# FastAPI entrypoint; validates auth, loads active flavor, and delegates to orchestrator/pipeline; returns grounded answers

"""
FastAPI entrypoint for the RAG system.
Validates auth, loads active flavor, and delegates to orchestrator/pipeline.
"""

from datetime import datetime
import os
from sys import exc_info

from fastapi.routing import APIRoute
from requests import Session

from memory.database import DatabaseManager, get_db
from memory.repository import ChatMessageRepository, ConversationRepository
from memory.service import ConversationService
from memory.utils import estimate_tokens, format_context_string
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from fastapi import APIRouter, FastAPI, HTTPException, Depends, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
import faulthandler
import json
import dotenv

from orchestrator.pipeline import Pipeline
from orchestrator.registry import registry
from auth.auth_routes import router as auth_router
from auth.auth_manager import auth_manager
from auth.security_middleware import (
    SecurityHeadersMiddleware,
    HTTPSEnforcementMiddleware,
    TokenBlacklistMiddleware,
    SecurityLoggingMiddleware,
    RateLimitMiddleware,
    AuditLoggingMiddleware,
)
from starlette.middleware.trustedhost import TrustedHostMiddleware
from memory.memory_routes import router as memory_router
from documents.doc_routes import router as document_router

from auth.rbac_dependencies import (
    verify_jwt_token,
    require_admin,
    require_viewer,
    require_editor,
    require_permission,
    rate_limit_check
)

dotenv.load_dotenv()

faulthandler.enable()

# Initialize FastAPI app
app = FastAPI(
    title="Modular RAG API",
    description="A modular, plug-and-play RAG system",
    version="1.1.0"
)
"""
FastAPI application instance.
Configured with CORS middleware to allow cross-origin requests from any domain.
All endpoints are mounted on this app instance.
Provides auto-generated OpenAPI documentation at /docs and /redoc.
"""

# ==================== SECURITY MIDDLEWARE STACK ====================

# Add security middleware in order of execution
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(SecurityLoggingMiddleware)
app.add_middleware(TokenBlacklistMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
app.add_middleware(HTTPSEnforcementMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "veritlyai.com",
        "www.veritlyai.com",
        "api.veritlyai.com",
        "*.veritlyai.com",      # Catch all subdomains
        "*.azurefd.net",        # Allow Front Door
    ]
)



# ==================== ROUTERS ====================

router = APIRouter(prefix="/api/base", tags=["base"])

app.include_router(router)
app.include_router(auth_router)
app.include_router(memory_router)
app.include_router(document_router)

# ==================== CORS MIDDLEWARE ====================

FRONTEND_DOMAINS = [
    "https://veritlyai.com",
    "https://www.veritlyai.com",
    "http://localhost:5173",      # Local dev
    "http://localhost:3000",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_DOMAINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",              # ✅ Critical for JWT
        "Accept",
        "Accept-Language",
        "Accept-Encoding",
        "Origin",
    ],
    expose_headers=[
        "Content-Disposition",
        "Content-Type",
        "Content-Length",
        "Access-Control-Allow-Origin",
    ],
    max_age=86400,
)

# Update CSP headers to allow iframes
@app.middleware("http")
async def set_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # ✅ Updated CSP to allow Swagger UI from CDN
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "  # ✅ Allow CDN scripts
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "  # ✅ Allow CDN styles
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "frame-src 'self' blob:; "
        "object-src 'none'; "
    )
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), "
        "microphone=(), "
        "camera=(), "
        "payment=(), "
        "usb=(), "
        "magnetometer=(), "
        "gyroscope=(), "
        "accelerometer=()"
    )
    
    return response

# ==================== Request/Response Models ====================

class QueryRequest(BaseModel):
    """
    Request model for query endpoint.
    Contains the user's query string and optional k parameter for top-k retrieval.
    Used in: POST /query and POST /query/stream
    """
    query: str
    k: int = 10
    conversation_id: Optional[str] = None

class RetrieveRequest(BaseModel):
    """
    Request model for direct retrieval endpoint.
    Allows users to specify query, number of results (k), and which retriever to use.
    Supports: hybrid, bm25, or qdrant retrievers.
    Used in: POST /retrieve
    """
    query: str
    k: int = 10
    retriever_type: str = "hybrid"  # "hybrid", "bm25", "qdrant"

class GenerateRequest(BaseModel):
    """
    Request model for direct generation endpoint.
    Takes a query and pre-provided context chunks to generate answer without retrieval.
    Useful for testing generation independently from retrieval.
    Used in: POST /generate
    """
    query: str
    context: List[Dict[str, Any]]  # List of chunks with id, content, metadata

class ConfigUpdateRequest(BaseModel):
    """
    Request model for configuration updates.
    Contains component name and new configuration dictionary.
    Used in: POST /config/update (if implemented)
    """
    component_name: str
    config: Dict[str, Any]

class IndexBuildRequest(BaseModel):
    """
    Request model for building retriever indexes.
    Contains list of chunks (with id, content, metadata) and retriever type.
    Allows building fresh indexes for BM25 or Qdrant.
    Used in: POST /index/build
    """
    chunks: List[Dict[str, Any]]  # chunks with id, content, metadata
    retriever_type: str = "bm25"  # "bm25" or "qdrant"

# ==================== Global Instances ====================

pipeline = Pipeline()
"""
Global Pipeline instance.
Main orchestrator for query processing.
Used by /query and /query/stream endpoints.
"""

db_manager = DatabaseManager()

# ==================== AUTH DEPENDENCY ====================

def verify_jwt(authorization: str = Header(None)) -> dict:
    """
    Dependency to verify JWT token.
    Add this to any route that needs authentication.
    Returns user payload with: sub (user_id), email, tier, exp
    """
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    
    token = authorization.replace("Bearer ", "").strip()
    payload = auth_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload

# ==================== UTILITY FUNCTIONS ====================

def get_db_session():
    """Get database session for memory operations."""
    return db_manager.get_session()


# ==================== QUERY ENDPOINTS ====================

@router.get("/")
async def base_root():
    """Root endpoint that returns API information."""
    from fastapi.routing import APIRoute
    
    routes = [
        {
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods - {"HEAD", "OPTIONS"})
        }
        for route in app.routes
        if isinstance(route, APIRoute)
    ]
    
    return {
        "message": "Veritly AI - Secure RAG API",
        "version": "2.0.0",
        "security_features": [
            "Email verification",
            "Password reset with tokens",
            "Refresh token rotation",
            "Role-based access control",
            "HTTPS enforcement",
            "Security headers (CSP, HSTS, etc.)",
            "Token blacklisting",
            "Audit logging",
            "Rate limiting",
            "Azure Key Vault integration"
        ],
        "routes": routes
    }

# ==================== QUERY ENDPOINTS (PROTECTED) ====================

@app.post("/api/query")
async def query_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(rate_limit_check)
):
    """
    Full pipeline query endpoint.
    
    Requires: Authenticated user with verified email
    Rate limited based on subscription tier
    """

    try:
        logger.info(f"Query from user {user['sub']}: {request.query[:50]}...")
        # NEW: Handle conversation
        if not request.conversation_id:
            conv = ConversationService.create_conversation(
                db=db,
                user_id=user["sub"],
                title=request.query[:100],
                description="Auto-created from query"
            )
            conversation_id = conv["id"]
            context_string = ""  # No prior context for new conversation
            context_tokens = 0
            logger.info(f'Created a new conversation')
        else:
            # Use existing conversation with TOKEN LIMITING
            try:
                context_string, context_tokens, messages_loaded = \
                    ConversationService.load_context_with_token_limit(
                        db=db,
                        conversation_id=request.conversation_id,
                        user_id=user["sub"],
                        max_tokens=5000,  # ← HARD LIMIT: 2000 tokens max
                        max_messages=10   # ← Also limit to last 10 messages
                    )
                
                conversation_id = request.conversation_id
                
                logger.info(
                    f"Using existing conversation: {conversation_id} "
                    f"({messages_loaded} messages, {context_tokens} tokens)"
                )
            except ValueError as e:
                logger.error(f"Access denied: {e}")
                raise HTTPException(status_code=403, detail=str(e))

        result = pipeline.query(request.query, k=request.k,context=context_string)

        logger.info(f"Retrieved {len(result.retrieved_chunks)} chunks")

        # NEW: Store to memory
        user_msg = ChatMessageRepository.create(
            db=db,
            conversation_id=conversation_id,
            user_id=user["sub"],
            role="user",
            content=request.query,
            sources=None,
            tokens_used=estimate_tokens(request.query)
        )
        db.commit()
        logger.info(f"Stored user message: {user_msg.id}")

        source_objects = []
        for chunk in result.retrieved_chunks:
            source_obj = {
                "id": str(chunk.id),  # Document ID
                "content": chunk.content,  # Full text excerpt
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {},  # Metadata dict
                "confidence": getattr(chunk, 'confidence', None)  # Optional confidence score
            }
            source_objects.append(source_obj)


        assistant_msg = ChatMessageRepository.create(
            db=db,
            conversation_id=conversation_id,
            user_id=user["sub"],
            role="assistant",
            content=result.answer,
            sources=source_objects,
            tokens_used=estimate_tokens(result.answer),
            # Store retrieval statistics
            metadata={
                "retrieved_count": len(result.retrieved_chunks),
                "generation_time_s": result.metadata.get("generation_time_s"),
                "total_time_s": result.metadata.get("total_time_s"),
                "context_tokens": context_tokens,
                "context_used": context_string is not None and len(context_string) > 0
            }
        )
        db.commit()
        logger.info(f"Stored assistant message: {assistant_msg.id}")
        
        response = {
            "query": result.query,
            "answer": result.answer,
            "conversation_id": str(conversation_id),
            "user_message_id": str(user_msg.id),
            "assistant_message_id": str(assistant_msg.id),
            "retrieved_chunks": [
                {
                    "id": str(chunk.id),
                    "content": chunk.content[:500],  # Truncate for response
                    "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                }
                for chunk in result.retrieved_chunks
            ],
            "memory": {
                "context_used": context_string[:200] if context_string else None,
                "context_available": len(context_string) > 0 if context_string else False
            },
            "metadata": {
                "retrieval_time_s": result.metadata.get("retrieval_time_s"),
                "generation_time_s": result.metadata.get("generation_time_s"),
                "total_time_s": result.metadata.get("total_time_s"),
                "user_tier": user["tier"]
            }
        }
        
        db.close()
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(rate_limit_check)
):
    """
    Streaming query endpoint with memory integration.
    Streams tokens as they're generated while maintaining context.
    """
    async def event_generator():
        # Initialize all variables at start so they exist in any execution path
        retrieved_chunks = []
        conversation_id = None
        context_tokens = 0
        messages_loaded = 0
        tokens_streamed = 0
        full_answer = ""
        generation_start = None

        try:
            logger.info(f"Streaming query from user {user['sub']}: {request.query[:50]}...")
            
            # Conversation management (with token limiting and verification)
            if not request.conversation_id:
                conv_data = ConversationService.create_conversation(
                    db=db,
                    user_id=user["sub"],
                    title=request.query[:100],
                    description="Auto-created from streaming query"
                )
                conversation_id = conv_data["id"]
                context_string = ""
                context_tokens = 0
                messages_loaded = 0
                
                logger.info(f'Created new conversation: {conversation_id}')
                
                yield f"data: {json.dumps({'event': 'conversation_created', 'conversation_id': str(conversation_id)})}\n\n"
            else:
                try:
                    context_string, context_tokens, messages_loaded = ConversationService.load_context_with_token_limit(
                            db=db,
                            conversation_id=request.conversation_id,
                            user_id=user["sub"],
                            max_tokens=5000,  # Hard token limit
                            max_messages=10   # Limit number of messages considered
                        )
                    conversation_id = request.conversation_id
                    
                    logger.info(
                        f"Using existing conversation: {conversation_id} "
                        f"({messages_loaded} messages, {context_tokens} tokens)"
                    )
                    
                    yield f"data: {json.dumps({'event': 'context_loaded', 'context_tokens': context_tokens, 'messages_loaded': messages_loaded, 'context_available': len(context_string) > 0})}\n\n"
                    
                except ValueError as e:
                    logger.error(f"Access denied: {e}")
                    yield f"data: {json.dumps({'event': 'error', 'error': 'Access denied', 'detail': str(e), 'status_code': 403})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            # Start streaming generation
            generation_start = datetime.now()
            
            for chunk_dict in pipeline.query_stream(request.query, k=request.k, context=context_string):
                if "choices" in chunk_dict and len(chunk_dict["choices"]) > 0:
                    delta = chunk_dict["choices"][0].get("delta", {})
                    if delta.get("content"):
                        token = delta["content"]
                        full_answer += token
                        tokens_streamed += 1
                        
                        yield f"data: {json.dumps({'event': 'token', 'content': token, 'tokens_streamed': tokens_streamed})}\n\n"
            
            generation_time_ms = (datetime.now() - generation_start).total_seconds() * 1000
            
            logger.info(f"Generation complete - Tokens: {tokens_streamed}, Time: {generation_time_ms:.0f}ms")
            
            yield f"data: {json.dumps({'event': 'generation_complete', 'total_tokens': tokens_streamed, 'generation_time_ms': generation_time_ms})}\n\n"

            # Retrieve chunks for metadata if any
            try:
                retrieval_result = pipeline.query(request.query, k=request.k, context=context_string)
                retrieved_chunks = retrieval_result.retrieved_chunks
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks for metadata")
            except Exception as e:
                logger.warning(f"Could not retrieve chunks for metadata: {e}")
                retrieved_chunks = []

            # Store user message
            user_msg = ChatMessageRepository.create(
                db=db,
                conversation_id=conversation_id,
                user_id=user["sub"],
                role="user",
                content=request.query,
                sources=None,
                tokens_used=estimate_tokens(request.query)
            )
            db.commit()
            logger.info(f"Stored user message: {user_msg.id}")

            source_objects = []
            for chunk in retrieval_result.retrieved_chunks:
                source_obj = {
                    "id": str(chunk.id),  # Document ID
                    "content": chunk.content,  # Full text excerpt
                    "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {},  # Metadata dict
                    "confidence": getattr(chunk, 'confidence', None)  # Optional confidence score
                }
                source_objects.append(source_obj)

            # Store assistant message
            assistant_msg = ChatMessageRepository.create(
                db=db,
                conversation_id=conversation_id,
                user_id=user["sub"],
                role="assistant",
                content=full_answer,
                sources=source_objects,
                tokens_used=estimate_tokens(full_answer),
                metadata={
                    "retrieved_count": len(retrieved_chunks),
                    "generation_time_ms": generation_time_ms,
                    "context_tokens": context_tokens,
                    "context_used": context_tokens > 0,
                    "streaming": True,
                    "tokens_streamed": tokens_streamed
                }
            )
            db.commit()
            logger.info(f"Stored assistant message: {assistant_msg.id}")

            yield f"data: {json.dumps({'event': 'stored', 'user_message_id': str(user_msg.id), 'assistant_message_id': str(assistant_msg.id)})}\n\n"

            yield f"data: {json.dumps({'event': 'complete', 'conversation_id': str(conversation_id), 'answer_length': len(full_answer), 'total_tokens': tokens_streamed, 'retrieved_chunks': len(retrieved_chunks), 'context_tokens': context_tokens})}\n\n"

            yield "data: [DONE]\n\n"

        except HTTPException as e:
            logger.error(f"HTTP Exception in streaming: {e.detail}")
            yield f"data: {json.dumps({'event': 'error', 'error': e.detail, 'status_code': e.status_code})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'error': 'Internal server error', 'detail': str(e), 'status_code': 500})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            try:
                db.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked"
        }
    )


# ==================== RETRIEVAL ENDPOINTS ====================

@app.post("/api/retrieve")
async def retrieve_endpoint(request: RetrieveRequest, user: dict = Depends(verify_jwt)):
    """
    Direct retrieval endpoint without generation.
    Retrieves top-k chunks using specified retriever type.
    Allows testing retrieval independently from generation.
    Supports three retriever types: bm25 (keyword), qdrant (semantic), hybrid (both).
    
    Dependencies: bm25_retriever.retrieve() OR qdrant_retriever.retrieve() OR hybrid_retriever.retrieve()
    Status Codes: 200 (success), 500 (invalid retriever type or error)
    Returns: query, retriever_type, chunks[] with id, content, metadata
    """
    try:
        # Optional: Check rate limit
        if not auth_manager.check_query_limit(user["sub"]):
            raise HTTPException(status_code=429, detail="Query limit exceeded")
        
        if request.retriever_type == "bm25":
            retriever = registry.get("bm25_retriever")
        elif request.retriever_type == "qdrant":
            retriever = registry.get("qdrant_retriever")
        elif request.retriever_type == "hybrid":
            retriever = registry.get("hybrid_retriever")
        else:
            raise ValueError(f"Unknown retriever type: {request.retriever_type}")
        
        chunks = retriever.retrieve(request.query, k=request.k)
        auth_manager.log_query(user["sub"], request.query, "success")
        
        return {
            "query": request.query,
            "retriever_type": request.retriever_type,
            "user_id": user["sub"],
            "user_tier": user["tier"],
            "chunks": [
                {"id": c.id, "content": c.content, "metadata": c.metadata}
                for c in chunks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_endpoint(request: GenerateRequest, user: dict = Depends(verify_jwt)):
    """
    PROTECTED: Direct generation endpoint.
    Generates answer from user-provided context.
    """
    try:
        from ingestion.dataprep.chunkers.base import Chunk
        
        chunks = [
            Chunk(
                id=c.get("id", ""),
                content=c.get("content", ""),
                metadata=c.get("metadata", {})
            )
            for c in request.context
        ]
        
        generator = registry.get("generator")
        answer = generator.generate(request.query, chunks)
        auth_manager.log_query(user["sub"], request.query, "success")
        
        return {
            "query": request.query,
            "answer": answer,
            "chunks_used": len(chunks),
            "user_id": user["sub"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== INDEX MANAGEMENT ENDPOINTS ====================

# @router.post("/index/build")
# async def build_index_endpoint(request: IndexBuildRequest, user: dict = Depends(verify_jwt)):
#     """
#     PROTECTED + ADMIN ONLY: Build/rebuild retriever indexes.
#     """
#     # Check admin role
#     if user.get("tier") != "admin":
#         raise HTTPException(status_code=403, detail="Admin access required")
    
#     try:
        
#         if request.retriever_type == "bm25":
#             retriever = registry.get("bm25_retriever")
#         elif request.retriever_type == "qdrant":
#             retriever = registry.get("qdrant_retriever")
#         else:
#             raise ValueError(f"Unknown retriever type: {request.retriever_type}")
        
#         retriever.build_index(chunks)
#         logger.info(f"Admin {user['email']} built {request.retriever_type} index")
        
#         return {
#             "status": "success",
#             "retriever_type": request.retriever_type,
#             "chunks_indexed": len(chunks),
#             "admin": user["email"]
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/index/stats/{retriever_type}")
async def get_index_stats(retriever_type: str):
    """
    Get statistics about a specific retriever's index.
    Returns number of indexed chunks, index size, and other metrics.
    Useful for monitoring index health and size.
    
    Dependencies: retriever.get_stats()
    Supports: bm25, qdrant, hybrid
    Status Codes: 200 (success), 500 (invalid type or error)
    Returns: Index statistics dictionary (varies by retriever)
    """
    try:
        if retriever_type == "bm25":
            retriever = registry.get("bm25_retriever")
        elif retriever_type == "qdrant":
            retriever = registry.get("qdrant_retriever")
        elif retriever_type == "hybrid":
            retriever = registry.get("hybrid_retriever")
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        return retriever.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# ==================== CONFIGURATION ENDPOINTS ====================

@app.get("/api/config")
async def get_config():
    """
    Get all system configuration.
    Returns complete configuration, registered components, and config sections.
    Useful for understanding system setup and component dependencies.
    
    Dependencies: registry.config, registry.list_components(), registry.list_config_sections()
    Status Codes: 200 (success), 500 (error)
    Returns: config dict, components list, config_sections list
    """
    try:
        return {
            "config": registry.config,
            "components": registry.list_components(),
            "config_sections": registry.list_config_sections()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/config/{section}")
async def get_config_section(section: str):
    """
    Get specific configuration section.
    Retrieves configuration for a particular component or section.
    
    Dependencies: registry.get_config(section)
    Status Codes: 200 (success), 404 (section not found), 500 (error)
    Returns: Configuration dictionary for specified section
    """
    try:
        return registry.get_config(section)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/config/reload")
async def reload_config(user: dict = Depends(verify_jwt)):
    """
    PROTECTED + ADMIN ONLY: Reload configuration.
    """
    if user.get("tier") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        registry.reload_config()
        logger.info(f"Admin {user['email']} reloaded config")
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== COMPONENT ENDPOINTS ====================

@app.get("/api/components")
async def list_components():
    """
    List all registered components in the system.
    Returns component names and configuration sections.
    Useful for understanding system architecture.
    
    Dependencies: registry.list_components(), registry.list_config_sections()
    Status Codes: 200 (success), 500 (error)
    Returns: components list, config_sections list
    """
    try:
        return {
            "components": registry.list_components(),
            "config_sections": registry.list_config_sections()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/component/{component_name}/info")
async def get_component_info(component_name: str):
    """
    Get detailed information about a specific component.
    Returns component metadata, model info, or statistics.
    Supports: generator, bm25_retriever, qdrant_retriever, hybrid_retriever
    
    Dependencies: registry.get(component_name) -> component.get_model_info() OR get_stats()
    Status Codes: 200 (success), 404 (component not found), 500 (error)
    Returns: Component-specific information dictionary
    """
    try:
        component = registry.get(component_name)
        
        # Get component-specific info
        if hasattr(component, 'get_model_info'):
            return component.get_model_info()
        elif hasattr(component, 'get_stats'):
            return component.get_stats()
        else:
            return {
                "component": component_name,
                "type": str(type(component)),
                "info": "No additional info available"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# ==================== RETRIEVER SPECIFIC ENDPOINTS ====================

@app.post("/api/bm25/clear-cache")
async def clear_bm25_cache(user: dict = Depends(verify_jwt)):
    """
    PROTECTED + ADMIN ONLY: Clear BM25 cache.
    """
    if user.get("tier") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        retriever = registry.get("bm25_retriever")
        retriever.clear_cache()
        logger.info(f"Admin {user['email']} cleared BM25 cache")
        return {"status": "success", "message": "BM25 cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hybrid/config")
async def get_hybrid_config():
    """
    Get hybrid retriever configuration and statistics.
    Returns RRF (Reciprocal Rank Fusion) parameter and sub-retriever stats.
    Useful for understanding hybrid retriever setup.
    
    Dependencies: hybrid_retriever.get_stats()
    Status Codes: 200 (success), 500 (error)
    Returns: hybrid retriever stats including k_rrf and sub-retriever metrics
    """
    try:
        hybrid = registry.get("hybrid_retriever")
        return hybrid.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HEALTH & INFO ENDPOINTS ====================

@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring system status.
    Verifies that all core components are initialized and accessible.
    Used by load balancers and monitoring systems.
    
    Dependencies: registry.get() for all core components
    Status Codes: Always 200 (check 'status' field: 'healthy' or 'unhealthy')
    Returns: status (healthy/unhealthy), component types
    """

    try:
        components = {
            "qdrant_retriever": registry.get("qdrant_retriever"),
            "bm25_retriever": registry.get("bm25_retriever"),
            "hybrid_retriever": registry.get("hybrid_retriever"),
            "generator": registry.get("generator")
        }
        return {
            "status": "healthy",
            "components": {name: type(comp).__name__ for name, comp in components.items()}
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
    
# ==================== STARTUP EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    try:
        logger.info("Initializing auth database...")
        from auth.models import init_database
        init_database()
        logger.info("✓ Auth database initialized")
    except Exception as e:
        logger.warning(f"Auth database init: {e}")
    
    # NEW: Initialize memory system database
    try:
        logger.info("Initializing memory database...")
        from memory.database import DatabaseManager
        DatabaseManager.initialize()
        logger.info("✓ Memory database initialized and tables created")
    except Exception as e:
        logger.error(f"Memory database init failed: {e}", exc_info=True)
        raise