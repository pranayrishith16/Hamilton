# FastAPI entrypoint with all necessary routes and middleware

from datetime import datetime
import os
from sys import exc_info
from fastapi.routing import APIRoute
from requests import Session

from memory.database import DatabaseManager, get_db
from memory.repository import ChatMessageRepository, ConversationRepository
from memory.service import ConversationService
from memory.utils import estimate_tokens, format_context_string
from starlette.middleware.gzip import GZipMiddleware

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

# ==================== SECURITY MIDDLEWARE STACK ====================
# Add security middleware in order of execution (ONLY ONCE)

app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(SecurityLoggingMiddleware)
app.add_middleware(TokenBlacklistMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)  
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=[
#         "veritlyai.com",
#         "www.veritlyai.com",
#         "*.veritlyai.com",
#         "*.azurefd.net",
#         "localhost",  # for local testing
#     ]
# )
app.add_middleware(SecurityHeadersMiddleware)


# ==================== CORS MIDDLEWARE ====================

FRONTEND_DOMAINS = [
    "https://veritlyai.com",
    "https://www.veritlyai.com",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_DOMAINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
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

# ==================== SECURITY HEADERS MIDDLEWARE ====================

@app.middleware("http")
async def set_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    response.headers["Content-Security-Policy"] = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https:; "
    "connect-src 'self' https:; "  # ← Allow https connections
    "frame-src 'self' blob: data:; "  # ← Allow data: URIs for PDFs
    "object-src 'self' https:; "  # ← Allow object tag for PDFs
    "media-src 'self' blob: data:; "  # ← Allow blob for media
    "worker-src 'self' blob:; "  # ← Allow web workers
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

# ==================== REQUEST/RESPONSE MODELS ====================

class QueryRequest(BaseModel):
    query: str
    k: int = 10
    conversation_id: Optional[str] = None

class RetrieveRequest(BaseModel):
    query: str
    k: int = 10
    retriever_type: str = "hybrid"

class GenerateRequest(BaseModel):
    query: str
    context: List[Dict[str, Any]]

class ConfigUpdateRequest(BaseModel):
    component_name: str
    config: Dict[str, Any]

class IndexBuildRequest(BaseModel):
    chunks: List[Dict[str, Any]]
    retriever_type: str = "bm25"

# ==================== GLOBAL INSTANCES ====================

pipeline = Pipeline()
db_manager = DatabaseManager()

# ==================== AUTH DEPENDENCY ====================

def verify_jwt(authorization: str = Header(None)) -> dict:
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    
    token = authorization.replace("Bearer ", "").strip()
    payload = auth_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload

# ==================== UTILITY FUNCTIONS ====================

def get_db_session():
    return db_manager.get_session()

# ==================== BASE ROUTER - CRITICAL ==================== 

router = APIRouter(prefix="/api/base", tags=["base"])

@router.get("/")
async def base_root():
    """Root endpoint that returns API information and routes."""
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

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring system status."""
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

# ==================== ROUTER REGISTRATION (ONLY ONCE) ====================

app.include_router(router)              # /api/base
app.include_router(auth_router)         # /api/auth
app.include_router(memory_router)       # /api/memory
app.include_router(document_router)     # /api/documents

# ==================== ROOT ENDPOINT ====================

@app.get("/")
async def root():
    """Root endpoint - returns simple welcome message."""
    return {
        "message": "Veritly AI Backend",
        "status": "running",
        "docs_url": "/docs",
        "api_base": "/api"
    }

# ==================== QUERY ENDPOINTS (PROTECTED) ====================

@app.post("/api/query")
async def query_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(rate_limit_check)
):
    """Full pipeline query endpoint."""
    try:
        logger.info(f"Query from user {user['sub']}: {request.query[:50]}...")
        
        if not request.conversation_id:
            conv = ConversationService.create_conversation(
                db=db,
                user_id=user["sub"],
                title=request.query[:100],
                description="Auto-created from query"
            )
            conversation_id = conv["id"]
            context_string = ""
            context_tokens = 0
            logger.info(f'Created a new conversation')
        else:
            try:
                context_string, context_tokens, messages_loaded = \
                ConversationService.load_context_with_token_limit(
                    db=db,
                    conversation_id=request.conversation_id,
                    user_id=user["sub"],
                    max_tokens=5000,
                    max_messages=10
                )
                conversation_id = request.conversation_id
                logger.info(
                    f"Using existing conversation: {conversation_id} "
                    f"({messages_loaded} messages, {context_tokens} tokens)"
                )
            except ValueError as e:
                logger.error(f"Access denied: {e}")
                raise HTTPException(status_code=403, detail=str(e))
        
        result = pipeline.query(request.query, k=request.k, context=context_string)
        logger.info(f"Retrieved {len(result.retrieved_chunks)} chunks")
        
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
                "id": str(chunk.id),
                "content": chunk.content,
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {},
                "confidence": getattr(chunk, 'confidence', None)
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
                    "content": chunk.content[:500],
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
        conversation_id = None
        context_tokens = 0
        messages_loaded = 0
        tokens_streamed = 0
        full_answer = ""
        generation_start = None
        retrieved_chunks = []
        sources_received = False

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
                
                conversation_data = {
                    'event': 'conversation_created',
                    'conversation': {
                        'id': str(conversation_id),
                        'title': request.query[:100],
                        'description': "Auto-created from streaming query",
                        'created_at': datetime.now().isoformat(),
                        'user_id': user['sub']
                    }
                }

                yield f"data: {json.dumps(conversation_data)}\n\n"


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
            
            async for chunk_dict in pipeline.query_stream(request.query, k=request.k, context=context_string):
                # Handle sources event
                if chunk_dict.get('event') == 'sources' and chunk_dict.get('sources'):
                    retrieved_chunks = chunk_dict['sources']
                    sources_received = True
                    
                    logger.info(f"🔍 Received {len(retrieved_chunks)} sources from pipeline")
                    
                    # Emit analyzing status FIRST
                    yield f"data: {json.dumps({'event': 'analyzing', 'status': 'Analyzing sources...'})}\n\n"
                    
                    # Stream each source with CORRECT event name
                    for i, source in enumerate(retrieved_chunks):
                        # Format source with correct field names for Redux
                        formatted_source = {
                            'id': source.get('id', f'source_{i}'),
                            'content': source.get('snippet', source.get('source', '')),  # Use snippet as content
                            'metadata': {
                                'source_name': source.get('source', 'Unknown'),
                                'rank': i + 1,
                                'context_used': source.get('context_used', False)
                            }
                        }
                    
                    yield f"data: {json.dumps({'event': 'source_retrieved', 'rank': i+1, 'source': formatted_source})}\n\n"
                    
                    logger.info(f"✅ Finished streaming {len(retrieved_chunks)} sources")
                    continue
                
                # Handle token events
                if 'choices' in chunk_dict and len(chunk_dict['choices']) > 0:
                    delta = chunk_dict['choices'][0].get('delta', {})
                    if delta.get('content'):
                        token = delta['content']
                        
                        # Skip DONE marker - frontend will see it
                        if token == 'DONE':
                            logger.info(f"✅ Generation complete: {tokens_streamed} tokens")
                            break
                        
                        full_answer += token
                        tokens_streamed += 1
                        
                        # Emit token
                        yield f"data: {json.dumps({'event': 'token', 'content': token, 'tokens_streamed': tokens_streamed})}\n\n"
            
            generation_time_ms = (datetime.now() - generation_start).total_seconds() * 1000
            
            logger.info(f"Generation complete - Tokens: {tokens_streamed}, Time: {generation_time_ms:.0f}ms")
            
            yield f"data: {json.dumps({'event': 'generation_complete', 'total_tokens': tokens_streamed, 'generation_time_ms': generation_time_ms})}\n\n"

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
            logger.info(f"Stored user message: {user_msg.id}")

            source_objects = []
            for chunk in retrieved_chunks:
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

@app.post("/api/retrieve")
async def retrieve_endpoint(request: RetrieveRequest, user: dict = Depends(verify_jwt)):
    """Direct retrieval endpoint without generation."""
    try:
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

@app.get("/api/config")
async def get_config():
    """Get all system configuration."""
    try:
        return {
            "config": registry.config,
            "components": registry.list_components(),
            "config_sections": registry.list_config_sections()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

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
    
    try:
        logger.info("Initializing memory database...")
        from memory.database import DatabaseManager
        DatabaseManager.initialize()
        logger.info("✓ Memory database initialized and tables created")
    except Exception as e:
        logger.error(f"Memory database init failed: {e}", exc_info=True)
        raise