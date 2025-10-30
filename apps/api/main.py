# FastAPI entrypoint; validates auth, loads active flavor, and delegates to orchestrator/pipeline; returns grounded answers

"""
FastAPI entrypoint for the RAG system.
Validates auth, loads active flavor, and delegates to orchestrator/pipeline.
"""

import os

from fastapi.routing import APIRoute
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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
    AuditLoggingMiddleware
)

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

# ==================== CORS MIDDLEWARE ====================

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.veritlyai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ROUTERS ====================

app.include_router(auth_router)

# ==================== Request/Response Models ====================

class QueryRequest(BaseModel):
    """
    Request model for query endpoint.
    Contains the user's query string and optional k parameter for top-k retrieval.
    Used in: POST /query and POST /query/stream
    """
    query: str
    k: int = 10

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


# ==================== QUERY ENDPOINTS ====================

@app.get("/")
async def root():
    """
    Root endpoint that returns API information and available endpoints.
    Displays features and all available routes for self-documentation/debugging.
    """
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

@app.post("/query")
async def query_endpoint(
    request: QueryRequest,
    user: dict = Depends(rate_limit_check)
):
    """
    Full pipeline query endpoint.
    
    Requires: Authenticated user with verified email
    Rate limited based on subscription tier
    """
    try:
        # Check rate limit
        if not auth_manager.check_query_limit(user["sub"]):
            raise HTTPException(
                status_code=429,
                detail="Daily query limit exceeded"
            )
        
        result = pipeline.query(request.query, k=request.k)
        
        # Log successful query
        from auth.models import get_db_session, QueryLog
        session = get_db_session()
        query_log = QueryLog(
            user_id=user["sub"],
            query_text=request.query,
            status="success"
        )
        session.add(query_log)
        session.commit()
        session.close()
        
        return {
            "query": result.query,
            "answer": result.answer,
            "retrieved_chunks": [
                {"id": c.id, "content": c.content, "metadata": c.metadata}
                for c in result.retrieved_chunks
            ],
            "metadata": result.metadata,
            "user_tier": user["tier"],
            "user_id": user["sub"]
        }
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query/stream")
async def query_stream(request: Request, user: dict = Depends(rate_limit_check)):
    """
    Streaming query endpoint using Server-Sent Events.
    
    Requires: Authenticated user
    Returns: Streamed answer tokens
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        k = body.get("k", 5)
        
        if not query:
            raise ValueError("Query is required")
        
        # Check rate limit
        if not auth_manager.check_query_limit(user["sub"]):
            raise HTTPException(status_code=429, detail="Daily query limit exceeded")
    
    except Exception as exc:
        async def error_generator():
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    
    def event_generator():
        yield ": ping\n\n"
        try:
            for chunk_dict in pipeline.query_stream(query, k=k):
                yield f"data: {json.dumps(chunk_dict)}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ==================== RETRIEVAL ENDPOINTS ====================

@app.post("/retrieve")
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

@app.post("/generate")
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

# @app.post("/index/build")
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

@app.get("/index/stats/{retriever_type}")
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

@app.get("/config")
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


@app.get("/config/{section}")
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

@app.post("/config/reload")
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

@app.get("/components")
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

@app.get("/component/{component_name}/info")
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

@app.post("/bm25/clear-cache")
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


@app.get("/hybrid/config")
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

@app.get("/health")
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
    """Initialize database and check connections on startup"""
    try:
        logger.info('Initializing database')
        from auth.models import init_database
        init_database()
    except Exception as e:
        print(f"⚠ Database initialization: {e}")
    
    try:
        from auth.cache_manager import redis_manager
        print("✓ In-memory cache initialized")
    except Exception as e:
        print(f"⚠ Cache initialization: {e}")