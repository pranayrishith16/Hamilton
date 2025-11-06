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

# ==================== SECURITY MIDDLEWARE STACK ====================
# Add security middleware in order of execution (ONLY ONCE)

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
        "*.veritlyai.com",
        "*.azurefd.net",
    ]
)

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