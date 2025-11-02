"""
Memory system API endpoints.

Exposed endpoints:
- GET /api/memory/conversations - List user's conversations
- POST /api/memory/conversations - Create conversation
- GET /api/memory/conversations/{id} - Get conversation details
- GET /api/memory/conversations/{id}/messages - Get chat history
- GET /api/memory/search - Search past messages
- GET /api/memory/stats - User memory stats
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from uuid import UUID
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from typing import Annotated


from memory.database import DatabaseManager
from memory.service import ConversationService, ChatMessageService
from memory.schemas import (
    CreateConversationRequest,
    ConversationResponse,
    ChatMessageResponse,
    SearchMemoryRequest,
    MemoryStatsResponse
)

security = HTTPBearer()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Extract user_id from JWT token in Authorization header.
    
    Expected format:
        Authorization: Bearer <JWT_TOKEN>
    
    Token should contain:
        {"user_id": "550e8400-e29b-41d4-a716-446655440000", ...}
    
    Returns:
        user_id extracted from token
    
    Raises:
        HTTPException with 401 if token invalid
    """
    try:
        token = credentials.credentials
        
        # Decode JWT token
        payload = jwt.decode(
            token,
            key=os.getenv("JWT_SECRET_KEY", "your-secret-key"),  # Get from env!
            algorithms=["HS256"]
        )
        
        # Extract user_id
        user_id = payload.get("user_id")
        
        if not user_id:
            logger.warning("Token missing user_id claim")
            raise HTTPException(status_code=401, detail="Invalid token: missing user_id")
        
        logger.debug(f"Authenticated user: {user_id}")
        return user_id
    
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    db: Session = Depends(DatabaseManager.get_session)
):
    """
    Create a new conversation.
    
    Example request:
        {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "SEC Filing Analysis",
            "description": "Analyzing quarterly compliance"
        }
    
    Returns:
        Created conversation with ID
    """
    try:
        result = ConversationService.create_conversation(
            db=db,
            user_id=request.user_id,
            title=request.title,
            description=request.description
        )
        return result
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    user_id: str = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    include_archived: bool = Query(False),
    db: Session = Depends(DatabaseManager.get_session)
):
    """
    List user's conversations (paginated).
    """
    try:
        conversations = ConversationService.list_conversations(
            db=db,
            user_id=user_id,
            skip=skip,
            limit=limit,
            include_archived=include_archived
        )
        return conversations
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(DatabaseManager.get_session)
):
    """
    Get conversation details.
    """
    try:
        conversation = ConversationService.get_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversations/{conversation_id}/messages", response_model=list[ChatMessageResponse])
async def get_conversation_history(
    conversation_id: str,
    user_id: str = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(DatabaseManager.get_session)
):
    """
    Get full conversation history.
    """
    try:
        messages = ChatMessageService.get_conversation_history(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            skip=skip,
            limit=limit
        )
        return messages
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/search", response_model=list[ChatMessageResponse])
async def search_memory(
    user_id: str = Depends(get_current_user),
    query: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20),
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(DatabaseManager.get_session)
):
    """
    Search user's past messages.
    """
    try:
        results = ChatMessageService.search_memory(
            db=db,
            user_id=user_id,
            query=query,
            limit=limit,
            days=days
        )
        return results.results
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(DatabaseManager.get_session)
):
    """
    Get memory usage statistics.
    """
    try:
        stats = ChatMessageService.get_memory_stats(
            db=db,
            user_id=user_id
        )
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=400, detail=str(e))
