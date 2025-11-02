"""
Data access layer for memory system.

The repository pattern isolates database operations from business logic.
This makes testing easier and database queries more reusable.

Repository methods:
- Conversation: create, get, list, update, delete, archive
- ChatMessage: create, get, list_by_conversation, search
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, func
from uuid import UUID
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from memory.models import Conversation, ChatMessage, Base
from memory.schemas import (
    CreateConversationRequest, ChatMessageRequest
)

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Repository for Conversation database operations.
    
    Encapsulates all SQL queries related to conversations.
    """
    
    @staticmethod
    def create(
        db: Session,
        user_id: UUID,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            db: Database session
            user_id: User's UUID
            title: Conversation title
            description: Description
            metadata: Custom JSON metadata
        
        Returns:
            Created Conversation object
        """
        conversation = Conversation(
            user_id=user_id,
            title=title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            description=description,
            custom_metadata=metadata or {}
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        logger.info(f"Created conversation {conversation.id} for user {user_id}")
        return conversation
    
    @staticmethod
    def get_by_id(
        db: Session,
        conversation_id: UUID,
        user_id: Optional[UUID] = None
    ) -> Optional[Conversation]:
        """
        Get conversation by ID (with optional user verification).
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            user_id: Optional - if provided, verify user owns conversation
        
        Returns:
            Conversation object or None if not found
        """
        query = db.query(Conversation).filter(Conversation.id == conversation_id)
        
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        
        return query.first()
    
    @staticmethod
    def list_by_user(
        db: Session,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
        include_archived: bool = False
    ) -> List[Conversation]:
        """
        Get all conversations for a user, paginated.
        
        Args:
            db: Database session
            user_id: User UUID
            skip: Pagination offset
            limit: Page size
            include_archived: Whether to include archived conversations
        
        Returns:
            List of Conversation objects
        """
        query = db.query(Conversation).filter(
            Conversation.user_id == user_id
        )
        
        if not include_archived:
            query = query.filter(Conversation.is_archived == False)
        
        return query.order_by(
            desc(Conversation.updated_at)
        ).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(
        db: Session,
        conversation_id: UUID,
        **updates
    ) -> Conversation:
        """
        Update conversation fields.
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            **updates: Fields to update (title, description, metadata, etc.)
        
        Returns:
            Updated Conversation object
        """
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        for key, value in updates.items():
            if hasattr(conversation, key):
                setattr(conversation, key, value)
        
        db.commit()
        db.refresh(conversation)
        
        logger.info(f"Updated conversation {conversation_id}")
        return conversation
    
    @staticmethod
    def archive(
        db: Session,
        conversation_id: UUID
    ) -> Conversation:
        """
        Soft delete: archive a conversation (don't actually delete).
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
        
        Returns:
            Updated Conversation object
        """
        return ConversationRepository.update(
            db,
            conversation_id,
            is_archived=True
        )
    
    @staticmethod
    def delete(
        db: Session,
        conversation_id: UUID
    ) -> bool:
        """
        Hard delete a conversation and all its messages.
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
        
        Returns:
            True if deleted, False if not found
        """
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            return False
        
        db.delete(conversation)
        db.commit()
        
        logger.warning(f"Deleted conversation {conversation_id}")
        return True


class ChatMessageRepository:
    """
    Repository for ChatMessage database operations.
    
    Encapsulates all SQL queries related to chat messages.
    """
    
    @staticmethod
    def create(
        db: Session,
        conversation_id: UUID,
        user_id: UUID,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None
    ) -> ChatMessage:
        """
        Create and save a new chat message.
        
        Args:
            db: Database session
            conversation_id: Parent conversation UUID
            user_id: Message owner
            role: 'user' or 'assistant'
            content: Message text
            sources: Retrieved documents (for assistant messages)
            embedding: Vector representation
            metadata: Custom JSON metadata
            tokens_used: Estimated tokens
            latency_ms: Generation latency
        
        Returns:
            Created ChatMessage object
        """
        message = ChatMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            sources=sources,
            embedding=embedding,
            metadata=metadata or {},
            tokens_used=tokens_used,
            latency_ms=latency_ms
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        
        logger.info(
            f"Created {role} message {message.id} in conversation {conversation_id}"
        )
        return message
    
    @staticmethod
    def get_by_id(
        db: Session,
        message_id: UUID
    ) -> Optional[ChatMessage]:
        """
        Get a message by ID.
        
        Args:
            db: Database session
            message_id: Message UUID
        
        Returns:
            ChatMessage object or None
        """
        return db.query(ChatMessage).filter(
            ChatMessage.id == message_id
        ).first()
    
    @staticmethod
    def list_by_conversation(
        db: Session,
        conversation_id: UUID,
        skip: int = 0,
        limit: int = 100,
        reverse_order: bool = False
    ) -> List[ChatMessage]:
        """
        Get all messages in a conversation, chronologically ordered.
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            skip: Pagination offset
            limit: Max messages
            reverse_order: If True, newest first
        
        Returns:
            List of ChatMessage objects
        """
        query = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation_id
        )
        
        if reverse_order:
            query = query.order_by(desc(ChatMessage.created_at))
        else:
            query = query.order_by(ChatMessage.created_at)
        
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def get_recent(
        db: Session,
        conversation_id: UUID,
        count: int = 10
    ) -> List[ChatMessage]:
        """
        Get most recent messages from a conversation (last k messages).
        
        Used for loading context for augmentation.
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            count: Number of messages to retrieve
        
        Returns:
            List of ChatMessage objects (oldest to newest)
        """
        messages = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation_id
        ).order_by(
            desc(ChatMessage.created_at)
        ).limit(count).all()
        
        # Reverse to chronological order (oldest first)
        return list(reversed(messages))
    
    @staticmethod
    def search_by_user(
        db: Session,
        user_id: UUID,
        role: Optional[str] = None,
        days: int = 30,
        limit: int = 20
    ) -> List[ChatMessage]:
        """
        Search user's messages across all conversations.
        
        Args:
            db: Database session
            user_id: User UUID
            role: Filter by 'user' or 'assistant' (None = all)
            days: Only search messages from last N days
            limit: Max results
        
        Returns:
            List of ChatMessage objects
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(ChatMessage).filter(
            and_(
                ChatMessage.user_id == user_id,
                ChatMessage.created_at >= cutoff_date
            )
        )
        
        if role:
            query = query.filter(ChatMessage.role == role)
        
        return query.order_by(
            desc(ChatMessage.created_at)
        ).limit(limit).all()
    
    @staticmethod
    def get_stats(
        db: Session,
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        Get usage statistics for a user.
        
        Args:
            db: Database session
            user_id: User UUID
        
        Returns:
            Dictionary with stats
        """
        # Use SQL functions to do aggregation efficiently
        result = db.query(
            func.count(ChatMessage.id).label('total_messages'),
            func.sum(ChatMessage.tokens_used).label('total_tokens'),
            func.avg(ChatMessage.latency_ms).label('avg_latency_ms'),
            func.min(ChatMessage.created_at).label('oldest_message'),
            func.max(ChatMessage.created_at).label('newest_message')
        ).filter(
            ChatMessage.user_id == user_id
        ).first()
        
        # Handle empty result
        if not result or result.total_messages == 0:
            return {
                'total_messages': 0,
                'total_tokens': 0,
                'avg_latency_ms': 0.0,
                'oldest_message': None,
                'newest_message': None,
            }
        
        # Format result
        return {
            'total_messages': result.total_messages or 0,
            'total_tokens': result.total_tokens or 0,
            'avg_latency_ms': float(result.avg_latency_ms or 0),
            'oldest_message': result.oldest_message.isoformat() if result.oldest_message else None,
            'newest_message': result.newest_message.isoformat() if result.newest_message else None,
        }