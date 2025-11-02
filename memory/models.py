"""
Database models for conversation memory system.

This module defines SQLAlchemy ORM models for storing chat history,
conversations, and metadata. These are the foundation of the memory system.

Models:
- Conversation: Groups messages by user session
- ChatMessage: Individual user/assistant exchanges
- ConversationMetadata: Session-specific settings (optional)
"""

from sqlalchemy import (
    CheckConstraint, Column, String, DateTime, Text, Integer, ForeignKey, 
    Index, Boolean, Float, desc, event
)
from sqlalchemy import JSON
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import json

Base = declarative_base()

class Conversation(Base):
    """
    Represents a single chat session between a user and the RAG system.
    
    Attributes:
        id: Unique conversation identifier (UNIQUEIDENTIFIER)
        user_id: Owner of the conversation
        title: Human-readable name (e.g., "SEC Filing Analysis")
        description: Optional summary of the conversation topic
        created_at: When the conversation started
        updated_at: Last message timestamp
        is_archived: Soft delete flag (archived conversations)
        metadata: Custom JSON data (tags, shared_with, etc.)
    
    Relationships:
        messages: All ChatMessage objects in this conversation
    """
    
    __tablename__ = "conversations"

    # Primary key
    id = Column(
        UNIQUEIDENTIFIER(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        doc="Unique conversation identifier"
    )
    
    # Foreign key
    user_id = Column(
        UNIQUEIDENTIFIER(as_uuid=True), 
        nullable=False, 
        index=True,
        doc="User who owns this conversation"
    )
    
    # Content fields
    title = Column(
        String(255),
        nullable=True,
        doc="Conversation title (e.g., 'Case Analysis - XYZ Corp')"
    )
    
    description = Column(
        Text,
        nullable=True,
        doc="Optional description of conversation purpose"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True,
        doc="When conversation was created"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        index=True,
        doc="When last message was added"
    )
    
    # Soft delete
    is_archived = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Archive conversations without deletion"
    )
    
    # Custom data
    custom_metadata = Column(
        JSON,
        nullable=True,
        default=lambda:{},
        doc="Extra data: {tags: [], shared_with: [], priority: 'high'}"
    )
    
    # Relationship
    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="dynamic",
        doc="All messages in this conversation",
        foreign_keys="ChatMessage.conversation_id"
    )
    
    # Composite indexes for common queries
    __table_args__ = (
        # Index for: Get recent conversations for a user
        Index('idx_user_updated', user_id, desc(updated_at)),
        
        # Index for: Get unarchived conversations
        Index('idx_user_active', user_id, is_archived, desc(updated_at)),
        
        # Index for: Archive operations
        Index('idx_archived', is_archived, user_id),
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id}, title='{self.title}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'title': self.title,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_archived': self.is_archived,
            'custom_metadata': self.custom_metadata or {},
        }
    

class ChatMessage(Base):
    """
    Represents a single message in a conversation (user or assistant).
    
    The heart of the memory system. Each query and response is stored here
    with full metadata for retrieval, debugging, and audit trails.
    
    Attributes:
        id: Unique message identifier
        conversation_id: Parent conversation
        user_id: Message owner
        role: 'user' or 'assistant'
        content: The actual message text
        embedding: Vector representation (for semantic search)
        sources: Retrieved legal documents used to generate response
        metadata: Performance metrics and debugging info
        tokens_used: Estimated LLM tokens for this exchange
        latency_ms: How long generation took
        created_at: When message was created
    """

    __tablename__ = "chat_messages"

    # Primary key
    id = Column(
        UNIQUEIDENTIFIER(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique message identifier"
    )
    
    # Foreign keys
    conversation_id = Column(
        UNIQUEIDENTIFIER(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
        doc="Parent conversation"
    )
    
    user_id = Column(
        UNIQUEIDENTIFIER(as_uuid=True),
        nullable=False,
        index=True,
        doc="Message owner (for access control)"
    )
    
    # Message content
    role = Column(
        String(10),
        nullable=False,
        default='user',
        doc="'user' or 'assistant' (or 'system' for future extensions)"
    )
    
    content = Column(
        Text,
        nullable=False,
        doc="Full message text (question or answer)"
    )
    
    # Vector embedding (for semantic search)
    embedding = Column(
        JSON,
        nullable=True,
        doc="Embedding vector as JSON array (1536 dimensions for text-embedding-3-small)"
    )
    
    # Retrieved documents (what the RAG used to generate response)
    sources = Column(
        JSON,
        nullable=True,
        default=lambda:[],
        doc="""Retrieved documents used in generation:
        [
            {
                'id': 'doc-uuid',
                'source': 'Case name or statute',
                'text': 'Excerpt',
                'custom_metadata': {'year': 2023, 'court': 'SDNY'},
                'confidence': 0.95
            }
        ]"""
    )
    
    # Performance & debugging metadata
    custom_metadata = Column(
        JSON,
        nullable=True,
        default=lambda:{},
        doc="""Debugging and performance info:
        {
            'query_type': 'case_search|statute_lookup|analysis',
            'retrieval_time_ms': 45,
            'generation_time_ms': 2300,
            'model_used': 'gpt-3.5-turbo',
            'retrieval_count': 5,
            'reranker_used': true,
            'error': null
        }"""
    )
    
    # Performance metrics
    tokens_used = Column(
        Integer,
        nullable=True,
        doc="Estimated tokens for this message pair"
    )
    
    latency_ms = Column(
        Integer,
        nullable=True,
        doc="Generation time in milliseconds"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True,
        doc="When message was created"
    )
    
    # Relationship
    conversation = relationship(
        "Conversation",
        back_populates="messages",
        lazy="joined",
        foreign_keys=[conversation_id]
    )
    
    # Production indexes
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')"),
        
        # CRITICAL: Get last N messages for a conversation (ordered by time)
        Index('idx_conversation_messages_time', conversation_id, desc(created_at)),
        
        # For auditing: Get all messages by user
        Index('idx_user_messages_time', user_id, desc(created_at)),
        
        # For pagination: Combine user + role + time
        Index('idx_user_role_messages', user_id, role, desc(created_at)),
        
        # For token counting: Get messages for a conversation
        Index('idx_conversation_tokens', conversation_id, tokens_used),
    )
    
    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<ChatMessage(id={self.id}, role={self.role}, content='{content_preview}')>"
    
    def to_dict(self, include_embedding=False):
        data = {
            'id': str(self.id),
            'conversation_id': str(self.conversation_id),
            'role': self.role,
            'content': self.content,
            'sources': self.sources or [],
            'custom_metadata': self.custom_metadata or {},
            'tokens_used': self.tokens_used,
            'latency_ms': self.latency_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        
        if include_embedding and self.embedding:
            data['embedding'] = self.embedding
        
        return data
    
class ConversationMetadata(Base):
    """
    Optional: Extended metadata for conversations (users, tags, sharing, etc.)
    
    This is separate from Conversation.metadata for better indexing
    and query performance. Use this for:
    - Shared conversations
    - Tags and categorization
    - Access control lists
    - Custom properties
    """
    
    __tablename__ = "conversation_metadata"
    
    id = Column(
        UNIQUEIDENTIFIER(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    conversation_id = Column(
        UNIQUEIDENTIFIER(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    
    # Tags and categorization
    tags = Column(
        JSON,
        default=list,
        doc="Tags for searching: ['sec-filing', 'compliance', 'urgent']"
    )
    
    # Sharing
    shared_with = Column(
        JSON,
        default=list,
        doc="List of user IDs who have access"
    )
    
    # Custom fields
    custom_properties = Column(
        JSON,
        default=dict,
        doc="User-defined metadata"
    )
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============ Database event listeners ============
@event.listens_for(Conversation, 'before_update')
def receive_before_update(mapper, connection, target):
    """Automatically update modified_at timestamp"""
    target.updated_at = datetime.utcnow()
