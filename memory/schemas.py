"""
Pydantic schemas for memory system API validation and serialization.

These schemas handle:
1. Request validation (what frontend sends)
2. Response serialization (what API returns)
3. Type hints and documentation
"""

from pydantic import BaseModel, Field, field_validator, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID


# ============ Request Schemas ============

class CreateConversationRequest(BaseModel):
    """
    Request to create a new conversation.
    
    Example:
        {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "SEC Filing Analysis - Q3 2024",
            "description": "Analyzing regulatory compliance"
        }
    """
    user_id: str = Field(..., description="UUID of the user")
    title: Optional[str] = Field(
        None, 
        max_length=255,
        description="Conversation title"
    )
    description: Optional[str] = Field(
        None,
        description="Conversation description"
    )

    @field_validator('user_id')
    def validate_user_id(cls, v):
        from uuid import UUID
        try:
            UUID(v)
            return v
        except:
            raise ValueError(f"Invalid UUID: {v}")

class ChatMessageRequest(BaseModel):
    """
    Request to save a chat message.
    
    Internal use (called by RAG pipeline after generation).
    """
    conversation_id: str = Field(..., description="Conversation UUID")
    user_id: str = Field(..., description="User UUID")
    role: str = Field(..., pattern="^(user|assistant)$", description="'user' or 'assistant'")
    content: str = Field(..., max_length=50000, description="Message text (max 50KB)")
    sources: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Retrieved documents"
    )
    embedding: Optional[List[float]] = Field(
        None,
        description="Message embedding vector"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Performance and debug info"
    )
    tokens_used: Optional[int] = Field(None, description="Tokens used")
    latency_ms: Optional[int] = Field(None, description="Generation latency")

    @field_validator('user_id')
    def validate_user_id(cls, v):
        from uuid import UUID
        try:
            UUID(v)
            return v
        except:
            raise ValueError(f"Invalid UUID: {v}")
        
    @field_validator('embedding')
    def validate_embedding(cls, v):
        """Ensure embedding is 1536 dimensions (OpenAI text-embedding-3-small)"""
        if v is None:
            return v  # Optional, so None is OK
        
        if not isinstance(v, list):
            raise ValueError("embedding must be a list")
        
        if len(v) != 1536:
            raise ValueError(f"embedding must be 1536 dimensions, got {len(v)}")
        
        return v


class LoadContextRequest(BaseModel):
    """
    Request to load recent conversation context (for memory service).
    
    Example:
        {
            "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "550e8400-e29b-41d4-a716-446655440001",
            "num_pairs": 2
        }
    """
    conversation_id: str = Field(..., description="Conversation UUID")
    user_id: str = Field(..., description="User UUID")
    num_pairs: int = Field(
        2,
        ge=1,
        le=10,
        description="Number of Q&A pairs to load (2 = last 4 messages)"
    )

    @field_validator('user_id')
    def validate_user_id(cls, v):
        from uuid import UUID
        try:
            UUID(v)
            return v
        except:
            raise ValueError(f"Invalid UUID: {v}")


class SearchMemoryRequest(BaseModel):
    """
    Request to search conversation history by semantic similarity.
    
    Uses embeddings to find similar past conversations.
    """
    user_id: str
    query: str = Field(..., description="Query text to search for")
    limit: int = Field(5, ge=1, le=20, description="Max results")
    similarity_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity (0-1)"
    )

    @field_validator('user_id')
    def validate_user_id(cls, v):
        from uuid import UUID
        try:
            UUID(v)
            return v
        except:
            raise ValueError(f"Invalid UUID: {v}")


# ============ Response Schemas ============

class ChatMessageResponse(BaseModel):
    """
    Response for a single chat message.
    """
    id: str
    conversation_id: str
    role: str
    content: str
    sources: Optional[List[Union[Dict[str, Any], str]]] = Field(
        default_factory=list,
        description="Retrieved documents or document IDs"
    )
    custom_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadata about the message"
    )
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
    created_at: str

    class Config:
        orm_mode = True


class ConversationResponse(BaseModel):
    """
    Response for a conversation summary.
    """
    id: str
    user_id: str
    title: Optional[str]
    description: Optional[str]
    created_at: str
    updated_at: str
    is_archived: bool
    custom_metadata: dict = Field(default_factory=dict) 
    
    class Config:
        orm_mode = True

    @field_validator('user_id')
    def validate_user_id(cls, v):
        from uuid import UUID
        try:
            UUID(v)
            return v
        except:
            raise ValueError(f"Invalid UUID: {v}")


class ConversationDetailResponse(ConversationResponse):
    """
    Full conversation with all messages.
    """
    messages: List[ChatMessageResponse]


class ContextWindowResponse(BaseModel):
    """
    Response containing recent conversation context (for RAG augmentation).
    
    This is what gets passed to the RAG system to augment the query.
    """
    context_string: str = Field(
        ...,
        description="Formatted context (Q: ...\nA: ...)"
    )
    messages: List[ChatMessageResponse] = Field(
        ...,
        description="Raw message objects"
    )
    total_chars: int = Field(
        ...,
        description="Total characters in context"
    )
    total_tokens_estimate: int = Field(
        ...,
        description="Estimated tokens (for token counting)"
    )


class SearchResultsResponse(BaseModel):
    """
    Response from semantic search.
    """
    query: str
    results: List[ChatMessageResponse]
    count: int
    similarity_scores: List[float]


class MemoryStatsResponse(BaseModel):
    """
    Statistics about user's memory usage.
    """
    user_id: str
    total_conversations: int
    total_messages: int
    total_tokens_used: int
    avg_latency_ms: float
    oldest_message_date: Optional[str]
    newest_message_date: Optional[str]
    memory_size_mb: float

    @field_validator('user_id')
    def validate_user_id(cls, v):
        from uuid import UUID
        try:
            UUID(v)
            return v
        except:
            raise ValueError(f"Invalid UUID: {v}")


# ============ Query Service Schemas ============

class QueryWithMemoryRequest(BaseModel):
    """
    Complete query request including memory context.
    
    This is what your RAG endpoint receives.
    """
    user_id: str
    conversation_id: str
    query_text: str = Field(..., description="User's question")
    include_memory: bool = Field(
        True,
        description="Whether to load and use conversation history"
    )
    num_memory_pairs: int = Field(
        2,
        description="Number of previous Q&A pairs to include"
    )

    @field_validator('user_id')
    def validate_user_id(cls, v):
        from uuid import UUID
        try:
            UUID(v)
            return v
        except:
            raise ValueError(f"Invalid UUID: {v}")


class QueryWithMemoryResponse(BaseModel):
    """
    Response from RAG query with memory integration.
    
    Extends normal RAG response with memory metadata.
    """
    response: str = Field(..., description="Generated response")
    sources: List[Dict[str, Any]] = Field(..., description="Retrieved documents")
    message_id: str = Field(..., description="ID of saved response message")
    conversation_id: str
    
    # Memory-specific
    memory_context_used: str = Field(
        ...,
        description="Formatted context that was passed to LLM"
    )
    memory_messages_loaded: int
    latency_ms: int
    tokens_used: int
