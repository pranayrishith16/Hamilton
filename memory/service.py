"""
Business logic for memory system.

The service layer sits between API endpoints and repositories.
It handles:
- Orchestrating complex operations
- Validating business rules
- Formatting responses
- Handling errors gracefully
"""

from sqlalchemy.orm import Session
from uuid import UUID
from typing import List, Optional, Dict, Any, Tuple
import logging
from datetime import datetime

from memory.repository import ConversationRepository, ChatMessageRepository
from memory.schemas import (
    ContextWindowResponse, ChatMessageResponse, SearchResultsResponse
)
from memory.utils import format_context_string, estimate_tokens, rebuild_context_string

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Business logic for conversation operations.
    
    Methods handle creating conversations, listing, updating, etc.
    """
    
    @staticmethod
    def create_conversation(
        db: Session,
        user_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation.
        
        Args:
            db: Database session
            user_id: User UUID (as string)
            title: Conversation title
            description: Conversation description
        
        Returns:
            Dictionary with conversation data
        
        Example:
            result = ConversationService.create_conversation(
                db,
                user_id="550e8400-e29b-41d4-a716-446655440000",
                title="SEC Filing Q3 2024"
            )
            # {
            #     'id': 'conv-uuid',
            #     'user_id': 'user-uuid',
            #     'title': 'SEC Filing Q3 2024',
            #     'message_count': 0,
            #     'created_at': '2024-11-01T...'
            # }
        """
        try:
            conversation = ConversationRepository.create(
                db=db,
                user_id=UUID(user_id),
                title=title,
                description=description
            )
            
            logger.info(f"Created conversation {conversation.id} for user {user_id}")
            
            return conversation.to_dict()
        
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    @staticmethod
    def get_conversation(
        db: Session,
        conversation_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get conversation details (without messages).
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            user_id: User UUID (for access control)
        
        Returns:
            Conversation data or None if not found/unauthorized
        """
        try:
            conversation = ConversationRepository.get_by_id(
                db=db,
                conversation_id=UUID(conversation_id),
                user_id=UUID(user_id)  # Verify ownership
            )
            
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found for user {user_id}")
                return None
            
            return conversation.to_dict()
        
        except Exception as e:
            logger.error(f"Error fetching conversation: {e}")
            raise
    
    @staticmethod
    def list_conversations(
        db: Session,
        user_id: str,
        skip: int = 0,
        limit: int = 20,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all conversations for a user (paginated).
        
        Args:
            db: Database session
            user_id: User UUID
            skip: Pagination offset
            limit: Page size (max 100)
            include_archived: Include archived conversations
        
        Returns:
            List of conversation dictionaries
        """
        # Enforce max limit for performance
        limit = min(limit, 100)
        
        conversations = ConversationRepository.list_by_user(
            db=db,
            user_id=UUID(user_id),
            skip=skip,
            limit=limit,
            include_archived=include_archived
        )
        
        return [c.to_dict() for c in conversations]
    
    @staticmethod
    def archive_conversation(
        db: Session,
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Archive a conversation (soft delete).
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            user_id: User UUID (for access control)
        
        Returns:
            Updated conversation data
        """
        # Verify ownership
        conversation = ConversationRepository.get_by_id(
            db=db,
            conversation_id=UUID(conversation_id),
            user_id=UUID(user_id)
        )
        
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        updated = ConversationRepository.archive(db, UUID(conversation_id))
        logger.info(f"Archived conversation {conversation_id}")
        
        return updated.to_dict()
    
    @staticmethod
    def load_context_with_token_limit(
        db: Session,
        conversation_id: str,
        user_id: str,
        max_tokens: int = 2000,
        max_messages: int = 10
    ) -> Tuple[str, int, int]:
        """
        Load context with intelligent token limiting.
        
        Automatically removes oldest messages if token count exceeds limit.
        
        Args:
            db: Database session
            conversation_id: UUID of conversation
            user_id: UUID of user (for security check)
            max_tokens: Maximum tokens allowed (default: 2000)
            max_messages: Maximum messages to consider (default: 10)
        
        Returns:
            Tuple of (context_string, token_count, messages_loaded)
        
        Raises:
            ValueError: If conversation not found or access denied
        """
        try:
            from uuid import UUID
            
            # Verify conversation exists and belongs to user
            conversation = ConversationRepository.get_by_id(
            db=db,
            conversation_id=UUID(conversation_id),
            user_id=UUID(user_id)
        )
            
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return "", 0, 0
            
            # Security: verify user owns this conversation
            if str(conversation.user_id) != str(user_id):
                logger.warning(
                    f"User {user_id} attempted to access conversation {conversation_id}"
                )
                raise ValueError("Access denied: Conversation does not belong to user")
            
            # Load recent messages
            messages = ChatMessageRepository.list_by_conversation(
                db=db,
                conversation_id=UUID(conversation_id),
                limit=max_messages
            )
            
            if not messages:
                logger.debug(f"No messages in conversation {conversation_id}")
                return "<NO_CONTEXT>", 0, 0
            
            # Build initial context
            context_string = rebuild_context_string(messages)
            token_count = estimate_tokens(context_string)
            original_token_count = token_count
            
            logger.debug(
                f"Initial context: {len(messages)} messages, "
                f"{token_count} tokens (limit: {max_tokens})"
            )
            
            # Drop oldest messages if exceeding token limit
            dropped_pairs = 0
            if token_count > max_tokens and len(messages) > 0:
                # Binary search for max messages that fit
                messages_to_keep = len(messages)
                while token_count > max_tokens and messages_to_keep > 0:
                    messages_to_keep -= 2  # Remove pairs
                    messages_subset = messages[-messages_to_keep:] if messages_to_keep > 0 else []
                    context_string = rebuild_context_string(messages_subset)
                    token_count = estimate_tokens(context_string)
                
                if messages_to_keep > 0:
                    messages = messages[-messages_to_keep:]
                else:
                    messages = []
            
            # Log final status
            messages_loaded = len(messages)
            tokens_saved = original_token_count - token_count
            
            if dropped_pairs > 0:
                logger.info(
                    f"Conversation {conversation_id}: "
                    f"Dropped {dropped_pairs} message pairs to stay under "
                    f"{max_tokens} tokens. "
                    f"Original: {original_token_count}, Final: {token_count}, "
                    f"Saved: {tokens_saved} tokens ({messages_loaded} messages loaded)"
                )
            else:
                logger.debug(
                    f"Conversation {conversation_id}: "
                    f"All {messages_loaded} messages fit within {max_tokens} token limit"
                )
            
            return context_string, token_count, messages_loaded
            
        except Exception as e:
            logger.error(f"Error loading context with token limit: {e}")
            raise


class ChatMessageService:
    """
    Business logic for chat messages and memory operations.
    """
    
    @staticmethod
    def save_message(
        db: Session,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Save a message to the conversation.
        
        Called by RAG pipeline after query generation.
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            user_id: User UUID
            role: 'user' or 'assistant'
            content: Message text
            sources: Retrieved documents (for assistant)
            embedding: Vector embedding
            metadata: Custom metadata
            tokens_used: Token count
            latency_ms: Generation time
        
        Returns:
            Saved message data
        
        Example:
            ChatMessageService.save_message(
                db=db,
                conversation_id="conv-id",
                user_id="user-id",
                role='assistant',
                content="Insider trading is...",
                sources=[{
                    'source': 'United States v. O'Brien',
                    'text': '...',
                    'confidence': 0.95
                }],
                latency_ms=2300,
                tokens_used=1500
            )
        """
        try:
            message = ChatMessageRepository.create(
                db=db,
                conversation_id=UUID(conversation_id),
                user_id=UUID(user_id),
                role=role,
                content=content,
                sources=sources,
                embedding=embedding,
                metadata=metadata,
                tokens_used=tokens_used,
                latency_ms=latency_ms
            )
            
            logger.info(
                f"Saved {role} message in conversation {conversation_id} "
                f"({tokens_used} tokens, {latency_ms}ms)"
            )
            
            return message.to_dict()
        
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise
    
    @staticmethod
    def load_context(
        db: Session,
        conversation_id: str,
        user_id: str,
        num_pairs: int = 2,
        max_context_tokens: int = 2000
    ) -> ContextWindowResponse:
        """
        Load recent conversation context for RAG augmentation.
        
        This is called BEFORE generating a response to load past context.
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            user_id: User UUID (for verification)
            num_pairs: Number of Q&A pairs to load (e.g., 2 = last 4 messages)
        
        Returns:
            ContextWindowResponse with formatted context string and metadata
        
        Example:
            context = ChatMessageService.load_context(
                db=db,
                conversation_id="conv-id",
                user_id="user-id",
                num_pairs=2
            )
            
            # Returns:
            # {
            #     'context_string': 'Q: What is insider trading?\n...',
            #     'messages': [...],
            #     'total_chars': 1200,
            #     'total_tokens_estimate': 280
            # }
            
            # Now include context_string in your RAG prompt:
            # augmented_query = f"{context.context_string}\n\nCurrent Q: {user_query}"
        """
        try:
            # Verify conversation belongs to user
            conversation = ConversationRepository.get_by_id(
                db=db,
                conversation_id=UUID(conversation_id),
                user_id=UUID(user_id)
            )
            
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return ContextWindowResponse(
                    context_string="",
                    messages=[],
                    total_chars=0,
                    total_tokens_estimate=0
                )
            
            # Load recent messages (num_pairs * 2 = pairs converted to individual messages)
            messages = ChatMessageRepository.get_recent(
                db=db,
                conversation_id=UUID(conversation_id),
                count=num_pairs * 2
            )
            
            # Format messages
            context_string = format_context_string(messages)
            token_estimate = estimate_tokens(context_string)
            
            # NEW: Drop oldest messages if too many tokens
            while token_estimate > max_context_tokens and len(messages) > 2:
                messages = messages[2:]  # Remove oldest Q&A pair
                context_string = format_context_string(messages)
                token_estimate = estimate_tokens(context_string)
                logger.debug(f"Reduced context to {token_estimate} tokens")
            
            # Convert to response format
            message_responses = [
                ChatMessageResponse.from_orm(m) for m in messages
            ]
            
            logger.debug(
                f"Loaded context for conversation {conversation_id}: "
                f"{len(messages)} messages, {token_estimate} tokens"
            )
            
            return ContextWindowResponse(
                context_string=context_string,
                messages=message_responses,
                total_chars=len(context_string),
                total_tokens_estimate=token_estimate
            )
        
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            raise
    
    @staticmethod
    def get_conversation_history(
        db: Session,
        conversation_id: str,
        user_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get full conversation history (all messages).
        
        Args:
            db: Database session
            conversation_id: Conversation UUID
            user_id: User UUID (for verification)
            skip: Pagination offset
            limit: Page size
        
        Returns:
            List of message dictionaries
        """
        # Verify ownership
        conversation = ConversationRepository.get_by_id(
            db=db,
            conversation_id=UUID(conversation_id),
            user_id=UUID(user_id)
        )
        
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found")
            return []
        
        messages = ChatMessageRepository.list_by_conversation(
            db=db,
            conversation_id=UUID(conversation_id),
            skip=skip,
            limit=limit,
            reverse_order=False  # Chronological order
        )
        
        return [m.to_dict() for m in messages]
    
    @staticmethod
    def search_memory(
        db: Session,
        user_id: str,
        query: str,
        limit: int = 5,
        days: int = 30
    ) -> SearchResultsResponse:
        """
        Search user's past messages (simple text search, can upgrade to semantic).
        
        Args:
            db: Database session
            user_id: User UUID
            query: Search query
            limit: Max results
            days: Search within last N days
        
        Returns:
            SearchResultsResponse with matching messages
        
        Note:
            Current implementation is basic text search.
            Future: Add semantic search using message embeddings.
        """
        try:
            # Get user's recent messages
            messages = ChatMessageRepository.search_by_user(
                db=db,
                user_id=UUID(user_id),
                role='user',  # Only search user messages
                days=days,
                limit=limit * 3  # Get more, then filter
            )
            
            # Simple text search (case-insensitive)
            query_lower = query.lower()
            results = [
                m for m in messages
                if query_lower in m.content.lower()
            ][:limit]
            
            logger.info(f"Found {len(results)} search results for user {user_id}")
            
            return SearchResultsResponse(
                query=query,
                results=[ChatMessageResponse.from_orm(m) for m in results],
                count=len(results),
                similarity_scores=[1.0] * len(results)  # Placeholder
            )
        
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            raise
    
    @staticmethod
    def get_memory_stats(
        db: Session,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get usage statistics for a user.
        
        Args:
            db: Database session
            user_id: User UUID
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = ChatMessageRepository.get_stats(
                db=db,
                user_id=UUID(user_id)
            )
            
            return {
                'user_id': user_id,
                **stats
            }
        
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise
