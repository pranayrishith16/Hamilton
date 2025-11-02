"""
Utility functions for memory system.

Handles:
- Formatting context strings
- Token estimation
- Text truncation
- Validation
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


def format_context_string(messages: List) -> str:
    """
    Format chat messages as a context string for prompt augmentation.
    
    Args:
        messages: List of ChatMessage objects (in chronological order)
    
    Returns:
        Formatted string like:
            Q: What is insider trading?
            A: Insider trading is... (truncated)
            
            Q: What are penalties?
            A: Penalties include... (truncated)
    
    Example:
        >>> from memory.models import ChatMessage
        >>> messages = [user_msg, assistant_msg, user_msg2, assistant_msg2]
        >>> context = format_context_string(messages)
        >>> print(context)
        Q: What is insider trading?
        A: Insider trading is the purchase or sale of a security by someone...
        
        Q: What are the penalties?
        A: Penalties can include fines and imprisonment...
    """
    context_parts = []
    
    for msg in messages:
        if msg.role == 'user':
            context_parts.append(f"Q: {msg.content}")
        else:
            # Truncate long responses to save tokens
            content = truncate_text(msg.content, max_chars=300)
            context_parts.append(f"A: {content}")
    
    # Join with double newlines for clarity
    context_string = "\n\n".join(context_parts)
    
    logger.debug(f"Formatted context string: {len(context_string)} chars")
    return context_string


def truncate_text(text: str, max_chars: int = 300) -> str:
    """
    Truncate text and add ellipsis if needed.
    
    Args:
        text: Text to truncate
        max_chars: Maximum length
    
    Returns:
        Truncated text with "..." if truncated
    
    Example:
        >>> truncate_text("Hello world this is a very long text...", max_chars=20)
        'Hello world this i...'
    """
    if len(text) <= max_chars:
        return text
    
    return text[:max_chars].rstrip() + "..."


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Simple approximation: ~1 token ≈ 4 characters (for English text).
    For precise counting, use tiktoken library.
    
    Args:
        text: Text to estimate tokens for
    
    Returns:
        Estimated token count
    
    Example:
        >>> estimate_tokens("Hello world")  # 2 words ≈ 3 tokens
        3
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to rough estimation
        return max(1, len(text) // 4)


def count_messages_tokens(messages: List) -> int:
    """
    Count total tokens in a list of messages.
    
    Args:
        messages: List of ChatMessage objects
    
    Returns:
        Total token count
    """
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
    return total


def rebuild_context_string(messages: List) -> str:
    """
    Rebuild context string from messages.
    
    Args:
        messages: List of ChatMessage objects
    
    Returns:
        Formatted context string
    """
    if not messages:
        return ""
    context = ""
    for msg in messages:
        if not msg or not msg.content:
            continue
        
        role = msg.role.upper() if msg.role else "UNKNOWN"
        context += f"{role}: {msg.content}\n\n"
    
    return context


def validate_uuid(value: str) -> bool:
    """
    Validate if string is a valid UUID.
    
    Args:
        value: String to validate
    
    Returns:
        True if valid UUID, False otherwise
    
    Example:
        >>> validate_uuid("550e8400-e29b-41d4-a716-446655440000")
        True
        >>> validate_uuid("not-a-uuid")
        False
    """
    try:
        from uuid import UUID
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def validate_role(value: str) -> bool:
    """
    Validate message role.
    
    Args:
        value: Role string
    
    Returns:
        True if 'user' or 'assistant', False otherwise
    """
    return value in ('user', 'assistant', 'system')


def estimate_storage_mb(message_count: int) -> float:
    """
    Estimate database storage size in MB.
    
    Rough estimate:
    - Each message ≈ 5KB (content + metadata)
    - Indexes add ~30%
    
    Args:
        message_count: Number of messages
    
    Returns:
        Estimated storage in MB
    
    Example:
        >>> estimate_storage_mb(10000)
        55.0  # ~55 MB for 10k messages
    """
    bytes_per_message = 5000  # 5KB average
    total_bytes = message_count * bytes_per_message
    
    # Add 30% for indexes
    total_with_indexes = total_bytes * 1.3
    
    return total_with_indexes / (1024 * 1024)
