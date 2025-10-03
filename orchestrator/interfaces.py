from abc import ABC, abstractmethod
from ingestion.dataprep.chunkers.base import Chunk
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np

@dataclass
class QueryResult:
    """Query result with retrieved context and metadata."""
    query: str
    answer: str
    retrieved_chunks: List[Dict[str,Any]]
    metadata: Dict[str, Any]
    score: Optional[float] = None

class Embedder(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

class Generator(ABC):
    """Abstract base class for generation models."""
    
    @abstractmethod
    def generate(self, query: str, context: List[Dict[str,Any]], **kwargs) -> str:
        """Generate answer given query and context."""
        pass

class Memory(ABC):
    """Abstract base class for conversation memory."""
    
    @abstractmethod
    def add_turn(self, query: str, response: str) -> None:
        """Add a conversation turn to memory."""
        pass
    
    @abstractmethod
    def get_context(self) -> str:
        """Get conversation context."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear memory."""
        pass

class Reranker(ABC):
    """Abstract base class for reranker"""

    @abstractmethod
    def rerank(self,query:str,candidates:List[Chunk],k:int=5) -> List[Chunk]:
        pass


class Storage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        """Store value with key."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Any:
        """Retrieve value by key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value by key."""
        pass

class MLOpsBackend(ABC):
    """Abstract base class for MLOps backends."""
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics."""
        pass
    
    @abstractmethod
    def log_artifact(self, path: str, name: str) -> None:
        """Log artifact."""
        pass