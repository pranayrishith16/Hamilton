from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Retriever(ABC):
    """Abstract base class for all retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str,Any]]:
        """Retrieve k most relevant chunks for the query."""
        pass
    
    @abstractmethod
    def build_index(self, chunks: List[Dict[str,Any]]) -> None:
        """Build or update the retrieval index."""
        pass
