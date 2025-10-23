from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LoaderConfig:
    """Configuration for loaders"""
    batch_size: int = 50
    file_extensions: Optional[List[str]] = None
    skip_errors: bool = True

class BaseLoader(ABC):
    """
    Abstract base class for data loaders.
    Loaders implement lazy_load (streaming) and batch_load methods.
    """
    
    def __init__(self, config: LoaderConfig = None):
        self.config = config or LoaderConfig()
    
    @abstractmethod
    def lazy_load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Lazy load files one at a time (streaming).
        Yields: {'file_name': str, 'content': bytes}
        """
        pass
    
    def load(self) -> List[Dict[str, Any]]:
        """Eagerly load all files into memory."""
        return list(self.lazy_load())
    
    def batch_load(self, batch_size: Optional[int] = None) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Batch lazy loading - yields batches of files.
        Ideal for RAG ingestion pipelines.
        """
        batch_size = batch_size or self.config.batch_size
        batch = []
        
        for file_data in self.lazy_load():
            batch.append(file_data)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch