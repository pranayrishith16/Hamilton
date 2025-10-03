# generation/postprocessor/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ingestion.dataprep.chunkers.base import Chunk

class PostProcessorAdapter(ABC):
    """
    Abstract base for post‐processing generation output.
    """

    @abstractmethod
    def process(self, raw_answer: str, context: List[Chunk]) -> str:
        """
        Given the raw generated answer and its metadata,
        return a cleaned or enhanced version.
        """
        raise NotImplementedError
