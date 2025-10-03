from abc import ABC, abstractmethod
import os
import httpx
from typing import List, Dict, Any, Optional
from loguru import logger

from ingestion.dataprep.chunkers.base import Chunk

class GeneratorAdapter(ABC):
    @abstractmethod
    def generate(self,query:str,context:List[Chunk],**kwargs) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def get_model_info(self) -> Dict[str,Any]:
        raise NotImplementedError