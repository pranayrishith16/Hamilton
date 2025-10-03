from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Chunk:
    id:str
    content:str
    metadata:Dict[str,Any]


class Chunker:
    def split(self,pages:List[Dict[str,Any]]) -> List[Chunk]:
        raise NotImplementedError