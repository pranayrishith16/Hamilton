from abc import ABC
from dataclasses import dataclass
from typing import Optional, Any, List, Dict

@dataclass
class RawPage:
    page_number:int
    text:str
    metadata:Optional[Dict[str,Any]] = None

class Parser(ABC):
    def parse(self,path:str) -> List[RawPage]:
        raise NotImplementedError