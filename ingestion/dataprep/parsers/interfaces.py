from abc import ABC
from dataclasses import dataclass
from typing import Optional, Any, List, Dict




class Parser(ABC):
    def parse(self,path:str) -> List[Dict[str,Any]]:
        raise NotImplementedError


@dataclass
class StatuteSection:
    id:str
    title_num:str
    chapter_num:Optional[str]
    section_num:str
    heading:str
    content:str
    metadata:Dict[str,Any]