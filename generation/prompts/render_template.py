from typing import Dict, List, Any
import yaml
from jinja2 import Template

from ingestion.dataprep.chunkers.base import Chunk


with open("configs/prompts.yaml") as f:
    prompts: Dict[str, Dict[str, str]] = yaml.safe_load(f)


def render_messages(query: str, context: List[Chunk]) -> List[Dict[str, Any]]:
    """
    Render a system+user message pair using the specified prompt template.
    """
    # Select the template configuration
    cfg = prompts["legal_rag_metadata"]

    # Render the system message (no variables)
    system_msg = Template(cfg["system"]).render()
    
    # Prepare data for the user template
    chunk_dicts = []
    for chunk in context:
        chunk_dicts.append({
            "metadata": {
                "source": chunk.metadata.get("source", ""),
                "page_number": chunk.metadata.get("page_number", ""),
                "holdings": chunk.metadata.get("holdings",""),
                "disposition": chunk.metadata.get("disposition",""),
                "statutes_guidelines":chunk.metadata.get("statutes_guidelines",""),
                "court": chunk.metadata.get("court", ""),
                "date": chunk.metadata.get("date", ""),
                "parties": chunk.metadata.get("parties",""),
                "case_name": chunk.metadata.get("case_name",""),
                "procedural_posture":chunk.metadata.get("procedural_posture","")
            },
            "content": chunk.content
        })
    
    # Render the user message, passing in chunks and query
    user_msg = Template(cfg["user"]).render(chunks=chunk_dicts, query=query)
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
