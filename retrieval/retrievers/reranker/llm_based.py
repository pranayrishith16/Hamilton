# retrieval/rerankers/llm_reranker.py

from typing import List
from openai import OpenAI
from orchestrator.interfaces import Reranker
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.observability import trace_request, log_metrics

class LLMBasedReranker(Reranker):
    """LLM-based reranker using OpenAI scoring via prompt."""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b:free"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def rerank(self, query: str, candidates: List[Chunk], k: int = 5) -> List[Chunk]:
        """Ask the LLM to score each candidate’s relevance."""
        with trace_request("rerank", "llm_reranker.rerank"):
            scored = []
            for chunk in candidates:
                prompt = (
                    f"Query: {query}\n"
                    f"Passage: {chunk.content}\n"
                    "On a scale of 1–10, how relevant is this passage to the query?"
                )
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=0,
                    temperature=0
                )
                # Assume response is a number
                score = float(resp.choices[0].message["content"].strip())
                scored.append((chunk, score))
            
            ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
            results = []
            for rank, (chunk, score) in enumerate(ranked):
                chunk_copy = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={**chunk.metadata, "llm_score": score, "rank": rank}
                )
                results.append(chunk_copy)
            
            log_metrics({"rerank.llm": len(results)})
            return results
