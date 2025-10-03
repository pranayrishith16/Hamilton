from typing import List, Dict, Any
from loguru import logger
import numpy as np
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics

class HybridRetriever(Retriever):
    """Combines BM25 and dense retriever"""

    def __init__(self,alpha:float=0.5):

        self.alpha = alpha
        self.bm25_retriever = None
        self.dense_retriever = None
        self.logger = logger

    def _ensure_retrievers(self):
        if self.bm25_retriever is None:
            self.bm25_retriever = registry.get('bm25_retriever')
        if self.dense_retriever is None:
            self.dense_retriever = registry.get('faiss_retriever')
    
    def retrieve(self, query:str, k:int = 10) -> List[Chunk]:
        self._ensure_retrievers()
        with trace_request("retrieve", "hybrid_retriever.retrieve"):
            # Get top candidates from each
            bm25_cands = self.bm25_retriever.retrieve(query, k=k * 2)
            dense_cands = self.dense_retriever.retrieve(query, k=k * 2)

            # Build a map from chunk id to best scores
            scores: Dict[str, Dict[str, float]] = {}
            for chunk in bm25_cands:
                scores.setdefault(chunk.id, {})["bm25"] = chunk.metadata.get("bm25_score", 0.0)
            for chunk in dense_cands:
                scores.setdefault(chunk.id, {})["dense"] = chunk.metadata.get("retrieval_score", 0.0)

            # Normalize scores to [0,1]
            bm25_vals = [v.get("bm25", 0.0) for v in scores.values()]
            dense_vals = [v.get("dense", 0.0) for v in scores.values()]
            bm25_min, bm25_max = min(bm25_vals, default=0), max(bm25_vals, default=1)
            dense_min, dense_max = min(dense_vals, default=0), max(dense_vals, default=1)

            fused: List[Dict[str, Any]] = []
            for chunk_id, v in scores.items():
                bm = (v.get("bm25", 0.0) - bm25_min) / (bm25_max - bm25_min + 1e-8)
                dn = (v.get("dense", 0.0) - dense_min) / (dense_max - dense_min + 1e-8)
                final_score = self.alpha * bm + (1 - self.alpha) * dn

                # Retrieve the actual Chunk object (from either retriever)
                chunk_obj = next(
                    (c for c in bm25_cands + dense_cands if c.id == chunk_id),
                    None
                )
                if chunk_obj:
                    fused.append({
                        "chunk": chunk_obj,
                        "score": final_score
                    })

            # Sort by fused score descending, take top-k
            fused.sort(key=lambda x: x["score"], reverse=True)
            top_k = fused[:k]

            # Attach fused score and rank in metadata
            results: List[Chunk] = []
            for rank, entry in enumerate(top_k):
                chunk = entry["chunk"]
                chunk_copy = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "hybrid_score": float(entry["score"]),
                        "rank": rank
                    }
                )
                results.append(chunk_copy)

            log_metrics({"hybrid.retrieved": len(results)})
            return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "bm25_indexed": self.bm25_retriever.get_stats().get("indexed_chunks", 0),
            "dense_indexed": self.dense_retriever.get_stats().get("chunks_count", 0)
        }
    
    def build_index(self, chunks):
        pass