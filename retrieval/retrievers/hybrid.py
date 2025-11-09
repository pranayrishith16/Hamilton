from typing import List, Dict, Any
from loguru import logger
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics
import asyncio

class HybridRetriever(Retriever):
    """Optimized hybrid retriever using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, k_rrf: float = 60.0):
        """
        Initialize with RRF parameter.
        
        Args:
            k_rrf: RRF parameter, typically 60. Lower values give more weight to top ranks.
        """
        self.k_rrf = k_rrf
        self.bm25_retriever = None
        self.dense_retriever = None
        self.logger = logger
        
        # Cache retrievers to avoid repeated registry lookups
        self._ensure_retrievers()
    
    def _ensure_retrievers(self):
        """Cache retrievers to avoid registry lookups during each query."""
        if self.bm25_retriever is None:
            self.bm25_retriever = registry.get('bm25_retriever')
        if self.dense_retriever is None:
            self.dense_retriever = registry.get('qdrant_retriever')  # Changed from faiss_retriever
    
    async def retrieve(self, query: str, k: int = 10) -> List[Chunk]:
        """Retrieve using optimized RRF fusion."""
        self._ensure_retrievers()
        with trace_request("retrieve", "hybrid_retriever.retrieve"):
            # Get candidates from both retrievers
            # Use k instead of k*2 to reduce computation
            bm25_task = asyncio.to_thread(self.bm25_retriever.retrieve, query, min(k * 3, 50))
            dense_task = asyncio.to_thread(self.dense_retriever.retrieve, query, min(k * 3, 50))

            # Wait for BOTH at the same time
            bm25_cands, dense_cands = await asyncio.gather(bm25_task, dense_task)
                        
            # Use RRF for more efficient fusion
            rrf_scores = self._compute_rrf_scores(bm25_cands, dense_cands)
            
            # Sort by RRF score and take top-k
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Build final results with optimized chunk creation
            results = []
            chunk_map = self._build_chunk_map(bm25_cands, dense_cands)
            
            for rank, (chunk_id, rrf_score) in enumerate(sorted_results):
                if chunk_id in chunk_map:
                    original_chunk = chunk_map[chunk_id]
                    # Create result chunk with minimal metadata
                    result_chunk = Chunk(
                        id=original_chunk.id,
                        content=original_chunk.content,
                        metadata={
                            **original_chunk.metadata,
                            "rrf_score": float(rrf_score),
                            "rank": rank
                        }
                    )
                    results.append(result_chunk)
            
            log_metrics({"hybrid.retrieved": len(results)})
            return results
    
    def _compute_rrf_scores(self, bm25_cands: List[Chunk], dense_cands: List[Chunk]) -> Dict[str, float]:
        """Compute RRF scores efficiently."""
        rrf_scores = {}

        # Process BM25 candidates
        for rank, chunk in enumerate(bm25_cands):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.k_rrf + rank + 1)

        # Process dense candidates
        for rank, chunk in enumerate(dense_cands):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.k_rrf + rank + 1)

        return rrf_scores
    
    def _build_chunk_map(self, bm25_cands: List[Chunk], dense_cands: List[Chunk]) -> Dict[str, Chunk]:
        """Build a map from chunk IDs to chunks for efficient lookup."""
        chunk_map = {}

        for chunk in bm25_cands:
            chunk_map[chunk.id] = chunk

        for chunk in dense_cands:
            chunk_map[chunk.id] = chunk

        return chunk_map
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "k_rrf": self.k_rrf,
            "bm25_indexed": self.bm25_retriever.get_stats().get("indexed_chunks", 0) if self.bm25_retriever else 0,
            "dense_indexed": self.dense_retriever.get_stats().get("chunks_count", 0) if self.dense_retriever else 0
        }
    
    def build_index(self, chunks):
        """Pass-through to underlying retrievers."""
        pass