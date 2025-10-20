from typing import List, Dict, Any
from loguru import logger
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics

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
            self.dense_retriever = registry.get('faiss_retriever')
    
    def retrieve(self, query: str, k: int = 10) -> List[Chunk]:
        """Retrieve using optimized RRF fusion."""
        with trace_request("retrieve", "hybrid_retriever.retrieve"):
            # Get candidates from both retrievers
            # Use k instead of k*2 to reduce computation
            bm25_cands = self.bm25_retriever.retrieve(query, k=min(k * 3, 50))  # Cap max candidates
            dense_cands = self.dense_retriever.retrieve(query, k=min(k * 3, 50))
            
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
        
        # Add BM25 candidates
        for chunk in bm25_cands:
            chunk_map[chunk.id] = chunk
        
        # Add dense candidates (will overwrite if same ID, which is fine)
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


class WeightedHybridRetriever(Retriever):
    """Alternative hybrid approach with weighted combination - more traditional but optimized."""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize with alpha weighting.
        
        Args:
            alpha: Weight for BM25 scores (1-alpha for dense scores)
        """
        self.alpha = alpha
        self.bm25_retriever = None
        self.dense_retriever = None
        self.logger = logger
        
        # Pre-cache retrievers
        self._ensure_retrievers()
    
    def _ensure_retrievers(self):
        """Cache retrievers to avoid registry lookups."""
        if self.bm25_retriever is None:
            self.bm25_retriever = registry.get('bm25_retriever')
        if self.dense_retriever is None:
            self.dense_retriever = registry.get('faiss_retriever')
    
    def retrieve(self, query: str, k: int = 10) -> List[Chunk]:
        """Optimized weighted retrieval."""
        with trace_request("retrieve", "weighted_hybrid_retriever.retrieve"):
            # Get candidates
            bm25_cands = self.bm25_retriever.retrieve(query, k=min(k * 2, 40))
            dense_cands = self.dense_retriever.retrieve(query, k=min(k * 2, 40))
            
            # Extract scores efficiently
            bm25_scores = {chunk.id: chunk.metadata.get("bm25_score", 0.0) for chunk in bm25_cands}
            dense_scores = {chunk.id: chunk.metadata.get("retrieval_score", 0.0) for chunk in dense_cands}
            
            # Get all unique chunk IDs
            all_chunk_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
            
            # Normalize scores using min-max normalization
            bm25_vals = list(bm25_scores.values())
            dense_vals = list(dense_scores.values())
            
            bm25_min, bm25_max = (min(bm25_vals), max(bm25_vals)) if bm25_vals else (0, 1)
            dense_min, dense_max = (min(dense_vals), max(dense_vals)) if dense_vals else (0, 1)
            
            bm25_range = bm25_max - bm25_min + 1e-8
            dense_range = dense_max - dense_min + 1e-8
            
            # Compute fused scores
            fused_scores = {}
            for chunk_id in all_chunk_ids:
                bm25_norm = (bm25_scores.get(chunk_id, 0.0) - bm25_min) / bm25_range
                dense_norm = (dense_scores.get(chunk_id, 0.0) - dense_min) / dense_range
                fused_scores[chunk_id] = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm
            
            # Sort and get top-k
            sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Build result chunks
            chunk_map = self._build_chunk_map(bm25_cands, dense_cands)
            results = []
            
            for rank, (chunk_id, score) in enumerate(sorted_results):
                if chunk_id in chunk_map:
                    original_chunk = chunk_map[chunk_id]
                    result_chunk = Chunk(
                        id=original_chunk.id,
                        content=original_chunk.content,
                        metadata={
                            **original_chunk.metadata,
                            "hybrid_score": float(score),
                            "rank": rank
                        }
                    )
                    results.append(result_chunk)
            
            log_metrics({"weighted_hybrid.retrieved": len(results)})
            return results
    
    def _build_chunk_map(self, bm25_cands: List[Chunk], dense_cands: List[Chunk]) -> Dict[str, Chunk]:
        """Build chunk ID to chunk mapping."""
        chunk_map = {}
        for chunk in bm25_cands + dense_cands:
            chunk_map[chunk.id] = chunk
        return chunk_map
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "bm25_indexed": self.bm25_retriever.get_stats().get("indexed_chunks", 0) if self.bm25_retriever else 0,
            "dense_indexed": self.dense_retriever.get_stats().get("chunks_count", 0) if self.dense_retriever else 0
        }
    
    def build_index(self, chunks):
        pass