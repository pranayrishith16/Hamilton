import re
from typing import List, Dict, Any, Tuple
from loguru import logger
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics

class HybridRetriever(Retriever):
    """Optimized hybrid retriever using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, fusion_method: str = "rrf", k_rrf: float = 60.0, alpha: float = 0.5, ):
        """
        Initialize hybrid retriever with multiple fusion strategies.
        
        Args:
            fusion_method: rrf, weighted, convex, or adaptive
            k_rrf: RRF parameter for rank-based fusion
            alpha: Weight for BM25 scores in weighted fusion (1-alpha for dense)
            enable_legal_boost: Enable legal-specific score boosting
            enable_query_routing: Enable query-type based routing
        """
        self.fusion_method = fusion_method
        self.k_rrf = k_rrf
        self.alpha = alpha
        
        self.bm25_retriever = None
        self.dense_retriever = None
        self.logger = logger
        
        # Legal patterns for boosting
        self.legal_patterns = {
            'citation': [
                r'\b\d+\s+U\.S\.C\.\s*§?\s*\d+',
                r'\b\d+\s+F\.\s*\d+d?\s*\d+',
                r'\bNo\.\s*\d+[-–]\d+'
            ],
            'procedural': [
                r'\b(?:motion|petition|appeal|writ)\b',
                r'\b(?:summary judgment|injunction|mandamus)\b'
            ],
            'legal_terms': [
                r'\b(?:negligence|breach|liability|damages)\b',
                r'\b(?:contract|tort|constitutional|statutory)\b'
            ]
        }
        
        # Cache retrievers
        self._ensure_retrievers()
    
    def _ensure_retrievers(self):
        """Cache retrievers to avoid registry lookups during each query."""
        if self.bm25_retriever is None:
            self.bm25_retriever = registry.get('bm25_retriever')
        if self.dense_retriever is None:
            self.dense_retriever = registry.get('faiss_retriever')

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for routing decisions."""
        query_lower = query.lower()
        
        # Citation queries - favor BM25
        for pattern_list in self.legal_patterns['citation']:
            if re.search(pattern_list, query, re.IGNORECASE):
                return "citation"
        
        # Procedural queries - balanced
        for pattern_list in self.legal_patterns['procedural']:
            if re.search(pattern_list, query, re.IGNORECASE):
                return "procedural"
        
        # Conceptual queries - favor dense
        if any(word in query_lower for word in ['what', 'how', 'why', 'explain', 'analyze']):
            return "conceptual"
        
        # Default
        return "general"

    def _route_query(self, query: str, k: int) -> Tuple[List[Chunk], List[Chunk]]:
        """Route queries to appropriate retrievers based on query type."""
        if not self.enable_query_routing:
            # Standard approach - get candidates from both
            bm25_cands = self.bm25_retriever.retrieve(query, k=min(k * 3, 50))
            dense_cands = self.dense_retriever.retrieve(query, k=min(k * 3, 50))
            return bm25_cands, dense_cands
        
        query_type = self._classify_query_type(query)
        
        if query_type == "citation":
            # Citation queries: Heavy BM25, light dense
            bm25_cands = self.bm25_retriever.retrieve(query, k=min(k * 4, 60))
            dense_cands = self.dense_retriever.retrieve(query, k=min(k * 2, 30))
        elif query_type == "conceptual":
            # Conceptual queries: Heavy dense, light BM25
            bm25_cands = self.bm25_retriever.retrieve(query, k=min(k * 2, 30))
            dense_cands = self.dense_retriever.retrieve(query, k=min(k * 4, 60))
        else:
            # Balanced approach
            bm25_cands = self.bm25_retriever.retrieve(query, k=min(k * 3, 50))
            dense_cands = self.dense_retriever.retrieve(query, k=min(k * 3, 50))
        
        return bm25_cands, dense_cands
    
    def retrieve(self, query: str, k: int = 10) -> List[Chunk]:
        """Enhanced retrieve with multiple fusion strategies."""
        with trace_request("retrieve", f"enhanced_hybrid_retriever.retrieve_{self.fusion_method}"):
            # Route query and get candidates
            bm25_cands, dense_cands = self._route_query(query, k)
            
            # Apply fusion strategy
            if self.fusion_method == "rrf":
                results = self._rrf_fusion(bm25_cands, dense_cands, k)
            elif self.fusion_method == "weighted":
                results = self._weighted_fusion(bm25_cands, dense_cands, k)
            elif self.fusion_method == "convex":
                results = self._convex_fusion(bm25_cands, dense_cands, k)
            elif self.fusion_method == "adaptive":
                results = self._adaptive_fusion(query, bm25_cands, dense_cands, k)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
            # Apply legal boosting
            if self.enable_legal_boost:
                results = self._apply_legal_boost(query, results)
            
            log_metrics({
                f"hybrid_{self.fusion_method}.retrieved": len(results),
                "hybrid.bm25_candidates": len(bm25_cands),
                "hybrid.dense_candidates": len(dense_cands)
            })
            
            return results
        
        def _rrf_fusion(self, bm25_cands: List[Chunk], dense_cands: List[Chunk], k: int) -> List[Chunk]:
            """Reciprocal Rank Fusion - your current method."""
        rrf_scores = {}
        
        # Process BM25 candidates
        for rank, chunk in enumerate(bm25_cands):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        
        # Process dense candidates
        for rank, chunk in enumerate(dense_cands):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        
        # Sort and build results
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        chunk_map = self._build_chunk_map(bm25_cands, dense_cands)
        
        results = []
        for rank, (chunk_id, rrf_score) in enumerate(sorted_results):
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                result_chunk = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "rrf_score": float(rrf_score),
                        "fusion_method": "rrf",
                        "rank": rank
                    }
                )
                results.append(result_chunk)
        
        return results
    
    def _weighted_fusion(self, bm25_cands: List[Chunk], dense_cands: List[Chunk], k: int) -> List[Chunk]:
        """Weighted score fusion with normalization."""
        # Extract and normalize scores
        bm25_scores = {chunk.id: chunk.metadata.get("bm25_score", 0.0) for chunk in bm25_cands}
        dense_scores = {chunk.id: chunk.metadata.get("retrieval_score", 0.0) for chunk in dense_cands}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        # Normalize scores using min-max
        bm25_vals = list(bm25_scores.values())
        dense_vals = list(dense_scores.values())
        
        if bm25_vals:
            bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
            bm25_range = bm25_max - bm25_min + 1e-8
        else:
            bm25_min, bm25_range = 0, 1
        
        if dense_vals:
            dense_min, dense_max = min(dense_vals), max(dense_vals)
            dense_range = dense_max - dense_min + 1e-8
        else:
            dense_min, dense_range = 0, 1
        
        # Compute fused scores
        fused_scores = {}
        for chunk_id in all_chunk_ids:
            bm25_norm = (bm25_scores.get(chunk_id, 0.0) - bm25_min) / bm25_range
            dense_norm = (dense_scores.get(chunk_id, 0.0) - dense_min) / dense_range
            fused_scores[chunk_id] = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm
        
        # Sort and build results
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        chunk_map = self._build_chunk_map(bm25_cands, dense_cands)
        
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_results):
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                result_chunk = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "weighted_score": float(score),
                        "fusion_method": "weighted",
                        "rank": rank
                    }
                )
                results.append(result_chunk)
        
        return results
    
    def _convex_fusion(self, bm25_cands: List[Chunk], dense_cands: List[Chunk], k: int) -> List[Chunk]:
        """Convex combination using rank-based weights."""
        # Convert to rank-based scores (higher rank = lower score)
        bm25_rank_scores = {chunk.id: 1.0 / (rank + 1) for rank, chunk in enumerate(bm25_cands)}
        dense_rank_scores = {chunk.id: 1.0 / (rank + 1) for rank, chunk in enumerate(dense_cands)}
        
        all_chunk_ids = set(bm25_rank_scores.keys()) | set(dense_rank_scores.keys())
        
        # Convex combination with automatic weight adjustment
        fused_scores = {}
        for chunk_id in all_chunk_ids:
            bm25_score = bm25_rank_scores.get(chunk_id, 0.0)
            dense_score = dense_rank_scores.get(chunk_id, 0.0)
            
            # Dynamic weighting based on score presence
            if bm25_score > 0 and dense_score > 0:
                # Both retrievers found it - balanced weight
                weight = 0.5
            elif bm25_score > 0:
                # Only BM25 found it - slight penalty
                weight = 0.7
            else:
                # Only dense found it - slight penalty
                weight = 0.3
            
            fused_scores[chunk_id] = weight * bm25_score + (1 - weight) * dense_score
        
        # Sort and build results
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        chunk_map = self._build_chunk_map(bm25_cands, dense_cands)
        
        results = []
        for rank, (chunk_id, score) in enumerate(sorted_results):
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                result_chunk = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "convex_score": float(score),
                        "fusion_method": "convex",
                        "rank": rank
                    }
                )
                results.append(result_chunk)
        
        return results
    
    def _adaptive_fusion(self, query: str, bm25_cands: List[Chunk], dense_cands: List[Chunk], k: int) -> List[Chunk]:
        """Adaptive fusion that changes strategy based on query characteristics."""
        query_type = self._classify_query_type(query)
        
        # Adapt fusion weights based on query type
        if query_type == "citation":
            # Heavy BM25 weight for citation queries
            adaptive_alpha = 0.8
        elif query_type == "conceptual":
            # Heavy dense weight for conceptual queries
            adaptive_alpha = 0.2
        elif query_type == "procedural":
            # Balanced for procedural queries
            adaptive_alpha = 0.5
        else:
            # Default balanced
            adaptive_alpha = 0.5
        
        # Use weighted fusion with adaptive alpha
        original_alpha = self.alpha
        self.alpha = adaptive_alpha
        results = self._weighted_fusion(bm25_cands, dense_cands, k)
        self.alpha = original_alpha  # Restore original
        
        # Update metadata to reflect adaptive nature
        for result in results:
            result.metadata["fusion_method"] = "adaptive"
            result.metadata["adaptive_alpha"] = adaptive_alpha
            result.metadata["query_type"] = query_type
        
        return results
    
    def _build_chunk_map(self, bm25_cands: List[Chunk], dense_cands: List[Chunk]) -> Dict[str, Chunk]:
        """Build chunk ID to chunk mapping."""
        chunk_map = {}
        for chunk in bm25_cands + dense_cands:
            chunk_map[chunk.id] = chunk
        return chunk_map
    
    def retrieve_with_filters(self, 
                             query: str, 
                             k: int = 10,
                             jurisdiction: Optional[str] = None,
                             court_level: Optional[str] = None,
                             date_range: Optional[Tuple[datetime.date, datetime.date]] = None) -> List[Chunk]:
        """Retrieve with metadata filtering."""
        # Get initial results
        all_results = self.retrieve(query, k * 3)  # Get more candidates
        
        # Apply filters
        filtered_results = []
        for result in all_results:
            metadata = result.metadata
            
            # Apply filters (similar to BM25 implementation)
            if jurisdiction and jurisdiction.lower() not in metadata.get("court_name", "").lower():
                continue
            
            if court_level and court_level.lower() not in metadata.get("court_name", "").lower():
                continue
            
            if date_range:
                case_date_str = metadata.get("case_date", "")
                try:
                    case_date = datetime.datetime.strptime(case_date_str, "%B %d, %Y").date()
                    if not (date_range[0] <= case_date <= date_range[1]):
                        continue
                except ValueError:
                    continue
            
            filtered_results.append(result)
            if len(filtered_results) >= k:
                break
        
        return filtered_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive retriever statistics."""
        base_stats = {
            "fusion_method": self.fusion_method,
            "k_rrf": self.k_rrf,
            "alpha": self.alpha,
            "enable_legal_boost": self.enable_legal_boost,
            "enable_query_routing": self.enable_query_routing
        }
        
        # Add underlying retriever stats
        if self.bm25_retriever:
            bm25_stats = self.bm25_retriever.get_stats()
            base_stats.update({f"bm25_{k}": v for k, v in bm25_stats.items()})
        
        if self.dense_retriever:
            dense_stats = self.dense_retriever.get_stats()
            base_stats.update({f"dense_{k}": v for k, v in dense_stats.items()})
        
        return base_stats
    
    def build_index(self, chunks):
        """Pass-through to underlying retrievers."""
        pass