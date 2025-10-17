import faiss
from loguru import logger
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
import hashlib
import re
import os
from functools import lru_cache
import datetime

from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics

class DenseRetriever(Retriever):
    """FAISS-based dense retriever."""
    
    def __init__(self, index_path: Optional[str] = None, metric: str = "cosine",use_ivf: bool = True,enable_cache=True,cache_size:int=1000,):
        self.index_path = index_path or "storage/faiss_index"
        self.metric = metric
        self.use_ivf = use_ivf
        self.enable_cache = enable_cache
        self.cache_size = cache_size

        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.dimension: Optional[int] = None
        self.logger = logger

        #cache embedder
        self.embedder = None

        # Query cache
        self.query_cache = {} if enable_cache else None
        
        # Load existing index if available
        if os.path.exists(f"{self.index_path}.index"):
            self.load_index()

    def _get_embedder(self):
        """Get cached embedder or fetch from registry."""
        if self.embedder is None:
            self.embedder = registry.get("embedder")
        return self.embedder
    
    def _optimize_index_config(self,embeddings_count:int) -> str:
        """Dynamic index configuration based on corpus size."""
        if embeddings_count < 10000:
            return "flat"
        elif embeddings_count < 50000:
            return "ivf"
        elif embeddings_count < 500000:
            return "hnsw"
        else:
            return "ivf_pq"
        
    def preprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Optimize embeddings for better retrieval."""
        # Ensure float32 dtype
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32, copy=False)
        
        # L2 normalization for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
            
        return embeddings
    
    def _cache_key(self, query: str, k: int) -> str:
        """Generate cache key for query."""
        return hashlib.md5(f"{query}:{k}".encode()).hexdigest()
    
    def _build_results(self, scores: np.ndarray, indices: np.ndarray) -> List[Chunk]:
        """Build result chunks from scores and indices."""
        results = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk_copy = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "retrieval_score": float(score),
                        "rank": i
                    }
                )
                results.append(chunk_copy)
        return results
    
    def retrieve(self, query: str, k: int = 10, section_weights: Optional[Dict[str, float]] = None) -> List[Chunk]:
        """Retrieve k most relevant chunks for the query."""
        if self.index is None or len(self.chunks) == 0:
            return []

        # Check cache first
        if self.enable_cache:
            cache_key = self._cache_key(query, k)
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
        
        with trace_request("retrieve", "dense_retriever.retrieve"):
            queries_to_search = [query]
            # Get cached embedder and encode query
            embedder = self._get_embedder()
            all_results = []

            query_embedding = embedder.encode(query)
            
            query_embedding = self.preprocess_embeddings(query_embedding)
            
            # Search index
            scores, indices = self.index.search(query_embedding, k * 2)  # Get more candidates
            results = self._build_results(scores[0], indices[0])
            all_results.extend(results)
            
            # Remove duplicates and merge results
            seen_ids = set()
            merged_results = []
            for result in all_results:
                if result.id not in seen_ids:
                    merged_results.append(result)
                    seen_ids.add(result.id)

            # Sort by final scores and return top k
            final_results = sorted(
                merged_results, 
                key=lambda x: x.metadata.get("retrieval_score", 0), 
                reverse=True
            )[:k]

            # Cache results
            if self.enable_cache:
                if len(self.query_cache) >= self.cache_size:
                    # Simple LRU: remove oldest entries
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                self.query_cache[cache_key] = final_results
            
            log_metrics({"retrieval.results": len(final_results)})
            return final_results
        
    def retrieve_batch(self, queries: List[str], k: int = 10) -> List[List[Chunk]]:
        """Batch retrieval for multiple queries - more efficient."""
        if not queries or self.index is None:
            return [[] for _ in queries]
        
        with trace_request("retrieve_batch", "enhanced_dense_retriever.retrieve_batch"):
            embedder = self._get_embedder()
            
            # Batch encode all queries at once
            query_embeddings = embedder.encode(queries)
            
            query_embeddings = self.preprocess_embeddings(query_embeddings)
            
            # Single FAISS search call for all queries
            scores, indices = self.index.search(query_embeddings, k)
            
            # Build results for each query
            results = []
            for i in range(len(queries)):
                query_results = self._build_results(scores[i], indices[i])
                results.append(query_results)
            
            return results
        
    def retrieve_with_filters(self, 
                             query: str, 
                             k: int = 10,
                             jurisdiction: Optional[str] = None,
                             court_level: Optional[str] = None,
                             date_range: Optional[Tuple[datetime.date, datetime.date]] = None,
                             case_type: Optional[str] = None) -> List[Chunk]:
        """Retrieve with metadata-based filtering."""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        with trace_request("retrieve_with_filters", "enhanced_dense_retriever.retrieve_with_filters"):
            # Pre-filter chunks by metadata
            eligible_indices = []
            eligible_chunks = []
            
            for i, chunk in enumerate(self.chunks):
                metadata = chunk.metadata
                
                # Apply filters
                if jurisdiction and jurisdiction.lower() not in metadata.get("court_name", "").lower():
                    continue
                
                if court_level and court_level.lower() not in metadata.get("court_name", "").lower():
                    continue
                
                if case_type and case_type.lower() not in metadata.get("disposition", "").lower():
                    continue
                
                if date_range:
                    case_date_str = metadata.get("case_date", "")
                    try:
                        case_date = datetime.datetime.strptime(case_date_str, "%B %d, %Y").date()
                        if not (date_range[0] <= case_date <= date_range[1]):
                            continue
                    except ValueError:
                        continue  # Skip if date parsing fails
                
                eligible_indices.append(i)
                eligible_chunks.append(chunk)
            
            if not eligible_chunks:
                return []
            
            # Create filtered embeddings for search
            embedder = self._get_embedder()
            query_embedding = embedder.encode([query])
            
            query_embedding = self.preprocess_embeddings(query_embedding)
            
            # Search only among eligible chunks
            # For simplicity, we'll compute similarities manually
            # In production, you might want to create a temporary sub-index
            chunk_embeddings = []
            for idx in eligible_indices:
                # This assumes you have embeddings stored separately
                # You might need to re-embed or store embeddings alongside chunks
                pass
            
            # Fallback: retrieve from full index and filter results
            all_results = self.retrieve(query, k * 5)  # Get more candidates
            filtered_results = [
                r for r in all_results 
                if any(r.id == chunk.id for chunk in eligible_chunks)
            ]
            
            return filtered_results[:k]
    
    def build_index(self, chunks: List[Chunk], embeddings: Optional[np.ndarray] = None) -> None:
        """Build or update the retrieval index with optimizations."""
        if not chunks:
            return
        
        with trace_request("build_index", "enhanced_dense_retriever.build_index"):
            self.chunks = chunks
            
            if embeddings is None:
                embedder = self._get_embedder()
                embeddings = embedder.encode([chunk.content for chunk in chunks])
            
            # Ensure embeddings is a numpy array
            if not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
                raise ValueError("No embeddings to index")
            
            self.logger.info(f"Original embeddings shape: {embeddings.shape}")
            
            # Preprocess embeddings
            embeddings = self.preprocess_embeddings(embeddings)
            self.dimension = embeddings.shape[1]
            
            self.logger.info(f"Final embeddings shape: {embeddings.shape}")
            
            # Choose optimal index type
            index_type = self._optimize_index_config(len(chunks))
            
            if index_type == "flat":
                # Flat index for small datasets
                if self.metric == "cosine":
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)
                    
            elif index_type == "hnsw":
                # HNSW for medium datasets
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # M=32
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
                
            elif index_type == "ivf":
                # IVF for larger datasets (your current approach)
                nlist = min(4096, max(64, int(np.sqrt(len(chunks)))))
                
                if self.metric == "cosine":
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                else:
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                
                self.index.train(embeddings)
                self.index.nprobe = min(32, nlist // 4)
                
            elif index_type == "ivf_pq":
                # IVF + PQ for very large datasets
                nlist = min(4096, max(64, int(np.sqrt(len(chunks)))))
                m = 8  # Number of sub-quantizers
                bits = 8  # Bits per sub-quantizer
                
                if self.metric == "cosine":
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, bits)
                else:
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, bits)
                
                self.index.train(embeddings)
                self.index.nprobe = min(32, nlist // 4)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            log_metrics({
                "index.chunks_count": len(chunks),
                "index.dimension": self.dimension,
                "index.type": index_type,
            })
            
            self.logger.info(f"Built {index_type} index with {len(chunks)} chunks")
            
            # Save index
            self.save_index()
    
    def save_index(self) -> None:
        """Save the FAISS index and chunks to disk."""
        if self.index is None:
            return
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        # Save chunks using pickle protocol 4 for better performance
        with open(f"{self.index_path}.chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "metric": self.metric,
            "chunks_count": len(self.chunks),
            "use_ivf": self.use_ivf
        }
        
        with open(f"{self.index_path}.meta.pkl", "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Saved index to {self.index_path}")
    
    def load_index(self) -> None:
        """Load the FAISS index and chunks from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{self.index_path}.index")
            
            # Load chunks
            with open(f"{self.index_path}.chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load metadata
            with open(f"{self.index_path}.meta.pkl", "rb") as f:
                metadata = pickle.load(f)
                self.dimension = metadata["dimension"]
                self.metric = metadata["metric"]
                self.use_ivf = metadata.get("use_ivf", False)
            
            print(f"Loaded index with {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"Failed to load index: {e}")
            self.index = None
            self.chunks = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "chunks_count": len(self.chunks),
            "dimension": self.dimension,
            "metric": self.metric,
            "index_size": self.index.ntotal if self.index else 0,
            "use_ivf": self.use_ivf,
            "enable_cache": self.enable_cache,
            "cache_size": len(self.query_cache) if self.query_cache else 0,
        }
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self.query_cache:
            self.query_cache.clear()
            self.logger.info("Cleared query cache")