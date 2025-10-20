import os
import pickle
from typing import List, Dict, Any, Optional
import re
import numpy as np
from rank_bm25 import BM25Okapi
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.observability import trace_request, log_metrics

class BM25Retriever(Retriever):
    def __init__(self,index_path:Optional[str]=None,k1:float=1.2,b:float=0.75):
        self.index_path = index_path or "storage/bm25_index"
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []

        # Compile regex for better tokenization performance
        self.tokenizer_pattern = re.compile(r'\b\w+\b')

        # Cache for tokenized queries to avoid re-tokenizing identical queries
        self.query_cache: Dict[str, List[str]] = {}
        self.cache_size = 1000  # Limit cache size

        # Load existing index if present
        if os.path.exists(f"{self.index_path}.pkl"):
            self.load_index()

    def load_index(self) -> None:
        """Load BM25 index and chunks from disk."""
        try:
            with open(f"{self.index_path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.bm25 = data["bm25"]
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            self.bm25 = None
            self.chunks = []

    def _tokenize(self, text: str) -> List[str]:
        """Optimized tokenization using compiled regex."""
        # Convert to lowercase and extract words
        return [token.lower() for token in self.tokenizer_pattern.findall(text)]
    
    def _tokenize_query_cached(self, query: str) -> List[str]:
        """Tokenize query with caching for repeated queries."""
        if query in self.query_cache:
            return self.query_cache[query]
        
        tokens = self._tokenize(query)
        
        # Manage cache size
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[query] = tokens
        return tokens

    def build_index(self, chunks: List[Chunk], **kwargs) -> None:
        """Build BM25 index from a list of Chunk objects."""
        if not chunks:
            return

        with trace_request("build_index", "bm25_retriever.build_index"):
            self.chunks = chunks
            # Tokenize each chunk content using optimized tokenizer
            corpus = [self._tokenize(c.content) for c in chunks]
            self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

            # Persist index using highest protocol for better performance
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(f"{self.index_path}.pkl", "wb") as f:
                pickle.dump(
                    {"chunks": self.chunks, "bm25": self.bm25}, 
                    f, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            
            log_metrics({
                "bm25.indexed_chunks": len(self.chunks),
                "bm25.k1": self.k1,
                "bm25.b": self.b
            })

    def retrieve(self, query: str, k: int = 10) -> List[Chunk]:
        """Retrieve top-k chunks by BM25 score for the query."""
        if not self.bm25 or not self.chunks:
            return []

        with trace_request("retrieve", "bm25_retriever.retrieve"):
            # Use cached tokenization
            tokens = self._tokenize_query_cached(query)

            # Get scores and rank
            scores = self.bm25.get_scores(tokens)
            # Use numpy argpartition for faster top-k selection when k << n
            if k < len(scores) // 10:  # Only use argpartition if k is much smaller than n
                top_indices = np.argpartition(scores, -k)[-k:]
                # Sort only the top k results
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            else:
                # For larger k, use regular sorting
                top_indices = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )[:k]
            
            results: List[Chunk] = []
            for rank, idx in enumerate(top_indices):
                chunk = self.chunks[idx]
                # Create result with minimal metadata copying
                chunk_copy = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        "bm25_score": float(scores[idx]),
                        "rank": rank
                    }
                )
                results.append(chunk_copy)
            
            log_metrics({"bm25.retrieved": len(results)})
            return results
    
    def clear_cache(self):
        """Clear the query tokenization cache."""
        self.query_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "indexed_chunks": len(self.chunks),
            "k1": self.k1,
            "b": self.b
        }