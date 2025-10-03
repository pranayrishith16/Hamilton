import faiss
from loguru import logger
import numpy as np
from typing import List, Dict, Any, Optional
import pickle
import os
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics

class DenseRetriever(Retriever):
    """FAISS-based dense retriever."""
    
    def __init__(self, index_path: Optional[str] = None, metric: str = "cosine"):
        self.index_path = index_path or "storage/faiss_index"
        self.metric = metric
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.dimension: Optional[int] = None
        self.logger = logger
        
        # Load existing index if available
        if os.path.exists(f"{self.index_path}.index"):
            self.load_index()
    
    def retrieve(self, query: str, k: int = 10) -> List[Chunk]:
        """Retrieve k most relevant chunks for the query."""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        with trace_request("retrieve", "dense_retriever.retrieve"):
            # Get embedder and encode query
            embedder = registry.get("embedder")
            query_embedding = embedder.encode([query])
            
            # Search index
            if self.metric == "cosine":
                # Normalize for cosine similarity
                faiss.normalize_L2(query_embedding.astype(np.float32))
            
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            # Get corresponding chunks
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    # Add retrieval metadata
                    chunk_copy = Chunk(
                        id=chunk.id,
                        content=chunk.content,
                        metadata={**chunk.metadata, "retrieval_score": float(score), "rank": i}
                    )
                    results.append(chunk_copy)
            
            log_metrics({"retrieval.results": len(results)})
            return results
    
    def build_index(self, chunks: List[Chunk],embeddings: Optional[np.ndarray] = None) -> None:
        """Build or update the retrieval index."""
        if not chunks:
            return
        
        with trace_request("build_index", "dense_retriever.build_index"):
            self.chunks = chunks
            self.logger.info(embeddings.shape)

            
            if embeddings is None:
                # Extract text content for encoding
                embedder = registry.get("embedder")
                embeddings = embedder.encode(chunks)

            # Ensure embeddings is a non-empty numpy array
            if not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
                raise ValueError("No embeddings to index")

            # get embeddings dimension
            self.dimension = embeddings.shape[1]
            
            # Create FAISS index
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings.astype(np.float32))
            else:
                self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance
            
            # Add embeddings to index
            self.index.add(embeddings.astype(np.float32))
            
            log_metrics({
                "index.chunks_count": len(chunks),
                "index.dimension": self.dimension
            })
            
            # Save index
            self.save_index()
    
    def save_index(self) -> None:
        """Save the FAISS index and chunks to disk."""
        if self.index is None:
            return
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        # Save chunks
        with open(f"{self.index_path}.chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "metric": self.metric,
            "chunks_count": len(self.chunks)
        }
        with open(f"{self.index_path}.meta.pkl", "wb") as f:
            pickle.dump(metadata, f)
    
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
            "index_size": self.index.ntotal if self.index else 0
        }