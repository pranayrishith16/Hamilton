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
    
    def __init__(self, index_path: Optional[str] = None, metric: str = "cosine",use_ivf: bool = True):
        self.index_path = index_path or "storage/faiss_index"
        self.metric = metric
        self.use_ivf = use_ivf
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.dimension: Optional[int] = None
        self.logger = logger

        #cache embedder
        self.embedder = None
        
        # Load existing index if available
        if os.path.exists(f"{self.index_path}.index"):
            self.load_index()

    def _get_embedder(self):
        """Get cached embedder or fetch from registry."""
        if self.embedder is None:
            self.embedder = registry.get("embedder")
        return self.embedder
    
    def retrieve(self, query: str, k: int = 10) -> List[Chunk]:
        """Retrieve k most relevant chunks for the query."""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        with trace_request("retrieve", "dense_retriever.retrieve"):
            # Get cached embedder and encode query
            embedder = self._get_embedder()
            query_embedding = embedder.encode([query])
            
            # Ensure float32 dtype from start to avoid copies
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32, copy=False)
            
            # Search index
            if self.metric == "cosine":
                # Normalize for cosine similarity
                faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, k)
            
            # Build results more efficiently - avoid creating new Chunk objects
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks) and idx >= 0:  # Check for valid index
                    chunk = self.chunks[idx]
                    # Create lightweight result without full metadata copy
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
                embedder = self._get_embedder()
                embeddings = embedder.encode(chunks)

            # Ensure embeddings is a non-empty numpy array
            if not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
                raise ValueError("No embeddings to index")
            
            # Ensure float32 dtype
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32, copy=False)

            # get embeddings dimension
            self.dimension = embeddings.shape[1]
            
            # Choose index type based on dataset size and performance needs
            if self.use_ivf and len(chunks) > 10000:
                # Use IVF for better performance on large datasets
                nlist = min(4096, max(64, int(np.sqrt(len(chunks)))))  # Rule of thumb
                
                if self.metric == "cosine":
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(embeddings)
                else:
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                
                # Train the index
                self.index.train(embeddings)
                # Set nprobe for search (balance between speed and accuracy)
                self.index.nprobe = min(32, nlist // 4)
            else:
                # Use flat index for smaller datasets or when exact search is needed
                if self.metric == "cosine":
                    self.index = faiss.IndexFlatIP(self.dimension)
                    faiss.normalize_L2(embeddings)
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            log_metrics({
                "index.chunks_count": len(chunks),
                "index.dimension": self.dimension,
                "index.type": "IVF" if self.use_ivf and len(chunks) > 10000 else "Flat"
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
            "use_ivf": self.use_ivf
        }