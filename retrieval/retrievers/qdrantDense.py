import os
from loguru import logger
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics

class QdrantDenseRetriever(Retriever):
    """FAISS-based dense retriever."""
    
    def __init__(
            self, 
            collection_name='legal-docs',
            metric: str = "cosine",
            qdrant_url: Optional[str] = None,
            qdrant_api_key: Optional[str] = None):
        """
        Initialize Qdrant Dense Retriever.

        Args:
            collection_name: Name of Qdrant collection
            metric: Distance metric (cosine, euclid, dot)
            qdrant_url: Qdrant service URL (from Azure container)
            qdrant_api_key: API key for Qdrant (optional)
        """
        self.collection_name = collection_name
        self.metric = metric
        self.dimension:Optional[int] = None
        self.logger = logger

        # Get Qdrant connection from environment or params
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if self.qdrant_api_key:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        else:
            self.client = QdrantClient(url=self.qdrant_url)

        self.logger.info(f"Connected to Qdrant at {self.qdrant_url}")

        # Cache embedder
        self.embedder = None

    def _get_embedder(self):
        """Get cached embedder or fetch from registry."""
        if self.embedder is None:
            self.embedder = registry.get("embedder")
        return self.embedder
    
    def _map_metric_to_distance(self, metric: str) -> Distance:
        """Map metric name to Qdrant Distance enum."""
        metric_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT
        }
        return metric_map.get(metric.lower(), Distance.COSINE)
    
    def retrieve(self, query: str, k: int = 10) -> List[Chunk]:
        """Retrieve k most relevant chunks for the query."""
        with trace_request("retrieve", "dense_retriever.retrieve"):
            # Get cached embedder and encode query
            embedder = self._get_embedder()
            query_embedding = embedder.encode([query])
            
            # Ensure float32 and flatten to 1D list
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
            query_vector = query_embedding.tolist()
            
            # Search Qdrant
            try:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=k,
                    with_payload=True
                )

                # Build results from Qdrant points
                results = []
                for i, hit in enumerate(search_results):
                    payload = hit.payload
                    chunk = Chunk(
                        id=payload.get("chunk_id", str(hit.id)),
                        content=payload.get("content", ""),
                        metadata={
                            **payload.get("metadata", {}),
                            "retrieval_score": float(hit.score),
                            "rank": i
                        }
                    )
                    results.append(chunk)

                log_metrics({"retrieval.results": len(results)})
                return results

            except Exception as e:
                self.logger.error(f"Qdrant search error: {e}")
                return []
    
    def build_index(self, chunks: List[Chunk],embeddings: Optional[np.ndarray] = None) -> None:
        """Build or update the retrieval index."""
        if not chunks:
            return
        
        with trace_request("build_index", "dense_retriever.build_index"):
            self.logger.info(f"Building Qdrant index with {len(chunks)} chunks")

            if embeddings is None:
                # Generate embeddings
                embedder = self._get_embedder()
                embeddings = embedder.encode(chunks)

            # Ensure float32 dtype
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            # Get embedding dimension
            self.dimension = embeddings.shape

            # Create or recreate collection
            try:
                self.client.delete_collection(collection_name=self.collection_name)
                self.logger.info(f"Deleted existing collection: {self.collection_name}")
            except:
                pass

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=self._map_metric_to_distance(self.metric)
                )
            )
            self.logger.info(f"Created Qdrant collection: {self.collection_name}")

            # Prepare points for upsert
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk.id,
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                )
                points.append(point)

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                self.logger.info(f"Upserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

            log_metrics({
                "index.chunks_count": len(chunks),
                "index.dimension": self.dimension,
                "index.type": "Qdrant"
            })

            self.logger.info(f"✓ Qdrant index built successfully")
    
    def save_index(self) -> None:
        """Qdrant persists automatically - no action needed."""
        self.logger.info("Qdrant persists data automatically")
    
    def load_index(self) -> None:
        """Qdrant loads automatically - no action needed."""
        self.logger.info("Qdrant collection loaded from server")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "chunks_count": collection_info.points_count,
                "dimension": self.dimension,
                "metric": self.metric,
                "collection_name": self.collection_name,
                "qdrant_url": self.qdrant_url
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {
                "chunks_count": 0,
                "dimension": self.dimension,
                "metric": self.metric
            }