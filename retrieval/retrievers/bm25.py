import os
import pickle
from typing import List, Dict, Any, Optional
from pyparsing import Opt
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

    def build_index(self, chunks: List[Chunk], **kwargs) -> None:
        """Build BM25 index from a list of Chunk objects."""
        if not chunks:
            return

        with trace_request("build_index", "bm25_retriever.build_index"):
            self.chunks = chunks
            # Tokenize each chunk content (simple whitespace split)
            corpus = [c.content.split() for c in chunks]
            self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

            # Persist index
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(f"{self.index_path}.pkl", "wb") as f:
                pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)

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
            # Tokenize query
            tokens = query.split()
            # Get scores and rank
            scores = self.bm25.get_scores(tokens)
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:k]

            results: List[Chunk] = []
            for rank, idx in enumerate(top_indices):
                chunk = self.chunks[idx]
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

    def get_stats(self) -> Dict[str, Any]:
        return {
            "indexed_chunks": len(self.chunks),
            "k1": self.k1,
            "b": self.b
        }