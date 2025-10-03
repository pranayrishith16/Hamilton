#Query-time DAG implementation: route→retrieve→rerank→generate→verify; emits traces/metrics

from typing import Iterator, List, Optional, Dict, Any, AsyncGenerator
import time
import uuid

from loguru import logger
from orchestrator.interfaces import QueryResult
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics

class Pipeline:
    """Main pipeline orchestrator for query processing."""
    
    def __init__(self, pipeline_name: str = "default"):
        self.pipeline_name = pipeline_name 
        self.request_id: Optional[str] = None
        self.logger = logger
        
    def query(self, query: str, k: int = 10, rerank_k: int = 5) -> QueryResult:
        """Process a query through the full pipeline."""
        start_time = time.time()
        self.request_id = str(uuid.uuid4())
        
        with trace_request(self.request_id, "pipeline.query"):
            try:
                # Step 1: Retrieve relevant chunks
                self.logger.info("Retrieval started")
                t0 = time.time()
                retriever = registry.get("hybrid_retriever")
                chunks = retriever.retrieve(query, k=k)
                t1 = time.time()
                self.logger.info(f"Retrieval completed in {t1 - t0:.3f}s, retrieved {len(chunks)} chunks")
                
                # # Step 2: Rerank if configured
                # chunks = retrieved_chunks
                # if "cross_encoder_reranker" in registry.config:
                #     reranker = registry.get("cross_encoder_reranker")
                #     chunks = reranker.rerank(query, retrieved_chunks, k=rerank_k)
                
                # Step 3: Generate answer
                self.logger.info("Generation started")
                t0 = time.time()
                generator = registry.get("generator")
                raw_answer = generator.generate(query, chunks)
                t1 = time.time()
                self.logger.info(f"Generation completed in {t1 - t0:.3f}s")

                # Step 4: Post-process if available
                answer = raw_answer
                if "postprocessor" in registry.config:
                    pp = registry.get("postprocessor")
                    answer = pp.process(raw_answer, chunks)

                # Step 5: Verify if available
                verified = answer
                if "verifier" in registry.config:
                    verifier = registry.get("verifier")
                    verified = verifier.verify(answer, query, chunks)
                
                # Build metadata
                duration = time.time() - start_time
                metadata: Dict[str, Any] = {
                    "pipeline": self.pipeline_name,
                    "request_id": self.request_id,
                    "retrieved_count": len(chunks),
                    "reranked_count": len(chunks),
                    "generation_time": duration
                }

                # Log overall metrics
                log_metrics({
                    "pipeline.duration": duration,
                    "pipeline.retrieved": len(chunks),
                    "pipeline.final_chunks": len(chunks)
                })

                return QueryResult(
                    query=query,
                    answer=verified,
                    retrieved_chunks=chunks,
                    metadata=metadata
                )

            except Exception:
                log_metrics({"pipeline.errors": 1})
                raise

    def query_stream(self, query: str, k: int) -> Iterator[Dict[str, Any]]:
        """
        Sync generator that:
        - Emits one event with metadata (list of sources)
        - Then yields token deltas without metadata
        - Finally signals completion
        """
        # 1. Retrieve context with metadata
        t0 = time.perf_counter()
        retriever = registry.get("hybrid_retriever")
        result = retriever.retrieve(query, k=k)
        context_chunks = result
        t1 = time.perf_counter()

        retrieval_ms = (t1 - t0) * 1000
        print(f"[retrieval] {retrieval_ms:.1f} ms for k={k}")


        # 2. Build a single metadata payload
        sources = [
            {
                "source": c.metadata.get("source"),
                "page_number": c.metadata.get("page_number"),
                "id": c.id,
                'content':c.content[:500]
            }
            for c in context_chunks
        ]
        # Emit initial metadata event
        yield {"metadata": sources, "choices": [{"delta": {"content": ""}}]}

        # 3. Stream text deltas
        generator = registry.get("generator")
        for chunk in generator.stream_generate(query, context_chunks):
            # Each chunk is {"choices":[{"delta":{"content": "..."}}]}
            yield chunk

        # 4. End of stream signal (if your SSE layer doesn’t do [DONE] itself)
        yield {"choices": [{"delta": {"content": "[DONE]"}}]}

    
    def batch_query(self, queries: List[str], **kwargs) -> List[QueryResult]:
        """Process multiple queries in batch."""
        results = []
        for query in queries:
            result = self.query(query, **kwargs)
            results.append(result)
        return results