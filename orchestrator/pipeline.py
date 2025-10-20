#Query-time DAG implementation: route→retrieve→rerank→generate→verify; emits traces/metrics

from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any, AsyncGenerator
import time
import uuid
import json
from loguru import logger
from orchestrator.interfaces import QueryResult
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
        
        with trace_request(self.request_id, "pipeline.query"):
            # Step 1: Retrieve
            t0 = time.time()
            retriever = registry.get("hybrid_retriever")
            chunks = retriever.retrieve(query, k=k)
            t1 = time.time()
            retrieval_time = t1 - t0

            # Save retrieved metadata artifact
            sources = [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
                ]

            self.logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.3f}s")
            
            # Step 3: Generate answer
            self.logger.info("Generation started")
            t0 = time.time()
            generator = registry.get("generator")
            raw_answer = generator.generate(query, chunks)
            t1 = time.time()
            gen_time = t1 - t0

            # log_metric("generation.time_s", gen_time)
            self.logger.info(f"Generated answer in {gen_time:.3f}s")

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
                
                # Final metrics
            total_time = time.time() - start_time

        metadata: Dict[str, Any] = {
            "pipeline": self.pipeline_name,
            "request_id": self.request_id,
            "retrieved_count": len(chunks),
            "generation_time_s": gen_time,
            "total_time_s": total_time,
        }

        return QueryResult(
            query=query,
            answer=verified,
            retrieved_chunks=chunks,
            metadata=metadata,
        )

    def query_stream(self, query: str, k: int) -> Iterator[Dict[str, Any]]:
        """
        Sync generator that:
        - Emits one event with metadata (list of sources)
        - Then yields token deltas without metadata
        - Finally signals completion
        """

        # Retrieve
        t0 = time.time()
        retriever = registry.get("hybrid_retriever")
        chunks = retriever.retrieve(query, k=k)
        t1 = time.time()

        # Emit metadata event
        sources = [
            {"id": c.id, "source": c.metadata.get("source"), "snippet": c.content[:200]}
            for c in chunks
        ]
        yield {"metadata": sources, "choices": [{"delta": {"content": ""}}]}

        # Stream generation
        generator = registry.get("generator")
        for delta in generator.stream_generate(query, chunks):
            yield delta

        # Done signal
        yield {"choices": [{"delta": {"content": "[DONE]"}}]}

        # end_run()

    
    def batch_query(self, queries: List[str], **kwargs) -> List[QueryResult]:
        """Process multiple queries in batch."""
        results = []
        for query in queries:
            result = self.query(query, **kwargs)
            results.append(result)
        return results