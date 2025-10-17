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

# from orchestrations.mlflow.tracking import (
#     start_run,
#     log_param,
#     log_metric,
#     log_artifact,
#     end_run,
# )

class Pipeline:
    """Main pipeline orchestrator for query processing."""
    
    def __init__(self, pipeline_name: str = "default"):
        self.pipeline_name = pipeline_name 
        self.request_id: Optional[str] = None
        self.logger = logger
        self.artifacts_base = Path("orchestrations") / "artifacts"
        
    def query(self, query: str, k: int = 10, rerank_k: int = 5) -> QueryResult:
        """Process a query through the full pipeline."""
        self.request_id = str(uuid.uuid4())
        run_name = f"query_{self.request_id}"
        run_artifacts = self.artifacts_base / run_name
        run_artifacts.mkdir(parents=True, exist_ok=True)

        # start_run(run_name, pipeline=self.pipeline_name, query=query, k=k, rerank_k=rerank_k)
        # log_param("pipeline", self.pipeline_name)
        # log_param("request_id", self.request_id)

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
            retrieved_path = run_artifacts / "retrieved.json"
            retrieved_path.write_text(json.dumps(sources, indent=2))
            # log_artifact(str(retrieved_path), artifact_path="retrieved")

            # log_metric("retrieval.time_s", retrieval_time)
            # log_metric("retrieval.count", len(chunks))
            self.logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.3f}s")
            
            # Step 3: Generate answer
            self.logger.info("Generation started")
            t0 = time.time()
            generator = registry.get("generator")
            raw_answer = generator.generate(query, chunks)
            t1 = time.time()
            gen_time = t1 - t0

            # Save raw answer artifact
            answer_path = run_artifacts / "raw_answer.txt"
            answer_path.write_text(raw_answer)
            # log_artifact(str(answer_path), artifact_path="raw_answer")

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
            # log_metric("pipeline.total_time_s", total_time)
            # log_metric("pipeline.final_chunks", len(chunks))

            # Save final answer artifact
            # final_path = run_artifacts / "final_answer.txt"
            # final_path.write_text(verified)
            # log_artifact(str(final_path), artifact_path="final_answer")

        # end_run()

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
        request_id = str(uuid.uuid4())
        run_name = f"stream_{request_id}"
        run_artifacts = self.artifacts_base / run_name
        run_artifacts.mkdir(parents=True, exist_ok=True)

        # start_run(run_name, pipeline=self.pipeline_name, query=query, k=k, stream=True)
        # log_param("pipeline", self.pipeline_name)
        # log_param("request_id", request_id)

        # Retrieve
        t0 = time.time()
        retriever = registry.get("hybrid_retriever")
        chunks = retriever.retrieve(query, k=k)
        t1 = time.time()
        # log_metric("retrieval.time_s", t1 - t0)
        # log_metric("retrieval.count", len(chunks))


        # Emit metadata event
        sources = [
            {"id": c.id, "source": c.metadata.get("source"), "snippet": c.content[:200]}
            for c in chunks
        ]
        # retrieved_path = run_artifacts / "retrieved.json"
        # retrieved_path.write_text(json.dumps(sources, indent=2))
        # log_artifact(str(retrieved_path), artifact_path="retrieved")
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