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
        
    def query(self, query: str, k: int = 10, context:str="") -> QueryResult:
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
            raw_answer = generator.generate(query, chunks,context=context)
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

    def query_stream(
        self, 
        query: str, 
        k: int = 10,
        context: str = "" 
    ) -> Iterator[Dict[str, Any]]:
        """
        Streaming query with memory context.
        
        Emits:
        1. Metadata event with context info
        2. Token delta events
        3. [DONE] signal
        """
        # RETRIEVE
        t0 = time.time()
        retriever = registry.get("hybrid_retriever")
        chunks = retriever.retrieve(query, k=k)
        t1 = time.time()
        
        # EMIT METADATA EVENT
        sources = []
        for c in chunks:
            try:
                # Safely extract metadata - handle both dict and object
                if isinstance(c.metadata, dict):
                    source_name = c.metadata.get("source", "Unknown")
                else:
                    # metadata is an object
                    source_name = getattr(c.metadata, "source", "Unknown") if c.metadata else "Unknown"
                
                sources.append({
                    "id": c.id,
                    "source": source_name,
                    "snippet": c.content[:200] if c.content else "",
                    "context_used": len(context) > 0
                })
            except Exception as e:
                self.logger.warning(f"Error processing chunk metadata: {e}")
                sources.append({
                    "id": getattr(c, "id", "unknown"),
                    "source": "Unknown",
                    "snippet": "",
                    "context_used": len(context) > 0
                })
        
        yield {
        "event": "sources",
        "sources": sources
        }
        
        # STREAM GENERATION (with context)
        generator = registry.get("generator")
        for delta in generator.stream_generate(query, chunks, context=context):  # NEW: context
            yield delta
        
        # DONE SIGNAL
        yield {"choices": [{"delta": {"content": "[DONE]"}}]}
    
    def batch_query(
        self, 
        queries: List[str], 
        context_per_query: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> List[QueryResult]:
        """
        Process multiple queries in batch WITH optional per-query context.
        
        Args:
            queries: List of query strings
            context_per_query: Dict mapping query to context string
            **kwargs: Additional args for query()
        
        Returns:
            List of QueryResults
        """
        results = []
        context_per_query = context_per_query or {}
        
        for query in queries:
            context = context_per_query.get(query, "")
            result = self.query(query, context=context, **kwargs)
            results.append(result)
        
        return results