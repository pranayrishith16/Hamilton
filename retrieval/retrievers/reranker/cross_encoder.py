from typing import List
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from orchestrator.interfaces import Reranker
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.observability import trace_request, log_metrics

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = 'cpu'):
        # Force CPU-only environment
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        self.device = "cpu"  # Force CPU regardless of input
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to("cpu")


    def rerank(self, query:str, candidates:List[Chunk], k:int = 5) -> List[Chunk]:
        """Rerank top-k candidates by cross-encoder scores"""
        with trace_request('rerank','cross_encoder.rerank'):
            inputs = self.tokenizer(
                [query] * len(candidates),
                [chunk.content for chunk in candidates],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            outputs = self.model(**inputs)
            scores = outputs.logits.view(-1).detach().cpu().numpy()
            
            ranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            results = []
            for rank, (chunk, score) in enumerate(ranked):
                chunk_copy = Chunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata={**chunk.metadata, "cross_score": float(score), "rank": rank}
                )
                results.append(chunk_copy)
            
            log_metrics({"rerank.cross_encoder": len(results)})
            return results