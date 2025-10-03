# retrieval/rerankers/bi_encoder_reranker.py

from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from orchestrator.interfaces import Reranker
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.observability import trace_request, log_metrics

class BiEncoderReranker(Reranker):
    """Bi-encoder with interaction MLP for faster reranking."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hidden_dim: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.mlp.to(self.device)
    
    def rerank(self, query: str, candidates: List[Chunk], k: int = 5) -> List[Chunk]:
        """Encode query and passages separately, then score via interaction MLP."""
        with trace_request("rerank", "bi_encoder.rerank"):
            # Encode query
            q_inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
            q_emb = self.encoder(**q_inputs).last_hidden_state.mean(dim=1)
            
            # Encode passages in batch
            passages = [c.content for c in candidates]
            p_inputs = self.tokenizer(passages, return_tensors="pt", truncation=True, padding=True).to(self.device)
            p_embs = self.encoder(**p_inputs).last_hidden_state.mean(dim=1)
            
            # Compute interaction scores
            q_rep = q_emb.expand(p_embs.size(0), -1)
            interactions = torch.cat([q_rep, p_embs], dim=1)
            scores = self.mlp(interactions).squeeze(-1).detach().cpu().numpy()
            
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
                    metadata={**chunk.metadata, "bi_score": float(score), "rank": rank}
                )
                results.append(chunk_copy)
            
            log_metrics({"rerank.bi_encoder": len(results)})
            return results
