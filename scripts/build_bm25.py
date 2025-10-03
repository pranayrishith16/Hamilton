# scripts/build_bm25.py
from orchestrator.registry import registry

# 1) Load your existing chunks from FAISS retriever
faiss = registry.get("faiss_retriever")
chunks = faiss.chunks

# 2) Build BM25 index
bm25 = registry.get("bm25_retriever")
bm25.build_index(chunks)

print(f"BM25 index built with {len(chunks)} chunks at {bm25.index_path}.pkl")
