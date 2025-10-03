# 6. Dense Retriever
dense_retriever_content = """from typing import List, Dict, Any, Optional
import numpy as np
import logging
from ..indexing.faiss_indexer import FAISSIndexer
from ...ingestion.embeddings.sentence_transformers import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        indexer: FAISSIndexer,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ):
        \"\"\"
        Initialize dense retriever.
        
        Args:
            embedder: Sentence transformer embedder
            indexer: FAISS indexer
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score to include
        \"\"\"
        self.embedder = embedder
        self.indexer = indexer
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        \"\"\"
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query text
            top_k: Override default top_k
            filters: Optional filters (e.g., source, page_range)
            
        Returns:
            List of retrieved documents with metadata and scores
        \"\"\"
        if top_k is None:
            top_k = self.top_k
        
        try:
            # Encode the query
            query_embedding = self.embedder.encode_single(query)
            
            # Search in FAISS index
            results = self.indexer.search(
                query_embedding, 
                k=top_k * 2,  # Get more results to allow for filtering
                threshold=None  # We'll filter by similarity later
            )
            
            # Apply similarity threshold
            filtered_results = [
                result for result in results 
                if result.get('similarity', 0) >= self.similarity_threshold
            ]
            
            # Apply custom filters if provided
            if filters:
                filtered_results = self._apply_filters(filtered_results, filters)
            
            # Return top_k results
            final_results = filtered_results[:top_k]
            
            logger.info(f"Retrieved {len(final_results)} documents for query: '{query[:50]}...'")
            return final_results
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return []
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        \"\"\"Apply additional filters to search results.\"\"\"
        filtered = results.copy()
        
        # Filter by source document
        if 'source' in filters:
            filtered = [r for r in filtered if r.get('source') == filters['source']]
        
        # Filter by page range
        if 'page_min' in filters or 'page_max' in filters:
            page_min = filters.get('page_min', 0)
            page_max = filters.get('page_max', float('inf'))
            filtered = [
                r for r in filtered 
                if page_min <= r.get('page', 0) <= page_max
            ]
        
        # Filter by document type
        if 'doc_type' in filters:
            filtered = [r for r in filtered if r.get('doc_type') == filters['doc_type']]
        
        return filtered
    
    def get_context_string(self, results: List[Dict[str, Any]], max_length: int = 4000) -> str:
        \"\"\"
        Convert retrieval results to context string for LLM.
        
        Args:
            results: List of retrieval results
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        \"\"\"
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # Format each chunk with metadata
            chunk_text = result.get('text', '')
            source = result.get('source', 'Unknown')
            page = result.get('page', 'Unknown')
            similarity = result.get('similarity', 0)
            
            chunk_header = f"\\n--- Document {i+1} (Source: {source}, Page: {page}, Score: {similarity:.3f}) ---\\n"
            chunk_content = f"{chunk_text}\\n"
            
            chunk_full = chunk_header + chunk_content
            
            if current_length + len(chunk_full) > max_length:
                break
            
            context_parts.append(chunk_full)
            current_length += len(chunk_full)
        
        return "".join(context_parts)
    
    def get_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"
        Extract citation information from results.
        
        Returns:
            List of citation objects with source, page, and snippet info
        \"\"\"
        citations = []
        for i, result in enumerate(results):
            citation = {
                "id": i + 1,
                "source": result.get('source', 'Unknown'),
                "page": result.get('page', 'Unknown'),
                "similarity_score": result.get('similarity', 0),
                "snippet": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', '')
            }
            citations.append(citation)
        return citations

# Example usage
if __name__ == "__main__":
    # This would typically be initialized with proper embedder and indexer
    from ...ingestion.embeddings.sentence_transformers import SentenceTransformerEmbedder
    from ..indexing.faiss_indexer import FAISSIndexer
    
    # Initialize components
    embedder = SentenceTransformerEmbedder()
    indexer = FAISSIndexer(embedder.get_dimension())
    retriever = DenseRetriever(embedder, indexer)
    
    print("Dense retriever initialized successfully")
"""

with open("pdf_rag/src/retrieval/retrievers/dense_retriever.py", "w") as f:
    f.write(dense_retriever_content)

print("Created dense_retriever.py")