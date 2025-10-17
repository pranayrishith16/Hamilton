import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from retrieval.retrievers.interface import Retriever
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.observability import trace_request, log_metrics
import datetime
from functools import lru_cache
import hashlib

class LegalTokenizer:
    """Advanced tokenizer specifically designed for legal documents."""
    
    def __init__(self):
        # Legal citation patterns that should be preserved as single tokens
        self.legal_patterns = [
            # USC citations: "42 U.S.C. § 1983"
            r'\b\d+\s+U\.?S\.?C?\.?\s*§?\s*\d+(?:\([a-z0-9]+\))?',
            # Federal reporter citations: "123 F.3d 456"
            r'\b\d+\s+F\.\s*(?:2d|3d)?\s*\d+',
            # Docket numbers: "No. 1-18-2708"
            r'\bNo\.\s*\d+[-–]\d+[-–]?\d*',
            # Section symbols with numbers: "§ 1983"
            r'§\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))?',
            # Rule citations: "Rule 23(e)(1)"
            r'\bRule\s+\d+(?:\([a-z0-9]+\))*',
            # State codes: "Cal. Code Civ. Proc. § 425.16"
            r'\b[A-Z][a-z]+\.\s*(?:Code|Rev\.|Stat\.)[^§]*§\s*\d+(?:\.\d+)*',
            # Court case citations with year: "Smith v. Jones (2020)"
            r'\b[A-Z][a-zA-Z]+\s+v\.\s+[A-Z][a-zA-Z]+\s*\(\d{4}\)',
        ]
        
        # Compile patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.legal_patterns]
        
        # Standard word tokenizer (exclude legal patterns)
        self.word_pattern = re.compile(r'\b\w+\b')
        
        # Legal abbreviations that should not be split
        self.legal_abbrevs = {
            'U.S.C.', 'F.2d', 'F.3d', 'F.Supp.', 'S.Ct.', 'L.Ed.',
            'Cal.App.', 'N.Y.', 'Ill.App.', 'Fed.R.Civ.P.', 'Fed.R.Evid.'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Advanced tokenization preserving legal citations and terms."""
        if not text:
            return []
        
        tokens = []
        text_lower = text.lower()
        used_spans = set()
        
        # First, find all legal patterns
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                if not any(start < used_end and end > used_start for used_start, used_end in used_spans):
                    tokens.append(match.group().lower())
                    used_spans.add((start, end))
        
        # Then tokenize remaining text
        remaining_text = text_lower
        for start, end in sorted(used_spans, reverse=True):
            remaining_text = remaining_text[:start] + ' ' * (end - start) + remaining_text[end:]
        
        # Extract regular words from remaining text
        word_tokens = [token for token in self.word_pattern.findall(remaining_text) if len(token) > 1]
        tokens.extend(word_tokens)
        
        return tokens

class BM25Retriever(Retriever):
    def __init__(self,index_path:Optional[str]=None,k1:float=2.0,b:float=0.9,bm25_variant: str = "okapi",cache_size: int = 1000,enable_legal_tokenizer: bool = True,):
        self.index_path = index_path or "storage/bm25_index"
        self.k1 = k1
        self.b = b
        self.bm25_variant = bm25_variant
        self.enable_legal_tokenizer = enable_legal_tokenizer
        self.chunks: List[Chunk] = []
        self.cache_size = cache_size

        self.bm25 = None
        self.chunks: List[Chunk] = []


        # Initialize tokenizer
        if enable_legal_tokenizer:
            self.tokenizer = LegalTokenizer()
        else:
            self.tokenizer_pattern = re.compile(r'\b\w+\b')
        
        # Query processing cache
        self.query_cache: Dict[str, List[str]] = {}
        self.score_cache: Dict[str, np.ndarray] = {}
        
        # Load existing index if present
        if os.path.exists(f"{self.index_path}.pkl"):
            self.load_index()

    def _create_bm25_index(self, corpus: List[List[str]]) -> Any:
        """Create BM25 index based on specified variant."""
        if self.bm25_variant.lower() == "plus":
            # BM25+ adds query term proximity bonus
            return BM25Plus(corpus, k1=self.k1, b=self.b)
        elif self.bm25_variant.lower() == "l":
            # BM25L handles document length bias better
            return BM25L(corpus, k1=self.k1, b=self.b)
        else:
            # Default to BM25 Okapi
            return BM25Okapi(corpus, k1=self.k1, b=self.b)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using legal-aware or standard tokenizer."""
        if self.enable_legal_tokenizer:
            return self.tokenizer.tokenize(text)
        else:
            return [token.lower() for token in self.tokenizer_pattern.findall(text)] 
    
    def _get_query_cache_key(self, query: str, k: int) -> str:
        """Generate cache key for query results."""
        return hashlib.md5(f"{query}:{k}:{self.bm25_variant}".encode()).hexdigest()
    
    def _tokenize_query_cached(self, query: str) -> List[str]:
        """Tokenize query with caching."""
        if query in self.query_cache:
            return self.query_cache[query]
        
        tokens = self._tokenize(query)
        
        # Manage cache size
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[query] = tokens
        return tokens
    
    def build_index(self, chunks: List[Chunk], **kwargs) -> None:
        """Build BM25 index with legal optimizations."""
        if not chunks:
            return
        
        with trace_request("build_index", "enhanced_bm25_retriever.build_index"):
            self.chunks = chunks
            
            # Tokenize each chunk content
            print(f"Tokenizing {len(chunks)} chunks with legal-aware tokenizer...")
            corpus = []
            for i, chunk in enumerate(chunks):
                if i % 1000 == 0:
                    print(f"Processed {i}/{len(chunks)} chunks")
                tokens = self._tokenize(chunk.content)
                corpus.append(tokens)
            
            # Create BM25 index
            print(f"Building {self.bm25_variant.upper()} index...")
            self.bm25 = self._create_bm25_index(corpus)
            
            # Save index
            self.save_index()
            
            log_metrics({
                "bm25.indexed_chunks": len(self.chunks),
                "bm25.variant": self.bm25_variant,
                "bm25.k1": self.k1,
                "bm25.b": self.b,
                "bm25.legal_tokenizer": self.enable_legal_tokenizer
            })
            
            print(f"Built {self.bm25_variant.upper()} index with {len(chunks)} chunks")
    
    def retrieve(self, 
                 query: str, 
                 k: int = 10,
                 boost_citations: bool = True,
                 boost_exact_matches: bool = True,
                 section_weights: Optional[Dict[str, float]] = None) -> List[Chunk]:
        """Enhanced retrieve with legal-specific optimizations."""
        if not self.bm25 or not self.chunks:
            return []
        
        # Check score cache first
        cache_key = self._get_query_cache_key(query, k)
        if cache_key in self.score_cache:
            scores = self.score_cache[cache_key]
        else:
            with trace_request("retrieve", "enhanced_bm25_retriever.retrieve"):
                # Query expansion
                expanded_queries = self._expand_query(query)
                all_scores = []
                
                # Get scores for all query variations
                for expanded_query in expanded_queries:
                    tokens = self._tokenize_query_cached(expanded_query)
                    if tokens:  # Only process if tokens exist
                        query_scores = self.bm25.get_scores(tokens)
                        all_scores.append(query_scores)
                
                if not all_scores:
                    return []
                
                # Combine scores (take maximum across query variations)
                scores = np.maximum.reduce(all_scores) if len(all_scores) > 1 else all_scores[0]
                
                # Cache the scores
                if len(self.score_cache) >= self.cache_size:
                    oldest_key = next(iter(self.score_cache))
                    del self.score_cache[oldest_key]
                self.score_cache[cache_key] = scores
        
        # Get top-k indices efficiently
        if k < len(scores) // 10:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1][:k]
        
        results: List[Chunk] = []
        for rank, idx in enumerate(top_indices):
            if idx >= len(self.chunks):  # Safety check
                continue
                
            chunk = self.chunks[idx]
            base_score = float(scores[idx])
            
            # Apply boosting factors
            final_score = base_score
            boost_factors = {}
            
            # Create result chunk
            chunk_copy = Chunk(
                id=chunk.id,
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "bm25_score": base_score,
                    "final_score": final_score,
                    "rank": rank,
                    **boost_factors
                }
            )
            results.append(chunk_copy)
        
        # Sort by final score
        results.sort(key=lambda x: x.metadata["final_score"], reverse=True)
        
        log_metrics({"bm25.retrieved": len(results)})
        return results
    
    def retrieve_with_filters(self,
                             query: str,
                             k: int = 10,
                             jurisdiction: Optional[str] = None,
                             court_level: Optional[str] = None,
                             date_range: Optional[Tuple[datetime.date, datetime.date]] = None,
                             case_type: Optional[str] = None) -> List[Chunk]:
        """Retrieve with metadata filtering."""
        if not self.bm25 or not self.chunks:
            return []
        
        with trace_request("retrieve_with_filters", "enhanced_bm25_retriever.retrieve_with_filters"):
            # First get all results
            all_results = self.retrieve(query, k * 3)  # Get more candidates
            
            # Apply filters
            filtered_results = []
            for result in all_results:
                metadata = result.metadata
                
                # Jurisdiction filter
                if jurisdiction:
                    court_name = metadata.get("court_name", "").lower()
                    if jurisdiction.lower() not in court_name:
                        continue
                
                # Court level filter
                if court_level:
                    court_name = metadata.get("court_name", "").lower()
                    if court_level.lower() not in court_name:
                        continue
                
                # Case type filter
                if case_type:
                    disposition = metadata.get("disposition", "").lower()
                    if case_type.lower() not in disposition:
                        continue
                
                # Date range filter
                if date_range:
                    case_date_str = metadata.get("case_date", "")
                    try:
                        case_date = datetime.datetime.strptime(case_date_str, "%B %d, %Y").date()
                        if not (date_range[0] <= case_date <= date_range[1]):
                            continue
                    except ValueError:
                        continue
                
                filtered_results.append(result)
                if len(filtered_results) >= k:
                    break
            
            return filtered_results
    
    def retrieve_batch(self, queries: List[str], k: int = 10) -> List[List[Chunk]]:
        """Batch retrieval for multiple queries."""
        if not queries or not self.bm25:
            return [[] for _ in queries]
        
        with trace_request("retrieve_batch", "enhanced_bm25_retriever.retrieve_batch"):
            results = []
            for query in queries:
                query_results = self.retrieve(query, k)
                results.append(query_results)
            return results
    
    def save_index(self) -> None:
        """Save BM25 index and chunks to disk."""
        if self.bm25 is None:
            return
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save main index data
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "bm25": self.bm25,
                "variant": self.bm25_variant,
                "k1": self.k1,
                "b": self.b,
                "legal_tokenizer": self.enable_legal_tokenizer,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved BM25 index to {self.index_path}")
    
    def load_index(self) -> None:
        """Load BM25 index and chunks from disk."""
        try:
            with open(f"{self.index_path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.bm25 = data["bm25"]
                self.bm25_variant = data.get("variant", "okapi")
                self.k1 = data.get("k1", 2.0)
                self.b = data.get("b", 0.9)
                self.enable_legal_tokenizer = data.get("legal_tokenizer", True)
            
            print(f"Loaded {self.bm25_variant.upper()} index with {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            self.bm25 = None
            self.chunks = []
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        self.score_cache.clear()
        print("Cleared BM25 caches")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive retriever statistics."""
        return {
            "indexed_chunks": len(self.chunks),
            "bm25_variant": self.bm25_variant,
            "k1": self.k1,
            "b": self.b,
            "legal_tokenizer": self.enable_legal_tokenizer,
            "query_cache_size": len(self.query_cache),
            "score_cache_size": len(self.score_cache),
            "legal_synonyms": len(self.legal_synonyms)
        }