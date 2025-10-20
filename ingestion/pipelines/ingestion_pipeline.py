"""
Main ingestion pipeline that orchestrates document processing.
Glue pipeline composing all prep→embed→index steps.
"""

from datetime import datetime
import json
from typing import List, Optional, Dict, Any
import os
from pathlib import Path

import numpy as np
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics
from loguru import logger

class IngestionPipeline:
    """Main pipeline for document ingestion and indexing."""

    def __init__(self, pipeline_name: str = "default"):
        self.pipeline_name = pipeline_name
        self.logger = logger

    def ingest_directory(self, directory_path: str, file_pattern: str = "*.pdf") -> Dict[str, Any]:
        """Ingest all files matching pattern from directory."""
        directory = Path(directory_path)
        files = list(directory.glob(file_pattern))

        if not files:
            logger.warning(f"No files found matching {file_pattern}")
            # log_metric("ingestion.files", 0)
            # end_run()
            return {"status": "error", "message": "No files"}
        
        # ===================== 1. PARSE =======================
        all_pages = [] 

        for file_path in files:
            parser = self._get_parser(str(file_path))
            doc = parser.parse(str(file_path))
            if doc:
                all_pages.append(doc)

        logger.info(f"✓ Parsed {len(all_pages)} pages from {len(files)} files")

        if not all_pages:
            return {"status": "error", "message": "No pages extracted"}
        
        # 2. CLEAN
        cleaner = registry.get("cleaner")
        
        cleaned = cleaner.clean(all_pages)
        
        logger.info(f"✓ Cleaned {len(cleaned)} pages")

        # 3. ANNOTATE
        annotator = registry.get("annotator")
        
        annotated = annotator.annotate(cleaned)

        logger.info(f"✓ Annotated {len(annotated)} pages")

        # 4. CHUNK
        chunker = registry.get("chunker")
        
        chunks = chunker.split(annotated)
        
        logger.info(f"✓ Created {len(chunks)} chunks")

        # 5. EMBED
        embedder = registry.get("embedder")
        
        embeddings = embedder.encode(chunks)
        
        logger.info(f"✓ Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}D)")

        # 6. INDEX FAISS
        faiss_retr = registry.get("faiss_retriever")
        
        faiss_retr.build_index(chunks, embeddings)
        
        # Save index file

        faiss_retr.save_index()

        # 7. INDEX BM25
        bm25_retr = registry.get("bm25_retriever")
        
        bm25_retr.build_index(chunks)
        
        # Log BM25 stats
        if hasattr(bm25_retr, 'get_stats'):
            stats = bm25_retr.get_stats()
            
        logger.info(f"✓ Built BM25 index")
        
        # Save summary
        summary = {
            "status": "success",
            "files_processed": len(files),
            "total_pages": len(all_pages),
            "total_chunks": len(chunks),
            "embedding_dimension": embeddings.shape[1],
            "timestamp": datetime.now().isoformat()
        }

        return summary

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single file through the full pipeline."""
        with trace_request("ingest_file", f"ingestion.ingest_file"):
            try:
                # Step 1: Parse document
                parser = self._get_parser(file_path)
                pages = parser.parse(file_path)


                if not pages:
                    return {
                        "file": file_path,
                        "status": "error",
                        "error": "No pages extracted"
                    }

                # Step 2: Chunk text
                chunker = registry.get("chunker")
                chunks = chunker.split(pages)

                if not chunks:
                    return {
                        "file": file_path,
                        "status": "error", 
                        "error": "No chunks created"
                    }

                # Step 3: Update retrieval index
                retriever = registry.get("retriever")

                # Get existing chunks if any
                existing_chunks = getattr(retriever, 'chunks', [])

                # Add new chunks
                all_chunks = existing_chunks + chunks

                # Rebuild index with all chunks
                retriever.build_index(all_chunks)

                log_metrics({
                    "ingestion.pages": len(pages),
                    "ingestion.chunks": len(chunks),
                    "ingestion.total_chunks": len(all_chunks)
                })

                return {
                    "file": file_path,
                    "status": "success",
                    "pages_count": len(pages),
                    "chunks_count": len(chunks),
                    "total_chunks": len(all_chunks)
                }

            except Exception as e:
                log_metrics({"ingestion.errors": 1})
                return {
                    "file": file_path,
                    "status": "error",
                    "error": str(e)
                }

    def _get_parser(self, file_path: str):
        """Get approp
        riate parser based on file extension."""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            from ingestion.dataprep.parsers.pdf_parser import FitzPDFParser
            return FitzPDFParser()
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion pipeline statistics."""
        try:
            retriever = registry.get("retriever")
            stats = retriever.get_stats() if hasattr(retriever, 'get_stats') else {}
            return {
                "pipeline": self.pipeline_name,
                "retriever_stats": stats
            }
        except Exception as e:
            return {"error": str(e)}
        

    def _serialize_doc(self, doc) -> dict:
        """Serialize document object to dict for JSON storage."""
        if hasattr(doc, 'to_dict'):
            return doc.to_dict()
        elif hasattr(doc, '__dict__'):
            return {k: str(v) for k, v in doc.__dict__.items()}
        else:
            return {"content": str(doc)}

    def _serialize_chunk(self, chunk) -> dict:
        """Serialize chunk object to dict for JSON storage."""
        if hasattr(chunk, 'to_dict'):
            return chunk.to_dict()
        elif hasattr(chunk, '__dict__'):
            return {k: str(v) if not isinstance(v, (int, float, bool, list, dict)) else v 
                    for k, v in chunk.__dict__.items()}
        else:
            return {"text": str(chunk)}