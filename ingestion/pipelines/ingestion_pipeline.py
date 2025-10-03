"""
Main ingestion pipeline that orchestrates document processing.
Glue pipeline composing all prep→embed→index steps.
"""

from typing import List, Optional, Dict, Any
import os
from pathlib import Path
from ingestion.dataprep.chunkers.base import Chunk
from orchestrator.registry import registry
from orchestrator.observability import trace_request, log_metrics
from ingestion.dataprep.parsers.interfaces import RawPage
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
            logger.warning(f'No files found: {file_pattern}')
            return {"status": "error", "message": f"No files found matching {file_pattern}"}

        all_pages = []       # type: List[RawPage]
        file_page_counts = {}
        for file_path in files:
            parser = self._get_parser(str(file_path))
            pages = parser.parse(str(file_path))
            if pages:
                all_pages.extend(pages)
                file_page_counts[str(file_path)] = len(pages)
        self.logger.info('Parser Done')

        if not all_pages:
            return {"status": "error", "message": "No pages extracted from any file"}
        
        # cleaner
        cleaner = registry.get('cleaner')
        cleaned_pages = cleaner.clean(all_pages)

        self.logger.info('Cleaner Done')

        # annotator
        annotator = registry.get('annotator')
        annotated_pages = annotator.annotate(cleaned_pages)

        self.logger.info('Annotator Done')

        # Chunk all pages at once
        chunker = registry.get("chunker")
        all_chunks = chunker.split(annotated_pages)
        if not all_chunks:
            return {"status": "error", "message": "No chunks created"}

        # Embed in one pass
        embedder = registry.get("embedder")
        embeddings = embedder.encode(all_chunks)
    
        # Build faiss index
        faiss_retriever = registry.get("faiss_retriever")
        faiss_retriever.build_index(all_chunks, embeddings)

        # Build bm25 index
        bm25_retriever = registry.get("bm25_retriever")
        bm25_retriever.build_index(all_chunks)  

        log_metrics({
            "ingestion.files": len(files),
            "ingestion.pages": len(all_pages),
            "ingestion.chunks": len(all_chunks),
        })

        return {
            "status": "success",
            "files_processed": len(files),
            "total_pages": len(all_pages),
            "total_chunks": len(all_chunks)
        }

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
        
