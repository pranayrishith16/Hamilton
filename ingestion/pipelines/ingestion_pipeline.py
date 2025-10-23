"""
Main ingestion pipeline that orchestrates document processing.
Glue pipeline composing all prep→embed→index steps.
"""

from datetime import datetime
import json
from typing import List, Optional, Dict, Any
import os
from pathlib import Path
import io

import numpy as np
from orchestrator.registry import registry
from ingestion.dataprep.loaders.base_loader import LoaderConfig
from ingestion.dataprep.loaders.loader_factory import LoaderFactory
from orchestrator.observability import trace_request, log_metrics
from loguru import logger

class IngestionPipeline:
    """Main pipeline for document ingestion and indexing."""

    def __init__(self, pipeline_name: str = "default", config_env: str = 'default'):
        """
        Initialize pipeline.
        
        Args:
            pipeline_name: Name of this pipeline
            config_env: Configuration environment (default, dev, prod)
        """
        self.pipeline_name = pipeline_name
        self.logger = logger

    def ingest_from_azure(self) -> Dict[str, Any]:
        """
        Ingest files from Azure Blob Storage using batch streaming.
        
        All configuration comes from YAML + environment variables.
        No hardcoded values.
        
        Returns:
            Dict with ingestion summary:
            {
                "status": "success",
                "files_processed": 5000,
                "batches_processed": 100,
                "total_pages": 15000,
                "total_chunks": 25000,
                "embedding_dimension": 768,
                "method": "azure_batch_stream",
                "timestamp": "2025-10-22T20:15:00"
            }
        
        Example:
            pipeline = IngestionPipeline("legal_rag", config_env='prod')
            result = pipeline.ingest_from_azure()
            print(result)
        """
        with trace_request("ingest_from_azure", "ingestion.ingest_from_azure"):
            try:
                logger.info("Starting Azure Blob Storage ingestion pipeline")
                
                # Step 1: Get Azure configuration from registry
                loader_config_dict = registry.get_config('azure_loader')
                
                if not loader_config_dict:
                    raise ValueError("Azure loader configuration not found in config")
                
                logger.info(f"Configuration loaded: {loader_config_dict}")
                
                # Step 2: Create LoaderConfig object
                loader_config = LoaderConfig(
                    batch_size=loader_config_dict.get('batch_size', 50),
                    file_extensions=loader_config_dict.get('file_extensions', ['.pdf']),
                    skip_errors=loader_config_dict.get('skip_errors', True)
                )
                
                logger.info(f"Loader config: batch_size={loader_config.batch_size}, "
                           f"extensions={loader_config.file_extensions}, "
                           f"skip_errors={loader_config.skip_errors}")
                
                # Step 3: Create Azure Blob Storage loader
                loader = LoaderFactory.create_loader(
                    loader_type='azure',
                    config=loader_config,
                    connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
                    account_url=loader_config_dict.get('account_url'),
                    container_name=loader_config_dict.get('container_name')
                )
                
                logger.info(f"Azure Blob loader created successfully")
                
                # Step 4: Get container statistics
                stats = loader.get_container_stats()
                logger.info(f"Container statistics: {stats}")
                
                # Step 5: Process files in batches
                return self._ingest_batch_stream(loader)
                
            except Exception as e:
                logger.error(f"Azure ingestion error: {e}", exc_info=True)
                log_metrics({"ingestion.errors": 1})
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

    def _ingest_batch_stream(self, loader) -> Dict[str, Any]:
        """
        Internal method: Process files in batches using streaming.
        
        Workflow:
        1. For each batch of N files from loader:
           - Parse all files to extract text
           - Clean the text
           - Annotate with metadata
           - Chunk into smaller pieces
        2. After all batches:
           - Generate embeddings for all chunks
           - Build FAISS index (semantic search)
           - Build BM25 index (keyword search)
           - Save indices to disk
        3. Return ingestion summary
        
        Args:
            loader: A BaseLoader instance (Azure, Local, etc.)
        
        Returns:
            Ingestion summary dict
        """
        with trace_request("ingest_batch_stream", "ingestion._ingest_batch_stream"):
            all_pages = []
            all_chunks = []
            total_files = 0
            batch_count = 0
            
            # Get pipeline components from registry
            cleaner = registry.get("cleaner")
            annotator = registry.get("annotator")
            chunker = registry.get("chunker")
            embedder = registry.get("embedder")
            faiss_retr = registry.get("qdrant_retriever")
            bm25_retr = registry.get("bm25_retriever")
            
            logger.info("Pipeline components loaded from registry")
            
            try:
                # ==================== BATCH STREAMING LOOP ====================
                logger.info("Starting batch streaming loop...")
                
                for file_batch in loader.batch_load():
                    batch_count += 1
                    batch_pages = []
                    
                    logger.info(f"Processing batch {batch_count}: {len(file_batch)} files")
                    
                    # ===================== 1. PARSE =======================
                    logger.debug(f"Parsing {len(file_batch)} files in batch {batch_count}...")
                    
                    for file_data in file_batch:
                        try:
                            parser = self._get_parser(file_data['file_name'])
                            
                            # Parse from bytes (not file path)
                            doc = parser.parse_from_bytes(
                                file_data['content'],
                                file_data['file_name']
                            )
                            
                            if doc:
                                batch_pages.extend(doc if isinstance(doc, list) else [doc])
                                total_files += 1
                                # logger.debug(f"Successfully parsed {file_data['file_name']}")
                        
                        except Exception as e:
                            logger.error(f"Error parsing {file_data['file_name']}: {e}")
                            if not loader.config.skip_errors:  
                                raise
                            continue
                    
                    if not batch_pages:
                        logger.warning(f"Batch {batch_count}: No pages extracted")
                        continue
                    
                    logger.info(f"✓ Batch {batch_count}: Parsed {len(batch_pages)} pages from {len(file_batch)} files")
                    all_pages.extend(batch_pages)
                    
                    # ===================== 2. CLEAN =======================
                    logger.debug(f"Cleaning {len(batch_pages)} pages...")
                    cleaned = cleaner.clean(batch_pages)
                    logger.info(f"✓ Cleaned {len(cleaned)} pages")
                    
                    # ===================== 3. ANNOTATE =======================
                    logger.debug(f"Annotating {len(cleaned)} pages...")
                    annotated = annotator.annotate(cleaned)
                    logger.info(f"✓ Annotated {len(annotated)} pages")
                    
                    # ===================== 4. CHUNK =======================
                    logger.debug(f"Chunking {len(annotated)} pages...")
                    chunks = chunker.split(annotated)
                    all_chunks.extend(chunks)
                    logger.info(f"✓ Created {len(chunks)} chunks from batch {batch_count} "
                              f"(Total so far: {len(all_chunks)})")
                
                logger.info(f"✓ Batch streaming complete: {batch_count} batches processed, "
                           f"{total_files} files processed, {len(all_chunks)} total chunks")
                
                if not all_chunks:
                    logger.error("No chunks created from any batch")
                    return {
                        "status": "error",
                        "message": "No chunks created",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # ===================== 5. EMBED =======================
                logger.info(f"Starting embedding generation for {len(all_chunks)} chunks...")
                embeddings = embedder.encode(all_chunks)
                logger.info(f"✓ Generated {embeddings.shape[0]} embeddings "
                           f"(Dimension: {embeddings.shape[1]}D)")
                
                # ===================== 6. INDEX FAISS =======================
                logger.info("Building FAISS index...")
                faiss_retr.build_index(all_chunks, embeddings)
                faiss_retr.save_index()
                logger.info("✓ Built and saved FAISS index")
                
                # ===================== 7. INDEX BM25 =======================
                logger.info("Building BM25 index...")
                bm25_retr.build_index(all_chunks)
                if hasattr(bm25_retr, 'get_stats'):
                    bm25_stats = bm25_retr.get_stats()
                    logger.info(f"BM25 index stats: {bm25_stats}")
                logger.info("✓ Built BM25 index")
                
                # ===================== SUMMARY =======================
                summary = {
                    "status": "success",
                    "files_processed": total_files,
                    "batches_processed": batch_count,
                    "total_pages": len(all_pages),
                    "total_chunks": len(all_chunks),
                    "embedding_dimension": embeddings.shape[1],
                    "method": "azure_batch_stream",
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"✓ INGESTION COMPLETE")
                logger.info(f"Summary: {summary}")
                
                log_metrics({
                    "ingestion.files": total_files,
                    "ingestion.batches": batch_count,
                    "ingestion.pages": len(all_pages),
                    "ingestion.chunks": len(all_chunks),
                    "ingestion.success": 1
                })
                
                return summary
                
            except Exception as e:
                logger.error(f"Error during batch stream processing: {e}", exc_info=True)
                log_metrics({"ingestion.errors": 1})
                raise

    def ingest_directory(self, directory_path: str, file_pattern: str = "*.pdf") -> Dict[str, Any]:
        """
        DEPRECATED: Use ingest_from_azure() instead.
        Ingest all files matching pattern from local directory.
        """
        directory = Path(directory_path)
        files = list(directory.glob(file_pattern))

        if not files:
            logger.warning(f"No files found matching {file_pattern}")
            return {"status": "error", "message": "No files"}
        
        return self._process_files(files)

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

    def _process_files(self, files: List[Path]) -> Dict[str, Any]:
        """Internal method to process list of file paths."""
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
        faiss_retr = registry.get("qdrant_retriever")
        faiss_retr.build_index(chunks, embeddings)
        faiss_retr.save_index()

        # 7. INDEX BM25
        bm25_retr = registry.get("bm25_retriever")
        bm25_retr.build_index(chunks)
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

    def _get_parser(self, file_path: str):
        """Get appropriate parser based on file extension."""
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