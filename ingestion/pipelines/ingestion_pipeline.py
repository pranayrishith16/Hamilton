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
        all_pages = []
        all_chunks = []
        total_files = 0
        batch_count = 0
        
        # Get pipeline components from registry
        cleaner = registry.get("cleaner")
        annotator = registry.get("annotator")
        chunker = registry.get("chunker")
        embedder = registry.get("embedder")
        qdrant_retr = registry.get("qdrant_retriever")
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
            
            # ===================== 6. INDEX QDRANT =======================
            logger.info("Building QDRANT index...")
            qdrant_retr.build_index(all_chunks, embeddings)
            qdrant_retr.save_index()
            logger.info("✓ Built and saved QDRANT index")
            
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
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during batch stream processing: {e}", exc_info=True)
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

            return {
                "file": file_path,
                "status": "success",
                "pages_count": len(pages),
                "chunks_count": len(chunks),
                "total_chunks": len(all_chunks)
            }

        except Exception as e:
            return {
                "file": file_path,
                "status": "error",
                "error": str(e)
            }
        
    def ingest_non_ingested_batch(self, max_files: int = 1000) -> Dict[str, Any]:
        """
        Ingest only non-ingested files up to a maximum limit.
        
        Filters for files with metadata['ingested'] != 'true' and processes
        only the first `max_files` files.
        
        Args:
            max_files: Maximum number of files to ingest (default 1000)
            
        Returns:
            Dict with ingestion summary including files_processed count
        """
        try:
            logger.info(f"Starting batch ingestion for up to {max_files} non-ingested files")
            
            # Step 1: Get Azure configuration
            loader_config_dict = registry.get_config('azure_loader')
            self.logger.info('Azure loader configuration loaded')
            if not loader_config_dict:
                raise ValueError("Azure loader configuration not found")
            
            # Step 2: Create loader config
            loader_config = LoaderConfig(
                batch_size=loader_config_dict.get('batch_size', 50),
                file_extensions=loader_config_dict.get('file_extensions', ['.pdf']),
                skip_errors=loader_config_dict.get('skip_errors', True)
            )
            
            # Step 3: Create Azure loader
            loader = LoaderFactory.create_loader(
                loader_type='azure',
                config=loader_config,
                connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
                account_url=loader_config_dict.get('account_url'),
                container_name=loader_config_dict.get('container_name')
            )
            
            # Step 4: Get non-ingested files with limit
            non_ingested_files = self._get_non_ingested_files(loader, max_files)
            
            if not non_ingested_files:
                logger.info("No non-ingested files found")
                return {
                    "status": "success",
                    "files_processed": 0,
                    "batches_processed": 0,
                    "total_chunks": 0,
                    "message": "No non-ingested files to process",
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"Found {len(non_ingested_files)} non-ingested files to process")
            
            # Step 5: Process the filtered batch
            return self._ingest_filtered_batch(loader, non_ingested_files)
        
        except Exception as e:
            logger.error(f"Batch ingestion error: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


    def _get_non_ingested_files(self, loader, max_files: int) -> List[str]:
        """
        Get list of blob names that have not been ingested yet.
        
        Args:
            loader: Azure blob loader instance
            max_files: Maximum number of files to return
            
        Returns:
            List of blob names (up to max_files) that need ingestion
        """
        from azure.storage.blob import BlobServiceClient
        
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("AZURE_CONTAINER_NAME")
        
        client = BlobServiceClient.from_connection_string(conn_str)
        container_client = client.get_container_client(container_name)
        
        non_ingested = []
        
        logger.info(f"Scanning blobs for non-ingested files (limit: {max_files})...")
        
        for blob in container_client.list_blobs():
            # Stop if we've reached the limit
            if len(non_ingested) >= max_files:
                break
            
            # Check if file extension matches
            if not any(blob.name.endswith(ext) for ext in loader.config.file_extensions):
                continue
            
            # Check metadata for ingested flag
            blob_client = container_client.get_blob_client(blob.name)
            try:
                properties = blob_client.get_blob_properties()
                metadata = properties.metadata or {}
                
                # Only include if not ingested
                if metadata.get("ingested", "false").lower() != "true":
                    non_ingested.append(blob.name)
                    
            except Exception as e:
                logger.warning(f"Could not check metadata for {blob.name}: {e}")
                # If we can't check, assume not ingested
                non_ingested.append(blob.name)
        
        logger.info(f"Found {len(non_ingested)} non-ingested files")
        return non_ingested


    def _ingest_filtered_batch(self, loader, file_list: List[str]) -> Dict[str, Any]:
        """
        Process a specific list of files and mark them as ingested.
        
        Args:
            loader: Azure blob loader
            file_list: List of blob names to process
            
        Returns:
            Ingestion summary dict
        """
        all_pages = []
        all_chunks = []
        total_files = 0
        batch_count = 0
        
        # Get pipeline components
        cleaner = registry.get("cleaner")
        self.logger.info(f'Cleaner loaded')
        annotator = registry.get("annotator")
        self.logger.info(f'Annotator loaded')
        chunker = registry.get("chunker")
        self.logger.info(f'Chunker loaded')
        embedder = registry.get("embedder")
        self.logger.info(f'Embedder loaded')
        qdrant_retr = registry.get("qdrant_retriever")
        self.logger.info(f'Qdrant loader loaded')
        bm25_retr = registry.get("bm25_retriever")
        self.logger.info(f'bm25 loaded')
        
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("AZURE_CONTAINER_NAME")
        
        client = BlobServiceClient.from_connection_string(conn_str)
        container_client = client.get_container_client(container_name)
        
        try:
            # Process files in batches
            batch_size = loader.config.batch_size
            
            for i in range(0, len(file_list), batch_size):
                batch_files = file_list[i:i + batch_size]
                batch_count += 1
                batch_pages = []
                
                logger.info(f"Processing batch {batch_count}: {len(batch_files)} files")
                
                # Download and parse each file
                for blob_name in batch_files:
                    try:
                        # Download blob content
                        blob_client = container_client.get_blob_client(blob_name)
                        blob_data = blob_client.download_blob().readall()
                        
                        # Parse the file
                        parser = self._get_parser(blob_name)
                        doc = parser.parse_from_bytes(blob_data, blob_name)
                        
                        if doc:
                            batch_pages.extend(doc if isinstance(doc, list) else [doc])
                            total_files += 1
                            
                            # Mark as ingested in metadata
                            self._mark_blob_as_ingested(container_client, blob_name)
                            
                    except Exception as e:
                        logger.error(f"Error processing {blob_name}: {e}")
                        if not loader.config.skip_errors:
                            raise
                        continue
                
                if not batch_pages:
                    logger.warning(f"Batch {batch_count}: No pages extracted")
                    continue
                
                logger.info(f"✓ Batch {batch_count}: Parsed {len(batch_pages)} pages")
                all_pages.extend(batch_pages)
                
                # Clean, annotate, chunk
                cleaned = cleaner.clean(batch_pages)
                annotated = annotator.annotate(cleaned)
                chunks = chunker.split(annotated)
                all_chunks.extend(chunks)
                
                logger.info(f"✓ Batch {batch_count}: Created {len(chunks)} chunks "
                        f"(Total: {len(all_chunks)})")
            
            if not all_chunks:
                return {
                    "status": "success",
                    "files_processed": 0,
                    "total_chunks": 0,
                    "message": "No chunks created",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Embed and index
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = embedder.encode(all_chunks)
            
            logger.info("Building QDRANT index...")
            qdrant_retr.build_index(all_chunks, embeddings)
            qdrant_retr.save_index()
            
            logger.info("Building BM25 index...")
            bm25_retr.build_index(all_chunks)
            
            # Summary
            summary = {
                "status": "success",
                "files_processed": total_files,
                "batches_processed": batch_count,
                "total_pages": len(all_pages),
                "total_chunks": len(all_chunks),
                "embedding_dimension": embeddings.shape[1],
                "method": "limited_batch_ingestion",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✓ LIMITED BATCH INGESTION COMPLETE: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error during filtered batch processing: {e}", exc_info=True)
            raise


    def _mark_blob_as_ingested(self, container_client, blob_name: str):
        """
        Mark a blob as ingested by setting metadata['ingested'] = 'true'.
        
        Args:
            container_client: Azure container client
            blob_name: Name of the blob to mark
        """
        try:
            blob_client = container_client.get_blob_client(blob_name)
            
            # Get existing metadata
            properties = blob_client.get_blob_properties()
            metadata = properties.metadata or {}
            
            # Update metadata
            metadata['ingested'] = 'true'
            metadata['ingested_at'] = datetime.now().isoformat()
            
            # Set updated metadata
            blob_client.set_blob_metadata(metadata)
            
            logger.debug(f"Marked {blob_name} as ingested")
            
        except Exception as e:
            logger.error(f"Error marking {blob_name} as ingested: {e}")

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

        # 6. INDEX QDRANT
        qdrant_retr = registry.get("qdrant_retriever")
        qdrant_retr.build_index(chunks, embeddings)
        qdrant_retr.save_index()

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