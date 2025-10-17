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
# from orchestrations.mlflow.tracking import start_run, log_artifact, log_metric, end_run, log_param
from loguru import logger

class IngestionPipeline:
    """Main pipeline for document ingestion and indexing."""

    def __init__(self, pipeline_name: str = "default"):
        self.pipeline_name = pipeline_name
        self.logger = logger
        # self.artifacts_base = Path("orchestrations") / "artifacts"

    def ingest_directory(self, directory_path: str, file_pattern: str = "*.pdf") -> Dict[str, Any]:
        """Ingest all files matching pattern from directory."""
        directory = Path(directory_path)
        files = list(directory.glob(file_pattern))

        # Create batch ID with timestamp
        batch_id = f"batch_{directory.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_artifacts_path = self.artifacts_base / batch_id
        batch_artifacts_path.mkdir(parents=True, exist_ok=True)

        # Top‐level batch run
        # start_run(batch_id, pipeline=self.pipeline_name, file_count=len(files))

        if not files:
            logger.warning(f"No files found matching {file_pattern}")
            # log_metric("ingestion.files", 0)
            # end_run()
            return {"status": "error", "message": "No files"}
        
        # ===================== 1. PARSE =======================
        all_pages = [] 
        # start_run("parse_all", stage="parse", nested=True)
        # log_param("parser_type", "FitzPDFParser")
        # log_param("file_pattern", file_pattern)
        # log_param("source_directory", str(directory))
        for file_path in files:
            parser = self._get_parser(str(file_path))
            doc = parser.parse(str(file_path))
            if doc:
                all_pages.append(doc)

        # Save parsed output as artifact
        parsed_path = batch_artifacts_path / "parsed_docs.json"
        with open(parsed_path, "w") as f:
            json.dump([self._serialize_doc(p) for p in all_pages], f, indent=2)
        # log_artifact(str(parsed_path), artifact_path="parsed")
        # log_metric("parse.pages", len(all_pages))
        # log_metric("parse.files_processed", len(files))
        # end_run()
        logger.info(f"✓ Parsed {len(all_pages)} pages from {len(files)} files")

        if not all_pages:
            # end_run()
            return {"status": "error", "message": "No pages extracted"}
        
        # 2. CLEAN
        # start_run("clean_all", stage="clean", nested=True)
        cleaner = registry.get("cleaner")
        # log_param("cleaner_name", cleaner.__class__.__name__) 
        
        cleaned = cleaner.clean(all_pages)
        
        # Persist cleaned pages
        clean_path = batch_artifacts_path / "cleaned_docs.json"
        with open(clean_path, "w") as f:
            json.dump([self._serialize_doc(c) for c in cleaned], f, indent=2)
        # log_artifact(str(clean_path), artifact_path="cleaned")
        # log_metric("clean.pages", len(cleaned))
        # log_metric("clean.retention_rate", len(cleaned) / len(all_pages) if all_pages else 0)
        # end_run()
        logger.info(f"✓ Cleaned {len(cleaned)} pages")

        # 3. ANNOTATE
        # start_run("annotate_all", stage="annotate", nested=True)
        annotator = registry.get("annotator")
        # log_param("annotator_name", annotator.__class__.__name__) 
        
        annotated = annotator.annotate(cleaned)
        
        ann_path = batch_artifacts_path / "annotated_docs.json"
        with open(ann_path, "w") as f:
            json.dump([self._serialize_doc(a) for a in annotated], f, indent=2)
        # log_artifact(str(ann_path), artifact_path="annotated")
        # log_metric("annotate.pages", len(annotated))
        # end_run()
        logger.info(f"✓ Annotated {len(annotated)} pages")

        # 4. CHUNK
        # start_run("chunk_all", stage="chunk", nested=True)
        chunker = registry.get("chunker")
        # log_param("chunk_size", getattr(chunker, 'chunk_size', 'unknown'))
        # log_param("chunk_overlap", getattr(chunker, 'chunk_overlap', 'unknown'))
        # log_param("chunker_type", chunker.__class__.__name__)
        
        chunks = chunker.split(annotated)
        
        chunk_path = batch_artifacts_path / "chunks.json"
        with open(chunk_path, "w") as f:
            json.dump([self._serialize_chunk(c) for c in chunks], f, indent=2)
        # log_artifact(str(chunk_path), artifact_path="chunks")
        # log_metric("chunk.count", len(chunks))
        # log_metric("chunk.avg_per_page", len(chunks) / len(annotated) if annotated else 0)
        # end_run()
        logger.info(f"✓ Created {len(chunks)} chunks")

        # 5. EMBED
        # start_run("embed_all", stage="embed", nested=True)
        embedder = registry.get("embedder")
        # log_param("embedder_model", getattr(embedder, 'model_name', 'unknown'))
        # log_param("embedder_type", embedder.__class__.__name__)
        
        embeddings = embedder.encode(chunks)
        
        # Save as numpy array artifact
        emb_path = batch_artifacts_path / "embeddings.npy"
        np.save(str(emb_path), embeddings)
        # log_artifact(str(emb_path), artifact_path="embeddings")
        # log_metric("embed.dim", embeddings.shape[1])
        # log_metric("embed.count", embeddings.shape[0])
        # log_metric("embed.total_size_mb", emb_path.stat().st_size / (1024**2))
        # end_run()
        logger.info(f"✓ Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}D)")

        # 6. INDEX FAISS
        # start_run("index_faiss", stage="index", nested=True)
        faiss_retr = registry.get("faiss_retriever")
        # log_param("faiss_index_type", getattr(faiss_retr, 'index_type', 'unknown'))
        
        faiss_retr.build_index(chunks, embeddings)
        
        # Save index file

        faiss_retr.save_index()
        # log_artifact(str(idx_path), artifact_path="faiss_index")
        # log_metric("faiss.index_size_mb", idx_path.stat().st_size / (1024**2))
    
        # Get FAISS stats if available
        # if hasattr(faiss_retr, 'get_stats'):
        #     faiss_stats = faiss_retr.get_stats()
        #     for key, value in faiss_stats.items():
        #         if isinstance(value, (int, float)):
        #             log_metric(f"faiss.{key}", value)

        # 7. INDEX BM25
        # start_run("index_bm25", stage="index", nested=True)
        bm25_retr = registry.get("bm25_retriever")
        # log_param("bm25_analyzer", getattr(bm25_retr, 'analyzer', 'unknown'))
        # log_param("bm25_type", bm25_retr.__class__.__name__)
        
        bm25_retr.build_index(chunks)
        
        # Save BM25 index if possible
        if hasattr(bm25_retr, 'save_index'):
            bm25_path = batch_artifacts_path / "bm25.pkl"
            bm25_retr.save_index(str(bm25_path))
            # log_artifact(str(bm25_path), artifact_path="bm25_index")
        
        # Log BM25 stats
        if hasattr(bm25_retr, 'get_stats'):
            stats = bm25_retr.get_stats()
            # log_metric("bm25.vocab_size", stats.get("vocab_size", 0))
            # log_metric("bm25.doc_count", stats.get("doc_count", 0))
            
            # Save stats as JSON
            bm25_stats_path = batch_artifacts_path / "bm25_stats.json"
            with open(bm25_stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            # log_artifact(str(bm25_stats_path), artifact_path="bm25_stats")
        
        # end_run()
        logger.info(f"✓ Built BM25 index")

        # Final batch metrics
        # log_metric("ingestion.files", len(files))
        # log_metric("ingestion.pages", len(all_pages))
        # log_metric("ingestion.chunks", len(chunks))
        # log_metric("ingestion.embeddings", embeddings.shape[0])
        # log_param("artifacts_path", str(batch_artifacts_path))
        
        # Save summary
        summary = {
            "batch_id": batch_id,
            "status": "success",
            "files_processed": len(files),
            "total_pages": len(all_pages),
            "total_chunks": len(chunks),
            "embedding_dimension": embeddings.shape[1],
            "artifacts_location": str(batch_artifacts_path),
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = batch_artifacts_path / "ingestion_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        # log_artifact(str(summary_path), artifact_path="summary")
        
        # end_run()  # Close batch run
        logger.info(f"✓ Ingestion complete: {batch_id}")

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