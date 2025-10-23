# FastAPI entrypoint; validates auth, loads active flavor, and delegates to orchestrator/pipeline; returns grounded answers

"""
FastAPI entrypoint for the RAG system.
Validates auth, loads active flavor, and delegates to orchestrator/pipeline.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from azure.storage.blob import BlobServiceClient

import uvicorn
from orchestrator.pipeline import Pipeline
from orchestrator.registry import registry
from ingestion.pipelines.ingestion_pipeline import IngestionPipeline
from pathlib import Path
from datetime import datetime
from loguru import logger
import faulthandler
import json

import dotenv

dotenv.load_dotenv()

faulthandler.enable()

# Initialize FastAPI app
app = FastAPI(
    title="Modular RAG API",
    description="A modular, plug-and-play RAG system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for document discovery
DOCUMENT_DIRECTORIES = [
    "data/",
    # "documents/", 
    # "uploads/",
    # "content/"
]

FILE_PATTERNS = ["*.pdf", "*.txt", "*.docx", "*.md"]

class IngestRequest(BaseModel):
    file_path: Optional[str] = None
    directory_path: Optional[str] = None
    file_pattern: str = "*.pdf"

class IngestResponse(BaseModel):
    status: str
    message: Optional[str] = None
    files_processed: Optional[int] = None
    total_chunks: Optional[int] = None
    results: Optional[List[Dict[str, Any]]] = None

class AutoIngestResponse(BaseModel):
    status: str
    message: Optional[str] = None
    directories_scanned: List[str]
    files_found: int
    files_processed: int
    total_chunks: int
    processing_time: float
    results: Optional[List[Dict[str, Any]]] = None

class DocumentDiscoveryResponse(BaseModel):
    directories_found: List[str]
    total_files: int
    files_by_type: Dict[str, int]
    latest_file: Optional[str] = None
    oldest_file: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    k: int = 10

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Global instances
pipeline = Pipeline()
ingestion_pipeline = IngestionPipeline()

# Track last processing time to avoid reprocessing
last_auto_ingest_time = None
auto_ingest_in_progress = False

def discover_documents() -> Dict[str, Any]:
    """Depreciated: Automatically discover documents in predefined directories."""
    discovered_files = []
    directories_found = []
    files_by_type = {}
    
    for dir_name in DOCUMENT_DIRECTORIES:
        dir_path = Path(__file__).parent.parent.parent / dir_name
        if dir_path.exists() and dir_path.is_dir():
            directories_found.append(dir_name)
            
            for pattern in FILE_PATTERNS:
                files = list(dir_path.rglob(pattern))  # Recursive search
                for file_path in files:
                    if file_path.is_file():
                        discovered_files.append(str(file_path))
                        ext = file_path.suffix.lower()
                        files_by_type[ext] = files_by_type.get(ext, 0) + 1
    
    # Get file timestamps for latest/oldest
    latest_file = None
    oldest_file = None
    
    if discovered_files:
        file_times = [(f, os.path.getmtime(f)) for f in discovered_files]
        file_times.sort(key=lambda x: x[1])
        oldest_file = file_times[0][0]
        latest_file = file_times[-1][0]
    
    return {
        "files": discovered_files,
        "directories_found": directories_found,
        "files_by_type": files_by_type,
        "latest_file": latest_file,
        "oldest_file": oldest_file
    }

def discover_azure_documents() -> Dict[str, Any]:
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("AZURE_CONTAINER_NAME")
        if not connection_string or not container_name:
            raise ValueError("Azure Storage credentials not configured")

        client = BlobServiceClient.from_connection_string(connection_string)
        container_client = client.get_container_client(container_name)

        blob_list = container_client.list_blobs()
        files = []
        files_by_type = {}
        file_times = []

        for blob in blob_list:
            files.append(blob.name)
            ext = os.path.splitext(blob.name)[1].lower()
            files_by_type[ext] = files_by_type.get(ext, 0) + 1
            # last_modified is datetime, convert to timestamp
            if blob.last_modified:
                file_times.append((blob.name, blob.last_modified.timestamp()))

        oldest_file = None
        latest_file = None

        if file_times:
            file_times.sort(key=lambda x: x[1])
            oldest_file = file_times[0][0]
            latest_file = file_times[-1][0]

        return {
            "files": files,
            "directories_found": [container_name],
            "files_by_type": files_by_type,
            "latest_file": latest_file,
            "oldest_file": oldest_file
        }
    except Exception as e:
        # handle or propagate error
        raise e


async def auto_process_documents() -> Dict[str, Any]:
    """Automatically process all discovered documents."""
    global auto_ingest_in_progress, last_auto_ingest_time
    
    if auto_ingest_in_progress:
        return {"status": "error", "message": "Auto-ingestion already in progress"}
    
    try:
        auto_ingest_in_progress = True
        start_time = datetime.now()

        # Call Azure ingestion instead of local directory
        ingest_result = ingestion_pipeline.ingest_from_azure()

        # Extract metrics from batch result
        files_processed = ingest_result.get("files_processed", 0)
        batches_processed = ingest_result.get("batches_processed", 0)
        total_chunks = ingest_result.get("total_chunks", 0)
        method = ingest_result.get("method", "")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        last_auto_ingest_time = end_time

        return {
            "status": ingest_result.get("status", "success"),
            "directories_scanned": [f"azure container: {os.getenv('AZURE_CONTAINER_NAME')}"],
            "files_found": files_processed,
            "files_processed": files_processed,
            "batches_processed": batches_processed,
            "total_chunks": total_chunks,
            "method": method,
            "processing_time": processing_time,
            "results": ingest_result.get("results", [])
        }

    finally:
        auto_ingest_in_progress = False

@app.get("/")
async def root():
    """Root endpoint with API information."""
    for route in app.routes:
        print(f"{route.path} [{','.join(route.methods)}]")
    return {
        "message": "Modular RAG API with Auto-Discovery",
        "version": "1.0.0",
        "features": [
            "Automatic document discovery",
            "Multi-directory scanning",
            "Background processing",
            "Real-time indexing"
        ],
        "configured_directories": DOCUMENT_DIRECTORIES,
        "supported_formats": [p.replace("*", "") for p in FILE_PATTERNS]
    }


@app.get("/discover", response_model=DocumentDiscoveryResponse)
async def discover_endpoint():
    """Discover documents in configured directories without processing them."""
    try:
        discovery_result = discover_azure_documents()
        
        return DocumentDiscoveryResponse(
            directories_found=discovery_result["directories_found"],
            total_files=len(discovery_result["files"]),
            files_by_type=discovery_result["files_by_type"],
            latest_file=discovery_result["latest_file"],
            oldest_file=discovery_result["oldest_file"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto-ingest", response_model=AutoIngestResponse)
async def auto_ingest_endpoint(background_tasks: BackgroundTasks):
    """Automatically discover and ingest all documents from configured directories."""
    try:
        # Run the auto-processing
        result = await auto_process_documents()
        return AutoIngestResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto-ingest-background")
async def auto_ingest_background_endpoint(background_tasks: BackgroundTasks):
    """Start auto-ingestion as a background task."""
    if auto_ingest_in_progress:
        return {"status": "info", "message": "Auto-ingestion already in progress"}
    
    background_tasks.add_task(auto_process_documents)
    return {"status": "started", "message": "Auto-ingestion started in background"}

@app.get("/ingest-status")
async def get_ingest_status():
    """Get the current status of auto-ingestion."""
    return {
        "in_progress": auto_ingest_in_progress,
        "last_run": last_auto_ingest_time.isoformat() if last_auto_ingest_time else None,
        "configured_directories": DOCUMENT_DIRECTORIES,
        "supported_patterns": FILE_PATTERNS
    }

@app.post("/query",response_model=QueryResponse)
async def query_endpoint(request:QueryRequest):
    """Query the RAG system"""
    try:
        # Delegate to the pipeline instead of manual retrieval/generation
        result = pipeline.query(request.query, k=request.k)

        # Build response from QueryResult
        chunks_dict = [
            {"id": c.id, "content": c.content, "metadata": c.metadata}
            for c in result.retrieved_chunks
        ]

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            retrieved_chunks=chunks_dict,
            metadata=result.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: Request):  # Use Request instead of Pydantic model
    """
    Streams incremental answer tokens as SSE.
    """
    try:
        # Manually parse the request body
        body = await request.json()
        query = body.get("query", "")
        k = body.get("k", 5)
        
        if not query:
            raise ValueError("Query is required")
            
    except Exception as exc:
        # Capture the exception message in the outer scope
        error_message = str(exc)
        
        async def error_generator():
            payload = {"error": f"Invalid request: {error_message}"}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    def event_generator():

        yield ": ping\n\n"
        try:
            # Call your pipeline
            # 2) Stream each chunk from pipeline
            for chunk_dict in pipeline.query_stream(query, k=k):
                # chunk_dict should now include both:
                #   choices[0].delta.content AND metadata: {source,page}
                sse = json.dumps(chunk_dict)
                yield f"data: {sse}\n\n"

            # 3) Signal done
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_chunk = {
                "choices": [{"delta": {"content": ""}}],
                "error": str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control":"no-cache",
            "X-Accel-Buffering":"no" # disable buffering
        }
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if core components are available
        faiss_retriever = registry.get("qdrant_retriever")
        bm25_retriever = registry.get("bm25_retriever")
        generator = registry.get("generator")
        
        # Check document directories
        available_dirs = [d for d in DOCUMENT_DIRECTORIES if Path(d).exists()]
        
        return {
            "status": "healthy",
            "components": {
                "faiss_retriever": type(faiss_retriever).__name__,
                "bm25_retriever": type(bm25_retriever).__name__,
                "generator": type(generator).__name__,
            },
            "document_directories": {
                "configured": DOCUMENT_DIRECTORIES,
                "available": available_dirs,
                "missing": [d for d in DOCUMENT_DIRECTORIES if d not in available_dirs]
            },
            "auto_ingest": {
                "in_progress": auto_ingest_in_progress,
                "last_run": last_auto_ingest_time.isoformat() if last_auto_ingest_time else None
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/stats")
async def get_stats():
    """Get system statistics including document discovery stats."""
    try:
        ingestion_stats = ingestion_pipeline.get_stats()
        component_stats = registry.list_components()
        discovery_stats = discover_documents()
        
        return {
            "ingestion": ingestion_stats,
            "components": component_stats,
            "document_discovery": {
                "total_files_found": len(discovery_stats["files"]),
                "directories_scanned": discovery_stats["directories_found"],
                "files_by_type": discovery_stats["files_by_type"],
                "latest_file": discovery_stats["latest_file"],
                "oldest_file": discovery_stats["oldest_file"]
            },
            "auto_ingest": {
                "in_progress": auto_ingest_in_progress,
                "last_run": last_auto_ingest_time.isoformat() if last_auto_ingest_time else None
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/reload-config")
async def reload_config():
    """Reload system configuration."""
    try:
        registry.reload_config()
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Startup event to create directories if they don't exist
@app.on_event("startup")
async def startup_event():
    """Create document directories on startup if they don't exist."""
    for dir_name in DOCUMENT_DIRECTORIES:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")