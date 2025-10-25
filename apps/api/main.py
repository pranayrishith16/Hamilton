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
from pathlib import Path
from loguru import logger
import faulthandler
import json
import dotenv

from orchestrator.pipeline import Pipeline
from orchestrator.registry import registry

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

# ==================== Request/Response Models ====================

class QueryRequest(BaseModel):
    query: str
    k: int = 10

class RetrieveRequest(BaseModel):
    query: str
    k: int = 10
    retriever_type: str = "hybrid"  # "hybrid", "bm25", "qdrant"

class GenerateRequest(BaseModel):
    query: str
    context: List[Dict[str, Any]]  # List of chunks with id, content, metadata

class ConfigUpdateRequest(BaseModel):
    component_name: str
    config: Dict[str, Any]

class IndexBuildRequest(BaseModel):
    chunks: List[Dict[str, Any]]  # chunks with id, content, metadata
    retriever_type: str = "bm25"  # "bm25" or "qdrant"

# ==================== Global Instances ====================

pipeline = Pipeline()

# ==================== QUERY ENDPOINTS ====================

# Track last processing time to avoid reprocessing
last_auto_ingest_time = None
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


@app.post("/query")
async def query_endpoint(request:QueryRequest):
    """
    Full pipeline query (retrieve + generate).
    Uses: pipeline.query() -> hybrid_retriever.retrieve() -> generator.generate()
    """
    try:
        result = pipeline.query(request.query, k=request.k)
        return {
            "query": result.query,
            "answer": result.answer,
            "retrieved_chunks": [
                {"id": c.id, "content": c.content, "metadata": c.metadata}
                for c in result.retrieved_chunks
            ],
            "metadata": result.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: Request):  # Use Request instead of Pydantic model
    """
    Streaming query endpoint.
    Uses: pipeline.query_stream() -> generator.stream_generate()
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        k = body.get("k", 5)
        if not query:
            raise ValueError("Query is required")
    except Exception as exc:
        error_message = str(exc)
        async def error_generator():
            yield f"data: {json.dumps({'error': f'Invalid request: {error_message}'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    def event_generator():
        yield ": ping\n\n"
        try:
            for chunk_dict in pipeline.query_stream(query, k=k):
                yield f"data: {json.dumps(chunk_dict)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ==================== RETRIEVAL ENDPOINTS ====================

@app.post("/retrieve")
async def retrieve_endpoint(request: RetrieveRequest):
    """
    Direct retrieval without generation.
    Uses: bm25_retriever.retrieve() OR qdrant_retriever.retrieve() OR hybrid_retriever.retrieve()
    """
    try:
        if request.retriever_type == "bm25":
            retriever = registry.get("bm25_retriever")
        elif request.retriever_type == "qdrant":
            retriever = registry.get("qdrant_retriever")
        elif request.retriever_type == "hybrid":
            retriever = registry.get("hybrid_retriever")
        else:
            raise ValueError(f"Unknown retriever type: {request.retriever_type}")
        
        chunks = retriever.retrieve(request.query, k=request.k)
        return {
            "query": request.query,
            "retriever_type": request.retriever_type,
            "chunks": [
                {
                    "id": c.id,
                    "content": c.content,
                    "metadata": c.metadata
                }
                for c in chunks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    """
    Direct generation with provided context (no retrieval).
    Uses: generator.generate()
    """
    try:
        from ingestion.dataprep.chunkers.base import Chunk
        
        # Convert dict chunks to Chunk objects
        chunks = [
            Chunk(
                id=c.get("id", ""),
                content=c.get("content", ""),
                metadata=c.get("metadata", {})
            )
            for c in request.context
        ]
        
        generator = registry.get("generator")
        answer = generator.generate(request.query, chunks)
        
        return {
            "query": request.query,
            "answer": answer,
            "chunks_used": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== INDEX MANAGEMENT ENDPOINTS ====================

@app.post("/index/build")
async def build_index_endpoint(request: IndexBuildRequest):
    """
    Build index for BM25 or Qdrant retriever.
    Uses: bm25_retriever.build_index() OR qdrant_retriever.build_index()
    """
    try:
        from ingestion.dataprep.chunkers.base import Chunk
        
        chunks = [
            Chunk(
                id=c.get("id", ""),
                content=c.get("content", ""),
                metadata=c.get("metadata", {})
            )
            for c in request.chunks
        ]
        
        if request.retriever_type == "bm25":
            retriever = registry.get("bm25_retriever")
        elif request.retriever_type == "qdrant":
            retriever = registry.get("qdrant_retriever")
        else:
            raise ValueError(f"Unknown retriever type: {request.retriever_type}")
        
        retriever.build_index(chunks)
        
        return {
            "status": "success",
            "retriever_type": request.retriever_type,
            "chunks_indexed": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/stats/{retriever_type}")
async def get_index_stats(retriever_type: str):
    """
    Get statistics for a specific retriever.
    Uses: retriever.get_stats()
    """
    try:
        if retriever_type == "bm25":
            retriever = registry.get("bm25_retriever")
        elif retriever_type == "qdrant":
            retriever = registry.get("qdrant_retriever")
        elif retriever_type == "hybrid":
            retriever = registry.get("hybrid_retriever")
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        return retriever.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CONFIGURATION ENDPOINTS ====================

@app.get("/config")
async def get_config():
    """
    Get all configuration from registry.
    Uses: registry.config
    """
    try:
        return {
            "config": registry.config,
            "components": registry.list_components(),
            "config_sections": registry.list_config_sections()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/{section}")
async def get_config_section(section: str):
    """
    Get specific configuration section.
    Uses: registry.get_config()
    """
    try:
        return registry.get_config(section)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/config/reload")
async def reload_config():
    """
    Reload configuration from YAML.
    Uses: registry.reload_config()
    """
    try:
        registry.reload_config()
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== COMPONENT ENDPOINTS ====================

@app.get("/components")
async def list_components():
    """
    List all registered components.
    Uses: registry.list_components()
    """
    try:
        return {
            "components": registry.list_components(),
            "config_sections": registry.list_config_sections()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/component/{component_name}/info")
async def get_component_info(component_name: str):
    """
    Get information about a specific component.
    Supports: generator, bm25_retriever, qdrant_retriever, hybrid_retriever
    """
    try:
        component = registry.get(component_name)
        
        # Get component-specific info
        if hasattr(component, 'get_model_info'):
            return component.get_model_info()
        elif hasattr(component, 'get_stats'):
            return component.get_stats()
        else:
            return {
                "component": component_name,
                "type": str(type(component)),
                "info": "No additional info available"
            }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# ==================== RETRIEVER SPECIFIC ENDPOINTS ====================

@app.post("/bm25/clear-cache")
async def clear_bm25_cache():
    """
    Clear BM25 query cache.
    Uses: bm25_retriever.clear_cache()
    """
    try:
        retriever = registry.get("bm25_retriever")
        retriever.clear_cache()
        return {"status": "success", "message": "BM25 cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hybrid/config")
async def get_hybrid_config():
    """
    Get hybrid retriever configuration.
    Returns k_rrf parameter and sub-retriever stats.
    """
    try:
        hybrid = registry.get("hybrid_retriever")
        return hybrid.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HEALTH & INFO ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "Modular RAG API - Full Component Control",
        "version": "2.0.0",
        "endpoints": {
            "query": {
                "/query": "Full pipeline query (retrieve + generate)",
                "/query/stream": "Streaming query"
            },
            "retrieval": {
                "/retrieve": "Direct retrieval (bm25/qdrant/hybrid)",
                "/generate": "Direct generation with context"
            },
            "index": {
                "/index/build": "Build retriever index",
                "/index/stats/{type}": "Get index statistics"
            },
            "config": {
                "/config": "Get all config",
                "/config/{section}": "Get specific config section",
                "/config/reload": "Reload config from YAML"
            },
            "components": {
                "/components": "List all components",
                "/component/{name}/info": "Get component info"
            },
            "specialized": {
                "/bm25/clear-cache": "Clear BM25 cache",
                "/hybrid/config": "Get hybrid retriever config"
            }
        }
    }

@app.get("/health")
async def health_check():
    try:
        components = {
            "qdrant_retriever": registry.get("qdrant_retriever"),
            "bm25_retriever": registry.get("bm25_retriever"),
            "hybrid_retriever": registry.get("hybrid_retriever"),
            "generator": registry.get("generator")
        }
        return {
            "status": "healthy",
            "components": {name: type(comp).__name__ for name, comp in components.items()}
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}