# Project Documentation

Auto-generated documentation from Python source code.

**Generated:** 2025-10-26 11:06:03

## Table of Contents

- [apps.api.main](#apps-api-main)
- [doc_generator](#doc_generator)
- [generation.models.adapters.interface](#generation-models-adapters-interface)
- [generation.models.adapters.openrouter](#generation-models-adapters-openrouter)
- [generation.postprocessors.base](#generation-postprocessors-base)
- [generation.prompts.render_template](#generation-prompts-render_template)
- [ingestion.dataprep.annotators.pipeline](#ingestion-dataprep-annotators-pipeline)
- [ingestion.dataprep.chunkers.base](#ingestion-dataprep-chunkers-base)
- [ingestion.dataprep.chunkers.langchain_chunker](#ingestion-dataprep-chunkers-langchain_chunker)
- [ingestion.dataprep.cleaners.pipeline](#ingestion-dataprep-cleaners-pipeline)
- [ingestion.dataprep.loaders.azure_loader](#ingestion-dataprep-loaders-azure_loader)
- [ingestion.dataprep.loaders.base_loader](#ingestion-dataprep-loaders-base_loader)
- [ingestion.dataprep.loaders.loader_factory](#ingestion-dataprep-loaders-loader_factory)
- [ingestion.dataprep.parsers.interfaces](#ingestion-dataprep-parsers-interfaces)
- [ingestion.dataprep.parsers.ocr_parser](#ingestion-dataprep-parsers-ocr_parser)
- [ingestion.dataprep.parsers.pdf_parser](#ingestion-dataprep-parsers-pdf_parser)
- [ingestion.dataprep.parsers.xml_parser](#ingestion-dataprep-parsers-xml_parser)
- [ingestion.embeddings.models.all_miniLM_v2](#ingestion-embeddings-models-all_minilm_v2)
- [ingestion.embeddings.models.legal_bert](#ingestion-embeddings-models-legal_bert)
- [ingestion.pipelines.ingestion_pipeline](#ingestion-pipelines-ingestion_pipeline)
- [orchestrations.mlflow.tracking](#orchestrations-mlflow-tracking)
- [orchestrator.interfaces](#orchestrator-interfaces)
- [orchestrator.observability](#orchestrator-observability)
- [orchestrator.pipeline](#orchestrator-pipeline)
- [orchestrator.registry](#orchestrator-registry)
- [retrieval.retrievers.bm25](#retrieval-retrievers-bm25)
- [retrieval.retrievers.hybrid](#retrieval-retrievers-hybrid)
- [retrieval.retrievers.interface](#retrieval-retrievers-interface)
- [retrieval.retrievers.qdrantDense](#retrieval-retrievers-qdrantdense)
- [retrieval.retrievers.reranker.bi_encoder](#retrieval-retrievers-reranker-bi_encoder)
- [retrieval.retrievers.reranker.cross_encoder](#retrieval-retrievers-reranker-cross_encoder)
- [retrieval.retrievers.reranker.llm_based](#retrieval-retrievers-reranker-llm_based)

---

## apps.api.main

### Class: `QueryRequest`

**Inherits from:** `BaseModel`

**Description:**

Request model for query endpoint.
Contains the user's query string and optional k parameter for top-k retrieval.
Used in: POST /query and POST /query/stream

### Class: `RetrieveRequest`

**Inherits from:** `BaseModel`

**Description:**

Request model for direct retrieval endpoint.
Allows users to specify query, number of results (k), and which retriever to use.
Supports: hybrid, bm25, or qdrant retrievers.
Used in: POST /retrieve

### Class: `GenerateRequest`

**Inherits from:** `BaseModel`

**Description:**

Request model for direct generation endpoint.
Takes a query and pre-provided context chunks to generate answer without retrieval.
Useful for testing generation independently from retrieval.
Used in: POST /generate

### Class: `ConfigUpdateRequest`

**Inherits from:** `BaseModel`

**Description:**

Request model for configuration updates.
Contains component name and new configuration dictionary.
Used in: POST /config/update (if implemented)

### Class: `IndexBuildRequest`

**Inherits from:** `BaseModel`

**Description:**

Request model for building retriever indexes.
Contains list of chunks (with id, content, metadata) and retriever type.
Allows building fresh indexes for BM25 or Qdrant.
Used in: POST /index/build


---

## doc_generator

### Class: `DocumentationGenerator`

**Description:**

Generate markdown documentation from Python project.

**Methods:**

- `__init__(project_root)`
- `extract_docstring(node)`: Extract docstring from AST node.
- `extract_function_info(node)`: Extract function signature and metadata.
- `extract_class_info(node)`: Extract class information including methods.
- `parse_file(file_path)`: Parse Python file and extract public functions and classes.
- `scan_project(exclude_dirs)`: Scan entire project and extract documentation.
- `generate_markdown(file_structure)`: Generate complete markdown documentation.
- `_generate_function_docs(func)`: Generate markdown documentation for a function.
- `_generate_class_docs(cls)`: Generate markdown documentation for a class.

### Function: `main`

```
def main():
```

**Description:**

Main entry point.


---

## generation.models.adapters.interface

### Class: `GeneratorAdapter`

**Inherits from:** `ABC`

*No description provided.*

**Methods:**

- `generate(query, context)`
- `get_model_info()`


---

## generation.models.adapters.openrouter

### Class: `OpenRouterAdapter`

**Inherits from:** `GeneratorAdapter`

*No description provided.*

**Methods:**

- `__init__(api_url, api_key, model, temperature, max_tokens)`
- `generate(query, context)`: Generate an answer given query and context.
- `_build_messages(query, context)`
- `get_model_info()`
- `stream_generate(query, context)`: Streams OpenRouter/OpenAI chunks directly as they arrive


---

## generation.postprocessors.base

### Class: `PostProcessorAdapter`

**Inherits from:** `ABC`

**Description:**

Abstract base for post‐processing generation output.

**Methods:**

- `process(raw_answer, context)`: Given the raw generated answer and its metadata,


---

## generation.prompts.render_template

### Function: `render_messages`

```
def render_messages(query, context) -> List[Dict[str, Any]]:
```

**Description:**

Render a system+user message pair using the specified prompt template.

**Parameters:**

- `query`
- `context`

**Returns:**

- `List[Dict[str, Any]]`


---

## ingestion.dataprep.annotators.pipeline

### Class: `Section`

*No description provided.*

### Class: `PageAnnotator`

*No description provided.*

### Class: `LegalAnnotator`

*No description provided.*

**Methods:**

- `extract_procedural(text)`
- `extract_statutes(text)`
- `extract_precedents(text)`
- `extract_holdings_disposition(text)`
- `annotate_page(doc)`
- `annotate(docs)`


---

## ingestion.dataprep.chunkers.base

### Class: `Chunk`

*No description provided.*

### Class: `Chunker`

*No description provided.*

**Methods:**

- `split(pages)`


---

## ingestion.dataprep.chunkers.langchain_chunker

### Class: `LangChainChunker`

**Inherits from:** `Chunker`

*No description provided.*

**Methods:**

- `__init__(chunk_size, chunk_overlap, separators, length_function, use_token_splitter, token_chunk_size, token_chunk_overlap, encoding_name)`
- `split(pages)`


---

## ingestion.dataprep.cleaners.pipeline

### Class: `RawDoc`

*No description provided.*

### Class: `Section`

*No description provided.*

### Class: `Cleaner`

*No description provided.*

**Methods:**

- `__init__()`
- `court_name(text)`
- `docket_number(text)`
- `case_date(text)`
- `case_name(metadata)`
- `extract_headings(text)`: Returns list of {'name', 'start', 'end'} for each heading.
- `split_by_sections(text, metadata)`: Split cleaned text into sections, return list of RawDoc.
- `extract_disposition(text)`
- `whitespace(text)`
- `remove_page_number(text)`
- `remove_case_date_page_lines(text)`
- `dehyphenate(text)`
- `clean(docs)`


---

## ingestion.dataprep.loaders.azure_loader

### Class: `AzureBlobLoader`

**Inherits from:** `BaseLoader`

**Description:**

Loads files from Azure Blob Storage with batch streaming support.
Automatically handles authentication and file streaming.

Example:
    loader = AzureBlobLoader(
        connection_string="your-connection-string",
        container_name="legal-documents",
        config=LoaderConfig(batch_size=50, file_extensions=['.pdf'])
    )
    
    for batch in loader.batch_load():
        # Process batch of files
        process_batch(batch)

**Methods:**

- `__init__(container_name)`: Loads files from Azure Blob Storage with batch streaming support.
- `lazy_load()`: Stream files from Azure Blob one at a time.
- `list_files()`: List all files in container (for inspection)
- `get_container_stats()`: Get container statistics


---

## ingestion.dataprep.loaders.base_loader

### Class: `LoaderConfig`

**Description:**

Configuration for loaders

### Class: `BaseLoader`

**Inherits from:** `ABC`

**Description:**

Abstract base class for data loaders.
Loaders implement lazy_load (streaming) and batch_load methods.

**Methods:**

- `__init__(config)`
- `lazy_load()`: Lazy load files one at a time (streaming).
- `load()`: Eagerly load all files into memory.
- `batch_load(batch_size)`: Batch lazy loading - yields batches of files.


---

## ingestion.dataprep.loaders.loader_factory

### Class: `LoaderFactory`

**Description:**

Factory for creating loaders based on configuration.

**Methods:**

- `register_loader(cls, name, loader_class)`: Register a new loader class
- `create_loader(cls, loader_type, config)`: Create a loader instance.


---

## ingestion.dataprep.parsers.interfaces

### Class: `Parser`

**Inherits from:** `ABC`

*No description provided.*

**Methods:**

- `parse(path)`

### Class: `StatuteSection`

*No description provided.*


---

## ingestion.dataprep.parsers.ocr_parser

### Class: `OCRPDFParser`

**Inherits from:** `Parser`

*No description provided.*

**Methods:**

- `__init__(dpi, lang)`
- `parse(path)`


---

## ingestion.dataprep.parsers.pdf_parser

### Class: `RawDoc`

*No description provided.*

### Class: `FitzPDFParser`

**Inherits from:** `Parser`

*No description provided.*

**Methods:**

- `parse(path)`
- `parse_from_bytes(content_bytes, source_name)`: Parse PDF directly from bytes instead of a file path.


---

## ingestion.dataprep.parsers.xml_parser

### Class: `XMLParser`

**Inherits from:** `Parser`

**Description:**

Universal XML parser for USLM statute XML (and any similar hierarchical XML).
Extracts every <section> as a Document, capturing its full hierarchy in metadata.

**Methods:**

- `__init__(namespace)`
- `parse(path)`
- `_extract_sections(elem, path)`: Recursively traverse XML. Yield dicts with:


---

## ingestion.embeddings.models.all_miniLM_v2

### Class: `SentenceTransformerEmbedder`

*No description provided.*

**Methods:**

- `__init__(model_name)`
- `_load_model()`
- `encode(chunks, batch_size)`: Encode texts into embeddings.
- `encode_single(text)`: Encode single text string
- `get_dimension()`: Get dimension


---

## ingestion.embeddings.models.legal_bert

### Class: `SentenceTransformerEmbedder`

*No description provided.*

**Methods:**

- `__init__(model_name)`
- `_load_model()`
- `encode(chunks, batch_size)`: Encode texts into embeddings using SentenceTransformer's built-in encode method.
- `encode_single(text)`: Encode single text string
- `get_dimension()`: Get dimension


---

## ingestion.pipelines.ingestion_pipeline

### Class: `IngestionPipeline`

**Description:**

Main pipeline for document ingestion and indexing.

**Methods:**

- `__init__(pipeline_name, config_env)`: Initialize pipeline.
- `ingest_from_azure()`: Ingest files from Azure Blob Storage using batch streaming.
- `_ingest_batch_stream(loader)`: Internal method: Process files in batches using streaming.
- `ingest_directory(directory_path, file_pattern)`: DEPRECATED: Use ingest_from_azure() instead.
- `ingest_file(file_path)`: Ingest a single file through the full pipeline.
- `_process_files(files)`: Internal method to process list of file paths.
- `_get_parser(file_path)`: Get appropriate parser based on file extension.
- `get_stats()`: Get ingestion pipeline statistics.
- `_serialize_doc(doc)`: Serialize document object to dict for JSON storage.
- `_serialize_chunk(chunk)`: Serialize chunk object to dict for JSON storage.


---

## orchestrations.mlflow.tracking

### Function: `start_run`

```
def start_run(run_name, nested):
```

**Description:**

Configure MLflow, then start a run, apply default tags and log params.

**Parameters:**

- `run_name`
- `nested`

### Function: `log_param`

```
def log_param(key, value):
```

**Description:**

Log a single parameter to MLflow.

**Parameters:**

- `key`
- `value`

### Function: `log_artifact`

```
def log_artifact(local_path, artifact_path):
```

**Description:**

Log a local file or directory as an MLflow artifact.

**Parameters:**

- `local_path`
- `artifact_path`

### Function: `log_metric`

```
def log_metric(key, value, step):
```

**Description:**

Log a metric value to MLflow.

**Parameters:**

- `key`
- `value`
- `step`

### Function: `end_run`

```
def end_run():
```

**Description:**

End the active MLflow run.


---

## orchestrator.interfaces

### Class: `QueryResult`

**Description:**

Query result with retrieved context and metadata.

### Class: `Embedder`

**Inherits from:** `ABC`

**Description:**

Abstract base class for embedding models.

**Methods:**

- `encode(texts, batch_size)`: Encode texts into embeddings.
- `get_dimension()`: Get embedding dimension.

### Class: `Generator`

**Inherits from:** `ABC`

**Description:**

Abstract base class for generation models.

**Methods:**

- `generate(query, context)`: Generate answer given query and context.

### Class: `Memory`

**Inherits from:** `ABC`

**Description:**

Abstract base class for conversation memory.

**Methods:**

- `add_turn(query, response)`: Add a conversation turn to memory.
- `get_context()`: Get conversation context.
- `clear()`: Clear memory.

### Class: `Reranker`

**Inherits from:** `ABC`

**Description:**

Abstract base class for reranker

**Methods:**

- `rerank(query, candidates, k)`

### Class: `Storage`

**Inherits from:** `ABC`

**Description:**

Abstract base class for storage backends.

**Methods:**

- `store(key, value)`: Store value with key.
- `retrieve(key)`: Retrieve value by key.
- `delete(key)`: Delete value by key.

### Class: `MLOpsBackend`

**Inherits from:** `ABC`

**Description:**

Abstract base class for MLOps backends.

**Methods:**

- `log_params(params)`: Log parameters.
- `log_metrics(metrics)`: Log metrics.
- `log_artifact(path, name)`: Log artifact.


---

## orchestrator.observability

### Class: `TraceSpan`

**Description:**

Represents a trace span.

### Class: `ObservabilityManager`

**Description:**

Manages observability operations.

**Methods:**

- `__init__()`
- `start_span(request_id, operation, metadata)`: Start a new trace span.
- `end_span(span_id)`: End a trace span.
- `log_metric(name, value)`: Log a metric value.
- `log_metrics(metrics)`: Log multiple metrics.
- `get_metrics()`: Get all metrics.
- `clear_metrics()`: Clear all metrics.

### Function: `trace_request`

```
def trace_request(request_id, operation, metadata):
```

**Description:**

Context manager for tracing operations.

**Parameters:**

- `request_id`
- `operation`
- `metadata`

### Function: `log_metrics`

```
def log_metrics(metrics) -> None:
```

**Description:**

Log metrics using the global observability manager.

**Parameters:**

- `metrics`

**Returns:**

- `None`

### Function: `log_metric`

```
def log_metric(name, value) -> None:
```

**Description:**

Log a single metric.

**Parameters:**

- `name`
- `value`

**Returns:**

- `None`


---

## orchestrator.pipeline

### Class: `Pipeline`

**Description:**

Main pipeline orchestrator for query processing.

**Methods:**

- `__init__(pipeline_name)`
- `query(query, k, rerank_k)`: Process a query through the full pipeline.
- `query_stream(query, k)`: Sync generator that:
- `batch_query(queries)`: Process multiple queries in batch.


---

## orchestrator.registry

### Class: `Registry`

*No description provided.*

**Methods:**

- `__init__(config_path)`
- `_load_config()`
- `register(name, component)`
- `get(name)`: Get a component instance, creating it if needed.
- `get_config(name)`: Get raw configuration values (not a component).
- `_create_component(config)`: Create component from configuration.
- `list_components()`: List all available components.
- `list_config_sections()`: List all configuration sections in YAML.
- `reload_config(config_path)`: Reload configuration and clear cached components.


---

## retrieval.retrievers.bm25

### Class: `BM25Retriever`

**Inherits from:** `Retriever`

*No description provided.*

**Methods:**

- `__init__(index_path, k1, b)`
- `load_index()`: Load BM25 index and chunks from disk.
- `_tokenize(text)`: Optimized tokenization using compiled regex.
- `_tokenize_query_cached(query)`: Tokenize query with caching for repeated queries.
- `build_index(chunks)`: Build BM25 index from a list of Chunk objects.
- `retrieve(query, k)`: Retrieve top-k chunks by BM25 score for the query.
- `clear_cache()`: Clear the query tokenization cache.
- `get_stats()`


---

## retrieval.retrievers.hybrid

### Class: `HybridRetriever`

**Inherits from:** `Retriever`

**Description:**

Optimized hybrid retriever using Reciprocal Rank Fusion (RRF).

**Methods:**

- `__init__(k_rrf)`: Initialize with RRF parameter.
- `_ensure_retrievers()`: Cache retrievers to avoid registry lookups during each query.
- `retrieve(query, k)`: Retrieve using optimized RRF fusion.
- `_compute_rrf_scores(bm25_cands, dense_cands)`: Compute RRF scores efficiently.
- `_build_chunk_map(bm25_cands, dense_cands)`: Build a map from chunk IDs to chunks for efficient lookup.
- `get_stats()`: Get retriever statistics.
- `build_index(chunks)`: Pass-through to underlying retrievers.


---

## retrieval.retrievers.interface

### Class: `Retriever`

**Inherits from:** `ABC`

**Description:**

Abstract base class for all retrievers.

**Methods:**

- `retrieve(query, k)`: Retrieve k most relevant chunks for the query.
- `build_index(chunks)`: Build or update the retrieval index.


---

## retrieval.retrievers.qdrantDense

### Class: `QdrantDenseRetriever`

**Inherits from:** `Retriever`

**Description:**

FAISS-based dense retriever.

**Methods:**

- `__init__(collection_name, metric, qdrant_url, qdrant_api_key)`: Initialize Qdrant Dense Retriever.
- `_get_embedder()`: Get cached embedder or fetch from registry.
- `_map_metric_to_distance(metric)`: Map metric name to Qdrant Distance enum.
- `retrieve(query, k)`: Retrieve k most relevant chunks for the query.
- `build_index(chunks, embeddings)`: Build or update the retrieval index.
- `save_index()`: Qdrant persists automatically - no action needed.
- `load_index()`: Qdrant loads automatically - no action needed.
- `get_stats()`: Get retriever statistics.


---

## retrieval.retrievers.reranker.bi_encoder

### Class: `BiEncoderReranker`

**Inherits from:** `Reranker`

**Description:**

Bi-encoder with interaction MLP for faster reranking.

**Methods:**

- `__init__(model_name, hidden_dim)`
- `rerank(query, candidates, k)`: Encode query and passages separately, then score via interaction MLP.


---

## retrieval.retrievers.reranker.cross_encoder

### Class: `CrossEncoderReranker`

**Inherits from:** `Reranker`

*No description provided.*

**Methods:**

- `__init__(model_name, device)`
- `rerank(query, candidates, k)`: Rerank top-k candidates by cross-encoder scores


---

## retrieval.retrievers.reranker.llm_based

### Class: `LLMBasedReranker`

**Inherits from:** `Reranker`

**Description:**

LLM-based reranker using OpenAI scoring via prompt.

**Methods:**

- `__init__(api_key, model)`
- `rerank(query, candidates, k)`: Ask the LLM to score each candidate’s relevance.


---
