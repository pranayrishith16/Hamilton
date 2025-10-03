Top level
README.md: Project overview, setup, flavor selection, and workflows; serves as the entry point for maintainers to understand pipeline components and responsibilities.

pyproject.toml / requirements.txt: Pinned dependencies for deterministic builds; extras for mlops/observability backends.

.env.example: Template for credentials and endpoints used by adapters; copy to .env for local/dev secrets management.

Makefile: Common commands for linting, tests, ingestion runs, evals, and deployment to standardize ops across environments.

Configs
configs/pipelines/*.yaml: Select RAG flavor and modules (retriever/reranker/memory/graph/prompt) and thresholds; registry reads this to assemble the pipeline at runtime.

configs/datasources.yaml: Data connectors, access scopes, and refresh cadence to drive ingestion jobs and indexing schedules.

configs/eval.yaml: Evaluation datasets, metrics, schedules, and deploy gates to enforce quality regressions prevention.

configs/security_policies.yaml: RBAC/ABAC/tenancy and PII/PHI handling rules enforced by middleware and retrieval filters.

configs/prompts.yaml: Prompt templates and versions mapping to domains and tasks to ensure reproducibility and rollback.

configs/mlops.yaml: Selection and credentials for mlops backend (mlflow or zenml) so tracking and lineage are swapped without code changes.

configs/observability.yaml: Tracing/metrics sampling, dashboards, budgets, and alert thresholds for runtime monitoring.

Apps (serving and workers)
apps/api/main.py: FastAPI entrypoint; validates auth, loads active flavor, and delegates to orchestrator/pipeline; returns grounded answers.

apps/api/routers/query.py: REST routes for query and chat, with domain tags to route legal/medical/general workloads.

apps/api/routers/admin.py: Health, metrics, config reloads, and index flip endpoints for SRE/operators.

apps/api/middleware/authentication.py: OAuth2/OIDC/SAML; attaches identity and tenant context to requests.

apps/api/middleware/authorization.py: Enforces RBAC/ABAC and policy decisions before retrieval; blocks unauthorized scopes.

apps/api/middleware/audit_logger.py: Writes immutable audit events correlating to traces for compliance.

apps/workers/ingest_runner.py: Launches ingestion/pipelines/ingestion_pipeline in batch/stream modes; runs via cron/queue.

apps/workers/eval_runner.py: Schedules offline evals and canary checks; posts reports and raises deploy gates.

apps/workers/index_swapper.py: Performs blue/green alias flips to new index versions after eval success, ensuring zero-downtime updates.

Orchestrator (composition and cross-cutting)
orchestrator/pipeline.py: Query-time DAG implementation: route→retrieve→rerank→generate→verify; emits traces/metrics.

orchestrator/planner.py: Optional agentic planner for multi-hop retrieval/tool use when queries are complex or underspecified.

orchestrator/registry.py: Dependency-injection container that loads concrete implementations by name/version from configs; the core “plug-and-play” engine.

orchestrator/interfaces.py: Abstract base interfaces for all component types (Retriever, Reranker, Memory, Generator, Verifier, Storage, MLOps, Metrics) to guarantee swappability.

orchestrator/adapters/mlops/mlflow_adapter.py: Records params, metrics, and artifacts to MLflow when mlops.backend=mlflow.

orchestrator/adapters/mlops/zenml_adapter.py: Integrates with ZenML pipelines/metadata when mlops.backend=zenml.

orchestrator/adapters/observability/otel_adapter.py: OpenTelemetry tracing setup and spans correlation for requests/jobs.

orchestrator/adapters/observability/metrics_adapter.py: Metrics sink abstraction for Prometheus/Datadog; exposes counters/histograms/gauges.

orchestrator/adapters/storage/*_adapter.py: Adapters that unify vector/search/object backends behind stable CRUD/search contracts.

orchestrator/rate_limits.py: Per-tenant throttling and backpressure to protect upstreams and ensure fair-use SLAs.

orchestrator/guards.py: Governance/safety gates (e.g., denylist topics, PHI/PII redaction enforcement) at generation boundaries.

orchestrator/observability.py: Utilities that standardize labels, span attributes, and metric naming across modules for consistent dashboards.

Ingestion (data prep → embeddings → indexing)
ingestion/dataprep/loaders/*.py: Source connectors (files, web, APIs, mailboxes) to pull raw content into staging.

ingestion/dataprep/parsers/*.py: PDF/DOCX/HTML/OCR to structured text blocks with metadata; critical for high-quality retrieval later.

ingestion/dataprep/cleaners/*.py: Dedupe, noise removal, and PII/PHI redaction for compliance and improved retrieval precision.

ingestion/dataprep/annotators/*.py: NER, citations, section/factor tagging to add structure and improve targeted retrieval.

ingestion/dataprep/chunkers/*.py: Semantic/hierarchical/window chunking; the most impactful step for retrieval effectiveness.

ingestion/embeddings/jobs.py: Batch/stream embedding orchestration; ensures dimensionality and model consistency.

ingestion/embeddings/models/*.py: Domain-tuned embedders (legal/medical/general) selected by pipeline config.

ingestion/indexing/vector_indexer.py: Upserts to vector DB; handles schema/versioning and index migrations.

ingestion/indexing/bm25_indexer.py: Full-text index build and analyzer settings for lexical retrieval.

ingestion/indexing/graph_indexer.py: Builds citation/entity graphs for authority/relationship-aware retrieval.

ingestion/pipelines/ingestion_pipeline.py: Glue pipeline composing all prep→embed→index steps; logs lineage via MLOps adapter.

ingestion/pipelines/refresh_policies.yaml: Triggers and schedules; defines delta/full refresh behavior.

Retrieval (serving)
retrieval/retrievers/{dense,bm25,hybrid,multi_vector,temporal}.py: Swappable retrievers implementing the Retriever interface, including fusion and time-aware variants.

retrieval/rerankers/{cross_encoder,llm}.py: Stage-2 rerankers to boost precision at top-k; selected by flavor config.

retrieval/memory/{summary_buffer,vector_memory}.py: Conversational memory strategies for multi-turn contexts.

retrieval/graphs/{citation,entity}.py: Graph-based expansion and authority weighting for structured/CBR-style retrieval.

retrieval/routers/{intent,domain}.py: Query classification and routing logic to choose retriever/prompt stacks.

retrieval/filters/{security,jurisdiction,date}.py: Policy and scope filters applied before context is passed to the generator.

retrieval/cache/result_cache.py: Layered caching for retrieval and LLM results to reduce latency/cost.

Generation
generation/prompts/*: Domain/task-specific templates; pinned via configs/prompts.yaml for reproducibility and rollback.

generation/models/adapters/{openai,anthropic,local}.py: Model adapters hiding vendor differences; selected by config and routed by SLA/cost.

generation/models/routers/cost_latency.py: Chooses model per latency/cost/SLA constraints for each request.

generation/postprocessors/{style_normalizer,redactor,guardrails}.py: Style unification, PHI/PII scrubbing, and policy enforcement on outputs.

Verification
verification/claim_checkers/{quote_grounder,faithfulness_checker}.py: Checks grounding by matching claims to retrieved spans and scoring faithfulness.

verification/fact_explainers/{support_explainer,gap_finder}.py: Explains support status and triggers corrective retrieval loops for unsupported claims.

Evaluation
evaluation/datasets/{legalbench_rag,clinical_qa}/: Gold sets for domain evaluation to avoid regressions and measure impact of swaps.

evaluation/runners/{ragas,giskard,triad}.py: Offline metrics runners; the RAG Triad pinpoints where to optimize (retrieval vs generation).

evaluation/reports/{html,csv}/: Human-friendly dashboards and raw CSVs for audit and trend analysis.

evaluation/ab_testing/configs/*.yaml, runner.py: A/B harnesses to compare retrievers, rerankers, and prompts under consistent datasets.

Governance
governance/policies/{security,privacy,tenancy}.yaml: Machine-enforceable policies surfaced in middleware/filters/guards.

governance/audits/{event_logger,retention}.py: WORM audit logs and retention scheduling required for compliance-sensitive domains.

governance/approvals/{prompt_change,model_change}.md: Change-control templates to formalize risky updates.

Observability (monitoring)
observability/tracing/{otel_setup,request_id}.py: Distributed tracing to follow a request end-to-end across modules.

observability/metrics/{exporters,quality_kpis,cost}.py: KPIs such as context relevance proxies, groundedness, answer relevance, and spend.

observability/alerts/{slo,security,quality}.yaml: Alert policies triggering on SLA breaches, leakage, or quality regressions.

Orchestrations (tool swaps)
orchestrations/zenml/pipelines/{ingestion,eval}.py and stack.yaml: ZenML wrappers if mlops.backend=zenml, mapping native steps to this repo’s modules.

orchestrations/mlflow/{tracking,registry}.py: MLflow tracking/registry utilities if mlops.backend=mlflow; business logic remains unchanged.

orchestrations/adapters.md: Instructions and patterns for adding or switching MLOps backends without touching core logic.

Storage
storage/object_store/{buckets,versioning}.py: Raw/processed/embeddings buckets and immutable snapshots for reproducibility.

storage/vector_store/{weaviate,pinecone,faiss, schema_manager}.py: Multiple vector DB clients behind an adapter; schema/version management.

storage/search_index/{opensearch,elastic,pipelines}.py: Lexical index clients and indexing pipelines for BM25.

storage/relational/postgres.py: RDBMS for metadata, lineage, and audit indices.

Tests
tests/contracts/*: Contract tests ensure any new component conforms to interfaces; key to safe plug-and-play swaps.

tests/e2e/*: End-to-end tests for legal/medical workflows exercising full stack under realistic inputs.

tests/load/locustfile.py: Load/performance tests to validate SLAs and capacity plans.

tests/quality/test_online_sampling.py: Live sampling checks to detect silent degradations in production.

tests/security/*: Leakage/access control tests that gate deploys in regulated environments.

If a map of dependencies is needed, a Mermaid diagram or an ADR can outline module relationships and how configs feed the registry to assemble different RAG flavors end-to-end.

