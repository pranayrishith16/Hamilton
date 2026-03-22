# Hamilton - Legal RAG Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382C)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production-grade FastAPI backend for AI-powered legal research platform with RAG (Retrieval-Augmented Generation), authentication, conversation memory, and document management.

**Live API:** [https://www.veritlyai.com](https://www.veritlyai.com)  
**Frontend:** [ronaldo](https://github.com/pranayrishith16/ronaldo) | **Ingestion:** [verstappen](https://github.com/pranayrishith16/verstappen)

---

## 🎯 Overview

Hamilton is the backend engine powering VERITLY AI, a conversational legal research assistant for attorneys. It combines:

- **Hybrid RAG Pipeline** - BM25 + dense retrieval with reranking for precision
- **JWT Authentication** - Secure auth with refresh token rotation
- **Conversation Memory** - PostgreSQL-backed chat history for context-aware responses
- **Document Management** - Azure Blob Storage integration with secure PDF streaming
- **Tier-based Quotas** - Free, Basic, Pro, Enterprise subscription tiers
- **MLOps Integration** - MLflow tracking for observability

### Key Capabilities

✅ **Streaming Responses** - Real-time answer generation via Server-Sent Events  
✅ **Source Citations** - Every answer includes legal document references  
✅ **Conversation Context** - Auto-loads previous Q&A for follow-up questions  
✅ **Multi-Retriever** - Hybrid search (BM25 + Qdrant dense + reranking)  
✅ **Secure PDF Viewing** - Token-protected document streaming  
✅ **Production-Ready** - Azure SQL, connection pooling, error handling

---

## 🏗️ Architecture

### System Design

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│   FastAPI    │─────▶│  Auth DB    │
│  (ronaldo)  │      │   Backend    │      │ (Azure SQL) │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
     ┌──────▼─────┐  ┌─────▼──────┐ ┌─────▼──────┐
     │  RAG       │  │  Memory    │ │ Document   │
     │  Pipeline  │  │  System    │ │ Storage    │
     └────────────┘  └────────────┘ └────────────┘
            │               │               │
     ┌──────▼─────┐  ┌─────▼──────┐ ┌─────▼──────┐
     │  Qdrant    │  │ PostgreSQL │ │   Azure    │
     │  (Vector)  │  │  (Memory)  │ │   Blob     │
     └────────────┘  └────────────┘ └────────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Framework** | FastAPI 0.115+ | Async API server |
| **Database** | Azure SQL Server | User auth, subscriptions |
| **Memory DB** | PostgreSQL/SQLite | Conversation history |
| **Vector DB** | Qdrant Cloud | Dense retrieval |
| **Storage** | Azure Blob Storage | PDF documents |
| **LLM Provider** | OpenRouter API | Model-agnostic generation |
| **ORM** | SQLAlchemy 2.0 | Database abstraction |
| **Auth** | JWT + bcrypt | Secure authentication |
| **Observability** | MLflow | Experiment tracking |

### Project Structure

```
hamilton/
├── apps/
│   └── api/
│       └── main.py                  # FastAPI app entry point
├── auth/
│   ├── auth_manager.py              # Core auth logic (JWT, bcrypt)
│   ├── auth_routes.py               # Login, signup, refresh endpoints
│   ├── models.py                    # SQLAlchemy auth models
│   ├── rbac_dependencies.py         # Role-based access control
│   ├── cache_manager.py             # In-memory session cache
│   └── tier_config.py               # Subscription tier limits
├── memory/
│   ├── database.py                  # Memory DB connection
│   ├── models.py                    # Conversation, ChatMessage models
│   ├── repository.py                # Data access layer
│   ├── service.py                   # Business logic
│   ├── schemas.py                   # Pydantic request/response
│   └── memory_routes.py             # Conversation history API
├── documents/
│   └── doc_routes.py                # Secure PDF viewing
├── ingestion/
│   ├── dataprep/
│   │   ├── parsers/                 # PDF, XML parsers
│   │   ├── chunkers/                # Text chunking strategies
│   │   ├── cleaners/                # Text preprocessing
│   │   └── loaders/                 # Azure Blob loader
│   ├── embeddings/
│   │   └── models/                  # Sentence transformers
│   └── pipelines/
│       └── ingestion_pipeline.py    # Full ingestion workflow
├── retrieval/
│   ├── retrievers/
│   │   ├── bm25.py                  # BM25 sparse retrieval
│   │   ├── qdrantDense.py           # Qdrant dense retrieval
│   │   ├── hybrid.py                # RRF fusion
│   │   └── reranker/                # Cross-encoder reranking
│   └── interface.py                 # Retriever abstraction
├── generation/
│   ├── models/
│   │   └── adapters/
│   │       └── openrouter.py        # OpenRouter LLM adapter
│   └── prompts/
│       └── render_template.py       # Jinja2 prompt templates
├── orchestrator/
│   ├── pipeline.py                  # Main RAG orchestration
│   ├── registry.py                  # Component registry
│   └── observability.py             # Tracing and metrics
├── configs/
│   ├── components.yaml              # Component configuration
│   └── prompts.yaml                 # LLM prompt templates
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** and pip
- **Azure SQL Server** (or PostgreSQL for memory)
- **Qdrant Cloud** account (or local Qdrant)
- **Azure Blob Storage** account
- **OpenRouter API** key

### Installation

```bash
# Clone repository
git clone https://github.com/pranayrishith16/hamilton.git
cd hamilton

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
# Database
AZURE_SQL_CONNECTION_STRING=Server=tcp:your-server.database.windows.net,1433;Initial Catalog=veritly_db;User ID=admin;Password=your-password
MEMORY_DATABASE_URL=postgresql://user:password@localhost/memory_db

# Authentication
JWT_SECRET=your-super-secret-jwt-key-min-32-chars

# Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your-azure-storage-connection-string
AZURE_BLOB_CONTAINER=legal-docs

# LLM
OPENROUTER_API_KEY=your-openrouter-api-key

# MLflow (optional)
MLFLOW_TRACKING_URI=http://localhost:5050
```

### Database Setup

```bash
# Initialize auth database
python -c "from auth.models import init_database; init_database()"

# Initialize memory database
python -c "from memory.database import DatabaseManager; DatabaseManager.initialize(); DatabaseManager.create_tables()"
```

### Run Development Server

```bash
# Start FastAPI server
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

# API available at: http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### Optional: Run MLflow Tracking

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow_artifacts \
  --host 0.0.0.0 \
  --port 5050
```

---

## 🔐 Authentication Flow

Hamilton uses JWT-based authentication with refresh token rotation:

```
1. User registers → Email verification (optional)
2. User logs in → Receive access_token + refresh_token
3. Access token expires (1 hour) → Use refresh_token to get new access_token
4. Refresh token rotates (max 5 times) → Force re-login for security
```

### Example: Login Request

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "attorney@lawfirm.com",
    "password": "SecurePassword123"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "long_refresh_token_string",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "attorney@lawfirm.com",
  "tier": "free"
}
```

### Subscription Tiers

| Tier | Daily Queries | Price | Features |
|------|--------------|-------|----------|
| **Free** | 5 | $0 | Basic legal search, community support |
| **Basic** | 50 | $29/mo | Advanced search, email support |
| **Pro** | 200 | $99/mo | API access, team collaboration, priority support |
| **Enterprise** | Unlimited | Custom | Custom integrations, SLA, 24/7 support |

---

## 💬 RAG Pipeline Architecture

### Query Flow

```
User Query
    ↓
[1. Load Conversation Context]  ← Memory system (last 2 Q&A pairs)
    ↓
[2. Hybrid Retrieval]           ← BM25 (sparse) + Qdrant (dense)
    ↓
[3. RRF Fusion]                 ← Reciprocal Rank Fusion
    ↓
[4. Reranking]                  ← Cross-encoder (optional)
    ↓
[5. Generation]                 ← OpenRouter LLM with context
    ↓
[6. Save to Memory]             ← Store Q&A for future context
    ↓
Response + Sources
```

### Streaming Implementation

**Endpoint:** `POST /api/query/stream`

```python
# Server-Sent Events (SSE) format
data: {"content": "Insider trading is...", "sources": [...]}
data: {"content": " the purchase or sale...", "sources": []}
data: {"content": " of a security...", "sources": []}
data: DONE
```

**Frontend consumption:**
```javascript
const eventSource = new EventSource('/api/query/stream');
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.content) appendToChat(data.content);
};
```

### Retrieval Components

**1. BM25 Retriever** (`retrieval/retrievers/bm25.py`)
- Sparse lexical search
- TF-IDF based scoring
- Fast, exact keyword matching

**2. Qdrant Dense Retriever** (`retrieval/retrievers/qdrantDense.py`)
- Semantic search using embeddings
- Sentence transformers (all-MiniLM-L6-v2)
- Handles synonyms and paraphrasing

**3. Hybrid Retriever** (`retrieval/retrievers/hybrid.py`)
- RRF (Reciprocal Rank Fusion)
- Combines BM25 + dense results
- Weights: 60% dense, 40% BM25

**4. Reranker** (optional)
- Cross-encoder for final ranking
- Dramatically improves top-k precision
- Trade-off: adds 200-500ms latency

---

## 📊 Conversation Memory System

### Database Schema

**Conversations Table:**
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    title VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE
);
```

**ChatMessages Table:**
```sql
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    user_id UUID NOT NULL,
    role VARCHAR(10) CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources JSON,
    metadata JSON,
    tokens_used INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP
);
```

### Memory-Augmented Query Example

**Request:**
```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the penalties?",
    "conversation_id": "conv-uuid",
    "k": 5
  }'
```

**Behind the scenes:**
1. Load last 2 Q&A pairs from `conversation_id`
2. Build context string:
   ```
   Q: What is insider trading?
   A: Insider trading is the purchase or sale of a security...
   
   Current Q: What are the penalties?
   ```
3. Retrieve relevant documents with augmented query
4. Generate answer with full context
5. Save new Q&A to database

---

## 📄 Document Management

### Secure PDF Viewing

**Endpoint:** `GET /api/documents/view/{blob_path}`

**Flow:**
1. Frontend requests PDF with JWT in `Authorization` header
2. Backend verifies JWT, checks user permissions
3. Streams PDF from Azure Blob Storage
4. Sets `Content-Disposition: inline` for browser viewing

**Example:**
```bash
curl -X GET "http://localhost:8000/api/documents/view/case_rcpdfs/sharon_kennell_v._diahann_gates.pdf" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  --output document.pdf
```

**Frontend (React):**
```jsx
<iframe 
  src={`/api/documents/view/${documentPath}`}
  headers={{ Authorization: `Bearer ${accessToken}` }}
/>
```

### File Upload Pipeline

See [verstappen](https://github.com/pranayrishith16/verstappen) for the ingestion pipeline that:
1. Uploads PDFs to Azure Blob Storage
2. Parses PDFs with PyMuPDF
3. Extracts metadata (case name, court, docket number)
4. Chunks text (LangChain recursive splitter)
5. Generates embeddings
6. Indexes to Qdrant

---

## 🛠️ API Reference

### Core Endpoints

#### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create new user account |
| POST | `/api/auth/login` | Login and receive tokens |
| POST | `/api/auth/refresh-token` | Refresh access token |
| POST | `/api/auth/logout` | Revoke tokens |

#### RAG Query

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query/stream` | Stream AI response (SSE) |
| POST | `/api/query` | Get full response (non-streaming) |

#### Memory

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/memory/conversations` | List user's conversations |
| POST | `/api/memory/conversations` | Create new conversation |
| GET | `/api/memory/conversations/{id}/messages` | Get chat history |
| DELETE | `/api/memory/conversations/{id}` | Delete conversation |

#### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents/view/{path}` | Stream PDF securely |

### Example Requests

**Create Conversation:**
```bash
curl -X POST http://localhost:8000/api/memory/conversations \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-uuid",
    "title": "SEC Filing Analysis"
  }'
```

**Query with Streaming:**
```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is insider trading?",
    "conversation_id": "conv-uuid",
    "k": 5
  }'
```

**Full API Documentation:** Visit `/docs` (Swagger UI) when server is running

---

## 🔧 Configuration

### Component Registry

All components are configured in `configs/components.yaml`:

```yaml
embedder:
  type: SentenceTransformerEmbedder
  config:
    model_name: all-MiniLM-L6-v2

qdrant_retriever:
  type: QdrantDenseRetriever
  config:
    collection_name: legal-docs
    metric: cosine
    qdrant_url: ${QDRANT_URL}
    qdrant_api_key: ${QDRANT_API_KEY}

hybrid_retriever:
  type: HybridRetriever
  config:
    k_rrf: 60

generator:
  type: OpenRouterAdapter
  config:
    model: anthropic/claude-3.5-sonnet
    temperature: 0.0
    max_tokens: 2500
```

### Prompt Templates

Prompts are managed in `configs/prompts.yaml`:

```yaml
legal_rag_metadata:
  system: |
    You are a legal research assistant. Provide accurate, well-cited answers based on the provided legal documents.
    Always cite your sources using case names and page numbers.
  
  user: |
    {% if context %}
    Previous conversation:
    {{ context }}
    {% endif %}
    
    Retrieved documents:
    {% for chunk in chunks %}
    [{{ loop.index }}] {{ chunk.metadata.case_name }}
    {{ chunk.content }}
    {% endfor %}
    
    Question: {{ query }}
```

---

## 📈 Observability & Monitoring

### MLflow Integration

Track experiments, models, and metrics:

```python
# Automatic tracking in orchestrator/pipeline.py
with trace_request("query", "rag_pipeline.query"):
    log_metrics({
        "retrieval_time_ms": 45,
        "generation_time_ms": 2300,
        "total_tokens": 1500,
        "sources_retrieved": 5
    })
```

**View metrics:** http://localhost:5050 (MLflow UI)

### Logging

```python
# Structured logging with Loguru
logger.info(f"User {user_id} query processed in {latency_ms}ms")
logger.error(f"Auth failed for user {email}: {error}")
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "database": "connected", "qdrant": "connected"}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_auth.py
```

**Test Structure:**
```
tests/
├── contracts/          # Interface contract tests
├── e2e/                # End-to-end workflow tests
├── security/           # Security & access control tests
└── quality/            # Response quality sampling
```

---

## 🚧 Known Issues & Roadmap

### Current Issues (from TODO.md)

- [ ] Refresh token encoding verification needed (check DB vs stored)
- [ ] Backend deployment update pending

### Planned Features

- [ ] Semantic search in conversation history (using embeddings)
- [ ] Multi-file document upload
- [ ] Citation export (BibTeX, APA format)
- [ ] Advanced reranking (LLM-based)
- [ ] Query caching for common questions
- [ ] Real-time collaboration (WebSockets)
- [ ] GraphQL API option
- [ ] Rate limiting per endpoint
- [ ] PII detection and redaction
- [ ] ABAC (Attribute-Based Access Control)

---

## 📦 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t hamilton-backend .
docker run -p 8000:8000 --env-file .env hamilton-backend
```

### Azure App Service

```bash
# Deploy to Azure
az webapp up --name veritly-backend --resource-group veritly-rg --runtime "PYTHON:3.11"
```

### Environment Variables (Production)

```env
# Critical: Change these in production!
JWT_SECRET=<generate-with-secrets.token_urlsafe(64)>
AZURE_SQL_CONNECTION_STRING=<production-connection-string>
QDRANT_API_KEY=<production-qdrant-key>
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Write tests for new features
4. Ensure all tests pass (`pytest`)
5. Commit changes (`git commit -m 'Add AmazingFeature'`)
6. Push to branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

**Development Guidelines:**
- Follow PEP 8 style guide
- Add docstrings to all functions
- Use type hints
- Write unit tests
- Update README for new features

---

## 📧 Contact

**Pranay Rishith Bondugula**  
- GitHub: [@pranayrishith16](https://github.com/pranayrishith16)
- Email: pranayrishith@example.com
- Website: [https://www.veritlyai.com](https://www.veritlyai.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Qdrant](https://qdrant.tech/) - Vector similarity search engine
- [LangChain](https://www.langchain.com/) - Text chunking utilities
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [OpenRouter](https://openrouter.ai/) - Multi-model LLM API
- [SQLAlchemy](https://www.sqlalchemy.org/) - Python ORM
- [MLflow](https://mlflow.org/) - Experiment tracking

---

**⭐ If you find this project helpful, please give it a star!**

## Related Repositories

- **Frontend:** [ronaldo](https://github.com/pranayrishith16/ronaldo) - React + Redux chat interface
- **Ingestion Pipeline:** [verstappen](https://github.com/pranayrishith16/verstappen) - Document processing and indexing
- **LLM Orchestration:** [trancepoint](https://github.com/pranayrishith16/trancepoint) - LLM monitoring library
