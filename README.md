# iBola — Production Agentic RAG Chatbot

[![Version](https://img.shields.io/badge/version-1.0.1-005AF0?style=for-the-badge)](https://github.com/bolablg/agentic-rag-chatbot/releases)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Pro-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Cloud Run](https://img.shields.io/badge/Cloud_Run-deployed-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://chat.bolablg.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-FF6F00?style=for-the-badge)](https://www.truma.com/chromadb)
[![Docker](https://img.shields.io/badge/Docker-container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-proprietary-grey?style=for-the-badge)]()

A production-grade, multi-agent Retrieval-Augmented Generation chatbot powering [chat.bolablg.com](https://chat.bolablg.com). Built with LangGraph, Google Gemini 2.5 Pro, hybrid search (BM25 + vector + RRF), and deployed on Google Cloud Run.

---

## Architecture

```
User Query
    │
    ▼
┌────────────┐   score ≥ 60   ┌──────────┐   relevant?   ┌──────────┐
│ Guardrail  │ ──────────────▶ │ Retrieve │ ────────────▶ │ Generate │ ──▶ Answer
│ (0-100)    │                 │ BM25+Vec │               │          │
└────────────┘                 │ +RRF     │               └──────────┘
    │ score < 60               └──────────┘                    ▲
    ▼                               │ not relevant             │
┌────────────┐               ┌──────────────┐                  │
│ Out of     │               │ Rewrite      │──── retry ───────┘
│ Scope      │               │ Query        │  (max 2 attempts)
└────────────┘               └──────────────┘
```

**LangGraph workflow** with 6 nodes: `guardrail` → `retrieve` → `grade_documents` → `generate` / `rewrite_query` / `out_of_scope`. All parsing-critical LLM calls use **Pydantic structured outputs** at temperature 0.0. Every node has graceful fallbacks.

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Agentic RAG** | LangGraph state machine with document grading, query rewriting, and retry logic |
| **Hybrid Search** | BM25 (rank-bm25) + ChromaDB vector MMR + Reciprocal Rank Fusion |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` post-retrieval reranker |
| **Multi-Agent** | Professional, Education, Learning, Redirect — domain-specific prompts and retrievers |
| **Guardrails** | Structured LLM scoring (0-100) with configurable threshold |
| **SSE Streaming** | Real-time token delivery via Server-Sent Events |
| **Multilingual** | 10+ languages with automatic detection |
| **Observability** | Langfuse tracing, Google Cloud Logging, feedback endpoint |
| **Caching** | Multi-level TTL cache (response, session, language) |
| **Security** | Input validation, rate limiting, CORS, XSS/SQL injection guards |

---

## Tech Stack

### Core

| Component | Technology | Version / Details |
|-----------|-----------|-------------------|
| **Language** | Python | 3.12+ |
| **LLM** | Google Gemini | 2.5 Pro |
| **Agent Framework** | LangGraph | 0.2+ — stateful workflow graph |
| **Chain Framework** | LangChain | 0.3+ — LCEL, retrievers, prompts |
| **Embeddings** | Google Generative AI | `models/embedding-001` |
| **Structured Outputs** | Pydantic | v2 — `GuardrailScoring`, `GradeDocuments`, `QueryRewrite` |

### Search & Retrieval

| Component | Technology | Details |
|-----------|-----------|---------|
| **Vector Store** | ChromaDB | SQLite backend, MMR search |
| **Keyword Search** | rank-bm25 | BM25Okapi with EN+FR stopwords |
| **Score Fusion** | Reciprocal Rank Fusion | `rank_constant=60` |
| **Reranker** | sentence-transformers | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Chunking** | Custom `IntelligentChunker` | Section-based, metadata-enriched, category-tagged |

### API & Web

| Component | Technology | Details |
|-----------|-----------|---------|
| **Framework** | FastAPI | 0.115+ with async support |
| **Server** | Uvicorn | ASGI, HTTP/1.1 |
| **Streaming** | SSE (Server-Sent Events) | Via `sse-starlette` |
| **Validation** | Pydantic | Request/response models |
| **Config** | pydantic-settings | Type-safe, hierarchical, `.env` support |

### Infrastructure & DevOps

| Component | Technology | Details |
|-----------|-----------|---------|
| **Containerization** | Docker | Python 3.12-slim base |
| **Deployment** | Google Cloud Run | us-central1, public |
| **Container Registry** | Google Container Registry | `gcr.io` |
| **CI/CD** | GitHub Actions | GitFlow: feature → staging → main |
| **Domain** | `chat.bolablg.com` | Cloud Run domain mapping |

### Observability & Monitoring

| Component | Technology | Details |
|-----------|-----------|---------|
| **Tracing** | Langfuse | Per-span pipeline tracing (opt-in) |
| **Logging** | Google Cloud Logging | Structured JSON, severity filtering |
| **Feedback** | Custom endpoint | `POST /feedback` → Langfuse scores |
| **Metrics** | psutil + custom | CPU, memory, cache stats, rate limits |
| **Alerts** | Google Chat webhook | Contact requests, redirect limits |
| **Analytics** | Google Sheets | Redirect event logging |

### Caching & Storage

| Component | Technology | Details |
|-----------|-----------|---------|
| **Response Cache** | TTLCache | 30-minute TTL |
| **Session Cache** | TTLCache | 1-hour TTL |
| **Language Cache** | TTLCache | 2-hour TTL |
| **Redis** | Optional | Production cache backend |
| **Chat History** | Redis / in-memory | Per-session storage |

### Security

| Component | Technology | Details |
|-----------|-----------|---------|
| **Rate Limiting** | Sliding window | Per-IP, per-endpoint + global |
| **Input Validation** | Pydantic + custom | XSS, SQL injection, length checks |
| **CORS** | FastAPI middleware | Regex-based origin matching |
| **Dependency Scanning** | pip-audit, safety | CVE detection in CI |
| **Static Analysis** | Bandit | Python security linting |
| **Container Scanning** | Trivy | CRITICAL + HIGH severity |

### Code Quality

| Tool | Purpose |
|------|---------|
| **Black** | Code formatting (line length 88) |
| **isort** | Import sorting (Black-compatible) |
| **Flake8** | Static analysis (max line 100) |
| **mypy** | Type checking |
| **pre-commit** | Git hook automation |
| **pytest** | Testing (80% coverage target) |

---

## Project Structure

```
├── app/
│   ├── graph/              # LangGraph agentic workflow
│   │   ├── state.py        #   GraphState + Pydantic structured outputs
│   │   ├── nodes.py        #   Guardrail, retrieve, grade, rewrite, generate, out_of_scope
│   │   ├── workflow.py     #   StateGraph definition with conditional edges
│   │   └── service.py      #   AgenticRAGService (production wrapper)
│   ├── agents/             # Legacy multi-agent system (backward compatible)
│   ├── routes/             # API routes
│   │   ├── streaming.py    #   /ask-agentic (SSE), /ask (simple RAG)
│   │   └── feedback.py     #   POST /feedback
│   ├── services/           # Shared services
│   │   ├── advanced_rag.py #   Hybrid search: BM25 + Vector + RRF + reranker
│   │   ├── tracing.py      #   Langfuse integration
│   │   ├── cache_service.py
│   │   ├── rate_limiting.py
│   │   └── logging_service.py
│   ├── settings.py         # pydantic-settings configuration
│   └── main.py             # FastAPI app
├── pipeline/
│   ├── chunker.py          # Intelligent section-based chunking
│   ├── update_vectorstore.py
│   └── sync.py             # Google Drive sync
├── data/                   # Knowledge base (14 documents)
├── chroma_db/              # ChromaDB vector store
├── tests/
├── docs/                   # Documentation
├── static/                 # Frontend
├── VERSION                 # Release version (drives git tags)
└── CLAUDE.md               # Claude Code project context
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask-agentic` | Full agentic RAG pipeline (LangGraph). Supports `stream=true` for SSE. |
| `POST` | `/ask` | Simple RAG — direct retrieval + generation, no agent routing. Fast (2-5s). |
| `POST` | `/chat` | Legacy chat endpoint (orchestrator-based routing). |
| `POST` | `/welcome` | Localized welcome messages based on browser language. |
| `POST` | `/feedback` | Submit user quality ratings (feeds into Langfuse). |
| `POST` | `/contact-alert` | Forward contact requests to Google Chat. |
| `GET` | `/health` | System health check with resource metrics. |
| `GET` | `/performance/metrics` | CPU, memory, cache, rate limit stats. |
| `GET` | `/session/{id}/stats` | Session analytics. |
| `DELETE` | `/session/{id}` | Reset session. |

Interactive docs at `/docs` (Swagger) and `/redoc`.

---

## Quick Start

### Prerequisites

- Python 3.12+
- Google Gemini API key

### Install & Run

```bash
# Clone
git clone https://github.com/bolablg/agentic-rag-chatbot.git
cd agentic-rag-chatbot

# Install
pip install -r requirements.txt

# Configure
cp sample.env .env
# Edit .env and set GEMINI_API_KEY

# Update vector store with knowledge base
python pipeline/update_vectorstore.py

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker-compose build && docker-compose up -d
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `GCP_PROJECT_ID` | No | Google Cloud project ID |
| `GCP_SA_CREDENTIALS_PATH` | No | Path to service account JSON |
| `GCHAT_WEBHOOK_URL` | No | Google Chat webhook for alerts |
| `REDIRECT_LOG_SHEET_ID` | No | Google Sheets ID for redirect logging |
| `LANGFUSE_ENABLED` | No | Enable Langfuse tracing (`true`/`false`) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |

Full configuration reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md)

---

## CI/CD Pipeline

GitFlow with automated promotion and version tagging:

```
Feature branch ─── lint + format + test ───▶ Auto-merge to staging
                                                     │
Staging ─────── pip-audit, bandit, safety, trivy ───▶ Auto-create PR to main
                                                     │
Main ──────────────── merge PR ─────────────────────▶ Deploy to Cloud Run + tag vX.Y.Z
```

Version is read from the `VERSION` file at project root. Update it before merging to main.

| Stage | Tools |
|-------|-------|
| **Lint & Format** | Black, isort, Flake8 |
| **Test** | pytest with coverage |
| **Security** | pip-audit, safety, Bandit, Trivy |
| **Deploy** | Docker → GCR → Cloud Run |
| **Tag** | Annotated git tag from `VERSION` file |

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov=app

# Lint & format
black app/ tests/ pipeline/ && isort app/ tests/ pipeline/ && flake8 app/ tests/ pipeline/

# Update knowledge base
python pipeline/update_vectorstore.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Agentic RAG Pipeline](docs/AGENTIC_RAG_PIPELINE.md) | LangGraph workflow, nodes, structured outputs, state machine |
| [Hybrid Search](docs/HYBRID_SEARCH.md) | BM25 + Vector + RRF fusion, cross-encoder reranking |
| [Data Pipeline](docs/DATA_PIPELINE.md) | Knowledge base, intelligent chunking, vectorstore updates |
| [API Reference](docs/API_REFERENCE.md) | All endpoints with request/response examples |
| [Configuration](docs/CONFIGURATION.md) | Environment variables, pydantic-settings, legacy config |
| [CI/CD Pipeline](docs/CI_CD.md) | GitFlow, security scanning, deployment, tagging |
| [Observability](docs/OBSERVABILITY.md) | Langfuse tracing, logging, rate limiting, feedback |
| [Architecture Assessment](docs/ARCHITECTURE_ASSESSMENT.md) | Original architecture review |
| [Release Notes](docs/RELEASE_NOTES.md) | Version history |

---

## License

This project is a professional portfolio chatbot for [Bolaji BALOGOUN](https://bolablg.com).

---

*Built with LangGraph, Google Gemini, and modern agentic RAG patterns.*
