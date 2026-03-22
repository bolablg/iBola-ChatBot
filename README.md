# iBola — Production Agentic RAG Chatbot

A production-grade, multi-agent Retrieval-Augmented Generation chatbot powering [chat.bolablg.com](https://chat.bolablg.com). Built with LangGraph, Google Gemini 2.5 Pro, hybrid search (BM25 + vector + RRF), and deployed on Google Cloud Run.

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
├── docs/                   # Additional documentation
└── static/                 # Frontend
```

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

## CI/CD Pipeline

GitFlow with automated promotion:

```
Feature branch ─── lint + format + test ───▶ Auto-merge to staging
                                                     │
Staging ─────── pip-audit, bandit, safety, trivy ───▶ Auto-create PR to main
                                                     │
Main ──────────────── merge PR ─────────────────────▶ Deploy to Cloud Run
```

| Stage | Tools |
|-------|-------|
| **Lint & Format** | Black, isort, Flake8 |
| **Test** | pytest with coverage |
| **Security** | pip-audit, safety, Bandit, Trivy |
| **Deploy** | Docker → GCR → Cloud Run |

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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 2.5 Pro |
| **Agent Framework** | LangGraph + LangChain |
| **Embeddings** | Google Generative AI (`embedding-001`) |
| **Vector Store** | ChromaDB |
| **Keyword Search** | rank-bm25 (BM25Okapi) |
| **Reranker** | sentence-transformers cross-encoder |
| **API** | FastAPI + Uvicorn |
| **Config** | pydantic-settings |
| **Tracing** | Langfuse |
| **Caching** | Redis / in-memory TTLCache |
| **Deployment** | Docker → Google Cloud Run |
| **CI/CD** | GitHub Actions (GitFlow) |

## Documentation

| Document | Description |
|----------|-------------|
| [Agentic RAG Pipeline](docs/AGENTIC_RAG_PIPELINE.md) | LangGraph workflow, nodes, structured outputs, state machine |
| [Hybrid Search](docs/HYBRID_SEARCH.md) | BM25 + Vector + RRF fusion, cross-encoder reranking |
| [Data Pipeline](docs/DATA_PIPELINE.md) | Knowledge base, intelligent chunking, vectorstore updates |
| [API Reference](docs/API_REFERENCE.md) | All endpoints with request/response examples |
| [Configuration](docs/CONFIGURATION.md) | Environment variables, pydantic-settings, legacy config |
| [CI/CD Pipeline](docs/CI_CD.md) | GitFlow, security scanning, deployment |
| [Observability](docs/OBSERVABILITY.md) | Langfuse tracing, logging, rate limiting, feedback |
| [Architecture Assessment](docs/ARCHITECTURE_ASSESSMENT.md) | Original architecture review |
| [Release Notes](docs/RELEASE_NOTES.md) | Version history |

## License

This project is a professional portfolio chatbot for [Bolaji BALOGOUN](https://bolablg.com).

---

*Built with LangGraph, Google Gemini, and modern agentic RAG patterns.*
