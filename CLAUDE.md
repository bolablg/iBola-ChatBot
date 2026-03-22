# iBola Agentic RAG Chatbot

## Architecture

This is a **production-grade agentic multi-agent RAG chatbot** built with:

- **LangGraph** for the agentic workflow (guardrail → retrieve → grade → generate / rewrite)
- **Google Gemini 2.5 Pro** as the LLM
- **ChromaDB** for vector storage
- **BM25 + Vector + RRF** hybrid search
- **FastAPI** for the API layer
- **Google Cloud Run** for deployment

## Key Directories

- `app/graph/` — LangGraph agentic workflow (state, nodes, workflow, service)
- `app/agents/` — Legacy multi-agent system (orchestrator, specialized agents)
- `app/services/` — Shared services (search, cache, rate limiting, tracing, logging)
- `app/routes/` — API routes (streaming, feedback)
- `pipeline/` — Data ingestion (chunker, vectorstore update, Google Drive sync)
- `data/` — Knowledge base documents
- `tests/` — Test suite

## API Endpoints

- `POST /chat` — Legacy chat (uses orchestrator)
- `POST /ask-agentic` — Full agentic RAG pipeline (LangGraph, supports SSE streaming)
- `POST /ask` — Simple RAG (fast, no agent routing)
- `POST /feedback` — User feedback for quality monitoring
- `GET /health` — System health check

## Configuration

Settings loaded via `pydantic-settings` from `.env` file. See `app/settings.py`.
Legacy config in `config.py` still works for backward compatibility.

## CI/CD

GitFlow: feature → staging (lint+test+auto-merge) → main (security scan → deploy)

## Commands

```bash
# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Update vector store
python pipeline/update_vectorstore.py

# Run tests
pytest tests/ -v --cov=app

# Lint
black app/ tests/ pipeline/ && isort app/ tests/ pipeline/ && flake8 app/ tests/ pipeline/
```
