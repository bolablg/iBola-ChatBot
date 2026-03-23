# iBola Agentic RAG Chatbot

## Architecture

This is a **production-grade agentic multi-agent RAG chatbot** built with:

- **LangGraph** for the agentic workflow (guardrail → retrieve → grade → generate / rewrite)
- **Google Gemini 2.5 Flash** as the LLM (configurable via `LLM_MODEL_NAME`)
- **ChromaDB** for vector storage with **gemini-embedding-001** (sync client)
- **BM25 + Vector MMR + RRF** hybrid search with optional cross-encoder reranking
- **FastAPI** for the API layer (sync workflows run in thread pool via `run_in_executor`)
- **Google Cloud Run** for deployment (2 GiB memory, min-instances=1)
- **Google Secret Manager** for runtime secrets (API keys, SA credentials)

## Key Directories

- `app/graph/` — LangGraph agentic workflow (state, nodes, workflow, service)
- `app/agents/` — Legacy multi-agent system (orchestrator, specialized agents)
- `app/services/` — Shared services (search, cache, rate limiting, tracing, logging)
- `app/routes/` — API routes (streaming, feedback)
- `pipeline/` — Data ingestion (chunker, vectorstore update, Google Drive sync)
- `data/` — Knowledge base documents
- `tests/` — Test suite
- `scripts/` — Lint, format, test, security, CI scripts
- `docs/` — Full documentation

## API Endpoints

- `POST /ask-agentic` — **Primary**: Full agentic RAG pipeline (LangGraph, supports SSE streaming, 12s timeout)
- `POST /chat` — Legacy chat (uses orchestrator, kept for backward compatibility)
- `POST /ask` — Simple RAG (fast, no agent routing)
- `POST /feedback` — User feedback for quality monitoring
- `GET /health` — System health check

## Production RAG Design Principles

- **Intent-first routing**: Guardrail classifies before retrieval. Deterministic intents (contact, opportunity) skip the RAG pipeline entirely.
- **Bounded latency**: All external calls have hard timeouts (LLM=8s, pipeline=12s, collector=4s).
- **No heavyweight init on request path**: Models and indexes load at startup or first request, never mid-pipeline.
- **Hybrid search with fallbacks**: Vector MMR → BM25 → keyword overlap. Never returns empty without trying all paths.
- **Sync embedding client**: `utils/embedder.py` uses `google.genai.Client` (sync) to avoid uvicorn event-loop corruption.

## Configuration

Settings loaded via `pydantic-settings` from `.env` file. See `app/settings.py`.
Legacy config in `config.py` still works for backward compatibility.

## CI/CD

GitFlow: feature → staging (lint+test+auto-merge) → main (security scan → deploy)
Version from `VERSION` file drives git tags on deploy.

## Quality Scripts

Scripts lint/format changed files first, then verify ALL project files pass (including `utils/`).

```bash
bash scripts/format.sh    # Auto-format changed .py files (Black + isort)
bash scripts/lint.sh      # Lint changed .py files (Flake8)
bash scripts/test.sh      # Run full test suite with coverage
bash scripts/security.sh  # Dependency audit (pip-audit) + Bandit on changed files
bash scripts/check.sh     # Format + lint + security (quick pre-commit check)
bash scripts/ci-local.sh  # Full CI pipeline: format + lint + test + security
```

## Before Completing Any Task

**Always run `bash scripts/ci-local.sh` before marking a task as done.**
This runs the full local CI pipeline (format → lint → test → security) — same checks as GitHub Actions.
If any step fails, fix the issues before proceeding.

At minimum, run `bash scripts/check.sh` for a quick pre-commit check (skips tests for speed).

## Commands

```bash
# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run with Docker
docker compose build && docker compose up -d

# Update vector store (inside Docker)
docker compose run --rm app python pipeline/update_vectorstore.py

# Run full local CI (format + lint + test + security)
bash scripts/ci-local.sh

# Run tests only
bash scripts/test.sh

# Run a single test
bash scripts/test.sh -k test_health

# Run tests without coverage
bash scripts/test.sh --no-cov
```

## Commit Rules

- Never include `Co-Authored-By` lines in commit messages
- Never move CLAUDE.md out of the project root
