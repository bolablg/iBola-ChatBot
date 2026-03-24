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
- `scripts/` — Lint, format, test, security, CI scripts
- `docs/` — Full documentation

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
Version from `VERSION` file drives git tags on deploy.

## Quality Scripts

All scripts lint/format **changed files only** (vs main) for speed.

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
- Never move AGENTS.md out of the project root
