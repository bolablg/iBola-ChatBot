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

# Sync KB from the website canon (llms-full.txt) and upsert the vector store
python pipeline/sync_website.py

# KB freshness monitor (exit 1 if KB lags the site or has stale phrases)
python pipeline/sync_website.py --check

# Rebuild the vector store from scratch
python pipeline/update_vectorstore.py --rebuild

# Golden-QA eval (full run, LLM judge; report under local/eval_reports/)
python scripts/run_eval.py

# Eval smoke subset / against a deployment / model sweep candidate
python scripts/run_eval.py --tags smoke
python scripts/run_eval.py --base-url https://<cloud-run-url>
python scripts/run_eval.py --model gemini-2.5-pro

# Gate a report against the accepted baseline (CI does this on every PR)
python scripts/eval_gate.py --report local/eval_reports/eval_<ts>.json

# Chunk-quality audit (sample the vector store, flag noise)
python scripts/audit_chunks.py --flagged-only
```

## Eval discipline

- `eval/golden.jsonl` is the golden QA set; `eval/accepted_baseline.json` is the
  committed baseline the CI gate compares against.
- Every retrieval or generation change must show its eval delta before merging.
  Accept a new baseline deliberately with `python scripts/run_eval.py --accept`.
- Context-budget sweeps are env changes, not code changes:
  `SEARCH_GENERATION_CONTEXT_DOCS=3 python scripts/run_eval.py`.

## Commit Rules

- Never include `Co-Authored-By` lines in commit messages
- Never move CLAUDE.md out of the project root


## Standing rules (owner's instructions)

1. **Cross-check everything substantive with Codex.** Copy, branding, SEO, strategy, and factual changes must be reviewed with the Codex CLI (`codex exec --sandbox read-only -C <repo> "<brief>"`) before implementation. Plan with it, then verify with it.
2. **No em dashes (—) anywhere** in rendered content, titles, seoTitles, or machine files. Use commas, colons, parentheses, `·`, or en dash (–) for numeric ranges.
3. **No filler copy.** Every sentence must state a verifiable fact, a concrete capability, or a specific way of working. Banned: "effective use of data", "measurable business impact", "empowering", "seamless", unqualified "impact"/"high-impact", stock values lists.