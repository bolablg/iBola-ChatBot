# iBola, Production Agentic RAG Chatbot

[![Version](https://img.shields.io/badge/version-1.1.0-005AF0?style=for-the-badge)](https://github.com/bolablg/iBola-ChatBot/releases)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Cloud Run](https://img.shields.io/badge/Cloud_Run-deployed-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://chat.bolablg.com)
[![Langfuse](https://img.shields.io/badge/Langfuse-observability-0A0A0A?style=for-the-badge)](https://langfuse.com/)
[![Docker](https://img.shields.io/badge/Docker-container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
![License](https://img.shields.io/badge/license-proprietary-grey?style=for-the-badge)

An agentic, multi-node Retrieval-Augmented Generation chatbot powering [chat.bolablg.com](https://chat.bolablg.com). It answers questions about Bolaji BALOGOUN's professional profile from a curated knowledge base, in English and French, with grounding verification on every answer. Built with LangGraph, Google Gemini 2.5 Flash, hybrid search (BM25 + vector MMR + RRF), and a golden-QA eval gate: a smoke subset gates every pull request and the full 117-row set runs nightly.

---

## Design principles

- **Intent-first routing.** A guardrail node classifies each query before any retrieval. Deterministic intents (contact requests, prompt-exfiltration attempts, identity questions) are answered directly and never touch the RAG pipeline.
- **Bounded latency.** External calls have hard timeouts (per-node LLM 15s, collector 4s, end-to-end request 45s). Models and indexes load at startup or first request, never mid-pipeline.
- **Hybrid retrieval with fallbacks.** Vector MMR, BM25, and keyword overlap fuse via Reciprocal Rank Fusion. The pipeline never returns empty without trying every path.
- **Grounding, not just generation.** A claim-level verifier runs after generation and fails closed on unsupported profile claims, so the bot does not invent facts about a real person.
- **Everything gated by eval.** Retrieval and generation changes must show their delta against a per-tag golden baseline before merge. A smoke subset gates every PR, the full set runs nightly, and canonical facts are held at 100 percent.

---

## Architecture

```
User query
    │
    ▼
┌────────────┐  in scope   ┌──────────────┐     ┌──────────┐     ┌──────────────┐
│ Guardrail  │ ──────────▶ │ Condense     │ ──▶ │ Retrieve │ ──▶ │ Grade        │
│ intent +   │             │ query        │     │ BM25 +   │     │ documents    │
│ score 0-100│             │ (standalone, │     │ Vec MMR  │     └──────┬───────┘
└─────┬──────┘             │  EN retrieval│     │ + RRF    │            │
      │ out of scope       │  query)      │     └──────────┘     relevant│  not relevant
      ▼                    └──────────────┘                             │       │
┌────────────┐                                  ┌──────────┐            │       ▼
│ Out of     │                                  │ Generate │ ◀──────────┘  ┌──────────────┐
│ scope      │                                  └────┬─────┘               │ Rewrite query│
└────────────┘                                       │                     └──────┬───────┘
                                              ┌───────▼────────┐    retry (max 2)  │
                                              │ Verify         │ ◀─────────────────┘
                                              │ grounding      │
                                              └───────┬────────┘
                                                      ▼
                                                    Answer
```

The LangGraph workflow has eight nodes: `guardrail` → `condense_query` → `retrieve` → `grade_documents` → `generate` → `verify_grounding`, plus `rewrite_query` (retry loop, max 2 attempts) and `out_of_scope`. Parsing-critical LLM calls (guardrail, grading, grounding) use Pydantic structured outputs at temperature 0.0, each with a 15s timeout. Every node has a graceful fallback.

Full pipeline detail: [docs/AGENTIC_RAG_PIPELINE.md](docs/AGENTIC_RAG_PIPELINE.md).

---

## Key features

| Feature | Implementation |
|---------|----------------|
| Agentic workflow | LangGraph state machine: guardrail, condense, retrieve, grade, generate, verify, rewrite, out-of-scope |
| Condense-first retrieval | Pronoun and elliptical follow-ups become standalone queries, normalized to English, before the first retrieval |
| Hybrid search | BM25 (rank-bm25) + ChromaDB vector MMR + Reciprocal Rank Fusion, with temporal boost and multi-part fan-out |
| Optional multilingual reranking | `BAAI/bge-reranker-v2-m3` cross-encoder, env-gated via `ENABLE_CROSS_ENCODER_RERANKER` (multilingual by design, off by default to avoid startup latency) |
| Grounding verification | Claim-level verifier after generation, fails closed on unsupported profile claims |
| Bilingual answers | English and French parity, automatic reply-language detection, answer-language validation |
| Security guardrails | Intent-first classification, prompt-exfiltration defense, PII masking, input validation, rate limiting |
| SSE streaming | Real-time token delivery on `/ask-agentic` |
| Observability | Langfuse v4 tracing, typed feedback scores, Google Cloud Logging, monthly assessment job |
| Eval gate | 117-row golden QA set, per-tag baselines, deterministic fact checks plus pinned LLM judge; smoke subset on every PR, full set nightly |

---

## Tech stack

### Core

| Component | Technology | Details |
|-----------|-----------|---------|
| Language | Python | 3.12 |
| LLM | Google Gemini | `gemini-2.5-flash` (configurable via `LLM_MODEL_NAME`) |
| Agent framework | LangGraph | 0.2+, stateful workflow graph |
| Chain framework | LangChain | 0.3+, LCEL, retrievers, prompts |
| Embeddings | `gemini-embedding-001` | Sync `google.genai` client (avoids event-loop corruption under Uvicorn) |
| Structured outputs | Pydantic | v2: `GuardrailScoring`, `GradeDocuments`, `QueryRewrite`, `CondensedQuery`, `GroundingVerdict` |

### Search and retrieval

| Component | Technology | Details |
|-----------|-----------|---------|
| Vector store | ChromaDB | SQLite backend, MMR search (fetch_k 24, top_k 8) |
| Keyword search | rank-bm25 | BM25Okapi with EN + FR stopwords |
| Score fusion | Reciprocal Rank Fusion | `rrf_rank_constant=60` |
| Reranker | sentence-transformers | `BAAI/bge-reranker-v2-m3` (multilingual, opt-in via `ENABLE_CROSS_ENCODER_RERANKER`) |
| Generation budget | Top graded chunks | `generation_context_docs=5`, near-duplicate filtered |
| Chunking | Custom `IntelligentChunker` | Section-based, metadata-enriched, `latest_year` temporal tags |

### API and web

| Component | Technology | Details |
|-----------|-----------|---------|
| Framework | FastAPI | 0.115+ |
| Server | Uvicorn | ASGI; sync workflows run in a thread pool via `run_in_executor` |
| Streaming | Server-Sent Events | via `sse-starlette` |
| Validation | Pydantic | Request and response models |
| Config | pydantic-settings | Type-safe, hierarchical, `.env` support |

### Infrastructure and DevOps

| Component | Technology | Details |
|-----------|-----------|---------|
| Containerization | Docker | `python:3.12-slim` base |
| Deployment | Google Cloud Run | us-central1, 2 GiB, scale-to-zero with canary warmup |
| Secrets | Google Secret Manager | Gemini key, Langfuse keys, SA credentials at runtime |
| Release strategy | Canary + smoke gate | Deploy canary, warm it, run smoke eval, then promote traffic |
| CI/CD | GitHub Actions | GitFlow: feature → staging → main |
| Domain | `chat.bolablg.com` | Cloud Run domain mapping |

### Observability

| Component | Technology | Details |
|-----------|-----------|---------|
| Tracing | Langfuse v4 | Per-turn traces with per-node child spans, PII masked, salted pseudonymous visitor id |
| Feedback | `POST /feedback` | Typed scores: thumbs (boolean), thumbs reason (categorical), session CSAT (session-level) |
| Assessment | `scripts/monthly_assessment.py` | Builds a dataset from low-confidence and thumbs-down turns, posts a Google Chat digest |
| Logging | Google Cloud Logging | Structured JSON, severity filtering |
| Alerts | Google Chat webhook | Contact requests, redirect limits |

### Quality and security

| Component | Technology | Details |
|-----------|-----------|---------|
| Formatting | Black + isort | Line length 88, Black-compatible imports |
| Linting | Flake8 | Static analysis |
| Testing | pytest | 11 test modules, coverage tracked |
| Eval | Golden QA + LLM judge | `run_eval.py`, `eval_gate.py`, per-tag baselines, strict mode in CI |
| Dependency scan | pip-audit | CVE detection with reviewed allowlist |
| Static analysis | Bandit | Python security linting |
| Container scan | Trivy | Filesystem scan, blocking on CRITICAL |

---

## Project structure

```
├── app/
│   ├── graph/                  # LangGraph agentic workflow
│   │   ├── state.py            #   WorkflowState + Pydantic structured outputs
│   │   ├── nodes.py            #   guardrail, condense, retrieve, grade, rewrite, generate, verify, out-of-scope
│   │   ├── workflow.py         #   StateGraph definition with conditional edges
│   │   ├── prompts.py          #   Versioned prompt registry + exfiltration echo guard
│   │   └── service.py          #   AgenticRAGService (production wrapper, deterministic intents)
│   ├── routes/
│   │   ├── streaming.py        #   /ask-agentic (SSE), /ask (simple RAG)
│   │   └── feedback.py         #   POST /feedback (typed Langfuse scores)
│   ├── services/               # Shared services
│   │   ├── advanced_rag.py     #   Hybrid search: BM25 + vector MMR + RRF + reranker
│   │   ├── tracing.py          #   Langfuse v4 integration, PII masking, visitor hashing
│   │   ├── cache_service.py, rate_limiting.py, logging_service.py
│   │   ├── language_detection.py, google_chat_alert.py, google_sheets_logger.py
│   │   └── public_facts.py
│   ├── settings.py             # pydantic-settings configuration
│   └── main.py                 # FastAPI app + /chat, /welcome, /health, admin endpoints
├── pipeline/
│   ├── chunker.py              # Section-based chunking, dedup, temporal metadata
│   ├── update_vectorstore.py   # Source-level upsert / rebuild
│   ├── sync_website.py         # Sync KB from the website canon (llms-full.txt)
│   └── sync.py                 # Google Drive sync
├── data/                       # Knowledge base (00-17 profile docs, website canon, role timeline, public_facts.yaml)
├── eval/
│   ├── golden.jsonl            # 117-row golden QA set (EN + FR twins, temporal, metric, security tags)
│   ├── accepted_baseline*.json # Per-tag accepted baselines
│   └── langfuse_experiment.py  # PR experiment gate (Langfuse experiment-action contract)
├── scripts/                    # lint, format, test, security, ci-local, run_eval, eval_gate, audit_chunks, ...
├── tests/                      # pytest suite (agentic, grounding, security, feedback, retrieval, ...)
├── docs/                       # Full documentation
├── static/                     # Frontend (chat UI with per-message feedback)
├── VERSION                     # Release version (drives git tags)
└── CLAUDE.md                   # Claude Code project context
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask-agentic` | Primary. Full agentic RAG pipeline (LangGraph). Supports `stream=true` for SSE. 12s timeout. |
| `POST` | `/ask` | Simple RAG: direct retrieval and generation, no agent routing. |
| `POST` | `/chat` | Chat endpoint, routed through the agentic service (kept for backward compatibility). |
| `POST` | `/welcome` | Localized welcome messages based on browser language. |
| `POST` | `/feedback` | Typed quality scores (thumbs, thumbs reason, session CSAT) into Langfuse. |
| `POST` | `/contact-alert` | Forward contact requests to Google Chat. |
| `GET` | `/health` | System health check with resource metrics. |
| `GET` | `/performance/metrics` | CPU, memory, cache, and rate-limit stats. |
| `GET` | `/cache/stats`, `/rate-limit/stats` | Cache and rate-limit introspection. |
| `POST` | `/cache/clear` | Clear the response cache. |
| `GET` | `/session/{id}/stats` | Session analytics. |
| `DELETE` | `/session/{id}` | Reset a session. |

Interactive docs at `/docs` (Swagger) and `/redoc`. Full reference: [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

---

## Quick start

### Prerequisites

- Python 3.12.14
- A Google Gemini API key

### Install and run

```bash
git clone https://github.com/bolablg/iBola-ChatBot.git
cd iBola-ChatBot

pip install --require-hashes -r requirements.lock

# Configure: copy the sample env and set GEMINI_API_KEY
cp sample.env .env

# Build the vector store from the knowledge base
python pipeline/update_vectorstore.py

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker compose build && docker compose up -d

# Build the vector store inside the container
docker compose run --rm app python pipeline/update_vectorstore.py
```

### Dependency locks

The checked-in lock files pin direct and transitive packages with hashes. Use
`requirements.lock` for runtime work and `requirements-dev.lock` for tests and
security tooling. Regenerate both only when intentionally reviewing a
dependency update:

```bash
bash scripts/compile_requirements.sh
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `LLM_MODEL_NAME` | No | Override the LLM (default `gemini-2.5-flash`) |
| `GCP_PROJECT_ID` | No | Google Cloud project id |
| `GCP_SA_CREDENTIALS_PATH` | No | Path to service account JSON |
| `GCHAT_WEBHOOK_URL` | No | Google Chat webhook for alerts |
| `REDIRECT_LOG_SHEET_ID` | No | Google Sheets id for redirect logging |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | No | Langfuse credentials (tracing auto-enables when both are present) |
| `LANGFUSE_BASE_URL` | No | Langfuse host (for example `https://us.cloud.langfuse.com`) |

Full configuration reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

---

## Evaluation

The golden QA set (`eval/golden.jsonl`, 117 rows) is the correctness contract. Every retrieval or generation change must show its eval delta before merge.

```bash
# Full run (deterministic fact checks + pinned LLM judge); report under local/eval_reports/
python scripts/run_eval.py

# Smoke subset, or against a live deployment
python scripts/run_eval.py --tags smoke
python scripts/run_eval.py --base-url https://chat.bolablg.com

# Gate a report against the accepted baseline (CI gates the smoke report on every PR, strict)
python scripts/eval_gate.py --report local/eval_reports/eval_<ts>.json

# Accept a new baseline deliberately
python scripts/run_eval.py [--tags smoke] --accept
```

Baselines are per tag set: `eval/accepted_baseline.json` (full) and `eval/accepted_baseline_smoke.json` (smoke). In CI the gate runs `--strict`: a missing baseline, any hard-error row, or a latency p95 regression fails the build. The security tag is held at 100 percent.

---

## CI/CD pipeline

GitFlow with automated promotion, an eval gate, and version tagging:

```
Feature branch ── lint + test + smoke eval gate ──▶ auto-merge to staging
                                                          │
Staging ──── pip-audit + bandit + trivy ──▶ auto-create release PR to main
                                                          │
Main ──── merge PR ──▶ deploy canary + warm + smoke eval ──▶ promote traffic + tag vX.Y.Z
```

| Stage | Tools |
|-------|-------|
| Lint and format | Black, isort, Flake8 |
| Test | pytest with coverage |
| Eval | `run_eval.py` + `eval_gate.py`, smoke subset on push/PR (per-tag baselines, strict) |
| Security | pip-audit, Bandit, Trivy (blocking on CRITICAL, with a reviewed allowlist) |
| Deploy | Docker → GCR → Cloud Run canary → smoke gate → promote |
| Tag | Annotated git tag from the `VERSION` file |

Two scheduled workflows keep the system honest: a nightly full 117-row eval and a daily KB refresh that routes the new knowledge base through the same lint, test, and eval funnel as code. See [docs/CI_CD.md](docs/CI_CD.md).

---

## Development

```bash
# Auto-format changed files (Black + isort)
bash scripts/format.sh

# Lint changed files (Flake8)
bash scripts/lint.sh

# Full test suite with coverage
bash scripts/test.sh

# Run a single test
bash scripts/test.sh -k test_health

# Dependency audit + Bandit
bash scripts/security.sh

# Quick pre-commit check (format + lint + security)
bash scripts/check.sh

# Full local CI (format + lint + test + security), run before every task
bash scripts/ci-local.sh
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Agentic RAG Pipeline](docs/AGENTIC_RAG_PIPELINE.md) | LangGraph workflow, nodes, structured outputs, state machine |
| [Hybrid Search](docs/HYBRID_SEARCH.md) | BM25 + vector MMR + RRF fusion, cross-encoder reranking |
| [Data Pipeline](docs/DATA_PIPELINE.md) | Knowledge base, intelligent chunking, vectorstore updates |
| [API Reference](docs/API_REFERENCE.md) | All endpoints with request and response examples |
| [Configuration](docs/CONFIGURATION.md) | Environment variables, pydantic-settings, legacy config |
| [CI/CD Pipeline](docs/CI_CD.md) | GitFlow, security scanning, deployment, tagging |
| [Observability](docs/OBSERVABILITY.md) | Langfuse tracing, logging, rate limiting, feedback |
| [Embed Integration](docs/EMBED_INTEGRATION.md) | Embedding the chat widget on an external site |
| [Architecture Assessment](docs/ARCHITECTURE_ASSESSMENT.md) | Architecture review |
| [Release Notes](docs/RELEASE_NOTES.md) | Version history |

---

## License

Proprietary. This is a professional portfolio chatbot for [Bolaji BALOGOUN](https://bolablg.com).

---

*Built with LangGraph, Google Gemini, and hybrid RAG, gated by a golden-QA eval on every change.*
