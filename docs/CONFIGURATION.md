# Configuration

## Overview

The project supports two configuration approaches:
- **`app/settings.py`**: Modern, type-safe configuration via `pydantic-settings` (recommended)
- **`config.py`**: Legacy flat configuration (backward compatible)

Both load from `.env` files and environment variables.

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |

### Google Cloud (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | - | Google Cloud project ID |
| `GCP_SA_CREDENTIALS_PATH` | `_conf/ibola_agent_sa.json` | Service account key path |
| `GCHAT_WEBHOOK_URL` | - | Google Chat webhook for alerts |
| `REDIRECT_LOG_SHEET_ID` | - | Google Sheets ID for redirect logging |
| `GOOGLE_OAUTH_CREDENTIALS_PATH` | - | OAuth credentials path |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALLOWED_ORIGIN_REGEX` | `https://(.+\.)?bolablg\.com` | CORS origin regex |

### Session

| Variable | Default | Description |
|----------|---------|-------------|
| `SESSION_TIMEOUT_MINUTES` | `30` | Session timeout |
| `MAX_REDIRECT_COUNT` | `3` | Max redirects before chat ends |

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `gemini-2.5-flash` | Model name |
| `LLM_GUARDRAIL_TEMPERATURE` | `0.0` | Temperature for guardrail scoring |
| `LLM_GRADING_TEMPERATURE` | `0.0` | Temperature for document grading |
| `LLM_REWRITE_TEMPERATURE` | `0.3` | Temperature for query rewriting |
| `LLM_GENERATION_TEMPERATURE` | `0.7` | Temperature for answer generation |
| `LLM_GUARDRAIL_THRESHOLD` | `60` | Score threshold for relevance (0-100) |
| `LLM_MAX_RETRIEVAL_ATTEMPTS` | `2` | Max query rewrite + re-retrieve cycles |

### Search Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_USE_HYBRID` | `true` | Enable BM25 alongside vector search |
| `SEARCH_USE_RERANKER` | `true` | Allow cross-encoder reranking in the search pipeline |
| `SEARCH_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model (multilingual) |
| `ENABLE_CROSS_ENCODER_RERANKER` | `false` | Load the cross-encoder at startup (off by default: adds startup latency) |
| `SEARCH_RRF_RANK_CONSTANT` | `60` | RRF fusion constant |
| `SEARCH_VECTOR_TOP_K` | `8` | Results to return |

### Tracing (Langfuse)

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFUSE_ENABLED` | `false` | Enable Langfuse tracing |
| `LANGFUSE_PUBLIC_KEY` | - | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | - | Langfuse secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse server URL (also accepts the `LANGFUSE_BASE_URL` alias) |

### Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_RESPONSE_TTL` | `1800` | Response cache TTL in seconds |
| `CACHE_SESSION_TTL` | `3600` | Session cache TTL |
| `CACHE_REDIS_URL` | - | Redis URL (falls back to in-memory) |

## Configuration Hierarchy

```
pydantic-settings (app/settings.py)
  └── reads from: environment variables → .env file → defaults
  └── sub-settings: LLMSettings, SearchSettings, CacheSettings, TracingSettings

legacy config (config.py)
  └── reads from: .env file → os.getenv() → hardcoded defaults
```

## Files

| File | Purpose |
|------|---------|
| `app/settings.py` | `AppSettings` (pydantic-settings) with nested sub-settings |
| `config.py` | Legacy flat config (still used by existing agents) |
| `.env` | Environment variables (never committed) |
| `sample.env` | Template for `.env` |
