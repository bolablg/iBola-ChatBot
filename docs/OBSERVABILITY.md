# Observability & Monitoring

## Tracing (Langfuse)

### Setup

Set environment variables to enable:
```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### How It Works

`RAGTracer` in `app/services/tracing.py` provides:
- **Traces**: One per request, linked by `session_id`
- **Spans**: One per pipeline stage (guardrail, retrieve, grade, generate)
- **Scores**: User feedback ratings via `POST /feedback`

All tracing is wrapped so **failures never break the main flow**. When `LANGFUSE_ENABLED=false` (default), all operations are silent no-ops.

### Feedback Loop

`POST /feedback` accepts a score (0.0-1.0) and optional comment. This feeds into Langfuse for quality monitoring dashboards.

## Google Cloud Logging

`app/services/logging_service.py` sends structured JSON logs to Google Cloud Logging:
- Request/response metadata
- Session ID, language, agent type
- Response time
- Exception details with stack traces

Filters by severity: WARNING, ERROR, DEBUG only (to control costs).

## Health Monitoring

### `GET /health`

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "system": {
    "memory_usage": "45.2%",
    "cpu_usage": "12.3%",
    "memory_available": "2048MB"
  },
  "services": {
    "orchestrator": "healthy",
    "language_service": "healthy",
    "cache_service": "healthy",
    "rate_limiter": "healthy"
  }
}
```

### `GET /performance/metrics`

System-level metrics: CPU, memory, disk, plus application-level: cache stats, rate limit stats, active sessions.

## Rate Limiting

Sliding window per client IP (`app/services/rate_limiting.py`):

| Endpoint | Per Minute | Per Hour |
|----------|-----------|----------|
| `/chat` | 30 | 500 |
| `/welcome` | 10 | 100 |
| `/health` | 60 | 1000 |
| `/session` | 20 | 200 |
| Global | 60 | 1000 |

Burst protection: max 10 requests in rapid succession.

## Redirect Analytics

Off-topic queries are logged to Google Sheets via `app/services/google_sheets_logger.py`:
- Timestamp, session ID, user input
- Confidence score, agent type, redirect reason
- Device type, browser language, referrer
- Cache hit status, response time

Google Chat alerts are sent when a session reaches 3+ redirects.

## Files

| File | Purpose |
|------|---------|
| `app/services/tracing.py` | `RAGTracer` — Langfuse integration |
| `app/routes/feedback.py` | `POST /feedback` endpoint |
| `app/services/logging_service.py` | Google Cloud Logging |
| `app/services/rate_limiting.py` | Sliding window rate limiter |
| `app/services/google_sheets_logger.py` | Redirect event logging |
| `app/services/google_chat_alert.py` | Google Chat webhook alerts |
