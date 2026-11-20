# API Reference

Base URL: `https://chat.bolablg.com` (production) or `http://localhost:8000` (local)

Interactive docs: `/docs` (Swagger UI) | `/redoc` (ReDoc)

---

## Chat Endpoints

### `POST /ask-agentic`

Full agentic RAG pipeline (LangGraph). Recommended for production use.

Send `X-Ibola-No-Trace: 1` to disable optional Langfuse turn recording for the current request. The trimmed header must equal `1`; other values preserve normal tracing. This preference also works with `/chat` and `/ask`, including SSE, and is not stored in sessions or caches. Evaluation clients send it on prefix turns, final turns and latency probes.

Send `X-Ibola-No-Notifications: 1` on the same chat endpoints to suppress redirect alerts, redirect Sheets logging and lead-message webhooks for the current request. These requests bypass response caching. Suppressed message completion states that no message was sent. Evaluation clients send both opt-out headers on every turn and probe. The flags are independent and do not disable Gemini processing or ordinary application logs.

The tracing opt-out does not disable request logs, Gemini processing, contact behavior or the separate `/feedback` path.

**Request:**
```json
{
  "user_input": "What is Bolaji's experience with BigQuery?",
  "session_id": "abc-123",
  "user_language": "en",
  "stream": false
}
```

**Response:**
```json
{
  "answer": "Bolaji has extensive experience with BigQuery...",
  "actions": [],
  "agent_type": "professional",
  "confidence": 0.85,
  "language": "en",
  "redirect_count": 0,
  "session_id": "abc-123",
  "response_time": 3.241,
  "cached": false,
  "should_end_chat": false
}
```

**SSE Streaming** (`stream: true`): Returns `text/event-stream` with events:
- `progress` — `{"status": "processing", "node": "guardrail", "action": "scored"}`
- `token` — `{"token": "chunk of text"}`
- `done` — `{"status": "done", "agent_type": "professional", "confidence": 0.85}`

Concatenate `token` fragments directly to preserve the answer's Markdown spacing and line breaks. The `done` event also includes `should_end_chat`.

### `POST /ask`

Simple RAG — direct retrieval + generation without agent routing. Faster (2-5s).

**Request/Response**: Same schema as `/ask-agentic` (without `stream`).

### `POST /chat`

Legacy endpoint using orchestrator-based routing. Backward compatible.

**Request:**
```json
{
  "user_input": "Tell me about Bolaji",
  "session_id": "abc-123",
  "user_language": "en"
}
```

### `POST /welcome`

Localized welcome messages based on browser language.

**Request:**
```json
{
  "session_id": "abc-123",
  "browser_language": "fr-FR"
}
```

---

## Feedback

### `POST /feedback`

Submit user quality ratings. Feeds into Langfuse when enabled.

**Request:**
```json
{
  "session_id": "abc-123",
  "message_index": 0,
  "score": 0.8,
  "comment": "Helpful answer"
}
```

---

## Session Management

### `GET /session/{session_id}/stats`

Returns redirect count, last agent, language, and active status.

### `DELETE /session/{session_id}`

Resets session data.

---

## Monitoring

### `GET /health`

System health check: CPU, memory, service status, cache stats, rate limit stats.

### `GET /performance/metrics`

Comprehensive metrics: system resources + application stats.

### `GET /cache/stats` | `GET /rate-limit/stats`

Cache and rate limiter statistics.

### `POST /cache/clear`

Admin endpoint to clear all caches.

---

## Alerts

### `POST /contact-alert`

Forward contact requests (booking/email) to Google Chat webhook.

**Request:**
```json
{
  "contact_type": "booking",
  "session_id": "abc-123",
  "chat_history": [["user msg", "bot reply"]],
  "timestamp": "2026-03-22T10:00:00"
}
```
