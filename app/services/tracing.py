"""
Langfuse tracing integration for the agentic RAG pipeline.

Wraps all tracing calls so failures never break the main application flow.
When LANGFUSE_ENABLED=false (default), all operations are silent no-ops.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger("ibola.tracing")

import hashlib as _hashlib  # noqa: E402
import re as _re  # noqa: E402

# PII patterns masked before any text reaches Langfuse (Codex #8: traces
# previously carried raw query/answer/IP/UA/referrer). EU-visitor safe.
_PII_PATTERNS = [
    (_re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (
        _re.compile(
            r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){3,5}\d{2,4}"
        ),
        "[phone]",
    ),
    (_re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "[ip]"),
    (_re.compile(r"https?://\S+"), "[url]"),
]


def mask_pii(text):
    """Redact emails, phone numbers, IPs, and URLs from free text."""
    if not text:
        return text
    out = str(text)
    for pattern, repl in _PII_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def hash_visitor(*parts):
    """Stable anonymous visitor id from non-PII request signals (hashed)."""
    raw = "|".join(str(p) for p in parts if p)
    if not raw:
        return None
    return "anon-" + _hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# Optional dependency
try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.info("langfuse not installed — tracing disabled")


class RAGTracer:
    """Instruments the agentic RAG pipeline with Langfuse spans."""

    def __init__(
        self,
        enabled: bool = False,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
    ):
        self.enabled = enabled and LANGFUSE_AVAILABLE
        self.client: Optional[Any] = None

        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("Langfuse tracing enabled at %s", host)
            except Exception as exc:
                logger.warning("Langfuse init failed (tracing disabled): %s", exc)
                self.enabled = False

    def create_trace(
        self,
        name: str,
        session_id: str = "",
        user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Create a new trace. Returns trace object or None."""
        if not self.enabled:
            return None
        try:
            return self.client.trace(
                name=name,
                session_id=session_id or None,
                user_id=user_id or None,
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.debug("Trace creation failed: %s", exc)
            return None

    @contextmanager
    def span(
        self,
        trace: Optional[Any],
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for a tracing span. No-op if trace is None."""
        if trace is None or not self.enabled:
            yield None
            return

        start = time.time()
        span_obj = None
        try:
            span_obj = trace.span(name=name, input=input_data or {})
            yield span_obj
        except Exception as exc:
            logger.debug("Span %s error: %s", name, exc)
            yield None
        finally:
            if span_obj is not None:
                try:
                    elapsed_ms = (time.time() - start) * 1000
                    span_obj.end(metadata={"duration_ms": round(elapsed_ms, 1)})
                except Exception:
                    pass

    def score(
        self,
        trace: Optional[Any],
        name: str,
        value: float,
        comment: str = "",
    ):
        """Add a score (e.g., user feedback) to a trace."""
        if trace is None or not self.enabled:
            return
        try:
            trace.score(name=name, value=value, comment=comment)
        except Exception as exc:
            logger.debug("Score recording failed: %s", exc)

    def record_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        payload: Optional[Dict[str, Any]] = None,
        steps: Optional[Any] = None,
        user_id: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> Optional[str]:
        """Record one full agentic turn as a trace with per-node child spans.

        Uses the Langfuse v3+ OTEL API (start_span / update_trace). The
        required payload keys make every answer reproducible from its trace:
        raw and rewritten query, retrieved chunk IDs with scores, the final
        context order, the answer, and latency. Query and answer are PII-masked
        before they leave the process (Codex #8). ``user_id`` is an anonymous
        hashed visitor id; ``tags`` carry lang/category/endpoint/release/prompt
        versions. Returns the trace_id so feedback can link back to this turn.
        """
        if not self.enabled or self.client is None:
            return None
        try:
            from langfuse import propagate_attributes

            masked_query = mask_pii(query)
            masked_answer = mask_pii(answer)

            # v4 API: session_id/user_id/tags are trace-level attributes set
            # via propagate_attributes; the root observation's I/O becomes the
            # trace I/O. Each graph node is a child span.
            with propagate_attributes(
                session_id=session_id or None,
                user_id=user_id or None,
                tags=tags or None,
            ):
                with self.client.start_as_current_observation(
                    name="chat-response",
                    as_type="span",
                    input=masked_query,
                    metadata=payload or {},
                ) as root:
                    trace_id = root.trace_id
                    for step in steps or []:
                        node = getattr(step, "node", None) or (
                            step.get("node") if isinstance(step, dict) else "step"
                        )
                        action = getattr(step, "action", None) or (
                            step.get("action") if isinstance(step, dict) else ""
                        )
                        detail = getattr(step, "detail", None) or (
                            step.get("detail") if isinstance(step, dict) else ""
                        )
                        try:
                            child = root.start_observation(
                                name=str(node),
                                as_type="span",
                                input={"action": action},
                                metadata={"detail": detail},
                            )
                            child.end()
                        except Exception:
                            continue
                    root.update(output=masked_answer)
                    root.set_trace_io(input=masked_query, output=masked_answer)
            return trace_id
        except Exception as exc:
            logger.debug("record_turn failed: %s", exc)
            return None

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value,
        comment: str = "",
        data_type: str = "NUMERIC",
    ):
        """Attach a typed score to an existing trace by id.

        ``data_type`` is passed explicitly: a BOOLEAN score with value 1 is
        otherwise inferred NUMERIC (skill user-feedback Common Mistakes).
        CATEGORICAL scores take a string value.
        """
        if not self.enabled or self.client is None or not trace_id:
            return
        try:
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type=data_type,
                comment=comment or None,
            )
        except Exception as exc:
            logger.debug("score_trace failed: %s", exc)

    def score_session(
        self,
        session_id: str,
        name: str,
        value,
        comment: str = "",
        data_type: str = "NUMERIC",
    ):
        """Attach a SESSION-level score (e.g. session-csat), not a trace score
        (Codex #5): CSAT is about the conversation, not one turn."""
        if not self.enabled or self.client is None or not session_id:
            return
        try:
            self.client.create_score(
                session_id=session_id,
                name=name,
                value=value,
                data_type=data_type,
                comment=comment or None,
            )
        except Exception as exc:
            logger.debug("score_session failed: %s", exc)

    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception:
                pass


# Module-level singleton — configured from settings
_tracer: Optional[RAGTracer] = None


def get_tracer() -> RAGTracer:
    """Return the global tracer instance (lazy init)."""
    global _tracer
    if _tracer is None:
        try:
            from app.settings import get_settings

            settings = get_settings()
            _tracer = RAGTracer(
                enabled=settings.tracing.is_active,
                public_key=settings.tracing.public_key,
                secret_key=settings.tracing.secret_key,
                host=settings.tracing.host,
            )
        except Exception:
            _tracer = RAGTracer(enabled=False)
    return _tracer
