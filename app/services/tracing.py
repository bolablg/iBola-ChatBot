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
    ) -> Optional[str]:
        """Record one full agentic turn as a trace with per-node child spans.

        Uses the Langfuse v3+ OTEL API (start_span / update_trace). The
        required payload keys make every answer reproducible from its trace:
        raw and rewritten query, retrieved chunk IDs with scores, the final
        context order, the answer, and latency. Returns the trace_id so the
        response (and later user feedback) can link back to it.
        """
        if not self.enabled or self.client is None:
            return None
        try:
            root = self.client.start_span(
                name="agentic_turn",
                input={"query": query},
                metadata=payload or {},
            )
            try:
                root.update_trace(
                    session_id=session_id or None,
                    input=query,
                    output=answer,
                )
            except Exception:
                pass

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
                    child = root.start_span(
                        name=str(node),
                        input={"action": action},
                        metadata={"detail": detail},
                    )
                    child.end()
                except Exception:
                    continue

            root.update(output={"answer": answer})
            trace_id = getattr(root, "trace_id", None)
            root.end()
            return trace_id
        except Exception as exc:
            logger.debug("record_turn failed: %s", exc)
            return None

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str = "",
    ):
        """Attach a score to an existing trace by id (links user feedback to
        the originating turn instead of creating a disconnected trace)."""
        if not self.enabled or self.client is None or not trace_id:
            return
        try:
            self.client.create_score(
                trace_id=trace_id, name=name, value=value, comment=comment or None
            )
        except Exception as exc:
            logger.debug("score_trace failed: %s", exc)

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
                enabled=settings.tracing.enabled,
                public_key=settings.tracing.public_key,
                secret_key=settings.tracing.secret_key,
                host=settings.tracing.host,
            )
        except Exception:
            _tracer = RAGTracer(enabled=False)
    return _tracer
