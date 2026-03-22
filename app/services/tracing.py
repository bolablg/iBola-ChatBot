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
                    span_obj.end(
                        metadata={"duration_ms": round(elapsed_ms, 1)}
                    )
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
