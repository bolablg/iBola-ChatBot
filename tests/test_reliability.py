"""Regression tests for provider retries, cache fallback, and eval diagnostics."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from app.graph.nodes import _invoke_with_retry
from app.graph.prompts import GENERATE_SYSTEM_PROMPTS, PROMPT_VERSIONS
from app.graph.state import AgentCategory
from scripts.run_eval import _extract_workflow_error


def test_transient_gemini_failure_is_retried_once():
    call = Mock(side_effect=[RuntimeError("503 UNAVAILABLE"), "ok"])

    with patch("app.graph.nodes.time.sleep") as sleep:
        assert _invoke_with_retry(call, "test generation") == "ok"

    assert call.call_count == 2
    sleep.assert_called_once_with(0.5)


def test_non_transient_gemini_failure_is_not_retried():
    call = Mock(side_effect=ValueError("invalid JSON"))

    with patch("app.graph.nodes.time.sleep") as sleep:
        with pytest.raises(ValueError, match="invalid JSON"):
            _invoke_with_retry(call, "test generation")

    assert call.call_count == 1
    sleep.assert_not_called()


def test_eval_surfaces_workflow_diagnostics():
    result = {"workflow_errors": ["generate: provider unavailable"]}

    assert _extract_workflow_error(result) == "generate: provider unavailable"


def test_cache_service_without_cachetools_is_safe():
    from app.services.cache_service import CacheService

    with patch("app.services.cache_service.CACHE_AVAILABLE", False):
        service = CacheService()

    assert (
        asyncio.run(service.get_localized_content("welcome:en-US", "welcome")) is None
    )
    assert service.get_cache_stats()["status"] == "disabled"
    service.clear_all_caches()


def test_professional_prompt_requires_complete_multipart_answers():
    prompt = GENERATE_SYSTEM_PROMPTS[AgentCategory.PROFESSIONAL]

    assert PROMPT_VERSIONS["generate_professional"] == "3.2"
    assert "include every requested number, title, date, and category" in prompt
