"""Regression tests for provider retries, cache fallback, and eval diagnostics."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.graph.nodes import _invoke_with_retry, _StructuredOutput
from app.graph.prompts import GENERATE_SYSTEM_PROMPTS, PROMPT_VERSIONS
from app.graph.state import AgentCategory, GeneratedAnswer, GroundingVerdict
from scripts.run_eval import _extract_workflow_error, _extract_workflow_warning


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


def test_structured_output_recovers_from_truncated_json():
    class FakeModels:
        def __init__(self):
            self.responses = [
                SimpleNamespace(text='{"answer":"Bolaji was Data Director'),
                SimpleNamespace(
                    text='{"answer":"Bolaji was Data Director at Gozem.",'
                    '"confidence":0.9}'
                ),
            ]
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            return self.responses.pop(0)

    models = FakeModels()
    llm = SimpleNamespace(
        _client=SimpleNamespace(models=models),
        _model="test-model",
        _temperature=0.0,
        _max_output_tokens=750,
        _thinking_budget=0,
    )

    prompts = []
    original_generate_content = models.generate_content

    def generate_content(**kwargs):
        prompts.append(kwargs["contents"])
        return original_generate_content(**kwargs)

    models.generate_content = generate_content
    result = _StructuredOutput(llm, GeneratedAnswer).invoke(["Return JSON"])

    assert result == GeneratedAnswer(
        answer="Bolaji was Data Director at Gozem.", confidence=0.9
    )
    assert models.calls == 2
    assert "previous structured response was invalid" in prompts[1]


def test_structured_output_recovers_grounding_verdict():
    class FakeModels:
        def __init__(self):
            self.responses = [
                SimpleNamespace(
                    text='{"is_grounded":false,"addresses_question":true,'
                    '"unsupported_claims":["unfinished'
                ),
                SimpleNamespace(
                    text='{"is_grounded":true,"addresses_question":true,'
                    '"unsupported_claims":[],"corrected_answer":""}'
                ),
            ]
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            return self.responses.pop(0)

    models = FakeModels()
    llm = SimpleNamespace(
        _client=SimpleNamespace(models=models),
        _model="test-model",
        _temperature=0.0,
        _max_output_tokens=750,
        _thinking_budget=0,
    )

    result = _StructuredOutput(llm, GroundingVerdict).invoke(["Return JSON"])

    assert result == GroundingVerdict(is_grounded=True)
    assert models.calls == 2


def test_structured_output_prefers_sdk_parsed_model():
    parsed = GeneratedAnswer(answer="Parsed by the SDK", confidence=0.9)

    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(text='{"answer":"truncated', parsed=parsed)

    models = FakeModels()
    llm = SimpleNamespace(
        _client=SimpleNamespace(models=models),
        _model="test-model",
        _temperature=0.0,
        _max_output_tokens=750,
        _thinking_budget=0,
    )

    result = _StructuredOutput(llm, GeneratedAnswer).invoke(["Return JSON"])

    assert result is parsed
    assert models.calls == 1


def test_eval_surfaces_workflow_diagnostics():
    result = {"workflow_errors": ["generate: provider unavailable"]}

    assert _extract_workflow_error(result) == "generate: provider unavailable"


def test_eval_surfaces_nonfatal_workflow_warnings():
    result = {"workflow_warnings": ["verify_grounding: unavailable"]}

    assert _extract_workflow_warning(result) == "verify_grounding: unavailable"


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
