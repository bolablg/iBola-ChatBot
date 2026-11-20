"""Real truncation envelopes with explicitly synthetic SDK recovery controls."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.graph import nodes
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import (
    BatchGradeResult,
    CondensedQuery,
    GeneratedAnswer,
    GroundingVerdict,
    GuardrailScoring,
)

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/answer_delivery_failures.json").read_text()
)


def fake_llm(responses, configured=750):
    llm = object.__new__(nodes._SyncGeminiLLM)
    llm._model = "gemini-3.5-flash-lite"
    llm._temperature = 0
    llm._max_output_tokens = configured
    llm._thinking_budget = None
    llm._thinking_level = "low"
    generate = Mock(side_effect=responses)
    llm._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    return llm, generate


@pytest.mark.parametrize("capture", CAPTURE["truncations"], ids=lambda c: c["case_id"])
@pytest.mark.parametrize("recover", [True, False])
def test_actual_invalid_envelope_recovers_boundedly_or_raises(capture, recover):
    assert capture["max_output_tokens"] == 750
    assert "MAX_TOKENS" in capture["finish_reasons"][0]
    assert capture["usage"]["thoughts_token_count"] > 680
    with pytest.raises(ValueError):
        json.loads(capture["response_text"])
    broken = SimpleNamespace(text=capture["response_text"], parsed=None)
    parsed = GeneratedAnswer(answer="Synthetic complete recovery.", confidence=0.8)
    llm, generate = fake_llm(
        [broken, SimpleNamespace(parsed=parsed, text="") if recover else broken]
    )
    if recover:
        assert (
            nodes._StructuredOutput(llm, GeneratedAnswer).invoke(["Fixture"]) is parsed
        )
    else:
        with pytest.raises(ValueError):
            nodes._StructuredOutput(llm, GeneratedAnswer).invoke(["Fixture"])
    assert generate.call_count == 2
    assert all(
        c.kwargs["config"].max_output_tokens == 2048 for c in generate.call_args_list
    )
    assert llm._max_output_tokens == 750


@pytest.mark.parametrize("configured", [750, 8192])
@pytest.mark.parametrize(
    "schema,parsed,floor",
    [
        (
            GeneratedAnswer,
            GeneratedAnswer(answer="Synthetic answer.", confidence=1),
            2048,
        ),
        (GroundingVerdict, GroundingVerdict(is_grounded=True), 4096),
        (
            GuardrailScoring,
            GuardrailScoring(score=100, category="professional", reasoning="Fixture"),
            0,
        ),
        (BatchGradeResult, BatchGradeResult(relevant_indices=[0]), 0),
        (
            CondensedQuery,
            CondensedQuery(standalone_query="Fixture", retrieval_query_en="Fixture"),
            0,
        ),
    ],
)
def test_schema_headroom_preserves_routing_and_configured_limits(
    schema, parsed, floor, configured
):
    llm, generate = fake_llm([SimpleNamespace(parsed=parsed, text="")], configured)
    ideal = nodes._StructuredOutput(llm, schema).invoke(["Fixture"])
    assert ideal is parsed
    conf = generate.call_args.kwargs["config"]
    assert conf.max_output_tokens == max(configured, floor)
    assert str(conf.thinking_config.thinking_level).lower().endswith("low")
    assert conf.thinking_config.thinking_budget is None
    assert llm._max_output_tokens == configured


def state():
    return {
        "query": "Which languages does Bolaji speak?",
        "answer": "A synthetic incorrect draft.",
        "agent_type": "professional",
        "user_language": "en",
        "context_documents": [
            Document(
                page_content="Bolaji speaks French and English.",
                metadata={"source": "synthetic-scope.txt"},
            )
        ],
    }


def test_repair_truncation_never_adds_a_fourth_sdk_call(monkeypatch):
    broken = SimpleNamespace(
        text=CAPTURE["truncations"][0]["response_text"], parsed=None
    )
    llm, generate = fake_llm(
        [
            SimpleNamespace(
                parsed={"is_grounded": False, "corrected_answer": ""}, text=""
            ),
            broken,
            broken,
        ]
    )
    monkeypatch.setattr(nodes, "_get_llm", lambda **kwargs: llm)
    result = nodes.verify_grounding_node(state())
    assert generate.call_count == 3
    assert not result["grounding_checked"]
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    assert nodes._grounding_provider_calls.get() is None


@pytest.mark.parametrize(
    "grounded,on_topic", [(True, True), (False, True), (True, False)]
)
def test_larger_repair_still_requires_independent_final_acceptance(
    monkeypatch, grounded, on_topic
):
    repaired = "Bolaji speaks French and English."
    llm, generate = fake_llm(
        [
            SimpleNamespace(
                parsed={"is_grounded": False, "corrected_answer": ""}, text=""
            ),
            SimpleNamespace(parsed={"answer": repaired, "confidence": 0.8}, text=""),
            SimpleNamespace(
                parsed={"is_grounded": grounded, "addresses_question": on_topic},
                text="",
            ),
        ]
    )
    monkeypatch.setattr(nodes, "_get_llm", lambda **kwargs: llm)
    result = nodes.verify_grounding_node(state())
    assert generate.call_count == 3
    assert [c.kwargs["config"].max_output_tokens for c in generate.call_args_list] == [
        4096,
        2048,
        4096,
    ]
    assert result["grounding_checked"] is (grounded and on_topic)
    assert result["answer"] == (
        repaired if grounded and on_topic else INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    )
