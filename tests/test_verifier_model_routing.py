"""Dedicated verification routing without weakening source or provider-call gates."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.graph import nodes
from app.services.evidence_policy import (
    constraint_failures,
    preserve_requested_evidence,
)
from app.settings import LLMSettings


@pytest.mark.parametrize(
    "final_grounded,final_on_topic", [(True, True), (False, True), (True, False)]
)
def test_actual_sdk_routes_verify_repair_and_final_verification(
    monkeypatch, final_grounded, final_on_topic
):
    case = next(
        c
        for c in json.loads(
            Path("tests/fixtures/live_scope_relationships.json").read_text()
        )
        if "adi" in c["id"]
    )
    good = "Bolaji a été la première embauche data en 2019. L'équipe initiale a atteint 4 personnes. L'équipe est passée de 4 à plus de 14 personnes durant Head of Data, octobre 2022 - mars 2025. Elle a soutenu cinq marchés opérationnels africains. Singapour était uniquement le siège de l'entreprise."
    replies = [
        {"is_grounded": False, "corrected_answer": ""},
        {"answer": good, "confidence": 0.8},
        {"is_grounded": final_grounded, "addresses_question": final_on_topic},
    ]
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(parsed=replies.pop(0), text="")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: client)
    monkeypatch.setattr(nodes, "_llms", {})
    result = nodes.verify_grounding_node(
        {
            "query": case["question"],
            "answer": case["answer"],
            "context_documents": [Document(**d) for d in case["documents"]],
            "user_language": "fr",
        }
    )
    assert len(calls) == 3
    assert [c["model"] for c in calls] == [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
    ]
    for call in (calls[0], calls[2]):
        conf = call["config"]
        assert conf.max_output_tokens == 4096
        assert str(conf.thinking_config.thinking_level).lower().endswith("low")
        assert conf.thinking_config.thinking_budget is None
    assert calls[1]["config"].thinking_config.thinking_budget is None
    assert (
        str(calls[1]["config"].thinking_config.thinking_level).lower().endswith("low")
    )
    assert all(call["config"].temperature is None for call in calls)
    assert any(
        good in part.text for content in calls[2]["contents"] for part in content.parts
    )
    assert result["grounding_checked"] is (final_grounded and final_on_topic)
    if result["grounding_checked"]:
        assert result["answer"] == good
    else:
        assert result["answer"] != good


def test_model_cache_and_rollback_keep_effective_thinking_config_separate(monkeypatch):
    monkeypatch.setattr("google.genai.Client", Mock())
    monkeypatch.setattr(nodes, "_llms", {})
    default = nodes._get_llm(temperature=0, model="gemini-2.5-flash")
    verifier = nodes._get_llm(
        temperature=0, model="gemini-3.5-flash-lite", thinking_level="low"
    )
    assert default is not verifier
    assert (
        nodes._get_llm(
            temperature=0, model="gemini-3.5-flash-lite", thinking_level="low"
        )
        is verifier
    )
    rollback = LLMSettings(verifier_model_name="gemini-2.5-flash").verifier_parameters()
    assert rollback == {"model": "gemini-2.5-flash", "thinking_budget": 0}
    old = nodes._get_llm(temperature=0, **rollback)
    assert old._thinking_config() == {"thinking_budget": 0}
    assert verifier._thinking_config() == {"thinking_level": "low"}
    with pytest.raises(ValueError, match="not both"):
        nodes._get_llm(
            model="gemini-3.5-flash-lite", thinking_level="low", thinking_budget=512
        )
    with pytest.raises(ValueError, match="not both"):
        nodes._SyncGeminiLLM(
            model="test",
            api_key="test",
            temperature=0,
            max_output_tokens=750,
            thinking_level="low",
            thinking_budget=512,
        )


@pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", "gemini-2.5-flash"])
def test_provenance_records_effective_verifier_settings(monkeypatch, model):
    from app.services import provenance

    settings = LLMSettings(verifier_model_name=model)
    monkeypatch.setattr(
        "app.settings.get_settings", lambda: SimpleNamespace(llm=settings)
    )
    result = provenance.runtime_provenance()
    assert result["verifier_model"] == model
    assert result["verifier_thinking_level"] == ("low" if "3.5" in model else None)
    assert result["verifier_thinking_budget"] == (0 if "2.5" in model else None)


def test_strategy_progress_requires_exact_section_and_does_not_cleanse_completion():
    text = "Result: Turned fragmented AI experimentation into a structured enterprise roadmap, identifying 49 opportunities and advancing 6 high-priority initiatives into implementation."
    doc = Document(
        page_content=text,
        metadata={
            "source": "90_website_canon_llms_full.txt",
            "section_header": "AI strategy & investment portfolio",
        },
    )
    q = "For the AI strategy how did he prioritize the six implementations? What return is documented?"
    draft = "He used a scoring framework. Measured return is not documented."
    result, ids = preserve_requested_evidence(draft, [doc], q, "English")
    assert ids == ["requested_AI_implementation_progress_missing"]
    assert text in result
    assert not constraint_failures(result, [doc], q)
    for bad in [
        Document(
            page_content=text,
            metadata={**doc.metadata, "section_header": "Another project"},
        ),
        Document(
            page_content=text, metadata={**doc.metadata, "source": "different.txt"}
        ),
    ]:
        assert preserve_requested_evidence(draft, [bad], q, "English") == (
            "He used a scoring framework. The retrieved evidence does not establish measured return.",
            [],
        )
    assert preserve_requested_evidence(
        draft, [doc], "What languages does he speak?", "English"
    ) == (
        "He used a scoring framework. The retrieved evidence does not establish measured return.",
        [],
    )
    wrong = "He completed six AI use cases."
    result, _ = preserve_requested_evidence(wrong, [doc], q, "English")
    assert "implementation_status" in constraint_failures(result, [doc], q)


def test_reference_history_cannot_expand_the_current_answer_scope():
    state = {
        "original_query": "Where is the public article for this framework?",
        "query": "Where is the public article for the Dataform framework?",
        "chat_history": [
            ("How does this compare with dbt?", "Untrusted previous answer.")
        ],
    }
    assert "dbt" not in nodes._verification_scope(state)
    assert "public article" in nodes._verification_scope(state)


@pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", "gemini-2.5-flash"])
def test_all_graph_factory_payloads_use_compatible_effective_config(monkeypatch, model):
    from app.graph.state import GeneratedAnswer
    from app.settings import get_settings

    settings = get_settings().model_copy(update={"llm": LLMSettings(model_name=model)})
    monkeypatch.setattr("app.settings.get_settings", lambda: settings)
    monkeypatch.setattr(nodes, "_llms", {})
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            text="Supported reply.",
            parsed=GeneratedAnswer(answer="Supported reply.", confidence=1.0),
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: client)
    ordinary = nodes._get_llm(temperature=0.3)
    ordinary.invoke(["A routing or rewrite input."])
    generator = nodes._get_llm(
        temperature=0.7, **settings.llm.thinking_parameters(generation=True)
    )
    generator.with_structured_output(GeneratedAnswer).invoke(["A generation input."])
    assert all(c["model"] == model for c in calls)
    if model == "gemini-3.5-flash-lite":
        assert calls[0]["config"]["thinking_config"] == {"thinking_level": "low"}
        assert "temperature" not in calls[0]["config"]
        assert calls[1]["config"].temperature is None
        assert calls[1]["config"].thinking_config.thinking_budget is None
        assert (
            str(calls[1]["config"].thinking_config.thinking_level)
            .lower()
            .endswith("low")
        )
        assert nodes._get_llm(temperature=0.9) is nodes._get_llm(temperature=0.0)
        with pytest.raises(ValueError, match="thinking level"):
            nodes._get_llm(thinking_budget=0)
    else:
        assert calls[0]["config"]["thinking_config"] == {"thinking_budget": 0}
        assert calls[0]["config"]["temperature"] == 0.3
        assert calls[1]["config"].thinking_config.thinking_budget == 256
        assert calls[1]["config"].thinking_config.thinking_level is None
        assert calls[1]["config"].temperature == 0.7
        assert nodes._get_llm(temperature=0.9) is not nodes._get_llm(temperature=0.0)


def test_modern_constructor_rejects_legacy_budgets_before_any_sdk_call(monkeypatch):
    client = Mock()
    monkeypatch.setattr("google.genai.Client", client)
    with pytest.raises(ValueError, match="thinking level"):
        nodes._SyncGeminiLLM(
            model="gemini-3.5-flash-lite",
            api_key="test",
            temperature=0,
            max_output_tokens=750,
            thinking_budget=0,
        )
    client.assert_not_called()
