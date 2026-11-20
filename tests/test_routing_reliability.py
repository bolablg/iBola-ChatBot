"""Frozen failures must improve without loosening the accepted assertions."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph import nodes
from app.graph.prompts import GUARDRAIL_PROMPT
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    _record_limit_subject,
    constraint_failures,
    describe_constraints,
    documentation_limits,
    prepare_evidence,
    retained_facet_indices,
    supplement_facets,
)
from app.settings import LLMSettings
from scripts.run_eval import check_facts, load_golden

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/routing_reliability_failures.json").read_text()
)
CASES = {c["id"]: c for c in FIXTURE["cases"]}
SPARK = CASES["recruiter-sde-03-t2"]
TEAM = CASES["recruiter-adi-05-t5"]
HUB = CASES["skill-004-fr"]
ECHO = "superseded_growth_interval_echo"
ADOPTION = "platform_definition_adoption_scope_missing"


def docs(case):
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in case["evidence"]
        ],
        case["question"],
    )


def test_exact_documentation_limit_is_recognized_without_rewriting_delivered_text():
    assert not documentation_limits(SPARK["answer"], ["Spark Structured Streaming"])
    assert not constraint_failures(SPARK["answer"], docs(SPARK), SPARK["question"])
    row = next(r for r in load_golden() if r["id"] == SPARK["id"])
    assert check_facts(SPARK["answer"], row, result=SPARK)[0]


@pytest.mark.parametrize("prefix", ["", "Additionally, "])
def test_existential_limit_extracts_only_the_bound_tool(prefix):
    text = (
        prefix
        + "there is no documented personal production experience or named project proving the deployment of Spark Structured Streaming"
    )
    assert (
        _record_limit_subject(text, ["Spark Structured Streaming"]).strip()
        == "Spark Structured Streaming"
    )
    assert not documentation_limits(text, ["Spark Structured Streaming"])
    assert documentation_limits(text, ["Kafka"])


@pytest.mark.parametrize(
    "text",
    [
        "If there is no documented personal production experience or named project proving the deployment of Spark Structured Streaming",
        "Is there no documented personal production experience or named project proving the deployment of Spark Structured Streaming?",
        "There is no personal production experience or named project proving the deployment of Spark Structured Streaming",
        "There is no documented personal production experience or named project proving the deployment of Spark Structured Streaming and he operated it",
        "There is no documented personal production experience or named project proving the deployment of Spark Structured Streaming if needed",
        "There is no documented personal production experience or named project proving the deployment of Spark Structured Streaming unless requested",
        "There is no documented personal production experience or named project proving the deployment of Spark Structured Streaming and he runs it in production",
        "There is no documented personal production experience or named project proving the deployment of Spark Structured Streaming and he owns it",
    ],
)
def test_conditional_unqualified_or_finite_claims_are_not_nominal_record_limits(text):
    assert _record_limit_subject(text, ["Spark Structured Streaming"]) is None


def test_existential_subjects_are_literal_and_finite_suffix_cannot_hide_from_both_consumers():
    nominal = "There is no documented personal production experience or named project proving the deployment of Spark Structured Streaming"
    assert _record_limit_subject(nominal, ["Spark.*"]) is None
    bad = nominal + " and he runs it in production"
    assert documentation_limits(bad, ["Spark Structured Streaming"])
    assert constraint_failures(bad, docs(SPARK), SPARK["question"] + " In production?")


def test_valid_gap_cannot_hide_adjacent_affirmative_production_claim():
    answer = (
        SPARK["answer"]
        + " Bolaji personally operated Spark Structured Streaming in production."
    )
    assert constraint_failures(
        answer, docs(SPARK), SPARK["question"] + " In production?"
    )


def test_absolute_golden_interval_prohibition_stays_unchanged_for_the_exact_denial():
    row = next(r for r in load_golden() if r["id"] == TEAM["id"])
    assert not check_facts(TEAM["answer"], row, result=TEAM)[0]
    assert ECHO in constraint_failures(TEAM["answer"], docs(TEAM), TEAM["question"])
    assert "without echoing" in describe_constraints([ECHO])
    assert ECHO not in constraint_failures(TEAM["answer"], [], TEAM["question"])
    assert ECHO not in constraint_failures(
        TEAM["answer"], docs(TEAM), "Which programming language?"
    )


@pytest.mark.parametrize(
    "case,good,rule",
    [
        (
            TEAM,
            "Bolaji a été la première embauche data en 2019. L'équipe initiale a atteint 4 personnes pendant Global Data Analyst de novembre 2019 à septembre 2022. Elle est ensuite passée de 4 à plus de 14 professionnels durant Head of Data, octobre 2022 à mars 2025. Elle soutenait cinq marchés opérationnels africains. Singapour était uniquement le siège social.",
            ECHO,
        ),
        (
            HUB,
            "Le Gozem Data Hub est la plateforme centrale de données et d'IA que Bolaji a architecturée. Elle servait plus de 700 personnes, dont des employés, agents terrain et partenaires, dans cinq pays africains à la fin de son mandat.",
            ADOPTION,
        ),
    ],
)
def test_exact_missing_or_editorial_detail_requires_bounded_repair_and_final_verification(
    case, good, rule
):
    assert rule in constraint_failures(case["answer"], docs(case), case["question"])
    assert rule not in constraint_failures(good, docs(case), case["question"])
    verifier = Mock()
    verifier.invoke.return_value = GroundingVerdict(is_grounded=True)
    repair = Mock()
    repair.invoke.return_value = GeneratedAnswer(answer=good, confidence=0.9)
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = nodes.verify_grounding_node(
            {
                "query": case["question"],
                "original_query": case["question"],
                "answer": case["answer"],
                "context_documents": docs(case),
                "user_language": case["lang"],
                "agent_type": "professional",
            }
        )
    assert result["grounding_checked"] and result["answer"] == good
    assert verifier.invoke.call_count == 2 and repair.invoke.call_count == 1
    assert (
        verifier.invoke.call_args.args[0][-1].content.rsplit("Answer to verify:\n", 1)[
            1
        ]
        == good
    )


@pytest.mark.parametrize(
    "bad",
    [
        "The Hub served 700 people.",
        "The Hub served over 700 employees.",
        "The Hub had 700+ active people.",
        "The Hub served over 700 internal people, all employees.",
        "The Hub served over 700 people, all employees.",
        "The Hub served over 700 people, all are employees.",
        "The Hub served over 700 people, all of whom were employees.",
        "The Hub served over 700 people, employees only.",
        "Le Hub servait plus de 700 personnes, uniquement des employés.",
    ],
)
def test_platform_bound_population_and_activity_remain_distinct(bad):
    assert ADOPTION in constraint_failures(bad, docs(HUB), HUB["question"])
    assert ADOPTION not in constraint_failures(bad, [], HUB["question"])
    assert ADOPTION not in constraint_failures(
        bad, docs(HUB), "Which AWS service did the Hub use?"
    )


@pytest.mark.parametrize(
    "good",
    [
        "The Hub served over 700 people, including internal employees, field agents and external partners.",
        "The Hub served over 700 internal and external stakeholders.",
        "Le Hub servait plus de 700 personnes, dont des employés, agents terrain et partenaires.",
    ],
)
def test_neutral_or_explicitly_mixed_adoption_population_stays_supported(good):
    assert ADOPTION not in constraint_failures(good, docs(HUB), HUB["question"])


def test_definition_facet_retains_only_the_named_platform_section():
    documents = supplement_facets(
        SimpleNamespace(documents=docs(HUB)), HUB["question"], docs(HUB)
    )
    selected = retained_facet_indices(documents, HUB["question"])
    assert any(
        documents[i].metadata["source"] == "10_apps_and_projects.txt" for i in selected
    )
    wrong = [
        Document(
            page_content=documents[0].page_content,
            metadata={**documents[0].metadata, "section_header": "ANOTHER PROJECT"},
        )
    ]
    assert not retained_facet_indices(wrong, HUB["question"])


@pytest.mark.parametrize(
    "model,configured,expected",
    [
        ("gemini-3.5-flash-lite", None, {"thinking_level": "minimal"}),
        ("gemini-3.5-flash-lite", "low", {"thinking_level": "low"}),
        ("gemini-2.5-flash", None, {"thinking_budget": 0}),
        ("gemini-3.8-flash", None, {"thinking_level": "low"}),
    ],
)
def test_effective_routing_is_model_aware_and_does_not_change_other_thinking(
    model, configured, expected
):
    settings = LLMSettings(model_name=model, routing_thinking_level=configured)
    assert settings.routing_parameters() == expected
    assert settings.verifier_parameters()["thinking_level"] == "low"
    if not model.startswith("gemini-2.5"):
        assert settings.thinking_parameters(generation=True) == {
            "thinking_level": "low"
        }


def test_real_sdk_boundary_limits_minimal_thinking_to_guard_and_batch_grade(
    monkeypatch,
):
    replies = [
        {
            "score": 95,
            "category": "professional",
            "reasoning": "Professional question",
            "standalone_query": "Which programming language?",
            "retrieval_query_en": "Which programming language?",
        },
        {"relevant_indices": [0]},
        {
            "answer": "Bolaji's primary programming language is Python.",
            "confidence": 0.9,
        },
        {"is_grounded": True, "addresses_question": True},
    ]
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(parsed=replies.pop(0), text="")

    monkeypatch.setattr(
        "google.genai.Client",
        lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
    )
    monkeypatch.setattr(nodes, "_llms", {})
    doc = Document(
        page_content="Bolaji's primary programming language is Python.",
        metadata={"source": "05_skills_and_tools.txt"},
    )
    monkeypatch.setattr(
        nodes, "_get_hybrid_search", lambda: SimpleNamespace(documents=[doc])
    )
    state = {
        "query": "Which programming language?",
        "agent_type": "professional",
        "context_documents": [doc],
        "graded_documents": [doc],
        "user_language": "en",
    }
    nodes.guardrail_node(state)
    nodes.grade_documents_node({**state, "documents": [doc]})
    state.update(nodes.generate_node(state))
    result = nodes.verify_grounding_node(state)
    assert result["grounding_checked"]
    assert len(calls) == 4
    assert [
        str(c["config"].thinking_config.thinking_level).lower().split(".")[-1]
        for c in calls
    ] == ["minimal", "minimal", "low", "low"]
    assert [c["config"].max_output_tokens for c in calls] == [750, 750, 2048, 4096]
    assert all(c["config"].temperature is None for c in calls)
    assert "iSheero" not in GUARDRAIL_PROMPT.messages[0].prompt.template


def test_routing_setting_changes_cache_namespace_and_effective_provenance(monkeypatch):
    from app.services.cache_service import CacheService
    from app.services.provenance import runtime_provenance
    from app.settings import get_settings

    settings = get_settings().model_copy(deep=True)
    monkeypatch.setattr("app.settings.get_settings", lambda: settings)
    minimal = CacheService._knowledge_namespace()
    assert runtime_provenance()["routing_thinking_config"] == {
        "thinking_level": "minimal"
    }
    settings.llm.routing_thinking_level = "low"
    assert CacheService._knowledge_namespace() != minimal
    assert runtime_provenance()["generation_thinking_config"] == {
        "thinking_level": "low"
    }


@pytest.mark.parametrize(
    "query,category,score",
    [
        ("What is Bolaji's role at an unknown organization?", "professional", 95),
        ("What is tomorrow's weather?", "out_of_scope", 0),
    ],
)
def test_minimal_guard_preserves_intent_routing_and_request_question(
    monkeypatch, query, category, score
):
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            parsed={
                "score": score,
                "category": category,
                "reasoning": "Intent classification",
                "standalone_query": query,
                "retrieval_query_en": query,
            },
            text="",
        )

    monkeypatch.setattr(
        "google.genai.Client",
        lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        ),
    )
    monkeypatch.setattr(nodes, "_llms", {})
    result = nodes.guardrail_node({"query": query})
    assert result["category"].value == category
    assert result["query"] == query
    assert len(calls) == 1
    assert (
        str(calls[0]["config"].thinking_config.thinking_level)
        .lower()
        .endswith("minimal")
    )
    assert "Classify the request's intent" in calls[0]["contents"]
