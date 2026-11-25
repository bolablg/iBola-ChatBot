"""Exact e8e4813 failures with separately labeled synthetic controls."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    _normalize_non_use_reference,
    constraint_failures,
    documentation_limits,
    prepare_evidence,
)
from scripts.run_eval import check_facts, load_golden
from tests.prompt_fields import prompt_field

FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/confirmed_reference_scope_failures.json"
    ).read_text()
)
CASES = {c["id"]: c for c in FIXTURE["cases"]}
STATUS = CASES["recruiter-sde-03-t2"]
PIPELINE = CASES["recruiter-sde-01-t1"]
BUDGET = CASES["recruiter-hda-03-t4"]
ROLE = "cdc_role_attribution_requires_direct_evidence"
NON_USE = "non_use_requires_explicit_negative_evidence"


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


def initial(case):
    return json.loads(
        next(r["text"] for r in case["sdk_responses"] if r["sequence"] == 3)
    )["answer"]


def state(case, answer):
    return {
        "query": case["question"],
        "original_query": case["question"],
        "answer": answer,
        "context_documents": docs(case),
        "agent_type": "professional",
        "user_language": case["lang"],
    }


def test_exact_initial_tool_status_was_true_and_llm_approved_before_false_refusal():
    answer = initial(STATUS)
    verdict = json.loads(
        next(r["text"] for r in STATUS["sdk_responses"] if r["sequence"] == 4)
    )
    assert verdict["is_grounded"] and verdict["addresses_question"]
    assert "has never used it" in answer
    assert not constraint_failures(answer, docs(STATUS), STATUS["question"])
    row = next(r for r in load_golden() if r["id"] == STATUS["id"])
    assert check_facts(answer, row, result={**STATUS, "grounding_checked": True})[0]
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(STATUS, answer))
    assert result.get("answer", answer) == answer and result["grounding_checked"]
    assert llm.with_structured_output.return_value.invoke.call_count == 1
    text = llm.with_structured_output.return_value.invoke.call_args.args[0][-1].content
    assert prompt_field(text, "draft_answer") == answer


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji has no Kafka experience at all and has never used it.",
        "Regarding Kafka, Bolaji has no Kafka experience at all and has never used it.",
        "He has no experience with Kafka and has never used it.",
    ],
)
def test_synthetic_local_reference_keeps_existing_source_predicate_check(answer):
    assert _normalize_non_use_reference(answer, docs(STATUS)).endswith("used Kafka.")
    assert NON_USE not in constraint_failures(answer, docs(STATUS), STATUS["question"])
    absence_only = [
        Document(
            page_content="Bolaji has no Kafka experience at all.",
            metadata={"source": "20_owner_confirmed_production_scope.txt"},
        )
    ]
    assert NON_USE in constraint_failures(answer, absence_only, STATUS["question"])


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji has no Kafka or Spark experience and has never used it.",
        "Bolaji has no Kafka experience at all. He has never used it.",
        "Bolaji has no Kafka experience, discussed Spark, and has never used it.",
        "His Kafka experience is not documented and he has never used it.",
        "Bolaji has no Kafka experience and has never used Spark.",
    ],
)
def test_synthetic_ambiguous_unknown_or_different_tool_never_borrows_confirmation(
    answer,
):
    assert _normalize_non_use_reference(answer, docs(STATUS)) == answer
    assert constraint_failures(answer, docs(STATUS), STATUS["question"])


@pytest.mark.parametrize(
    "source,kind",
    [
        ("05_skills_and_tools.txt", "curated"),
        ("20_owner_confirmed_production_scope.txt", "interpretation"),
        ("20_owner_confirmed_production_scope.txt", "pending_confirmation"),
    ],
)
def test_synthetic_reference_requires_eligible_retrieved_confirmation(source, kind):
    answer = "Bolaji has no Kafka experience at all and has never used it."
    evidence = [
        Document(
            page_content="Bolaji has no Kafka experience at all. Bolaji has never used Kafka.",
            metadata={"source": source, "evidence_type": kind},
        )
    ]
    assert _normalize_non_use_reference(answer, evidence) == answer


def test_synthetic_record_limit_stays_local_and_cannot_hide_operation_claims():
    gap = "Spark Structured Streaming is unsupported by direct production evidence."
    assert not documentation_limits(gap, ["Spark Structured Streaming"])
    question = (
        "Has Bolaji personally operated Spark Structured Streaming in production?"
    )
    assert not constraint_failures(gap, docs(STATUS), question)
    for affirmative in [
        " He operated Spark in production.",
        " He personally deployed Spark in production.",
    ]:
        assert constraint_failures(gap + affirmative, docs(STATUS), question)
    assert documentation_limits(
        "Spark Structured Streaming is unsupported.", ["Spark Structured Streaming"]
    )


@pytest.mark.parametrize(
    "status", ["is not documented", "remains unknown", "has not been confirmed"]
)
def test_synthetic_nominal_production_experience_limit_is_not_an_operation_assertion(
    status,
):
    answer = (
        "Personal production experience with Spark Structured Streaming " + status + "."
    )
    question = (
        "Has Bolaji personally operated Spark Structured Streaming in production?"
    )
    assert not constraint_failures(answer, docs(STATUS), question)
    assert not documentation_limits(answer, ["Spark Structured Streaming"])


def test_exact_repair_cannot_date_cdc_from_the_adjacent_migration():
    assert ROLE in constraint_failures(
        PIPELINE["answer"], docs(PIPELINE), PIPELINE["question"]
    )
    assert "personal_pipeline_project_identity_missing" not in constraint_failures(
        initial(PIPELINE), docs(PIPELINE), PIPELINE["question"]
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "As Global Data Analyst,",
        "In his Global Data Analyst role,",
        "During his earlier role as Global Data Analyst,",
        "While he was Global Data Analyst,",
    ],
)
def test_synthetic_cdc_role_claim_needs_same_project_role_proof(prefix):
    answer = (
        prefix
        + " Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB change data capture pipeline at Gozem."
    )
    assert ROLE in constraint_failures(answer, docs(PIPELINE), PIPELINE["question"])
    unrelated = Document(
        page_content="As Global Data Analyst, Bolaji operated Dataflow for the cloud migration."
    )
    assert ROLE in constraint_failures(
        answer, docs(PIPELINE) + [unrelated], PIPELINE["question"]
    )
    direct = Document(
        page_content="As Global Data Analyst, Bolaji personally designed and operated the MongoDB CDC pipeline at Gozem."
    )
    assert ROLE not in constraint_failures(
        answer, docs(PIPELINE) + [direct], PIPELINE["question"]
    )


def test_synthetic_role_headings_adjacent_claims_and_unknown_periods():
    answer = "Global Data Analyst\nBolaji personally operated the MongoDB CDC pipeline at Gozem."
    assert ROLE in constraint_failures(answer, docs(PIPELINE), PIPELINE["question"])
    direct = Document(page_content=answer)
    assert ROLE not in constraint_failures(
        answer, docs(PIPELINE) + [direct], PIPELINE["question"]
    )
    adjacent = "As Global Data Analyst, Bolaji built a warehouse. He personally operated the MongoDB CDC pipeline."
    assert ROLE in constraint_failures(adjacent, docs(PIPELINE), PIPELINE["question"])
    scoped = adjacent.replace("He personally", "Separately, he personally")
    assert ROLE not in constraint_failures(scoped, docs(PIPELINE), PIPELINE["question"])
    unknown = "The retrieved evidence does not establish whether he designed the MongoDB CDC pipeline as Global Data Analyst."
    assert ROLE not in constraint_failures(
        unknown, docs(PIPELINE), PIPELINE["question"]
    )


def test_synthetic_source_quote_can_add_undated_proof_but_cannot_cleanse_a_wrong_role():
    quote = 'Named production implementation (OWNER-CONFIRMED PRODUCTION RESPONSIBILITY): "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."'
    migration = "As Global Data Analyst, Bolaji built 50+ scheduled ETL pipelines and migrated 10+ TB within a single quarter. "
    assert ROLE not in constraint_failures(
        migration + quote, docs(PIPELINE), PIPELINE["question"]
    )
    assert ROLE in constraint_failures(
        PIPELINE["answer"] + quote, docs(PIPELINE), PIPELINE["question"]
    )
    assert ROLE not in constraint_failures(
        "Global Data Analyst\n" + migration + quote,
        docs(PIPELINE),
        PIPELINE["question"],
    )


def test_synthetic_unrelated_cdc_mention_cannot_prove_role_delivery():
    answer = (
        "As Global Data Analyst, Bolaji personally operated the MongoDB CDC pipeline."
    )
    unrelated = Document(
        page_content="As Global Data Analyst, Bolaji operated Dataflow for the cloud migration, independently of the MongoDB CDC pipeline."
    )
    assert ROLE in constraint_failures(
        answer, docs(PIPELINE) + [unrelated], PIPELINE["question"]
    )
    other_role = Document(
        page_content="As Global Data Analyst, Bolaji operated Dataflow for the cloud migration while he operated the MongoDB CDC pipeline as Head of Data."
    )
    assert ROLE in constraint_failures(
        answer, docs(PIPELINE) + [other_role], PIPELINE["question"]
    )
    assert ROLE not in constraint_failures(
        answer.replace("Global Data Analyst", "Head of Data"),
        docs(PIPELINE) + [other_role],
        PIPELINE["question"],
    )


@pytest.mark.parametrize(
    "answer",
    [
        "It is unknown whether Bolaji operated the MongoDB CDC pipeline as Global Data Analyst.",
        "Whether Bolaji operated the MongoDB CDC pipeline as Global Data Analyst remains unconfirmed.",
        "Bolaji never operated the MongoDB CDC pipeline as Global Data Analyst.",
        "As Global Data Analyst, it is unknown whether Bolaji operated the MongoDB CDC pipeline.",
        "As Global Data Analyst, Bolaji did not operate the MongoDB CDC pipeline.",
    ],
)
def test_synthetic_unknown_or_denied_role_is_not_an_affirmative_role_claim(answer):
    # This rule checks affirmative role attribution. The LLM still verifies
    # whether any negative biographical assertion is actually supported.
    assert ROLE not in constraint_failures(answer, docs(PIPELINE), PIPELINE["question"])


def test_synthetic_bounded_budget_repair_preserves_known_outcome_and_final_verification():
    repaired = "At Gozem, Bolaji's FinOps work recorded a 42.57% Google Cloud cost reduction in one quarter. The retrieved evidence does not establish department-budget ownership, investment amount, planning horizon, approval authority or rejected options."
    verifier = Mock()
    verifier.invoke.side_effect = [
        GroundingVerdict(is_grounded=True),
        GroundingVerdict(is_grounded=True),
    ]
    repair = Mock()
    repair.invoke.return_value = GeneratedAnswer(answer=repaired, confidence=0.8)
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(BUDGET, initial(BUDGET)))
    assert result["grounding_checked"] and "42.57%" in result["answer"]
    assert "outcome" not in result["answer"].lower()
    assert verifier.invoke.call_count == 2
    actual = prompt_field(verifier.invoke.call_args.args[0][-1].content, "draft_answer")
    assert actual == result["answer"]


def test_synthetic_rejection_still_withholds_the_original_tool_status_text():
    answer = initial(STATUS)
    verifier = Mock()
    verifier.invoke.return_value = GroundingVerdict(
        is_grounded=False, corrected_answer=""
    )
    repair = Mock()
    repair.invoke.return_value = GeneratedAnswer(answer=answer, confidence=1)
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(STATUS, answer))
    assert not result["grounding_checked"]
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS["en"]


@pytest.mark.parametrize("valid", [True, False])
def test_synthetic_single_project_repair_cannot_inherit_a_wrong_cdc_role(valid):
    repaired = "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."
    verifier = Mock()
    verifier.invoke.return_value = GroundingVerdict(
        is_grounded=True, corrected_answer=""
    )
    repair = Mock()
    repair.invoke.return_value = GeneratedAnswer(
        answer=repaired if valid else PIPELINE["answer"], confidence=0.8
    )
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(PIPELINE, initial(PIPELINE)))
    assert result["grounding_checked"] is valid
    assert result["answer"] == (
        repaired if valid else INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    )
    assert verifier.invoke.call_count == (2 if valid else 1)
