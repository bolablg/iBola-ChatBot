"""Framework domain and record-bound unknowns retain their actual source meaning."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    bound_record_limits,
    constraint_failures,
    prepare_evidence,
    preserve_requested_evidence,
)
from tests.prompt_fields import prompt_field

CASES = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_framework_record_limits.json"
    ).read_text()
)
RULE = "performance_framework_domain_requires_direct_evidence"


def docs(row):
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in row["evidence"]
        ],
        row["question"],
    )


def test_captured_team_framework_claim_needs_matching_domain_design_proof():
    row = CASES["recruiter-hda-03-t3"]
    assert row["scores"]["faithfulness"] == 4
    assert RULE in constraint_failures(row["answer"], docs(row), row["question"])
    claim = "Bolaji designed a team performance framework."
    for text in (
        "Bolaji designed a driver performance framework.",
        "His colleague designed a team performance framework.",
        "If Bolaji designed a team performance framework, ask for details.",
        "Source: file.txt | Section: Bolaji designed a team performance framework.",
    ):
        assert RULE in constraint_failures(
            claim, docs(row) + [Document(page_content=text)], row["question"]
        )
    assert RULE not in constraint_failures(
        claim, docs(row) + [Document(page_content=claim)], row["question"]
    )
    for text in (
        "He defined competency frameworks and partnered with HR on performance management.",
        "If he designed a team performance framework, ask him.",
    ):
        assert RULE not in constraint_failures(text, docs(row), row["question"])


def test_exact_captured_undocumented_fields_are_bounded_to_retrieved_record():
    row = CASES["recruiter-hda-05-t6"]
    answer, _ = preserve_requested_evidence(
        row["answer"], docs(row), row["question"], "en"
    )
    assert (
        "The retrieved evidence does not establish build versus vendor comparisons"
        in answer
    )
    assert bound_record_limits(answer, "en") == answer
    assert not constraint_failures(answer, docs(row), row["question"])


@pytest.mark.parametrize(
    "language,answer",
    [
        ("en", "Budget amounts and approval authority are undocumented."),
        ("fr", "Les montants du budget sont inconnus."),
    ],
)
def test_complete_nominal_limits_are_scoped_and_idempotent(language, answer):
    result = bound_record_limits(answer, language)
    assert result != answer and ("retrieved evidence" in result or "dossier" in result)
    assert bound_record_limits(result, language) == result


@pytest.mark.parametrize(
    "answer",
    [
        "The retrieved record does not establish budget amounts.",
        "He has no Kafka experience.",
        "If budget amounts are unknown, ask him.",
        '"Budget amounts are unknown."',
        "He operated a data platform and budget amounts are unknown.",
        "The framework that supports production is unknown.",
        "Budget amounts vary and approval authority is unknown.",
        "Budget amounts fluctuate and approval authority is unknown.",
    ],
)
def test_scope_normalization_cannot_change_absence_conditional_quotes_or_extra_predicates(
    answer,
):
    assert bound_record_limits(answer, "en") == answer


@pytest.mark.parametrize("accept", [False, True])
def test_record_scoping_still_requires_verification_of_exact_final_text(accept):
    row = CASES["recruiter-hda-05-t6"]
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=accept
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "original_query": row["question"],
                "query": row["question"],
                "answer": row["answer"],
                "context_documents": docs(row),
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert bool(result["grounding_checked"]) == accept
    if accept:
        assert "The retrieved evidence" in result["answer"]
        assert (
            prompt_field(
                llm.with_structured_output.return_value.invoke.call_args.args[0][
                    -1
                ].content,
                "draft_answer",
            )
            == result["answer"]
        )
