"""Cloud platform choice and later FinOps reduction have separate scope."""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.evidence_policy import constraint_failures, prepare_evidence

CASE = json.loads(Path("tests/fixtures/live_finops_relationship.json").read_text())


def test_actual_live_causal_attribution_is_rejected_even_with_correct_fact_appended():
    docs = [
        Document(
            page_content=e["content_preview"],
            metadata={"source": e["source"], "section_header": e["section"]},
        )
        for e in CASE["evidence"]
    ]
    for answer in [
        CASE["answer"],
        CASE["answer"] + " Separately, later FinOps work recorded 42.57% lower costs.",
    ]:
        assert "initial_cloud_choice_not_finops_cause" in constraint_failures(
            answer, prepare_evidence(docs), CASE["question"]
        )


@pytest.mark.parametrize(
    "answer",
    [
        "The initial cloud-first choice contributed to a 42.57% reduction.",
        "He chose a cloud-first approach. This decision reduced costs by 42.57%.",
        "He chose a cloud-first approach. Budget authority is unverified, this decision contributed to the 42.57% reduction.",
    ],
)
def test_initial_platform_choice_cannot_borrow_later_finops_result(answer):
    assert "initial_cloud_choice_not_finops_cause" in constraint_failures(
        answer, [], CASE["question"]
    )


@pytest.mark.parametrize(
    "answer",
    [
        "As Global Data Analyst he chose a cloud-first approach. Separately, later Head of Data FinOps work using Dataform reduced costs by 42.57% in one quarter.",
        "He chose a cloud-first approach. Later he led Dataform and FinOps work. That initiative reduced costs by 42.57%.",
        "The initial cloud-first choice did not cause the 42.57% reduction.",
        "The record does not establish that the initial cloud-first choice contributed to the 42.57% reduction.",
    ],
)
def test_separate_correct_facts_and_explicit_limits_are_not_rejected(answer):
    assert "initial_cloud_choice_not_finops_cause" not in constraint_failures(
        answer, [], CASE["question"]
    )


def test_summary_exclusion_does_not_use_tool_equivalence_exception():
    summary = Document(
        page_content="Narrative pitch.", metadata={"source": "17_highlights_pitch.txt"}
    )
    assert prepare_evidence([summary], "What budget authority is documented?") == []
    assert (
        prepare_evidence([summary], "Are his tools transferable or equivalent to dbt?")
        == []
    )
    assert prepare_evidence(
        [summary], "Describe his professional narrative positioning."
    )


@pytest.mark.parametrize(
    "section",
    [
        "FinOps and analytical serving",
        "Data platform modernization and FinOps transformation",
    ],
)
def test_both_canonical_finops_sections_preserve_metric_scope(section):
    doc = Document(
        page_content="Recorded Google Cloud costs fell by 42.57% in one quarter.",
        metadata={
            "source": "90_website_canon_llms_full.txt",
            "section_header": section,
        },
    )
    prepared = prepare_evidence([doc])[0]
    scope = prepared.metadata["metric_scope"]
    assert scope["id"] == "finops_cost_reduction"
    assert scope["role"] == "Head of Data"
    assert "initial cloud" in scope["not_established"]
