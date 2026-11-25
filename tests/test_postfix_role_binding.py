"""A role after the same migration action group is valid, another actor is not."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import constraint_failures, prepare_evidence
from tests.prompt_fields import prompt_field

CAPTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_postfix_role_failure.json"
    ).read_text()
)
RECORD = CAPTURE["record"]
VALID = RECORD["response"]["unsupported_claims"][0]
RULE = "migration_earlier_role_missing"


def docs():
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in RECORD["response"]["evidence"]
        ],
        RECORD["question"],
    )


def state(answer):
    return {
        "query": RECORD["question"],
        "answer": answer,
        "context_documents": docs(),
        "agent_type": "professional",
        "user_language": "en",
    }


def test_actual_candidate_reported_by_api_has_an_explicit_local_postfix_role():
    assert "as a Global Data Analyst" in VALID
    assert RULE not in constraint_failures(VALID, docs(), RECORD["question"])


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji migrated 10+ TB as a Global Data Analyst.",
        "As Head of Data, Bolaji designed the MongoDB CDC pipeline. He migrated 10+ TB as a Global Data Analyst.",
        "Bolaji migrated 10+ TB within a single quarter without operational interruption while establishing 50+ scheduled ETL pipelines as a Global Data Analyst.",
        "Bolaji designed 50+ scheduled ETL pipelines as a Global Data Analyst at Gozem.",
    ],
)
def test_same_actor_migration_action_group_can_have_a_trailing_role(answer):
    assert RULE not in constraint_failures(answer, docs(), RECORD["question"])


@pytest.mark.parametrize(
    "answer",
    [
        "As Head of Data, Bolaji migrated 10+ TB as a Global Data Analyst.",
        "As Global Data Analyst, Bolaji migrated 10+ TB as Head of Data.",
        "Bolaji migrated 10+ TB as Head of Data.",
        "His profile mentions Head of Data. Bolaji migrated 10+ TB. As Global Data Analyst, he designed 50+ scheduled ETL pipelines.",
        "Bolaji migrated 10+ TB while his colleague established 50+ scheduled ETL pipelines as a Global Data Analyst.",
        "Bolaji migrated 10+ TB as Head of Data while his colleague migrated 10+ TB as a Global Data Analyst.",
        "Bolaji migrated 10+ TB and designed Data Hub before migrating 10+ TB as a Global Data Analyst.",
        "Bolaji migrated 10+ TB while mentoring a team as a Global Data Analyst.",
        "Bolaji migrated 10+ TB and designed Data Hub as a Global Data Analyst.",
        "Bolaji migrated 10+ TB; he worked as a Global Data Analyst.",
        "Bolaji migrated 10+ TB. Global Data Analyst:\nHe designed 50+ scheduled ETL pipelines.",
    ],
)
def test_postfix_role_cannot_cleanse_a_wrong_claim_or_borrow_a_different_action(answer):
    assert RULE in constraint_failures(answer, docs(), RECORD["question"])


def test_valid_postfix_correction_still_needs_independent_exact_text_approval():
    # Synthetic correction removes the reported candidate's redundant trailing
    # CDC sentence, which otherwise inherits the migration role.
    corrected = VALID.rsplit(" At Gozem,", 1)[0]
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=corrected),
        GroundingVerdict(is_grounded=True),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            state("As Head of Data, Bolaji migrated 10+ TB.")
        )
    assert result["grounding_checked"] and result["answer"].startswith(corrected)
    assert invoke.call_count == 2
    assert (
        prompt_field(invoke.call_args.args[0][-1].content, "draft_answer")
        == result["answer"]
    )


def test_wrong_local_role_remains_rejected_after_positive_verdict_and_repair():
    wrong = "As Head of Data, Bolaji migrated 10+ TB as a Global Data Analyst."
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=wrong),
        GeneratedAnswer(answer=wrong),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(wrong))
    assert not result["grounding_checked"] and invoke.call_count == 2
