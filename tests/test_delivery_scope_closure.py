"""Exact delivered failures and separate synthetic scope/repair controls."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    preserve_requested_evidence,
)

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/answer_delivery_failures.json").read_text()
)
CASES = {r["id"]: r for r in CAPTURE["cases"]}
PIPELINE = CASES["recruiter-sde-01-t1"]
WORK = CASES["recruiter-adi-05-t6"]
IDENTITY = "personal_pipeline_project_identity_missing"
LOCATION = "onsite_availability_requires_same_actor_evidence"


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


def state(case, answer=None):
    return {
        "query": case["question"],
        "original_query": case["question"],
        "answer": answer or case["answer"],
        "context_documents": docs(case),
        "agent_type": "professional",
        "user_language": case["lang"],
    }


def test_exact_delivered_tools_only_proof_restores_its_named_implementation_once():
    assert IDENTITY in constraint_failures(
        PIPELINE["answer"], docs(PIPELINE), PIPELINE["question"]
    )
    restored, ids = preserve_requested_evidence(
        PIPELINE["answer"], docs(PIPELINE), PIPELINE["question"], "English"
    )
    assert IDENTITY in ids
    assert (
        "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."
        in restored
    )
    assert not constraint_failures(restored, docs(PIPELINE), PIPELINE["question"])
    assert preserve_requested_evidence(
        restored, docs(PIPELINE), PIPELINE["question"], "English"
    ) == (restored, [])
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(PIPELINE))
    assert result["answer"] == restored
    actual = llm.with_structured_output.return_value.invoke.call_args.args[0][
        -1
    ].content
    assert actual.rsplit("Answer to verify:\n", 1)[1] == restored


def test_synthetic_named_warehouse_alternative_does_not_inject_cdc():
    answer = "At Gozem, Bolaji built a BigQuery warehouse with scheduled ETL pipelines."
    assert IDENTITY not in constraint_failures(
        answer, docs(PIPELINE), PIPELINE["question"]
    )
    assert preserve_requested_evidence(
        answer, docs(PIPELINE), PIPELINE["question"], "English"
    ) == (answer, [])


@pytest.mark.parametrize(
    "draft",
    [
        "Bolaji personally designed and operated Pub/Sub and Dataflow in production for Gozem's MongoDB CDC pipeline.",
        "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for a MongoDB CDC pipeline.",
        "Bolaji personally designed and operated Pub/Sub and Dataflow in production at Gozem for the MongoDB CDC pipeline.",
        "Chez Gozem, Bolaji a personnellement conçu et exploité Pub/Sub et Dataflow en production pour le pipeline CDC MongoDB.",
    ],
)
def test_synthetic_valid_possessive_and_indefinite_pipeline_identity(draft):
    assert IDENTITY not in constraint_failures(
        draft, docs(PIPELINE), PIPELINE["question"]
    )


@pytest.mark.parametrize(
    "draft,expected",
    [
        (
            "At Gozem, Bolaji never used Dataflow for the MongoDB CDC pipeline.",
            "non_use_requires_explicit_negative_evidence",
        ),
        (
            "At Gozem, his colleague operated Pub/Sub for the MongoDB CDC pipeline.",
            IDENTITY,
        ),
    ],
)
def test_synthetic_denial_and_other_actor_do_not_establish_candidate_pipeline_proof(
    draft,
    expected,
):
    assert expected in constraint_failures(draft, docs(PIPELINE), PIPELINE["question"])


def test_synthetic_negated_tool_does_not_activate_cdc_restoration():
    draft = "At Gozem, Bolaji built a BigQuery warehouse, not Dataflow."
    assert IDENTITY not in constraint_failures(
        draft, docs(PIPELINE), PIPELINE["question"]
    )
    _, ids = preserve_requested_evidence(
        draft, docs(PIPELINE), PIPELINE["question"], "English"
    )
    assert IDENTITY not in ids


@pytest.mark.parametrize(
    "tail",
    ["and did not use Dataflow", "without Dataflow", "without Pub/Sub or Dataflow"],
)
def test_synthetic_use_negation_keeps_its_tool_scope(tail):
    draft = "At Gozem, Bolaji built a BigQuery warehouse " + tail + "."
    assert IDENTITY not in constraint_failures(
        draft, docs(PIPELINE), PIPELINE["question"]
    )


def test_synthetic_candidate_action_cannot_borrow_a_colleagues_pipeline():
    draft = "At Gozem, Bolaji operated Pub/Sub for a warehouse while his colleague operated Dataflow for the MongoDB CDC pipeline."
    assert IDENTITY in constraint_failures(draft, docs(PIPELINE), PIPELINE["question"])


@pytest.mark.parametrize(
    "suffix",
    [
        " The MongoDB CDC pipeline is unrelated to this example.",
        " This is not for the MongoDB CDC pipeline.",
        " The MongoDB CDC pipeline is unknown.",
    ],
)
def test_synthetic_detached_or_denied_project_name_is_not_identity_proof(suffix):
    answer = (
        "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production."
        + suffix
    )
    assert IDENTITY in constraint_failures(answer, docs(PIPELINE), PIPELINE["question"])


def test_synthetic_wrong_source_never_restores_a_named_implementation():
    wrong = [
        Document(
            page_content="For the MongoDB CDC pipeline at Gozem, Bolaji operated Pub/Sub and Dataflow.",
            metadata={"source": "unrelated.txt", "section_header": "Fixture"},
        )
    ]
    _, ids = preserve_requested_evidence(
        PIPELINE["answer"], wrong, PIPELINE["question"], "English"
    )
    assert IDENTITY not in ids


def test_exact_french_work_arrangement_fails_source_scope_check():
    assert LOCATION in constraint_failures(WORK["answer"], docs(WORK), WORK["question"])


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji is available for on-site work.",
        "Il est disponible pour des missions en présentiel.",
        "Bolaji est disponible en mode present.",
        "Compensation is unknown, but Bolaji is available for on-site work.",
    ],
)
def test_synthetic_remote_or_employer_policy_never_proves_onsite_availability(answer):
    evidence = [
        Document(
            page_content="Bolaji is available for remote work and full-time roles. His employer supports on-site work."
        )
    ]
    assert LOCATION in constraint_failures(
        answer, evidence, "What work arrangements are confirmed?"
    )


@pytest.mark.parametrize(
    "answer",
    [
        "On-site work is not documented.",
        "The evidence does not establish whether he is available for on-site work.",
        "Le dossier ne précise pas s'il est disponible en présentiel.",
        "Bolaji is not available for on-site work.",
        "Bolaji is unavailable for on-site work.",
        "The evidence does not confirm whether Bolaji is currently available for on-site work.",
        "À présent, Bolaji est disponible pour des missions à distance.",
        "Bolaji is currently available for remote work.",
    ],
)
def test_synthetic_location_guard_preserves_unknown_negation_and_temporal_wording(
    answer,
):
    assert LOCATION not in constraint_failures(
        answer, [], "What work arrangements are confirmed?"
    )


@pytest.mark.parametrize(
    "source",
    [
        "Bolaji is available for on-site work.",
        "Bolaji est disponible pour des missions en présentiel.",
        "Bolaji is currently available for on-site work.",
        "Bolaji is available for on-site work and information security roles.",
    ],
)
def test_synthetic_same_actor_same_arrangement_can_be_established(source):
    assert LOCATION not in constraint_failures(
        source, [Document(page_content=source)], "Availability?"
    )


@pytest.mark.parametrize(
    "source",
    [
        "His colleague is available for on-site work.",
        "Bolaji is available for remote work while his colleague works on-site.",
        "Bolaji was previously available for on-site work during a teaching bootcamp.",
        "Bolaji is available for on-site interviews.",
        "Bolaji is available for on-site work in his former role.",
        "Bolaji is available for on-site work, but only for interviews.",
        "Bolaji is available for on-site work. Only for interviews.",
        "Bolaji is available for on-site work, but this applies only to interviews.",
        "Bolaji is available for on-site work, but availability is limited to interviews.",
        "Bolaji is available for on-site work.\n\nThis is restricted to interviews.",
        "Bolaji est disponible pour des missions en présentiel, mais cela s'applique uniquement aux entretiens.",
    ],
)
def test_synthetic_other_actor_history_or_interviews_are_not_job_location_proof(source):
    assert LOCATION in constraint_failures(
        "Bolaji is available for on-site work.",
        [Document(page_content=source)],
        "Availability?",
    )


def test_synthetic_unrelated_interview_sentence_does_not_cancel_current_availability():
    source = "Bolaji is currently available for on-site work. His colleague arranges interviews."
    assert LOCATION not in constraint_failures(
        "Bolaji is available for on-site work.",
        [Document(page_content=source)],
        "Availability?",
    )


@pytest.mark.parametrize("repair_valid", [True, False])
def test_falsely_approving_verifier_cannot_bypass_location_boundary(repair_valid):
    repaired = WORK["answer"].replace("en mode present", "à temps plein")
    verifier = Mock()
    verifier.invoke.return_value = GroundingVerdict(
        is_grounded=True, corrected_answer=""
    )
    repair = Mock()
    repair.invoke.return_value = GeneratedAnswer(
        answer=repaired if repair_valid else WORK["answer"], confidence=0.8
    )
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(WORK))
    assert result["grounding_checked"] is repair_valid
    assert result["answer"] == (
        repaired if repair_valid else INSUFFICIENT_EVIDENCE_ANSWERS["fr"]
    )
    assert verifier.invoke.call_count == (2 if repair_valid else 1)
