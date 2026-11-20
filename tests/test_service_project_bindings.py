"""Captured failed drafts, plus labeled synthetic scope counterexamples."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.services.contextual_handoff import contextual_handoff
from app.services.evidence_policy import (
    constraint_failures,
    documentation_limits,
    prepare_evidence,
    preserve_requested_evidence,
)
from app.settings import TracingSettings
from scripts.run_eval import InProcessTarget

FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/service_project_binding_failures.json"
    ).read_text()
)
CASES = {row["id"]: row for row in FIXTURE["cases"]}
AWS = CASES["recruiter-sda-05-t2"]


def docs(case=AWS):
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


def test_actual_repaired_list_keeps_ec2_project_and_s3_unknown_independent():
    draft = AWS["captured_candidates"][-1]["result"]["answer"]
    assert not constraint_failures(draft, docs(), AWS["question"])


def test_actual_mixed_architect_draft_is_not_laundered_into_valid_scope():
    case = CASES["recruiter-sda-01-t1"]
    draft = case["captured_candidates"][0]["result"]["answer"]
    assert constraint_failures(draft, docs(case), case["question"])


def test_actual_correction_role_prefix_already_has_local_role_binding():
    case = CASES["recruiter-sde-01-t1"]
    corrected = next(
        c["result"]["corrected_answer"]
        for c in case["captured_candidates"]
        if c["schema"] == "GroundingVerdict"
    )
    assert "migration_earlier_role_missing" not in constraint_failures(
        corrected, docs(case), case["question"]
    )


@pytest.mark.parametrize(
    "draft",
    [
        "Bolaji personally designed and operated AWS EC2 and AWS S3 in production for the Gozem Data Hub.",
        "Bolaji personally designed and operated AWS EC2 in production for the MongoDB CDC pipeline at Gozem.",
        "Bolaji personally designed and operated AWS EC2 for the Gozem Data Hub and AWS S3 in production at Gozem (the employer for S3 is not specified).",
        "Bolaji personally designed and operated AWS EC2 for the Gozem Data Hub and Pub/Sub and Dataflow in production for Rintio.",
    ],
)
def test_synthetic_wrong_project_and_conflicting_disclaimer_still_fail(draft):
    assert any(
        f.startswith("requested_personal_production_scope:")
        for f in constraint_failures(draft, docs(), AWS["question"])
    )


def test_synthetic_separately_bound_projects_with_shared_predicate_pass():
    draft = "Bolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub and Pub/Sub and Dataflow in production for the MongoDB CDC pipeline at Gozem. Bolaji has no Kafka experience at all."
    assert not constraint_failures(draft, docs(), AWS["question"])


def test_synthetic_prior_project_does_not_bind_a_new_personal_predicate():
    draft = "At Gozem, Bolaji architected the Gozem Data Hub and personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."
    assert not constraint_failures(
        draft, docs(), CASES["recruiter-sda-01-t1"]["question"]
    )
    wrong = draft.replace(
        "Pub/Sub and Dataflow in production for the MongoDB CDC pipeline",
        "AWS S3 in production",
    )
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        wrong, docs(), CASES["recruiter-sda-01-t1"]["question"]
    )


@pytest.mark.parametrize("separator", [", ", " and "])
def test_synthetic_three_item_list_preserves_the_middle_unknown_employer(separator):
    draft = (
        "Bolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub"
        + separator
        + "AWS S3 in production (the employer or project is not specified)"
        + separator
        + "Pub/Sub and Dataflow in production for the MongoDB CDC pipeline at Gozem. "
        "Bolaji has no Kafka experience at all."
    )
    assert not constraint_failures(draft, docs(), AWS["question"])
    wrong = draft.replace(
        "AWS S3 in production (the employer or project is not specified)",
        "AWS S3 in production for the Gozem Data Hub (the employer is not specified)",
    )
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        wrong, docs(), AWS["question"]
    )


def test_synthetic_unknown_employer_never_proves_personal_service_operation():
    draft = AWS["captured_candidates"][-1]["result"]["answer"]
    evidence = [
        Document(
            page_content="For the Gozem Data Hub, Bolaji personally designed and operated AWS EC2 in production. Inventory: AWS S3.",
            metadata={"source": "20_owner_confirmed_production_scope.txt"},
        )
    ]
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        draft, evidence, AWS["question"]
    )


@pytest.mark.parametrize(
    "answer,missing",
    [
        (
            "The evidence does not establish direct-report counts or manager-of-managers layers.",
            False,
        ),
        ("Direct-report counts and manager-of-managers layers are documented.", True),
        ("Budget authority is unknown. He managed two managers.", True),
        ("He onboarded 12 hires and had 12 active manager users.", True),
    ],
)
def test_synthetic_management_aliases_keep_the_local_unknown(answer, missing):
    assert (
        bool(
            documentation_limits(
                answer,
                [
                    "direct reports|manager layers|reporting lines|authority over managers"
                ],
            )
        )
        is missing
    )


def test_synthetic_unknown_count_does_not_excuse_manager_layer_assertion():
    answer = "Direct-report counts are unknown, but he managed two managers."
    assert "quantified_manager_layers_require_direct_evidence" in constraint_failures(
        answer, [], "What manager layers did he lead?"
    )


def test_synthetic_oncall_gap_requires_retrieved_limitation_and_explicit_boundary():
    question = "What exact on-call rotation and uptime SLA did Bolaji personally own for the production MongoDB CDC pipeline?"
    bare = "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline. Specific details regarding an on-call rotation and uptime SLA remain unknown."
    eligible = [
        d
        for d in docs()
        if Path(d.metadata["source"]).name == "20_owner_confirmed_production_scope.txt"
    ]
    assert "production_on_call_scope_unknown" in constraint_failures(
        bare, eligible, question
    )
    restored, ids = preserve_requested_evidence(bare, eligible, question, "English")
    assert "production_on_call_scope_unknown" in ids
    assert (
        "The retrieved evidence does not establish on-call responsibility." in restored
    )
    assert contextual_handoff(
        {
            "answer": restored,
            "grounding_checked": True,
            "agent_type": "professional",
            "evidence": [{}],
        },
        question,
        enabled=True,
    )
    assert not contextual_handoff(
        {
            "answer": restored,
            "grounding_checked": True,
            "agent_type": "professional",
            "evidence": [{}],
        },
        question + " Do not contact me.",
        enabled=True,
    )
    # A filename alone cannot activate a production on-call documentation gap.
    empty = [
        Document(
            page_content="Service inventory only.",
            metadata={"source": "20_owner_confirmed_production_scope.txt"},
        )
    ]
    assert "production_on_call_scope_unknown" not in constraint_failures(
        bare, empty, question
    )


@pytest.mark.parametrize(
    "enabled,keys,expected",
    [
        (False, True, False),
        (None, True, True),
        (None, False, False),
        (True, False, True),
    ],
)
def test_explicit_tracing_setting_takes_precedence(
    monkeypatch, enabled, keys, expected
):
    for name in ("LANGFUSE_ENABLED", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    kwargs = {
        "public_key": "fixture" if keys else None,
        "secret_key": "fixture" if keys else None,
    }
    if enabled is not None:
        kwargs["enabled"] = enabled
    assert TracingSettings(**kwargs).is_active is expected


def test_in_process_eval_uses_request_local_no_trace():
    target = InProcessTarget.__new__(InProcessTarget)
    target.service = Mock()
    target.ask("A fixture question", history=[["Earlier", "Answer"]])
    assert target.service.process_query.call_args.args[-1] == {"no_trace": True}
