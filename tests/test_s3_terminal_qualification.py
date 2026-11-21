"""Captured repaired S3 scope and controls for nominal unknown affiliation."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    confirmed_use_conflicts,
    constraint_failures,
    policy_data,
)

CAPTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_s3_terminal_qualification.json"
    ).read_text()
)
RECORD = CAPTURE["record"]
REPAIRED = CAPTURE["captured_candidates"][-1]["result"]["answer"]
QUERY = RECORD["question"]


def docs():
    return [
        Document(
            page_content=e["content_preview"],
            metadata={**e, "section_header": e["section"]},
        )
        for e in RECORD["response"]["evidence"]
    ]


def test_actual_repair_distinguishes_ec2_project_from_unknown_s3_affiliation():
    assert not constraint_failures(REPAIRED, docs(), QUERY)
    assert not constraint_failures(REPAIRED, docs(), QUERY + " At Gozem")
    original = CAPTURE["captured_candidates"][0]["result"]["answer"]
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        original, docs(), QUERY
    )


@pytest.mark.parametrize("accepted", [True, False])
def test_actual_repair_still_requires_independent_final_verification(accepted):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=accepted
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": REPAIRED,
                "original_query": QUERY,
                "query": QUERY,
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": docs(),
            }
        )
    assert result["grounding_checked"] is accepted
    assert llm.with_structured_output.return_value.invoke.call_count >= 1
    assert any(
        REPAIRED in str(call)
        for call in llm.with_structured_output.return_value.invoke.call_args_list
    )


@pytest.mark.parametrize(
    "tail",
    [
        "with the employer or project for that S3 bucket remaining unknown",
        "the employer for that AWS S3 bucket remains unknown",
        "with its project for the S3 bucket unspecified",
    ],
)
def test_terminal_nominal_qualification_is_local_to_s3(tail):
    answer = (
        "Bolaji personally operated AWS EC2 in production for Gozem Data Hub and personally operated AWS S3 in production, "
        + tail
        + ". Bolaji has no Kafka experience."
    )
    assert not constraint_failures(answer, docs(), QUERY + " At Gozem")
    explicit = answer.replace(
        "AWS S3 in production,", "AWS S3 in production for Gozem Data Hub,"
    )
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        explicit, docs(), QUERY
    )


@pytest.mark.parametrize(
    "tail",
    [
        "with the employer or project for that S3 bucket not unknown",
        "with the employer or project for that S3 bucket remaining unknown and operationally attributable to Gozem",
        "with the employer or project for that S3 bucket that he designed remaining unknown",
        "with the employer or project he operated for that S3 bucket remaining unknown",
        "with the employer or project for that EC2 bucket remaining unknown",
    ]
    + [
        word + " the employer or project for that S3 bucket is unknown"
        for word in (
            "if",
            "unless",
            "whether",
            "assuming",
            "suppose",
            "hypothetical",
            "could",
            "would",
            "might",
        )
    ],
)
def test_non_nominal_conditional_or_wrong_tool_cannot_erase_requested_affiliation(tail):
    answer = (
        "Bolaji personally operated AWS S3 in production, "
        + tail
        + ". Bolaji has no Kafka experience."
    )
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        answer, docs(), QUERY + " At Gozem"
    )


def test_unknown_affiliation_does_not_supply_operation_proof_or_erase_known_ec2_project():
    answer = "Bolaji personally operated AWS S3 in production, with the employer or project for that S3 bucket remaining unknown. Bolaji has no Kafka experience."
    evidence = [
        Document(
            page_content="Inventory: AWS S3. Bolaji has no Kafka experience at all."
        )
    ]
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        answer, evidence, QUERY
    )
    wrong = answer.replace("AWS S3", "AWS EC2").replace("S3 bucket", "EC2 bucket")
    assert "requested_personal_production_scope:aws ec2" in constraint_failures(
        wrong, docs(), QUERY + " At Rintio"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Bolaji has explicit owner-confirmed absence of Kafka experience",
        "Bolaji has an absence of Kafka experience",
        "Bolaji has a lack of Kafka experience",
        "Bolaji has an absence of experience with Kafka",
        "He has confirmed absence of experience in Kafka",
    ],
)
def test_complete_nominal_absence_is_not_positive_experience(phrase):
    assert not confirmed_use_conflicts(phrase, docs())


@pytest.mark.parametrize(
    "phrase",
    [
        "Bolaji has experience with Kafka.",
        "Bolaji has an absence of doubts about Kafka experience.",
        "Bolaji has a lack of uncertainty about his Kafka experience.",
        "Bolaji has an absence of Kafka experience, but he operated Kafka.",
        "Bolaji has no absence of Kafka experience. He used Kafka.",
        "Bolaji has explicit owner-confirmed absence of Kafka experience and he deployed Kafka.",
        "Bolaji has an absence of Spark experience with Kafka.",
    ],
)
def test_absence_exception_cannot_hide_other_nominal_meanings_or_positive_use(phrase):
    assert "Kafka" in confirmed_use_conflicts(phrase, docs())


def test_terminal_qualifier_cannot_match_only_an_accepted_prefix():
    import re

    rule = policy_data()["request_scopes"][0]
    valid = "AWS S3 in production, with the employer or project for that S3 bucket remaining unknown"
    assert re.search(rule["unknown_project_qualification_pattern"], valid)
    assert not re.search(
        rule["unknown_project_qualification_pattern"], valid + " and he operated it"
    )
