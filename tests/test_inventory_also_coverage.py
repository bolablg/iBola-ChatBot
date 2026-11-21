"""A real inventory continuation cannot be relabeled as additional services."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INVENTORY_FOLLOWUP_TEMPLATES
from app.services.service_inventory import inventory_followup

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/local_1_2_11_inventory_also.json").read_text()
)["records"]


def state(records=CAPTURE):
    return {
        "original_query": records[1]["question"],
        "query": records[1]["question"],
        "chat_history": [(records[0]["question"], records[0]["response"]["answer"])],
        "agent_type": "professional",
        "user_language": "en",
        "context_documents": [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in records[1]["response"]["evidence"]
        ],
    }


def test_actual_two_turn_followup_recognizes_all_fourteen_previously_listed_labels():
    s = state()
    answer = inventory_followup(s, s["context_documents"], "en")
    assert answer == INVENTORY_FOLLOWUP_TEMPLATES["en"]["covered"].format(
        scope="production-service"
    )
    proof = next(
        d.metadata["conversation_coverage"]
        for d in s["context_documents"]
        if d.metadata.get("conversation_coverage")
    )
    assert len(proof["previously_listed_labels"]) == 14
    assert proof["remaining_labels"] == []
    s["answer"] = answer
    with patch("app.graph.nodes._get_llm") as llm:
        result = verify_grounding_node(s)
    llm.assert_not_called()
    assert result["grounding_checked"] and result["answer"] == answer
    assert result["verification_method"] == "retrieved_inventory_conversation_coverage"


@pytest.mark.parametrize(
    "prefix",
    [
        "His production service inventory does not include",
        "If his production service inventory also includes",
        '"His production service inventory also includes',
        "Another engineer's production service inventory also includes",
        "His production service inventory might also include",
    ],
)
def test_nonaffirmative_or_other_actor_inventory_cannot_mark_those_labels_as_covered(
    prefix,
):
    s = state()
    question, prior = s["chat_history"][0]
    s["chat_history"] = [
        (
            question,
            prior.replace("His production service inventory also includes", prefix),
        )
    ]
    answer = inventory_followup(s, s["context_documents"], "en")
    assert "Additional" in answer
    proof = next(
        d.metadata["conversation_coverage"]
        for d in s["context_documents"]
        if d.metadata.get("conversation_coverage")
    )
    assert len(proof["previously_listed_labels"]) == 4
    assert len(proof["remaining_labels"]) == 10


PREAMBLES = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_inventory_preambles.json"
    ).read_text()
)["records"]


def test_actual_project_and_discourse_preambles_keep_ec2_s3_already_covered():
    s = state(PREAMBLES)
    answer = inventory_followup(s, s["context_documents"], "en")
    assert "already covered" in answer
    proof = next(
        d.metadata["conversation_coverage"]
        for d in s["context_documents"]
        if d.metadata.get("conversation_coverage")
    )
    assert len(proof["previously_listed_labels"]) == 14
    assert proof["remaining_labels"] == []


@pytest.mark.parametrize(
    "replacement",
    [
        "For the Gozem Data Hub, another engineer personally designed and operated AWS EC2.",
        "If he personally designed and operated AWS EC2 for the Gozem Data Hub, he could help.",
        '"For the Gozem Data Hub, he personally designed and operated AWS EC2."\n',
        "For the Gozem Data Hub, he did not personally operate AWS EC2.",
    ],
)
def test_original_clause_exclusions_survive_prefix_normalization(replacement):
    s = state(PREAMBLES)
    q, prior = s["chat_history"][0]
    s["chat_history"] = [
        (
            q,
            prior.replace(
                "For the Gozem Data Hub, he personally designed and operated AWS EC2.",
                replacement,
            ),
        )
    ]
    answer = inventory_followup(s, s["context_documents"], "en")
    assert "Additional" in answer and "AWS EC2" in answer
    assert "AWS S3" not in answer


def test_comparison_prefix_mentions_are_not_counted_as_listed_service_objects():
    s = state(PREAMBLES)
    q, prior = s["chat_history"][0]
    s["chat_history"] = [
        (
            q,
            prior.replace(
                "For the Gozem Data Hub, he personally designed and operated AWS EC2.",
                "For comparison with AWS EC2, he personally operated AWS S3.",
            ),
        )
    ]
    answer = inventory_followup(s, s["context_documents"], "en")
    assert "Additional" in answer and "AWS EC2" in answer
    assert "AWS S3" not in answer
    proof = next(
        d.metadata["conversation_coverage"]
        for d in s["context_documents"]
        if d.metadata.get("conversation_coverage")
    )
    assert proof["remaining_labels"] == ["AWS EC2"]
