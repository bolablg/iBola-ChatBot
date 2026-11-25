"""Knowledge-source URLs cannot be presented as public artifact destinations."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import _evidence_context, verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import constraint_failures
from tests.prompt_fields import prompt_field

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/local_1_2_11_kb_url_leak.json").read_text()
)
RECORD = CAPTURE["record"]
BAD = RECORD["answer"]
RULE = "knowledge_snapshot_url_not_public_reference"
GOOD = "The documented project page for Bolaji's advanced Dataform framework is https://www.bolablg.com/projects/dataform-modeling-lab/. The retrieved evidence does not establish a separate public article URL."


def docs():
    return [
        Document(
            page_content=e["content_preview"],
            metadata={
                **e,
                "section_header": e["section"],
                **{
                    key: value
                    for key, value in CAPTURE["captured_context_metadata"]
                    .get(e["chunk_id"], {})
                    .items()
                    if value is not None
                },
            },
        )
        for e in RECORD["evidence"]
    ]


def test_actual_wrong_article_destination_is_rejected_despite_correct_project_link():
    assert RULE in constraint_failures(BAD, docs(), RECORD["question"])
    assert CAPTURE["captured_candidates"][-1]["result"]["is_grounded"]
    assert RECORD["scores"]["accuracy"] < 8


def test_model_context_retains_documented_artifact_and_identity_but_not_raw_source_url():
    evidence = docs()
    before = copy.deepcopy(evidence)
    assert any(
        (d.metadata.get("canonical_url") or "").endswith("/llms-full.txt")
        for d in evidence
    )
    context = _evidence_context(evidence)
    assert "canonical_url" not in context
    assert "https://www.bolablg.com/llms-full.txt" not in context
    assert "https://www.bolablg.com/projects/dataform-modeling-lab/" in context
    assert all(d.metadata["chunk_id"] in context for d in evidence)
    assert evidence == before


@pytest.mark.parametrize("path", ["llms.txt", "llms-full.txt"])
@pytest.mark.parametrize(
    "form",
    [
        "https://www.bolablg.com/{path}",
        "https://www.bolablg.com/{path}.",
        "https://www.bolablg.com/{path}?download=1#article",
        "[Article](https://www.bolablg.com/{path})",
        "<https://www.bolablg.com/{path}>",
        "www.bolablg.com/{path}",
        "/{path}",
        "[article]: https://www.bolablg.com/{path}",
    ],
)
def test_snapshot_destinations_are_unconditional_even_without_evidence(path, form):
    assert RULE in constraint_failures(form.format(path=path), [], "Anything")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.bolablg.com/projects/dataform-modeling-lab/",
        "https://example.org/blog/dataform/",
        "https://github.com/example/framework",
        "https://example.org/blog/llms-full.txt-guide/",
        "https://example.org/llms.txt.example",
    ],
)
def test_normal_artifact_and_guide_destinations_are_not_snapshot_paths(url):
    assert RULE not in constraint_failures(url, [], "Public reference")


@pytest.mark.parametrize("accepted", [True, False])
def test_model_approval_cannot_override_guard_and_repair_requires_final_exact_verdict(
    accepted,
):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = [
        GroundingVerdict(is_grounded=True),
        GeneratedAnswer(answer=GOOD, confidence=1),
        GroundingVerdict(is_grounded=accepted),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": BAD,
                "original_query": RECORD["question"],
                "query": RECORD["question"],
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": docs(),
            }
        )
    calls = llm.with_structured_output.return_value.invoke.call_args_list
    assert len(calls) == 3
    assert prompt_field(calls[-1].args[0][-1].content, "draft_answer") == GOOD
    assert result["grounding_checked"] is accepted
    assert result["answer"] == (
        GOOD if accepted else INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    )
    assert "llms-full.txt" not in result["answer"]
