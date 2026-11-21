"""The strongest customer analytics example is one coherent source record."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import _select_context_docs, generate_node, verify_grounding_node
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    focus_single_case,
    prepare_evidence,
    supplement_facets,
)

CASE = json.loads(
    (Path(__file__).parent / "fixtures/sdk_1_2_11_customer_case_merge.json").read_text()
)
SECTION = "Customer intelligence & lifecycle analytics"


def docs():
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in CASE["evidence"]
        ],
        CASE["question"],
    )


def promoted():
    original = docs()
    return supplement_facets(
        Mock(documents=original), CASE["question"], original, protected_only=True
    )


def test_actual_merge_capture_focuses_one_validated_customer_platform_record():
    assert CASE["scores"]["faithfulness"] == 6
    selected = _select_context_docs(
        promoted(), CASE["question"], original_query=CASE["question"]
    )
    assert selected and {d.metadata["section_header"] for d in selected} == {SECTION}
    assert all(
        "9%" not in d.page_content and "SEASONAL DRIVER" not in d.page_content
        for d in selected
    )
    assert "RFM segmentation" in selected[0].page_content


@pytest.mark.parametrize(
    "question",
    [
        "Compare his customer analytics platform with the seasonal driver-retention case.",
        "Give several customer analytics examples.",
        "Give his strongest customer analytics example and contrast it with his retention work.",
        "Give his strongest customer analytics example with a quantified outcome.",
        "What is his strongest customer analytics example about agricultural churn?",
        "What is his strongest customer analytics example with a 9% retention outcome?",
        "Earlier I asked for his strongest customer analytics example. Now list all projects.",
        "Give one customer analytics example and another project.",
        "What production services did he personally operate?",
    ],
)
def test_comparison_other_named_project_plural_and_historical_requests_keep_context(
    question,
):
    original = promoted()
    assert focus_single_case(original, question, original_query=question) == original


@pytest.mark.parametrize("invalid", ["source", "section", "support", "retention"])
def test_customer_focus_requires_validated_source_section_and_body(invalid):
    documents = promoted()
    doc = next(
        d
        for d in documents
        if d.metadata.get("required_facet")
        == "single_customer_intelligence_implementation"
    )
    if invalid == "source":
        doc.metadata["source"] = "unrelated.txt"
    elif invalid == "section":
        doc.metadata["section_header"] = "Other case"
    elif invalid == "support":
        doc.page_content = "An unrelated record."
    else:
        doc.metadata["retain_requested_facet"] = False
    assert focus_single_case(documents, CASE["question"]) == documents


def test_generation_and_exact_corrected_verification_share_only_the_customer_case():
    answer = "Bolaji designed and built a customer intelligence platform combining rider and driver profiling, RFM segmentation, and cohort analysis. Marketing used its segments and behavioral insights to tailor acquisition, retention and re-engagement strategies."
    generator, verifier = Mock(), Mock()
    generator.with_structured_output.return_value.invoke.return_value = GeneratedAnswer(
        answer=answer
    )
    verifier.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    state = {
        "original_query": CASE["question"],
        "query": CASE["question"],
        "graded_documents": docs(),
        "agent_type": "professional",
        "user_language": "en",
    }
    with (
        patch(
            "app.graph.nodes._get_hybrid_search", return_value=Mock(documents=docs())
        ),
        patch("app.graph.nodes._get_llm", side_effect=[generator, verifier]),
    ):
        generated = generate_node(state)
        result = verify_grounding_node({**state, **generated})
    assert result["grounding_checked"]
    assert {d.metadata["section_header"] for d in generated["context_documents"]} == {
        SECTION
    }
    content = verifier.with_structured_output.return_value.invoke.call_args.args[0][
        -1
    ].content
    assert "SEASONAL DRIVER-RETENTION" not in content and "9%" not in content
    assert content.rsplit("Answer to verify:\n", 1)[1] == answer
