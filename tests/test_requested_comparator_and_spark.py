"""Actual local comparator refusal and missing-current-proof Spark follow-up."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import (
    _select_context_docs,
    grade_documents_node,
    verify_grounding_node,
)
from app.graph.state import BatchGradeResult, GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    retained_facet_indices,
    supplement_facets,
)
from scripts.run_eval import _confirmed_experience_limit, check_facts
from tests.prompt_fields import prompt_field

CAPTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/local_1_2_11_comparator_and_spark.json"
    ).read_text()
)
COMPARATOR = CAPTURE["comparator"]
SPARK = CAPTURE["spark"]
RULE = "latency_threshold_wording_requires_source"


def docs(record):
    result = record.get("response", record)
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in result["evidence"]
        ],
        record["question"],
    )


def test_requested_comparator_can_be_unknown_without_becoming_source_evidence():
    # Synthetic valid draft; the captured API refusal does not expose its draft.
    answer = "Bolaji designed the MongoDB CDC pipeline using Pub/Sub and Dataflow. Its end-to-end latency is not documented, so a comparison with sub-second streaming cannot be established."
    evidence = docs(COMPARATOR)
    assert RULE not in constraint_failures(answer, evidence, COMPARATOR["question"])
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": COMPARATOR["question"],
                "answer": answer,
                "context_documents": evidence,
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert result["grounding_checked"]
    assert llm.with_structured_output.return_value.invoke.call_count == 1
    submitted = prompt_field(
        llm.with_structured_output.return_value.invoke.call_args.args[0][-1].content,
        "draft_answer",
    )
    assert submitted.startswith(answer)
    assert submitted == result["answer"]


@pytest.mark.parametrize(
    "question",
    [
        COMPARATOR["question"],
        "What latency is documented?",
        'Do not claim "sub-second" latency.',
        "What is documented?\nPrior user: compare it with sub-second streaming.",
    ],
)
def test_user_benchmark_never_proves_false_literal_source_attribution(question):
    answer = (
        "The documentation explicitly notes that sub-second latency is not established."
    )
    assert RULE in constraint_failures(answer, docs(COMPARATOR), question)


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji achieved sub-second latency for the MongoDB CDC pipeline.",
        "Its latency is unknown. The MongoDB CDC pipeline nevertheless delivered sub-second latency.",
        "The MongoDB CDC pipeline had the public-event command center's minute-level end-to-end latency.",
    ],
)
def test_unproven_capability_or_wrong_system_still_requires_final_source_verification(
    answer,
):
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=False),
        GeneratedAnswer(answer=answer),
        GroundingVerdict(is_grounded=False),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": COMPARATOR["question"],
                "answer": answer,
                "context_documents": docs(COMPARATOR),
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert not result["grounding_checked"]
    assert invoke.call_count <= 3


def spark_proof():
    text = Path("data/04_previous_roles_early.txt").read_text()
    body = text.split("Rintio is a technology company", 1)[1].split("\n---", 1)[0]
    return Document(
        page_content="Rintio is a technology company" + body,
        metadata={
            "source": "04_previous_roles_early.txt",
            "section_header": "DATA SCIENTIST - RINTIO",
        },
    )


def test_generic_spark_proof_is_retrieved_into_the_current_turn_and_survives_grading():
    row = SPARK["result"]
    current = docs(row)
    assert not any("100GB+" in d.page_content for d in current)
    selected = supplement_facets(
        Mock(documents=[spark_proof(), *current]),
        row["question"],
        current,
        protected_only=True,
    )
    assert (
        selected[0].metadata["required_facet"]
        == "generic_spark_implementation_comparison"
    )
    assert retained_facet_indices(selected, row["question"])
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = BatchGradeResult(
        relevant_indices=[]
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        graded = grade_documents_node(
            {"query": row["question"], "documents": selected}
        )["graded_documents"]
    final_docs = _select_context_docs(graded, row["question"])
    assert any(
        "100GB+" in d.page_content
        and d.metadata["source"] == "04_previous_roles_early.txt"
        for d in final_docs
    )
    assert not any(
        "Spark Structured Streaming" in d.page_content
        for d in final_docs
        if d.metadata["source"] == "04_previous_roles_early.txt"
    )


@pytest.mark.parametrize(
    "kind", ["source", "section", "support", "unrelated", "history_only"]
)
def test_generic_spark_facet_cannot_be_promoted_from_wrong_or_irrelevant_records(kind):
    doc = spark_proof()
    query = SPARK["result"]["question"]
    if kind == "source":
        doc.metadata["source"] = "other.txt"
    elif kind == "section":
        doc.metadata["section_header"] = "Unrelated project"
    elif kind == "support":
        doc.page_content = "Spark is a tool."
    else:
        query = "What latency is documented?" + (
            "\nEarlier user: Spark production experience"
            if kind == "history_only"
            else ""
        )
    assert not supplement_facets(Mock(documents=[doc]), query, [], protected_only=True)


def test_exact_documented_deployment_gap_is_recognized_only_after_retrieval_and_grounding():
    row = SPARK["result"]
    assert _confirmed_experience_limit(row["answer"], "Spark Structured Streaming")
    assert check_facts(row["answer"], SPARK["golden"], result=row)[0]
    # The historical faithfulness failure remains real: this recognizer fixes
    # grammar only. The retained-facet test separately repairs missing proof.
    assert row["scores"]["faithfulness"] == 4
    for override in ({"grounding_checked": False}, {"evidence": []}):
        assert not check_facts(
            row["answer"], SPARK["golden"], result={**row, **override}
        )[0]


@pytest.mark.parametrize(
    "answer",
    [
        "If there is no documented production experience deploying Spark Structured Streaming, ask him.",
        "There is no production experience deploying Spark Structured Streaming.",
        "There is no documented production experience deploying Spark.",
        "There is no documented production experience deploying Spark Structured Streaming and Dataflow.",
        "There is no documented production experience deploying Spark Structured Streaming, though he deployed it.",
        "There is no documented production experience deploying Spark Structured Streaming, though Spark Structured Streaming is documented for a project.",
        "There is no documented production experience deploying Spark Structured Streaming, though Apache Spark is documented for a project and he operated Spark Structured Streaming.",
        "There is no documented production experience deploying Spark Structured Streaming. He deployed Spark Structured Streaming.",
    ],
)
def test_deployment_limit_cannot_hide_conditionals_different_tools_or_extra_claims(
    answer,
):
    assert not _confirmed_experience_limit(answer, "Spark Structured Streaming")
