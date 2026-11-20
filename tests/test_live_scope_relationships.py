"""Relationships missed in the v1.2.7 actual-prefix live recruiter replay."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import grade_documents_node, verify_grounding_node
from app.graph.state import BatchGradeResult, GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    retained_facet_indices,
    supplement_facets,
)
from scripts.run_eval import check_facts, load_golden

CASES = json.loads(
    (Path(__file__).parent / "fixtures/live_scope_relationships.json").read_text()
)
GOOD = {
    "recruiter-adi-05-t5": "Bolaji a été le premier recrutement data en 2019. L'équipe initiale a grandi jusqu'à 4 personnes pendant sa période de Global Data Analyst. L'équipe est ensuite passée de 4 à plus de 14 personnes durant le mandat Head of Data, d'octobre 2022 à mars 2025. Le périmètre couvre cinq marchés opérationnels africains. Singapour était uniquement le siège de l'entreprise.",
    "recruiter-sda-05-t2": "Bolaji architected and led the Gozem Data Hub; its documented stack includes AWS EC2. The retrieved inventory lists AWS S3 and streaming tools. Personal production operation of EC2, S3, Pub/Sub, Dataflow, Spark, Apache NiFi and Lakeflow is not established by this evidence. Personal Kafka architecture delivery is not documented.",
    "recruiter-sde-03-t4": "A separate streaming workload averaged 45 million telemetry and operational events per day. That is not an individual MongoDB CDC pipeline count, throughput ceiling, sub-second latency or availability guarantee. Those CDC measurements are not established. The Vodun Days command center provided minute-level Operations visibility; that is a separate system.",
}


def documents(case):
    return [Document(**d) for d in case["documents"]]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_captured_relationship_failures_and_independent_correct_controls(case):
    failures = constraint_failures(case["answer"], documents(case), case["question"])
    assert failures
    row = next(r for r in load_golden() if r["id"] == case["id"])
    assert not check_facts(
        case["answer"],
        row,
        result={
            "grounding_checked": True,
            "agent_type": "professional",
            "evidence": case["documents"],
        },
    )[0]
    if case["id"].startswith("recruiter-adi"):
        assert "first_hire_title_requires_explicit_role" in failures
        assert "initial_team_growth_stage_missing" in failures
    if case["id"].startswith("recruiter-sda"):
        assert "requested_personal_production_scope:aws s3" in failures
        assert "requested_personal_production_scope:aws ec2" in failures
        for tool in ("pub/sub", "dataflow", "spark", "apache nifi", "lakeflow"):
            assert "requested_personal_production_scope:" + tool in failures
    if case["id"].startswith("recruiter-sde"):
        assert "requested_streaming_workload_average_missing" in failures
        assert "system_visibility_requires_own_evidence" in failures
    assert not constraint_failures(GOOD[case["id"]], documents(case), case["question"])
    row = next(r for r in load_golden() if r["id"] == case["id"])
    assert check_facts(
        GOOD[case["id"]],
        row,
        result={
            "grounding_checked": True,
            "evidence": case["documents"],
            "agent_type": "professional",
        },
    )[0]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("recheck", [True, False])
def test_wrong_approval_or_unchanged_correction_cannot_release_captured_answer(
    case, recheck
):
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=case["answer"]),
        GeneratedAnswer(answer=GOOD[case["id"]], confidence=0.8),
        GroundingVerdict(is_grounded=recheck),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "query": case["question"],
                "answer": case["answer"],
                "agent_type": "professional",
                "user_language": case["language"],
                "confidence": 0.9,
                "context_documents": documents(case),
            }
        )
    assert result["grounding_checked"] is recheck
    assert result["answer"] != case["answer"]
    assert invoke.call_count <= 3
    if recheck:
        assert not constraint_failures(
            result["answer"], documents(case), case["question"]
        )


@pytest.mark.parametrize(
    "answer,evidence,question,bad",
    [
        (
            "He was the first Data Engineer in 2019.",
            "He was the first data hire in 2019, as Global Data Analyst.",
            "What was his first role?",
            True,
        ),
        (
            "He was the first Data Engineer.",
            "He was the first Data Engineer.",
            "What was his first role?",
            False,
        ),
        (
            "He operated AWS S3 in production.",
            "Tools: AWS S3. He used EC2 in production.",
            "Which services did he personally operate in production?",
            True,
        ),
        (
            "He operated AWS S3 in production.",
            "He operated AWS S3 in production.",
            "Which services did he personally operate in production?",
            False,
        ),
        (
            "He has experience with Pub/Sub and Lakeflow.",
            "Tool inventory: Pub/Sub, Lakeflow.",
            "Which streaming tools did he personally operate in production?",
            True,
        ),
        (
            "Listed tools: Pub/Sub, Lakeflow. Their personal production operation is unverified.",
            "Tool inventory: Pub/Sub, Lakeflow.",
            "Which tools did he personally operate in production?",
            False,
        ),
        (
            "The CDC pipeline gave minute-level Operations visibility.",
            "The Vodun command center gave minute-level Operations visibility. He built a CDC pipeline.",
            "Which latency measurements exist?",
            True,
        ),
        (
            "Minute-level visibility was delivered by the CDC pipeline.",
            "The Vodun command center gave minute-level visibility. He built a CDC pipeline.",
            "Which latency measurements exist?",
            True,
        ),
        (
            "The CDC pipeline gave minute-level visibility.",
            "The CDC pipeline gave minute-level visibility.",
            "What is known?",
            False,
        ),
        (
            "Minute-level visibility belongs to the Vodun command center, not the CDC pipeline.",
            "The Vodun command center provided minute-level visibility.",
            "What is known?",
            False,
        ),
    ],
)
def test_relationship_scope_mutations_and_positive_controls(
    answer, evidence, question, bad
):
    assert (
        bool(constraint_failures(answer, [Document(page_content=evidence)], question))
        is bad
    )


def test_exact_requested_facets_survive_incorrect_grading_but_unmatched_flags_do_not():
    corpus = [
        Document(
            page_content="He built a MongoDB CDC pipeline publishing events through Pub/Sub and processing them in Dataflow.",
            metadata={"source": "90_website_canon_llms_full.txt"},
        )
    ]
    query = "Which streaming technologies did he personally design in production?"
    selected = prepare_evidence(
        supplement_facets(Mock(documents=corpus), query, []), query
    )
    assert selected and retained_facet_indices(selected, query) == {0}
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = BatchGradeResult(
        relevant_indices=[]
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = grade_documents_node({"query": query, "documents": selected})
    assert result["graded_documents"] == selected
    invalid = Document(
        page_content="Unrelated inventory.", metadata=selected[0].metadata
    )
    assert retained_facet_indices([invalid], query) == set()
    assert retained_facet_indices(selected, "What languages does he speak?") == set()


@pytest.mark.parametrize(
    "answer",
    [
        "EC2 operation is unverified, he operated AWS S3 in production.",
        "EC2 operation is unknown while AWS S3 was personally operated in production.",
        "S3 use is unverified, which demonstrates experience with Dataflow.",
    ],
)
def test_unknown_for_one_claim_cannot_qualify_another_operation(answer):
    assert constraint_failures(
        answer,
        [Document(page_content="Inventory: AWS S3, Dataflow.")],
        "Which services did he personally operate in production?",
    )


@pytest.mark.parametrize(
    "answer,evidence,bad",
    [
        (
            "He personally operated AWS S3 in production. Personal AWS S3 production operation is unverified.",
            "Inventory: AWS S3.",
            True,
        ),
        (
            "He personally operated AWS S3 in production alongside unverified Lakeflow.",
            "Inventory: AWS S3, Lakeflow.",
            True,
        ),
        (
            "Bolaji personally operates AWS EC2 in production.",
            "Bolaji personally operates AWS EC2 in production.",
            False,
        ),
        (
            "He personally operated AWS S3 in production at the Data Hub.",
            "He personally operated AWS S3 in production at Rintio.",
            True,
        ),
        (
            "He personally operated AWS S3 in production at the Data Hub.",
            "He personally operated AWS S3 in production at the Data Hub.",
            False,
        ),
        (
            "The unrelated prototype project documented stack includes AWS EC2.",
            "Bolaji architected and led the Gozem Data Hub. The published case documents AWS EC2.",
            True,
        ),
        (
            "AWS EC2 is documented as part of the Gozem Data Hub.",
            "Bolaji architected and led the Gozem Data Hub. The published case documents AWS EC2.",
            False,
        ),
    ],
)
def test_reviewed_proof_qualification_project_and_tense_controls(answer, evidence, bad):
    assert (
        bool(
            constraint_failures(
                answer,
                [Document(page_content=evidence)],
                "Which services did he personally operate in production?",
            )
        )
        is bad
    )


def test_short_followup_keeps_standalone_production_requirements_in_verification():
    case = next(c for c in CASES if c["id"] == "recruiter-sda-05-t2")
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=GOOD[case["id"]]),
        GroundingVerdict(is_grounded=True),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "original_query": "And AWS S3?",
                "query": case["question"],
                "answer": "He personally operated AWS S3 in production.",
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": documents(case),
            }
        )
    assert result["grounding_checked"]
    assert result["answer"] == GOOD[case["id"]]
    assert invoke.call_count == 2


def test_project_stack_reference_requires_immediate_unambiguous_project():
    doc = Document(
        page_content="Bolaji architected and led the Gozem Data Hub. The published case documents AWS EC2."
    )
    q = "Which services did he personally operate in production?"
    assert not constraint_failures(
        "Bolaji architected and led the Gozem Data Hub; its documented stack includes AWS EC2.",
        [doc],
        q,
    )
    assert constraint_failures(
        "Bolaji architected and led the Gozem Data Hub. He also built an unrelated prototype. Its documented stack includes AWS EC2.",
        [doc],
        q,
    )


def test_present_operation_is_not_cleared_by_another_tools_unknown():
    assert constraint_failures(
        "AWS EC2 operation is unverified, he operates AWS S3 in production.",
        [Document(page_content="Inventory: AWS EC2, AWS S3.")],
        "Which services did he personally operate in production?",
    )


def test_verification_restoration_language_comes_from_original_user_message():
    from app.graph.nodes import _detect_reply_language

    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    with (
        patch("app.graph.nodes._get_llm", return_value=llm),
        patch(
            "app.graph.nodes._detect_reply_language", wraps=_detect_reply_language
        ) as detect,
    ):
        result = verify_grounding_node(
            {
                "original_query": "And AWS S3?",
                "query": "Quels services a-t-il personnellement exploites en production ?",
                "answer": "Personal AWS S3 production operation is unverified.",
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": [Document(page_content="Inventory: AWS S3.")],
            }
        )
    assert result["grounding_checked"]
    assert detect.call_args_list and all(
        c.args[0] == "And AWS S3?" for c in detect.call_args_list
    )


def test_explicit_personal_production_stack_keeps_its_tools_and_project():
    source = Document(
        page_content="At Rintio, Bolaji built a production Hadoop and Apache Spark data stack with Elasticsearch to process agricultural data."
    )
    q = "Which tools did he personally use in production?"
    assert not constraint_failures(
        "At Rintio, Bolaji built a production Apache Spark data stack.", [source], q
    )
    assert constraint_failures(
        "At the Gozem Data Hub, Bolaji personally operated Spark in production.",
        [source],
        q,
    )
