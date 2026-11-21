"""Captured release failures plus explicitly synthetic partial drafts and boundaries."""

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
    confirmed_use_conflicts,
    constraint_failures,
    prepare_evidence,
    preserve_requested_evidence,
    retained_facet_indices,
    supplement_facets,
)

FIXTURES = Path(__file__).parent / "fixtures"
LOCAL = json.loads((FIXTURES / "local_1_2_11_failures.json").read_text())["records"]
SDK = json.loads((FIXTURES / "sdk_1_2_11_failures.json").read_text())["results"]
AWS, EXPERIENCE, WORKLOAD = LOCAL
GOV = SDK[1]


def evidence(record):
    response = record.get("response", record)
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in response["evidence"]
        ],
        record["question"],
    )


def test_synthetic_aws_only_draft_restores_the_missing_streaming_answer():
    draft = "Bolaji has no Kafka experience at all."
    docs = evidence(AWS)
    answer, changes = preserve_requested_evidence(draft, docs, AWS["question"], "en")
    assert "requested_streaming_production_inventory_missing" in changes
    assert (
        "At Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."
        in answer
    )
    assert preserve_requested_evidence(answer, docs, AWS["question"], "en") == (
        answer,
        [],
    )
    assert len(changes) <= 3


@pytest.mark.parametrize(
    "question",
    [
        "Which other streaming technologies has he personally operated besides Pub/Sub and Dataflow?",
        "Which streaming services has he personally operated, only Kafka?",
        "What AWS services has he personally operated?",
    ],
)
def test_streaming_restoration_does_not_answer_a_different_inventory(question):
    _, changes = preserve_requested_evidence(
        "A partial answer.", evidence(AWS), question, "en"
    )
    assert "requested_streaming_production_inventory_missing" not in changes


@pytest.mark.parametrize(
    "language,expected",
    [
        ("en", "Bolaji has no experience with Kafka."),
        ("fr", "Bolaji n'a aucune expérience avec Kafka."),
    ],
)
def test_synthetic_partial_documentation_limit_preserves_confirmed_non_use(
    language, expected
):
    draft = "Personal production deployment of Spark Structured Streaming is not documented."
    docs = evidence(EXPERIENCE)
    answer, changes = preserve_requested_evidence(
        draft, docs, EXPERIENCE["question"], language
    )
    assert "confirmed_non_use_requires_answer:Kafka" in changes
    assert expected in answer
    assert "no experience with Spark" not in answer
    assert "confirmed_non_use_requires_answer:Kafka" not in constraint_failures(
        answer, docs, EXPERIENCE["question"]
    )
    assert preserve_requested_evidence(
        answer, docs, EXPERIENCE["question"], language
    ) == (answer, [])


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji deployed Kafka in production.",
        "Bolaji operated Kafka. The employer for S3 is unknown.",
        "His Kafka pipeline was personally designed by Bolaji.",
        "Bolaji has experience with Kafka.",
        "Bolaji worked with Kafka in production.",
        "Bolaji ran Kafka in production.",
    ],
)
def test_confirmed_absence_cannot_be_appended_to_cancel_a_positive_claim(answer):
    docs = evidence(EXPERIENCE)
    assert confirmed_use_conflicts(answer, docs) == {"Kafka"}
    updated, changes = preserve_requested_evidence(
        answer, docs, EXPERIENCE["question"], "en"
    )
    assert "confirmed_non_use_requires_answer:Kafka" not in changes
    assert "confirmed_non_use_contradicted:Kafka" in constraint_failures(
        updated, docs, EXPERIENCE["question"]
    )
    assert "confirmed_non_use_contradicted:Kafka" in constraint_failures(
        updated + " Bolaji has no Kafka experience at all.",
        docs,
        EXPERIENCE["question"],
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji has never used Kafka.",
        "Bolaji has an explicit owner-confirmed absence of experience with Kafka and has never used it.",
        "Bolaji didn't deploy Kafka.",
        "It is not documented whether Bolaji deployed Kafka.",
        "Another engineer used Kafka.",
        "Kafka was operated by another engineer.",
        "If Bolaji used Kafka, that would meet the vacancy.",
    ],
)
def test_negation_hypotheticals_and_other_actors_are_not_personal_use(answer):
    assert not confirmed_use_conflicts(answer, evidence(EXPERIENCE))


def test_owner_confirmation_cannot_be_created_by_wrong_source_or_tool_inventory():
    docs = [
        Document(
            page_content="Kafka is a streaming service.",
            metadata={"source": "other.txt"},
        )
    ]
    answer, changes = preserve_requested_evidence(
        "A partial answer.", docs, EXPERIENCE["question"], "en"
    )
    assert "confirmed_non_use_requires_answer:Kafka" not in changes
    assert "no experience" not in answer


def test_actual_workload_failure_stays_rejected_after_correct_quotes_are_added():
    docs = evidence(WORKLOAD)
    answer, _ = preserve_requested_evidence(
        WORKLOAD["response"]["answer"], docs, WORKLOAD["question"], "en"
    )
    failures = constraint_failures(answer, docs, WORKLOAD["question"])
    assert "requested_streaming_workload_average_missing:false_withholding" in failures
    assert "latency_threshold_wording_requires_source" in failures
    assert all("sub-second" not in str(d.metadata.get("metric_scope")) for d in docs)


@pytest.mark.parametrize(
    "denial",
    [
        "The record does not document throughput measurements for these systems.",
        "No event volume measurements are documented for these systems.",
        "No measured throughput or event volume is documented for these systems.",
        "Throughput measurements for these systems are not documented.",
        "No throughput, end-to-end latency or availability measurements are documented for these systems.",
    ],
)
@pytest.mark.parametrize("denial_first", [True, False])
def test_broad_unknown_measurements_remain_a_conflict_in_either_order(
    denial, denial_first
):
    docs = evidence(WORKLOAD)
    known, _ = preserve_requested_evidence(
        "The separate streaming workload averaged 45 million events per day.",
        docs,
        WORKLOAD["question"],
        "en",
    )
    answer = denial + " " + known if denial_first else known + " " + denial
    assert (
        "requested_streaming_workload_average_missing:false_withholding"
        in constraint_failures(answer, docs, WORKLOAD["question"])
    )


@pytest.mark.parametrize(
    "limit",
    [
        "Maximum throughput capacity is not documented for these systems.",
        "No end-to-end latency or availability guarantees are documented for the MongoDB CDC pipeline.",
        "The individual MongoDB CDC pipeline's throughput is not documented.",
        'The statement "No throughput measurements are documented for these systems" is incorrect.',
        "If no throughput measurements are documented for these systems, do not estimate them.",
    ],
)
def test_local_unknowns_and_rejected_denials_do_not_erase_workload_averages(limit):
    failures = constraint_failures(limit, evidence(WORKLOAD), WORKLOAD["question"])
    assert (
        "requested_streaming_workload_average_missing:false_withholding" not in failures
    )


def test_unsupported_latency_threshold_is_rejected_even_in_a_negative_claim():
    docs = evidence(WORKLOAD)
    assert "latency_threshold_wording_requires_source" in constraint_failures(
        "Sub-second latency is not established.", docs
    )
    actual = [Document(page_content="Sub-second latency is not documented.")]
    assert "latency_threshold_wording_requires_source" not in constraint_failures(
        "Sub-second latency is not established.", actual
    )


def test_fallback_restoration_delivers_facts_without_internal_section_labels():
    answer, changes = preserve_requested_evidence(
        "The separate streaming workload averaged 45 million events per day.",
        evidence(WORKLOAD),
        WORKLOAD["question"],
        "en",
    )
    assert "requested_batch_workload_measurement_missing" in changes
    assert "5 TB" in answer
    assert "Workload and measurement scope" not in answer
    assert "Separate batch workload measurement" not in answer
    assert ".txt" not in answer


def test_governance_known_user_outcomes_do_not_cancel_a_false_withholding():
    docs = evidence(GOV)
    answer, changes = preserve_requested_evidence(
        GOV["answer"], docs, GOV["question"], "en"
    )
    assert set(changes) >= {
        "enterprise_governance_single_truth_missing",
        "enterprise_governance_user_access_outcome_missing",
    }
    assert (
        "Selected senior users could query enterprise data using natural language within their existing Google Cloud IAM permissions."
        in answer
    )
    assert (
        "enterprise_governance_user_access_outcome_missing:false_withholding"
        in constraint_failures(answer, docs, GOV["question"])
    )


@pytest.mark.parametrize(
    "suffix",
    [
        "The country-specific negotiation process is not documented.",
        "Quantitative user outcomes are not documented.",
        "User adoption counts are not documented.",
        "For the MongoDB CDC pipeline, user outcomes are unknown.",
        "User outcomes are unknown for the MongoDB CDC pipeline.",
        "For the separate Snowflake platform, user outcomes are unknown.",
        "For the separate Snowflake reporting platform, user outcomes are unknown.",
        "User outcomes are unknown for the separate Snowflake reporting platform.",
    ],
)
def test_known_governance_outcomes_coexist_with_specific_unknowns(suffix):
    draft = "He introduced MCP-enabled BigQuery integration in Claude. " + suffix
    docs = evidence(GOV)
    answer, changes = preserve_requested_evidence(draft, docs, GOV["question"], "en")
    assert "enterprise_governance_user_access_outcome_missing" in changes
    assert not any(
        "false_withholding" in f
        for f in constraint_failures(answer, docs, GOV["question"])
    )
    assert preserve_requested_evidence(answer, docs, GOV["question"], "en") == (
        answer,
        [],
    )


@pytest.mark.parametrize(
    "question,draft",
    [
        (
            "What governance and access safeguards changed for Data Hub reporting users?",
            "He built Data Hub RBAC.",
        ),
        (
            "What quality and access safeguards changed for observability reporting users?",
            "He built pipeline observability.",
        ),
        (GOV["question"], "He built the Metabase platform."),
        (GOV["question"], "He used Claude to write code for the Metabase platform."),
        (GOV["question"], "For the Data Hub, he introduced governed AI access."),
        (GOV["question"], "He built Data Hub metrics governance using Claude."),
    ],
)
def test_governance_outcomes_never_transfer_to_another_implementation(question, draft):
    answer, changes = preserve_requested_evidence(draft, evidence(GOV), question, "en")
    assert not any(c.startswith("enterprise_governance_") for c in changes)
    assert "Selected senior users" not in answer


@pytest.mark.parametrize("record", [AWS, GOV])
def test_wrong_sources_cannot_supply_restored_facts(record):
    docs = evidence(record)
    for doc in docs:
        doc.metadata["source"] = "other.txt"
    _, changes = preserve_requested_evidence(
        "A partial answer.", docs, record["question"], "en"
    )
    assert not changes


def test_actual_governance_withholding_needs_repair_and_independent_verification():
    docs = evidence(GOV)
    corrected, _ = preserve_requested_evidence(
        "He introduced MCP-enabled BigQuery integration in Claude.",
        docs,
        GOV["question"],
        "en",
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=corrected),
        GroundingVerdict(is_grounded=True),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": GOV["answer"],
                "query": GOV["question"],
                "context_documents": docs,
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert result["grounding_checked"] is True
    assert llm.with_structured_output.return_value.invoke.call_count == 2
    assert "does not specify what changed" not in result["answer"]


@pytest.mark.parametrize(
    "record,answer",
    [
        (GOV, GOV["answer"]),
        (WORKLOAD, WORKLOAD["response"]["answer"]),
        (
            EXPERIENCE,
            "Bolaji operated Kafka in production. Bolaji has no Kafka experience at all.",
        ),
    ],
)
def test_positive_verdict_and_unchanged_repair_cannot_override_hard_conflicts(
    record, answer
):
    verifier, repair = Mock(), Mock()
    verifier.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True, corrected_answer=answer
    )
    repair.with_structured_output.return_value.invoke.return_value = GeneratedAnswer(
        answer=answer
    )
    with patch("app.graph.nodes._get_llm", side_effect=[verifier, repair]) as factory:
        result = verify_grounding_node(
            {
                "answer": answer,
                "query": record["question"],
                "context_documents": evidence(record),
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert result["grounding_checked"] is False
    assert result["answer"] != answer
    assert factory.call_count == 2
    assert verifier.with_structured_output.return_value.invoke.call_count == 1


@pytest.mark.parametrize(
    "record,source,section,body",
    [
        (
            SDK[0],
            "10_apps_and_projects.txt",
            "iBOLA - AI-POWERED PORTFOLIO CHATBOT",
            "A production-grade agentic RAG chatbot serving as a professional portfolio assistant. LangGraph architecture.",
        ),
        (
            GOV,
            "90_website_canon_llms_full.txt",
            "Enterprise metrics governance & AI-ready semantic layer",
            "An enterprise metrics governance framework. MCP-enabled BigQuery integration in Claude.",
        ),
    ],
)
def test_exact_sections_survive_grading_and_context_without_expanding_budgets(
    record, source, section, body
):
    doc = Document(
        page_content=body, metadata={"source": source, "section_header": section}
    )
    docs = supplement_facets(
        Mock(documents=[doc]), record["question"], [], protected_only=True
    )
    assert retained_facet_indices(docs, record["question"]) == {0}
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = BatchGradeResult(
        relevant_indices=[]
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        graded = grade_documents_node({"query": record["question"], "documents": docs})[
            "graded_documents"
        ]
    assert _select_context_docs(graded, record["question"]) == docs
    doc.metadata["section_header"] = "Different project"
    assert not supplement_facets(
        Mock(documents=[doc]), record["question"], [], protected_only=True
    )
