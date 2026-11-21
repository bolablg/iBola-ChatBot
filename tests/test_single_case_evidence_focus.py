"""Captured cross-case synthesis, S3 record wording, and provider500 regression."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import (
    _grounding_provider_calls,
    _invoke_with_retry,
    _is_transient_llm_error,
    _provider_generate,
    _select_context_docs,
    generate_node,
    verify_grounding_node,
)
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    focus_single_case,
    policy_data,
    prepare_evidence,
    supplement_facets,
)

CAPTURE = json.loads(
    (
        Path(__file__).parent
        / "fixtures/sdk_1_2_11_case_scope_and_provider_failures.json"
    ).read_text()
)
CASES = {r["id"]: r for r in CAPTURE["results"]}
GOV = CASES["recruiter-hda-04-t6"]
AWS = CASES["recruiter-sda-05-t2"]


def docs(row):
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in row["evidence"]
        ],
        row["question"],
    )


def promoted():
    original = docs(GOV)
    return supplement_facets(
        Mock(documents=original), GOV["question"], original, protected_only=True
    )


def test_captured_one_case_request_selects_only_its_validated_source_section():
    selected = _select_context_docs(promoted(), GOV["question"])
    assert selected
    assert {(d.metadata["source"], d.metadata["section_header"]) for d in selected} == {
        (
            "90_website_canon_llms_full.txt",
            "Enterprise metrics governance & AI-ready semantic layer",
        )
    }
    failures = constraint_failures(
        "He introduced MCP-enabled BigQuery integration in Claude.",
        selected,
        GOV["question"],
    )
    assert "requested_documented_quality_checks_missing" not in failures
    assert "requested_report_user_outcome_missing" not in failures


@pytest.mark.parametrize(
    "question",
    [
        GOV["question"].replace("one documented case", "several documented cases"),
        "Compare one documented case of metrics governance with his other projects.",
        "Do not give one case of metric governance; list several examples.",
        "What architecture experience does he have?",
        "Describe one documented case of metrics governance for the Data Hub.",
        "Describe one documented case of metrics governance during Head of Data.",
        "Describe one documented case of metrics governance and pipeline observability.",
        "Describe one documented case of metrics governance and two other cases of data quality.",
        "Previously I asked for one documented case of metric governance; now list all relevant projects.",
        "Describe one documented case of metrics governance.\nAlso include all relevant projects.",
        "Earlier I asked for one documented case of metric governance; now summarize his work.",
        "Donne un cas documente de gouvernance des metrics et deux autres projets.",
        "Describe one documented case of metrics governance and another case of quality control.",
    ],
)
def test_plural_comparative_historical_and_unrelated_queries_keep_normal_selection(
    question,
):
    documents = promoted()
    assert focus_single_case(documents, question) == documents


def test_complete_original_question_is_separate_from_expanded_history():
    documents = promoted()
    latest = "What metric governance access safeguards are documented?"
    scope = latest + "\n" + GOV["question"]
    assert focus_single_case(documents, scope, original_query=latest) == documents
    assert len(_select_context_docs(documents, scope, original_query=latest)) > 1
    multiline = "Describe one documented case.\nWhat metric governance and access safeguards changed for report users?"
    assert len(focus_single_case(documents, multiline, original_query=multiline)) == 1


@pytest.mark.parametrize("invalid", ["source", "section", "support", "retention"])
def test_unvalidated_metadata_cannot_focus_context(invalid):
    documents = promoted()
    primary = next(
        d
        for d in documents
        if d.metadata.get("required_facet") == "enterprise_governance_user_outcomes"
    )
    if invalid == "source":
        primary.metadata["source"] = "other.txt"
    elif invalid == "section":
        primary.metadata["section_header"] = "A different case"
    elif invalid == "support":
        primary.page_content = "An unrelated paragraph."
    else:
        primary.metadata["retain_requested_facet"] = False
    assert focus_single_case(documents, GOV["question"]) == documents


def test_multiple_validated_focus_identities_do_not_arbitrarily_discard_a_case():
    documents = promoted()
    policy = copy.deepcopy(policy_data())
    policy["facets"].append(
        {
            "id": "synthetic_second_case",
            "sources": ["other_case.txt"],
            "section_pattern": "^Other case$",
            "support_pattern": "Synthetic proof",
            "query_pattern": "(?i)metric",
            "single_case_query_pattern": "(?i)one documented case",
            "retain_for_query": True,
        }
    )
    documents.append(
        Document(
            page_content="Synthetic proof",
            metadata={
                "source": "other_case.txt",
                "section_header": "Other case",
                "required_facet": "synthetic_second_case",
                "retain_requested_facet": True,
            },
        )
    )
    with patch("app.services.evidence_policy.policy_data", return_value=policy):
        assert focus_single_case(documents, GOV["question"]) == documents


def test_generation_completion_and_verification_share_the_focused_evidence():
    documents = docs(GOV)
    answer = "Bolaji established an enterprise metrics governance framework and a single source of truth for business metrics. He introduced MCP-enabled BigQuery integration in Claude. Selected senior users could query enterprise data using natural language within their existing Google Cloud IAM permissions."
    generator, verifier = Mock(), Mock()
    generator.with_structured_output.return_value.invoke.return_value = GeneratedAnswer(
        answer=answer
    )
    verifier.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    state = {
        "query": GOV["question"],
        "original_query": GOV["question"],
        "graded_documents": documents,
        "agent_type": "professional",
        "user_language": "en",
    }
    with (
        patch(
            "app.graph.nodes._get_hybrid_search", return_value=Mock(documents=documents)
        ),
        patch("app.graph.nodes._get_llm", side_effect=[generator, verifier]),
    ):
        generated = generate_node(state)
        verified = verify_grounding_node({**state, **generated})
    assert verified["grounding_checked"]
    assert verified["context_documents"] == generated["context_documents"]
    assert len(generated["context_documents"]) == 1
    submitted = verifier.with_structured_output.return_value.invoke.call_args.args[0][
        -1
    ].content
    for unrelated in (
        "13_gozem_internal_projects.txt",
        "700 people",
        "250+ production pipelines",
        "3 access tiers",
    ):
        assert unrelated not in submitted
    assert "single source of truth" in submitted and "IAM permissions" in submitted


def sourced_aws_answer():
    text = Path("data/20_owner_confirmed_production_scope.txt").read_text()
    paragraphs = text.split("\n\n")
    parts = [
        next(p for p in paragraphs if p.startswith(prefix))
        for prefix in (
            "At Gozem, Bolaji personally",
            "For the Gozem Data Hub",
            "Bolaji personally set up",
            "Bolaji has no Kafka",
        )
    ]
    return "\n\n".join(parts)


def test_canonical_record_limit_keeps_known_s3_operation_separate_from_missing_employer():
    answer = sourced_aws_answer()
    assert (
        "This clarification does not identify an employer or project for the S3 bucket"
        in answer
    )
    assert "requested_personal_production_scope:aws s3" not in constraint_failures(
        answer, docs(AWS), AWS["question"] + " At Gozem"
    )


@pytest.mark.parametrize(
    "qualification",
    [
        "If this clarification does not identify an employer or project for the S3 bucket, ask him.",
        "This clarification does not not identify an employer or project for the S3 bucket.",
        "This clarification does not identify an employer or project for the EC2 service.",
        '"This clarification does not identify an employer or project for the S3 bucket."',
    ],
)
def test_nonasserted_or_different_tool_record_limit_cannot_cancel_s3_question_scope(
    qualification,
):
    answer = "Bolaji personally operated AWS S3 in production. " + qualification
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        answer, docs(AWS), AWS["question"] + " At Gozem"
    )


def test_record_limit_never_cancels_an_explicit_gozem_s3_claim():
    answer = (
        sourced_aws_answer()
        + " At Gozem, Bolaji personally operated AWS S3 in production."
    )
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        answer, docs(AWS), AWS["question"]
    )


@pytest.mark.parametrize("code", [500, 502, 504])
def test_provider_status_errors_receive_only_the_existing_one_retry(code):
    error = RuntimeError("Provider failure")
    error.code = code
    assert _is_transient_llm_error(error)
    call = Mock(side_effect=[error, "ok"])
    with patch("app.graph.nodes.time.sleep") as sleep:
        assert _invoke_with_retry(call, "generation") == "ok"
    assert call.call_count == 2
    sleep.assert_called_once_with(0.5)


def test_captured_500_retries_once_and_exhaustion_still_fails():
    error = RuntimeError(CASES["tmp-003"]["error"])
    call = Mock(side_effect=error)
    with patch("app.graph.nodes.time.sleep"):
        with pytest.raises(RuntimeError):
            _invoke_with_retry(call, "guardrail")
    assert call.call_count == 2


def test_provider_retry_cannot_add_a_call_after_the_shared_grounding_budget():
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=Mock(side_effect=RuntimeError("500 INTERNAL"))
        )
    )
    token = _grounding_provider_calls.set(1)
    try:
        with (
            patch("app.graph.nodes.time.sleep"),
            pytest.raises(RuntimeError, match="budget exhausted"),
        ):
            _invoke_with_retry(lambda: _provider_generate(client), "grounding")
    finally:
        _grounding_provider_calls.reset(token)
    assert client.models.generate_content.call_count == 1
