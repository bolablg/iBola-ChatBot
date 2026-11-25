"""Actual valid corrections, evidence limits and public references must survive."""

import json
import shlex
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import (
    _record_limit_subject,
    constraint_failures,
    documentation_limits,
    prepare_evidence,
    preserve_requested_evidence,
)
from scripts.run_eval import check_facts
from tests.prompt_fields import prompt_field

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/reference_completion_failures.json").read_text()
)
CASES = {c["id"]: c for c in FIXTURE["cases"]}
PIPELINE = CASES["recruiter-sde-01-t1"]
SPARK = CASES["recruiter-sde-03-t2"]
LINK = CASES["recruiter-sde-04-t6"]
MIGRATION = "migration_earlier_role_missing"
REFERENCE = "requested_dataform_public_reference_missing"
URL = "https://www.bolablg.com/projects/dataform-modeling-lab/"


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


def response(case, sequence):
    return json.loads(
        next(
            r["response_text"]
            for r in case["sdk_responses"]
            if r["sequence"] == sequence
        )
    )


def reference_query():
    return response(LINK, 1)["standalone_query"]


def state(case, answer):
    return {
        "query": reference_query() if case["id"] == LINK["id"] else case["question"],
        "original_query": case["question"],
        "answer": answer,
        "context_documents": docs(case),
        "agent_type": "professional",
        "user_language": case["lang"],
    }


def test_exact_valid_verdict_correction_is_rechecked_without_unnecessary_repair():
    initial = response(PIPELINE, 3)["answer"]
    first = response(PIPELINE, 4)
    corrected = first["corrected_answer"]
    repaired = response(PIPELINE, 5)["answer"]
    assert MIGRATION in constraint_failures(
        initial, docs(PIPELINE), PIPELINE["question"]
    )
    for answer in (corrected, repaired):
        assert MIGRATION not in constraint_failures(
            answer, docs(PIPELINE), PIPELINE["question"]
        )
    verifier = Mock()
    verifier.invoke.side_effect = [
        GroundingVerdict(**first),
        GroundingVerdict(is_grounded=True),
    ]
    llm = Mock()
    llm.with_structured_output.return_value = verifier
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(PIPELINE, initial))
    assert result["grounding_checked"] and result["answer"].startswith(corrected)
    assert verifier.invoke.call_count == 2
    assert all(
        call.args[0] is GroundingVerdict
        for call in llm.with_structured_output.call_args_list
    )
    assert (
        prompt_field(verifier.invoke.call_args.args[0][-1].content, "draft_answer")
        == result["answer"]
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "During his earlier",
        "During his former",
        "During his previous",
        "In his earlier",
        "In his previous",
    ],
)
def test_bounded_role_adjectives_still_require_the_documented_local_role(prefix):
    claim = (
        prefix
        + " Global Data Analyst role, Bolaji migrated 10+ TB within a single quarter."
    )
    assert MIGRATION not in constraint_failures(
        claim, docs(PIPELINE), PIPELINE["question"]
    )
    assert MIGRATION in constraint_failures(
        claim.replace("Global Data Analyst", "Head of Data"),
        docs(PIPELINE),
        PIPELINE["question"],
    )
    assert MIGRATION in constraint_failures(
        "His CV mentions Global Data Analyst. He migrated 10+ TB within a single quarter.",
        docs(PIPELINE),
        PIPELINE["question"],
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "During his earlier",
        "During his former",
        "During his previous",
        "In his earlier",
        "In his previous",
    ],
)
def test_new_role_prefix_does_not_assign_a_phase_to_the_undated_cdc_project(prefix):
    answer = (
        "At Gozem, "
        + prefix
        + " Global Data Analyst role, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."
    )
    rule = "cdc_role_attribution_requires_direct_evidence"
    assert rule in constraint_failures(answer, docs(PIPELINE), PIPELINE["question"])
    direct = Document(
        page_content=answer,
        metadata={"source": "synthetic-role-proof.txt", "evidence_type": "curated"},
    )
    assert rule not in constraint_failures(
        answer, docs(PIPELINE) + [direct], PIPELINE["question"]
    )
    wrong = Document(
        page_content=answer.replace("Global Data Analyst", "Head of Data"),
        metadata=direct.metadata,
    )
    assert rule in constraint_failures(
        answer, docs(PIPELINE) + [wrong], PIPELINE["question"]
    )


def test_exact_short_deployment_limit_preserves_kafka_confirmation_and_unknown_spark():
    assert not documentation_limits(SPARK["answer"], ["Spark Structured Streaming"])
    assert check_facts(SPARK["answer"], SPARK["golden_assertions"], result=SPARK)[0]


@pytest.mark.parametrize(
    "prefix",
    [
        "There is no documented project proving personal deployment of",
        "There is no documented named project proving personal production deployment of",
        "There is no documented project proving deployment of",
    ],
)
def test_short_carrier_keeps_complete_literal_subject_and_rejects_finite_suffix(prefix):
    text = prefix + " Spark Structured Streaming"
    assert (
        _record_limit_subject(text, ["Spark Structured Streaming"])
        == "Spark Structured Streaming"
    )
    assert documentation_limits(text, ["Kafka"])
    query = "Has Bolaji personally deployed Spark Structured Streaming in production?"
    assert not constraint_failures(text, docs(SPARK), query)
    for suffix in (
        " and he runs it in production",
        " if needed",
        " and he personally operated Spark Structured Streaming in production",
    ):
        bad = text + suffix
        assert _record_limit_subject(bad, ["Spark Structured Streaming"]) is None
        assert any(
            failure.startswith("requested_personal_production_scope:")
            for failure in constraint_failures(bad, docs(SPARK), query)
        )
    assert _record_limit_subject("If " + text, ["Spark Structured Streaming"]) is None
    assert (
        _record_limit_subject(text + " and he owns it", ["Spark Structured Streaming"])
        is None
    )


@pytest.mark.parametrize("language", ["en", "fr"])
def test_exact_retrieved_url_restores_clickable_reference_and_is_idempotent(language):
    documents = docs(LINK)
    final, added = preserve_requested_evidence(
        LINK["answer"], documents, reference_query(), language
    )
    assert added == [REFERENCE] and URL in final
    assert check_facts(final, LINK["golden_assertions"], result=LINK)[0]
    assert not constraint_failures(final, documents, reference_query())
    assert preserve_requested_evidence(
        final, documents, reference_query(), language
    ) == (final, [])


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_source",
        "wrong_section",
        "missing_url",
        "duplicate_url",
        "excluded_evidence",
        "longer_path",
        "dot_path",
        "query_suffix",
    ],
)
def test_url_restoration_never_guesses_or_uses_ineligible_evidence(failure):
    document = next(
        d
        for d in docs(LINK)
        if d.metadata["source"] == "90_website_canon_llms_full.txt"
    )
    if failure == "wrong_source":
        document.metadata["source"] = "another.txt"
    elif failure == "wrong_section":
        document.metadata["section_header"] = "Another framework"
    elif failure == "missing_url":
        document.page_content = document.page_content.replace(URL, "")
    elif failure == "duplicate_url":
        document.page_content += "\n" + URL
    elif failure in {"longer_path", "dot_path", "query_suffix"}:
        suffix = {
            "longer_path": "other-page",
            "dot_path": ".hidden",
            "query_suffix": "?view=other",
        }[failure]
        document.page_content = document.page_content.replace(URL, URL + suffix)
    else:
        document.metadata["evidence_type"] = "interpretation"
    assert preserve_requested_evidence(
        LINK["answer"], [document], reference_query(), "en"
    ) == (LINK["answer"], [])


def test_dataform_url_is_not_forced_into_a_different_framework_request():
    assert preserve_requested_evidence(
        LINK["answer"],
        docs(LINK),
        "Point me to public documentation for the dbt framework",
        "en",
    ) == (LINK["answer"], [])


def test_exact_url_can_be_followed_by_sentence_punctuation_without_losing_idempotence():
    document = next(
        d
        for d in docs(LINK)
        if d.metadata["source"] == "90_website_canon_llms_full.txt"
    )
    document.page_content = document.page_content.replace(URL, URL + ". ")
    final, added = preserve_requested_evidence(
        LINK["answer"], [document], reference_query(), "en"
    )
    assert added == [REFERENCE] and URL in final
    assert preserve_requested_evidence(final, [document], reference_query(), "en") == (
        final,
        [],
    )


@pytest.mark.parametrize("accepted", [True, False])
def test_restored_reference_requires_verification_of_the_actual_final_text(accepted):
    verifier = Mock()
    verifier.invoke.return_value = GroundingVerdict(is_grounded=accepted)
    repair = Mock()
    repair.invoke.side_effect = ValueError("Synthetic repair failure")
    llm = Mock()
    llm.with_structured_output.side_effect = lambda schema: (
        verifier if schema is GroundingVerdict else repair
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(LINK, LINK["answer"]))
    if accepted:
        assert result["grounding_checked"] and URL in result["answer"]
        assert (
            prompt_field(verifier.invoke.call_args.args[0][-1].content, "draft_answer")
            == result["answer"]
        )
    else:
        assert not result["grounding_checked"] and URL not in result["answer"]


def test_ci_profile_warmups_send_both_request_optouts_without_changing_flow():
    workflow = yaml.safe_load(Path(".github/workflows/deploy.yml").read_text())
    commands = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            script = step.get("run", "").replace("\\\n", " ")
            for line in script.splitlines():
                if "curl " in line and "/ask-agentic" in line and "-X POST" in line:
                    tokens = shlex.split(line)
                    headers = {
                        tokens[i + 1].strip()
                        for i, t in enumerate(tokens[:-1])
                        if t == "-H"
                    }
                    assert {
                        "X-Ibola-No-Trace: 1",
                        "X-Ibola-No-Notifications: 1",
                    } <= headers
                    assert '"user_input":"What is Bolaji latest role?"' in line
                    commands.append(line)
    assert len(commands) == 2
