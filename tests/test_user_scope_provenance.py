"""Search hints cannot add biography constraints or alter the user's question."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import (
    _sanitize_answer,
    _verification_scope,
    generate_node,
    verify_grounding_node,
)
from app.graph.prompts import GENERATE_PROMPT, VERIFY_GROUNDING_PROMPT
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    request_scope,
    user_answer_scope,
)
from scripts.run_eval import _confirmed_experience_limit, check_facts

ROOT = Path(__file__).parent / "fixtures"
CAPTURE = json.loads((ROOT / "local_1_2_11_user_scope_contrast.json").read_text())
AWS = next(
    r
    for r in json.loads(
        (ROOT / "sdk_1_2_11_case_scope_and_provider_failures.json").read_text()
    )["results"]
    if r["id"] == "recruiter-sda-05-t2"
)
EC2 = json.loads((ROOT / "sdk_1_2_11_ec2_role_failure.json").read_text())["result"]


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


def test_rewrite_company_role_and_period_stay_search_hints_not_answer_scope():
    state = {
        "original_query": AWS["question"],
        "query": "Which production AWS services did Bolaji operate at Gozem as Head of Data during 2022?",
    }
    assert "Gozem" in request_scope(state)
    assert user_answer_scope(state) == _verification_scope(state) == AWS["question"]
    answer = "Bolaji personally operated AWS S3 in production."
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        answer, docs(AWS), request_scope(state)
    )
    assert "requested_personal_production_scope:aws s3" not in constraint_failures(
        answer, docs(AWS), _verification_scope(state)
    )


def test_explicit_user_company_scope_is_still_enforced():
    question = "At Gozem, which production AWS services did Bolaji personally operate?"
    state = {"original_query": question, "query": "List AWS services."}
    assert _verification_scope(state) == question
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        "Bolaji personally operated AWS S3 in production.",
        docs(AWS),
        _verification_scope(state),
    )


@pytest.mark.parametrize(
    "latest",
    ["What other services?", "What else?", "Tell me more", "Can you elaborate?"],
)
def test_followups_inherit_one_bounded_user_request_not_assistant_or_rewrite_facts(
    latest,
):
    state = {
        "original_query": latest,
        "query": "Synthetic employer FictionalCo",
        "chat_history": [
            ("Earlier unrelated user request", "Old assistant fact"),
            (
                "Which production services did he personally operate at Gozem?",
                "Assistant says he worked at FabricatedCo.",
            ),
        ],
    }
    scope = user_answer_scope(state)
    assert "production" in scope and "Gozem" in scope
    assert "FabricatedCo" not in scope and "FictionalCo" not in scope
    assert "unrelated" not in scope


def test_standalone_question_does_not_borrow_an_earlier_user_role_requirement():
    state = {
        "original_query": AWS["question"],
        "query": "At Gozem in Head of Data role",
        "chat_history": [
            ("Our vacancy needs a Head of Data managing25people.", "A response.")
        ],
    }
    assert user_answer_scope(state) == AWS["question"]


def test_legacy_direct_query_and_history_bound_remain_supported():
    assert (
        user_answer_scope({"query": "At Gozem, what did he operate?"})
        == "At Gozem, what did he operate?"
    )
    scope = user_answer_scope(
        {
            "original_query": "What other services?",
            "chat_history": [("x" * 2000, "Excluded assistant text")],
        }
    )
    assert len(scope.split("\n")[1]) == 1200 and "assistant" not in scope


def test_generation_and_verification_receive_original_scope_without_rewrite_additions():
    state = {
        "original_query": AWS["question"],
        "query": "Which AWS services did Bolaji operate at Gozem as Head of Data?",
        "graded_documents": docs(AWS),
        "agent_type": "professional",
        "user_language": "en",
    }
    generator, verifier = Mock(), Mock()
    generator.with_structured_output.return_value.invoke.return_value = GeneratedAnswer(
        answer="Bolaji personally operated AWS S3 in production."
    )
    verifier.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    generation_prompt, verification_prompt = Mock(wraps=GENERATE_PROMPT), Mock(
        wraps=VERIFY_GROUNDING_PROMPT
    )
    with (
        patch(
            "app.graph.nodes._get_hybrid_search", return_value=Mock(documents=docs(AWS))
        ),
        patch("app.graph.nodes._get_llm", side_effect=[generator, verifier]),
        patch("app.graph.nodes.GENERATE_PROMPT", generation_prompt),
        patch("app.graph.nodes.VERIFY_GROUNDING_PROMPT", verification_prompt),
    ):
        generated = generate_node(state)
        result = verify_grounding_node({**state, **generated})
    assert result["grounding_checked"]
    assert (
        generation_prompt.format_messages.call_args.kwargs["query"] == AWS["question"]
    )
    assert (
        verification_prompt.format_messages.call_args.kwargs["question"]
        == AWS["question"]
    )


def test_captured_correct_same_actor_other_tool_contrast_passes_only_with_evidence_and_grounding():
    row = CAPTURE["result"]
    assert row["scores"]["accuracy"] == row["scores"]["faithfulness"] == 10
    assert _confirmed_experience_limit(row["answer"], "Spark Structured Streaming")
    assert check_facts(row["answer"], CAPTURE["golden"], result=row)[0]
    for override in ({"grounding_checked": False}, {"evidence": []}):
        assert not check_facts(
            row["answer"], CAPTURE["golden"], result={**row, **override}
        )[0]


@pytest.mark.parametrize(
    "contrast",
    [
        "although he used Spark Structured Streaming at Rintio",
        "although he used it at Rintio",
        "although he could use Apache Spark at Rintio",
        "although he used Apache Spark at Rintio and deployed it",
        "although he used UnknownTool at Rintio",
        "although he used Apache Spark at Rintio then deployed Kafka",
        "although he used Apache Spark at Rintio then deployed it in production",
        "although he used Apache Spark at Rintio after deploying Kafka",
        "although he used Apache Spark at Rintio to deploy Kafka",
        "although he used Apache Spark at Rintio where Kafka powers production",
        "although he used Apache Spark at Rintio where it processes Kafka events",
        "although he used Apache Spark at Rintio because Kafka handles the pipeline",
        "although he used Apache Spark at Rintio where Kafka became the streaming platform",
    ],
)
def test_contrast_cannot_hide_same_tool_pronoun_hypothetical_or_extra_predicate(
    contrast,
):
    answer = (
        "There is no documented production experience with Spark Structured Streaming, "
        + contrast
        + "."
    )
    assert not _confirmed_experience_limit(answer, "Spark Structured Streaming")


@pytest.mark.parametrize("separator", [" ", "\n\n"])
def test_independent_ec2_sentence_does_not_inherit_the_cdc_role(separator):
    answer = (
        "During his Head of Data role at Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline."
        + separator
        + "Bolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub."
    )
    assert "ec2_role_attribution_requires_direct_evidence" not in constraint_failures(
        _sanitize_answer(answer), docs(EC2), EC2["question"]
    )


def test_same_sentence_and_explicit_heading_ec2_role_claims_still_need_role_evidence():
    for answer in (
        EC2["answer"],
        "Head of Data:\nBolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub.",
    ):
        assert "ec2_role_attribution_requires_direct_evidence" in constraint_failures(
            answer, docs(EC2), EC2["question"]
        )


def test_corrected_independent_ec2_sentences_are_reverified_exactly():
    corrected = "During his Head of Data role at Gozem, Bolaji personally designed and operated Pub/Sub and Dataflow in production for the MongoDB CDC pipeline. Bolaji personally designed and operated AWS EC2 in production for the Gozem Data Hub."
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=corrected),
        GroundingVerdict(is_grounded=True),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "original_query": EC2["question"],
                "query": EC2["question"],
                "answer": EC2["answer"],
                "context_documents": docs(EC2),
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert result["grounding_checked"] and result["answer"] == corrected
    assert invoke.call_count == 2
    assert (
        invoke.call_args.args[0][-1].content.rsplit("Answer to verify:\n", 1)[1]
        == corrected
    )


def test_short_tool_followup_uses_actual_user_production_scope():
    state = {
        "original_query": "And AWS S3?",
        "query": "Invented rewrite",
        "chat_history": [
            (
                "Which services did he personally operate in production?",
                "Assistant company claim",
            )
        ],
    }
    assert "personally operate in production" in user_answer_scope(state)
    assert "Assistant" not in user_answer_scope(state)


def test_reference_only_does_not_inherit_old_comparison_and_current_project_wins():
    history = [("How does this compare with dbt at Gozem?", "Untrusted assistant")]
    for question in (
        "Where is the public article for this framework?",
        "And what did he use at Rintio?",
    ):
        assert (
            user_answer_scope({"original_query": question, "chat_history": history})
            == question
        )


def test_explicit_personal_production_claim_requires_direct_proof_even_for_short_question():
    inventory = [
        Document(page_content="Tools: AWS S3", metadata={"source": "inventory.txt"})
    ]
    failure = "requested_personal_production_scope:aws s3"
    assert failure in constraint_failures(
        "He personally operated AWS S3 in production.", inventory, "And AWS S3?"
    )
    for answer in (
        "If he personally operated AWS S3 in production, ask for details.",
        "It is not documented whether he personally operated AWS S3 in production.",
    ):
        assert failure not in constraint_failures(answer, inventory, "And AWS S3?")


@pytest.mark.parametrize(
    "answer",
    [
        "AWS S3 was personally operated by Bolaji in production.",
        "He personally operated AWS S3 in production for an undocumented employer.",
    ],
)
def test_passive_claim_and_unknown_employer_do_not_prove_operation(answer):
    inventory = [
        Document(page_content="Tools: AWS S3", metadata={"source": "inventory.txt"})
    ]
    assert "requested_personal_production_scope:aws s3" in constraint_failures(
        answer, inventory, "And AWS S3?"
    )
