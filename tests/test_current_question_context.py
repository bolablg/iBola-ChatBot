"""The current request and earlier reference context have separate model inputs."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import generate_node, verify_grounding_node
from app.graph.prompts import (
    GENERATE_PROMPT,
    REPAIR_GROUNDING_PROMPT,
    VERIFY_GROUNDING_PROMPT,
)
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    preserve_requested_evidence,
)
from tests.prompt_fields import prompt_field

CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures/local_1_2_11_current_question.json").read_text()
)
LIMITS = {
    "self_service_measurement_period_unknown",
    "self_service_comparison_method_unknown",
}


def evidence(row):
    return prepare_evidence(
        [
            Document(
                page_content=e["content_preview"],
                metadata={**e, "section_header": e["section"]},
            )
            for e in row["evidence"]
        ],
        row.get("question", ""),
    )


def test_actual_powerbi_question_is_not_combined_with_old_efficiency_question():
    row = CAPTURE["powerbi"]
    state = {
        "original_query": row["question"],
        "query": "Synthetic search rewrite",
        "chat_history": CAPTURE["history"],
        "graded_documents": evidence(row),
        "agent_type": "professional",
        "user_language": "en",
    }
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GeneratedAnswer(
        answer="Power BI and DAX production use is not documented."
    )
    prompt = Mock(wraps=GENERATE_PROMPT)
    with (
        patch("app.graph.nodes._get_hybrid_search", return_value=Mock(documents=[])),
        patch("app.graph.nodes._get_llm", return_value=llm),
        patch("app.graph.nodes.GENERATE_PROMPT", prompt),
    ):
        generate_node(state)
    args = prompt.format_messages.call_args.kwargs
    assert prompt.format_messages.call_count == 1 and args["query"] == row["question"]
    assert CAPTURE["history"][-1][0] not in args["query"]
    assert CAPTURE["history"][-1][0] in args["chat_history"]


@pytest.mark.parametrize("path", ["correction", "repair", "second_correction"])
def test_other_services_reference_context_reaches_every_verifier_and_repair_path(path):
    question = "What other languages?"
    history = [
        ("Excluded old turn", "Excluded coverage"),
        ("Earlier languages", "Older covered Go."),
        ("Another turn", "Another response"),
        ("Another turn2", "Another response2"),
        ("Another turn3", "Another response3"),
        ("Which languages does he use?", "x" * 450 + " Already covered SQL."),
    ]
    answer = "Bolaji uses SQL."
    corrected = "Bolaji uses Python."
    alternative = "Bolaji has Python experience."
    llm, repair = Mock(), Mock()
    invoke = llm.with_structured_output.return_value.invoke
    if path == "repair":
        invoke.side_effect = [
            GroundingVerdict(is_grounded=False),
            GroundingVerdict(is_grounded=True),
        ]
    elif path == "second_correction":
        invoke.side_effect = [
            GroundingVerdict(is_grounded=False, corrected_answer=corrected),
            GroundingVerdict(is_grounded=False, corrected_answer=alternative),
            GroundingVerdict(is_grounded=True),
        ]
    else:
        invoke.side_effect = [
            GroundingVerdict(is_grounded=False, corrected_answer=corrected),
            GroundingVerdict(is_grounded=True),
        ]
    repair.with_structured_output.return_value.invoke.return_value = GeneratedAnswer(
        answer=corrected
    )
    verify_prompt, repair_prompt = Mock(wraps=VERIFY_GROUNDING_PROMPT), Mock(
        wraps=REPAIR_GROUNDING_PROMPT
    )
    state = {
        "original_query": question,
        "query": "Untrusted rewrite",
        "chat_history": history,
        "answer": answer,
        "context_documents": [
            Document(
                page_content="Bolaji uses SQL. Bolaji uses Python.",
                metadata={"source": "synthetic.txt"},
            )
        ],
        "agent_type": "professional",
        "user_language": "en",
    }
    with (
        patch("app.graph.nodes._get_llm", side_effect=[llm, repair]),
        patch("app.graph.nodes.VERIFY_GROUNDING_PROMPT", verify_prompt),
        patch("app.graph.nodes.REPAIR_GROUNDING_PROMPT", repair_prompt),
    ):
        result = verify_grounding_node(state)
    assert result["grounding_checked"] and "SQL" not in result["answer"]
    calls = (
        verify_prompt.format_messages.call_args_list
        + repair_prompt.format_messages.call_args_list
    )
    assert len(calls) == (2 if path == "correction" else 3)
    for call in calls:
        assert call.kwargs["question"] == question
        assert "Already covered SQL." in call.kwargs["conversation_context"]
        assert "Older covered Go." in call.kwargs["conversation_context"]
        assert "Excluded coverage" not in call.kwargs["conversation_context"]
        assert (
            "SQL" not in call.kwargs["context"]
            or "Bolaji uses SQL." in call.kwargs["context"]
        )
    assert repair_prompt.format_messages.call_count == (1 if path == "repair" else 0)


def test_prior_assistant_authority_never_becomes_retrieved_proof():
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=True
    )
    state = {
        "original_query": "What about that authority?",
        "chat_history": [
            (
                "What authority did he hold?",
                "Bolaji held formal performance-management authority.",
            )
        ],
        "answer": "Bolaji held formal performance-management authority.",
        "context_documents": [
            Document(page_content="Bolaji partnered with HR on performance management.")
        ],
        "agent_type": "professional",
        "user_language": "en",
    }
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    assert not result["grounding_checked"]


def self_service_docs():
    return evidence(CAPTURE["self_service"]["response"])


KNOWN = "L'adoption de Metabase a atteint plus de 12 utilisateurs managers actifs et a divise par deux le volume des requetes courantes adressees a l'equipe data centrale."


def test_exact_self_service_source_restores_both_requested_limits_within_existing_cap():
    question = CAPTURE["self_service"]["question"]
    docs = self_service_docs()
    assert LIMITS <= set(constraint_failures(KNOWN, docs, question))
    final, added = preserve_requested_evidence(KNOWN, docs, question, "fr")
    assert LIMITS <= set(added) and len(added) <= 3
    assert final.startswith(KNOWN) and "Metabase" in final and "12" in final
    assert not LIMITS.intersection(constraint_failures(final, docs, question))
    assert preserve_requested_evidence(final, docs, question, "fr") == (final, [])


@pytest.mark.parametrize(
    "kind",
    ["wrong_source", "wrong_section", "missing_scope", "missing_text", "unrequested"],
)
def test_limit_completion_requires_current_request_and_actual_scoped_source_proof(kind):
    docs = copy.deepcopy(self_service_docs())
    question = CAPTURE["self_service"]["question"]
    if kind == "wrong_source":
        docs[0].metadata["source"] = "unrelated.txt"
    elif kind == "wrong_section":
        docs[0].metadata["section_header"] = "Other project"
    elif kind == "missing_scope":
        docs[0].metadata.pop("metric_scope", None)
    elif kind == "missing_text":
        docs[0].page_content = "Metabase reached 12+ active users."
    else:
        question = "Quels outils a-t-il utilises?"
    final, added = preserve_requested_evidence(KNOWN, docs, question, "fr")
    assert not LIMITS.intersection(added)


@pytest.mark.parametrize("accept", [False, True])
def test_completed_self_service_answer_is_independently_verified_exactly(accept):
    row = CAPTURE["self_service"]
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=accept
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "original_query": row["question"],
                "query": row["question"],
                "answer": KNOWN,
                "context_documents": self_service_docs(),
                "agent_type": "professional",
                "user_language": "fr",
            }
        )
    assert bool(result["grounding_checked"]) == accept
    if accept:
        assert result["answer"].startswith(KNOWN)
        assert (
            prompt_field(
                llm.with_structured_output.return_value.invoke.call_args.args[0][
                    -1
                ].content,
                "draft_answer",
            )
            == result["answer"]
        )
