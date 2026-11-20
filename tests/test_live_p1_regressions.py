"""Captured live failures plus independently authored scope and repair controls."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import constraint_failures, supplement_facets
from scripts.run_eval import check_facts, load_golden

CAPTURES = json.loads(
    (Path(__file__).parent / "fixtures/live_p1_failures.json").read_text()
)
GOOD = {
    "recruiter-adi-02-t5": (
        "Power BI production use and DAX proficiency are not established in the "
        "record. This is a verification gap, not a confirmed experience gap."
    ),
    "recruiter-adi-05-t5": (
        "Bolaji a été le premier recrutement data en 2019. L'équipe initiale a "
        "grandi jusqu'à quatre personnes. Elle est ensuite passée de quatre à "
        "plus de quatorze pendant son mandat Head of Data, d'octobre 2022 à mars "
        "2025. Le périmètre couvre cinq marchés opérationnels africains. "
        "Singapour était uniquement le siège de l'entreprise."
    ),
    "recruiter-sda-05-t2": (
        "Bolaji architected and led the Gozem Data Hub, whose documented stack "
        "includes AWS EC2. Personal Kafka architecture delivery is not documented. "
        "The retrieved tool inventory alone does not establish production "
        "ownership of Pub/Sub or Dataflow."
    ),
}


def docs(case):
    return [Document(**doc) for doc in case["documents"]]


def state(case, answer):
    return {
        "answer": answer,
        "query": case["question"],
        "user_language": case["language"],
        "agent_type": "professional",
        "confidence": 0.9,
        "context_documents": docs(case),
    }


@pytest.mark.parametrize("case", CAPTURES, ids=lambda c: c["id"])
def test_exact_live_failures_and_independent_supported_answers(case):
    row = next(r for r in load_golden() if r["id"] == case["id"])
    result = {
        "agent_type": "professional",
        "grounding_checked": True,
        "evidence": case["documents"],
    }
    assert not check_facts(case["delivered_answer"], row, result=result)[0]
    assert constraint_failures(case["delivered_answer"], docs(case), case["question"])
    assert check_facts(GOOD[case["id"]], row, result=result)[0]
    assert not constraint_failures(GOOD[case["id"]], docs(case), case["question"])


@pytest.mark.parametrize("case", CAPTURES, ids=lambda c: c["id"])
def test_incorrect_model_approval_cannot_bypass_scope_checks(case):
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=GOOD[case["id"]]),
        GroundingVerdict(is_grounded=True),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(case, case["delivered_answer"]))
    assert result["answer"] == GOOD[case["id"]]
    assert result["grounding_checked"]
    assert invoke.call_count == 2
    assert GOOD[case["id"]] in "\n".join(
        str(m.content) for m in invoke.call_args.args[0]
    )


@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize(
    "initial", ["", "L'équipe est passée de zéro à quatre personnes."]
)
def test_french_recovery_is_bounded_and_checks_exact_final_text(accepted, initial):
    case = next(c for c in CAPTURES if c["language"] == "fr")
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=initial),
        GeneratedAnswer(answer=GOOD[case["id"]], confidence=0.8),
        GroundingVerdict(is_grounded=accepted),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(case, case["delivered_answer"]))
    assert invoke.call_count == 3
    assert GOOD[case["id"]] in "\n".join(
        str(m.content) for m in invoke.call_args.args[0]
    )
    assert result["grounding_checked"] is accepted
    assert result["answer"] == (
        GOOD[case["id"]] if accepted else INSUFFICIENT_EVIDENCE_ANSWERS["fr"]
    )


@pytest.mark.parametrize(
    "repair",
    [
        "",
        "Bolaji a été recruté en 2019.",
        "L'équipe est passée de zéro à quatre personnes.",
    ],
)
def test_incomplete_repair_never_retries_or_passes(repair):
    case = next(c for c in CAPTURES if c["language"] == "fr")
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=False),
        GeneratedAnswer(answer=repair, confidence=0.8),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state(case, case["delivered_answer"]))
    assert invoke.call_count == 2
    assert not result["grounding_checked"]


@pytest.mark.parametrize(
    "answer,evidence,bad",
    [
        (
            "The team grew from zero to 8.",
            "The founder was the first hire. The team grew to 8.",
            True,
        ),
        ("The team grew from zero to 8.", "The team grew from zero to 8.", False),
        ("The team grew from zero to 8.", "The team grew from zero to 5.", True),
        (
            "L'équipe est passée de zéro à quatre.",
            "L'équipe a grandi jusqu'à quatre.",
            True,
        ),
        (
            "L'équipe est passée de zéro à quatre.",
            "L'équipe est passée de zéro à 4.",
            False,
        ),
        (
            "Power BI is unverified, therefore this would be an experience gap.",
            "Power BI is comparable to another BI tool.",
            True,
        ),
        (
            "Power BI is not documented, indicating an experience gap.",
            "Power BI use is unknown.",
            True,
        ),
        (
            "There is an experience gap in Power BI.",
            "There is an experience gap in Kafka.",
            True,
        ),
        (
            "He lacks experience with Power BI.",
            "He lacks experience with Power BI.",
            False,
        ),
        (
            "This is not an experience gap in Power BI.",
            "Other BI tools are documented.",
            False,
        ),
        (
            "L'utilisation de DAX reste inconnue, donc cela constitue une lacune d'expérience.",
            "DAX est comparable a un autre outil.",
            True,
        ),
        (
            "He has production experience with Pub/Sub and Dataflow.",
            "Tools: Pub/Sub, Dataflow. Python is used in production.",
            True,
        ),
        (
            "He used Pub/Sub in production.",
            "He taught a Pub/Sub production streaming bootcamp.",
            True,
        ),
        (
            "He used Pub/Sub and Dataflow in production.",
            "He used Pub/Sub in production. Dataflow appears in his inventory.",
            True,
        ),
        ("He used Pub/Sub in production.", "He used Pub/Sub in production.", False),
        (
            "He used Pub/Sub and Dataflow in production.",
            "He personally used Pub/Sub and Dataflow in production.",
            False,
        ),
        (
            "Production use of Pub/Sub and Dataflow is not established.",
            "Tools: Pub/Sub, Dataflow.",
            False,
        ),
        (
            "Pub/Sub and Dataflow are not production examples.",
            "Tools: Pub/Sub, Dataflow.",
            False,
        ),
    ],
)
def test_independent_scope_mutations_and_supported_controls(answer, evidence, bad):
    assert (
        bool(
            constraint_failures(
                answer, [Document(page_content=evidence)], "Power BI DAX"
            )
        )
        is bad
    )


def test_streaming_facet_promotes_documented_implementation_not_inventory():
    inventory = Document(
        page_content="Tools: Pub/Sub Dataflow BigQuery",
        metadata={"source": "90_website_canon_llms_full.txt"},
    )
    implementation = Document(
        page_content="He built a MongoDB CDC pipeline publishing change events through Pub/Sub and processing them in Dataflow into BigQuery.",
        metadata={"source": "90_website_canon_llms_full.txt"},
    )
    search = Mock(documents=[inventory, implementation])
    result = supplement_facets(
        search, "Which streaming technologies has he designed?", [inventory]
    )
    assert result[0].page_content == implementation.page_content
    assert result[0].metadata["required_facet"] == "streaming_implementation"


@pytest.mark.parametrize(
    "answer,evidence,bad",
    [
        (
            "Power BI is undocumented, which constitutes an experience gap.",
            "Power BI use is unverified.",
            True,
        ),
        (
            "DAX is unverified and that indicates a lack of experience.",
            "DAX use is unverified.",
            True,
        ),
        (
            "No evidence of an experience gap in Power BI is provided.",
            "Power BI use is unverified.",
            False,
        ),
        (
            "He used Pub/Sub and Dataflow in production.",
            "He used Pub/Sub in production with Dataflow appearing only in the tool inventory.",
            True,
        ),
        (
            "He used Pub/Sub and Dataflow in production.",
            "He used Pub/Sub in production and Dataflow is listed only in the inventory.",
            True,
        ),
        (
            "He operated Dataflow in production.",
            "He operated Dataflow in production after completing training.",
            False,
        ),
        (
            "He operated Dataflow in production.",
            "His team operated Dataflow in production.",
            True,
        ),
        (
            "He operated Dataflow in production.",
            "He supervised a Dataflow production training module.",
            True,
        ),
    ],
)
def test_review_scope_bypasses_and_explicit_production_controls(answer, evidence, bad):
    assert (
        bool(
            constraint_failures(
                answer, [Document(page_content=evidence)], "Power BI DAX"
            )
        )
        is bad
    )


@pytest.mark.parametrize("first_failure", ["parse", "transient"])
def test_provider_budget_includes_malformed_and_transient_retries(first_failure):
    from types import SimpleNamespace

    from app.graph.nodes import _grounding_provider_calls, _SyncGeminiLLM

    case = next(c for c in CAPTURES if c["language"] == "fr")
    llm = object.__new__(_SyncGeminiLLM)
    llm._model = "test"
    llm._temperature = 0
    llm._max_output_tokens = 750
    llm._thinking_budget = 0
    llm._client = Mock()
    generate = llm._client.models.generate_content
    if first_failure == "parse":
        failure = SimpleNamespace(parsed=None, text="{bad-json")
    else:
        failure = RuntimeError("503 UNAVAILABLE")
    generate.side_effect = [
        failure,
        SimpleNamespace(parsed={"is_grounded": False, "corrected_answer": ""}, text=""),
        SimpleNamespace(
            parsed={"answer": GOOD[case["id"]], "confidence": 0.8}, text=""
        ),
        SimpleNamespace(parsed={"is_grounded": True}, text=""),
    ]
    with (
        patch("app.graph.nodes._get_llm", return_value=llm),
        patch("app.graph.nodes.time.sleep"),
    ):
        result = verify_grounding_node(state(case, case["delivered_answer"]))
    assert generate.call_count == 3
    assert not result["grounding_checked"]
    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS["fr"]
    assert _grounding_provider_calls.get() is None


@pytest.mark.parametrize(
    "answer,evidence,bad",
    [
        (
            "Power BI has an experience gap.",
            "It is unverified whether he has an experience gap in Power BI.",
            True,
        ),
        (
            "He operated Dataflow in production.",
            "He used Pub/Sub in production alongside Dataflow appearing only in the tool inventory.",
            True,
        ),
        (
            "He operated Dataflow in production.",
            "He confirmed that colleagues operated Dataflow in production.",
            True,
        ),
        (
            "He operated Dataflow in production.",
            "He operated Dataflow in production after completing a training module.",
            False,
        ),
        (
            "He operated Dataflow in production.",
            "Il a exploite Dataflow en production apres sa formation.",
            False,
        ),
        (
            "He operated Dataflow in production.",
            "He designed Dataflow production pipelines for a training module.",
            True,
        ),
    ],
)
def test_operation_objects_and_actor_remain_bound(answer, evidence, bad):
    assert (
        bool(constraint_failures(answer, [Document(page_content=evidence)], "Power BI"))
        is bad
    )
