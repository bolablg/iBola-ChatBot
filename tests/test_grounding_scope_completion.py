"""Captured release failures, with independent SDK-budget and scope controls."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.graph import nodes
from app.services.evidence_policy import (
    _postfixed_record_limit,
    constraint_failures,
    policy_data,
    prepare_evidence,
)
from scripts.run_eval import check_facts

FIXTURE = json.loads(Path("tests/fixtures/grounding_scope_failures.json").read_text())
CASES = {c["id"]: c for c in FIXTURE["cases"]}


def response(case, seq):
    return json.loads(
        next(r["response_text"] for r in case["sdk_responses"] if r["sequence"] == seq)
    )


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


def state(case):
    return {
        "query": response(case, 1)["standalone_query"],
        "original_query": case["question"],
        "answer": response(case, 3)["answer"],
        "context_documents": docs(case),
        "agent_type": "professional",
        "user_language": case["lang"],
    }


def sdk(monkeypatch, responses):
    llm = object.__new__(nodes._SyncGeminiLLM)
    llm._model = "gemini-3.5-flash-lite"
    llm._temperature = 0
    llm._max_output_tokens = 750
    llm._thinking_budget = None
    llm._thinking_level = "low"
    generate = Mock(side_effect=[SimpleNamespace(parsed=r, text="") for r in responses])
    llm._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(nodes, "_get_llm", lambda **kwargs: llm)
    return generate


@pytest.mark.parametrize("cid", ["recruiter-sda-01-t1", "recruiter-sda-05-t2"])
def test_actual_separate_project_repair_survives_and_is_independently_verified(
    monkeypatch, cid
):
    c = CASES[cid]
    repaired = response(c, 5)["answer"]
    assert not constraint_failures(repaired, docs(c), c["question"])
    generate = sdk(monkeypatch, [response(c, 4), response(c, 5), {"is_grounded": True}])
    result = nodes.verify_grounding_node(state(c))
    assert result["grounding_checked"]
    if cid == "recruiter-sda-05-t2":
        # The existing exact EC2 project quotation now makes the original draft
        # complete, so the former false non-use refusal needs no repair call.
        assert result["answer"].startswith(response(c, 3)["answer"])
        assert generate.call_count == 1
        assert check_facts(
            result["answer"], c["golden_assertions"], result={**c, **result}
        )[0]
    else:
        assert result["answer"] == repaired
        assert generate.call_count == 3
    assert result["answer"] in str(generate.call_args.kwargs["contents"])
    if cid == "recruiter-sda-01-t1":
        assert "requested_personal_production_scope:aws ec2" in constraint_failures(
            response(c, 3)["answer"], docs(c), c["question"]
        )


def test_actual_topic_introduction_does_not_hide_the_local_migration_role(monkeypatch):
    c = CASES["recruiter-sde-01-t1"]
    correction = response(c, 4)["corrected_answer"]
    assert not constraint_failures(correction, docs(c), c["question"])
    generate = sdk(monkeypatch, [response(c, 4), {"is_grounded": True}])
    result = nodes.verify_grounding_node(state(c))
    assert result["grounding_checked"] and result["answer"].startswith(correction)
    assert generate.call_count == 2
    for bad in [
        correction.replace("Global Data Analyst", "Head of Data"),
        "Regarding the Google Cloud migration, Bolaji migrated 10+ TB within a single quarter. His CV lists Global Data Analyst.",
        "Regarding the Google Cloud migration, Bolaji personally completed it in his Head of Data role.",
        "As for the cloud migration, Bolaji personally completed it as Data Director.",
        "Regarding the Google Cloud migration, Bolaji personally completed it as Head of Data, and as Global Data Analyst built 50+ scheduled ETL pipelines.",
    ]:
        assert "migration_earlier_role_missing" in constraint_failures(
            bad, docs(c), c["question"]
        )
    assert "migration_earlier_role_missing" not in constraint_failures(
        "Regarding the Google Cloud migration, Bolaji personally completed it in his Global Data Analyst role.",
        docs(c),
        c["question"],
    )


def test_actual_spark_absence_claim_is_repaired_into_a_documentation_boundary(
    monkeypatch,
):
    c = CASES["recruiter-sde-03-t2"]
    assert "non_use_requires_explicit_negative_evidence" in constraint_failures(
        response(c, 3)["answer"], docs(c), c["question"]
    )
    repaired = "Bolaji has never used Kafka. Personal production deployment of Spark Structured Streaming is not documented, and no named production project proves that deployment."
    generate = sdk(
        monkeypatch,
        [response(c, 4), {"answer": repaired, "confidence": 1}, {"is_grounded": True}],
    )
    result = nodes.verify_grounding_node(state(c))
    assert result["grounding_checked"] and result["answer"] == repaired
    assert check_facts(repaired, c["golden_assertions"], result={**c, **result})[0]
    assert generate.call_count == 3


def test_nominal_record_limit_cannot_swallow_actions_conditions_or_neighboring_claims():
    rule = policy_data()["request_scopes"][0]
    c = CASES["recruiter-sda-01-t1"]
    nominal = (
        response(c, 5)["answer"]
        .split("Remaining requested details", 1)[1]
        .strip()
        .rstrip(".")
    )
    nominal = "Remaining requested details" + " " + nominal
    assert _postfixed_record_limit(nominal, rule)
    for text in [
        "If " + nominal,
        nominal + " and he operated Spark in production",
        nominal.replace("Spark Structured Streaming", "he operated Spark"),
        nominal + ". He personally operated Spark in production.",
    ]:
        assert not _postfixed_record_limit(text, rule)
    assert "requested_personal_production_scope:spark" in constraint_failures(
        nominal + ". He personally operated Spark in production.",
        docs(c),
        c["question"],
    )


@pytest.mark.parametrize(
    "grounded,on_topic", [(True, True), (False, True), (True, False)]
)
def test_actual_french_second_correction_needs_a_third_independent_approval(
    monkeypatch, grounded, on_topic
):
    c = CASES["skill-003-fr"]
    generate = sdk(
        monkeypatch,
        [
            response(c, 4),
            response(c, 5),
            {"is_grounded": grounded, "addresses_question": on_topic},
        ],
    )
    result = nodes.verify_grounding_node(state(c))
    assert generate.call_count == 3
    assert result["grounding_checked"] is (grounded and on_topic)
    if grounded and on_topic:
        assert result["answer"] == response(c, 5)["corrected_answer"]
        assert result["answer"] in str(generate.call_args.kwargs["contents"])
    else:
        assert result["answer"] != response(c, 5)["corrected_answer"]
    assert nodes._grounding_provider_calls.get() is None


@pytest.mark.parametrize("mutation", ["empty", "unchanged", "whitespace", "invalid"])
def test_unapproved_second_correction_cannot_spend_an_extra_call(monkeypatch, mutation):
    c = CASES["skill-003-fr"]
    second = dict(response(c, 5))
    second["corrected_answer"] = {
        "empty": "",
        "unchanged": response(c, 4)["corrected_answer"],
        "whitespace": response(c, 4)["corrected_answer"].replace(" ", "  "),
        "invalid": "Bolaji has never used Spark Structured Streaming.",
    }[mutation]
    generate = sdk(monkeypatch, [response(c, 4), second])
    result = nodes.verify_grounding_node(state(c))
    assert not result["grounding_checked"] and generate.call_count == 2


def test_parse_recovery_consumes_the_spare_verifier_call(monkeypatch):
    c = CASES["skill-003-fr"]
    generate = sdk(monkeypatch, [response(c, 4), response(c, 5)])
    original = generate.side_effect
    generate.side_effect = iter(
        [SimpleNamespace(parsed=None, text="not JSON"), *list(original)]
    )
    result = nodes.verify_grounding_node(state(c))
    assert not result["grounding_checked"] and generate.call_count == 3


def test_unconfirmed_pronoun_and_other_tool_never_borrow_kafka_non_use():
    c = CASES["recruiter-sda-05-t2"]
    assert "non_use_requires_explicit_negative_evidence" not in constraint_failures(
        response(c, 5)["answer"], docs(c), c["question"]
    )
    for text in [
        "Bolaji has no Kafka or Spark experience and has never used it.",
        "Regarding Kafka architecture delivery, Bolaji has no Kafka experience and has never used Spark.",
    ]:
        assert "non_use_requires_explicit_negative_evidence" in constraint_failures(
            text, docs(c), c["question"]
        )
    assert constraint_failures(
        response(c, 5)["answer"],
        [Document(page_content="Tools: Kafka.")],
        c["question"],
    )


PHRASING = {
    c["id"]: c
    for c in json.loads(
        Path("tests/fixtures/scoped_phrasing_failures.json").read_text()
    )["cases"]
}


@pytest.mark.parametrize("cid", list(PHRASING))
def test_captured_scope_phrasing_retains_exact_final_independent_verification(
    monkeypatch, cid
):
    c = PHRASING[cid]
    responses = [response(c, 4)]
    if cid == "recruiter-hda-01-t1":
        assert not constraint_failures(response(c, 5)["answer"], docs(c), c["question"])
        responses += [response(c, 5), {"is_grounded": True}]
    generate = sdk(monkeypatch, responses)
    result = nodes.verify_grounding_node(state(c))
    assert result["grounding_checked"]
    assert generate.call_count == (3 if cid == "recruiter-hda-01-t1" else 1)
    assert result["answer"] in str(generate.call_args.kwargs["contents"])
    assert check_facts(
        result["answer"], c["golden_assertions"], result={**c, **result}
    )[0]


@pytest.mark.parametrize(
    "quantifier", ["multiple", "several", "two", "3", "two different", "12 separate"]
)
def test_coordinated_function_scope_does_not_authorize_explicit_multiplicity(
    quantifier,
):
    c = PHRASING["recruiter-hda-01-t1"]
    answer = f"He designed {quantifier} operating models, role charters, and competency frameworks for the data function."
    assert "operating_model_data_function_scope_missing" in constraint_failures(
        answer, docs(c), c["question"]
    )


@pytest.mark.parametrize(
    "answer",
    [
        "He designed operating models, role charters, and competency frameworks.",
        "He designed operating models, role charters, and competency frameworks for the whole company.",
        "He designed enterprise-wide operating models, role charters, and competency frameworks for the data function.",
        "He designed operating models. Separately, he designed the operating model for the data function.",
        "He designed operating models, role charters, and company-wide competency frameworks for the data function.",
    ],
)
def test_coordinated_scope_cannot_cleanse_unscoped_or_enterprise_claim(answer):
    c = PHRASING["recruiter-hda-01-t1"]
    assert "operating_model_data_function_scope_missing" in constraint_failures(
        answer, docs(c), c["question"]
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Bolaji has an explicit confirmed absence of experience with Kafka and has never used it.",
        "Regarding Kafka architecture delivery, he has a confirmed absence of experience with Kafka and has never used it.",
    ],
)
def test_confirmed_modifier_phrasing_still_requires_the_named_source_predicate(answer):
    c = PHRASING["recruiter-sda-05-t2"]
    assert "non_use_requires_explicit_negative_evidence" not in constraint_failures(
        answer, docs(c), c["question"]
    )
    absent = [
        Document(
            page_content="Bolaji has no Kafka experience at all.",
            metadata={"source": "20_owner_confirmed_production_scope.txt"},
        )
    ]
    assert "non_use_requires_explicit_negative_evidence" in constraint_failures(
        answer, absent, c["question"]
    )
    bad = answer.replace("with Kafka", "with Kafka and Spark")
    assert "non_use_requires_explicit_negative_evidence" in constraint_failures(
        bad, docs(c), c["question"]
    )
