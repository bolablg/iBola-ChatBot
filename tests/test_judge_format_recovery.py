"""Judge transport recovery must never select among completed grades."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts.run_eval import _JUDGE_DIAGNOSTICS, Judge, run_one

DIMENSIONS = ("relevance", "accuracy", "helpfulness", "faithfulness")


def response(data=None, *, text=None, finish="STOP", block=None):
    return SimpleNamespace(
        parsed=data,
        text=text,
        prompt_feedback=SimpleNamespace(block_reason=block),
        candidates=[SimpleNamespace(finish_reason=finish, safety_ratings=[])],
    )


def judge_with(*responses):
    judge = object.__new__(Judge)
    judge.model = "gemini-2.5-pro"
    judge.client = Mock()
    judge.client.models.generate_content.side_effect = responses
    return judge


def score(judge):
    return judge.score("question", "gold", "answer", "context")


@pytest.mark.parametrize("bad", [None, "", "{", "[]", '"string"', "null"])
def test_format_recovery_retains_first_completed_low_scores(bad):
    low = {key: 2 for key in DIMENSIONS}
    judge = judge_with(response(text=bad), response(text=json.dumps(low)))
    assert score(judge) == low | {"rationale": ""}
    assert judge.client.models.generate_content.call_count == 2
    diagnostics = _JUDGE_DIAGNOSTICS.get()
    assert diagnostics["retry_reasons"] == ["invalid_json_object"]
    assert diagnostics["decoding_source"] == "text"
    assert diagnostics["status"] == "success"


@pytest.mark.parametrize("data", [{key: 1 for key in DIMENSIONS}, {}])
def test_completed_low_or_empty_object_is_never_retried(data):
    judge = judge_with(response(data=data))
    actual = score(judge)
    assert {key: actual[key] for key in DIMENSIONS} == {
        key: data.get(key, 0) for key in DIMENSIONS
    }
    assert judge.client.models.generate_content.call_count == 1
    assert _JUDGE_DIAGNOSTICS.get()["retry_reasons"] == []
    assert _JUDGE_DIAGNOSTICS.get()["decoding_source"] == "parsed"


def test_invalid_score_conversion_does_not_trigger_resampling():
    judge = judge_with(response(data={"accuracy": "not a number"}))
    with pytest.raises(ValueError):
        score(judge)
    assert judge.client.models.generate_content.call_count == 1
    assert _JUDGE_DIAGNOSTICS.get()["status"] == "error"


def test_two_invalid_responses_remain_a_judge_error():
    judge = judge_with(response(), response(text="[]"))
    with pytest.raises(ValueError, match="after two calls"):
        score(judge)
    assert judge.client.models.generate_content.call_count == 2
    assert _JUDGE_DIAGNOSTICS.get()["status"] == "error"


class ProviderError(Exception):
    def __init__(self, code):
        self.code = code


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("transient_first", [True, False])
def test_provider_and_format_retries_share_one_two_call_budget(status, transient_first):
    responses = [ProviderError(status), response()]
    if not transient_first:
        responses.reverse()
    judge = judge_with(*responses)
    with (
        patch("scripts.run_eval.time.sleep"),
        pytest.raises((ProviderError, ValueError)),
    ):
        score(judge)
    assert judge.client.models.generate_content.call_count == 2
    assert len(_JUDGE_DIAGNOSTICS.get()["retry_reasons"]) == 1


@pytest.mark.parametrize(
    "metadata",
    [
        {"block": "SAFETY"},
        {"finish": "SAFETY"},
        {"finish": "RECITATION"},
        {"finish": "SPII"},
        {"finish": "NEW_UNKNOWN_REASON"},
    ],
)
@pytest.mark.parametrize("data", [None, {"accuracy": 10}])
def test_provider_denials_override_payload_and_are_never_retried(metadata, data):
    judge = judge_with(response(data=data, **metadata))
    with pytest.raises(ValueError, match="provider"):
        score(judge)
    assert judge.client.models.generate_content.call_count == 1
    assert _JUDGE_DIAGNOSTICS.get()["retry_reasons"] == []


def test_same_response_parsed_payload_wins_without_a_new_call():
    judge = judge_with(response(data={"accuracy": 3}, text=None))
    assert score(judge)["accuracy"] == 3
    assert judge.client.models.generate_content.call_count == 1


def test_non_json_parsed_object_uses_same_response_text():
    judge = judge_with(response(data={"object": object()}, text='{"accuracy": 4}'))
    assert score(judge)["accuracy"] == 4
    assert judge.client.models.generate_content.call_count == 1
    assert _JUDGE_DIAGNOSTICS.get()["decoding_source"] == "text"


def test_concurrent_rows_keep_their_own_attempt_diagnostics_even_when_facts_fail():
    judge = judge_with()

    def generate(**kwargs):
        if "question-first" in kwargs["contents"]:
            return response(data={"accuracy": 2})
        return response(text='{"accuracy": 5}')

    judge.client.models.generate_content.side_effect = generate
    target = SimpleNamespace(ask=lambda *args, **kwargs: {"answer": "actual answer"})

    def evaluate(name):
        return run_one(
            target,
            judge,
            {
                "id": name,
                "question": "question-" + name,
                "gold": "gold",
                "lang": "en",
                "category": "canonical",
                "must_contain": ["missing fact"],
            },
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(evaluate, ["first", "second"]))
    assert not first["fact_pass"] and not second["fact_pass"]
    assert first["scores"]["accuracy"] == 2
    assert second["scores"]["accuracy"] == 5
    assert first["judge_diagnostics"]["decoding_source"] == "parsed"
    assert second["judge_diagnostics"]["decoding_source"] == "text"
    assert judge.client.models.generate_content.call_count == 2
    unjudged = run_one(
        target,
        None,
        {"id": "third", "question": "q", "lang": "en", "category": "canonical"},
    )
    assert unjudged["judge_diagnostics"] is None
