"""The quality gate must reject routing failures and unjudged reports."""

import json
import sys

from scripts import eval_gate
from scripts.run_eval import check_facts


def test_regex_route_evidence_and_grounding_requirements_are_enforced():
    row = {
        "must_match": [r"4.*14"],
        "forbidden_agent_types": ["opportunity"],
        "require_evidence": True,
        "require_grounded": True,
    }
    ok, failures = check_facts("14 people", row, result={"agent_type": "opportunity"})
    assert not ok and len(failures) == 4
    assert check_facts(
        "4 to 14+",
        row,
        result={
            "agent_type": "professional",
            "evidence": [{"source": "confirmed"}],
            "grounding_checked": True,
        },
    )[0]


def test_critical_recruiter_failure_cannot_hide_in_good_aggregate(
    tmp_path, monkeypatch
):
    baseline = tmp_path / "baseline.json"
    report = tmp_path / "report.json"
    dims = {dim: 10 for dim in eval_gate.JUDGED_DIMENSIONS}
    baseline.write_text(json.dumps({"aggregates": dims, "gate": {}}))
    data = {
        "aggregates": dims,
        "results": [
            {"id": "recruiter-1", "tags": ["recruiter_regression"], "fact_pass": False}
        ],
    }
    report.write_text(json.dumps(data))
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_gate", "--report", str(report), "--baseline", str(baseline), "--strict"],
    )
    assert eval_gate.main() == 1
    data["results"][0]["fact_pass"] = True
    data["aggregates"]["judge_errors"] = 1
    report.write_text(json.dumps(data))
    assert eval_gate.main() == 1
    data["aggregates"]["judge_errors"] = 0
    del data["aggregates"]["accuracy"]
    report.write_text(json.dumps(data))
    assert eval_gate.main() == 1
    data["aggregates"]["accuracy"] = 10
    report.write_text(json.dumps(data))
    assert eval_gate.main() == 0


def test_unknown_wording_must_be_tied_to_each_requested_detail():
    row = {"must_mark_unknown": ["on.call", "compensation"]}
    assert not check_facts(
        "Compensation is not available. He welcomes on-call work.", row
    )[0]
    assert check_facts(
        "On-call willingness and compensation are not specified in the record.", row
    )[0]
    assert check_facts(
        "The record does not specify on-call willingness or compensation.", row
    )[0]


def test_localized_head_title_requires_its_disambiguating_period():
    from scripts.run_eval import load_golden

    row = next(r for r in load_golden() if r["id"] == "d1-2023-fr")
    assert check_facts(
        "En 2023 il etait Responsable des donnees d'octobre 2022 a mars 2025.", row
    )[0]
    assert not check_facts("Il etait Responsable des donnees de 2025 a 2026.", row)[0]
    assert not check_facts("Il etait Data Director d'octobre 2022 a mars 2025.", row)[0]


def test_current_head_question_accepts_complete_correct_dates_without_extra_title():
    from scripts.run_eval import load_golden

    row = next(r for r in load_golden() if r["id"] == "tmp-009")
    assert check_facts(
        "No. His Head tenure concluded in March 2025 and his Gozem employment ended in July 2026.",
        row,
    )[0]
    assert not check_facts("Yes, he is currently Head of Data.", row)[0]
    assert not check_facts(
        "No. His Head tenure concluded in March 2026 and Gozem employment ended in July 2027.",
        row,
    )[0]


def test_judge_retries_transient_failures_once_and_preserves_real_failure():
    from unittest.mock import Mock, patch

    import pytest

    from scripts.run_eval import Judge

    class ProviderFailure(Exception):
        def __init__(self, code):
            self.code = code

    judge = object.__new__(Judge)
    judge.model = "test-model"
    judge.client = Mock()
    generate = judge.client.models.generate_content
    response = Mock(text=json.dumps({key: 9 for key in eval_gate.JUDGED_DIMENSIONS}))
    for status in (429, 503):
        generate.reset_mock()
        generate.side_effect = [ProviderFailure(status), response]
        with patch("scripts.run_eval.time.sleep") as sleep:
            assert (
                judge.score("question", "gold", "answer", "evidence")["accuracy"] == 9
            )
        assert generate.call_count == 2
        sleep.assert_called_once_with(0.5)
        generate.reset_mock()
        generate.side_effect = ProviderFailure(status)
        with patch("scripts.run_eval.time.sleep"), pytest.raises(ProviderFailure):
            judge.score("question", "gold", "answer", "evidence")
        assert generate.call_count == 2
    generate.reset_mock()
    generate.side_effect = ProviderFailure(401)
    with patch("scripts.run_eval.time.sleep") as sleep, pytest.raises(ProviderFailure):
        judge.score("question", "gold", "answer", "evidence")
    assert generate.call_count == 1
    sleep.assert_not_called()
