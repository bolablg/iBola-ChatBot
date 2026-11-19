"""Retain scoped known facts after correction, without trusting failed rechecks."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS
from app.graph.state import GroundingVerdict
from scripts.run_eval import check_facts, load_golden

CASES = json.loads(
    (Path(__file__).parent / "fixtures/release_scope_corrections.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("accepted", [True, False])
def test_actual_release_scope_corrections_require_full_content_reverification(
    case, accepted
):
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(
            is_grounded=False,
            unsupported_claims=case["observed_unsupported_claims"],
            corrected_answer=case["corrected_answer"],
        ),
        GroundingVerdict(
            is_grounded=accepted, corrected_answer="Unverified second repair."
        ),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": case["draft"],
                "query": case["question"],
                "user_language": case["language"],
                "agent_type": "professional",
                "context_documents": [Document(**doc) for doc in case["documents"]],
                "confidence": 0.9,
            }
        )
    assert invoke.call_count == 2
    assert result["grounding_checked"] is accepted
    assert "Unverified second repair" not in result["answer"]
    recheck = "\n".join(
        str(message.content) for message in invoke.call_args_list[-1].args[0]
    )
    assert case["corrected_answer"] in recheck
    if accepted:
        assert result["answer"] == case["corrected_answer"]
        row = next(row for row in load_golden() if row["id"] == case["id"])
        assert check_facts(
            result["answer"],
            row,
            result={
                **result,
                "evidence": case["documents"],
                "agent_type": "professional",
            },
        )[0]
    else:
        assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS[case["language"]]
        assert result["confidence"] == 0.0
        assert result["answer"] != case["draft"]
