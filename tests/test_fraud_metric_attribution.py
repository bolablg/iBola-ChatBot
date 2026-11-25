"""Owner-confirmed fraud methods must not inherit another project's outcome."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import verify_grounding_node
from app.graph.state import GroundingVerdict
from app.services.evidence_policy import constraint_failures, prepare_evidence

CASE = json.loads(Path("tests/fixtures/fraud_metric_attribution.json").read_text())
RULE = "fraud_metrics_exclude_ml_attribution"


def evidence():
    return [
        Document(
            page_content=Path("data/18_confirmed_clarifications.txt")
            .read_text()
            .split("OWNER-CONFIRMED GOZEM MONEY FRAUD METRIC ATTRIBUTION", 1)[1],
            metadata={
                "source": "18_confirmed_clarifications.txt",
                "section_header": "OWNER-CONFIRMED GOZEM MONEY FRAUD METRIC ATTRIBUTION",
                "evidence_type": "candidate_confirmed",
            },
        )
    ]


@pytest.mark.parametrize(
    "answer",
    [
        CASE["captured_result"]["answer"],
        "ML screening and micro-batch rules were combined. They surfaced 150 incidents per month.",
        "ML screening and micro-batch rules were combined. These systems flagged fraud worth over 25M XOF.",
        "ML-based fraud screening surfaces approximately 150 incidents per month.",
        "The micro-batch pipeline and ML-based fraud screening systems surface 150 incidents per month and flagged fraud worth over 25M XOF.",
        "ML screening contributed to the fraud discovery worth over 25M XOF.",
        "A fraud worth over 25M XOF was flagged by ML screening.",
        "ML screening was used. This system surfaced 150 incidents per month.",
        "ML screening did not surface 150 incidents, but it flagged fraud worth over 25M XOF.",
        "Le filtrage ML a signale environ 150 incidents par mois.",
    ],
)
def test_direct_shared_partial_reverse_and_pronominal_ml_attribution_are_rejected(
    answer,
):
    assert RULE in constraint_failures(answer, evidence())


@pytest.mark.parametrize(
    "answer",
    [
        "Micro-batches, rules and EDA surfaced approximately 150 incidents per month and flagged fraud worth over 25M XOF. ML screening did not contribute to either result.",
        "ML screening did not contribute to the 150 incidents per month or the fraud discovery worth over 25M XOF.",
        "ML screening did not surface 150 incidents per month or discover fraud worth over 25M XOF.",
        "ML screening didn't surface 150 incidents per month or discover fraud worth over 25M XOF.",
        "The ML screening and payment automation project reports 7x faster field-agent payments, and micro-batches surfaced 150 incidents per month.",
        "ML screening is a separate project. Rules surfaced 150 incidents per month. They flagged fraud worth over 25M XOF.",
        "Unlike ML screening, the micro-batch pipeline surfaced 150 incidents per month.",
        "Micro-batch monitoring, not ML screening, surfaced 150 incidents per month.",
        "ML screening is a separate project, and micro-batch monitoring surfaced 150 incidents per month.",
        "ML screening is documented separately, while micro-batch monitoring surfaced 150 incidents per month.",
        "The ML screening and payment automation project reports 7x faster field-agent payment processing. Micro-batch monitoring surfaced 150 incidents per month.",
        "Fraud worth over 25M XOF was not discovered by ML screening.",
        "Le filtrage ML n'a pas contribue aux 150 incidents par mois.",
    ],
)
def test_correct_separation_explicit_negation_and_separate_payment_result_remain_valid(
    answer,
):
    assert RULE not in constraint_failures(answer, evidence())


@pytest.mark.parametrize("approved", [True, False])
def test_corrected_owner_attribution_still_requires_independent_verification(approved):
    corrected = "Micro-batches, rules, and EDA surfaced approximately 150 incidents per month and flagged fraud worth over 25M XOF. ML screening did not contribute to either result."
    llm = Mock()
    invoke = llm.with_structured_output.return_value.invoke
    invoke.side_effect = [
        GroundingVerdict(is_grounded=True, corrected_answer=corrected),
        GroundingVerdict(is_grounded=approved),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": CASE["captured_result"]["answer"],
                "query": "Which fraud results came from ML screening?",
                "agent_type": "professional",
                "user_language": "en",
                "context_documents": evidence(),
            }
        )
    assert invoke.call_count == 2
    assert result["grounding_checked"] is approved
    assert (result["answer"] == corrected) is approved


def test_scope_metadata_separates_incident_period_from_fraud_discovery_value():
    result = prepare_evidence(evidence(), "fraud")
    scope = result[0].metadata["metric_scope"]
    assert "per month" in scope["incident_rate"]
    assert "not an established monthly total" in scope["fraud_discovery_value"]
    assert (
        "ML-based fraud screening did not contribute" in scope["excluded_attribution"]
    )
