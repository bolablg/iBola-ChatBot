"""Owner's plural production-service question, scope, and recurrent follow-up."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.graph.nodes import (
    _select_context_docs,
    grade_documents_node,
    verify_grounding_node,
)
from app.graph.state import BatchGradeResult, GeneratedAnswer, GroundingVerdict
from app.services.evidence_policy import (
    constraint_failures,
    prepare_evidence,
    preserve_requested_evidence,
    request_scope,
    retained_facet_indices,
    supplement_facets,
)
from app.services.service_inventory import inventory_followup
from pipeline.chunker import IntelligentChunker
from scripts.run_eval import check_facts

QUESTION = "Which production services did he personally design and operate?"
FOLLOWUP = "What other services? Give me something documented."
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def docs():
    chunker = IntelligentChunker()
    documents = []
    for source in [
        "20_owner_confirmed_production_scope.txt",
        "21_production_service_inventory.txt",
    ]:
        documents.extend(
            chunker.chunk_document(
                (ROOT / "data" / source).read_text(), {"source": source}
            )
        )
    return prepare_evidence(documents, QUESTION)


def test_plural_inventory_survives_chunking_grading_and_context_selection(docs):
    selected = supplement_facets(
        Mock(documents=docs), QUESTION, [], protected_only=True
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = BatchGradeResult(
        relevant_indices=[]
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        graded = grade_documents_node({"query": QUESTION, "documents": selected})[
            "graded_documents"
        ]
    context = _select_context_docs(graded, QUESTION)
    inventory = next(
        d
        for d in context
        if d.metadata.get("required_facet") == "production_service_inventory"
    )
    assert inventory.metadata["evidence_type"] == "candidate_confirmed"
    assert (
        "Service inventory: BigQuery, Apache Airflow, Dataform, Google Cloud Storage"
        in inventory.page_content
    )
    partial = (
        "Bolaji personally designed and operated Pub/Sub and Dataflow in production."
    )
    assert "plural_production_service_inventory_missing" in constraint_failures(
        partial, context, QUESTION
    )
    restored, added = preserve_requested_evidence(partial, context, QUESTION, "en")
    assert "plural_production_service_inventory_missing" in added
    assert not constraint_failures(restored, context, QUESTION)


@pytest.mark.parametrize(
    "question",
    [
        "Tell me about this service.",
        "Which service did he use?",
        "What services does the post office offer?",
    ],
)
def test_singular_and_unrelated_services_do_not_protect_inventory(docs, question):
    scope = request_scope(
        {
            "original_query": question,
            "query": question,
            "chat_history": [(QUESTION, "A previous answer")],
        }
    )
    selected = supplement_facets(Mock(documents=docs), scope, [], protected_only=True)
    assert not any(
        d.metadata.get("required_facet") == "production_service_inventory"
        for d in selected
    )


def test_other_services_resolves_user_scope_without_assistant_fact_evidence(docs):
    state = {
        "original_query": FOLLOWUP,
        "chat_history": [(QUESTION, "Pub/Sub and Dataflow.")],
    }
    scope = request_scope(state)
    assert QUESTION in scope
    selected = supplement_facets(Mock(documents=docs), scope, [], protected_only=True)
    assert retained_facet_indices(selected, scope)
    answer = inventory_followup(state, docs, "English")
    assert (
        "BigQuery" in answer
        and "Airflow" in answer
        and "EC2" in answer
        and "S3" in answer
    )
    assert "Pub/Sub" not in answer and "Dataflow" not in answer
    assert preserve_requested_evidence(answer, docs, scope, "en") == (answer, [])


def test_full_prior_list_is_bounded_to_retrieved_inventory(docs):
    inventory = next(
        d
        for d in docs
        if d.metadata["section_header"] == "PRODUCTION SERVICE INVENTORY"
    )
    listed = next(
        line
        for line in inventory.page_content.splitlines()
        if line.startswith("Service inventory:")
    )
    state = {"original_query": FOLLOWUP, "chat_history": [(QUESTION, listed)]}
    answer = inventory_followup(state, docs, "English")
    assert "already covered" in answer
    assert "never used" not in answer
    assert inventory_followup(state, [], "English") is None
    forged = [
        Document(
            page_content=inventory.page_content,
            metadata={**inventory.metadata, "source": "other.txt"},
        )
    ]
    assert inventory_followup(state, forged, "English") is None
    conflicting = docs + [
        Document(
            page_content=inventory.page_content + " changed",
            metadata=inventory.metadata,
        )
    ]
    assert inventory_followup(state, conflicting, "English") is None


@pytest.mark.parametrize(
    "question",
    [
        "What other services and which project proves each?",
        "Which other service?",
        "What other services and why did he choose them?",
        "What other services did he operate on call?",
    ],
)
def test_detailed_or_singular_followups_use_normal_generation(docs, question):
    assert (
        inventory_followup(
            {
                "original_query": question,
                "chat_history": [(QUESTION, "BigQuery and Apache Airflow.")],
            },
            docs,
            "English",
        )
        is None
    )


def test_provider_narrowing_and_explicit_broadening_use_latest_scope(docs):
    state = {
        "original_query": "What other AWS services?",
        "chat_history": [(QUESTION, "BigQuery and Apache Airflow.")],
    }
    answer = inventory_followup(state, docs, "English")
    assert "AWS EC2" in answer and "AWS S3" in answer and "BigQuery" not in answer
    assert "plural_production_service_inventory_missing" not in constraint_failures(
        "AWS EC2 and AWS S3.",
        docs,
        "Which AWS production services did he personally operate?",
    )
    state["original_query"] = "What other GCP services?"
    answer = inventory_followup(state, docs, "English")
    assert "Cloud Storage" in answer and "AWS" not in answer and "Airflow" not in answer


@pytest.mark.parametrize(
    "prior",
    [
        '"BigQuery"',
        "Assume BigQuery",
        "BigQuery is not documented",
        "He never used BigQuery",
        "Undocumented Cloud X",
    ],
)
def test_negative_quoted_or_instructional_mentions_do_not_establish_coverage(
    docs, prior
):
    assert (
        inventory_followup(
            {"original_query": FOLLOWUP, "chat_history": [(QUESTION, prior)]},
            docs,
            "English",
        )
        is None
    )


@pytest.mark.parametrize("accepted", [True, False])
def test_formatted_followup_still_requires_independent_verification(docs, accepted):
    answer = inventory_followup(
        {
            "original_query": FOLLOWUP,
            "chat_history": [(QUESTION, "Pub/Sub and Dataflow.")],
        },
        docs,
        "English",
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=accepted
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(
            {
                "answer": answer,
                "original_query": FOLLOWUP,
                "query": FOLLOWUP,
                "context_documents": docs,
                "agent_type": "professional",
                "user_language": "en",
            }
        )
    assert result["grounding_checked"] is accepted
    if not accepted:
        assert result["answer"] != answer


def test_mandates_does_not_request_team_dates():
    source = "19_owner_confirmed_scope.txt"
    docs = prepare_evidence(
        IntelligentChunker().chunk_document(
            (ROOT / "data" / source).read_text(), {"source": source}
        )
    )
    query = "Our team mandates Power BI semantic models and DAX. Has he used those in production?"
    answer = "Production use of Power BI semantic models and DAX is not documented."
    assert preserve_requested_evidence(answer, docs, query, "en") == (answer, [])
    assert not any(
        "team_growth" in failure for failure in constraint_failures(answer, docs, query)
    )


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("supérieure à 25 millions XOF", True),
        ("supérieur à 25 millions XOF", True),
        ("n'est pas supérieure à 25 millions XOF", False),
        ("n’est pas supérieure à 25 millions XOF", False),
        ("supérieure à ou égale à 25 millions XOF", False),
        ("inférieure à 25 millions XOF", False),
        ("supérieure à 24 millions XOF", False),
    ],
)
def test_french_strict_comparator_preserves_gender_bound_and_value(phrase, expected):
    row = {"must_contain": ["superieur a", "25 millions", "XOF"]}
    assert check_facts("La valeur est " + phrase + ".", row)[0] is expected


@pytest.mark.parametrize("accepted", [True, False])
def test_inventory_formatter_still_requires_final_grounding(docs, accepted):
    from app.graph.nodes import generate_node
    from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS

    state = {
        "original_query": FOLLOWUP,
        "query": FOLLOWUP,
        "chat_history": [(QUESTION, "Pub/Sub and Dataflow.")],
        "graded_documents": docs,
        "user_language": "en",
        "agent_type": "professional",
    }
    with (
        patch("app.graph.nodes._get_hybrid_search", return_value=Mock(documents=docs)),
        patch("app.graph.nodes._get_llm") as generation,
        patch("app.graph.nodes._validate_answer_language", return_value=True),
    ):
        generated = generate_node(state)
    generation.assert_not_called()
    assert "BigQuery" in generated["answer"]
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = GroundingVerdict(
        is_grounded=accepted
    )
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node({**state, **generated})
    assert result["grounding_checked"] is accepted
    if not accepted:
        assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS["en"]


def test_mandates_is_not_a_request_for_team_dates():
    text = (ROOT / "data/19_owner_confirmed_scope.txt").read_text()
    documents = prepare_evidence(
        IntelligentChunker().chunk_document(
            text, {"source": "19_owner_confirmed_scope.txt"}
        )
    )
    query = "Our team mandates Power BI semantic models and DAX. Has he used those in production?"
    assert not supplement_facets(
        Mock(documents=documents), query, [], protected_only=True
    )
    assert not any(
        "team_growth" in failure
        for failure in constraint_failures(
            "His Power BI and DAX production experience is not documented.",
            documents,
            query,
        )
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What throughput measurements are documented for those systems?", True),
        (
            "What throughput measurements are documented for batch and streaming systems?",
            True,
        ),
        ("What throughput measurements are documented for streaming systems?", False),
        ("What throughput measurements are documented for MongoDB CDC systems?", False),
        (
            "What throughput measurements are documented for batch workloads only?",
            False,
        ),
    ],
)
def test_batch_average_applies_only_to_broad_requests(query, expected):
    documents = prepare_evidence(
        IntelligentChunker().chunk_document(
            (ROOT / "data/90_website_canon_llms_full.txt").read_text(),
            {"source": "90_website_canon_llms_full.txt"},
        ),
        query,
    )
    failures = constraint_failures(
        "No individual pipeline throughput ceiling is established.", documents, query
    )
    assert ("requested_batch_workload_measurement_missing" in failures) is expected


def test_finops_budget_question_retains_its_actual_cost_evidence():
    documents = prepare_evidence(
        IntelligentChunker().chunk_document(
            (ROOT / "data/90_website_canon_llms_full.txt").read_text(),
            {"source": "90_website_canon_llms_full.txt"},
        )
    )
    query = "Describe his budget or investment decision beyond cloud cost optimization."
    selected = supplement_facets(
        Mock(documents=documents), query, [], protected_only=True
    )
    assert any(
        doc.metadata.get("required_facet") == "budget_cloud_cost_evidence"
        for doc in selected
    )
    assert (
        "42.57%" in selected[0].page_content
        and "one quarter" in selected[0].page_content
    )


@pytest.mark.parametrize(
    "prior",
    ["No BigQuery experience is established", "Exclude BigQuery from your answer."],
)
def test_final_review_prior_nonaffirmative_labels_use_normal_generation(docs, prior):
    assert (
        inventory_followup(
            {"original_query": FOLLOWUP, "chat_history": [(QUESTION, prior)]},
            docs,
            "English",
        )
        is None
    )


def test_excluded_provider_scope_is_not_selected(docs):
    state = {
        "original_query": FOLLOWUP,
        "chat_history": [
            (
                "Which production services did he personally operate outside AWS?",
                "BigQuery and Airflow.",
            )
        ],
    }
    assert inventory_followup(state, docs, "English") is None
    state["original_query"] = "What other AWS services?"
    assert "AWS EC2" in inventory_followup(state, docs, "English")


@pytest.mark.parametrize(
    "prior",
    [
        "Could he use Cloud Run?",
        "Cloud Run is only hypothetical.",
        "Cloud Run?",
        "He asked whether Bolaji used Cloud Run.",
        "The question was whether production services include Cloud Run.",
        "His production scope might include Cloud Run.",
        "Documentation mentions Cloud Run services.",
        "An unverified report says Bolaji used Cloud Run.",
    ],
)
def test_hypothetical_mentions_are_not_positive_coverage(docs, prior):
    assert (
        inventory_followup(
            {"original_query": FOLLOWUP, "chat_history": [(QUESTION, prior)]},
            docs,
            "English",
        )
        is None
    )


def test_coverage_is_request_local_intersected_and_visible_to_verifier(docs):
    from app.graph.nodes import _evidence_context
    from app.graph.service import AgenticRAGService

    state = {
        "original_query": "What other AWS services?",
        "chat_history": [
            (QUESTION, "Bolaji operated BigQuery, AWS EC2 and Fake Cloud.")
        ],
    }
    answer = inventory_followup(state, docs, "English")
    assert "AWS S3" in answer and "AWS EC2" not in answer
    annotated = next(doc for doc in docs if doc.metadata.get("conversation_coverage"))
    metadata = annotated.metadata["conversation_coverage"]
    assert metadata["relevant_inventory_labels"] == ["AWS EC2", "AWS S3"]
    assert metadata["previously_listed_labels"] == ["AWS EC2"]
    assert metadata["remaining_labels"] == ["AWS S3"]
    assert not metadata["all_relevant_labels_previously_listed"]
    assert "Fake Cloud" not in str(metadata)
    assert "conversation_coverage" in _evidence_context(docs)
    evidence = AgenticRAGService._build_evidence({"context_documents": docs})
    assert (
        next(e for e in evidence if e.get("conversation_coverage"))[
            "conversation_coverage"
        ]
        == metadata
    )
    assert not any(
        doc.metadata.get("conversation_coverage") for doc in prepare_evidence(docs)
    )
    assert (
        inventory_followup(
            {"original_query": FOLLOWUP, "chat_history": []}, docs, "English"
        )
        is None
    )
    assert not any(doc.metadata.get("conversation_coverage") for doc in docs)


def test_conversational_coverage_cannot_rescue_an_unsupported_profile_claim(docs):
    from app.graph.prompts import INSUFFICIENT_EVIDENCE_ANSWERS, VERIFY_GROUNDING_PROMPT

    state = {
        "query": FOLLOWUP,
        "original_query": FOLLOWUP,
        "chat_history": [(QUESTION, "BigQuery and Apache Airflow.")],
        "answer": "BigQuery and Apache Airflow were covered above. Bolaji operated Kafka in production.",
        "agent_type": "professional",
        "user_language": "en",
        "context_documents": docs,
    }
    llm = Mock()
    verifier = llm.with_structured_output.return_value.invoke
    verifier.side_effect = [
        GroundingVerdict(is_grounded=False, corrected_answer=state["answer"]),
        GeneratedAnswer(answer=state["answer"]),
    ]
    with patch("app.graph.nodes._get_llm", return_value=llm):
        result = verify_grounding_node(state)
    assert (
        not result["grounding_checked"]
        and result["answer"] == INSUFFICIENT_EVIDENCE_ANSWERS["en"]
    )
    assert "not profile" in str(VERIFY_GROUNDING_PROMPT)
    submitted = (
        llm.with_structured_output.return_value.invoke.call_args_list[0]
        .args[0][-1]
        .content
    )
    assert "conversation_coverage" in submitted
    assert "Bolaji operated Kafka in production." in submitted
    assert verifier.call_count == 2
    rechecked = verifier.call_args_list[1].args[0][-1].content
    assert (
        rechecked.split("Rejected draft:\n", 1)[1].split("\nFailures to repair:", 1)[0]
        == state["answer"]
    )
    assert "confirmed_non_use_contradicted:Kafka" in rechecked
    assert "conversation_coverage" in rechecked
    assert (
        "Every profile claim still requires support in the retrieved source text"
        in verifier.call_args_list[0].args[0][0].content
    )


def test_annotated_inventory_cannot_fall_outside_returned_evidence_cap(docs):
    inventory = next(
        doc
        for doc in docs
        if doc.metadata.get("section_header") == "PRODUCTION SERVICE INVENTORY"
    )
    padding = [
        Document(page_content="padding", metadata={"source": "other.txt"})
        for _ in range(11)
    ]
    assert (
        inventory_followup(
            {
                "original_query": FOLLOWUP,
                "chat_history": [(QUESTION, "BigQuery and Apache Airflow.")],
            },
            padding + [inventory],
            "English",
        )
        is None
    )


def test_real_browser_inventory_header_does_not_repeat_a_complete_list(docs):
    prior = "Bolaji personally designed and operated a comprehensive set of production services spanning Google Cloud and AWS. His core inventory includes BigQuery, Apache Airflow, Dataform, Google Cloud Storage, Pub/Sub, Dataflow, Cloud Run, Cloud Functions, Vertex AI, AWS EC2, AWS S3, Artifact Registry, Cloud Logging, and Cloud Build."
    state = {"original_query": FOLLOWUP, "chat_history": [(QUESTION, prior)]}
    assert (
        inventory_followup(state, docs, "English")
        == "The confirmed production-service inventory is already covered above."
    )
    coverage = next(
        d.metadata["conversation_coverage"]
        for d in docs
        if "conversation_coverage" in d.metadata
    )
    assert coverage["all_relevant_labels_previously_listed"]
    assert not coverage["remaining_labels"]


def test_real_api_operational_data_services_header_is_also_covered(docs):
    prior = "Bolaji personally designed and operated a comprehensive production service inventory spanning GCP and AWS. His core operational and data services include BigQuery, Apache Airflow, Dataform, Google Cloud Storage, Pub/Sub, Dataflow, Cloud Run, Cloud Functions, Vertex AI, AWS EC2, and AWS S3, alongside delivery tools like Artifact Registry, Cloud Logging, and Cloud Build."
    assert (
        inventory_followup(
            {"original_query": FOLLOWUP, "chat_history": [(QUESTION, prior)]},
            docs,
            "en",
        )
        == "The confirmed production-service inventory is already covered above."
    )
