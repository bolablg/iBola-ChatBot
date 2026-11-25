"""Capabilities and technology evidence remain distinct in production retrieval."""

from pathlib import Path

import pytest

from pipeline.chunker import IntelligentChunker

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "05_skills_and_tools.txt"
CAPABILITY_HEADERS = {
    "CORE SKILLS AND CAPABILITIES:",
    "CAPABILITY SUMMARY IN ENGLISH:",
    "COMPETENCES PROFESSIONNELLES EN FRANCAIS:",
}


@pytest.fixture
def source_chunks():
    chunker = IntelligentChunker()
    return chunker.chunk_document(
        (ROOT / "data" / SOURCE).read_text(), {"source": SOURCE}
    )


@pytest.fixture
def lexical_search():
    # Import after the shared external-service fixtures have been installed.
    from app.services.advanced_rag import BM25Index

    chunker = IntelligentChunker()
    documents = []
    for path in sorted((ROOT / "data").glob("*.txt")):
        documents.extend(
            chunker.chunk_document(path.read_text(), {"source": path.name})
        )
    search = BM25Index()
    search.build(documents)
    return search


def test_chunk_boundaries_keep_capabilities_separate_from_technology_entries(
    source_chunks,
):
    capabilities = [
        doc
        for doc in source_chunks
        if doc.metadata["section_header"] in CAPABILITY_HEADERS
    ]
    assert {
        doc.metadata["section_header"] for doc in capabilities
    } == CAPABILITY_HEADERS
    for doc in capabilities:
        body = doc.page_content.split("\n\n", 1)[1]
        assert "Used at:" not in body
        assert all(
            tool not in body for tool in ("BigQuery", "Apache Airflow", "Python")
        )
        assert doc.metadata["source"] == SOURCE
    technologies = next(
        doc
        for doc in source_chunks
        if doc.metadata["section_header"] == "PROGRAMMING LANGUAGES:"
    )
    assert "Primary programming language: Python" in technologies.page_content
    assert "SQL (advanced, 10+ years)" in technologies.page_content


@pytest.mark.parametrize(
    "question",
    [
        "What are Bolaji's key skills?",
        "Summarize his core capabilities.",
        "Describe his professional competencies.",
        "Quelles sont les compétences clés de Bolaji ?",
        "Quelles sont ses competences professionnelles ?",
    ],
)
def test_capability_paraphrases_retrieve_capability_evidence(lexical_search, question):
    results = lexical_search.search(question, top_k=8)
    assert any(
        doc.metadata["source"] == SOURCE
        and doc.metadata["section_header"] in CAPABILITY_HEADERS
        for doc, _ in results
    )


@pytest.mark.parametrize(
    "question,required_terms",
    [
        ("Which programming languages does Bolaji use?", ("Python", "SQL")),
        (
            "Which tools support his data pipeline orchestration work?",
            ("Apache Airflow",),
        ),
        (
            "Which cloud platforms has Bolaji used?",
            ("Google Cloud Platform", "Amazon Web Services"),
        ),
    ],
)
def test_technology_queries_retain_their_application_evidence(
    lexical_search, question, required_terms
):
    evidence = "\n".join(
        doc.page_content for doc, _ in lexical_search.search(question, top_k=8)
    )
    assert all(term in evidence for term in required_terms)
    assert "Used at:" in evidence


def test_technology_details_keep_project_ownership_limits(source_chunks):
    text = "\n".join(doc.page_content for doc in source_chunks)
    assert (
        "does not establish sole implementation of the recommendation model, API, or mobile integration"
        in text
    )
    assert "comparison does not establish direct dbt use" in text
    assert (
        "does not establish Power BI, DAX, or Tableau production use or proficiency"
        in text
    )
