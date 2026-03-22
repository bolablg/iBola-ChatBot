"""
Specialized retrievers for different agent types.
"""

import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from utils.embedder import get_embeddings


def get_base_retriever():
    """Get the base retriever from the existing vector store."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.DB_PATH, embedding_function=embeddings
    )
    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20}
    )


def get_professional_retriever():
    """Retriever specialized for professional skills and experiences."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.DB_PATH, embedding_function=embeddings
    )

    def filter_professional_docs(doc):
        """Filter documents related to professional experience, skills, projects."""
        content = doc.page_content.lower()
        metadata = doc.metadata

        # Professional keywords
        prof_keywords = [
            "gozem",
            "rintio",
            "experience",
            "project",
            "skill",
            "technology",
            "data",
            "analytics",
            "machine learning",
            "ai",
            "engineering",
            "cloud",
            "bigquery",
            "python",
            "sql",
            "airflow",
            "looker",
            "dataform",
            "vertex",
            "gcp",
            "aws",
            "leadership",
            "team",
            "automation",
            "optimization",
        ]

        # Education keywords to exclude
        edu_keywords = [
            "master",
            "bachelor",
            "degree",
            "university",
            "diploma",
            "transcript",
            "statistics",
            "econometrics",
            "icmpa",
            "certificate",
            "certification",
        ]

        prof_score = sum(1 for keyword in prof_keywords if keyword in content)
        edu_score = sum(1 for keyword in edu_keywords if keyword in content)

        # Include if professional score > education score or if it's clearly professional
        return prof_score > edu_score or any(
            keyword in content for keyword in ["gozem", "rintio", "project"]
        )

    # Create a custom retriever that filters results
    class ProfessionalRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager=None
        ) -> List[Document]:
            # Get documents from base retriever
            base_retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20}
            )
            docs = base_retriever.invoke(query)

            # Filter the documents
            filtered_docs = [doc for doc in docs if filter_professional_docs(doc)]

            # If we have filtered results, return them
            if filtered_docs:
                return filtered_docs[:6]  # Return top 6 filtered results

            # Fallback to base retriever if no filtered results
            return docs[:6]

    return ProfessionalRetriever()


def get_education_retriever():
    """Retriever specialized for educational background."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.DB_PATH, embedding_function=embeddings
    )

    def filter_education_docs(doc):
        """Filter documents related to education, degrees, studies."""
        content = doc.page_content.lower()
        metadata = doc.metadata

        # Education keywords
        edu_keywords = [
            "master",
            "bachelor",
            "degree",
            "university",
            "diploma",
            "transcript",
            "statistics",
            "econometrics",
            "icmpa",
            "unesco",
            "abomey",
            "calavi",
            "education",
            "study",
            "academic",
            "thesis",
            "dissertation",
        ]

        # Professional keywords to exclude
        prof_keywords = [
            "gozem",
            "rintio",
            "project",
            "skill",
            "technology",
            "data",
            "analytics",
            "machine learning",
            "ai",
            "engineering",
            "cloud",
            "bigquery",
            "python",
            "sql",
            "airflow",
            "looker",
            "dataform",
            "vertex",
            "gcp",
            "aws",
        ]

        edu_score = sum(1 for keyword in edu_keywords if keyword in content)
        prof_score = sum(1 for keyword in prof_keywords if keyword in content)

        # Prioritize bachelor's degree information
        if "bachelor" in content or "College of Economics & Management" in content:
            return True

        # Include if education score > professional score
        return edu_score > prof_score or any(
            keyword in content for keyword in ["master", "university", "degree"]
        )

    # Create a custom retriever that filters results
    class EducationRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager=None
        ) -> List[Document]:
            # Get more documents from base retriever to ensure we capture bachelor's content
            base_retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 12, "fetch_k": 30}
            )
            docs = base_retriever.invoke(query)

            # Filter the documents
            filtered_docs = [doc for doc in docs if filter_education_docs(doc)]

            # If we have filtered results, return them
            if filtered_docs:
                return filtered_docs[
                    :12
                ]  # Return more docs to ensure bachelor's content is included

            # Fallback to base retriever if no filtered results
            return docs[:6]

    return EducationRetriever()


def get_classification_retriever():
    """Advanced retriever for classification agent with internet access and broader knowledge base."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.DB_PATH, embedding_function=embeddings
    )

    def filter_classification_docs(doc):
        """Filter documents that are useful for classification tasks."""
        content = doc.page_content.lower()
        metadata = doc.metadata

        # Classification-relevant keywords (broader than other agents)
        classification_keywords = [
            # Professional
            "work",
            "job",
            "career",
            "project",
            "skill",
            "experience",
            "company",
            "role",
            "achievement",
            "leadership",
            "team",
            "automation",
            "optimization",
            "business",
            # Education
            "degree",
            "university",
            "study",
            "academic",
            "school",
            "course",
            "education",
            "master",
            "bachelor",
            "diploma",
            "transcript",
            "grade",
            "gpa",
            "thesis",
            "dissertation",
            "statistics",
            "econometrics",
            "mathematics",
            "analysis",
            "research",
            # Learning
            "learn",
            "study",
            "how to",
            "tutorial",
            "course",
            "training",
            "advice",
            "guide",
            "beginner",
            "start",
            "career path",
            "skill development",
            "resources",
            "practice",
            "improve",
            "tips",
            "recommend",
            "teaching",
            "education",
            "knowledge",
            # General context
            "professional",
            "academic",
            "career",
            "development",
            "expertise",
            "qualification",
            "certification",
            "competence",
            "capability",
            "background",
            "experience",
        ]

        # Include documents that contain classification-relevant keywords
        return any(keyword in content for keyword in classification_keywords)

    class ClassificationRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager=None
        ) -> List[Document]:
            # Get more documents for broader context understanding
            base_retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 15, "fetch_k": 40}
            )
            docs = base_retriever.invoke(query)

            # Filter documents for classification relevance
            filtered_docs = [doc for doc in docs if filter_classification_docs(doc)]

            # If we have filtered results, return them
            if filtered_docs:
                return filtered_docs[:10]  # Return top 10 for broader context

            # Fallback to base retriever if no filtered results
            return docs[:10]

    return ClassificationRetriever()


def get_learning_retriever():
    """Retriever specialized for learning advice and skill development."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.DB_PATH, embedding_function=embeddings
    )

    def filter_learning_docs(doc):
        """Filter documents that could be useful for learning advice."""
        content = doc.page_content.lower()
        metadata = doc.metadata

        # Learning/advice keywords
        learning_keywords = [
            "learn",
            "study",
            "skill",
            "technology",
            "python",
            "sql",
            "data",
            "analytics",
            "machine learning",
            "ai",
            "cloud",
            "bigquery",
            "airflow",
            "looker",
            "dataform",
            "vertex",
            "gcp",
            "aws",
            "career",
            "development",
            "growth",
            "experience",
            "project",
            "challenge",
            "solution",
            "approach",
            "methodology",
        ]

        score = sum(1 for keyword in learning_keywords if keyword in content)

        # Include documents with learning-relevant content
        return score >= 2 or any(
            keyword in content
            for keyword in [
                "python",
                "sql",
                "data",
                "analytics",
                "machine learning",
                "ai",
            ]
        )

    def filter_learning_docs(doc):
        """Filter documents that could be useful for learning advice."""
        content = doc.page_content.lower()
        metadata = doc.metadata

        # Learning/advice keywords
        learning_keywords = [
            "learn",
            "study",
            "skill",
            "technology",
            "python",
            "sql",
            "data",
            "analytics",
            "machine learning",
            "ai",
            "cloud",
            "bigquery",
            "airflow",
            "looker",
            "dataform",
            "vertex",
            "gcp",
            "aws",
            "career",
            "development",
            "growth",
            "experience",
            "project",
            "challenge",
            "solution",
            "approach",
            "methodology",
        ]

        score = sum(1 for keyword in learning_keywords if keyword in content)

        # Include documents with learning-relevant content
        return score >= 2 or any(
            keyword in content
            for keyword in [
                "python",
                "sql",
                "data",
                "analytics",
                "machine learning",
                "ai",
            ]
        )

    # Create a custom retriever that filters results
    class LearningRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager=None
        ) -> List[Document]:
            # Get documents from base retriever
            base_retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20}
            )
            docs = base_retriever.invoke(query)

            # Filter the documents
            filtered_docs = [doc for doc in docs if filter_learning_docs(doc)]

            # If we have filtered results, return them
            if filtered_docs:
                return filtered_docs[:5]  # Return top 5 filtered results

            # Fallback to base retriever if no filtered results
            return docs[:5]

    return LearningRetriever()


def get_redirect_retriever():
    """Retriever for redirect agent - uses minimal context."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.DB_PATH, embedding_function=embeddings
    )

    # For redirect agent, we want minimal, general context
    def filter_general_docs(doc):
        """Filter for general, introductory content."""
        content = doc.page_content.lower()

        general_keywords = [
            "bolaji",
            "data",
            "science",
            "ai",
            "engineer",
            "professional",
            "background",
            "experience",
            "skill",
            "expertise",
        ]

        score = sum(1 for keyword in general_keywords if keyword in content)
        return score >= 1 and len(content.split()) < 100  # Short, general content

    def filter_general_docs(doc):
        """Filter for general, introductory content."""
        content = doc.page_content.lower()

        general_keywords = [
            "bolaji",
            "data",
            "science",
            "ai",
            "engineer",
            "professional",
            "background",
            "experience",
            "skill",
            "expertise",
        ]

        score = sum(1 for keyword in general_keywords if keyword in content)
        return score >= 1 and len(content.split()) < 100  # Short, general content

    # Create a custom retriever that filters results
    class RedirectRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager=None
        ) -> List[Document]:
            # Get documents from base retriever
            base_retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 6, "fetch_k": 15}
            )
            docs = base_retriever.invoke(query)

            # Filter the documents
            filtered_docs = [doc for doc in docs if filter_general_docs(doc)]

            # If we have filtered results, return them
            if filtered_docs:
                return filtered_docs[:3]  # Return top 3 filtered results

            # Fallback to base retriever if no filtered results
            return docs[:3]

    return RedirectRetriever()
