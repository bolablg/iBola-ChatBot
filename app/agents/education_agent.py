"""
Education Agent - Handles questions about educational background, degrees, and academic achievements.
"""

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import config

from .retrievers import get_education_retriever

# Specialized prompt for education agent
EDUCATION_QA_TEMPLATE = """
You are iBola, Bolaji's AI assistant. Succinct, captivating, straight to the point.

ABSOLUTE RULES:
1) DEFAULT: 2-3 sentences max. Each sentence ≤15 words.
2) DETAIL ONLY WHEN ASKED: elaborate only if user requests more info.
3) LANGUAGE: Reply in the SAME language the user writes in. French = French. English = English. Never mix.
4) Base answers ONLY on the given context. Never invent.
5) Never mention "documents", "context", "RAG".
6) ALWAYS talk about Bolaji in third person.
7) Focus: degrees, institutions, GPA, fields of study.
8) Off-topic: decline in one sentence, redirect to education/career topics.
9) Degrees, dates, institutions, certifications: answer ONLY from the context. Never state a profile fact that is not in the context.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

CONCISE ANSWER:"""

EDUCATION_CONDENSE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question about Bolaji's educational background, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


class EducationAgent:
    """Agent specialized in educational background."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=config.GEMINI_API_KEY,
        )
        self.retriever = get_education_retriever()
        self.qa_prompt = PromptTemplate.from_template(EDUCATION_QA_TEMPLATE)
        self.condense_prompt = PromptTemplate.from_template(EDUCATION_CONDENSE_PROMPT)

        self.agent = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=self.condense_prompt,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
        )

    def invoke(self, inputs):
        """Invoke the education agent."""
        return self.agent.invoke(inputs)

    def get_agent_type(self):
        """Return agent type identifier."""
        return "education"
