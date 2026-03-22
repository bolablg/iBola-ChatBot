"""
Learning Advice Agent - Provides guidance on learning professional skills and career development.
"""

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import config

from .retrievers import get_learning_retriever

# Specialized prompt for learning advice agent
LEARNING_QA_TEMPLATE = """
You are iBola, Bolaji's AI assistant. Succinct, captivating, encouraging.

ABSOLUTE RULES:
1) DEFAULT: 2-3 sentences max. Give one actionable tip or starting point.
2) DETAIL ONLY WHEN ASKED: expand into a full learning path only if user asks for more.
3) Match the user's language. Helpful and direct.
4) Never mention "documents", "context", "RAG".
5) ALWAYS talk about Bolaji in third person.
6) When elaborating: Prerequisites → Core Skills → Projects.
7) Emphasize hands-on learning and real projects.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

HELPFUL LEARNING ADVICE:"""

LEARNING_CONDENSE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question about learning professional skills related to Bolaji's expertise, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


class LearningAgent:
    """Agent specialized in providing learning advice."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.8,
            google_api_key=config.GEMINI_API_KEY,
        )  # Slightly higher temperature for creative advice
        self.retriever = get_learning_retriever()
        self.qa_prompt = PromptTemplate.from_template(LEARNING_QA_TEMPLATE)
        self.condense_prompt = PromptTemplate.from_template(LEARNING_CONDENSE_PROMPT)

        self.agent = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=self.condense_prompt,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
        )

    def invoke(self, inputs):
        """Invoke the learning advice agent."""
        return self.agent.invoke(inputs)

    def get_agent_type(self):
        """Return agent type identifier."""
        return "learning"
