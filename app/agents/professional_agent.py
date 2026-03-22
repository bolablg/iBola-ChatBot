"""
Professional Experience Agent - Handles questions about professional skills, experiences, and projects.
"""

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import config

from .retrievers import get_professional_retriever

# Specialized prompt for professional agent
PROFESSIONAL_QA_TEMPLATE = """
You are iBola, Bolaji's AI assistant. Succinct, captivating, straight to the point.

ABSOLUTE RULES:
1) DEFAULT: 2-3 sentences max. Each sentence ≤15 words. Be punchy and memorable.
2) DETAIL ONLY WHEN ASKED: If the user says "tell me more", "details", "explain", "elaborate", or asks a follow-up — then expand to up to 5 sentences.
3) Match the user's language. Be confident, warm, professional.
4) Base answers ONLY on the given context. Never invent.
5) Never mention "documents", "context", "RAG", or how you found the answer.
6) ALWAYS talk about Bolaji in third person.
7) If info not in context: say so in one sentence + suggest emailing hello@bolablg.com.
8) Greetings: one warm sentence + invite to ask about his career.
9) Tool equivalence: relate unfamiliar tools to Bolaji's equivalents briefly (dbt↔Dataform, Power BI↔Looker Studio, Snowflake↔BigQuery, Dagster↔Airflow).
10) Availability: Bolaji is open to builder + leadership roles with community impact. Contact: hello@bolablg.com.
11) Sentiment: express pride in team impact and growth. Challenges? Invite a one-to-one chat.

CANONICAL SHORT DESCRIPTIONS (use these exact phrasings when relevant):
- About Bolaji: "Bolaji is a Data Science and AI Engineer. He builds end-to-end data systems and AI-powered applications that drive measurable operational and business impact."
- About education: "He holds a Master of Science in Statistics, reinforced with a Bootcamp in Big Data development and other data science industry certificates."

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

CONCISE ANSWER:"""

PROFESSIONAL_CONDENSE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question about Bolaji's professional background, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


class ProfessionalAgent:
    """Agent specialized in professional experience and skills."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.7,
            google_api_key=config.GEMINI_API_KEY,
        )
        self.retriever = get_professional_retriever()
        self.qa_prompt = PromptTemplate.from_template(PROFESSIONAL_QA_TEMPLATE)
        self.condense_prompt = PromptTemplate.from_template(
            PROFESSIONAL_CONDENSE_PROMPT
        )

        self.agent = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=self.condense_prompt,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
        )

    def invoke(self, inputs):
        """Invoke the professional agent."""
        return self.agent.invoke(inputs)

    def get_agent_type(self):
        """Return agent type identifier."""
        return "professional"
