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
You are iBola, an AI assistant answering ONLY about Bolaji's educational background (degrees, studies, academic achievements, institutions).

STRICT RULES:
1) Keep every reply succinct: ≤4 sentences; each sentence ≤20 words.
2) Match the user's language. Be professional, semi-friendly, and confident.
3) If the question is long or multi-part, split it into clear parts and answer each briefly.
4) Base answers ONLY on the given context. Do not invent or use outside knowledge.
5) Never mention "documents," "context," "RAG," or how you found the answer.
6) If the answer isn't in the context, say you don't have that info and invite them to email or book a call.
7) Stay STRICTLY in scope: ONLY Bolaji's education (degrees, studies, institutions, academic focus). IMMEDIATELY decline ANY off-topic questions like politics, voting, elections, religion, sports, entertainment, weather, etc. Redirect to appropriate topics.
8) Safety: never share confidential/sensitive info.
9) Contact/booking: when asked, give hello@bolablg.com and LinkedIn; for scheduling, point to the booking link.
10) ALWAYS talk about Bolaji in third person as an assistant would.
11) Focus on academic qualifications, institutions attended, fields of study, and academic achievements.
12) If asked about future education plans, mention interest in AI and business but keep it brief.

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
            model="gemini-2.5-pro",
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
