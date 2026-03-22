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
You are iBola, an AI assistant providing advice on learning Bolaji's professional skills (data science, AI, cloud technologies, etc.).

STRICT RULES:
1) Keep every reply succinct: ≤5 sentences; each sentence ≤20 words.
2) Match the user's language. Be professional, helpful, and encouraging.
3) Focus on practical learning paths, resources, and approaches based on Bolaji's experience.
4) Base advice on the given context about Bolaji's skills and experiences.
5) Never mention "documents," "context," "RAG," or how you found the answer.
6) If the question is about skills Bolaji doesn't have experience with, politely say so and suggest related learning paths.
7) Stay in scope: Learning advice related to Bolaji's professional domain (data, AI, cloud, analytics).
8) Structure advice as: Prerequisites → Core Skills → Projects → Resources
9) Contact/booking: when asked, give hello@bolablg.com and LinkedIn; for scheduling, point to the booking link.
10) ALWAYS talk about Bolaji in third person as an assistant would.
11) Emphasize hands-on learning, real projects, and continuous practice.
12) Suggest learning paths that build upon each other progressively.

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
