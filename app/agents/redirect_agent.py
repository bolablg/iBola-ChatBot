"""
Redirect Agent - Handles off-topic questions and redirects users appropriately.
"""

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import config

from .retrievers import get_redirect_retriever

# Specialized prompt for redirect agent
REDIRECT_QA_TEMPLATE = """
You are iBola, an AI assistant for Bolaji. This question is not related to Bolaji's professional background, education, or learning advice about his skills.

STRICT RULES:
1) Keep reply succinct: ≤3 sentences; each sentence ≤15 words.
2) Match the user's language. Be polite and helpful.
3) Politely redirect to topics about Bolaji's professional journey or education.
4) Ask the user to ask a precise question about these specific topics.
5) Never mention "documents," "context," "RAG," or how you found the answer.
6) For second redirect: Explain it's not information you have and invite to contact options.
7) ALWAYS talk about Bolaji in third person as an assistant would.

QUESTION:
{question}

Context (for reference only):
{context}

POLITE REDIRECT:"""

REDIRECT_CONDENSE_PROMPT = """Given the following conversation and a follow up question that seems off-topic, rephrase it to acknowledge the question while redirecting to professional topics.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone redirect:"""


class RedirectAgent:
    """Agent specialized in redirecting off-topic questions."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.6,
            google_api_key=config.GEMINI_API_KEY,
        )
        self.retriever = get_redirect_retriever()
        self.qa_prompt = PromptTemplate.from_template(REDIRECT_QA_TEMPLATE)
        self.condense_prompt = PromptTemplate.from_template(REDIRECT_CONDENSE_PROMPT)

        self.agent = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=self.condense_prompt,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
        )

    def invoke(self, inputs):
        """Invoke the redirect agent."""
        return self.agent.invoke(inputs)

    def get_agent_type(self):
        """Return agent type identifier."""
        return "redirect"

    def generate_redirect_response(
        self, user_input, chat_history="", redirect_count=0, session_id=""
    ):
        """Generate a redirect response with enhanced progressive options."""
        base_response = self.invoke(
            {"question": user_input, "chat_history": chat_history}
        )

        # Progressive redirect actions based on redirect count
        redirect_actions = []

        # Determine response and actions based on redirect count
        if redirect_count == 0:
            # Initial redirect: Use the base response from the agent
            answer = base_response.get(
                "answer",
                "I specialize in questions about Bolaji's professional background.",
            )
            redirect_actions = []

        elif redirect_count == 1:
            # First redirection: Ask for precise question about professional journey or education
            # No button needed - just the text response is sufficient
            answer = "I can only answer questions about Bolaji's professional journey or education. Please ask a precise question about these specific topics."
            redirect_actions = []  # No actions for first redirect

        elif redirect_count == 2:
            # Second redirection: Explain it's not information you have, offer contact options, then end chat
            answer = "This is not information I have about Bolaji's professional journey or education. Please contact him directly for this information.\n\nChat ended. Thank you for your interest!"
            redirect_actions.extend(
                [
                    {
                        "text": "📧 Send email",
                        "type": "contact_email",
                        "url": "mailto:hello@bolablg.com",
                        "session_id": session_id,
                        "chat_history": chat_history,
                        "description": "Send an email to Bolaji",
                        "primary": True,
                        "end_chat": True,  # This will end the chat but keep buttons clickable
                    },
                    {
                        "text": "📅 Book appointment",
                        "type": "contact_booking",
                        "url": "https://calendar.app.google/Jg1r7af8Rk2jYqCV8",
                        "session_id": session_id,
                        "chat_history": chat_history,
                        "description": "Schedule a meeting with Bolaji",
                        "primary": True,
                        "end_chat": True,  # This will end the chat but keep buttons clickable
                    },
                ]
            )

        elif redirect_count >= 3:
            # Third or more redirections: Maintain contact options
            answer = "For questions outside Bolaji's professional journey or education, please contact him directly.\n\nChat ended. Thank you for your interest!"
            redirect_actions.extend(
                [
                    {
                        "text": "📧 Send email",
                        "type": "contact_email",
                        "url": "mailto:hello@bolablg.com",
                        "session_id": session_id,
                        "chat_history": chat_history,
                        "description": "Send an email to Bolaji",
                        "primary": True,
                        "end_chat": False,
                    },
                    {
                        "text": "📅 Book appointment",
                        "type": "contact_booking",
                        "url": "https://calendar.app.google/Jg1r7af8Rk2jYqCV8",
                        "session_id": session_id,
                        "chat_history": chat_history,
                        "description": "Schedule a meeting with Bolaji",
                        "primary": True,
                        "end_chat": False,
                    },
                ]
            )

        return {
            "answer": answer,
            "actions": redirect_actions,
            "agent_type": "redirect",
            "redirect_count": redirect_count,
            "session_id": session_id,
            "should_end_chat": redirect_count
            >= 2,  # End chat after second redirect attempt
        }
