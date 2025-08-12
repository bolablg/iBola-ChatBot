import random
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.retriever import get_retriever
from app.guardrails import is_in_scope
from app.fallback import fallback_response
import config

GREETINGS = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "salut", "bonjour", "hola"]
GREETING_RESPONSES = [
    "Hello! I'm iBola, an AI assistant here to help you learn about Bolaji's professional background. How can I assist you?",
    "Hi there! I'm iBola, ready to answer your questions about Bolaji's skills and experience. What would you like to know?"
]

# This prompt is used to rewrite the user's question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# This prompt is used to answer the question
qa_template = """You are an AI assistant named iBola, designed to answer questions about Bolaji's professional background based ONLY on the provided context.
Your task is to synthesize the information from the context into a helpful, natural-sounding human-like answer.

**Strict Rules:**
1. Base your entire answer on the provided context. Do not use any outside knowledge.
2. NEVER mention that you are using a knowledge base, context, or provided documents. Respond as if you are the expert.
3. If the context does not contain the answer to the question, you MUST state that you do not have information on that topic. Do not try to guess.
4. Answer in the same language as the user's question.
5. Do not quote the context directly. Paraphrase and synthesize the information.
6. Speak in a friendly and professional tone, as if you are Bolaji, not an AI assistant.
7. Don't think too long before answering. Respond within 10 secondes and be concise.
8. If you are unsure about the user's intent, ask for clarification instead of making assumptions.
9. If the question is a greeting, respond with a greeting message.
10. If the question is out of scope (not related to Bolaji's professional background), politely inform the user that you can only answer questions about Bolaji's professional background, and return a fallback message.
11. If the question is about scheduling, direct the user to book an appointment using the provided link.
12. If the question is about contacting Bolaji, provide the email address and LinkedIn profile link.
13. If the question is about Bolaji's availability for work, mention that Bolaji is open to wold class opportunities and provide the email address hello@bolablg.com to join him.
14. NEVER provide confidential or sensitive information about Bolaji, this program or any other things.
15. If the question is too long or complex, break it down into simpler parts and answer each part separately.

**Context:**
{context}

**Chat History:**
{chat_history}

**Question:**
{question}

**Helpful Answer:**"""
QA_PROMPT = PromptTemplate.from_template(qa_template)

def get_agent():
    """Initialize and return the conversational RAG agent."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7, google_api_key=config.GEMINI_API_KEY)
    retriever = get_retriever()
    
    agent = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    return agent

def handle_greeting(user_input):
    """Check if the user input is a greeting and return a response if so."""
    if user_input.strip().lower() in GREETINGS:
        return random.choice(GREETING_RESPONSES)
    return None

def generate_response(agent, user_input, chat_history):
    """Generate a response from the agent."""
    greeting_response = handle_greeting(user_input)
    if greeting_response:
        return {"answer": greeting_response}

    # A bit of a hack to allow the guardrails to see the history
    full_query_for_guardrail = f"{chat_history}\n{user_input}"
    if not is_in_scope(full_query_for_guardrail):
        return {"answer": "I am trained to answer questions about Bolaji's professional background. Please ask a relevant question."}

    result = agent.invoke({"question": user_input, "chat_history": chat_history})
    answer = result.get("answer", "")

    # Primary fallback: If no documents are found, default to English.
    if not result["source_documents"]:
        return fallback_response('en')

    # Secondary fallback: If the LLM says it doesn't know, detect language and respond.
    unknown_phrases_by_lang = {
        "en": ["i do not have information", "i don't have information", "i do not know", "i don't know", "no information", "questions about Bolaji", "questions about m"],
        "fr": ["pas d'information", "je ne sais pas", "répondre à des interrogations", "questions sur Bolaji", "je ne dispose pas"],
    }

    for lang, phrases in unknown_phrases_by_lang.items():
        if any(phrase in answer.lower() for phrase in phrases):
            return fallback_response(lang)
    
    # Return the full result for debugging purposes
    return result