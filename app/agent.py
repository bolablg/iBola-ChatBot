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
Your task is to synthesize the information from the context into a helpful, natural-sounding answer.

**Strict Rules:**
1. Base your entire answer on the provided context. Do not use any outside knowledge.
2. NEVER mention that you are using a knowledge base, context, or provided documents. Respond as if you are the expert.
3. If the context does not contain the answer to the question, you MUST state that you do not have information on that topic. Do not try to guess.
4. Answer in the same language as the user's question.
5. Do not quote the context directly. Paraphrase and synthesize the information.

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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7, google_api_key=config.GEMINI_API_KEY)
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
        "en": ["i do not have information", "i don't have information", "i do not know", "i don't know", "no information on that topic"],
        "fr": ["n'ai pas d'information", "je ne sais pas"],
        "es": ["no tengo información", "no lo sé"]
    }

    for lang, phrases in unknown_phrases_by_lang.items():
        if any(phrase in answer.lower() for phrase in phrases):
            return fallback_response(lang)
    
    # Return the full result for debugging purposes
    return result