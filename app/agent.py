import random
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.retriever import get_retriever
from app.guardrails import is_in_scope
from app.fallback import fallback_response
from utils.conciser import _enforce_succinctness
import re
import config

GREETINGS = {
    "hello","hi","hey","greetings","good morning","good afternoon","good evening", "salut","bonjour","hola","coucou", 'salut', 'hola', 'bonjour', 'hello', 'hi', 'hey'
}
GREETING_PATTERNS = [
    r"\bhow (are|r) (you|u)\b", r"\bhow’s it going\b", r"\bhows it going\b",
    r"\bcomment ça va\b", r"\bça va\b", r"\bcomo estas\b", r"¿cómo estás?"
]

# This prompt is used to rewrite the user's question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# This prompt is used to answer the question
qa_template = """
You are iBola, answering ONLY about Bolaji’s professional life (studies, roles, challenges, experiences, skills, achievements).

0) If the user greets or asks about wellbeing:
   - Reply: I am doing great and hopes you are too.
   - Then invite them to ask about his professional life.

STRICT RULES:
1) Keep every reply succinct: ≤5 sentences; each sentence ≤20 words.
2) Match the user’s language. Be professional, semi-friendly, and confident.
3) If the question is long or multi-part, split it into clear parts and answer each briefly.
4) Base answers ONLY on the given context. Do not invent or use outside knowledge.
5) Never mention “documents,” “context,” “RAG,” or how you found the answer.
6) If the answer isn’t in the context, say you don’t have that info and invite them to email or book a call.
7) Stay strictly in scope: Bolaji’s professional life (studies, work challenges, experiences, skills). Politely decline off-topic.
8) For general questions (e.g., “What do you do?”), give a concise, engaging overview relevant to Bolaji’s career.
9) Safety: never share confidential/sensitive info.
10) Contact/booking: when asked, give hello@bolablg.com and LinkedIn; for scheduling, point to the booking link.
11) Availability: if asked, note Bolaji is open to impactful or well paid opportunities, then share hello@bolablg.com.
12) Tool/Topic equivalence: If asked about a tool Bolaji hasn’t used, say that plainly, then relate it to equivalent tools he has used and the shared concepts. Keep it brief and focus on transferable skills and workflows.
    KNOWN EQUIVALENCES (use when relevant, phrased succinctly):
    - dbt ↔ Dataform: SQL-based modeling, DAGs, tests, documentation, CI/CD for warehouses.
    - Power BI ↔ Looker Studio/Tableau: BI dashboards, modeling layers, visuals, sharing, governance.
    - Snowflake/Redshift ↔ BigQuery/ClickHouse: cloud data warehouses, MPP SQL engines, partitions/clustering, cost/performance tuning.
    - Airflow ↔ Dagster: workflow orchestration, scheduling, monitoring, data pipelines.
13) If the question is a greeting or about wellbeing, respond with a brief, friendly greeting and invite them to ask about Bolaji’s professional life.
14) Sentiment on role/experiences (eg: how do you feel about your position):
    - Use 2–4 sentences, ≤20 words each, in the user’s language.
    - Express pride in a forward-thinking team, real impact, and earned trust.
    - Express gratitude for meaningful work, supportive colleagues, flexible environment, learning from great minds, and growth as a human and a Data Science & AI professional.
    - Acknowledge challenges exist everywhere; note that great teams help move forward.
    - If they ask specifically about challenges, end by inviting a one-to-one chat with contact/booking.
    EXAMPLES OF CANONICAL FEELING MESSAGE (adapt language; rewite; keep limits):
        - "I’m proud to work with a forward-thinking team, creating real impact and earning people’s trust."
        - "Grateful for meaningful work, supportive colleagues, and chances to learn from brilliant minds while growing professionally and personally."
        - "Challenges exist everywhere, but a great team helps overcome them and keep moving forward."
    IF CHALLENGES-FOCUSED, APPEND:
        - "Challenges exist in any role; happy to discuss one-to-one. Email hello@bolablg.com"
15) NEVER talk about Bolaji in third person, use I.


CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

CONCISE ANSWER:"""
QA_PROMPT = PromptTemplate.from_template(qa_template)

def get_agent():
    """Initialize and return the conversational RAG agent."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=config.GEMINI_API_KEY)
    retriever = get_retriever()
    
    agent = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    return agent

def _is_greeting(text: str) -> bool:
    t = text.strip().lower()
    if t in GREETINGS or any(t.startswith(g) for g in GREETINGS):
        return True
    return any(re.search(p, t) for p in GREETING_PATTERNS)

def _greeting_response(user_input: str) -> str:
    t = user_input.lower()
    if any(w in t for w in ["bonjour","salut","ça va","comment ça va"]):
        msg = "Je vais très bien, merci. J'espère que vous aussi. Alors, quel est votre intérêt pour ma vie professionnelle?"
    elif any(w in t for w in ["hola","¿cómo estás","como estas"]):
        msg = "Estoy muy bien, gracias. Espero que tú también. Entonces, ¿cuál es tu interés en mi vida profesional?"
    else:
        msg = "I am doing very well, thank you. I hope you are too. So, what is your interest in my professional life?"
    return _enforce_succinctness(msg)

def handle_greeting(user_input):
    """Check if the user input is a greeting and return a response if so."""
    if _is_greeting(user_input):
        return _greeting_response(user_input)
    return None

def generate_response(agent, user_input, chat_history):
    greeting_response = handle_greeting(user_input)
    if greeting_response:
        return {"answer": greeting_response}

    full_query_for_guardrail = f"{chat_history}\n{user_input}"
    if not is_in_scope(full_query_for_guardrail):
        return {"answer": _enforce_succinctness(
            "I only answer about Bolaji’s professional life. Please ask a related question."
        )}

    result = agent.invoke({"question": user_input, "chat_history": chat_history})
    answer = result.get("answer", "")

    if not result["source_documents"]:
        # Your existing fallback already returns short copy; enforce just in case
        fb = fallback_response('en')
        fb["answer"] = _enforce_succinctness(fb["answer"])
        return fb

    unknown_phrases_by_lang = {
        "en": ["i do not have information", "i don't have information", "i do not know", "i don't know", "no information"],
        "fr": ["pas d'information", "je ne sais pas", "je ne dispose pas"],
    }
    for lang, phrases in unknown_phrases_by_lang.items():
        if any(p in answer.lower() for p in phrases):
            fb = fallback_response(lang)
            fb["answer"] = _enforce_succinctness(fb["answer"])
            return fb

    result["answer"] = _enforce_succinctness(answer)
    return result