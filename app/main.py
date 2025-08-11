import sys
import os
from typing import List, Tuple
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.agent import get_agent, generate_response
from utils import database

app = FastAPI(
    title="iBola Agentic RAG Chatbot",
    description="A chatbot for answering questions about Bolaji's professional background.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    database.initialize_database()

# Mount the static directory to serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

agent = get_agent()

# In-memory store for chat histories. 
# For production, you would replace this with a persistent store like Redis or a database.
chat_histories = {}

class ChatInput(BaseModel):
    user_input: str
    session_id: str

class FeedbackInput(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int

@app.get("/", tags=["App"])
def read_root():
    return FileResponse('static/index.html')

@app.post("/chat", tags=["Chat"])
def chat(payload: ChatInput):
    """Chat with the agent."""
    session_id = payload.session_id
    user_input = payload.user_input

    # Get chat history or create a new one
    history = chat_histories.get(session_id, [])

    # Get the full response from the agent, which includes source documents
    result = generate_response(agent, user_input, chat_history=history)

    # --- DEBUG LOGGING: Print source documents to the console ---
    if "source_documents" in result:
        print("\n--- SOURCE DOCUMENTS ---")
        for doc in result["source_documents"]:
            print(f"Page {doc.metadata.get('page', '?')}:")
            print(doc.page_content)
        print("--- END SOURCE DOCUMENTS ---\\n")
    # ----------------------------------------------------------

    # Prepare the response for the frontend
    response_for_frontend = {
        "answer": result.get("answer"),
        "actions": result.get("actions")
    }

    # Update the history
    history.append((user_input, result.get("answer", "")))
    chat_histories[session_id] = history

    return response_for_frontend

@app.post("/feedback", tags=["Feedback"])
def feedback(payload: FeedbackInput):
    """Receive and store user feedback."""
    conn = database.create_connection()
    if conn:
        feedback_data = (
            payload.session_id,
            payload.question,
            payload.answer,
            payload.rating,
            datetime.now().isoformat()
        )
        database.add_feedback(conn, feedback_data)
        conn.close()
        return {"status": "success", "message": "Feedback received"}
    return {"status": "error", "message": "Failed to connect to database"}