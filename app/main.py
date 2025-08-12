from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.agent import get_agent, generate_response
from app.history_store import get_history, append_history

app = FastAPI(
    title="iBola Agentic RAG Chatbot",
    description="A chatbot for answering questions about Bolaji's professional background.",
    version="1.0.0"
)



# Mount the static directory to serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

agent = get_agent()



class ChatInput(BaseModel):
    user_input: str
    session_id: str



@app.get("/", tags=["App"])
def read_root():
    return FileResponse('static/index.html')

@app.post("/chat", tags=["Chat"])
def chat(payload: ChatInput):
    """Chat with the agent."""
    session_id = payload.session_id
    user_input = payload.user_input

    try:
        # Get chat history from the configured store
        history = get_history(session_id)

        # Get the full response from the agent, which includes source documents
        result = generate_response(agent, user_input, chat_history=history)

        # --- DEBUG LOGGING: Print source documents to the console ---
        if "source_documents" in result:
            print("\n--- SOURCE DOCUMENTS ---")
            for doc in result["source_documents"]:
                print(f"Page {doc.metadata.get('page', '?')}:")
                print(doc.page_content)
            print("--- END SOURCE DOCUMENTS ---\n")
        # ----------------------------------------------------------

        # Prepare the response for the frontend
        response_for_frontend = {
            "answer": result.get("answer"),
            "actions": result.get("actions")
        }

        # Update the history in the store
        append_history(session_id, (user_input, result.get("answer", "")))

        return response_for_frontend
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    

