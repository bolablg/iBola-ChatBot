import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from app.agent import get_agent, generate_response
from app.history_store import get_history, append_history

log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=numeric_level)
logger = logging.getLogger(__name__)
logger.setLevel(numeric_level)

app = FastAPI(
    title="iBola Agentic RAG Chatbot",
    description="A chatbot for answering questions about Bolaji's professional background.",
    version="1.0.0"
)

# Configure CORS
# It's recommended to use an environment variable for the regex to allow for more flexibility
# across different environments (e.g., development, staging, production).
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", r"https://(.+\.)?bolablg\.com")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    # allow_credentials=True is required to allow cookies to be sent from the
    # embedded iframe. This is necessary for session management.
    # However, it's important to be aware of the security implications of this,
    # as it can make the application more vulnerable to CSRF attacks.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Mount the static directory to serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

agent = get_agent()



class ChatInput(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(..., min_length=1)



@app.get("/", tags=["App"])
def read_root():
    # Set Content-Security-Policy to allow embedding in iframes on specified domains.
    # This is a more modern and flexible alternative to X-Frame-Options.
    # Note on secure cookies for iframes:
    # If you were to use cookies for authentication in the iframe, you would need to set
    # SameSite=None; Secure. This means the cookie will be sent with cross-site requests,
    # but only over HTTPS. FastAPI/Starlette session cookies can be configured accordingly.
    #
    # Note on postMessage:
    # For more complex interactions between the parent page and the iframe,
    # you can use the `postMessage` API to send messages securely between them.
    headers = {
        "Content-Security-Policy": "frame-ancestors 'self' https://bolablg.com https://*.bolablg.com"
    }
    return FileResponse('static/index.html', headers=headers)

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
            logger.debug("\n--- SOURCE DOCUMENTS ---")
            for doc in result["source_documents"]:
                logger.debug("Page %s:", doc.metadata.get('page', '?'))
                logger.debug(doc.page_content)
            logger.debug("--- END SOURCE DOCUMENTS ---\\n")
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
        logger.error("An error occurred: %s", e)
        raise e

    

