from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=config.GEMINI_API_KEY,
    )
