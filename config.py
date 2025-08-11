import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_TYPE = os.getenv("DB_TYPE", "mariadb")  # or "sqlite"
DB_NAME = os.getenv("DB_NAME", "feedback.db")
FEEDBACK_DB_CONN_URL = os.getenv("FEEDBACK_DB_CONN_URL")

# Vector store configuration
DB_PATH = os.getenv("DB_PATH", "chroma_db")

# Data configuration
DATA_PATH = os.getenv("DATA_PATH", "data")

# GCP Service Account Credentials
GCP_SA_CRENDIALS_PATH = os.getenv("GCP_SA_CRENDIALS_PATH")

# Google OAuth Credentials
GOOGLE_OAUTH_CREDENTIALS_PATH = os.getenv("GOOGLE_OAUTH_CREDENTIALS_PATH")

# Google API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# GCP Project ID
GCP_PROJECT = os.getenv("GCP_PROJECT")

# Google Drive Folder ID
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

# Google Chat Webhook URL
GCHAT_WEBHOOK_URL = os.getenv("GCHAT_WEBHOOK_URL")