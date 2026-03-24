"""
Configuration file for the multi-agent chatbot system.
"""

import os
from pathlib import Path
from typing import Optional


# Load environment variables from .env files
def load_env_files():
    """Load environment variables from .env files in order of priority."""
    # Get the project root directory
    project_root = Path(__file__).parent

    # List of .env files to load in order of priority (later files override earlier ones)
    env_files = [
        project_root / ".env.local",  # Local overrides (highest priority)
        project_root / ".env",  # Default environment file
    ]

    # Load .env files if they exist
    try:
        from dotenv import load_dotenv

        for env_file in env_files:
            if env_file.exists():
                print(f"Loading environment variables from {env_file}")
                load_dotenv(env_file)
    except ImportError:
        print("Warning: python-dotenv not installed. Using environment variables only.")
    except Exception as e:
        print(f"Warning: Error loading .env files: {e}")


# Load environment variables from .env files first
load_env_files()

# Google AI Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Google Cloud Configuration
GCP_SA_CREDENTIALS_PATH = os.getenv(
    "GCP_SA_CREDENTIALS_PATH",
    os.path.join(os.path.dirname(__file__), "_conf", "ibola_agent_sa.json"),
)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")

# Vector Database Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Data Directory Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Google Drive Sync
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

# Google Chat Integration (for contact alerts)
# To get a webhook URL:
# 1. Go to Google Chat
# 2. Create a new space or use existing one
# 3. Go to space settings -> Manage webhooks
# 4. Create a new webhook and copy the URL
GCHAT_WEBHOOK_URL = os.getenv("GCHAT_WEBHOOK_URL")

# Google Sheets Integration (for redirect logging)
# To get a spreadsheet ID:
# 1. Create a new Google Sheet
# 2. The ID is in the URL: https://docs.google.com/spreadsheets/d/[SPREADSHEET_ID]/edit
REDIRECT_LOG_SHEET_ID = os.getenv("REDIRECT_LOG_SHEET_ID")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# CORS Configuration
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", r"https://(.+\.)?bolablg\.com")

# Session Configuration
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
MAX_REDIRECT_COUNT = int(os.getenv("MAX_REDIRECT_COUNT", "3"))

# Dynamic Guardrails Configuration
GUARDRAILS_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DYNAMIC_PATTERNS_FILE = os.path.join(GUARDRAILS_DATA_DIR, "dynamic_patterns.json")
FEEDBACK_HISTORY_FILE = os.path.join(GUARDRAILS_DATA_DIR, "conversation_feedback.json")

# Language Support
SUPPORTED_LANGUAGES = [
    "en",  # English
    "fr",  # French
]

# Contact Information
CONTACT_EMAIL = "hello@bolablg.com"
CALENDAR_BOOKING_URL = "https://calendar.google.com/calendar/appointments/schedules/AcZssZ3YeidR5Og4YSGZIlxUIlDAf0AiRA6N8-MAzr-Sy55BtbKhBLXkfa8M_P_92eokXRnayLVlEXiW?gv=true"
LINKEDIN_URL = "https://linkedin.com/in/bolablg"


# Validation
def validate_config():
    """Validate that all required configuration is present."""
    required_vars = ["GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not globals().get(var)]

    if missing_vars:
        project_root = Path(__file__).parent
        env_file = project_root / ".env"
        env_example = project_root / "sample.env"

        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        error_msg += "\n\nTo fix this:"
        error_msg += "\n1. Copy the sample environment file:"
        error_msg += f"\n   cp {env_example} {env_file}"
        error_msg += "\n2. Edit .env and add your actual values"
        error_msg += "\n3. Or set environment variables directly:"
        error_msg += "\n   export GEMINI_API_KEY='your_actual_key_here'"

        raise ValueError(error_msg)

    # Optional but recommended
    if not GCHAT_WEBHOOK_URL:
        print(
            "Warning: GCHAT_WEBHOOK_URL not set. Contact alerts will not be sent to Google Chat."
        )


# Call validation on import
validate_config()
