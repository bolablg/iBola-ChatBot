"""
Logging service with Google Cloud Logging integration.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from google.auth import exceptions as auth_exceptions
    from google.cloud import logging as cloud_logging

    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print(
        "Google Cloud Logging not available. Install with: pip install google-cloud-logging"
    )

from config import GCP_PROJECT_ID, GCP_SA_CREDENTIALS_PATH, LOG_LEVEL


class GoogleCloudHandler(logging.Handler):
    """Custom logging handler for Google Cloud Logging."""

    def __init__(self, project_id: str, credentials_path: str):
        super().__init__()
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.client = None
        self.logger = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Google Cloud Logging client."""
        if not GOOGLE_CLOUD_AVAILABLE:
            print("Google Cloud Logging client not available")
            return

        try:
            # Set credentials path
            if os.path.exists(self.credentials_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

            # Initialize client
            self.client = cloud_logging.Client(project=self.project_id)
            self.logger = self.client.logger("iBola-chatbot")

            print("✅ Google Cloud Logging initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Google Cloud Logging: {e}")
            self.client = None
            self.logger = None

    def emit(self, record):
        """Emit log record to Google Cloud Logging."""
        if not self.logger:
            return

        try:
            # Only log WARNING, ERROR, and DEBUG to GCP (filter out INFO and below)
            if record.levelno not in [logging.WARNING, logging.ERROR, logging.DEBUG]:
                return

            # Format log entry
            log_entry = self.format(record)

            # Create structured log entry
            json_payload = {
                "message": record.getMessage(),
                "level": record.levelname,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "session_id": getattr(record, "session_id", None),
                "user_language": getattr(record, "user_language", None),
                "agent_type": getattr(record, "agent_type", None),
            }

            # Add exception info if available
            if record.exc_info:
                json_payload["exception"] = self.formatException(record.exc_info)

            # Add additional context for important logs
            if hasattr(record, "user_input"):
                json_payload["user_input"] = record.user_input
            if hasattr(record, "response_time"):
                json_payload["response_time"] = record.response_time

            # Log to Google Cloud with appropriate severity
            if record.levelno >= logging.ERROR:
                self.logger.log_struct(json_payload, severity="ERROR")
            elif record.levelno >= logging.WARNING:
                self.logger.log_struct(json_payload, severity="WARNING")
            else:  # DEBUG
                self.logger.log_struct(json_payload, severity="DEBUG")

        except Exception as e:
            # Fallback to stderr if cloud logging fails
            print(f"Failed to log to Google Cloud: {e}", file=sys.stderr)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def format(self, record):
        """Format log record with additional context."""
        # Add default format
        record.timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Create structured message
        log_data = {
            "timestamp": record.timestamp,
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add extra fields if available
        extra_fields = [
            "session_id",
            "user_language",
            "agent_type",
            "user_input",
            "response_time",
        ]
        for field in extra_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class LoggingService:
    """Centralized logging service with multiple handlers."""

    def __init__(self):
        self.logger = logging.getLogger("iBola")
        self.logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_formatter = StructuredFormatter()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Add file handler (only if directory exists or can be created)
        try:
            log_dir = "local/logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, "chatbot.log"),
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
            print(f"✅ File logging enabled: {os.path.join(log_dir, 'chatbot.log')}")
        except (OSError, PermissionError) as e:
            print(
                f"Warning: Could not set up file logging: {e}. Using console logging only."
            )

        # Add Google Cloud Logging handler if available
        if GOOGLE_CLOUD_AVAILABLE and os.path.exists(GCP_SA_CREDENTIALS_PATH):
            try:
                cloud_handler = GoogleCloudHandler(
                    GCP_PROJECT_ID, GCP_SA_CREDENTIALS_PATH
                )
                if cloud_handler.logger:
                    self.logger.addHandler(cloud_handler)
                    self.logger.info("Google Cloud Logging handler added successfully")
                else:
                    self.logger.warning(
                        "Google Cloud Logging handler failed to initialize"
                    )
            except Exception as e:
                self.logger.error(f"Failed to add Google Cloud Logging handler: {e}")
        else:
            self.logger.info(
                "Google Cloud Logging not configured or credentials not found"
            )

        # Create local logs directory if it doesn't exist
        os.makedirs("local/logs", exist_ok=True)

        self.logger.info("Logging service initialized")

    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance with the specified name."""
        if name:
            return logging.getLogger(f"iBola.{name}")
        return self.logger

    def log_chat_interaction(
        self,
        session_id: str,
        user_input: str,
        agent_type: str,
        response: str,
        response_time: float,
        user_language: str = "en",
    ):
        """Log chat interaction with structured data."""
        self.logger.info(
            f"Chat interaction: {len(user_input)} chars -> {len(response)} chars",
            extra={
                "session_id": session_id,
                "user_input": user_input[:100],  # Truncate for logging
                "agent_type": agent_type,
                "response_time": response_time,
                "user_language": user_language,
            },
        )

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context."""
        error_msg = f"{type(error).__name__}: {str(error)}"
        if context:
            error_msg += f" | Context: {json.dumps(context, ensure_ascii=False)}"

        self.logger.error(error_msg, exc_info=True)

    def log_performance_metric(
        self, metric_name: str, value: float, metadata: Dict[str, Any] = None
    ):
        """Log performance metric."""
        log_data = {"metric": metric_name, "value": value}
        if metadata:
            log_data.update(metadata)

        self.logger.info(f"Performance: {metric_name} = {value}", extra=log_data)


# Global logging service instance
logging_service = LoggingService()


def get_logger(name: str = None) -> logging.Logger:
    """Convenience function to get a logger instance."""
    return logging_service.get_logger(name)
