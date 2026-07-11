import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from app.agents.collector_agent import collector_agent
from app.history_store import append_history, get_history
from app.routes.feedback import router as feedback_router
from app.routes.streaming import router as streaming_router
from app.services.cache_service import cache_service
from app.services.google_chat_alert import google_chat_alert
from app.services.language_detection import language_service
from app.services.logging_service import get_logger, logging_service
from app.services.rate_limiting import rate_limiter

# Initialize logging service
app_logger = get_logger("main")

# Replace default logging with our service
logger = app_logger


def _read_version() -> str:
    """Read the release version from the project-root VERSION file.

    Single source of truth: the same file drives git tags on deploy, so the
    API version and /health never drift from the released tag. Falls back to
    the LLM_APP_VERSION env or "unknown" if the file is absent (e.g. a slim
    container that omitted it)."""
    version_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VERSION")
    try:
        with open(version_path, encoding="utf-8") as fh:
            version = fh.read().strip()
            if version:
                return version
    except OSError:
        pass
    return os.getenv("APP_VERSION", "unknown")


APP_VERSION = _read_version()

app = FastAPI(
    title="iBola Multi-Agent Chatbot API",
    description="""
    An intelligent multi-agent chatbot system for professional conversations.

    ## Features
    - 🤖 **Multi-Agent Architecture**: Specialized agents for different topics
    - 🌐 **Language Detection**: Automatic localization in 10+ languages
    - 🧠 **Dynamic Guardrails**: Learning system for improved routing
    - 📊 **Session Management**: Advanced conversation tracking
    - ☁️ **Google Cloud Integration**: Logging and monitoring
    - 🔒 **Security**: Input validation and error handling

    ## Agents
    - **Professional Agent**: Career experience and projects
    - **Education Agent**: Academic background and qualifications
    - **Learning Agent**: Advice on skill development
    - **Redirect Agent**: Polite handling of off-topic questions
    """,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
# It's recommended to use an environment variable for the regex to allow for more flexibility
# across different environments (e.g., development, staging, production).
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    r"https://(.+\.)?bolablg\.com|https://ibola-chatbot-.*\.run\.app",
)

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


# Mount new API routes (streaming, feedback)
app.include_router(streaming_router)
app.include_router(feedback_router)

# Mount the static directory to serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    """Serve robots.txt file for SEO."""
    return FileResponse("robots.txt", media_type="text/plain")


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap_xml():
    """Serve sitemap.xml file for SEO."""
    return FileResponse("sitemap.xml", media_type="application/xml")


@app.get("/manifest.json", include_in_schema=False)
async def web_manifest():
    """PWA manifest (PART 8.3). Served at root so scope covers the whole app."""
    return FileResponse("static/manifest.json", media_type="application/manifest+json")


@app.get("/sw.js", include_in_schema=False)
async def service_worker():
    """PWA service worker (PART 8.3), served from root for origin-wide scope.

    Served with no-store so a new Cloud Run revision's worker is detected on
    the next navigation, and Service-Worker-Allowed=/ so a file physically
    under /static could also claim root scope.
    """
    return FileResponse(
        "static/sw.js",
        media_type="text/javascript",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Service-Worker-Allowed": "/",
        },
    )


# The legacy /chat endpoint now runs the same LangGraph agentic pipeline as
# /ask-agentic. The legacy orchestrator (app/agents/orchestrator.py) baked
# profile facts into prompts and bypassed the guardrail flow; no endpoint may
# answer profile questions outside the graph.
def get_agentic_service():
    from app.routes.streaming import _get_service

    return _get_service()


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all endpoints."""
    import traceback
    from datetime import datetime

    error_details = {
        "error": str(exc),
        "error_code": type(exc).__name__,
        "timestamp": datetime.now().isoformat(),
        "path": str(request.url),
        "method": request.method,
    }

    # Log the error
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={
            "error_details": error_details,
            "user_agent": request.headers.get("user-agent"),
            "client_ip": request.client.host if request.client else "unknown",
        },
    )

    return JSONResponse(status_code=500, content=error_details)


# Request logging and rate limiting middleware
@app.middleware("http")
async def log_and_rate_limit_requests(request: Request, call_next):
    """Middleware to log requests and enforce rate limiting."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    # Check rate limits first
    allowed, rate_limit_info = await rate_limiter.check_rate_limit(
        client_ip, request.url.path
    )

    if not allowed:
        if rate_limit_info.get("blocked"):
            logger.warning(
                f"Blocked request from {client_ip}",
                extra={
                    "client_ip": client_ip,
                    "endpoint": request.url.path,
                    "reason": rate_limit_info.get("reason", "unknown"),
                },
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "retry_after": rate_limit_info.get("retry_after", 60),
                    "message": "Please wait before making more requests",
                },
                headers={"Retry-After": str(rate_limit_info.get("retry_after", 60))},
            )

        logger.warning(
            f"Rate limited request from {client_ip}",
            extra={
                "client_ip": client_ip,
                "endpoint": request.url.path,
                "reason": rate_limit_info.get("reason", "unknown"),
                "retry_after": rate_limit_info.get("retry_after", 60),
            },
        )

        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": rate_limit_info.get("retry_after", 60),
                "limit": rate_limit_info.get("limit", "unknown"),
            },
            headers={"Retry-After": str(rate_limit_info.get("retry_after", 60))},
        )

    # Log incoming request
    logger.info(
        f"Incoming {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "user_agent": request.headers.get("user-agent"),
            "client_ip": client_ip,
            "rate_limit_info": rate_limit_info,
        },
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log successful response
        logger.info(
            f"Completed {request.method} {request.url.path} -> {response.status_code}",
            extra={
                "status_code": response.status_code,
                "process_time": round(process_time, 3),
                "response_size": response.headers.get("content-length", 0),
                "client_ip": client_ip,
            },
        )

        # Log performance metric
        logging_service.log_performance_metric(
            "request_duration",
            process_time,
            {
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "client_ip": client_ip,
            },
        )

        return response

    except Exception as exc:
        process_time = time.time() - start_time

        # Log failed request
        logger.error(
            f"Failed {request.method} {request.url.path} -> {type(exc).__name__}",
            extra={
                "error": str(exc),
                "process_time": round(process_time, 3),
                "client_ip": client_ip,
            },
        )

        raise exc


class ChatInput(BaseModel):
    user_input: str = Field(
        ..., min_length=1, max_length=1000, description="User's message"
    )
    session_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique session identifier"
    )
    user_language: str = Field(
        default="en", max_length=5, description="User's preferred language code"
    )

    @validator("user_input")
    def validate_user_input(cls, v):
        """Validate user input for security and quality."""
        if not v or not v.strip():
            raise ValueError("Input cannot be empty")

        # Check for potentially harmful content
        harmful_patterns = [
            "<script",
            "javascript:",
            "onload=",
            "onerror=",
            "SELECT ",
            "INSERT ",
            "UPDATE ",
            "DELETE ",
            "DROP ",
        ]

        v_lower = v.lower()
        for pattern in harmful_patterns:
            if pattern.lower() in v_lower:
                raise ValueError("Invalid input detected")

        # Check length after cleaning
        cleaned_input = v.strip()
        if len(cleaned_input) < 1:
            raise ValueError("Input is too short")
        if len(cleaned_input) > 1000:
            raise ValueError("Input is too long (max 1000 characters)")

        return cleaned_input

    @validator("session_id")
    def validate_session_id(cls, v):
        """Validate session ID format."""
        if not v or not v.strip():
            raise ValueError("Session ID cannot be empty")

        # Check for valid characters (alphanumeric, hyphens, underscores)
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Session ID contains invalid characters")

        return v.strip()

    @validator("user_language")
    def validate_language(cls, v):
        """Validate language code."""
        from app.services.language_detection import language_service

        if v not in language_service.supported_languages:
            return "en"  # Default to English for unsupported languages
        return v.lower()


class WelcomeInput(BaseModel):
    session_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique session identifier"
    )
    browser_language: str = Field(
        default="en", max_length=10, description="Browser language tag"
    )

    @validator("session_id")
    def validate_session_id(cls, v):
        """Validate session ID format."""
        if not v or not v.strip():
            raise ValueError("Session ID cannot be empty")

        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Session ID contains invalid characters")

        return v.strip()


class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI response message")
    actions: list = Field(default_factory=list, description="Available action buttons")
    agent_type: str = Field(..., description="Agent that handled the request")
    confidence: float = Field(..., description="AI confidence in the response")
    language: str = Field(..., description="Response language")
    redirect_count: int = Field(default=0, description="Number of redirects in session")
    session_id: str = Field(..., description="Session identifier")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for debugging")
    timestamp: str = Field(..., description="Error timestamp")


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
        "Content-Security-Policy": "frame-ancestors 'self' https://bolablg.com https://*.bolablg.com https://ibola-chatbot-1055950842890.us-central1.run.app",
        # Revalidate the app shell so a new Cloud Run revision is picked up
        # without a hard refresh (PART 8.3 PWA freshness).
        "Cache-Control": "no-cache",
    }
    return FileResponse("static/index.html", headers=headers)


@app.post("/welcome", tags=["Chat"], response_model=Dict[str, Any])
async def get_welcome_message(payload: WelcomeInput):
    """Get localized welcome messages based on browser language (async with caching)."""
    start_time = time.time()

    try:
        session_id = payload.session_id
        browser_language = payload.browser_language

        logger.info(
            f"Welcome request for session {session_id}",
            extra={"session_id": session_id, "browser_language": browser_language},
        )

        # Check cache for localized content
        cache_key = f"welcome:{browser_language}"
        cached_content = await cache_service.get_localized_content(cache_key, "welcome")

        if cached_content:
            logger.info(f"Welcome cache hit for session {session_id}")
            return {
                "welcome_messages": json.loads(cached_content),
                "detected_language": browser_language.split("-")[0],
                "session_id": session_id,
                "cached": True,
            }

        # Detect user's preferred language
        detected_language = language_service.detect_language(browser_language)

        # Get localized welcome messages
        welcome_messages = language_service.get_welcome_messages(detected_language)

        # Cache the localized content
        await cache_service.set_localized_content(
            cache_key, "welcome", json.dumps(welcome_messages)
        )

        response = {
            "welcome_messages": welcome_messages,
            "detected_language": detected_language,
            "session_id": session_id,
            "cached": False,
        }

        # Log performance
        process_time = time.time() - start_time
        logging_service.log_performance_metric(
            "welcome_request",
            process_time,
            {"session_id": session_id, "detected_language": detected_language},
        )

        return response

    except Exception as e:
        logger.error(
            f"Welcome request failed: {e}", extra={"session_id": payload.session_id}
        )
        raise HTTPException(
            status_code=500, detail="Failed to generate welcome message"
        )


@app.options("/chat", tags=["Chat"])
async def chat_options():
    """Handle CORS preflight requests for /chat endpoint."""
    return JSONResponse(
        status_code=200,
        content={"message": "CORS preflight successful"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
        },
    )


@app.post("/chat", tags=["Chat"], response_model=Dict[str, Any])
async def chat(payload: ChatInput, request: Request):
    """Chat with the multi-agent system (async with caching)."""
    start_time = time.time()
    session_id = payload.session_id
    user_input = payload.user_input
    user_language = payload.user_language

    try:
        logger.info(
            f"Chat request from session {session_id}",
            extra={
                "session_id": session_id,
                "user_language": user_language,
                "input_length": len(user_input),
            },
        )

        # Check cache first for similar queries
        cached_response = await cache_service.get_cached_response(
            user_input, "unknown", user_language
        )

        if cached_response:
            logger.info(
                f"Cache hit for session {session_id}",
                extra={"session_id": session_id, "cached": True},
            )

            # Update cache metadata
            cached_response["cached"] = True
            cached_response["session_id"] = session_id
            # Never replay another turn's trace_id on a cache hit (Codex #6)
            cached_response["trace_id"] = None

            return cached_response

        # Get chat history from the configured store
        history = get_history(session_id)

        # Convert history format for ConversationalRetrievalChain (expects list of tuples)
        chat_history_tuples = [(h[0], h[1]) for h in history]

        # Collect request information for logging
        request_info = {
            "ip_address": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "referrer": request.headers.get("referer", "unknown"),
            "accept_language": request.headers.get("accept-language", "unknown"),
        }

        # Run the agentic LangGraph pipeline in a thread (the workflow is sync)
        import asyncio

        loop = asyncio.get_event_loop()
        service = get_agentic_service()
        result = await loop.run_in_executor(
            None,
            lambda: service.process_query(
                user_input, chat_history_tuples, session_id, user_language, request_info
            ),
        )

        # Calculate response time
        response_time = time.time() - start_time

        # Cache the response for future similar queries
        await cache_service.set_cached_response(
            user_input, result.get("agent_type", "unknown"), user_language, result
        )

        # Log chat interaction
        logging_service.log_chat_interaction(
            session_id=session_id,
            user_input=user_input,
            agent_type=result.get("agent_type", "unknown"),
            response=result.get("answer", ""),
            response_time=response_time,
            user_language=user_language,
            evidence=result.get("evidence", []),
            trace_id=result.get("trace_id"),
        )

        # Debug logging for development
        logger.debug(
            f"Agent: {result.get('agent_type', 'unknown')} | Confidence: {result.get('confidence', 0.0)}"
        )
        if "source_documents" in result:
            logger.debug(f"Source documents: {len(result['source_documents'])} found")

        # Prepare the response for the frontend with enhanced data
        response_for_frontend = {
            "answer": result.get("answer", ""),
            "actions": result.get("actions", []),
            "agent_type": result.get("agent_type", "redirect"),
            "confidence": result.get("confidence", 0.0),
            "language": result.get("language", user_language),
            "redirect_count": result.get("redirect_count", 0),
            "session_id": session_id,
            "response_time": round(response_time, 3),
            "cached": False,
            "should_end_chat": result.get(
                "should_end_chat", False
            ),  # Add the missing field
            "evidence": result.get("evidence", []),
            "trace_id": result.get("trace_id"),
        }

        # Run collector agent — detects opportunity intent and asks follow-up questions
        try:
            collector_result = collector_agent.check_and_respond(
                user_input=user_input,
                session_id=session_id,
                chat_history=chat_history_tuples,
                user_language=user_language,
                agent_response=result.get("answer", ""),
            )
            if collector_result and collector_result.get("follow_up_question"):
                response_for_frontend["answer"] += (
                    "\n\n" + collector_result["follow_up_question"]
                )
        except Exception as collector_err:
            logger.debug(f"Collector agent error (non-blocking): {collector_err}")

        # Update the history in the store
        append_history(
            session_id, (user_input, response_for_frontend.get("answer", ""))
        )

        return response_for_frontend

    except ValueError as ve:
        # Handle validation errors
        logger.warning(
            f"Validation error for session {session_id}: {ve}",
            extra={"session_id": session_id, "error_type": "validation"},
        )
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # Log the error with context
        error_context = {
            "session_id": session_id,
            "user_language": user_language,
            "input_length": len(user_input),
            "history_length": len(history) if "history" in locals() else 0,
            "user_agent": request.headers.get("user-agent"),
            "client_ip": request.client.host if request.client else "unknown",
        }

        logging_service.log_error(e, error_context)

        # Return user-friendly error message
        raise HTTPException(
            status_code=500,
            detail="I'm experiencing technical difficulties. Please try again in a moment.",
        )


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring (async)."""
    from datetime import datetime

    import psutil

    try:
        # Basic system metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        # Get cache statistics
        cache_stats = cache_service.get_cache_stats()

        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": APP_VERSION,
            "system": {
                "memory_usage": f"{memory.percent:.1f}%",
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_available": f"{memory.available / 1024 / 1024:.0f}MB",
            },
            "services": {
                "orchestrator": "available",
                "language_service": "healthy",
                "logging_service": "healthy",
                "cache_service": (
                    "healthy" if cache_stats.get("status") != "disabled" else "disabled"
                ),
                "rate_limiter": "healthy",
            },
            "performance": {
                "cache_stats": cache_stats,
                "rate_limit_stats": rate_limiter.get_global_stats(),
            },
        }

        # Check if services are accessible
        # Orchestrator is lazy-loaded, skip probing on health check

        logger.info("Health check performed", extra={"health_status": "healthy"})
        return health_data

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


@app.get("/cache/stats", tags=["Monitoring"])
async def get_cache_stats():
    """Get cache performance statistics."""
    try:
        stats = cache_service.get_cache_stats()
        return {"cache_stats": stats, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve cache statistics"
        )


@app.get("/rate-limit/stats", tags=["Monitoring"])
async def get_rate_limit_stats():
    """Get rate limiting statistics."""
    try:
        stats = rate_limiter.get_global_stats()
        return {"rate_limit_stats": stats, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Rate limit stats retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve rate limit statistics"
        )


@app.get("/performance/metrics", tags=["Monitoring"])
async def get_performance_metrics():
    """Get comprehensive performance metrics."""
    try:
        from datetime import datetime

        import psutil

        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage("/")

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / 1024 / 1024,
                "memory_available_mb": memory.available / 1024 / 1024,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            },
            "application": {
                "cache_stats": cache_service.get_cache_stats(),
                "rate_limit_stats": rate_limiter.get_global_stats(),
                "active_sessions": len(get_agentic_service().session_data),
            },
        }

        return metrics

    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve performance metrics"
        )


@app.post("/cache/clear", tags=["Maintenance"])
async def clear_cache():
    """Clear all cache data (admin endpoint)."""
    try:
        cache_service.clear_all_caches()
        logger.info("Cache cleared by admin request")
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@app.get("/session/{session_id}/stats", tags=["Session"], response_model=Dict[str, Any])
def get_session_stats(session_id: str):
    """Get statistics for a user session."""
    try:
        # Validate session ID
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="Invalid session ID")

        stats = get_agentic_service().get_session_stats(session_id)

        logger.info(
            f"Session stats retrieved for {session_id}",
            extra={"session_id": session_id, "stats": stats},
        )

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session stats for {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Unable to retrieve session statistics"
        )


@app.delete("/session/{session_id}", tags=["Session"])
def reset_session(session_id: str):
    """Reset a user session."""
    try:
        # Validate session ID
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="Invalid session ID")

        get_agentic_service().reset_session(session_id)

        logger.info(f"Session reset for {session_id}", extra={"session_id": session_id})

        return {"message": "Session reset successfully", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Unable to reset session")


# Contact Alert Models
class ContactAlertInput(BaseModel):
    contact_type: str = Field(
        ...,
        description="Type of contact request (e.g., booking_request, email_request)",
    )
    session_id: str = Field(..., description="Session ID of the chat")
    chat_history: List[Tuple[str, str]] = Field(
        ..., description="Full chat history leading to the contact request"
    )
    timestamp: str = Field(..., description="Timestamp of the request")


@app.post("/contact-alert", tags=["Chat"])
async def handle_contact_alert(payload: ContactAlertInput):
    """Handle contact alerts from frontend and forward to Google Chat."""
    try:
        logger.info(
            f"Received contact alert for session {payload.session_id}: {payload.contact_type}"
        )

        # Send alert to Google Chat (pass raw chat history, service will format it)
        result = google_chat_alert.send_contact_alert(
            contact_type=payload.contact_type,
            session_id=payload.session_id,
            chat_history=payload.chat_history,
        )

        if result:
            return {
                "status": "success",
                "message": "Contact alert sent to Google Chat.",
            }
        else:
            logger.warning(
                f"Contact alert not sent (Google Chat not configured) for session {payload.session_id}"
            )
            return {
                "status": "warning",
                "message": "Contact alert not sent - Google Chat not configured.",
            }

    except Exception as e:
        logger.error(f"Failed to send contact alert to Google Chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to send contact alert.")
