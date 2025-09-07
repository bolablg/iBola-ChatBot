# iBola Multi-Agent ChatBot 🤖

This project is an advanced Gemini-powered multi-agent RAG chatbot that helps people learn about Bolaji's professional background, education, and provides learning advice. The system features intelligent agent routing, dynamic guardrails, automatic language detection, and Google Chat integration.

## ✨ Key Features

### 🤖 Multi-Agent Architecture
- **Professional Agent**: Handles career, skills, projects, and work experience
- **Education Agent**: Focuses on academic background and qualifications
- **Learning Agent**: Provides advice on learning data science & AI skills
- **Redirect Agent**: Politely handles off-topic questions with helpful alternatives

### 🧠 Intelligent Features
- **Dynamic Guardrails**: Learns from conversations to improve routing accuracy
- **Automatic Language Detection**: Welcomes users in their browser's language (10+ languages supported)
- **Smart Query Routing**: Uses advanced pattern matching and context analysis
- **Session Tracking**: Monitors redirect counts and user behavior

### ⚡ Performance & Scalability
- **Async/Await Support**: Concurrent processing for better performance
- **Intelligent Caching**: TTLCache for responses, sessions, and localized content
- **Rate Limiting**: Advanced rate limiting with sliding windows and burst protection
- **Google Cloud Logging**: Structured logging with Cloud Logging integration

### 📱 Enhanced User Experience
- **Smooth Agent Transitions**: Animated switching between specialists
- **Intuitive Action Buttons**: Quick access to different conversation modes
- **Contact Integration**: Direct booking and email options with Google Chat alerts
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Real-time Feedback**: Loading states, typing indicators, and progress updates

### 🔒 Security & Reliability
- **Input Validation**: Comprehensive validation with security checks
- **Error Handling**: Global exception handling with detailed logging
- **Rate Limiting**: Protection against abuse and DoS attacks
- **Session Security**: Secure session management and validation

### 🔧 Advanced Functionality
- **Google Chat Alerts**: Automatic notifications for contact requests
- **Session Management**: Track and reset user sessions with analytics
- **Health Monitoring**: Comprehensive health checks and performance metrics
- **Cache Management**: Admin endpoints for cache monitoring and clearing
- **Fallback Handling**: Graceful error handling and recovery
- **Real-time Updates**: Live agent status and conversation flow

## 🚀 Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables:**

    Create a `.env` file in the project root and add the following variables:

    ```bash
    #----------------------------------------------------------------
    # Required: Gemini API Configuration
    #----------------------------------------------------------------
    GEMINI_API_KEY="your_gemini_api_key_here"

    #----------------------------------------------------------------
    # Optional: Google Chat Integration (for contact alerts)
    #----------------------------------------------------------------
    # Get webhook URL from: Google Chat -> Space Settings -> Manage Webhooks
    GCHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/.../messages?key=...&token=..."

    #----------------------------------------------------------------
    # Optional: Server Configuration
    #----------------------------------------------------------------
    HOST="0.0.0.0"
    PORT="8000"
    LOG_LEVEL="INFO"
    ALLOWED_ORIGIN_REGEX="https://(.+\\.)?bolablg\\.com"
    SESSION_TIMEOUT_MINUTES="30"
    MAX_REDIRECT_COUNT="3"
    ```

    #----------------------------------------------------------------
    # Database Configuration
    #----------------------------------------------------------------

    # DB_TYPE: The type of database to use. Can be "mariadb" or "sqlite".
    DB_TYPE="mariadb"

    # DB_NAME: The name of the SQLite database file.
    DB_NAME="feedback.db"

    # FEEDBACK_DB_CONN_URL: The connection URL for the MariaDB database.
    FEEDBACK_DB_CONN_URL="mariadb://user:password@host:port/database"

    #----------------------------------------------------------------
    # Chat History Storage
    #----------------------------------------------------------------

    # REDIS_URL: Connection URL for Redis used to store chat history.
    # If not set, the app falls back to an in-memory store.
    REDIS_URL="redis://localhost:6379/0"

    #----------------------------------------------------------------
    # Vector Store and Data Configuration
    #----------------------------------------------------------------

    # DB_PATH: The path to the vector store database.
    DB_PATH="chroma_db"

    # DATA_PATH: The path to the data directory.
    DATA_PATH="data"

    #----------------------------------------------------------------
    # Google Drive and Chat Configuration
    #----------------------------------------------------------------

    # GDRIVE_FOLDER_ID: The ID of the Google Drive folder to sync.
    GDRIVE_FOLDER_ID="your_google_drive_folder_id"

    # GCHAT_WEBHOOK_URL: The URL of the Google Chat webhook to send alerts to.
    GCHAT_WEBHOOK_URL="your_google_chat_webhook_url"

    #----------------------------------------------------------------
    # Credentials Configuration
    #----------------------------------------------------------------

    # GCP_SA_CRENDIALS_PATH: The path to the GCP service account credentials file.
    GCP_SA_CRENDIALS_PATH="_conf/ibola_agent_sa.json"

    # GOOGLE_OAUTH_CREDENTIALS_PATH: The path to the Google OAuth credentials file.
    GOOGLE_OAUTH_CREDENTIALS_PATH="_conf/ibola_agent_oauth.json"
    ```

    Setting the `REDIS_URL` environment variable enables Redis-backed
    chat history. If it is unset, the application stores history in
    memory, which clears on restart.

3.  **Google Drive API Setup:**

    *   Enable the Google Drive API in your Google Cloud Platform project.
    *   Create an OAuth 2.0 Client ID and download the `credentials.json` file.
    *   Move the `credentials.json` file to the path specified in the `GOOGLE_OAUTH_CREDENTIALS_PATH` environment variable.
    *   Share your Google Drive folder with the client email found in your `credentials.json` file.

## Running the Application with Docker

To run the application with Docker, you can use the provided `docker-compose.yml` file.

1.  **Build the Docker image:**

    ```bash
    docker-compose build
    ```

2.  **Run the Docker container:**

    ```bash
    docker-compose up -d
    ```

This will build the Docker image and run the container in the background. The cron job will be set up automatically and the vector store will be updated every day at midnight. An alert will be sent to the Google Chat webhook with the changes.

## 🧪 Testing

The project includes comprehensive testing with multiple test suites:

### Test Suites

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test end-to-end system functionality
- **Security Tests**: Test security vulnerabilities and input validation
- **Performance Tests**: Test system performance and scalability

### Running Tests

#### Run All Tests
```bash
# Run all test suites
python run_tests.py all

# Or use pytest directly
pytest tests/ -v --cov=app --cov-report=html
```

#### Run Specific Test Types
```bash
# Run only unit tests
python run_tests.py unit

# Run integration tests
python run_tests.py integration

# Run security tests
python run_tests.py security

# Run performance tests
python run_tests.py performance

# Run linting checks
python run_tests.py lint
```

#### CI/CD Mode
```bash
# Run tests in CI mode (exits with error code on failure)
python run_tests.py --ci unit integration security
```

### Test Coverage

The test suite provides:
- **Code Coverage**: >80% coverage target
- **Security Testing**: SQL injection, XSS, input validation
- **Performance Testing**: Response times, memory usage, concurrency
- **Integration Testing**: Full system workflow testing

### Test Reports

After running tests, reports are generated:
- **HTML Coverage Report**: `htmlcov/index.html`
- **XML Coverage Report**: `coverage.xml`
- **Test Results**: Console output with detailed results

## Running the Application Locally

### Development Mode
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
# Install production dependencies only
pip install -r requirements.txt

# Run with production settings
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Once running, the API documentation will be available at `http://127.0.0.1:8000/docs`.

### Environment Setup

The application supports loading environment variables from multiple sources with the following priority:

1. **Environment Variables** (highest priority)
2. **`.env.local`** file (local overrides)
3. **`.env`** file (default values)
4. **Default values** in code (lowest priority)

#### Quick Setup

1. **Copy the sample environment file:**
   ```bash
   cp sample.env .env
   ```

2. **Edit `.env` with your actual values:**
   ```bash
   # Required
   GEMINI_API_KEY="your_actual_gemini_api_key_here"

   # Optional - Google Cloud Integration
   GCHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/.../messages?key=...&token=..."
   GCP_SA_CREDENTIALS_PATH="/path/to/your/service-account.json"
   GCP_PROJECT_ID="your-gcp-project-id"

   # Optional - Performance & Security
   HOST="0.0.0.0"
   PORT="8000"
   LOG_LEVEL="INFO"
   SESSION_TIMEOUT_MINUTES="30"
   MAX_REDIRECT_COUNT="3"
   ```

3. **For local overrides (optional):**
   ```bash
   cp .env .env.local
   # Edit .env.local with your local-specific values
   ```

#### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | - | Google Gemini API key for AI functionality |
| `GCHAT_WEBHOOK_URL` | ❌ | - | Google Chat webhook for contact alerts |
| `GCP_SA_CREDENTIALS_PATH` | ❌ | `_conf/ibola_agent_sa.json` | Path to Google Cloud service account key |
| `GCP_PROJECT_ID` | ❌ | `your-gcp-project-id` | Google Cloud project ID |
| `HOST` | ❌ | `0.0.0.0` | Server host address |
| `PORT` | ❌ | `8000` | Server port |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `SESSION_TIMEOUT_MINUTES` | ❌ | `30` | Session timeout in minutes |
| `MAX_REDIRECT_COUNT` | ❌ | `3` | Maximum redirects before ending chat |

#### Testing Environment Loading

Test that your environment variables are loaded correctly:

```bash
# Test configuration loading
python test_env_loading.py

# Test with your actual API key
GEMINI_API_KEY="your_actual_key" python test_config_fix.py
```

#### Docker Environment Variables

When running with Docker, you can pass environment variables:

```bash
# Using environment variables
docker run -e GEMINI_API_KEY="your_key" -e PORT="8080" your-image

# Using .env file
docker run --env-file .env your-image
```

## 📋 API Endpoints

### Core Chat Endpoints
- `POST /welcome` - Get localized welcome messages based on browser language
- `POST /chat` - Main chat endpoint with multi-agent routing and caching
- `GET /health` - Comprehensive health check and system status

### Session Management
- `GET /session/{session_id}/stats` - Get session statistics and analytics
- `DELETE /session/{session_id}` - Reset user session data

### Monitoring & Analytics
- `GET /cache/stats` - Cache performance statistics and hit rates
- `GET /rate-limit/stats` - Rate limiting statistics and blocked clients
- `GET /performance/metrics` - Comprehensive system and application metrics
- `POST /cache/clear` - Clear all cache data (admin maintenance)

### Configuration & Environment
All configuration is handled through environment variables in a `.env` file:

```bash
# Required
GEMINI_API_KEY="your_gemini_api_key_here"

# Optional - Google Cloud Integration
GCHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/.../messages?key=...&token=..."
GCP_SA_CREDENTIALS_PATH="/path/to/service-account.json"
GCP_PROJECT_ID="your-gcp-project-id"

# Optional - Performance & Security
HOST="0.0.0.0"
PORT="8000"
LOG_LEVEL="INFO"
ALLOWED_ORIGIN_REGEX="https://(.+\\.)?bolablg\\.com"
SESSION_TIMEOUT_MINUTES="30"
MAX_REDIRECT_COUNT="3"
```

## 🔍 Monitoring & Observability

### Health Checks
The `/health` endpoint provides comprehensive system health information:
- System resource usage (CPU, memory, disk)
- Service status (orchestrator, cache, rate limiter, logging)
- Cache and rate limiting statistics
- Active session count and performance metrics

### Logging & Analytics
- **Local Logging**: Structured logs in `logs/chatbot.log`
- **Google Cloud Logging**: Automatic cloud logging when GCP credentials are configured
- **Performance Metrics**: Request duration, cache hit rates, error rates
- **Security Events**: Rate limiting violations, blocked requests, validation failures
- **Chat Analytics**: Agent usage, response times, user language preferences

### Cache & Performance Monitoring
- **Response Caching**: TTLCache for chat responses (30-minute TTL)
- **Session Caching**: User session data caching (1-hour TTL)
- **Language Caching**: Localized content caching (2-hour TTL)
- **Rate Limiting**: Sliding window rate limiting with burst protection

## 🛡️ Security Features

### Input Validation
- Comprehensive input sanitization and validation
- SQL injection and XSS protection
- Harmful content pattern detection
- Session ID format validation

### Rate Limiting
- Per-endpoint rate limits (30 requests/minute for chat)
- Global rate limits (60 requests/minute)
- Burst protection (10 requests per 10 seconds)
- Client blocking for abusive behavior

### Error Handling
- Global exception handlers
- User-friendly error messages
- Detailed internal logging
- Graceful service degradation

## Updating the Vectorstore

To sync your Google Drive folder and update the vectorstore, run the following command:

```bash
python pipeline/sync.py
```

This will sync the Google Drive folder with the local `data` folder and update the vectorstore only if there are changes.

### Automating with Cron Job

To automate the synchronization process, you can use the provided `setup_cron.sh` script to create a cron job.

1.  **Make the script executable:**

    ```bash
    chmod +x setup_cron.sh
    ```

2.  **Run the script:**

    ```bash
    ./setup_cron.sh
    ```

    The script will prompt you to enter the desired cron schedule.

## Deploying to Google Cloud

To deploy the main FastAPI application, we will use Google Cloud Run.

**1. Build the Docker Image:**

```bash
gcloud builds submit --tag gcr.io/<your-gcp-project-id>/ibola-chatbot
```

*   Replace `<your-gcp-project-id>` with your Google Cloud project ID.

**2. Deploy to Cloud Run:**

```bash
gcloud run deploy ibola-chatbot \ 
    --image gcr.io/<your-gcp-project-id>/ibola-chatbot \ 
    --platform managed \ 
    --region <your-gcp-region> \ 
    --allow-unauthenticated
```

*   Replace `<your-gcp-project-id>` with your Google Cloud project ID.
*   Replace `<your-gcp-region>` with the Google Cloud region where you want to deploy the application.