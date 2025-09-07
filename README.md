# iBola Multi-Agent ChatBot 🤖

An advanced Gemini-powered multi-agent RAG chatbot that helps people learn about Bolaji's professional background, education, and provides learning advice. The system features intelligent agent routing, dynamic guardrails, automatic language detection, and Google Chat integration.

## 🏗️ System Architecture

### Core Architecture Components

#### 🤖 **Multi-Agent Orchestration System**
The system uses a sophisticated agent-based architecture with intelligent routing:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Orchestrator  │───▶│ Classification  │───▶│   Specialized   │
│                 │    │     Agent       │    │     Agents      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                    │
         └────────────────────────┼────────────────────┘
                                  ▼
                    ┌─────────────────┐
                    │  Dynamic        │
                    │  Guardrails     │
                    └─────────────────┘
```

- **Orchestrator**: Central routing component that analyzes user queries and routes them to appropriate agents
- **Classification Agent**: Uses advanced pattern matching and context analysis to determine query intent
- **Specialized Agents**: Domain-specific agents for professional, education, learning, and redirect scenarios
- **Dynamic Guardrails**: Machine learning-powered system that learns from conversations to improve routing accuracy

#### 🧠 **Advanced AI Services Layer**
```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Advanced RAG   │ │  Memory Mgmt    │ │  Knowledge      │
│                 │ │                 │ │  Graph          │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                          ▼
                ┌─────────────────┐
                │  Continuous     │
                │  Learning       │
                └─────────────────┘
```

- **Advanced RAG**: Query expansion, semantic reranking, hybrid search (vector + keyword + TF)
- **Memory Management**: Summarization and compression for long-term conversation context
- **Knowledge Graph**: Entity extraction and dynamic graph construction for enhanced reasoning
- **Continuous Learning**: Automated model updates and performance tracking

#### 📊 **Data Processing Pipeline**
```
Google Drive ──▶ Local Sync ──▶ Vector Store ──▶ FAISS Index
     │                │              │              │
     └──────▶ Document Processing ───┘              │
                    │                              │
                    └─────────────▶ Reranking ──────┘
```

### Key Architectural Patterns

#### **1. Agent-Based Architecture**
- **Separation of Concerns**: Each agent specializes in a specific domain
- **Intelligent Routing**: Classification agent determines optimal agent selection
- **Fallback Mechanisms**: Graceful degradation when primary agents are unavailable

#### **2. Event-Driven Communication**
- **Inter-Agent Communication**: Task delegation and message passing between agents
- **Asynchronous Processing**: Non-blocking operations for better performance
- **Event Streaming**: Real-time updates and status notifications

#### **3. Multi-Level Caching Strategy**
- **Response Cache**: TTLCache for chat responses (30-minute TTL)
- **Session Cache**: User session data (1-hour TTL)
- **Language Cache**: Localized content (2-hour TTL)
- **Vector Cache**: Pre-computed embeddings and search indices

#### **4. Security-First Design**
- **Input Validation**: Multi-layer validation with security checks
- **Rate Limiting**: Sliding window protection against abuse
- **Session Security**: Secure session management and validation
- **Error Isolation**: Global exception handling with detailed logging

## ✨ Key Features

### 🤖 **Multi-Agent System**
- **Professional Agent**: Career, skills, projects, and work experience
- **Education Agent**: Academic background and qualifications
- **Learning Agent**: Data science & AI learning advice
- **Redirect Agent**: Off-topic handling with helpful alternatives

### 🧠 **Intelligent Features**
- **Dynamic Guardrails**: ML-powered routing accuracy improvement
- **Automatic Language Detection**: 10+ language support
- **Smart Query Routing**: Advanced pattern matching and context analysis
- **Session Analytics**: Redirect count monitoring and user behavior tracking

### ⚡ **Performance & Scalability**
- **Async Processing**: Concurrent operations for optimal performance
- **Intelligent Caching**: Multi-level TTLCache system
- **Advanced Rate Limiting**: Sliding windows with burst protection
- **Google Cloud Logging**: Structured logging and monitoring

### 🔒 **Security & Reliability**
- **Input Validation**: Comprehensive security checks
- **Error Handling**: Global exception handling with logging
- **Rate Limiting**: Protection against abuse and DoS attacks
- **Session Security**: Secure session management and validation

## 🚀 Setup

### Prerequisites
- Python 3.12+
- Google Gemini API key
- Google Cloud Project (optional, for cloud features)

### Installation

1. **Clone and Install Dependencies:**
   ```bash
   git clone <repository-url>
   cd ibola-chatbot
   pip install -r requirements.txt
   ```

2. **Environment Configuration:**

   Create a `.env` file in the project root with your configuration:

   ```bash
   # Required: AI API Key
   GEMINI_API_KEY=your_gemini_api_key_here

   # Optional: Google Cloud Integration
   GCHAT_WEBHOOK_URL=https://chat.googleapis.com/v1/spaces/.../messages?key=...&token=...
   GCP_SA_CREDENTIALS_PATH=_conf/ibola_agent_sa.json
   GCP_PROJECT_ID=your_project_id

   # Optional: Server Configuration
   HOST=0.0.0.0
   PORT=8000
   LOG_LEVEL=INFO
   SESSION_TIMEOUT_MINUTES=30
   MAX_REDIRECT_COUNT=3
   ```

3. **Google Cloud Setup (Optional):**

   For cloud features like logging and Google Drive integration:

   - Enable required Google Cloud APIs (Cloud Logging, Drive API)
   - Create service account with appropriate permissions
   - Download service account key to the path specified in `GCP_SA_CREDENTIALS_PATH`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose build
docker-compose up -d

# Or run with Docker directly
docker build -t ibola-chatbot .
docker run -p 8000:8000 --env-file .env ibola-chatbot
```

## 🧪 Testing & Development

### Running the Application

#### Development Mode
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode
```bash
# Run with production settings
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Once running, the API documentation is available at `http://127.0.0.1:8000/docs`.

### Test Suites

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end system testing
- **Security Tests**: Vulnerability and input validation
- **Performance Tests**: Load and scalability testing

#### Running Tests
```bash
# Run all tests with coverage
pytest tests/ -v --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_agents.py tests/test_services.py -v
pytest tests/test_integration.py -v
pytest tests/test_security.py -v
```

## 🔧 API Endpoints

### Core Chat Endpoints
- `POST /welcome` - Localized welcome messages based on browser language
- `POST /chat` - Main chat endpoint with multi-agent routing and caching
- `GET /health` - Comprehensive system health check

### Session Management
- `GET /session/{session_id}/stats` - Session statistics and analytics
- `DELETE /session/{session_id}` - Reset user session data

### Monitoring & Analytics
- `GET /cache/stats` - Cache performance statistics
- `GET /rate-limit/stats` - Rate limiting statistics
- `GET /performance/metrics` - System performance metrics
- `POST /cache/clear` - Clear all cache data (admin)

## 🚀 CI/CD & Deployment

### GitHub Actions Configuration

The project uses GitHub Actions for automated testing and deployment. Configure the following repository secrets:

#### Required Secrets
- `GCP_SA_KEY` - Google Cloud Service Account JSON key
- `GCP_PROJECT` - Google Cloud Project ID
- `GOOGLE_OAUTH_KEY` - OAuth client credentials JSON
- `GOOGLE_OAUTH_TOKEN` - OAuth refresh token JSON
- `GEMINI_API_KEY` - Google AI Gemini API key
- `GCHAT_WEBHOOK_URL` - Google Chat webhook URL
- `GDRIVE_FOLDER_ID` - Google Drive folder ID
- `REDIRECT_LOG_SHEET_ID` - Google Sheets ID for analytics

#### Optional Secrets
- `GOOGLE_OAUTH_KEY_PATH` - Path for OAuth credentials (default: `_conf/ibola_agent_oauth.json`)
- `GOOGLE_OAUTH_TOKEN_PATH` - Path for OAuth token (default: `_conf/token.json`)
- `DATA_PATH` - Path to data directory
- `DB_PATH` - Path to database/vector store

### Deployment to Google Cloud

```bash
# Build Docker image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/ibola-chatbot

# Deploy to Cloud Run
gcloud run deploy ibola-chatbot \
  --image gcr.io/YOUR_PROJECT/ibola-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### GitHub Actions Updates

The CI/CD pipeline uses the latest stable versions of all actions:

| Action | Version | Status |
|--------|---------|--------|
| `actions/checkout` | `v4` | ✅ Updated |
| `actions/upload-artifact` | `v4` | ✅ Updated |
| `codecov/codecov-action` | `v4` | ✅ Updated |
| `google-github-actions/auth` | `v2` | ✅ Updated |
| `google-github-actions/setup-gcloud` | `v2` | ✅ Updated |

## 🔍 Monitoring & Observability

### Health Monitoring
- **System Health**: CPU, memory, disk usage monitoring
- **Service Status**: Orchestrator, cache, rate limiter, logging status
- **Performance Metrics**: Request duration, cache hit rates, error rates
- **Session Analytics**: Active sessions, redirect patterns, user behavior

### Logging & Analytics
- **Structured Logging**: JSON-formatted logs with context
- **Google Cloud Logging**: Automatic cloud logging integration
- **Security Events**: Rate limiting violations, validation failures
- **Chat Analytics**: Agent usage, response times, language preferences

### Cache & Performance
- **Multi-Level Caching**: Response, session, language, and vector caching
- **Rate Limiting**: Sliding window protection with burst control
- **Resource Monitoring**: Real-time system resource tracking

## 🛡️ Security & Reliability

### Input Security
- **Validation**: Comprehensive input sanitization and validation
- **Injection Protection**: SQL injection and XSS prevention
- **Content Filtering**: Harmful content pattern detection
- **Session Security**: Secure session management and validation

### System Reliability
- **Error Handling**: Global exception handlers with graceful degradation
- **Rate Limiting**: Per-endpoint and global rate limiting
- **Health Monitoring**: Automated health checks and alerts
- **Fallback Mechanisms**: Graceful degradation when services fail

## 📊 Data Pipeline

### Vector Store Management
```bash
# Update vector store from Google Drive
python pipeline/sync.py
```

### Automated Updates
The system supports automated vector store updates via cron jobs or scheduled tasks for keeping the knowledge base current.

---

## 📄 License & Contributing

This project demonstrates advanced AI chatbot architecture with multi-agent systems, RAG, and cloud integration. For contributions or questions, please refer to the project documentation.

---

*Built with ❤️ using FastAPI, LangChain, Google Gemini, and modern AI practices.*