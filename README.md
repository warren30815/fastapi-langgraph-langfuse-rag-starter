# Email Marketing Strategy Agent

A comprehensive email marketing strategy recommendation system powered by LangGraph, FastAPI, and LangFuse for complete LLMOps monitoring.

## 🚀 Features

- **AI-Powered Strategy Generation**: LangGraph agent analyzes customer data and generates personalized email marketing strategies
- **RAG-Enhanced Knowledge Base**: Upload and query marketing research documents, best practices, and case studies
- **Comprehensive LLMOps Monitoring**: Complete observability with LangFuse integration
- **FAISS Vector Search**: Efficient similarity search for relevant marketing content
- **Multi-format Document Support**: Process PDF, DOCX, TXT, CSV, and Markdown files
- **OpenTelemetry Logging**: Standardized observability format with trace correlation and Langfuse request context
- **Docker Compose Deployment**: Easy setup with PostgreSQL, and LangFuse

## 🛠 Technology Stack

- **Backend**: FastAPI with Python 3.12
- **Agent Framework**: LangGraph for conversational AI workflows
- **LLMOps**: LangFuse for tracing, monitoring, and prompt management
- **Vector Database**: FAISS for efficient similarity search
- **Document Processing**: PyPDF2, python-docx, pandas for multi-format support
- **Logging**: structlog with OpenTelemetry format for standardized observability
- **Package Management**: uv for fast dependency management
- **Deployment**: Docker Compose with PostgreSQL

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local development)
- OpenAI API key
- SWIG (for FAISS installation): `brew install swig` on macOS

### 1. Clone and Setup

```bash
git clone <repository-url>
cd <repository-name>
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file with your credentials

### 3. Start Services

```bash
# Start all services (PostgreSQL, LangFuse, FastAPI)
docker-compose up -d --build

# View logs
docker-compose logs -f app
```

### 4. Access Applications

- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **LangFuse Dashboard**: http://localhost:3000

## 🧠 Agent Workflow

The LangGraph agent follows a structured workflow with complete observability:

1. **Request Context Setup** - Creates Langfuse trace and sets request correlation for OpenTelemetry logging
2. **Customer Context Fetch** - Retrieves user profile and business context from mock database with simulated latency
3. **Knowledge Retrieval** - Searches RAG system using combined user query and customer context
4. **Conditional Web Search** - Falls back to simulated web search if insufficient knowledge base results
5. **Strategy Generation** - Uses LLM with retrieved context to create targeted email marketing recommendations
6. **Strategy Finalization** - Packages strategy with metadata, iteration count, and source attribution

Each step is instrumented with Langfuse spans and structured logging, providing complete request traceability from HTTP request through agent execution.

## 📊 LangFuse Monitoring

The system provides comprehensive observability:

- **Trace Tracking**: Complete request flows with spans and context
- **LLM Metrics**: Token usage, costs, latency, and model performance
- **RAG Monitoring**: Retrieval quality, similarity scores, and source tracking
- **Custom Metrics**: Business KPIs and agent performance indicators
- **Error Tracking**: Detailed error logs with context and recovery information

## 🧪 Local Development

### Setup Development Environment

```bash
# Install dependencies with uv
uv sync

# Start local development server
uv run python run_local.py

# Start only external services for local dev
docker-compose up postgres langfuse-server -d
```

### Code Formatting

```bash
# Format code with Black and isort
uv run python format.py
```

## 🌱 Initial Data Setup

The project includes sample marketing documents and a seeding script:

```bash
# Seed the vector database with example documents
uv run python run_local.py
uv run python seed_data.py

# Or within Docker
docker-compose exec app python seed_data.py
```

Sample documents included:

- Customer segmentation strategies
- Email marketing best practices

## 📄 Example Usage

### 1. Upload Marketing Documents

Upload your marketing research, best practices, and case studies:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "files=@marketing_guide.pdf" \
  -F "files=@case_studies.docx" \
  -F "description=Email marketing resources" \
  -F "category=research"
```

### 2. Get Strategy Recommendation

Request a personalized email marketing strategy:

```bash
curl -X POST "http://localhost:8000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How can I use behavioral segmentation to improve the effectiveness of my email marketing campaigns?",
    "session_id": "session_id",
    "user_id": "user_id"
  }'
```

### 3. Query Knowledge Base (RAG)

Search for specific marketing information:

```bash
curl -X POST "http://localhost:8000/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can I use behavioral segmentation to improve the effectiveness of my email marketing campaigns?",
    "k": 3,
    "similarity_threshold": 0.5
  }'
```
