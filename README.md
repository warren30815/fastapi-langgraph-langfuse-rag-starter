# Email Marketing Strategy Agent

A comprehensive email marketing strategy recommendation system powered by LangGraph, FastAPI, and LangFuse for complete LLMOps monitoring.

## 🚀 Features

- **AI-Powered Strategy Generation**: LangGraph agent analyzes customer data and generates personalized email marketing strategies
- **RAG-Enhanced Knowledge Base**: Upload and query marketing research documents, best practices, and case studies
- **Comprehensive LLMOps Monitoring**: Complete observability with LangFuse integration
- **FAISS Vector Search**: Efficient similarity search for relevant marketing content
- **Multi-format Document Support**: Process PDF, DOCX, TXT, CSV, and Markdown files
- **Structured Logging**: JSON-formatted logs with request tracing for production monitoring
- **Docker Compose Deployment**: Easy setup with PostgreSQL, and LangFuse

## 🛠 Technology Stack

- **Backend**: FastAPI with Python 3.12
- **Agent Framework**: LangGraph for conversational AI workflows
- **LLMOps**: LangFuse for tracing, monitoring, and prompt management
- **Vector Database**: FAISS for efficient similarity search
- **Document Processing**: PyPDF2, python-docx, pandas for multi-format support
- **Logging**: structlog for structured JSON logging
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

The LangGraph agent follows a structured workflow:

1. **Customer Analysis** - Extracts business type, target audience, goals, and constraints
2. **Knowledge Retrieval** - Searches RAG system for relevant marketing materials
3. **Strategy Generation** - Creates comprehensive email marketing strategy
4. **Strategy Refinement** - Enhances strategy with specific recommendations
5. **Finalization** - Packages strategy with metadata and sources

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
    "session_id": "demo_session"
  }'
```

### 3. Query Knowledge Base

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

## 🌱 Initial Data Setup

The project includes sample marketing documents and a seeding script:

```bash
# Seed the vector database with example documents
uv run python seed_data.py

# Or within Docker
docker-compose exec app python seed_data.py
```

Sample documents included:

- Customer segmentation strategies
- Email marketing best practices
- Real-world case studies and examples
