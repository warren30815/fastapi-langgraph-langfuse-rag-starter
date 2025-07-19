from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agent, document
from app.config.logging_config import LoggingMiddleware, configure_logging, get_logger
from app.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger = get_logger("startup")

    # Configure logging
    configure_logging(settings.log_level)
    logger.info("Logging configured", log_level=settings.log_level)

    # Initialize vector store (will be created in rag.py)
    logger.info("Application startup complete")

    yield

    # Cleanup
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Email Marketing Agent",
    description="Email marketing strategy recommendation agent with LLMOps monitoring",
    version="0.1.0",
    lifespan=lifespan,
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent.router, prefix="/api/v1/agent", tags=["agent"])
app.include_router(document.router, prefix="/api/v1/documents", tags=["documents"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Email Marketing Agent API", "version": "0.1.0"}


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    logger = get_logger("health")
    logger.info("Health check requested")
    return {"status": "healthy", "service": "email-marketing-agent", "version": "0.1.0"}
