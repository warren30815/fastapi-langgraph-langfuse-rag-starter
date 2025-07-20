from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config.settings import settings


class ChatRequest(BaseModel):
    """Request model for chat with the email marketing agent."""

    message: str = Field(..., description="User message to the agent")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation tracking"
    )
    user_id: str = Field(
        ..., description="User ID for fetching customer context from the database"
    )
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Previous conversation messages"
    )


class DocumentUploadRequest(BaseModel):
    """Request model for document upload (used for form data validation)."""

    description: Optional[str] = Field(
        None, description="Description of the documents being uploaded"
    )
    category: Optional[str] = Field(
        None,
        description="Category of marketing materials (e.g., 'research', 'templates', 'case_studies')",
    )


class RAGQueryRequest(BaseModel):
    """Request model for direct RAG queries."""

    query: str = Field(
        ..., description="Search query for retrieving relevant documents"
    )
    k: Optional[int] = Field(
        settings.max_retrieval_results,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )
    similarity_threshold: Optional[float] = Field(
        settings.similarity_threshold,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score",
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Additional filters for document search"
    )
