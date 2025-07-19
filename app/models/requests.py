from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat with the email marketing agent."""

    message: str = Field(..., description="User message to the agent")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation tracking"
    )
    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Previous conversation messages"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "I need help creating an email marketing strategy for my e-commerce business selling sustainable fashion to millennials.",
                "session_id": "user_123_session_456",
                "conversation_history": [
                    {"role": "user", "content": "Hello", "timestamp": 1640995200.0},
                    {
                        "role": "assistant",
                        "content": "Hello! How can I help with your email marketing strategy?",
                        "timestamp": 1640995205.0,
                    },
                ],
            }
        }


class DocumentUploadRequest(BaseModel):
    """Request model for document upload (used for form data validation)."""

    description: Optional[str] = Field(
        None, description="Description of the documents being uploaded"
    )
    category: Optional[str] = Field(
        None,
        description="Category of marketing materials (e.g., 'research', 'templates', 'case_studies')",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "description": "Email marketing best practices and case studies",
                "category": "research",
            }
        }


class RAGQueryRequest(BaseModel):
    """Request model for direct RAG queries."""

    query: str = Field(
        ..., description="Search query for retrieving relevant documents"
    )
    k: Optional[int] = Field(
        5, ge=1, le=20, description="Number of documents to retrieve"
    )
    similarity_threshold: Optional[float] = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Additional filters for document search"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "best practices for email subject lines in e-commerce",
                "k": 5,
                "similarity_threshold": 0.75,
                "filters": {"category": "research", "file_type": "pdf"},
            }
        }


class SessionRequest(BaseModel):
    """Request model for session operations."""

    session_id: str = Field(..., description="Session ID to retrieve or manage")

    class Config:
        json_schema_extra = {"example": {"session_id": "user_123_session_456"}}
