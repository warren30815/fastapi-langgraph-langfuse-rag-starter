from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Response model for chat interactions."""

    success: bool = Field(..., description="Whether the request was successful")
    strategy: Optional[Dict[str, Any]] = Field(
        None, description="Generated email marketing strategy"
    )
    session_id: str = Field(..., description="Session ID for the conversation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")
    sources: Optional[List[Dict[str, Any]]] = Field(
        None, description="Source documents used"
    )
    error: Optional[str] = Field(None, description="Error message if request failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "strategy": {
                    "executive_summary": "Comprehensive email marketing strategy for sustainable fashion e-commerce...",
                    "target_audience": {
                        "primary": "Environmentally conscious millennials aged 25-40",
                        "secondary": "Gen Z early adopters interested in sustainable fashion",
                    },
                    "campaign_types": [
                        {
                            "type": "Welcome Series",
                            "frequency": "Triggered",
                            "description": "5-email onboarding sequence",
                        }
                    ],
                },
                "session_id": "user_123_session_456",
                "metadata": {
                    "iterations": 2,
                    "documents_used": 5,
                    "processing_steps": "completed",
                },
                "sources": [
                    {
                        "source": "email_marketing_best_practices.pdf",
                        "chunk_index": 0,
                        "similarity_score": 0.89,
                    }
                ],
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload operations."""

    success: bool = Field(..., description="Whether the upload was successful")
    message: str = Field(..., description="Status message")
    total_files: int = Field(..., description="Total number of files processed")
    successful_files: int = Field(
        ..., description="Number of files successfully processed"
    )
    total_chunks: int = Field(..., description="Total number of text chunks created")
    total_tokens: int = Field(..., description="Total tokens processed")
    processing_results: List[Dict[str, Any]] = Field(
        ..., description="Detailed processing results for each file"
    )
    error: Optional[str] = Field(None, description="Error message if upload failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Documents uploaded and processed successfully",
                "total_files": 2,
                "successful_files": 2,
                "total_chunks": 15,
                "total_tokens": 12000,
                "processing_results": [
                    {
                        "filename": "email_best_practices.pdf",
                        "status": "success",
                        "chunks_created": 8,
                        "total_tokens": 6500,
                    },
                    {
                        "filename": "case_studies.docx",
                        "status": "success",
                        "chunks_created": 7,
                        "total_tokens": 5500,
                    },
                ],
            }
        }


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries."""

    success: bool = Field(..., description="Whether the query was successful")
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(..., description="Retrieved documents")
    total_results: int = Field(..., description="Total number of results found")
    retrieval_time_ms: int = Field(
        ..., description="Time taken for retrieval in milliseconds"
    )
    error: Optional[str] = Field(None, description="Error message if query failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "email subject line best practices",
                "results": [
                    {
                        "id": 42,
                        "text": "Effective email subject lines should be concise, personalized, and create urgency...",
                        "source": "email_marketing_guide.pdf",
                        "similarity_score": 0.89,
                        "chunk_index": 5,
                        "file_type": "pdf",
                    }
                ],
                "total_results": 5,
                "retrieval_time_ms": 150,
            }
        }


class SessionResponse(BaseModel):
    """Response model for session information."""

    success: bool = Field(..., description="Whether the request was successful")
    session_id: str = Field(..., description="Session ID")
    conversation_history: List[Dict[str, Any]] = Field(
        ..., description="Conversation messages"
    )
    session_metadata: Dict[str, Any] = Field(..., description="Session metadata")
    error: Optional[str] = Field(None, description="Error message if request failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "session_id": "user_123_session_456",
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "I need help with email marketing",
                        "timestamp": 1640995200.0,
                    },
                    {
                        "role": "assistant",
                        "content": "I'd be happy to help! What's your business type?",
                        "timestamp": 1640995205.0,
                    },
                ],
                "session_metadata": {
                    "created_at": 1640995200.0,
                    "last_activity": 1640995205.0,
                    "total_messages": 2,
                    "strategies_generated": 1,
                },
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: float = Field(..., description="Response timestamp")
    dependencies: Optional[Dict[str, str]] = Field(
        None, description="Status of service dependencies"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "email-marketing-agent",
                "version": "0.1.0",
                "timestamp": 1640995200.0,
                "dependencies": {
                    "vector_store": "healthy",
                    "langfuse": "healthy",
                    "openai": "healthy",
                },
            }
        }


class RAGStatsResponse(BaseModel):
    """Response model for RAG system statistics."""

    success: bool = Field(..., description="Whether the request was successful")
    total_documents: int = Field(
        ..., description="Total number of documents in the vector store"
    )
    sources: Dict[str, int] = Field(..., description="Document count by source file")
    file_types: Dict[str, int] = Field(..., description="Document count by file type")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    chunk_size: int = Field(..., description="Configured chunk size")
    similarity_threshold: float = Field(
        ..., description="Configured similarity threshold"
    )
    error: Optional[str] = Field(None, description="Error message if request failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_documents": 150,
                "sources": {
                    "email_best_practices.pdf": 45,
                    "case_studies.docx": 38,
                    "market_research.csv": 67,
                },
                "file_types": {"pdf": 83, "docx": 38, "csv": 29},
                "embedding_dimension": 384,
                "chunk_size": 800,
                "similarity_threshold": 0.7,
            }
        }
