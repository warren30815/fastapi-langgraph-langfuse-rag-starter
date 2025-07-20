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


class SessionResponse(BaseModel):
    """Response model for session information."""

    success: bool = Field(..., description="Whether the request was successful")
    session_id: str = Field(..., description="Session ID")
    conversation_history: List[Dict[str, Any]] = Field(
        ..., description="Conversation messages"
    )
    session_metadata: Dict[str, Any] = Field(..., description="Session metadata")
    error: Optional[str] = Field(None, description="Error message if request failed")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: float = Field(..., description="Response timestamp")
    dependencies: Optional[Dict[str, str]] = Field(
        None, description="Status of service dependencies"
    )


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
