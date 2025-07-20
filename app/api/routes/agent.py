import uuid

from fastapi import APIRouter, HTTPException, status

from app.config.logging_config import get_logger
from app.core.agent import email_agent
from app.core.rag import rag_system
from app.models.requests import ChatRequest, RAGQueryRequest
from app.models.responses import (
    ChatResponse,
    RAGQueryResponse,
    RAGStatsResponse,
    SessionResponse,
)

router = APIRouter()
logger = get_logger("agent_routes")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest) -> ChatResponse:
    """Chat with the email marketing strategy agent."""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"{uuid.uuid4().hex}"

        logger.info(
            "Chat request received",
            session_id=session_id,
            message_length=len(request.message),
        )

        # Process request with agent
        result = await email_agent.process_request(
            message=request.message,
            user_id=request.user_id,
            session_id=session_id,
            conversation_history=request.conversation_history,
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error("Chat request failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}",
        )


@router.post("/query", response_model=RAGQueryResponse)
async def query_knowledge_base(request: RAGQueryRequest) -> RAGQueryResponse:
    """Query the RAG knowledge base directly."""
    try:
        logger.info(
            "RAG query received",
            query=request.query,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
        )

        # Retrieve documents
        documents = await rag_system.retrieve_relevant_documents(
            query=request.query,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            filters=request.filters,
        )

        return RAGQueryResponse(
            success=True,
            query=request.query,
            results=documents,
            total_results=len(documents),
            retrieval_time_ms=0,  # Could add timing here
        )

    except Exception as e:
        logger.error("RAG query failed", query=request.query, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query knowledge base: {str(e)}",
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    """Get session information and conversation history."""
    try:
        logger.info("Session info requested", session_id=session_id)

        # Note: In a real implementation, you'd store session data in a database
        # For now, return a placeholder response
        return SessionResponse(
            success=True,
            session_id=session_id,
            conversation_history=[],
            session_metadata={
                "created_at": 0,
                "last_activity": 0,
                "total_messages": 0,
                "strategies_generated": 0,
            },
        )

    except Exception as e:
        logger.error("Failed to get session", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session: {str(e)}",
        )


@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_statistics() -> RAGStatsResponse:
    """Get RAG system statistics."""
    try:
        logger.info("RAG statistics requested")

        stats = await rag_system.get_statistics()

        return RAGStatsResponse(success=True, **stats)

    except Exception as e:
        logger.error("Failed to get RAG statistics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}",
        )
