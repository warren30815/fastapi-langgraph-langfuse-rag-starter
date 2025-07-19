from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config.logging_config import get_logger
from app.core.rag import rag_system
from app.models.requests import DocumentUploadRequest
from app.models.responses import DocumentUploadResponse
from app.utils.document_processor import document_processor

router = APIRouter()
logger = get_logger("document_routes")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
) -> DocumentUploadResponse:
    """Upload and process marketing documents for the RAG system."""
    try:
        logger.info(
            "Document upload started",
            file_count=len(files),
            description=description,
            category=category,
        )

        # Validate file types
        supported_extensions = document_processor.get_supported_extensions()
        files_data = []

        for file in files:
            # Check file extension
            file_extension = None
            if file.filename:
                file_extension = "." + file.filename.split(".")[-1].lower()

            if not file_extension or file_extension not in supported_extensions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type. Supported types: {', '.join(supported_extensions)}",
                )

            # Read file content
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is empty",
                )

            # Validate file size (max 10MB per file)
            max_size = 10 * 1024 * 1024  # 10MB
            if len(content) > max_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File {file.filename} exceeds maximum size of 10MB",
                )

            files_data.append(
                {
                    "filename": file.filename,
                    "content": content,
                    "description": description,
                    "category": category,
                }
            )

        # Process files with RAG system
        result = await rag_system.add_documents_from_files(files_data)

        return DocumentUploadResponse(
            success=True,
            message="Documents uploaded and processed successfully",
            **result,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("Document upload failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload documents: {str(e)}",
        )


@router.get("/supported-types")
async def get_supported_file_types() -> dict:
    """Get list of supported file types for document upload."""
    try:
        supported_extensions = document_processor.get_supported_extensions()

        return {
            "supported_extensions": supported_extensions,
            "description": "List of file extensions supported for document upload",
            "max_file_size_mb": 10,
            "max_files_per_upload": 20,
        }

    except Exception as e:
        logger.error("Failed to get supported file types", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supported file types",
        )
