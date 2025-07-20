import time
from typing import Any, Dict, List, Optional

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.utils.document_processor import document_processor
from app.utils.vector_store import vector_store


class RAGSystem:
    """Retrieval-Augmented Generation system for email marketing knowledge."""

    def __init__(self):
        self.logger = get_logger("rag_system")
        self.vector_store = vector_store

    async def add_documents_from_files(
        self, files_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process and add documents from uploaded files."""
        try:
            all_documents = []
            processing_results = []

            for file_data in files_data:
                filename = file_data["filename"]
                content = file_data["content"]

                try:
                    # Process file into chunks
                    documents = await document_processor.process_file(content, filename)
                    all_documents.extend(documents)

                    processing_results.append(
                        {
                            "filename": filename,
                            "status": "success",
                            "chunks_created": len(documents),
                            "total_tokens": sum(
                                doc["token_count"] for doc in documents
                            ),
                        }
                    )

                except Exception as e:
                    self.logger.error(
                        "Failed to process file", filename=filename, error=str(e)
                    )
                    processing_results.append(
                        {"filename": filename, "status": "error", "error": str(e)}
                    )

            # Add all documents to vector store
            if all_documents:
                texts = [doc["text"] for doc in all_documents]
                metadatas = [
                    {k: v for k, v in doc.items() if k != "text"}
                    for doc in all_documents
                ]

                await self.vector_store.add_documents(texts, metadatas)

            result = {
                "total_files": len(files_data),
                "successful_files": len(
                    [r for r in processing_results if r["status"] == "success"]
                ),
                "total_chunks": len(all_documents),
                "total_tokens": sum(doc["token_count"] for doc in all_documents),
                "processing_results": processing_results,
            }

            self.logger.info(
                "Documents processing completed",
                **{k: v for k, v in result.items() if k != "processing_results"},
            )

            return result

        except Exception as e:
            self.logger.error("Failed to add documents from files", error=str(e))
            raise

    async def retrieve_relevant_documents(
        self,
        query: str,
        k: int = None,
        similarity_threshold: float = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        k = k or settings.max_retrieval_results
        similarity_threshold = similarity_threshold or settings.similarity_threshold

        start_time = time.time()

        try:
            # Perform similarity search
            results = await self.vector_store.similarity_search(
                query=query, k=k, similarity_threshold=similarity_threshold
            )

            # Apply additional filters if provided
            if filters:
                filtered_results = []
                for result in results:
                    match = True
                    for filter_key, filter_value in filters.items():
                        if result.get(filter_key) != filter_value:
                            match = False
                            break
                    if match:
                        filtered_results.append(result)
                results = filtered_results

            retrieval_time = int((time.time() - start_time) * 1000)

            self.logger.info(
                "Document retrieval completed",
                query=query,
                results_count=len(results),
                retrieval_time_ms=retrieval_time,
                avg_similarity=sum(r["similarity_score"] for r in results)
                / len(results)
                if results
                else 0,
            )

            return results

        except Exception as e:
            self.logger.error("Document retrieval failed", query=query, error=str(e))
            raise

    async def get_context_for_query(
        self,
        query: str,
        max_context_tokens: int = 4000,
        k: int = None,
        similarity_threshold: float = None,
    ) -> Dict[str, Any]:
        """Get formatted context for a query with token management."""
        try:
            # Retrieve relevant documents
            documents = await self.retrieve_relevant_documents(
                query=query, k=k, similarity_threshold=similarity_threshold
            )

            if not documents:
                return {
                    "context": "No relevant documents found.",
                    "sources": [],
                    "total_tokens": 0,
                    "documents_used": 0,
                }

            # Build context within token limit
            context_parts = []
            sources = []
            total_tokens = 0
            documents_used = 0

            for doc in documents:
                doc_tokens = doc.get("token_count", 0)

                # Check if adding this document would exceed token limit
                if total_tokens + doc_tokens <= max_context_tokens:
                    context_parts.append(f"Source: {doc.get('source', 'Unknown')}")
                    context_parts.append(f"Content: {doc['text']}")
                    context_parts.append("---")

                    sources.append(
                        {
                            "source": doc.get("source", "Unknown"),
                            "chunk_index": doc.get("chunk_index", 0),
                            "similarity_score": doc["similarity_score"],
                        }
                    )

                    total_tokens += doc_tokens
                    documents_used += 1
                else:
                    break

            context = "\n".join(context_parts)

            result = {
                "context": context,
                "sources": sources,
                "total_tokens": total_tokens,
                "documents_used": documents_used,
                "total_available": len(documents),
            }

            self.logger.info(
                "Context prepared for query",
                query=query,
                documents_used=documents_used,
                total_tokens=total_tokens,
                max_context_tokens=max_context_tokens,
            )

            return result

        except Exception as e:
            self.logger.error(
                "Failed to get context for query", query=query, error=str(e)
            )
            raise

    async def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        try:
            document_count = self.vector_store.get_document_count()

            # Analyze document types and sources
            source_stats = {}
            type_stats = {}

            for i in range(document_count):
                doc = self.vector_store.get_document_by_id(i)
                if doc:
                    source = doc.get("source", "Unknown")
                    file_type = doc.get("file_type", "Unknown")

                    source_stats[source] = source_stats.get(source, 0) + 1
                    type_stats[file_type] = type_stats.get(file_type, 0) + 1

            return {
                "total_documents": document_count,
                "sources": source_stats,
                "file_types": type_stats,
                "embedding_dimension": self.vector_store.embedding_dim,
                "chunk_size": settings.chunk_size,
                "similarity_threshold": settings.similarity_threshold,
            }

        except Exception as e:
            self.logger.error("Failed to get RAG statistics", error=str(e))
            raise


# Global RAG system instance
rag_system: RAGSystem = RAGSystem()
