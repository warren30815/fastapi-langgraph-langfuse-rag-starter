import os
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.config.logging_config import get_logger
from app.config.settings import settings


class FAISSVectorStore:
    """FAISS-based vector store for document embeddings."""

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.logger = get_logger("vector_store")

        # Initialize OpenAI embedding model
        self.embedding_model = OpenAIEmbeddings(
            model=embedding_model, api_key=settings.openai_api_key
        )
        # text-embedding-3-small has 1536 dimensions
        self.embedding_dim = 1536

        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []

        # File paths
        self.vector_db_path = settings.vector_db_path
        self.index_file = os.path.join(self.vector_db_path, "faiss_index.idx")
        self.metadata_file = os.path.join(self.vector_db_path, "metadata.pkl")

        # Ensure directory exists
        os.makedirs(self.vector_db_path, exist_ok=True)

        # Load existing index if available
        self._load_index()

    def _load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)

                # Load metadata
                with open(self.metadata_file, "rb") as f:
                    self.documents = pickle.load(f)

                self.logger.info(
                    "Loaded existing vector store",
                    documents_count=len(self.documents),
                    embedding_dim=self.embedding_dim,
                )
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(
                    self.embedding_dim
                )  # Inner product for cosine similarity
                self.documents = []
                self.logger.info(
                    "Created new vector store", embedding_dim=self.embedding_dim
                )

        except Exception as e:
            self.logger.error("Failed to load vector store", error=str(e))
            # Create new index on failure
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []

    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)

            # Save metadata
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.documents, f)

            self.logger.info("Vector store saved successfully")

        except Exception as e:
            self.logger.error("Failed to save vector store", error=str(e))
            raise

    async def add_documents(
        self, texts: List[str], metadatas: List[Dict[str, Any]], batch_size: int = 32
    ) -> None:
        """Add documents to the vector store."""
        try:
            # Generate embeddings using LangChain OpenAI
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Generate embeddings
                batch_embeddings = await self.embedding_model.aembed_documents(
                    batch_texts
                )

                # Convert to numpy array and normalize for cosine similarity
                batch_embeddings = np.array(batch_embeddings)
                batch_embeddings = batch_embeddings / np.linalg.norm(
                    batch_embeddings, axis=1, keepdims=True
                )
                all_embeddings.append(batch_embeddings)

            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings).astype("float32")

            # Add to FAISS index
            self.index.add(embeddings)

            # Add metadata
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                doc_metadata = {
                    "id": len(self.documents),
                    "text": text,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **metadata,
                }
                self.documents.append(doc_metadata)

            # Save to disk
            self._save_index()

            self.logger.info(
                "Documents added to vector store",
                count=len(texts),
                total_documents=len(self.documents),
            )

        except Exception as e:
            self.logger.error("Failed to add documents", error=str(e))
            raise

    async def similarity_search(
        self, query: str, k: int, similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            if len(self.documents) == 0:
                self.logger.warning("No documents in vector store")
                return []

            # Generate query embedding using LangChain OpenAI
            query_embedding_list = await self.embedding_model.aembed_query(query)

            # Convert to numpy array and normalize
            query_embedding = np.array([query_embedding_list])
            query_embedding = query_embedding / np.linalg.norm(
                query_embedding, axis=1, keepdims=True
            )
            query_embedding = query_embedding.astype("float32")

            # Search in FAISS index
            scores, indices = self.index.search(
                query_embedding, min(k, len(self.documents))
            )

            # Filter by similarity threshold and prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score >= similarity_threshold:
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(score)
                    results.append(doc)

            self.logger.info(
                "Similarity search completed",
                query=query,
                results_count=len(results),
                similarity_threshold=similarity_threshold,
            )

            return results

        except Exception as e:
            self.logger.error("Similarity search failed", error=str(e))
            raise

    def get_document_count(self) -> int:
        """Get total number of documents in the store."""
        return len(self.documents)

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]
        return None


# Global vector store instance
vector_store: FAISSVectorStore = FAISSVectorStore()
