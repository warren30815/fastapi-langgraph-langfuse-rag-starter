#!/usr/bin/env python3
"""
Seed data script for Email Marketing Agent RAG system.

This script loads example documents from the data/documents/examples folder
into the RAG system for development and demo purposes.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List
import httpx
import time

from app.config.logging_config import get_logger


def check_existing_vector_db() -> bool:
    """Check if vector database files already exist and are non-empty."""
    vector_db_path = Path("data/vector_db")
    faiss_index_path = vector_db_path / "faiss_index.idx"
    metadata_path = vector_db_path / "metadata.pkl"

    # Check if both files exist and are non-empty
    faiss_exists = faiss_index_path.exists() and faiss_index_path.stat().st_size > 0
    metadata_exists = metadata_path.exists() and metadata_path.stat().st_size > 0

    return faiss_exists or metadata_exists


async def check_server_health(base_url: str = "http://localhost:8000") -> bool:
    """Check if the FastAPI server is running and healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/v1/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


async def wait_for_server(base_url: str = "http://localhost:8000", max_attempts: int = 30) -> bool:
    """Wait for the server to become available."""
    logger = get_logger("seed_data")

    for attempt in range(max_attempts):
        if await check_server_health(base_url):
            logger.info(f"Server is ready at {base_url}")
            return True

        if attempt == 0:
            print(f"🔄 Waiting for server at {base_url}...")

        time.sleep(1)

    return False


async def load_documents_from_folder(folder_path: str) -> List[Dict[str, any]]:
    """Load all documents from a folder into the format expected by RAG system."""
    folder = Path(folder_path)
    documents = []
    logger = get_logger("seed_data")

    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return documents

    # Supported file extensions
    supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.csv'}

    for file_path in folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                logger.info(f"Loading file: {file_path.name}")

                # Read file content as bytes
                with open(file_path, 'rb') as file:
                    content = file.read()

                documents.append({
                    'filename': file_path.name,
                    'content': content
                })

            except Exception as e:
                logger.error(f"Failed to load file {file_path}: {str(e)}")

    return documents


async def upload_documents_via_api(
    documents: List[Dict[str, any]],
    base_url: str = "http://localhost:8000"
) -> Dict[str, any]:
    """Upload documents using the FastAPI upload endpoint."""
    logger = get_logger("seed_data")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Prepare files for multipart upload
            files = []
            for doc in documents:
                files.append(
                    ("files", (doc["filename"], doc["content"], "application/octet-stream"))
                )

            # Prepare form data
            data = {
                "description": "Example marketing documents for RAG system",
                "category": "examples"
            }

            # Upload files
            logger.info(f"Uploading {len(documents)} documents to {base_url}/api/v1/documents/upload")
            response = await client.post(
                f"{base_url}/api/v1/documents/upload",
                files=files,
                data=data
            )

            if response.status_code == 200:
                result = response.json()
                logger.info("Documents uploaded successfully via API")
                return result
            else:
                error_detail = response.text
                logger.error(f"API upload failed with status {response.status_code}: {error_detail}")
                raise Exception(f"API upload failed: {error_detail}")

    except Exception as e:
        logger.error(f"Failed to upload documents via API: {str(e)}")
        raise


async def seed_rag_system(force_seed: bool = False):
    """Seed the RAG system with example documents via HTTP API."""
    logger = get_logger("seed_data")
    base_url = "http://localhost:8000"

    try:
        # Check if vector database already exists
        if check_existing_vector_db():
            if force_seed:
                print("\n⚠️  Existing RAG System Detected! (Force mode: Overwriting existing data)")
                # Remove entire vector DB directory to ensure full overwrite
                vector_db_path = Path("data/vector_db")
                if vector_db_path.exists():
                    import shutil
                    shutil.rmtree(vector_db_path)
                # Ensure the directory is recreated before upload
                vector_db_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.warning("Existing vector database found - skipping seeding to prevent overwrite")
                print("\n⚠️  Existing RAG System Detected!")
                print("Vector database files (faiss_index.idx or metadata.pkl) already exist.")
                print("Skipping seeding to prevent accidental overwrite.")
                print("\nTo force re-seeding:")
                print("1. Delete data/vector_db/ folder")
                print("2. Run this script again")
                print("\nOr use --force flag: python seed_data.py --force")
                return False

        logger.info("Starting RAG system seeding process...")

        # Check if server is running
        if not await check_server_health(base_url):
            print(f"\n🚨 FastAPI server not found at {base_url}")
            print("Please start the server first:")
            print("  Option 1: uv run python run_local.py")
            print("  Option 2: docker-compose up -d")
            print("\nThen run this script again.")
            return False

        # Define the path to example documents
        examples_path = "data/documents/examples"

        # Load documents from the examples folder
        documents = await load_documents_from_folder(examples_path)

        if not documents:
            logger.warning("No documents found to seed the RAG system")
            return False

        logger.info(f"Loaded {len(documents)} documents for seeding")

        # Upload documents via API
        result = await upload_documents_via_api(documents, base_url)

        # Print results for user
        print("\n🌱 RAG System Seeding Complete!")
        print(f"📊 Files processed: {result.get('total_files', 0)}")
        print(f"✅ Successful files: {result.get('successful_files', 0)}")
        print(f"📄 Chunks created: {result.get('total_chunks', 0)}")
        print(f"🔢 Total tokens: {result.get('total_tokens', 0)}")

        if result.get('processing_results'):
            print("\n📋 Processing Details:")
            for file_result in result['processing_results']:
                status_emoji = "✅" if file_result.get('status') == 'success' else "❌"
                print(f"  {status_emoji} {file_result.get('filename', 'Unknown')}")
                if file_result.get('status') == 'success':
                    print(f"    └── Chunks: {file_result.get('chunks_created', 0)}, Tokens: {file_result.get('total_tokens', 0)}")
                else:
                    print(f"    └── Error: {file_result.get('error', 'Unknown error')}")

        print(f"\n📈 RAG System Updated Successfully!")
        print(f"  API Response: {result.get('message', 'Documents processed')}")

        return True

    except Exception as e:
        logger.error(f"Failed to seed RAG system: {str(e)}")
        print(f"\n❌ Seeding failed: {str(e)}")
        raise


async def test_rag_query(base_url: str = "http://localhost:8000"):
    """Test the RAG system with a sample query via HTTP API."""
    logger = get_logger("seed_data")

    try:
        logger.info("Testing RAG system with sample query...")

        test_query = "What are the best practices for email segmentation?"

        # Prepare query request
        query_data = {
            "query": test_query,
            "k": 3,
            "similarity_threshold": 0.7
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/api/v1/agent/query",
                json=query_data
            )

            if response.status_code == 200:
                result = response.json()
                results = result.get('results', [])

                print(f"\n🔍 Test Query: '{test_query}'")
                print(f"📊 Found {len(results)} relevant documents:")

                for i, doc in enumerate(results, 1):
                    print(f"  {i}. Source: {doc.get('source', 'Unknown')}")
                    print(f"     Similarity: {doc.get('similarity_score', 0):.3f}")
                    print(f"     Preview: {doc.get('text', '')}...")
                    print()
            else:
                error_detail = response.text
                logger.error(f"Query API failed with status {response.status_code}: {error_detail}")
                print(f"\n❌ Query test failed: {error_detail}")

    except Exception as e:
        logger.error(f"Failed to test RAG query: {str(e)}")
        print(f"\n❌ RAG test failed: {str(e)}")


if __name__ == "__main__":
    import sys

    # Check for --force flag
    force_seed = "--force" in sys.argv
    base_url = "http://localhost:8000"

    print("🚀 Email Marketing Agent - RAG System Seeding")
    print("=" * 50)
    print(f"🌐 Using API endpoint: {base_url}")

    # Run seeding (pass force_seed to ensure correct overwrite behavior)
    seeded = asyncio.run(seed_rag_system(force_seed=force_seed))

    # Only test if seeding was successful or if there's existing data
    if seeded or (not seeded and not force_seed):
        print("\n" + "=" * 50)
        if asyncio.run(check_server_health(base_url)):
            asyncio.run(test_rag_query(base_url))
            if seeded:
                print("\n✨ Seeding process complete! Your RAG system is ready to use.")
            else:
                print("\n✨ RAG system test completed with existing data.")
            print(f"💡 You can now use {base_url}/api/v1/agent/chat to interact with the agent.")
        else:
            print(f"\n⚠️  Server not available at {base_url} - skipping API test")
            print("Start the server to test the API endpoints:")
            print("  uv run python run_local.py")
    else:
        print("\n💡 To test the existing RAG system, start the server and run this script again.")
