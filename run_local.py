#!/usr/bin/env python3
"""
Local development server script for Email Marketing Agent.

This script starts the FastAPI application locally for development.
For production deployment, use the Docker Compose setup.
"""

import uvicorn
from app.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )