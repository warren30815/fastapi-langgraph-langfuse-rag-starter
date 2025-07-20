import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict

import structlog

# Context variables for request tracking
http_request_id_var: ContextVar[str] = ContextVar("http_request_id", default="")
langfuse_request_id_var: ContextVar[str] = ContextVar("langfuse_request_id", default="")


def opentelemetry_formatter(logger, method_name, event_dict):
    """Format log events according to OpenTelemetry standards."""
    # Get context variables
    http_request_id = http_request_id_var.get("")
    langfuse_request_id = langfuse_request_id_var.get("")

    # Create OpenTelemetry formatted log record
    otel_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "trace_id": http_request_id or None,
        "span_id": None,  # Can be extended in the future
        "severity_number": _get_severity_number(event_dict.get("level", "info")),
        "severity_text": event_dict.get("level", "info").upper(),
        "body": event_dict.get("event", ""),
        "resource": {
            "service.name": os.getenv("SERVICE_NAME", "fastapi-langgraph-rag"),
            "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
        },
        "attributes": {},
    }

    # Add logger name as attribute
    if "logger" in event_dict:
        otel_record["attributes"]["logger"] = event_dict["logger"]

    # Add langfuse request ID as attribute if available
    if langfuse_request_id:
        otel_record["attributes"]["langfuse_request_id"] = langfuse_request_id

    # Add all other fields as attributes
    for key, value in event_dict.items():
        if key not in ["event", "level", "logger", "timestamp"]:
            otel_record["attributes"][key] = value

    return otel_record


def _get_severity_number(level_name: str) -> int:
    """Convert log level name to OpenTelemetry severity number."""
    level_mapping = {"debug": 5, "info": 9, "warning": 13, "error": 17, "critical": 21}
    return level_mapping.get(level_name.lower(), 9)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured logging with structlog and trace context."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            opentelemetry_formatter,  # Format as OpenTelemetry standard
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def get_logger(name: str):
    """Get a structured logger instance with trace context."""
    return structlog.get_logger(name)


def set_http_request_context(http_request_id: str):
    """Set HTTP request context for current async context."""
    http_request_id_var.set(http_request_id)


def set_langfuse_request_context(langfuse_request_id: str):
    """Set Langfuse request context for current async context."""
    langfuse_request_id_var.set(langfuse_request_id)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4().hex)


class LoggingMiddleware:
    """Middleware for logging HTTP requests and responses with trace context."""

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("http")

    async def __call__(self, scope: Dict[str, Any], receive, send):
        if scope["type"] == "http":
            # Generate HTTP request ID and set context
            http_request_id = generate_request_id()
            set_http_request_context(http_request_id)

            method = scope["method"]
            path = scope["path"]
            query_string = scope.get("query_string", b"").decode()

            # Log request start
            self.logger.info(
                "Request started",
                method=method,
                path=path,
                query_string=query_string,
                user_agent=next(
                    (
                        h[1].decode()
                        for h in scope.get("headers", [])
                        if h[0] == b"user-agent"
                    ),
                    None,
                ),
            )

            async def send_with_logging(message):
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    self.logger.info(
                        "Request completed",
                        method=method,
                        path=path,
                        status_code=status_code,
                        status_category="success"
                        if 200 <= status_code < 400
                        else "error",
                    )
                await send(message)

            await self.app(scope, receive, send_with_logging)
        else:
            await self.app(scope, receive, send)
