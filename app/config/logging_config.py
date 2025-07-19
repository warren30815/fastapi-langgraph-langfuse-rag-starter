import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict

import structlog

# Context variables for request tracking
http_request_id_var: ContextVar[str] = ContextVar("http_request_id", default="")
langfuse_trace_id_var: ContextVar[str] = ContextVar("langfuse_trace_id", default="")


def add_trace_context(logger, method_name, event_dict):
    """Add trace context to log events."""
    http_request_id = http_request_id_var.get("")
    langfuse_trace_id = langfuse_trace_id_var.get("")

    # Include HTTP request ID for correlation
    if http_request_id:
        event_dict["http_request_id"] = http_request_id

    # Include Langfuse trace ID when available
    if langfuse_trace_id:
        event_dict["langfuse_trace_id"] = langfuse_trace_id

    return event_dict


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
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            add_trace_context,  # Add trace context to all logs
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance with trace context."""
    return structlog.get_logger(name)


def set_http_request_context(http_request_id: str):
    """Set HTTP request context for current async context."""
    http_request_id_var.set(http_request_id)


def set_langfuse_context(langfuse_trace_id: str):
    """Set Langfuse trace context without overwriting HTTP request context."""
    langfuse_trace_id_var.set(langfuse_trace_id)


def set_request_context(request_id: str, span_id: str = None):
    """Legacy function for backwards compatibility - sets HTTP request context."""
    # This function remains only for any existing code that might still use it
    set_http_request_context(request_id)


def get_trace_context() -> Dict[str, str]:
    """Get current trace context for correlation."""
    return {
        "http_request_id": http_request_id_var.get(""),
        "langfuse_trace_id": langfuse_trace_id_var.get(""),
    }


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
