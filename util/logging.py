"""
Structured logging configuration for TRACE.

Uses structlog for structured, context-aware logging with console and JSON output formats.
"""

from typing import Any

import structlog


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """Get a structured logger with optional initial context."""
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
