"""
Custom exception hierarchy for parser components.
"""

from __future__ import annotations

from typing import Any


class ParserError(Exception):
    """Base exception for all parser-related errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class FetchError(ParserError):
    """Raised when remote content cannot be fetched."""


class ParseError(ParserError):
    """Raised when the fetched payload cannot be parsed."""


class ValidationError(ParserError):
    """Raised when a parsed post fails validation."""


__all__ = ["ParserError", "FetchError", "ParseError", "ValidationError"]
