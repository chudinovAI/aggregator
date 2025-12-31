"""
Core aggregator package exposing parser utilities and exceptions.
"""

from .exceptions import FetchError, ParseError, ParserError, ValidationError

__all__ = ["FetchError", "ParseError", "ParserError", "ValidationError"]
