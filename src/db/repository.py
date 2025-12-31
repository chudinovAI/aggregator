"""
Compatibility layer for repository imports.

This module re-exports repositories from the new modular structure
to maintain backward compatibility with existing imports.
"""

from .repositories import PostRepository, UserRepository

__all__ = ["PostRepository", "UserRepository"]
