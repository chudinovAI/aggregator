"""
Repository classes for database access.

This module provides specialized repositories for different entity types:
- PostRepository: Data access for Post entities
- UserRepository: Data access for User entities
"""

from .base import BaseRepository
from .post import PostRepository
from .user import UserRepository

__all__ = ["BaseRepository", "PostRepository", "UserRepository"]
