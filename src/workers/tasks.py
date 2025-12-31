"""
Background tasks for the news aggregator.

This module re-exports all task functions for backwards compatibility.
The actual implementations have been moved to separate modules:
- types.py: ParseResult, TaskResult dataclasses
- parsers.py: Parser factory and REGISTERED_SOURCES
- parsing.py: parse_all_sources task
- training.py: retrain_classifier task
- cleanup.py: cleanup_old_posts task
- notifications.py: send_daily_digest task
"""

from __future__ import annotations

from .cleanup import cleanup_old_posts
from .notifications import send_daily_digest

# Re-export parser factory and registry
from .parsers import REGISTERED_SOURCES, create_parser

# Re-export tasks
from .parsing import parse_all_sources
from .training import retrain_classifier

# Re-export types
from .types import ParseResult, TaskResult

# Task Registry
TASK_REGISTRY = {
    "parse_all_sources": parse_all_sources,
    "retrain_classifier": retrain_classifier,
    "cleanup_old_posts": cleanup_old_posts,
    "send_daily_digest": send_daily_digest,
}


__all__ = [
    "ParseResult",
    "TaskResult",
    "REGISTERED_SOURCES",
    "TASK_REGISTRY",
    "cleanup_old_posts",
    "create_parser",
    "parse_all_sources",
    "retrain_classifier",
    "send_daily_digest",
]
