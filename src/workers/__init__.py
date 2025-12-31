"""
Background worker system for the news aggregator.

This package provides:
- Async task definitions for parsing, training, and cleanup
- APScheduler-based job scheduling
- Integration with the main application

Modules:
- types: Task result dataclasses (ParseResult, TaskResult)
- parsers: Parser factory and source registry
- parsing: Source parsing task
- training: Classifier retraining task
- cleanup: Data cleanup tasks
- notifications: User notification tasks
- scheduler: APScheduler-based job scheduling
"""

from .cleanup import cleanup_old_posts
from .notifications import send_daily_digest

# Parser factory
from .parsers import REGISTERED_SOURCES, create_parser

# Tasks
from .parsing import parse_all_sources
from .scheduler import create_scheduler, run_scheduler
from .scheduler import main as run_scheduler_main

# Task registry (for backwards compatibility with tasks.py)
from .tasks import TASK_REGISTRY
from .training import retrain_classifier

# Types
from .types import ParseResult, TaskResult

__all__ = [
    # Scheduler
    "create_scheduler",
    "run_scheduler",
    "run_scheduler_main",
    # Types
    "ParseResult",
    "TaskResult",
    # Parser factory
    "REGISTERED_SOURCES",
    "create_parser",
    # Tasks
    "parse_all_sources",
    "retrain_classifier",
    "cleanup_old_posts",
    "send_daily_digest",
    # Registry
    "TASK_REGISTRY",
]
