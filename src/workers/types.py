"""
Task result types for background workers.

This module defines dataclasses for task execution results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ParseResult:
    """Result of a parse_all_sources task run."""

    source: str
    posts_fetched: int
    posts_saved: int
    errors: list[str]
    duration_seconds: float


@dataclass
class TaskResult:
    """Generic task execution result."""

    task_name: str
    success: bool
    message: str
    details: dict[str, Any]
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


__all__ = [
    "ParseResult",
    "TaskResult",
]
