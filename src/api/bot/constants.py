"""
Bot constants and configuration values.
"""

from enum import Enum

# Available sources for selection (must match existing parsers)
AVAILABLE_SOURCES = [
    "reddit",
    "hackernews",
]

# Default topics suggestions
SUGGESTED_TOPICS = [
    "python",
    "machine learning",
    "web development",
    "devops",
    "security",
    "data science",
    "cloud",
    "kubernetes",
    "rust",
    "golang",
    "databases",
    "linux",
    "javascript",
    "typescript",
    "react",
    "ai",
    "llm",
    "open source",
    "career",
    "startups",
]

# Search period options (in days)
PERIOD_OPTIONS = {
    "1d": 1,
    "3d": 3,
    "7d": 7,
    "14d": 14,
    "30d": 30,
}


class SearchPeriod(str, Enum):
    """User-selectable search periods."""

    DAY_1 = "1d"
    DAY_3 = "3d"
    DAY_7 = "7d"
    DAY_14 = "14d"
    DAY_30 = "30d"


__all__ = [
    "AVAILABLE_SOURCES",
    "SUGGESTED_TOPICS",
    "PERIOD_OPTIONS",
    "SearchPeriod",
]
