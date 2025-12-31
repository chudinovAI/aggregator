"""
Telegram bot command handlers with inline keyboard navigation.

This module re-exports the router and utilities from the modular
`src/api/bot/` package for backward compatibility.
"""

from __future__ import annotations

# Re-export everything from the bot package for backward compatibility
from .bot import (
    # Constants
    AVAILABLE_SOURCES,
    PERIOD_OPTIONS,
    SUGGESTED_TOPICS,
    # States
    PeriodStates,
    SearchPeriod,
    SourceStates,
    TopicStates,
    # Keyboards
    build_back_keyboard,
    build_main_menu_keyboard,
    build_period_keyboard,
    build_posts_keyboard,
    build_sources_keyboard,
    build_topics_keyboard,
    # Utils
    escape_markdown,
    format_age,
    format_post_message,
    get_top_posts,
    get_user_data,
    # Main router
    router,
)

__all__ = [
    "router",
    # Constants
    "AVAILABLE_SOURCES",
    "SUGGESTED_TOPICS",
    "PERIOD_OPTIONS",
    "SearchPeriod",
    # States
    "TopicStates",
    "SourceStates",
    "PeriodStates",
    # Keyboards
    "build_main_menu_keyboard",
    "build_topics_keyboard",
    "build_sources_keyboard",
    "build_period_keyboard",
    "build_posts_keyboard",
    "build_back_keyboard",
    # Utils
    "get_user_data",
    "get_top_posts",
    "format_post_message",
    "format_age",
    "escape_markdown",
]
