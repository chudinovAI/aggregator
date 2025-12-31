"""
Telegram bot handlers module.

This module provides a modular structure for the Telegram bot:
- constants.py: Configuration values and enums
- states.py: FSM states for conversation flows
- keyboards.py: Inline keyboard builders
- utils.py: Helper functions
- commands.py: Command handlers (/start, /help, etc.)
- callbacks.py: Callback query handlers
"""

from aiogram import Router

from .callbacks import router as callbacks_router
from .commands import router as commands_router
from .constants import (
    AVAILABLE_SOURCES,
    PERIOD_OPTIONS,
    SUGGESTED_TOPICS,
    SearchPeriod,
)
from .keyboards import (
    build_back_keyboard,
    build_main_menu_keyboard,
    build_period_keyboard,
    build_posts_keyboard,
    build_sources_keyboard,
    build_topics_keyboard,
)
from .states import PeriodStates, SourceStates, TopicStates
from .utils import (
    escape_markdown,
    format_age,
    format_post_message,
    get_top_posts,
    get_user_data,
)

# Create main router and include sub-routers
router = Router(name="bot")
router.include_router(commands_router)
router.include_router(callbacks_router)

__all__ = [
    # Main router
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
