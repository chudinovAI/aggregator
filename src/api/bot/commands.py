"""
Command handlers for the Telegram bot.
"""

from __future__ import annotations

from typing import Any

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import Post
from ...db.repositories.post import PostRepository
from .constants import PERIOD_OPTIONS
from .keyboards import (
    build_back_keyboard,
    build_main_menu_keyboard,
    build_period_keyboard,
    build_sources_keyboard,
    build_topics_keyboard,
)
from .utils import (
    LOGGER,
    escape_markdown,
    format_age,
    get_user_data,
)

router = Router(name="commands")


@router.message(CommandStart())
async def cmd_start(message: Message, session: AsyncSession) -> None:
    """Handle /start command - show welcome message and menu."""
    if not message.from_user:
        return

    telegram_id = message.from_user.id
    user_data = await get_user_data(session, telegram_id)

    welcome_text = (
        f"Welcome to *News Aggregator Bot*\\!\n\n"
        f"I help you discover the most interesting tech news "
        f"from various sources, powered by ML classification\\.\n\n"
        f"*Your Settings:*\n"
        f"• Topics: {len(user_data['topics'])} configured\n"
        f"• Sources: {len(user_data['sources'])} active\n"
        f"• Period: Last {PERIOD_OPTIONS.get(user_data['period'], 7)} days\n\n"
        f"Use the menu below to get started\\!"
    )

    await message.answer(
        welcome_text,
        reply_markup=build_main_menu_keyboard(),
        parse_mode="MarkdownV2",
    )

    LOGGER.info("User %d started the bot", telegram_id)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command - show available commands."""
    help_text = (
        "*Available Commands*\n\n"
        "/start \\- Show main menu\n"
        "/top \\- Show top 10 interesting posts\n"
        "/topics \\- Manage your interest topics\n"
        "/sources \\- Choose data sources\n"
        "/period \\- Set search time period\n"
        "/settings \\- View your current settings\n"
        "/help \\- Show this help message\n\n"
        "*How it works:*\n"
        "1\\. Set your interested topics\n"
        "2\\. Choose your preferred sources\n"
        "3\\. Get personalized top posts\\!"
    )

    await message.answer(
        help_text,
        reply_markup=build_back_keyboard(),
        parse_mode="MarkdownV2",
    )


@router.message(Command("topics"))
async def cmd_topics(message: Message, session: AsyncSession) -> None:
    """Handle /topics command - manage interest topics."""
    if not message.from_user:
        return

    user_data = await get_user_data(session, message.from_user.id)
    topics = user_data["topics"]

    if topics:
        text = f"*Your Topics* \\({len(topics)}\\)\n\nTap to remove or add new ones:"
    else:
        text = (
            "*Your Topics*\n\n"
            "You haven't set any topics yet\\.\n"
            "Add topics to get personalized recommendations\\!"
        )

    await message.answer(
        text,
        reply_markup=build_topics_keyboard(topics),
        parse_mode="MarkdownV2",
    )


@router.message(Command("sources"))
async def cmd_sources(message: Message, session: AsyncSession) -> None:
    """Handle /sources command - manage data sources."""
    if not message.from_user:
        return

    user_data = await get_user_data(session, message.from_user.id)
    sources = user_data["sources"]

    text = (
        f"*Data Sources* \\({len(sources)} active\\)\n\n"
        f"Toggle sources to include/exclude from your feed:"
    )

    await message.answer(
        text,
        reply_markup=build_sources_keyboard(sources),
        parse_mode="MarkdownV2",
    )


@router.message(Command("period"))
async def cmd_period(message: Message, session: AsyncSession) -> None:
    """Handle /period command - set search period."""
    if not message.from_user:
        return

    user_data = await get_user_data(session, message.from_user.id)
    current_period = user_data.get("period", "7d")

    text = "*Search Period*\n\nChoose how far back to search for posts:"

    await message.answer(
        text,
        reply_markup=build_period_keyboard(current_period),
        parse_mode="MarkdownV2",
    )


@router.message(Command("top"))
async def cmd_top(message: Message, session: AsyncSession) -> None:
    """Handle /top command - show top 10 posts."""
    if not message.from_user:
        return

    await show_top_posts(message, session, message.from_user.id)


@router.message(Command("settings"))
async def cmd_settings(message: Message, session: AsyncSession) -> None:
    """Handle /settings command - show current settings."""
    if not message.from_user:
        return

    user_data = await get_user_data(session, message.from_user.id)

    topics_str = ", ".join(user_data["topics"][:5]) if user_data["topics"] else "None"
    if len(user_data["topics"]) > 5:
        topics_str += f" +{len(user_data['topics']) - 5} more"

    sources_str = ", ".join(s.replace("_", " ").title() for s in user_data["sources"][:4])
    if len(user_data["sources"]) > 4:
        sources_str += f" +{len(user_data['sources']) - 4} more"

    period_days = PERIOD_OPTIONS.get(user_data.get("period", "7d"), 7)

    text = (
        "*Your Settings*\n\n"
        f"*Topics:* {escape_markdown(topics_str)}\n\n"
        f"*Sources:* {escape_markdown(sources_str)}\n\n"
        f"*Period:* Last {period_days} days\n\n"
        f"*User ID:* `{message.from_user.id}`"
    )

    await message.answer(
        text,
        reply_markup=build_main_menu_keyboard(),
        parse_mode="MarkdownV2",
    )


async def show_top_posts(
    message: Any,
    session: AsyncSession,
    telegram_id: int,
    page: int = 0,
    edit: bool = False,
) -> None:
    """Show top posts from database, filtered by user topics."""
    user_data = await get_user_data(session, telegram_id)

    topics = user_data["topics"]
    sources = user_data["sources"]
    period = user_data.get("period", "7d")

    # Convert period to hours
    period_hours = PERIOD_OPTIONS.get(period, 7) * 24

    if not topics:
        no_topics_text = (
            "*No topics configured*\n\n"
            "Please add some topics first to get personalized posts\\.\n\n"
            "Use /topics or tap Topics in the menu\\."
        )
        if edit:
            await message.edit_text(
                no_topics_text, reply_markup=build_back_keyboard(), parse_mode="MarkdownV2"
            )
        else:
            await message.answer(
                no_topics_text, reply_markup=build_back_keyboard(), parse_mode="MarkdownV2"
            )
        return

    try:
        repo = PostRepository(session)

        # Get posts matching user topics from DB
        posts = await repo.get_posts_for_topics(
            topics=topics,
            limit=10,
            sources=sources if sources else None,
            hours=period_hours,
            min_score=0.1,
        )

        # Fallback: if no topic matches, get top posts by score
        if not posts:
            posts = await repo.get_top_posts_by_score(
                limit=10,
                sources=sources if sources else None,
                hours=period_hours,
                min_score=0.1,
            )

    except Exception as exc:
        LOGGER.exception("Failed to fetch posts from DB: %s", exc)
        error_text = "Failed to fetch posts\\. Please try again later\\."
        try:
            if edit:
                await message.edit_text(
                    error_text, reply_markup=build_back_keyboard(), parse_mode="MarkdownV2"
                )
            else:
                await message.answer(
                    error_text, reply_markup=build_back_keyboard(), parse_mode="MarkdownV2"
                )
        except TelegramBadRequest:
            pass
        return

    if not posts:
        no_posts_text = (
            "*No posts found*\n\n"
            "Posts are updated periodically\\. Try again later or adjust your topics/sources\\."
        )
        try:
            if edit:
                await message.edit_text(
                    no_posts_text, reply_markup=build_back_keyboard(), parse_mode="MarkdownV2"
                )
            else:
                await message.answer(
                    no_posts_text, reply_markup=build_back_keyboard(), parse_mode="MarkdownV2"
                )
        except TelegramBadRequest:
            pass
        return

    # Format posts
    text = format_db_posts(posts, topics)

    keyboard = build_back_keyboard()

    try:
        if edit:
            await message.edit_text(text, reply_markup=keyboard, parse_mode="MarkdownV2", disable_web_page_preview=True)
        else:
            await message.answer(text, reply_markup=keyboard, parse_mode="MarkdownV2", disable_web_page_preview=True)
    except TelegramBadRequest as e:
        if "message is not modified" not in str(e):
            LOGGER.warning("Failed to send/edit message: %s", e)


def format_db_posts(posts: list[Post], topics: list[str]) -> str:
    """Format posts from database for display."""
    if not posts:
        return "No posts found\\."

    topics_str = escape_markdown(", ".join(topics[:3]))
    if len(topics) > 3:
        topics_str += f" \\+{len(topics) - 3}"

    lines = [f"*Top 10 for:* {topics_str}\n"]

    for i, post in enumerate(posts, start=1):
        # Score as percentage (classifier_score is 0.0-1.0)
        score_pct = int(post.classifier_score * 100)

        # Title (truncated)
        title = post.title[:55] if post.title else "Untitled"
        if len(post.title or "") > 55:
            title += "..."

        # Source and age
        source = escape_markdown(post.source_name or "unknown")
        age = escape_markdown(format_age(post.published_at))

        # URL
        url = str(post.source_url) if post.source_url else "#"

        # Format: 1. [87%] Title (source, age)
        lines.append(
            f"{i}\\. \\[{score_pct}%\\] [{escape_markdown(title)}]({url})\n"
            f"   _{source}_ • {age}\n"
        )

    return "\n".join(lines)


__all__ = [
    "router",
    "show_top_posts",
]
