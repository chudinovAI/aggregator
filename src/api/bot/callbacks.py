"""
Callback query handlers for the Telegram bot.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.repository import UserRepository
from .commands import show_top_posts
from .constants import AVAILABLE_SOURCES, PERIOD_OPTIONS
from .keyboards import (
    build_back_keyboard,
    build_main_menu_keyboard,
    build_period_keyboard,
    build_sources_keyboard,
    build_topics_keyboard,
)
from .states import TopicStates
from .utils import escape_markdown, get_user_data, safe_edit_text

router = Router(name="callbacks")


async def safe_answer(callback: CallbackQuery, text: str | None = None) -> None:
    """Safely answer callback query, ignoring timeout errors."""
    try:
        await callback.answer(text)
    except TelegramBadRequest:
        # Callback query expired (user clicked too long ago)
        pass


# -----------------------------------------------------------------------------
# Action Callbacks
# -----------------------------------------------------------------------------


@router.callback_query(F.data == "action:menu")
async def callback_menu(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle back to menu action."""
    if not callback.message or not callback.from_user:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    user_data = await get_user_data(session, callback.from_user.id)

    text = (
        "*Main Menu*\n\n"
        f"Topics: {len(user_data['topics'])} \\| "
        f"Sources: {len(user_data['sources'])} \\| "
        f"Period: {user_data.get('period', '7d')}"
    )

    await safe_edit_text(
        callback.message,
        text,
        reply_markup=build_main_menu_keyboard(),
    )


@router.callback_query(F.data == "action:top")
async def callback_top(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle top posts action."""
    if not callback.message or not callback.from_user:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    await show_top_posts(callback.message, session, callback.from_user.id, edit=True)


@router.callback_query(F.data == "action:topics")
async def callback_topics(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle topics action."""
    if not callback.message or not callback.from_user:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    user_data = await get_user_data(session, callback.from_user.id)
    topics = user_data["topics"]

    if topics:
        text = f"*Your Topics* \\({len(topics)}\\)\n\nTap to remove or add new ones:"
    else:
        text = "*Your Topics*\n\nNo topics configured\\. Add some to get personalized posts\\!"

    await safe_edit_text(
        callback.message,
        text,
        reply_markup=build_topics_keyboard(topics),
    )


@router.callback_query(F.data == "action:sources")
async def callback_sources(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle sources action."""
    if not callback.message or not callback.from_user:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    user_data = await get_user_data(session, callback.from_user.id)
    sources = user_data["sources"]

    text = f"*Data Sources* \\({len(sources)} active\\)\n\nToggle to include/exclude:"

    await safe_edit_text(
        callback.message,
        text,
        reply_markup=build_sources_keyboard(sources),
    )


@router.callback_query(F.data == "action:period")
async def callback_period(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle period action."""
    if not callback.message or not callback.from_user:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    user_data = await get_user_data(session, callback.from_user.id)
    current_period = user_data.get("period", "7d")

    text = "*Search Period*\n\nChoose how far back to search:"

    await safe_edit_text(
        callback.message,
        text,
        reply_markup=build_period_keyboard(current_period),
    )


@router.callback_query(F.data == "action:settings")
async def callback_settings(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle settings action."""
    if not callback.message or not callback.from_user:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    user_data = await get_user_data(session, callback.from_user.id)

    topics_str = ", ".join(user_data["topics"][:5]) if user_data["topics"] else "None"
    sources_str = ", ".join(s.replace("_", " ").title() for s in user_data["sources"][:4])

    text = (
        "*Your Settings*\n\n"
        f"*Topics:* {escape_markdown(topics_str)}\n"
        f"*Sources:* {escape_markdown(sources_str)}\n"
        f"*Period:* {user_data.get('period', '7d')}"
    )

    await safe_edit_text(
        callback.message,
        text,
        reply_markup=build_main_menu_keyboard(),
    )


@router.callback_query(F.data == "action:help")
async def callback_help(callback: CallbackQuery) -> None:
    """Handle help action."""
    if not callback.message:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    help_text = (
        "*Help*\n\n"
        "• *Top Posts* \\- View best posts based on your preferences\n"
        "• *Topics* \\- Add/remove interest topics\n"
        "• *Sources* \\- Enable/disable news sources\n"
        "• *Period* \\- Set how far back to search\n"
    )

    await safe_edit_text(
        callback.message,
        help_text,
        reply_markup=build_back_keyboard(),
    )


# -----------------------------------------------------------------------------
# Topic Management Callbacks
# -----------------------------------------------------------------------------


@router.callback_query(F.data.startswith("topic:add:"))
async def callback_topic_add(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle adding a topic."""
    if not callback.message or not callback.from_user or not callback.data:
        return

    topic = callback.data.split(":", 2)[2]
    repo = UserRepository(session)
    user = await repo.get_or_create(callback.from_user.id)

    current_topics = list(user.topics or [])
    if topic not in current_topics:
        current_topics.append(topic)
        await repo.update_preferences(callback.from_user.id, topics=current_topics)
        await safe_answer(callback, f"Added: {topic}")
    else:
        await safe_answer(callback, "Topic already added")

    # Refresh the view
    text = f"*Your Topics* \\({len(current_topics)}\\)"
    await safe_edit_text(
        callback.message,
        text,
        reply_markup=build_topics_keyboard(current_topics),
    )


@router.callback_query(F.data.startswith("topic:remove:"))
async def callback_topic_remove(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle removing a topic."""
    if not callback.message or not callback.from_user or not callback.data:
        return

    topic = callback.data.split(":", 2)[2]
    repo = UserRepository(session)
    user = await repo.get_or_create(callback.from_user.id)

    current_topics = list(user.topics or [])
    # Find and remove the topic (handles truncated names)
    current_topics = [t for t in current_topics if not t.startswith(topic)]
    await repo.update_preferences(callback.from_user.id, topics=current_topics)

    await safe_answer(callback, f"Removed: {topic}")

    text = (
        f"*Your Topics* \\({len(current_topics)}\\)"
        if current_topics
        else "*Your Topics*\n\nNo topics configured\\."
    )
    await safe_edit_text(
        callback.message,
        text,
        reply_markup=build_topics_keyboard(current_topics),
    )


@router.callback_query(F.data == "topic:clear")
async def callback_topic_clear(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle clearing all topics."""
    if not callback.message or not callback.from_user:
        return

    repo = UserRepository(session)
    await repo.update_preferences(callback.from_user.id, topics=[])

    await safe_answer(callback, "All topics cleared")
    await safe_edit_text(
        callback.message,
        "*Your Topics*\n\nNo topics configured\\.",
        reply_markup=build_topics_keyboard([]),
    )


@router.callback_query(F.data == "topic:custom")
async def callback_topic_custom(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    """Handle custom topic input request."""
    if not callback.message:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    await state.set_state(TopicStates.waiting_for_topic)
    await safe_edit_text(
        callback.message,
        "*Add Custom Topic*\n\nSend me the topic you want to add\\.\n\nType /cancel to go back\\.",
    )


@router.message(TopicStates.waiting_for_topic)
async def handle_custom_topic(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
) -> None:
    """Handle custom topic input."""
    if not message.from_user or not message.text:
        return

    if message.text.lower() == "/cancel":
        await state.clear()
        user_data = await get_user_data(session, message.from_user.id)
        await message.answer(
            "*Your Topics*",
            reply_markup=build_topics_keyboard(user_data["topics"]),
            parse_mode="MarkdownV2",
        )
        return

    topic = message.text.strip()[:50]  # Limit length
    repo = UserRepository(session)
    user = await repo.get_or_create(message.from_user.id)

    current_topics = list(user.topics or [])
    if topic not in current_topics:
        current_topics.append(topic)
        await repo.update_preferences(message.from_user.id, topics=current_topics)

    await state.clear()
    await message.answer(
        f"*Your Topics* \\({len(current_topics)}\\)\n\nAdded: {escape_markdown(topic)}",
        reply_markup=build_topics_keyboard(current_topics),
        parse_mode="MarkdownV2",
    )


# -----------------------------------------------------------------------------
# Source Management Callbacks
# -----------------------------------------------------------------------------


@router.callback_query(F.data.startswith("source:enable:"))
async def callback_source_enable(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle enabling a source."""
    if not callback.message or not callback.from_user or not callback.data:
        return

    source = callback.data.split(":", 2)[2]
    repo = UserRepository(session)
    user = await repo.get_or_create(callback.from_user.id)

    current_sources = list(user.sources or [])
    if source not in current_sources:
        current_sources.append(source)
        await repo.update_preferences(callback.from_user.id, sources=current_sources)
        await safe_answer(callback, f"Enabled: {source}")
    else:
        await safe_answer(callback)

    await safe_edit_text(
        callback.message,
        f"*Data Sources* \\({len(current_sources)} active\\)",
        reply_markup=build_sources_keyboard(current_sources),
    )


@router.callback_query(F.data.startswith("source:disable:"))
async def callback_source_disable(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle disabling a source."""
    if not callback.message or not callback.from_user or not callback.data:
        return

    source = callback.data.split(":", 2)[2]
    repo = UserRepository(session)
    user = await repo.get_or_create(callback.from_user.id)

    current_sources = list(user.sources or [])
    if source in current_sources:
        current_sources.remove(source)
        await repo.update_preferences(callback.from_user.id, sources=current_sources)
        await safe_answer(callback, f"Disabled: {source}")
    else:
        await safe_answer(callback)

    await safe_edit_text(
        callback.message,
        f"*Data Sources* \\({len(current_sources)} active\\)",
        reply_markup=build_sources_keyboard(current_sources),
    )


@router.callback_query(F.data == "source:all")
async def callback_source_all(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle selecting all sources."""
    if not callback.message or not callback.from_user:
        return

    repo = UserRepository(session)
    await repo.update_preferences(callback.from_user.id, sources=AVAILABLE_SOURCES.copy())

    await safe_answer(callback, "All sources enabled")
    await safe_edit_text(
        callback.message,
        f"*Data Sources* \\({len(AVAILABLE_SOURCES)} active\\)",
        reply_markup=build_sources_keyboard(AVAILABLE_SOURCES.copy()),
    )


@router.callback_query(F.data == "source:none")
async def callback_source_none(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle clearing all sources."""
    if not callback.message or not callback.from_user:
        return

    repo = UserRepository(session)
    await repo.update_preferences(callback.from_user.id, sources=[])

    await safe_answer(callback, "All sources disabled")
    await safe_edit_text(
        callback.message,
        "*Data Sources* \\(0 active\\)",
        reply_markup=build_sources_keyboard([]),
    )


# -----------------------------------------------------------------------------
# Period Management Callbacks
# -----------------------------------------------------------------------------


@router.callback_query(F.data.startswith("period:set:"))
async def callback_period_set(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle setting the search period."""
    if not callback.message or not callback.from_user or not callback.data:
        return

    period = callback.data.split(":", 2)[2]

    # Save period to database
    repo = UserRepository(session)
    await repo.update_preferences(callback.from_user.id, period=period)

    await safe_answer(callback, f"Period set to {period}")
    await safe_edit_text(
        callback.message,
        f"*Search Period*\n\nNow searching the last {PERIOD_OPTIONS.get(period, 7)} days\\.",
        reply_markup=build_period_keyboard(period),
    )


# -----------------------------------------------------------------------------
# Posts Pagination Callbacks
# -----------------------------------------------------------------------------


@router.callback_query(F.data.startswith("posts:page:"))
async def callback_posts_page(callback: CallbackQuery, session: AsyncSession) -> None:
    """Handle posts pagination."""
    if not callback.message or not callback.from_user or not callback.data:
        return

    # Answer immediately to prevent timeout
    await safe_answer(callback)

    page = int(callback.data.split(":", 2)[2])
    await show_top_posts(callback.message, session, callback.from_user.id, page=page, edit=True)


@router.callback_query(F.data == "posts:noop")
async def callback_posts_noop(callback: CallbackQuery) -> None:
    """Handle no-op callback (page indicator)."""
    await safe_answer(callback)


__all__ = ["router"]
