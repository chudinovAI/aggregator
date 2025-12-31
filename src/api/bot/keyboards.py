"""
Keyboard builder functions for the Telegram bot.
"""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo

from ...config import get_settings
from ...db.models import Post
from .constants import AVAILABLE_SOURCES, SUGGESTED_TOPICS


def build_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Build the main menu inline keyboard."""
    settings = get_settings()

    buttons = [
        [
            InlineKeyboardButton(text="Top 10 Posts", callback_data="action:top"),
            InlineKeyboardButton(text="My Topics", callback_data="action:topics"),
        ],
        [
            InlineKeyboardButton(text="Sources", callback_data="action:sources"),
            InlineKeyboardButton(text="Period", callback_data="action:period"),
        ],
        [
            InlineKeyboardButton(text="My Settings", callback_data="action:settings"),
            InlineKeyboardButton(text="Help", callback_data="action:help"),
        ],
    ]

    # Add Web App button if URL is configured
    if settings.telegram.web_app_url:
        buttons.append([
            InlineKeyboardButton(
                text="Open App",
                web_app=WebAppInfo(url=settings.telegram.web_app_url),
            )
        ])

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def build_topics_keyboard(current_topics: list[str]) -> InlineKeyboardMarkup:
    """Build keyboard for topic management."""
    buttons: list[list[InlineKeyboardButton]] = []

    # Show current topics with remove option
    for topic in current_topics[:10]:  # Limit to 10
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"❌ {topic}",
                    callback_data=f"topic:remove:{topic[:20]}",
                )
            ]
        )

    # Add suggested topics that aren't already selected
    suggestions = [t for t in SUGGESTED_TOPICS if t not in current_topics][:4]
    if suggestions:
        suggestion_row = [
            InlineKeyboardButton(
                text=f"➕ {topic}",
                callback_data=f"topic:add:{topic}",
            )
            for topic in suggestions
        ]
        # Split into rows of 2
        for i in range(0, len(suggestion_row), 2):
            buttons.append(suggestion_row[i : i + 2])

    # Action buttons
    buttons.append(
        [
            InlineKeyboardButton(text="➕ Add Custom", callback_data="topic:custom"),
            InlineKeyboardButton(text="🗑 Clear All", callback_data="topic:clear"),
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(text="« Back to Menu", callback_data="action:menu"),
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def build_sources_keyboard(current_sources: list[str]) -> InlineKeyboardMarkup:
    """Build keyboard for source selection."""
    buttons: list[list[InlineKeyboardButton]] = []

    # Show all sources with toggle
    for source in AVAILABLE_SOURCES:
        is_enabled = source in current_sources
        icon = "✅" if is_enabled else "⬜"
        action = "disable" if is_enabled else "enable"
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{icon} {source.replace('_', ' ').title()}",
                    callback_data=f"source:{action}:{source}",
                )
            ]
        )

    # Action buttons
    buttons.append(
        [
            InlineKeyboardButton(text="✅ Select All", callback_data="source:all"),
            InlineKeyboardButton(text="⬜ Clear All", callback_data="source:none"),
        ]
    )
    buttons.append(
        [
            InlineKeyboardButton(text="« Back to Menu", callback_data="action:menu"),
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def build_period_keyboard(current_period: str = "7d") -> InlineKeyboardMarkup:
    """Build keyboard for period selection."""
    buttons: list[list[InlineKeyboardButton]] = []

    period_labels = {
        "1d": "Last 24 hours",
        "3d": "Last 3 days",
        "7d": "Last week",
        "14d": "Last 2 weeks",
        "30d": "Last month",
    }

    for period_key, label in period_labels.items():
        icon = "🔘" if period_key == current_period else "⚪"
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{icon} {label}",
                    callback_data=f"period:set:{period_key}",
                )
            ]
        )

    buttons.append(
        [
            InlineKeyboardButton(text="« Back to Menu", callback_data="action:menu"),
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def build_posts_keyboard(
    posts: list[Post],
    page: int = 0,
    total_pages: int = 1,
) -> InlineKeyboardMarkup:
    """Build keyboard for post navigation."""
    buttons: list[list[InlineKeyboardButton]] = []

    # Post buttons (open URL)
    for i, post in enumerate(posts):
        short_title = post.title[:40] + "..." if len(post.title) > 40 else post.title
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{i + 1}. {short_title}",
                    url=str(post.source_url),
                )
            ]
        )

    # Pagination
    nav_buttons: list[InlineKeyboardButton] = []
    if page > 0:
        nav_buttons.append(
            InlineKeyboardButton(text="« Prev", callback_data=f"posts:page:{page - 1}")
        )
    nav_buttons.append(
        InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="posts:noop")
    )
    if page < total_pages - 1:
        nav_buttons.append(
            InlineKeyboardButton(text="Next »", callback_data=f"posts:page:{page + 1}")
        )

    if nav_buttons:
        buttons.append(nav_buttons)

    buttons.append(
        [
            InlineKeyboardButton(text="🔄 Refresh", callback_data="action:top"),
            InlineKeyboardButton(text="« Menu", callback_data="action:menu"),
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def build_back_keyboard() -> InlineKeyboardMarkup:
    """Simple back to menu keyboard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="« Back to Menu", callback_data="action:menu")]]
    )


__all__ = [
    "build_main_menu_keyboard",
    "build_topics_keyboard",
    "build_sources_keyboard",
    "build_period_keyboard",
    "build_posts_keyboard",
    "build_back_keyboard",
]
