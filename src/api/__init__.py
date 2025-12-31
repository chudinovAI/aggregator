"""
FastAPI REST API and Telegram bot module for the news aggregator.
"""

from .main import app, create_app
from .routes import health_router, posts_router, settings_router

__all__ = [
    "app",
    "create_app",
    "health_router",
    "posts_router",
    "settings_router",
]

# Telegram bot components are imported separately to avoid
# loading aiogram when only FastAPI is needed:
#
# from .telegram_bot import main as run_telegram_bot
# from .handlers import router as telegram_router
