"""
Telegram bot initialization and runner with middleware and lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, Update
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..config import Settings, get_settings
from .handlers import router as handlers_router

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------------


class DatabaseMiddleware:
    """
    Middleware that injects a database session into each handler.

    The session is committed on success and rolled back on exception.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        """Process update with database session."""
        async with self._session_factory() as session:
            data["session"] = session
            try:
                result = await handler(event, data)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise


class LoggingMiddleware:
    """
    Middleware that logs all incoming updates.
    """

    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        """Log and process update."""
        user_id = None
        update_type = "unknown"

        if event.message:
            user_id = event.message.from_user.id if event.message.from_user else None
            update_type = "message"
            text = event.message.text or ""
            LOGGER.info(
                "Update [%s] from user %s: %s",
                update_type,
                user_id,
                text[:50] if text else "(no text)",
            )
        elif event.callback_query:
            user_id = event.callback_query.from_user.id if event.callback_query.from_user else None
            update_type = "callback"
            LOGGER.info(
                "Update [%s] from user %s: %s",
                update_type,
                user_id,
                event.callback_query.data or "(no data)",
            )

        try:
            return await handler(event, data)
        except Exception as exc:
            LOGGER.exception(
                "Error processing update [%s] from user %s: %s",
                update_type,
                user_id,
                exc,
            )
            raise


class ErrorHandlerMiddleware:
    """
    Middleware that handles errors gracefully and sends user-friendly messages.
    """

    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        """Handle errors and send user feedback."""
        try:
            return await handler(event, data)
        except Exception as exc:
            LOGGER.exception("Unhandled error in handler: %s", exc)

            # Try to send error message to user
            chat_id = None
            if event.message and event.message.chat:
                chat_id = event.message.chat.id
            elif event.callback_query and event.callback_query.message:
                chat_id = event.callback_query.message.chat.id

            if chat_id:
                try:
                    await self._bot.send_message(
                        chat_id,
                        "Sorry, something went wrong. Please try again later.",
                    )
                except Exception:
                    pass  # Ignore errors when sending error message

            # Re-raise to let dispatcher handle it
            raise


# -----------------------------------------------------------------------------
# Bot Setup
# -----------------------------------------------------------------------------


def create_bot(settings: Settings) -> Bot:
    """
    Create and configure the Telegram bot instance.

    Args:
        settings: Application settings containing the bot token.

    Returns:
        Configured Bot instance.

    Raises:
        ValueError: If bot token is not configured.
    """
    if not settings.telegram.token:
        raise ValueError(
            "Telegram bot token is not configured. Set TELEGRAM__TOKEN environment variable."
        )

    return Bot(
        token=settings.telegram.token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2),
    )


def create_dispatcher(
    bot: Bot,
    session_factory: async_sessionmaker[AsyncSession],
) -> Dispatcher:
    """
    Create and configure the dispatcher with routers and middleware.

    Args:
        bot: The Bot instance.
        session_factory: SQLAlchemy async session factory.

    Returns:
        Configured Dispatcher instance.
    """
    # Use memory storage for FSM (consider Redis for production with multiple workers)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)

    # Register middleware (order matters - executed in order)
    dp.update.middleware(LoggingMiddleware())
    dp.update.middleware(DatabaseMiddleware(session_factory))
    dp.update.middleware(ErrorHandlerMiddleware(bot))

    # Include routers
    dp.include_router(handlers_router)

    return dp


async def set_bot_commands(bot: Bot) -> None:
    """
    Set bot commands visible in Telegram UI.

    Args:
        bot: The Bot instance.
    """
    commands = [
        BotCommand(command="start", description="Start the bot and show menu"),
        BotCommand(command="top", description="Show top 10 interesting posts"),
        BotCommand(command="topics", description="Manage your interest topics"),
        BotCommand(command="sources", description="Choose data sources"),
        BotCommand(command="period", description="Set search time period"),
        BotCommand(command="settings", description="View your current settings"),
        BotCommand(command="help", description="Show help information"),
    ]
    await bot.set_my_commands(commands)
    LOGGER.info("Bot commands registered: %d commands", len(commands))


# -----------------------------------------------------------------------------
# Application Lifecycle
# -----------------------------------------------------------------------------


@asynccontextmanager
async def create_app_context(
    settings: Settings,
) -> AsyncIterator[tuple[Bot, Dispatcher, async_sessionmaker[AsyncSession]]]:
    """
    Create application context with all dependencies.

    Yields:
        Tuple of (bot, dispatcher, session_factory)
    """
    # Create database engine and session factory
    engine = create_async_engine(
        str(settings.database.url),
        pool_size=settings.database.pool_size,
        echo=settings.database.echo,
    )
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    # Create Redis client (optional, for future use)
    redis_client: Redis | None = None
    try:
        redis_client = Redis.from_url(
            str(settings.redis.url),
            max_connections=settings.redis.max_connections,
            decode_responses=True,
        )
        await redis_client.ping()
        LOGGER.info("Redis connection established")
    except Exception as exc:
        LOGGER.warning("Redis connection failed, continuing without cache: %s", exc)
        redis_client = None

    # Create bot and dispatcher
    bot = create_bot(settings)
    dp = create_dispatcher(bot, session_factory)

    LOGGER.info("Application context initialized")

    try:
        yield bot, dp, session_factory
    finally:
        LOGGER.info("Cleaning up application context...")

        # Close Redis connection
        if redis_client:
            await redis_client.aclose()  # type: ignore[attr-defined]

        # Dispose database engine
        await engine.dispose()

        # Close bot session
        await bot.session.close()

        LOGGER.info("Application context cleaned up")


async def run_polling(settings: Settings) -> None:
    """
    Run the bot in polling mode.

    This is the main entry point for running the bot.

    Args:
        settings: Application settings.
    """
    async with create_app_context(settings) as (bot, dp, _):
        # Set bot commands
        await set_bot_commands(bot)

        # Log bot info
        bot_info = await bot.get_me()
        LOGGER.info(
            "Starting bot @%s (ID: %d) in polling mode",
            bot_info.username,
            bot_info.id,
        )

        # Start polling
        try:
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
                drop_pending_updates=True,
            )
        except asyncio.CancelledError:
            LOGGER.info("Polling cancelled, shutting down...")
        finally:
            await dp.stop_polling()


async def run_webhook(
    settings: Settings,
    webhook_url: str,
    host: str = "0.0.0.0",
    port: int = 8443,
) -> None:
    """
    Run the bot in webhook mode (for production).

    Args:
        settings: Application settings.
        webhook_url: Public URL for the webhook.
        host: Host to bind the server to.
        port: Port to bind the server to.
    """
    from aiohttp import web

    async with create_app_context(settings) as (bot, dp, _):
        # Set bot commands
        await set_bot_commands(bot)

        # Set webhook
        await bot.set_webhook(
            url=webhook_url,
            drop_pending_updates=True,
        )

        bot_info = await bot.get_me()
        LOGGER.info(
            "Starting bot @%s in webhook mode at %s",
            bot_info.username,
            webhook_url,
        )

        # Create aiohttp app
        app = web.Application()

        async def handle_webhook(request: web.Request) -> web.Response:
            """Handle incoming webhook updates."""
            try:
                data = await request.json()
                update = Update(**data)
                await dp.feed_update(bot, update)
                return web.Response(status=200)
            except Exception as exc:
                LOGGER.exception("Error processing webhook: %s", exc)
                return web.Response(status=500)

        app.router.add_post("/webhook", handle_webhook)

        # Health check endpoint
        async def health_check(_: web.Request) -> web.Response:
            return web.Response(text="OK", status=200)

        app.router.add_get("/health", health_check)

        # Run server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)

        try:
            await site.start()
            LOGGER.info("Webhook server started on %s:%d", host, port)

            # Keep running until cancelled
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            LOGGER.info("Webhook server stopping...")
        finally:
            await bot.delete_webhook()
            await runner.cleanup()


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------


def configure_logging(level: str = "INFO") -> None:
    """Configure logging for the bot."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    # Reduce noise from aiogram
    logging.getLogger("aiogram").setLevel(logging.WARNING)


def main() -> int:
    """
    Main entry point for running the Telegram bot.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    settings = get_settings()
    configure_logging(settings.logging.level)

    LOGGER.info(
        "Starting News Aggregator Telegram Bot v%s (%s)",
        settings.app.version,
        settings.app.environment.value,
    )

    # Validate configuration
    if not settings.telegram.token:
        LOGGER.error("Telegram bot token not configured. Set TELEGRAM__TOKEN environment variable.")
        return 1

    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig: int, _: Any) -> None:
        LOGGER.info("Received signal %d, initiating shutdown...", sig)
        for task in asyncio.all_tasks(loop):
            task.cancel()

    if sys.platform != "win32":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(run_polling(settings))
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        return 0
    except Exception as exc:
        LOGGER.exception("Fatal error: %s", exc)
        return 1
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
