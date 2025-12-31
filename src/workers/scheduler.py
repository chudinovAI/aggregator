"""
APScheduler-based async scheduler for background tasks.

This module provides:
- Async scheduler initialization and configuration
- Job registration from tasks module
- Error handling and logging
- Graceful shutdown support
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    JobExecutionEvent,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..config import Settings, get_settings
from .cleanup import cleanup_old_posts
from .notifications import send_daily_digest
from .parsing import parse_all_sources
from .training import retrain_classifier
from .types import TaskResult

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Job Listener
# -----------------------------------------------------------------------------


def create_job_listener(settings: Settings) -> Callable[[JobExecutionEvent], None]:
    """
    Create a job event listener for logging and monitoring.

    Args:
        settings: Application settings

    Returns:
        Event listener function
    """

    def job_listener(event: JobExecutionEvent) -> None:
        """Handle job execution events."""
        job_id = event.job_id
        scheduled_time = event.scheduled_run_time

        if hasattr(event, "exception") and event.exception:
            LOGGER.error(
                "Job %s failed at %s: %s",
                job_id,
                scheduled_time,
                event.exception,
                exc_info=event.exception,
            )
        elif hasattr(event, "retval"):
            result = event.retval
            if isinstance(result, TaskResult):
                status = "SUCCESS" if result.success else "FAILED"
                LOGGER.info(
                    "Job %s completed [%s] in %.2fs: %s",
                    job_id,
                    status,
                    result.duration_seconds,
                    result.message,
                )
            else:
                LOGGER.info("Job %s completed at %s", job_id, scheduled_time)
        else:
            LOGGER.warning("Job %s missed at %s", job_id, scheduled_time)

    return job_listener


# -----------------------------------------------------------------------------
# Scheduler Factory
# -----------------------------------------------------------------------------


def create_scheduler(
    session_factory: async_sessionmaker[AsyncSession],
    redis_client: Redis | None,
    settings: Settings,
    *,
    telegram_bot: Any | None = None,
) -> AsyncIOScheduler:
    """
    Create and configure the async scheduler with all jobs.

    Args:
        session_factory: SQLAlchemy async session factory
        redis_client: Optional Redis client for caching
        settings: Application settings
        telegram_bot: Optional Telegram bot for notifications

    Returns:
        Configured AsyncIOScheduler instance
    """
    scheduler = AsyncIOScheduler(
        timezone="UTC",
        job_defaults={
            "coalesce": True,  # Combine missed runs into one
            "max_instances": 1,  # Only one instance per job
            "misfire_grace_time": 300,  # 5 minutes grace period
        },
    )

    # Add job listener
    listener = create_job_listener(settings)
    scheduler.add_listener(listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)

    # -------------------------------------------------------------------------
    # Job: Parse All Sources
    # -------------------------------------------------------------------------
    # Runs based on parsing.interval_seconds (default: every 5 minutes / 300s)
    # For production, you might want to run hourly

    parse_interval = max(settings.parsing.interval_seconds, 60)  # At least 1 minute

    scheduler.add_job(
        parse_all_sources,
        trigger=IntervalTrigger(seconds=parse_interval),
        id="parse_all_sources",
        name="Parse All Sources",
        kwargs={
            "session_factory": session_factory,
            "redis_client": redis_client,
            "settings": settings,
        },
        replace_existing=True,
    )
    LOGGER.info(
        "Registered job: parse_all_sources (every %d seconds)",
        parse_interval,
    )

    # -------------------------------------------------------------------------
    # Job: Retrain Classifier
    # -------------------------------------------------------------------------
    # Runs daily at 2:00 AM UTC

    scheduler.add_job(
        retrain_classifier,
        trigger=CronTrigger(hour=2, minute=0),
        id="retrain_classifier",
        name="Retrain Classifier",
        kwargs={
            "session_factory": session_factory,
            "settings": settings,
            "min_samples": 100,
        },
        replace_existing=True,
    )
    LOGGER.info("Registered job: retrain_classifier (daily at 02:00 UTC)")

    # -------------------------------------------------------------------------
    # Job: Cleanup Old Posts
    # -------------------------------------------------------------------------
    # Runs daily at 3:00 AM UTC

    scheduler.add_job(
        cleanup_old_posts,
        trigger=CronTrigger(hour=3, minute=0),
        id="cleanup_old_posts",
        name="Cleanup Old Posts",
        kwargs={
            "session_factory": session_factory,
            "settings": settings,
        },
        replace_existing=True,
    )
    LOGGER.info("Registered job: cleanup_old_posts (daily at 03:00 UTC)")

    # -------------------------------------------------------------------------
    # Job: Send Daily Digest (optional)
    # -------------------------------------------------------------------------
    # Runs daily at 8:00 AM UTC (if telegram bot is configured)

    if telegram_bot or settings.telegram.token:
        scheduler.add_job(
            send_daily_digest,
            trigger=CronTrigger(hour=8, minute=0),
            id="send_daily_digest",
            name="Send Daily Digest",
            kwargs={
                "session_factory": session_factory,
                "settings": settings,
                "telegram_bot": telegram_bot,
                "top_n": 10,
            },
            replace_existing=True,
        )
        LOGGER.info("Registered job: send_daily_digest (daily at 08:00 UTC)")

    return scheduler


# -----------------------------------------------------------------------------
# Application Context
# -----------------------------------------------------------------------------


@asynccontextmanager
async def create_scheduler_context(
    settings: Settings,
) -> AsyncIterator[tuple[AsyncIOScheduler, async_sessionmaker[AsyncSession], Redis[str] | None]]:
    """
    Create scheduler context with all dependencies.

    Yields:
        Tuple of (scheduler, session_factory, redis_client)
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

    # Create Redis client
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
        LOGGER.warning("Redis connection failed: %s", exc)
        redis_client = None

    # Create scheduler
    scheduler = create_scheduler(session_factory, redis_client, settings)

    LOGGER.info("Scheduler context initialized")

    try:
        yield scheduler, session_factory, redis_client
    finally:
        LOGGER.info("Cleaning up scheduler context...")

        # Shutdown scheduler
        if scheduler.running:
            scheduler.shutdown(wait=True)

        # Close Redis connection
        if redis_client:
            await redis_client.aclose()  # type: ignore[attr-defined]

        # Dispose database engine
        await engine.dispose()

        LOGGER.info("Scheduler context cleaned up")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


async def run_scheduler(settings: Settings) -> None:
    """
    Run the scheduler in the foreground.

    This is the main entry point for running the scheduler as a standalone process.

    Args:
        settings: Application settings
    """
    async with create_scheduler_context(settings) as (scheduler, _, _):
        LOGGER.info("Starting scheduler...")
        scheduler.start()

        # Print registered jobs
        jobs = scheduler.get_jobs()
        LOGGER.info("Scheduler running with %d jobs:", len(jobs))
        for job in jobs:
            LOGGER.info("  - %s: %s", job.id, job.next_run_time)

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            LOGGER.info("Scheduler cancelled, shutting down...")


def configure_logging(level: str = "INFO") -> None:
    """Configure logging for the scheduler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    # Reduce noise from APScheduler
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def main() -> int:
    """
    Main entry point for running the scheduler.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    settings = get_settings()
    configure_logging(settings.logging.level)

    LOGGER.info(
        "Starting News Aggregator Scheduler v%s (%s)",
        settings.app.version,
        settings.app.environment.value,
    )

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
        loop.run_until_complete(run_scheduler(settings))
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
