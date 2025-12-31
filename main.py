"""
Application entry points for the news aggregator service.

Usage:
    # Run FastAPI server (production - uses Granian)
    python main.py api

    # Run FastAPI server (development - uses Uvicorn with hot-reload)
    python main.py api --dev

    # Run Telegram bot
    python main.py bot

    # Run background scheduler
    python main.py scheduler
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


def run_api_granian() -> int:
    """Run the FastAPI application with Granian (production, 3x faster than uvicorn)."""
    try:
        from granian import Granian
        from granian.constants import Interfaces

        from src.config import get_settings

        settings = get_settings()

        # Get server config from environment or defaults
        host = os.getenv("SERVER__HOST", "0.0.0.0")
        port = int(os.getenv("SERVER__PORT", "8000"))
        workers = int(os.getenv("SERVER__WORKERS", "4"))
        backlog = int(os.getenv("SERVER__BACKLOG", "2048"))

        print(f"Starting Granian server on {host}:{port} with {workers} workers...")

        server = Granian(
            target="src.api.main:app",
            address=host,
            port=port,
            workers=workers,
            backlog=backlog,
            interface=Interfaces.ASGI,
            log_level="info" if not settings.app.debug else "debug",
        )
        server.serve()
        return 0
    except ImportError as exc:
        print(f"Error: {exc}. Install with: pip install '.[api]'", file=sys.stderr)
        return 1


def run_api_uvicorn() -> int:
    """Run the FastAPI application with Uvicorn (development, with hot-reload)."""
    try:
        import uvicorn

        from src.config import get_settings

        settings = get_settings()

        host = os.getenv("SERVER__HOST", "0.0.0.0")
        port = int(os.getenv("SERVER__PORT", "8000"))

        print(f"Starting Uvicorn dev server on {host}:{port} with hot-reload...")

        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=True,
            log_level="debug" if settings.app.debug else "info",
        )
        return 0
    except ImportError as exc:
        print(
            f"Error: {exc}. Install with: pip install '.[api-dev]'",
            file=sys.stderr,
        )
        return 1


def run_bot() -> int:
    """Run the Telegram bot."""
    try:
        from src.api.telegram_bot import run_polling
        from src.config import get_settings

        settings = get_settings()

        if not settings.telegram.token:
            print("Error: TELEGRAM__TOKEN not configured", file=sys.stderr)
            print("Set it in .env or as environment variable", file=sys.stderr)
            return 1

        asyncio.run(run_polling(settings))
        return 0
    except ImportError as exc:
        print(f"Error: {exc}. Install with: pip install '.[bot]'", file=sys.stderr)
        return 1


def run_scheduler() -> int:
    """Run the background task scheduler."""
    try:
        from src.config import get_settings
        from src.workers.scheduler import run_scheduler as _run_scheduler

        settings = get_settings()
        asyncio.run(_run_scheduler(settings))
        return 0
    except ImportError as exc:
        print(f"Error: {exc}. Install with: pip install '.[scheduler]'", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point with subcommand routing."""
    parser = argparse.ArgumentParser(
        description="News Aggregator Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # API subcommand
    api_parser = subparsers.add_parser("api", help="Run FastAPI server")
    api_parser.add_argument(
        "--dev",
        action="store_true",
        help="Use Uvicorn with hot-reload (development mode)",
    )

    # Bot subcommand
    subparsers.add_parser("bot", help="Run Telegram bot")

    # Scheduler subcommand
    subparsers.add_parser("scheduler", help="Run background task scheduler")

    args = parser.parse_args()

    if args.command == "api":
        if args.dev:
            return run_api_uvicorn()
        return run_api_granian()
    elif args.command == "bot":
        return run_bot()
    elif args.command == "scheduler":
        return run_scheduler()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
