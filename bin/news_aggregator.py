import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregator import AdvancedNewsAggregator  # noqa

REQUIRED_ENV_VARS: Sequence[str] = (
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "YOUTUBE_API_KEY",
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_environment() -> bool:
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        for var in missing:
            logging.error("Missing environment variable: %s", var)
        logging.error("Create a .env file with the required credentials.")
        return False
    return True


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()

    if not ensure_environment():
        return 1

    aggregator = AdvancedNewsAggregator(use_ml=True)
    result = aggregator.run()
    if not result:
        logging.error("Aggregation did not produce a report.")
        return 1

    logging.info(
        "Aggregation completed with %d posts. Report saved to %s",
        len(result.posts),
        result.report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
