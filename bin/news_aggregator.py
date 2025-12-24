import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregator import AdvancedNewsAggregator, AggregatorConfig  # noqa
from aggregator.secrets import load_runtime_secrets


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in payload or key in ("args", "msg", "name"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def main() -> int:
    load_runtime_secrets(required_keys=REQUIRED_SECRETS)
    configure_logging()

    try:
        config = AggregatorConfig()
    except ValidationError as exc:  # type: ignore[attr-defined]
        logging.error("Configuration validation failed", extra={"errors": exc.errors()})
        return 1

    aggregator = AdvancedNewsAggregator(config=config, use_ml=True)
    result = aggregator.run()
    if not result:
        logging.error("Aggregation did not produce a report.")
        return 1

    logging.info(
        "Aggregation completed with %d posts. Report saved to %s",
        len(result.posts),
        result.report_path,
        extra={
            "json_report": result.json_path,
            "markdown_report": result.markdown_path,
            "rss_report": result.rss_path,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
REQUIRED_SECRETS = {
    "REDDIT_CLIENT_ID": "Reddit API client id",
    "REDDIT_CLIENT_SECRET": "Reddit API client secret",
    "YOUTUBE_API_KEY": "YouTube API key",
}
