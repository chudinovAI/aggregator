from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import psycopg
    from psycopg import rows as psycopg_rows
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None
    psycopg_rows = None

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None

from .config import AggregatorConfig
from .types import Post

LOGGER = logging.getLogger(__name__)

SQLITE_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    source TEXT,
    title TEXT,
    url TEXT,
    created_utc TEXT,
    combined_score REAL,
    sentiment_score REAL,
    sentiment_label TEXT,
    payload TEXT,
    stored_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

POSTGRES_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS posts (
    id BIGSERIAL PRIMARY KEY,
    stage TEXT NOT NULL,
    source TEXT,
    title TEXT,
    url TEXT,
    created_utc TIMESTAMPTZ,
    combined_score DOUBLE PRECISION,
    sentiment_score DOUBLE PRECISION,
    sentiment_label TEXT,
    payload TEXT,
    stored_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
"""

SQLITE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_posts_stage_stored_at ON posts(stage, stored_at);
"""

POSTGRES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_posts_stage_stored_at ON posts(stage, stored_at);
"""


@dataclass
class StorageSummary:
    """Aggregate statistics pulled from persistent storage."""

    window_days: int
    total_posts: int
    source_breakdown: Dict[str, int]
    average_sentiment: Dict[str, float]
    daily_counts: List[Dict[str, Any]]
    sentiment_trend: List[Dict[str, Any]]
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_days": self.window_days,
            "total_posts": self.total_posts,
            "source_breakdown": self.source_breakdown,
            "average_sentiment": self.average_sentiment,
            "daily_counts": self.daily_counts,
            "sentiment_trend": self.sentiment_trend,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StorageSummary":
        data = payload.copy()
        generated_at = data.get("generated_at")
        if isinstance(generated_at, str):
            data["generated_at"] = datetime.fromisoformat(generated_at)
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str) -> "StorageSummary":
        return cls.from_dict(json.loads(payload))


class Storage:
    """Unified storage layer with SQLite/Postgres + Redis analytics cache."""

    def __init__(self, config: AggregatorConfig) -> None:
        self._config = config
        self._backend = config.storage_backend
        self._summary_days = config.storage_summary_days
        self._redis_ttl = config.redis_cache_ttl_seconds
        self._redis = self._init_redis()
        self._insert_sql = (
            """
            INSERT INTO posts (
                stage, source, title, url, created_utc, combined_score,
                sentiment_score, sentiment_label, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            if self._backend == "sqlite"
            else """
            INSERT INTO posts (
                stage, source, title, url, created_utc, combined_score,
                sentiment_score, sentiment_label, payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
        )
        self._init_db()

    def save_posts(self, posts: Iterable[Post], stage: str) -> None:
        """Persist posts for downstream analytics and feedback loops."""
        rows: List[Sequence[Any]] = []
        for post in posts:
            created = post.get("created_utc")
            if isinstance(created, datetime):
                created_iso = created.isoformat()
            else:
                created_iso = str(created or "")
            sentiment_score = post.get("sentiment_score")
            rows.append(
                (
                    stage,
                    post.get("source", ""),
                    post.get("title", ""),
                    post.get("url", ""),
                    created_iso,
                    float(post.get("combined_score", 0.0)),
                    float(sentiment_score) if sentiment_score is not None else None,
                    post.get("sentiment_label"),
                    json.dumps(post, ensure_ascii=False),
                )
            )

        if not rows:
            return

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(self._insert_sql, rows)
                conn.commit()
            finally:
                cursor.close()

        if stage == "report":
            self._invalidate_summary_cache()

    def summarize(self, days: Optional[int] = None) -> Optional[StorageSummary]:
        """Return cached analytics or recompute from persistent storage."""
        window = days or self._summary_days
        if window <= 0:
            return None

        cache_key = self._summary_cache_key(window)
        if self._redis:
            cached = self._redis.get(cache_key)
            if cached:
                try:
                    return StorageSummary.from_json(cached)
                except (ValueError, TypeError):
                    LOGGER.debug("Failed to decode cached summary, refreshing.")

        rows = self._fetch_summary_rows(window)
        if not rows:
            return None

        summary = self._build_summary(rows, window)
        if self._redis:
            self._redis.setex(cache_key, self._redis_ttl, summary.to_json())
        return summary

    def _connect(self):
        if self._backend == "sqlite":
            db_path = self._config.storage_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            return conn
        if psycopg is None:
            raise RuntimeError(
                "psycopg is required for Postgres storage backend but is not installed."
            )
        if not self._config.storage_pg_dsn:
            raise RuntimeError(
                "storage_pg_dsn must be configured for Postgres backend."
            )
        return psycopg.connect(
            self._config.storage_pg_dsn,
            row_factory=psycopg_rows.dict_row if psycopg_rows else None,
        )

    def _init_db(self) -> None:
        if self._backend == "sqlite":
            with self._connect() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(SQLITE_CREATE_TABLE)
                    cursor.execute(SQLITE_INDEX_SQL)
                    conn.commit()
                finally:
                    cursor.close()
            return

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(POSTGRES_CREATE_TABLE)
                cursor.execute(POSTGRES_INDEX_SQL)
                conn.commit()
            finally:
                cursor.close()

    def _init_redis(self):
        if not self._config.redis_enabled:
            return None
        if redis is None:
            raise RuntimeError(
                "redis package is required when redis_enabled=True but is missing."
            )
        client = redis.Redis.from_url(  # type: ignore[attr-defined]
            self._config.redis_url,
            decode_responses=True,
        )
        try:
            client.ping()
        except redis.RedisError as exc:  # type: ignore[attr-defined]
            LOGGER.warning("Redis unavailable, analytics cache disabled: %s", exc)
            return None
        return client

    def _summary_cache_key(self, window: int | str) -> str:
        return f"storage-summary:{self._backend}:{window}"

    def _invalidate_summary_cache(self) -> None:
        if not self._redis:
            return
        pattern = self._summary_cache_key("*")
        for key in self._redis.scan_iter(match=pattern):  # type: ignore[attr-defined]
            self._redis.delete(key)

    def _fetch_summary_rows(self, window: int) -> List[Dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(days=window)
        if self._backend == "sqlite":
            query = """
                SELECT source, stored_at, sentiment_score
                FROM posts
                WHERE stage = ? AND stored_at >= ?
                ORDER BY stored_at ASC;
            """
            params: Sequence[Any] = ("report", cutoff.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            query = """
                SELECT source, stored_at, sentiment_score
                FROM posts
                WHERE stage = %s AND stored_at >= %s
                ORDER BY stored_at ASC;
            """
            params = ("report", cutoff)

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                rows = cursor.fetchall()
            finally:
                cursor.close()

        normalized: List[Dict[str, Any]] = []
        for row in rows or []:
            row_dict = dict(row)
            row_dict["stored_at"] = self._parse_timestamp(row_dict.get("stored_at"))
            normalized.append(row_dict)
        return normalized

    def _build_summary(
        self, rows: Sequence[Dict[str, Any]], window: int
    ) -> StorageSummary:
        source_counter: Counter[str] = Counter()
        sentiment_by_source: Dict[str, List[float]] = defaultdict(list)
        per_day_counts: Dict[str, int] = defaultdict(int)
        per_day_sentiments: Dict[str, List[float]] = defaultdict(list)

        for row in rows:
            stored_at = row.get("stored_at")
            if not isinstance(stored_at, datetime):
                continue
            day_key = stored_at.strftime("%Y-%m-%d")
            source = (row.get("source") or "unknown").lower()
            source_counter[source] += 1
            per_day_counts[day_key] += 1

            sentiment_score = row.get("sentiment_score")
            if sentiment_score is None:
                continue
            sentiment = float(sentiment_score)
            sentiment_by_source[source].append(sentiment)
            per_day_sentiments[day_key].append(sentiment)

        total_posts = sum(source_counter.values())
        average_sentiment = {
            source: sum(values) / len(values)
            for source, values in sentiment_by_source.items()
        }
        daily_counts = [
            {"date": date_key, "count": per_day_counts[date_key]}
            for date_key in sorted(per_day_counts.keys())
        ]
        sentiment_trend = [
            {
                "date": date_key,
                "sentiment": sum(values) / len(values),
            }
            for date_key, values in sorted(per_day_sentiments.items())
        ]

        return StorageSummary(
            window_days=window,
            total_posts=total_posts,
            source_breakdown=dict(source_counter),
            average_sentiment=average_sentiment,
            daily_counts=daily_counts,
            sentiment_trend=sentiment_trend,
            generated_at=datetime.utcnow(),
        )

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
                return parsed.replace(tzinfo=None)
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    LOGGER.debug("Unparseable timestamp: %s", value)
                    return None
        return None


__all__ = ["Storage", "StorageSummary"]
