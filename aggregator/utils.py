from __future__ import annotations

import html
import logging
import re
from datetime import datetime
from typing import Any, Callable, Optional, Sequence, Tuple

from .types import Post

LOGGER = logging.getLogger(__name__)

ISO8601_DURATION_PATTERN = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def parse_iso8601_duration(duration: str) -> int:
    if not duration:
        raise ValueError("Duration string is empty.")
    match = ISO8601_DURATION_PATTERN.fullmatch(duration)
    if not match:
        raise ValueError(f"Invalid ISO 8601 duration: {duration}")
    hours, minutes, seconds = match.groups()
    return (int(hours or 0) * 3600) + (int(minutes or 0) * 60) + int(seconds or 0)


def format_datetime(value: Optional[datetime], fmt: str) -> str:
    if not value:
        return "-"
    return value.strftime(fmt)


def format_duration(seconds: Optional[int]) -> str:
    if not seconds:
        return "-"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def ml_badge(score: Optional[float]) -> str:
    if score is None:
        return ""
    return f"<span class='ml-score-badge'>{float(score):.3f}</span>"


def make_link_cell(post: Post) -> str:
    url = html.escape(post.get("url", "") or "")
    title = html.escape(post.get("title", "") or "")
    return f"<a href='{url}'>{title}</a>"


def escape_field(value: Any) -> str:
    return html.escape(str(value))


def build_html_table(
    items: Sequence[Post],
    columns: Sequence[Tuple[str, Callable[[Post], str]]],
) -> str:
    header_html = "".join(f"<th>{title}</th>" for title, _ in columns)
    rows = []
    for item in items:
        row_cells = "".join(f"<td>{renderer(item)}</td>" for _, renderer in columns)
        rows.append(f"<tr>{row_cells}</tr>")
    return (
        '<table class="sortable">'
        "<thead><tr>"
        f"{header_html}"
        "</tr></thead>"
        "<tbody>"
        f"{''.join(rows)}"
        "</tbody></table>"
    )


def log_post_summary(prefix: str, posts: Sequence[Post]) -> None:
    for index, post in enumerate(posts[:5], start=1):
        LOGGER.debug(
            "%s %02d. source=%s score=%.3f title=%s",
            prefix,
            index,
            post.get("source", "unknown"),
            float(post.get("combined_score", 0.0)),
            (post.get("title") or "")[:80],
        )
