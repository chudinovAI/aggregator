from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import AggregatorConfig
from .types import Post
from .utils import (
    build_html_table,
    escape_field,
    format_datetime,
    format_duration,
    make_link_cell,
    ml_badge,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLE_CLASS = "source-table"
SOURCE_TABLE_WIDTH = "100%"


class HtmlReporter:
    def __init__(self, config: AggregatorConfig) -> None:
        self._config = config

    def render(self, posts: Sequence[Post], filename: Optional[str] = None) -> str:
        output_path = self._resolve_output_path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._build_html(posts, output_path)
        output_path.write_text(content, encoding="utf-8")
        LOGGER.info("HTML report written to %s.", output_path)
        return str(output_path)

    def _resolve_output_path(self, filename: Optional[str]) -> Path:
        reports_dir = self._absolute_path(self._config.reports_dir)
        if filename:
            filename_path = Path(filename)
            return (
                filename_path
                if filename_path.is_absolute()
                else reports_dir / filename_path
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return reports_dir / f"weekly_digest_ml_{timestamp}.html"

    def _absolute_path(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _sorttable_script_tag(self, output_path: Path) -> str:
        sorttable_path = self._absolute_path(self._config.sorttable_js_path)
        if not sorttable_path.exists():
            LOGGER.warning(
                "Sorttable script not found at %s; tables will not be sortable.",
                sorttable_path,
            )
            return ""
        try:
            relative_path = Path(os.path.relpath(sorttable_path, output_path.parent))
        except ValueError:
            relative_path = sorttable_path
        return f"<script src='{relative_path.as_posix()}'></script>"

    @staticmethod
    def _apply_source_table_class(table_html: str) -> str:
        if "class=\"sortable\"" not in table_html:
            return table_html
        return table_html.replace(
            'class="sortable"', f'class="sortable {SOURCE_TABLE_CLASS}"', 1
        )

    def _build_html(self, posts: Sequence[Post], output_path: Path) -> str:
        grouped = self._group_by_source(posts)
        average_ml_score = (
            sum(float(post.get("ml_score", 0) or 0) for post in posts) / len(posts)
            if posts
            else 0.0
        )

        sections = self._build_sections()

        sections_html = []
        for source_key, title, columns in sections:
            items = grouped.get(source_key, [])
            sections_html.append(
                f"<h2 class='source-header'>{title}</h2>"
                + (
                    self._apply_source_table_class(
                        build_html_table(items, columns)
                    )
                    if items
                    else "<i>No posts collected for this source this week.</i>"
                )
            )

        script_block = self._sorttable_script_tag(output_path)

        now = datetime.now()
        filtering_mode = (
            "ML classifier"
            if posts and posts[0].get("ml_score") is not None
            else "keyword rules only"
        )

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Weekly Technology Digest (ML)</title>
            {script_block}
            <style>
            body {{ font-family: Arial, sans-serif; background: #f4f4f4; }}
            .container {{ background: #fff; max-width: 1150px; margin: 40px auto; border-radius:8px; box-shadow: 0 4px 20px #aaa5; padding: 24px; overflow-x:auto; }}
            h1, h2 {{ color: #007acc; }}
            h2.source-header {{ margin-top: 2em; color: #314b5f; }}
            .stats {{ margin: 12px 0 30px 0; font-size:18px; }}
            .ml-stats {{ padding: 12px 0 12px 20px; border-left: 4px solid #007acc; margin: 18px 0; background: #ecf8ff; }}
            table.sortable {{ border-collapse: collapse; margin:14px 0; }}
            .{SOURCE_TABLE_CLASS} {{ width: {SOURCE_TABLE_WIDTH}; }}
            th, td {{ border:1px solid #eee; padding:6px 12px; font-size:15px; text-align:left; white-space:nowrap; }}
            th {{ background:#e6ecf3; cursor: pointer; }}
            tr:hover {{ background:#f3fdff; }}
            .ml-score-badge {{ background: #ff6b35; color: white; border-radius:4px; padding:2px 7px; font-size:13px; }}
            .footer {{ margin-top:40px; color:#888;text-align:center; }}
            </style>
        </head>
        <body>
        <div class="container">
        <h1>Weekly Technology Digest</h1>
        <div class='stats'>
            <b>Reddit:</b> {len(grouped.get("reddit", []))} |
            <b>Hacker News:</b> {len(grouped.get("hackernews", []))} |
            <b>TED:</b> {len(grouped.get("ted_youtube", []))} |
            <b>Total:</b> {len(posts)}
            <br>
            <b>Report generated:</b> {now.strftime(self._config.html_timestamp_format)}
        </div>
        <div class='ml-stats'>
            <b>Average ML relevance score:</b> {average_ml_score:.2f}
        </div>
        {"".join(sections_html)}
        <div class="footer">
            <p>Digest created: {now.strftime(self._config.html_footer_format)}</p>
            <p>Sources: Reddit, Hacker News, TED Talks (YouTube)</p>
            <p>Filtering mode: {filtering_mode}</p>
            <p>Report path: {output_path}</p>
        </div>
        </div></body></html>
        """

    @staticmethod
    def _group_by_source(posts: Sequence[Post]) -> Dict[str, List[Post]]:
        grouped: Dict[str, List[Post]] = {
            "reddit": [],
            "hackernews": [],
            "ted_youtube": [],
        }
        for post in posts:
            source = post.get("source")
            if source in grouped:
                grouped[source].append(post)
        return grouped

    def _build_sections(
        self,
    ) -> Sequence[Tuple[str, str, Sequence[Tuple[str, Callable[[Post], str]]]]]:
        timestamp_fmt = self._config.html_timestamp_format
        return (
            (
                "reddit",
                "Reddit",
                (
                    ("Title", make_link_cell),
                    (
                        "Date",
                        lambda post: format_datetime(
                            post.get("created_utc"), timestamp_fmt
                        ),
                    ),
                    ("Ups", lambda post: escape_field(post.get("score", 0))),
                    (
                        "Comms",
                        lambda post: escape_field(post.get("num_comments", 0)),
                    ),
                    (
                        "Subreddit",
                        lambda post: escape_field(post.get("subreddit", "")),
                    ),
                    ("Score", lambda post: ml_badge(post.get("ml_score"))),
                ),
            ),
            (
                "hackernews",
                "Hacker News",
                (
                    ("Title", make_link_cell),
                    (
                        "Date",
                        lambda post: format_datetime(
                            post.get("created_utc"), timestamp_fmt
                        ),
                    ),
                    ("Ups", lambda post: escape_field(post.get("score", 0))),
                    (
                        "Comms",
                        lambda post: escape_field(post.get("num_comments", 0)),
                    ),
                    ("Author", lambda post: escape_field(post.get("author", ""))),
                    ("Score", lambda post: ml_badge(post.get("ml_score"))),
                ),
            ),
            (
                "ted_youtube",
                "TED Talks",
                (
                    ("Title", make_link_cell),
                    (
                        "Date",
                        lambda post: format_datetime(
                            post.get("created_utc"), timestamp_fmt
                        ),
                    ),
                    (
                        "Duration",
                        lambda post: format_duration(post.get("duration")),
                    ),
                    ("Views", lambda post: escape_field(post.get("view_count", 0))),
                    ("Channel", lambda post: escape_field(post.get("channel", ""))),
                    ("Score", lambda post: ml_badge(post.get("ml_score"))),
                ),
            ),
        )
