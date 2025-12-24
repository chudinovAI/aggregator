from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from email.utils import format_datetime as format_rfc2822
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring

from .config import AggregatorConfig
from .feedback import FeedbackMetrics
from .storage import StorageSummary
from .types import Post
from .validators import InputValidator
from .utils import (
    build_html_table,
    escape_field,
    format_datetime,
    format_duration,
    ml_badge,
    sentiment_badge,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TABLE_CLASS = "source-table"
SOURCE_TABLE_WIDTH = "100%"


class HtmlReporter:
    def __init__(self, config: AggregatorConfig) -> None:
        self._config = config
        self._validator = InputValidator(config.allowed_url_schemes)

    def render(
        self,
        posts: Sequence[Post],
        filename: Optional[str] = None,
        analytics: Optional[StorageSummary] = None,
        feedback: Optional[FeedbackMetrics] = None,
    ) -> str:
        output_path = self._resolve_output_path(filename, suffix="html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._build_html(
            posts, output_path, analytics=analytics, feedback=feedback
        )
        output_path.write_text(content, encoding="utf-8")
        LOGGER.info("HTML report written to %s.", output_path)
        return str(output_path)

    def render_json(
        self,
        posts: Sequence[Post],
        filename: Optional[str] = None,
        analytics: Optional[StorageSummary] = None,
        feedback: Optional[FeedbackMetrics] = None,
    ) -> str:
        output_path = self._resolve_output_path(filename, suffix="json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().strftime(self._config.html_timestamp_format),
            "count": len(posts),
            "posts": list(posts),
        }
        if analytics:
            payload["analytics"] = analytics.to_dict()
        if feedback:
            payload["feedback_metrics"] = feedback.to_dict()
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("JSON report written to %s.", output_path)
        return str(output_path)

    def render_markdown(
        self,
        posts: Sequence[Post],
        filename: Optional[str] = None,
        analytics: Optional[StorageSummary] = None,
        feedback: Optional[FeedbackMetrics] = None,
    ) -> str:
        output_path = self._resolve_output_path(filename, suffix="md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Weekly Technology Digest",
            "",
            f"Generated at: {datetime.now().strftime(self._config.html_timestamp_format)}",
            "",
            "| Source | Score | Sentiment | Title | URL |",
            "| --- | --- | --- | --- | --- |",
        ]
        for post in posts:
            title = escape_field(post.get("title", ""))
            url = escape_field(post.get("url", ""))
            source = escape_field(post.get("source", ""))
            score = f"{float(post.get('combined_score', 0.0)):.3f}"
            sent = sentiment_badge(
                post.get("sentiment_score"), post.get("sentiment_label")
            )
            lines.append(f"| {source} | {score} | {sent} | {title} | {url} |")
        if analytics:
            lines.extend(
                [
                    "",
                    f"## Trend analytics (last {analytics.window_days} days)",
                    f"- Total stored posts: {analytics.total_posts}",
                    "- Source breakdown:",
                ]
            )
            for source, count in analytics.source_breakdown.items():
                lines.append(f"  - {source}: {count}")
            if analytics.average_sentiment:
                lines.append("- Average sentiment:")
                for source, value in analytics.average_sentiment.items():
                    lines.append(f"  - {source}: {value:+.2f}")
        if feedback:
            lines.extend(
                [
                    "",
                    "## Feedback loop metrics",
                    f"- Processed labels: {feedback.processed}",
                    f"- Precision: {feedback.precision:.3f}"
                    if feedback.precision is not None
                    else "- Precision: n/a",
                    f"- Recall: {feedback.recall:.3f}"
                    if feedback.recall is not None
                    else "- Recall: n/a",
                ]
            )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        LOGGER.info("Markdown report written to %s.", output_path)
        return str(output_path)

    def render_rss(self, posts: Sequence[Post], filename: Optional[str] = None) -> str:
        output_path = self._resolve_output_path(filename, suffix="rss")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feed_bytes = self._build_rss(posts)
        output_path.write_bytes(feed_bytes)
        LOGGER.info("RSS feed written to %s.", output_path)
        return str(output_path)

    def _resolve_output_path(self, filename: Optional[str], suffix: str) -> Path:
        reports_dir = self._absolute_path(self._config.reports_dir)
        if filename:
            filename_path = Path(filename)
            return (
                filename_path.with_suffix(f".{suffix}")
                if filename_path.is_absolute()
                else reports_dir / filename_path.with_suffix(f".{suffix}")
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return reports_dir / f"weekly_digest_ml_{timestamp}.{suffix}"

    def _absolute_path(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _build_rss(self, posts: Sequence[Post]) -> bytes:
        now = datetime.now()
        rss = Element("rss", version="2.0")
        channel = SubElement(rss, "channel")
        SubElement(channel, "title").text = "Weekly Technology Digest"
        SubElement(channel, "link").text = "https://example.com"
        SubElement(
            channel, "description"
        ).text = "Technology news digest with ML relevance scoring."
        SubElement(channel, "pubDate").text = format_rfc2822(now)

        for post in posts:
            item = SubElement(channel, "item")
            SubElement(item, "title").text = post.get("title", "")
            SubElement(item, "link").text = post.get("url", "")
            summary = (post.get("selftext") or post.get("description") or "")[:500]
            SubElement(item, "description").text = summary
            created = post.get("created_utc") or now
            SubElement(item, "pubDate").text = format_rfc2822(created)
            guid_value = post.get("url") or post.get("title", "")
            SubElement(item, "guid").text = guid_value

        return tostring(rss, encoding="utf-8", xml_declaration=True)

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
        if 'class="sortable"' not in table_html:
            return table_html
        return table_html.replace(
            'class="sortable"', f'class="sortable {SOURCE_TABLE_CLASS}"', 1
        )

    def _build_feedback_section(self, feedback: Optional[FeedbackMetrics]) -> str:
        if not feedback:
            return ""
        precision = (
            f"{feedback.precision:.3f}" if feedback.precision is not None else "n/a"
        )
        recall = f"{feedback.recall:.3f}" if feedback.recall is not None else "n/a"
        return f"""
        <section class='analytics'>
            <h2>Feedback loop</h2>
            <p>Processed labels: <b>{feedback.processed}</b></p>
            <div class='analytics-grid'>
                <div>
                    <h3>Quality</h3>
                    <ul>
                        <li>Precision: {precision}</li>
                        <li>Recall: {recall}</li>
                        <li>True positives: {feedback.true_positive}</li>
                        <li>False positives: {feedback.false_positive}</li>
                        <li>False negatives: {feedback.false_negative}</li>
                    </ul>
                </div>
            </div>
        </section>
        """

    def _build_analytics_section(
        self, analytics: Optional[StorageSummary]
    ) -> Tuple[str, str, str]:
        if not analytics:
            return "", "", ""

        summary = analytics.to_dict()
        source_items = "".join(
            f"<li><b>{escape_field(source)}:</b> {count}</li>"
            for source, count in summary.get("source_breakdown", {}).items()
        )
        sentiment_items = "".join(
            f"<li><b>{escape_field(source)}:</b> {value:+.2f}</li>"
            for source, value in summary.get("average_sentiment", {}).items()
        )
        if not source_items:
            source_items = "<li>No stored posts in this window.</li>"
        if not sentiment_items:
            sentiment_items = "<li>Sentiment data unavailable.</li>"

        analytics_html = f"""
        <section class='analytics'>
            <h2>Trend analytics (last {summary.get("window_days", 0)} days)</h2>
            <p><b>Total stored posts:</b> {summary.get("total_posts", 0)}</p>
            <div class='analytics-grid'>
                <div>
                    <h3>Sources</h3>
                    <ul>{source_items}</ul>
                </div>
                <div>
                    <h3>Average sentiment</h3>
                    <ul>{sentiment_items}</ul>
                </div>
            </div>
            <div class='chart-grid'>
                <canvas id='trend-chart'></canvas>
                <canvas id='sentiment-chart'></canvas>
            </div>
        </section>
        """
        daily_counts = summary.get("daily_counts", [])
        sentiment_trend = summary.get("sentiment_trend", [])
        analytics_script = f"""
        <script>
        (() => {{
            const dailyCounts = {json.dumps(daily_counts)};
            const sentimentTrend = {json.dumps(sentiment_trend)};
            const trendCanvas = document.getElementById('trend-chart');
            const sentimentCanvas = document.getElementById('sentiment-chart');
            if (trendCanvas && window.Chart) {{
                new Chart(trendCanvas, {{
                    type: 'line',
                    data: {{
                        labels: dailyCounts.map(item => item.date),
                        datasets: [{{
                            label: 'Posts per day',
                            data: dailyCounts.map(item => item.count),
                            borderColor: '#0ea5e9',
                            backgroundColor: 'rgba(14,165,233,0.15)',
                            tension: 0.3,
                            fill: true,
                        }}],
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                ticks: {{ precision: 0 }},
                            }},
                        }},
                    }},
                }});
            }}
            if (sentimentCanvas && window.Chart) {{
                new Chart(sentimentCanvas, {{
                    type: 'line',
                    data: {{
                        labels: sentimentTrend.map(item => item.date),
                        datasets: [{{
                            label: 'Average sentiment',
                            data: sentimentTrend.map(item => item.sentiment),
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16,185,129,0.15)',
                            tension: 0.3,
                            fill: true,
                        }}],
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{
                                suggestedMin: -1,
                                suggestedMax: 1,
                            }},
                        }},
                    }},
                }});
            }}
        }})();
        </script>
        """
        chart_lib = "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
        return analytics_html, analytics_script, chart_lib

    def _build_html(
        self,
        posts: Sequence[Post],
        output_path: Path,
        analytics: Optional[StorageSummary] = None,
        feedback: Optional[FeedbackMetrics] = None,
    ) -> str:
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
                    self._apply_source_table_class(build_html_table(items, columns))
                    if items
                    else "<i>No posts collected for this source this week.</i>"
                )
            )

        analytics_html, analytics_script, analytics_lib = self._build_analytics_section(
            analytics
        )
        feedback_html = self._build_feedback_section(feedback)
        script_block = self._sorttable_script_tag(output_path)

        now = datetime.now()
        filtering_mode = (
            "ML classifier"
            if posts and posts[0].get("ml_score") is not None
            else "keyword rules only"
        )

        filter_script = """
        <script>
        (() => {
            const searchBox = document.getElementById('search-box');
            const themeToggle = document.getElementById('theme-toggle');
            const body = document.body;
            const filterRows = () => {
                const query = (searchBox?.value || '').toLowerCase();
                document.querySelectorAll('table.sortable tbody tr').forEach(row => {
                    const text = row.innerText.toLowerCase();
                    row.style.display = text.includes(query) ? '' : 'none';
                });
            };
            searchBox?.addEventListener('input', filterRows);
            themeToggle?.addEventListener('click', () => {
                body.classList.toggle('dark');
            });
        })();
        </script>
        """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="UTF-8">
        <title>Weekly Technology Digest (ML)</title>
            {script_block}
            {analytics_lib}
            <style>
            body {{ font-family: Arial, sans-serif; background: #f4f4f4; transition: background 0.2s, color 0.2s; }}
            body.dark {{ background: #0f172a; color: #e2e8f0; }}
            body.dark .container {{ background: #0b1221; box-shadow: 0 4px 20px #0008; }}
            body.dark h1, body.dark h2 {{ color: #7dd3fc; }}
            body.dark table {{ color: #e2e8f0; }}
            body.dark th {{ background: #1e293b; }}
            body.dark td {{ border-color: #1e293b; }}
            .container {{ background: #fff; max-width: 1150px; margin: 40px auto; border-radius:8px; box-shadow: 0 4px 20px #aaa5; padding: 24px; overflow-x:auto; }}
            h1, h2 {{ color: #007acc; }}
            h2.source-header {{ margin-top: 2em; color: #314b5f; }}
            .header {{ display:flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }}
            .controls {{ display:flex; gap:10px; align-items:center; }}
            .controls input {{ padding:6px 10px; border:1px solid #cbd5e1; border-radius:6px; min-width: 220px; }}
            .controls button {{ padding:6px 10px; background:#0ea5e9; color:#fff; border:none; border-radius:6px; cursor:pointer; }}
            .controls button:hover {{ background:#0284c7; }}
            .stats {{ margin: 12px 0 30px 0; font-size:18px; }}
            .ml-stats {{ padding: 12px 0 12px 20px; border-left: 4px solid #007acc; margin: 18px 0; background: #ecf8ff; }}
            table.sortable {{ border-collapse: collapse; margin:14px 0; }}
            .{SOURCE_TABLE_CLASS} {{ width: {SOURCE_TABLE_WIDTH}; }}
            th, td {{ border:1px solid #eee; padding:6px 12px; font-size:15px; text-align:left; white-space:nowrap; }}
            th {{ background:#e6ecf3; cursor: pointer; }}
            tr:hover {{ background:#f3fdff; }}
            .ml-score-badge {{ background: #ff6b35; color: white; border-radius:4px; padding:2px 7px; font-size:13px; }}
            .analytics {{ margin: 32px 0; padding: 20px; border-radius: 10px; background: #e0f2fe; border: 1px solid #bae6fd; }}
            .analytics h2 {{ margin-top: 0; }}
            .analytics-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
            .analytics-grid ul {{ list-style:none; padding-left:0; margin: 0; }}
            .chart-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-top: 20px; }}
            .chart-grid canvas {{ width: 100%; min-height: 260px; background: #fff; border-radius: 8px; padding: 10px; }}
            .footer {{ margin-top:40px; color:#888;text-align:center; }}
            </style>
        </head>
        <body>
        <div class="container">
            <div class="header">
                <h1>Weekly Technology Digest</h1>
                <div class="controls">
                    <input id="search-box" type="search" placeholder="Search posts..." />
                    <button id="theme-toggle" type="button">Тёмная тема</button>
                </div>
            </div>
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
            {feedback_html}
            {analytics_html}
            {"".join(sections_html)}
            <div class="footer">
                <p>Digest created: {now.strftime(self._config.html_footer_format)}</p>
                <p>Sources: Reddit, Hacker News, TED Talks (YouTube)</p>
                <p>Filtering mode: {filtering_mode}</p>
                <p>Report path: {output_path}</p>
            </div>
        </div>
        {filter_script}
        {analytics_script}
        </body></html>
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
        link_renderer = self._link_cell
        return (
            (
                "reddit",
                "Reddit",
                (
                    ("Title", link_renderer),
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
                    (
                        "Sentiment",
                        lambda post: sentiment_badge(
                            post.get("sentiment_score"), post.get("sentiment_label")
                        ),
                    ),
                ),
            ),
            (
                "hackernews",
                "Hacker News",
                (
                    ("Title", link_renderer),
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
                    (
                        "Sentiment",
                        lambda post: sentiment_badge(
                            post.get("sentiment_score"), post.get("sentiment_label")
                        ),
                    ),
                ),
            ),
            (
                "ted_youtube",
                "TED Talks",
                (
                    ("Title", link_renderer),
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
                    (
                        "Sentiment",
                        lambda post: sentiment_badge(
                            post.get("sentiment_score"), post.get("sentiment_label")
                        ),
                    ),
                ),
            ),
        )

    def _link_cell(self, post: Post) -> str:
        title = escape_field(post.get("title", "") or "")
        url = post.get("url", "") or ""
        if not url or not self._validator.is_safe_url(url):
            return f"<span>{title}</span>"
        safe_url = escape_field(url)
        return f"<a href='{safe_url}' rel='noopener noreferrer'>{title}</a>"
