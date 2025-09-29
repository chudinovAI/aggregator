from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class AggregatorConfig:
    collection_days: int = 7
    reddit_top_limit: int = 20
    hackernews_top_limit: int = 100
    ted_min_duration_seconds: int = 12 * 60
    top_posts_limit: int = 50
    ml_threshold: float = 0.6
    request_timeout_seconds: int = 10

    subreddits: Tuple[str, ...] = (
        "compsci",
        "computerscience",
        "csMajors",
        "datascience",
        "learnmachinelearning",
        "Python",
    )

    included_topics: Tuple[str, ...] = (
        "books",
        "textbook",
        "useful links",
        "c++",
        "cuda",
        "rust",
        "python",
        "machine learning",
        "data science",
        "artificial intelligence",
        "computer science",
        "ai",
        "computers",
        "software",
        "physics",
        "space",
        "science",
        "astronomy",
        "universe",
        "psychology",
        "brain",
        "education",
        "math",
        "investing",
        "business",
        "tutorial",
        "guide",
        "open source",
    )

    excluded_keywords: Tuple[str, ...] = (
        "lgbtq",
        "lgbt",
        "politics",
        "political",
        "war",
        "feminist",
        "feminism",
        "minority",
        "minorities",
        "transgender",
        "gay",
        "lesbian",
    )

    ted_topics: Tuple[str, ...] = (
        "artificial intelligence",
        "computers",
        "software",
        "physics",
        "space",
        "science",
        "astronomy",
        "universe",
        "personality",
        "memory",
        "psychology",
        "mental health",
        "brain",
        "sleep",
        "personal growth",
        "motivation",
        "mindfulness",
        "depression",
        "education",
        "math",
        "investing",
        "business",
    )

    ted_search_queries: Tuple[str, ...] = (
        "TED talks",
        "TED conference",
        "TEDx talks",
    )

    high_value_keywords: Tuple[str, ...] = (
        "tutorial",
        "guide",
        "research",
        "paper",
        "study",
        "breakthrough",
        "innovation",
        "open source",
        "free",
    )

    html_timestamp_format: str = "%Y-%m-%d %H:%M"
    html_footer_format: str = "%Y-%m-%d %H:%M:%S"
    reports_dir: Path = Path("docs")
    sorttable_js_path: Path = Path("static/sorttable.js")
    model_path: Path = Path("models/news_classifier.pkl")

    def lowercased_included_topics(self) -> Tuple[str, ...]:
        return tuple(topic.lower() for topic in self.included_topics)

    def lowercased_excluded_keywords(self) -> Tuple[str, ...]:
        return tuple(keyword.lower() for keyword in self.excluded_keywords)

    def lowercased_ted_topics(self) -> Tuple[str, ...]:
        return tuple(topic.lower() for topic in self.ted_topics)
