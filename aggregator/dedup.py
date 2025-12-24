from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Iterable, List, Set

from .types import Post

LOGGER = logging.getLogger(__name__)

NORMALIZE_RE = re.compile(r"\s+")


def _normalized_text(post: Post) -> str:
    title = post.get("title", "") or ""
    body = post.get("selftext") or post.get("description") or ""
    text = f"{title} {body}".lower()
    text = re.sub(r"[^a-z0-9а-яё\s]", " ", text)
    return NORMALIZE_RE.sub(" ", text).strip()


def deduplicate_posts(posts: Iterable[Post], similarity_threshold: float) -> List[Post]:
    """Remove near-duplicate posts by fuzzy matching their normalized text."""
    posts_list = list(posts)
    unique_posts: List[Post] = []
    seen_texts: Set[str] = set()

    for post in posts_list:
        normalized = _normalized_text(post)
        if not normalized:
            continue

        is_duplicate = False
        for candidate in seen_texts:
            if (
                SequenceMatcher(None, normalized, candidate).ratio()
                >= similarity_threshold
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_posts.append(post)
            seen_texts.add(normalized)

    if posts_list:
        LOGGER.info(
            "Deduplicated posts: %d -> %d (threshold=%.2f)",
            len(posts_list),
            len(unique_posts),
            similarity_threshold,
        )

    return unique_posts
