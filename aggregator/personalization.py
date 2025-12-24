from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Sequence

from .config import AggregatorConfig
from .types import Post

LOGGER = logging.getLogger(__name__)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class UserProfile:
    name: str
    interests: Sequence[str]
    excluded_sources: Sequence[str]
    email: str
    digest_limit: int = 10


@dataclass
class PersonalizedDigest:
    profile: UserProfile
    posts: List[Post]


class PersonalizationManager:
    def __init__(self, config: AggregatorConfig) -> None:
        self._enabled = config.personalization_enabled
        self._path = config.user_profiles_path
        self._profiles: List[UserProfile] = []
        if self._enabled:
            self._load_profiles()

    def _load_profiles(self) -> None:
        path = self._path
        if not path.exists():
            LOGGER.warning("Personalization enabled but no profiles file at %s", path)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.exception("Failed to parse user profiles JSON: %s", path)
            return
        for entry in data:
            try:
                email = entry["email"]
                if not EMAIL_PATTERN.match(email):
                    raise ValueError("invalid email format")
                digest_limit = int(entry.get("digest_limit", 10))
                if digest_limit <= 0:
                    raise ValueError("digest_limit must be positive")
                profile = UserProfile(
                    name=entry["name"],
                    interests=tuple(
                        item.lower() for item in entry.get("interests", [])
                    ),
                    excluded_sources=tuple(entry.get("excluded_sources", [])),
                    email=email,
                    digest_limit=digest_limit,
                )
                self._profiles.append(profile)
            except (KeyError, ValueError) as exc:
                LOGGER.warning("Skipping invalid user profile entry: %s", exc)

    def personalized_digests(self, posts: Sequence[Post]) -> List[PersonalizedDigest]:
        if not self._enabled or not self._profiles:
            return []

        digests: List[PersonalizedDigest] = []
        for profile in self._profiles:
            matches: List[Post] = []
            for post in posts:
                source = post.get("source")
                if source in profile.excluded_sources:
                    continue
                text = f"{post.get('title', '')} {post.get('selftext') or post.get('description') or ''}".lower()
                if any(interest in text for interest in profile.interests):
                    matches.append(post)
                if len(matches) >= profile.digest_limit:
                    break
            if matches:
                digests.append(PersonalizedDigest(profile=profile, posts=matches))
        return digests
