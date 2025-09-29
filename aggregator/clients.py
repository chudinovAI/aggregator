import os
from typing import Any

import praw
from googleapiclient.discovery import build


def build_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="AdvancedNewsAggregator/2.0",
    )


def build_youtube_client() -> Any:
    return build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
