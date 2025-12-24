from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .config import AggregatorConfig
from .types import Post

LOGGER = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    payloads: List[Dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping invalid feedback JSON line.")
    return payloads


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(serialized + ("\n" if serialized else ""), encoding="utf-8")


@dataclass
class FeedbackMetrics:
    processed: int
    positive: int
    negative: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: Optional[float]
    recall: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "processed": self.processed,
            "positive": self.positive,
            "negative": self.negative,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
        }


class FeedbackLoop:
    """Stores posts for annotation and folds labels into a training dataset."""

    def __init__(self, config: AggregatorConfig) -> None:
        self._queue_path = config.feedback_queue_path
        self._events_path = config.feedback_events_path
        self._dataset_path = config.feedback_dataset_path
        self._min_batch = config.feedback_min_training_batch

    def enqueue(self, posts: Sequence[Post], *, predicted: bool) -> None:
        queue = self._load_queue()
        new_entries = 0
        for post in posts:
            post_id = self._make_post_id(post)
            if post_id in queue:
                continue
            queue[post_id] = {
                "post_id": post_id,
                "predicted": predicted,
                "title": post.get("title", ""),
                "url": post.get("url", ""),
                "source": post.get("source", "unknown"),
                "snapshot": post,
            }
            new_entries += 1
        if new_entries:
            self._save_queue(queue)
            LOGGER.info(
                "Queued %d posts for feedback (predicted=%s).", new_entries, predicted
            )

    def integrate_feedback(self) -> Optional[FeedbackMetrics]:
        events = _read_jsonl(self._events_path)
        if not events:
            return None
        queue = self._load_queue()
        processed_entries: List[Dict] = []
        remaining_events: List[Dict] = []
        for event in events:
            post_id = event.get("post_id")
            label = self._normalize_label(event.get("label"))
            if not post_id or label is None:
                remaining_events.append(event)
                continue
            queued = queue.pop(post_id, None)
            if not queued:
                LOGGER.debug("Feedback refers to unknown post_id=%s", post_id)
                continue
            processed_entries.append(
                {
                    "post_id": post_id,
                    "label": label,
                    "predicted": bool(queued.get("predicted")),
                    "post": queued.get("snapshot"),
                }
            )

        if not processed_entries:
            _write_jsonl(self._events_path, remaining_events)
            self._save_queue(queue)
            return None

        self._append_dataset(processed_entries)
        metrics = self._calculate_metrics(processed_entries)
        LOGGER.info(
            "Feedback processed: processed=%d precision=%s recall=%s",
            metrics.processed,
            f"{metrics.precision:.3f}" if metrics.precision is not None else "n/a",
            f"{metrics.recall:.3f}" if metrics.recall is not None else "n/a",
        )
        _write_jsonl(self._events_path, remaining_events)
        self._save_queue(queue)
        return metrics

    def dataset_size(self) -> int:
        dataset = _read_jsonl(self._dataset_path)
        return len(dataset)

    def _append_dataset(self, entries: Sequence[Dict]) -> None:
        existing = _read_jsonl(self._dataset_path)
        serialized = [
            {
                "post_id": entry["post_id"],
                "label": entry["label"],
                "predicted": entry["predicted"],
                "title": entry["post"].get("title"),
                "text": entry["post"].get("selftext")
                or entry["post"].get("description"),
            }
            for entry in entries
        ]
        combined = existing + serialized
        _write_jsonl(self._dataset_path, combined)
        if len(combined) >= self._min_batch:
            LOGGER.info(
                "Feedback dataset size ready for training: %d samples.",
                len(combined),
            )

    def _load_queue(self) -> Dict[str, Dict]:
        queue_entries = _read_jsonl(self._queue_path)
        return {
            entry["post_id"]: entry for entry in queue_entries if "post_id" in entry
        }

    def _save_queue(self, queue: Dict[str, Dict]) -> None:
        _write_jsonl(self._queue_path, queue.values())

    @staticmethod
    def _make_post_id(post: Post) -> str:
        basis = (post.get("url") or "") + (post.get("title") or "")
        return sha1(basis.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_label(label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        normalized = label.strip().lower()
        if normalized in {"relevant", "positive", "pos", "1", "true"}:
            return "relevant"
        if normalized in {"irrelevant", "negative", "neg", "0", "false"}:
            return "irrelevant"
        return None

    @staticmethod
    def _calculate_metrics(entries: Sequence[Dict]) -> FeedbackMetrics:
        true_positive = sum(
            1
            for entry in entries
            if entry["predicted"] and entry["label"] == "relevant"
        )
        false_positive = sum(
            1
            for entry in entries
            if entry["predicted"] and entry["label"] != "relevant"
        )
        false_negative = sum(
            1
            for entry in entries
            if (not entry["predicted"]) and entry["label"] == "relevant"
        )
        positive = true_positive
        negative = false_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else None
        )
        return FeedbackMetrics(
            processed=len(entries),
            positive=positive,
            negative=negative,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=precision,
            recall=recall,
        )
