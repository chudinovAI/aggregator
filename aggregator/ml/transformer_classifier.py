from __future__ import annotations

import logging
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import torch
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "news_transformer"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "news_classification_dataset.csv"


class NewsDataset(Dataset):
    """Torch dataset for text classification."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-multilingual-cased"
    model_dir: Path = DEFAULT_MODEL_DIR
    base_dataset_path: Path = DEFAULT_DATASET_PATH
    feedback_dataset_path: Path | None = None
    feedback_weight: float = 2.0
    max_length: int = 256
    learning_rate: float = 3e-5
    batch_size: int = 8
    num_epochs: int = 2
    weight_decay: float = 0.01
    metrics_path: Path | None = None


class TransformerNewsClassifier:
    """Transformer-based news relevance classifier with training and inference."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        model_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or TransformerConfig()
        if model_path:
            self.config.model_dir = Path(model_path)
        if not self.config.metrics_path:
            self.config.metrics_path = self.config.model_dir / "training_metrics.jsonl"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.config.model_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_dir
            ).to(self.device)
            self.model.eval()
            LOGGER.info("Loaded transformer model from %s", self.config.model_dir)

    def train(
        self,
        dataset_path: Path | str | None = None,
        feedback_path: Path | None = None,
        feedback_weight: float | None = None,
    ) -> None:
        dataset_path = (
            Path(dataset_path) if dataset_path else Path(self.config.base_dataset_path)
        )
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        if "text" not in df or "label" not in df:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")

        feedback_path_value = feedback_path or self.config.feedback_dataset_path
        feedback_path_obj = Path(feedback_path_value) if feedback_path_value else None
        combined_df = df
        feedback_weight = feedback_weight or self.config.feedback_weight
        feedback_df = pd.DataFrame()
        if feedback_path_obj and feedback_path_obj.exists():
            feedback_df = self._load_feedback_dataframe(feedback_path_obj)
            if not feedback_df.empty:
                combined_df = self._augment_with_feedback(
                    df, feedback_df, weight=max(feedback_weight, 0.1)
                )
                LOGGER.info(
                    "Augmented transformer dataset with %d feedback samples (weight=%.2f).",
                    len(feedback_df),
                    feedback_weight,
                )
        else:
            LOGGER.info(
                "Feedback dataset not found at %s; training on base data only.",
                feedback_path_obj,
            )

        combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(
            drop=True
        )
        split = int(len(combined_df) * 0.85)
        train_df, eval_df = combined_df.iloc[:split], combined_df.iloc[split:]

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=2
        ).to(self.device)

        train_ds = NewsDataset(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            self.tokenizer,
            self.config.max_length,
        )
        eval_ds = NewsDataset(
            eval_df["text"].tolist(),
            eval_df["label"].tolist(),
            self.tokenizer,
            self.config.max_length,
        )

        training_args = TrainingArguments(
            output_dir=self.config.model_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )

        train_output = trainer.train()
        train_metrics = train_output.metrics or {}
        eval_metrics = trainer.evaluate()
        self.model.save_pretrained(self.config.model_dir)
        self.tokenizer.save_pretrained(self.config.model_dir)
        self.model.eval()
        self._record_training_metrics(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            train_samples=len(train_df),
            eval_samples=len(eval_df),
            feedback_samples=len(feedback_df),
        )
        LOGGER.info("Transformer model trained and saved to %s", self.config.model_dir)

    def predict_proba(self, texts: Sequence[str]) -> List[float]:
        if self.model is None or self.tokenizer is None:
            self._load_if_exists()
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not loaded or trained.")

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = softmax(logits, dim=1)[:, 1].cpu().tolist()
        return probs

    def filter_posts(self, posts: Iterable[dict], threshold: float = 0.6) -> List[dict]:
        texts = []
        post_refs = []
        for post in posts:
            title = post.get("title", "")
            content = post.get("selftext", "") or post.get("description", "")
            texts.append(f"{title} {content}")
            post_refs.append(post)

        probabilities = self.predict_proba(texts)
        relevant: List[dict] = []
        for post, prob in zip(post_refs, probabilities):
            if prob >= threshold:
                post_copy = post.copy()
                post_copy["ml_score"] = float(prob)
                relevant.append(post_copy)

        relevant.sort(key=lambda entry: entry["ml_score"], reverse=True)
        return relevant

    @staticmethod
    def _load_feedback_dataframe(path: Path) -> pd.DataFrame:
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (payload.get("text") or "")[:4096]
                title = payload.get("title") or ""
                combined = f"{title} {text}".strip()
                if not combined:
                    continue
                label_value = payload.get("label", "")
                label = (
                    1
                    if str(label_value).lower()
                    in {"1", "true", "relevant", "positive", "pos"}
                    else 0
                )
                records.append({"text": combined, "label": label})
        return pd.DataFrame(records)

    @staticmethod
    def _augment_with_feedback(
        base_df: pd.DataFrame, feedback_df: pd.DataFrame, weight: float
    ) -> pd.DataFrame:
        if feedback_df.empty or weight <= 0:
            return base_df
        frames = [base_df]
        if weight <= 1:
            sample = TransformerNewsClassifier._sample_rows(
                feedback_df, max(weight, 0.1)
            )
            frames.append(sample)
        else:
            repeat = int(math.floor(weight))
            fractional = max(0.0, weight - repeat)
            for _ in range(repeat):
                frames.append(feedback_df)
            if fractional > 0:
                frames.append(
                    TransformerNewsClassifier._sample_rows(feedback_df, fractional)
                )
        combined = pd.concat(frames, ignore_index=True)
        return combined

    @staticmethod
    def _sample_rows(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
        if df.empty or fraction <= 0:
            return df.head(0)
        fraction = min(fraction, 1.0)
        sample_size = max(1, int(math.ceil(fraction * len(df))))
        sample_size = min(sample_size, len(df))
        return df.sample(n=sample_size, replace=False, random_state=42)

    def _record_training_metrics(
        self,
        *,
        train_metrics: dict,
        eval_metrics: dict,
        train_samples: int,
        eval_samples: int,
        feedback_samples: int,
    ) -> None:
        metrics_path = self.config.metrics_path
        if not metrics_path:
            return
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "train_loss": train_metrics.get("train_loss"),
            "train_runtime": train_metrics.get("train_runtime"),
            "eval_loss": eval_metrics.get("eval_loss"),
            "eval_accuracy": eval_metrics.get("eval_accuracy"),
            "eval_f1": eval_metrics.get("eval_f1"),
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "feedback_samples": feedback_samples,
            "model_dir": str(self.config.model_dir),
            "model_name": self.config.model_name,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "weight_decay": self.config.weight_decay,
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        train_loss = (
            record["train_loss"] if record["train_loss"] is not None else float("nan")
        )
        eval_loss = (
            record["eval_loss"] if record["eval_loss"] is not None else float("nan")
        )
        LOGGER.info(
            "Training metrics recorded to %s (train_loss=%.4f eval_loss=%.4f).",
            metrics_path,
            train_loss,
            eval_loss,
        )
