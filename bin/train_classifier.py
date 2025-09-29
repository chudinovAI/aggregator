from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregator.ml import NewsClassifier  # noqa

LOGGER = logging.getLogger(__name__)

DATASET_FILE = PROJECT_ROOT / "data" / "news_classification_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "news_classifier.pkl"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    configure_logging()
    LOGGER.info("Starting classifier training utility.")

    if not DATASET_FILE.exists():
        LOGGER.error("Dataset file not found: %s", DATASET_FILE)
        LOGGER.error("Ensure the dataset is available in the data/ directory.")
        return 1

    try:
        classifier = NewsClassifier(model_path=MODEL_PATH)
        classifier.train_model(dataset_path=DATASET_FILE)

        LOGGER.info("Running sanity-check predictions on sample phrases.")
        samples = [
            "New Python 3.12 features for data science and machine learning",
            "CUDA programming tutorial for GPU acceleration in deep learning",
            "Best computer science textbooks for algorithms and data structures",
            "Trump election results and political implications",
            "LGBTQ rights movement in modern society",
            "Quantum computing breakthrough in IBM research lab",
            "War in Ukraine latest military conflict updates",
            "Open source machine learning frameworks comparison guide",
            "Feminist theory in workplace technology industry",
            "TED talk about artificial intelligence and future of humanity",
        ]

        for index, text in enumerate(samples, start=1):
            is_relevant, probability = classifier.predict_relevance(text)
            label = "relevant" if is_relevant else "irrelevant"
            LOGGER.info("%02d. %s (%.3f) -> %s", index, text, probability, label)

        LOGGER.info("Training finished successfully. The model is ready for use.")
        LOGGER.info(
            "Run `python bin/news_aggregator.py` to generate the weekly digest."
        )
        return 0

    except Exception as exc:
        LOGGER.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
