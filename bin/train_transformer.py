from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregator.config import AggregatorConfig  # noqa
from aggregator.ml import TransformerConfig, TransformerNewsClassifier  # noqa
from aggregator.secrets import load_runtime_secrets  # noqa

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    load_runtime_secrets()
    configure_logging()
    LOGGER.info("Starting transformer classifier training utility.")

    config = AggregatorConfig()

    dataset_path = config.transformer_dataset_path
    model_dir = config.transformer_model_dir

    if not dataset_path.exists():
        LOGGER.error("Dataset file not found: %s", dataset_path)
        return 1

    try:
        transformer_cfg = TransformerConfig(
            model_name=config.transformer_model_name,
            model_dir=model_dir,
            base_dataset_path=dataset_path,
            feedback_dataset_path=config.feedback_dataset_path,
            feedback_weight=config.transformer_feedback_weight,
            max_length=config.transformer_max_length,
            learning_rate=config.transformer_learning_rate,
            batch_size=config.transformer_batch_size,
            num_epochs=config.transformer_num_epochs,
            weight_decay=config.transformer_weight_decay,
        )
        classifier = TransformerNewsClassifier(config=transformer_cfg)
        classifier.train(
            dataset_path=dataset_path,
            feedback_path=config.feedback_dataset_path,
            feedback_weight=config.transformer_feedback_weight,
        )
        LOGGER.info("Transformer model trained and saved to %s", model_dir)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Transformer training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
