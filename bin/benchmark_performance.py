#!/usr/bin/env python3
"""
Performance benchmark script for the news aggregator.

This script benchmarks the key performance-critical functions:
1. TextClassifier.batch_predict - ML inference
2. TextVectorizer.transform - Text vectorization

Usage:
    python -m bin.benchmark_performance [--iterations N] [--batch-size N]

Example:
    python -m bin.benchmark_performance --iterations 50 --batch-size 100
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.classifier import ClassifierConfig, TextClassifier, TrainingDataset
from src.ml.vectorizer import TextVectorizer
from src.utils.profiling import PerformanceStats, run_benchmark

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)


# Sample texts for benchmarking
SAMPLE_TEXTS = [
    "Breaking: Major tech company announces new AI breakthrough",
    "Scientists discover new species in the Amazon rainforest",
    "Stock market reaches all-time high amid economic recovery",
    "Local community organizes charity event for homeless shelter",
    "New study reveals benefits of Mediterranean diet",
    "SpaceX successfully launches satellites into orbit",
    "Climate change report warns of rising sea levels",
    "Cybersecurity experts warn of new phishing attacks",
    "Electric vehicle sales surge as prices drop",
    "Researchers develop new cancer treatment method",
]


def generate_test_texts(n: int) -> list[str]:
    """Generate n test texts by cycling through samples."""
    return [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] + f" (variant {i})" for i in range(n)]


def generate_training_data(n: int) -> TrainingDataset:
    """Generate synthetic training data."""
    texts = generate_test_texts(n)
    labels = ["interesting" if i % 2 == 0 else "boring" for i in range(n)]
    return TrainingDataset(texts=texts, labels=labels)


def benchmark_vectorizer_tfidf(batch_size: int, iterations: int) -> PerformanceStats:
    """Benchmark TF-IDF vectorizer transform."""
    texts = generate_test_texts(batch_size)
    training_texts = generate_test_texts(500)

    vectorizer = TextVectorizer(vectorizer_type="tfidf", max_features=5000)
    vectorizer.fit_transform(training_texts)

    gc.collect()

    def transform_batch():
        vectorizer.reset_cache()  # Clear cache to measure actual transform
        return vectorizer.transform(texts)

    stats = run_benchmark(transform_batch, iterations=iterations, warmup=5)
    return stats


def benchmark_vectorizer_cached(batch_size: int, iterations: int) -> PerformanceStats:
    """Benchmark TF-IDF vectorizer with caching."""
    texts = generate_test_texts(batch_size)
    training_texts = generate_test_texts(500)

    vectorizer = TextVectorizer(vectorizer_type="tfidf", max_features=5000)
    vectorizer.fit_transform(training_texts)
    vectorizer.transform(texts)  # Populate cache

    gc.collect()

    def transform_cached():
        return vectorizer.transform(texts)

    stats = run_benchmark(transform_cached, iterations=iterations, warmup=5)
    return stats


def benchmark_classifier_batch(batch_size: int, iterations: int) -> PerformanceStats | None:
    """Benchmark classifier batch_predict."""
    import tempfile

    texts = generate_test_texts(batch_size)
    training_data = generate_training_data(500)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        config = ClassifierConfig(model_path=model_path, vectorizer_type="tfidf")
        classifier = TextClassifier(config)

        # Train the classifier
        classifier._train_sync(training_data)

        gc.collect()

        def batch_predict():
            return classifier.batch_predict(texts)

        stats = run_benchmark(batch_predict, iterations=iterations, warmup=5)
        return stats


def benchmark_row_to_predictions(
    batch_size: int, iterations: int
) -> tuple[PerformanceStats, PerformanceStats]:
    """Compare original vs optimized _row_to_predictions."""
    import tempfile

    training_data = generate_training_data(500)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        config = ClassifierConfig(model_path=model_path, vectorizer_type="tfidf")
        classifier = TextClassifier(config)
        classifier._train_sync(training_data)

        # Generate fake probability matrix
        n_classes = len(classifier._model.classes_) if classifier._model else 2
        probabilities = np.random.rand(batch_size, n_classes)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        gc.collect()

        # Benchmark individual row processing (old way)
        def old_row_by_row():
            return [classifier._row_to_predictions(row) for row in probabilities]

        # Benchmark batch processing (new way)
        def new_batch():
            return classifier._batch_rows_to_predictions(probabilities)

        old_stats = run_benchmark(old_row_by_row, iterations=iterations, warmup=5)
        new_stats = run_benchmark(new_batch, iterations=iterations, warmup=5)

        return old_stats, new_stats


def print_stats(name: str, stats: PerformanceStats) -> None:
    """Print formatted statistics."""
    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f"{'=' * 60}")
    print(f"  Iterations: {stats.total_calls}")
    print(f"  Total time: {stats.total_time_sec:.4f}s")
    print(f"  Avg time:   {stats.avg_time_sec * 1000:.3f}ms")
    print(f"  Min time:   {stats.min_time_sec * 1000:.3f}ms")
    print(f"  Max time:   {stats.max_time_sec * 1000:.3f}ms")
    print(f"  Std dev:    {stats.std_dev_sec * 1000:.3f}ms")
    print(f"  Throughput: {1 / stats.avg_time_sec:.1f} ops/sec")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark aggregator performance")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for predictions (default: 100)",
    )
    args = parser.parse_args()

    print(f"\nRunning benchmarks with {args.iterations} iterations, batch size {args.batch_size}")
    print("=" * 60)

    # Vectorizer benchmarks
    print("\n[1/4] Benchmarking TF-IDF vectorizer (no cache)...")
    vectorizer_stats = benchmark_vectorizer_tfidf(args.batch_size, args.iterations)
    print_stats("TF-IDF Vectorizer (no cache)", vectorizer_stats)

    print("\n[2/4] Benchmarking TF-IDF vectorizer (cached)...")
    cached_stats = benchmark_vectorizer_cached(args.batch_size, args.iterations)
    print_stats("TF-IDF Vectorizer (cached)", cached_stats)

    cache_speedup = vectorizer_stats.avg_time_sec / cached_stats.avg_time_sec
    print(f"\n  Cache speedup: {cache_speedup:.1f}x")

    # Classifier benchmark
    print("\n[3/4] Benchmarking classifier batch_predict...")
    classifier_stats = benchmark_classifier_batch(args.batch_size, args.iterations)
    if classifier_stats:
        print_stats("Classifier batch_predict", classifier_stats)
        texts_per_sec = args.batch_size / classifier_stats.avg_time_sec
        print(f"\n  Throughput: {texts_per_sec:.0f} texts/sec")

    # Row-to-predictions comparison
    print("\n[4/4] Comparing _row_to_predictions implementations...")
    old_stats, new_stats = benchmark_row_to_predictions(args.batch_size, args.iterations)
    print_stats("Row-by-row processing (original)", old_stats)
    print_stats("Batch processing (optimized)", new_stats)

    improvement = old_stats.avg_time_sec / new_stats.avg_time_sec
    print(f"\n  Optimization speedup: {improvement:.2f}x")
    print(
        f"  Time saved per batch: {(old_stats.avg_time_sec - new_stats.avg_time_sec) * 1000:.3f}ms"
    )

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  TF-IDF transform (no cache): {vectorizer_stats.avg_time_sec * 1000:.3f}ms")
    print(f"  TF-IDF transform (cached):   {cached_stats.avg_time_sec * 1000:.3f}ms")
    if classifier_stats:
        print(f"  Classifier batch_predict:    {classifier_stats.avg_time_sec * 1000:.3f}ms")
    print(f"  Row processing (old):        {old_stats.avg_time_sec * 1000:.3f}ms")
    print(f"  Row processing (new):        {new_stats.avg_time_sec * 1000:.3f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
