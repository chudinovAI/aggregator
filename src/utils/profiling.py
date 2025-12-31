"""
Performance profiling utilities with support for sync and async functions.

This module provides:
- Decorators for profiling sync/async functions
- Memory usage tracking
- Statistical aggregation of profiling results
- Benchmark runner for comparative analysis

Example usage:
    from src.utils.profiling import profile_sync, profile_async, run_benchmark

    @profile_sync(print_stats=True)
    def my_function():
        ...

    @profile_async(print_stats=True)
    async def my_async_function():
        ...

    # Run comparative benchmark
    results = run_benchmark(
        my_function,
        iterations=100,
        warmup=10,
    )
"""

from __future__ import annotations

import cProfile
import functools
import gc
import io
import logging
import pstats
import sys
import time
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

LOGGER = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(slots=True)
class PerformanceStats:
    """
    Statistical summary of performance measurements.
    """

    function_name: str
    total_calls: int
    total_time_sec: float
    avg_time_sec: float
    min_time_sec: float
    max_time_sec: float
    std_dev_sec: float
    memory_peak_mb: float
    timings: list[float] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        return (
            f"PerformanceStats({self.function_name}):\n"
            f"  Calls: {self.total_calls}\n"
            f"  Total: {self.total_time_sec:.4f}s\n"
            f"  Avg: {self.avg_time_sec:.6f}s\n"
            f"  Min: {self.min_time_sec:.6f}s\n"
            f"  Max: {self.max_time_sec:.6f}s\n"
            f"  StdDev: {self.std_dev_sec:.6f}s\n"
            f"  Memory Peak: {self.memory_peak_mb:.2f}MB"
        )


@dataclass(slots=True)
class ProfileResult:
    """
    Single profiling run result.
    """

    elapsed_sec: float
    memory_mb: float
    cprofile_stats: pstats.Stats | None = None


class AsyncProfiler:
    """
    Context manager for profiling async code blocks.

    Usage:
        async with AsyncProfiler("my_operation") as profiler:
            await some_async_operation()

        print(profiler.elapsed_sec)
    """

    def __init__(self, name: str = "unnamed", *, enable_cprofile: bool = False) -> None:
        self.name = name
        self.enable_cprofile = enable_cprofile
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._profiler: cProfile.Profile | None = None
        self._stats: pstats.Stats | None = None

    @property
    def elapsed_sec(self) -> float:
        """Return elapsed wall-clock time in seconds."""
        return self._end_time - self._start_time

    @property
    def stats(self) -> pstats.Stats | None:
        """Return cProfile stats if enabled."""
        return self._stats

    async def __aenter__(self) -> AsyncProfiler:
        if self.enable_cprofile:
            self._profiler = cProfile.Profile()
            self._profiler.enable()
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, *args: object) -> None:
        self._end_time = time.perf_counter()
        if self._profiler:
            self._profiler.disable()
            stream = io.StringIO()
            self._stats = pstats.Stats(self._profiler, stream=stream)
            self._stats.sort_stats(pstats.SortKey.CUMULATIVE)

    def print_stats(self, limit: int = 20) -> None:
        """Print the top N functions by cumulative time."""
        if self._stats:
            self._stats.print_stats(limit)


@contextmanager
def sync_profiler(
    name: str = "unnamed", *, enable_cprofile: bool = False
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager for profiling synchronous code blocks.

    Usage:
        with sync_profiler("my_operation") as result:
            some_operation()

        print(result["elapsed_sec"])
    """
    result: dict[str, Any] = {"name": name, "elapsed_sec": 0.0}
    profiler: cProfile.Profile | None = None

    if enable_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()

    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_sec"] = time.perf_counter() - start
        if profiler:
            profiler.disable()
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats(pstats.SortKey.CUMULATIVE)
            result["stats"] = stats


def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB (best effort)."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS returns bytes, Linux returns KB
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except (ImportError, AttributeError):
        return 0.0


def _calculate_std_dev(values: list[float], mean: float) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance**0.5


def profile_sync(
    *,
    print_stats: bool = False,
    log_stats: bool = True,
    enable_cprofile: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for profiling synchronous functions.

    Args:
        print_stats: Print timing to stdout after each call
        log_stats: Log timing via logger
        enable_cprofile: Enable detailed cProfile analysis
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            gc.collect()
            mem_before = _get_memory_usage_mb()

            profiler: cProfile.Profile | None = None
            if enable_cprofile:
                profiler = cProfile.Profile()
                profiler.enable()

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                mem_after = _get_memory_usage_mb()
                mem_delta = mem_after - mem_before

                if profiler:
                    profiler.disable()

                msg = f"{func.__qualname__}: {elapsed:.4f}s (memory delta: {mem_delta:+.2f}MB)"

                if print_stats:
                    print(msg)  # noqa: T201
                if log_stats:
                    LOGGER.info(msg)

                if enable_cprofile and profiler:
                    stats = pstats.Stats(profiler)
                    stats.sort_stats(pstats.SortKey.CUMULATIVE)
                    if print_stats:
                        stats.print_stats(20)

        return wrapper

    return decorator


def profile_async(
    *,
    print_stats: bool = False,
    log_stats: bool = True,
    enable_cprofile: bool = False,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for profiling async functions.

    Args:
        print_stats: Print timing to stdout after each call
        log_stats: Log timing via logger
        enable_cprofile: Enable detailed cProfile analysis
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            gc.collect()
            mem_before = _get_memory_usage_mb()

            profiler: cProfile.Profile | None = None
            if enable_cprofile:
                profiler = cProfile.Profile()
                profiler.enable()

            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                mem_after = _get_memory_usage_mb()
                mem_delta = mem_after - mem_before

                if profiler:
                    profiler.disable()

                msg = f"{func.__qualname__}: {elapsed:.4f}s (memory delta: {mem_delta:+.2f}MB)"

                if print_stats:
                    print(msg)  # noqa: T201
                if log_stats:
                    LOGGER.info(msg)

                if enable_cprofile and profiler:
                    stats = pstats.Stats(profiler)
                    stats.sort_stats(pstats.SortKey.CUMULATIVE)
                    if print_stats:
                        stats.print_stats(20)

        return wrapper

    return decorator


def run_benchmark(
    func: Callable[..., T],
    *args: Any,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs: Any,
) -> PerformanceStats:
    """
    Run a synchronous function multiple times and collect statistics.

    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not counted)
        **kwargs: Keyword arguments for the function

    Returns:
        PerformanceStats with aggregated results
    """
    # Warmup phase
    for _ in range(warmup):
        func(*args, **kwargs)

    gc.collect()
    mem_before = _get_memory_usage_mb()
    timings: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        timings.append(time.perf_counter() - start)

    mem_after = _get_memory_usage_mb()

    total_time = sum(timings)
    avg_time = total_time / len(timings)

    return PerformanceStats(
        function_name=func.__qualname__,
        total_calls=iterations,
        total_time_sec=total_time,
        avg_time_sec=avg_time,
        min_time_sec=min(timings),
        max_time_sec=max(timings),
        std_dev_sec=_calculate_std_dev(timings, avg_time),
        memory_peak_mb=mem_after - mem_before,
        timings=timings,
    )


async def run_benchmark_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs: Any,
) -> PerformanceStats:
    """
    Run an async function multiple times and collect statistics.

    Args:
        func: Async function to benchmark
        *args: Positional arguments for the function
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not counted)
        **kwargs: Keyword arguments for the function

    Returns:
        PerformanceStats with aggregated results
    """
    # Warmup phase
    for _ in range(warmup):
        await func(*args, **kwargs)

    gc.collect()
    mem_before = _get_memory_usage_mb()
    timings: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        await func(*args, **kwargs)
        timings.append(time.perf_counter() - start)

    mem_after = _get_memory_usage_mb()

    total_time = sum(timings)
    avg_time = total_time / len(timings)

    return PerformanceStats(
        function_name=func.__qualname__,
        total_calls=iterations,
        total_time_sec=total_time,
        avg_time_sec=avg_time,
        min_time_sec=min(timings),
        max_time_sec=max(timings),
        std_dev_sec=_calculate_std_dev(timings, avg_time),
        memory_peak_mb=mem_after - mem_before,
        timings=timings,
    )


def compare_implementations(
    baseline: Callable[..., T],
    optimized: Callable[..., T],
    *args: Any,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Compare two implementations and report speedup.

    Returns:
        Dictionary with baseline_stats, optimized_stats, and speedup ratio
    """
    baseline_stats = run_benchmark(baseline, *args, iterations=iterations, warmup=warmup, **kwargs)
    optimized_stats = run_benchmark(
        optimized, *args, iterations=iterations, warmup=warmup, **kwargs
    )

    speedup = (
        baseline_stats.avg_time_sec / optimized_stats.avg_time_sec
        if optimized_stats.avg_time_sec > 0
        else float("inf")
    )

    return {
        "baseline": baseline_stats,
        "optimized": optimized_stats,
        "speedup": speedup,
        "improvement_pct": (1 - 1 / speedup) * 100 if speedup > 0 else 0,
    }


__all__ = [
    "AsyncProfiler",
    "PerformanceStats",
    "ProfileResult",
    "compare_implementations",
    "profile_async",
    "profile_sync",
    "run_benchmark",
    "run_benchmark_async",
    "sync_profiler",
]
