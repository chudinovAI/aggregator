from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Iterable, Type, TypeVar

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def _compute_delay(base_delay: float, attempt: int) -> float:
    return base_delay * (2**attempt)


async def async_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    retry_exceptions: Iterable[Type[BaseException]] = (Exception,),
) -> T:
    for attempt in range(retries):
        try:
            return await operation()
        except retry_exceptions as exc:  # type: ignore[misc]
            if attempt == retries - 1:
                raise
            delay = _compute_delay(base_delay, attempt)
            LOGGER.debug("Retrying async operation after %.2fs due to %s", delay, exc)
            await asyncio.sleep(delay)
    raise RuntimeError("Async retry exhausted unexpectedly.")


def sync_retry(
    operation: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    retry_exceptions: Iterable[Type[BaseException]] = (Exception,),
) -> T:
    for attempt in range(retries):
        try:
            return operation()
        except retry_exceptions as exc:  # type: ignore[misc]
            if attempt == retries - 1:
                raise
            delay = _compute_delay(base_delay, attempt)
            LOGGER.debug("Retrying sync operation after %.2fs due to %s", delay, exc)
            time.sleep(delay)
    raise RuntimeError("Sync retry exhausted unexpectedly.")
