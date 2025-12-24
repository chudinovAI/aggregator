from __future__ import annotations

import asyncio
import time
from typing import Dict


class AsyncRateLimiter:
    """Simple per-key rate limiter enforcing a minimum interval between requests."""

    def __init__(
        self,
        requests_per_window: int,
        window_seconds: float,
        overrides: Dict[str, tuple[int, int]] | None = None,
    ) -> None:
        if requests_per_window <= 0 or window_seconds <= 0:
            self._min_interval = 0.0
        else:
            self._min_interval = window_seconds / float(requests_per_window)
        self._overrides: Dict[str, float] = {}
        if overrides:
            for key, profile in overrides.items():
                requests, window = profile
                if requests <= 0 or window <= 0:
                    continue
                self._overrides[key] = window / float(requests)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_call: Dict[str, float] = {}

    async def acquire(self, key: str) -> None:
        interval = self._overrides.get(key, self._min_interval)
        if interval <= 0:
            return
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        async with lock:
            now = time.monotonic()
            last_call = self._last_call.get(key, 0.0)
            wait_time = interval - (now - last_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call[key] = time.monotonic()
