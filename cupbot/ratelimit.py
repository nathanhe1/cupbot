"""Async token bucket, and backoff with full jitter."""

from __future__ import annotations

import asyncio
import random
import time


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: float = 0.0):
        self.rate = rate_per_s
        self.capacity = burst or max(1.0, rate_per_s)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    async def acquire(self, n: float = 1.0) -> None:
        async with self._lock:
            while True:
                self._refill()
                if self.tokens >= n:
                    self.tokens -= n
                    return
                await asyncio.sleep((n - self.tokens) / self.rate)

    def cooldown(self, seconds: float) -> None:
        """After a 429: drain the bucket so nothing is sent for `seconds`."""
        self.tokens = -seconds * self.rate
        self.updated = time.monotonic()


def backoff_s(attempt: int, base: float = 0.25, cap: float = 8.0) -> float:
    return random.uniform(0, min(cap, base * (2 ** attempt)))
