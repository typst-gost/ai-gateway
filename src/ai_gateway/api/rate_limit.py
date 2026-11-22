from __future__ import annotations

import time
from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = {}

    def check(self, key: str, *, max_requests: int, window_seconds: int) -> float:
        now = time.monotonic()
        cutoff = now - window_seconds
        bucket = self._buckets.setdefault(key, deque())

        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= max_requests:
            return max(0.1, bucket[0] + window_seconds - now)

        bucket.append(now)

        if len(self._buckets) > 4096:
            stale = [
                k for k, dq in self._buckets.items() if not dq or dq[-1] <= cutoff
            ]
            for k in stale:
                self._buckets.pop(k, None)

        return 0.0