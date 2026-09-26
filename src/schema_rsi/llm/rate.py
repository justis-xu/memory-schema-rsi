"""渐进式自适应并发控制器（AIMD）。

用法约定：
- 起始并发 start（默认 8），每 up_interval 秒无限流 +1，爬到 ceiling（默认 20）
- 遇限流按 ladder 依次降档（默认 [15, 10]，用完后减半），并短暂退避重试
- 线程池开 ceiling 个线程，任务用 acquire/release 包住实际工作
"""

from __future__ import annotations

import threading
import time


class AdaptiveLimiter:
    def __init__(self, start: int = 8, ceiling: int = 20, ladder=(15, 10), up_interval: float = 30.0):
        self._lock = threading.Lock()
        self._limit = start
        self._ceiling = ceiling
        self._ladder = list(ladder)
        self._up_interval = up_interval
        self._last_up = time.monotonic()
        self._sem = threading.Semaphore(start)

    @property
    def limit(self) -> int:
        return self._limit

    def acquire(self, timeout: float | None = None) -> bool:
        return self._sem.acquire(timeout=timeout) if timeout else self._sem.acquire()

    def release(self) -> None:
        self._sem.release()

    def on_rate_limited(self) -> int:
        """限流时降档，返回新档位。"""
        with self._lock:
            if self._ladder:
                target = self._ladder.pop(0)
            else:
                target = max(4, self._limit // 2)
            target = min(target, self._limit)
            drop = self._limit - target
            self._limit = target
        for _ in range(drop):
            self._sem.acquire()
        return self._limit

    def maybe_increase(self) -> None:
        with self._lock:
            if self._limit >= self._ceiling:
                return
            if time.monotonic() - self._last_up < self._up_interval:
                return
            self._limit += 1
            self._last_up = time.monotonic()
        self._sem.release()


def is_rate_limit_error(exc: Exception) -> bool:
    """识别 OpenAI-compatible 端点的限流错误（429 / RateLimitError / 智谱 1311/1302）。"""
    name = type(exc).__name__
    text = str(exc)
    return (
        "RateLimit" in name
        or "429" in text
        or "1311" in text
        or "1302" in text
        or "Too Many Requests" in text
    )
