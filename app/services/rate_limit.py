"""Per-identity minute/day limits with Redis-backed production counters."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.config import settings


class IdentityRateLimiter:
    _SCRIPT = """
    local count = redis.call('INCR', KEYS[1])
    if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
    return count
    """

    def __init__(self) -> None:
        self._redis: Any = None
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def initialize(self) -> None:
        if settings.RATE_LIMIT_REDIS_URL:
            import redis

            self._redis = redis.Redis.from_url(
                settings.RATE_LIMIT_REDIS_URL,
                socket_connect_timeout=3,
                socket_timeout=3,
                decode_responses=True,
            )
            self._redis.ping()
        elif settings.AUTH_MODE == "cognito":
            raise RuntimeError("Cognito mode requires RATE_LIMIT_REDIS_URL.")

    def check_ready(self) -> None:
        if self._redis is not None:
            self._redis.ping()
        elif settings.AUTH_MODE == "cognito":
            raise RuntimeError("Distributed rate limiting is unavailable.")

    def enforce(self, subject: str) -> None:
        user_key = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:32]
        now = time.time()
        if self._redis is not None:
            minute_key = f"rag:ratelimit:{user_key}:minute:{int(now // 60)}"
            day_key = f"rag:ratelimit:{user_key}:day:{datetime.now(timezone.utc):%Y%m%d}"
            try:
                minute_count = int(self._redis.eval(self._SCRIPT, 1, minute_key, 120))
                if minute_count > settings.RATE_LIMIT_PER_MINUTE:
                    raise HTTPException(
                        status_code=429,
                        detail="Per-user request limit reached. Retry in the next minute.",
                        headers={"Retry-After": "60"},
                    )
                day_count = int(self._redis.eval(self._SCRIPT, 1, day_key, 172800))
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Distributed rate limiting is temporarily unavailable.",
                ) from exc
        else:
            minute_count, day_count = self._enforce_local(user_key, now)

        if minute_count > settings.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="Per-user request limit reached. Retry in the next minute.",
                headers={"Retry-After": "60"},
            )
        if day_count > settings.RATE_LIMIT_PER_DAY:
            raise HTTPException(
                status_code=429,
                detail="Per-user daily request limit reached.",
                headers={"Retry-After": "3600"},
            )

    def _enforce_local(self, user_key: str, now: float) -> tuple[int, int]:
        with self._lock:
            minute_events = self._events[f"{user_key}:minute"]
            while minute_events and now - minute_events[0] >= 60:
                minute_events.popleft()
            minute_events.append(now)

            if len(minute_events) > settings.RATE_LIMIT_PER_MINUTE:
                return len(minute_events), len(self._events[f"{user_key}:day"])

            day_events = self._events[f"{user_key}:day"]
            while day_events and now - day_events[0] >= 86400:
                day_events.popleft()
            day_events.append(now)
            return len(minute_events), len(day_events)


rate_limiter = IdentityRateLimiter()
