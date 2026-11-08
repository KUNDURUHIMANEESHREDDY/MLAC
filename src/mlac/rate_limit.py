"""Shared Redis-backed sliding-window rate limiting for multi-process deployments."""

from __future__ import annotations

import hashlib
import math
from typing import Any
from uuid import uuid4


class RateLimiterUnavailable(Exception):
    """Raised when the shared rate-limit service cannot be reached."""


_REDIS_SLIDING_WINDOW = """
local time = redis.call('TIME')
local now = (tonumber(time[1]) * 1000) + math.floor(tonumber(time[2]) / 1000)
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window_ms)
local count = redis.call('ZCARD', KEYS[1])
if count >= limit then
  local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  local retry_ms = window_ms
  if oldest[2] then
    retry_ms = math.max(1, tonumber(oldest[2]) + window_ms - now)
  end
  return {0, math.ceil(retry_ms / 1000)}
end
redis.call('ZADD', KEYS[1], now, ARGV[3])
redis.call('PEXPIRE', KEYS[1], window_ms)
return {1, 0}
"""


class RedisRateLimiter:
    """Atomic Redis sliding-window limiter shared across API workers and hosts."""

    def __init__(
        self,
        client: Any,
        *,
        namespace: str,
        limit: int,
        window_seconds: float,
    ) -> None:
        """Bind a Redis client to a validated namespace and request budget."""
        if not namespace or not isinstance(namespace, str):
            raise ValueError("rate limiter namespace must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("rate limit must be a positive integer")
        if (
            isinstance(window_seconds, bool)
            or not isinstance(window_seconds, (int, float))
            or not math.isfinite(window_seconds)
            or window_seconds <= 0
        ):
            raise ValueError("rate limit window must be a finite positive number")
        self.client = client
        self.namespace = namespace
        self.limit = limit
        self.window_ms = max(1, int(window_seconds * 1000))

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        namespace: str,
        limit: int,
        window_seconds: float,
    ) -> "RedisRateLimiter":
        """Create a bounded-timeout Redis client from an operator-supplied URL."""
        try:
            from redis import Redis
        except ImportError as exc:
            raise RuntimeError("Install the production extra to use Redis rate limiting") from exc
        if not isinstance(url, str) or not url.startswith(("redis://", "rediss://")):
            raise ValueError("MLAC_RATE_LIMIT_REDIS_URL must use redis:// or rediss://")
        client = Redis.from_url(
            url,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
            decode_responses=True,
        )
        return cls(
            client,
            namespace=namespace,
            limit=limit,
            window_seconds=window_seconds,
        )

    def health_check(self) -> None:
        """Fail readiness when the shared rate-limit service is unavailable."""
        try:
            self.client.ping()
        except Exception as exc:
            raise RateLimiterUnavailable("shared rate-limit service is unavailable") from exc

    def check(self, identity: str) -> None:
        """Atomically admit one request or raise with a retry-after interval."""
        if not isinstance(identity, str) or not identity:
            raise ValueError("rate limit identity must be a non-empty string")
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        key = f"mlac:rate:{self.namespace}:{digest}"
        try:
            result = self.client.eval(
                _REDIS_SLIDING_WINDOW,
                1,
                key,
                self.window_ms,
                self.limit,
                uuid4().hex,
            )
        except Exception as exc:
            raise RateLimiterUnavailable("shared rate-limit service is unavailable") from exc
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise RateLimiterUnavailable("shared rate-limit service returned an invalid response")
        try:
            allowed, retry_after = int(result[0]), int(result[1])
        except (TypeError, ValueError) as exc:
            raise RateLimiterUnavailable("shared rate-limit service returned invalid counters") from exc
        if not allowed:
            raise RuntimeError(str(max(1, retry_after)))
