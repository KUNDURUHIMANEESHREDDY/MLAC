"""Tests for shared Redis-backed rate limiting."""

import pytest

from mlac.rate_limit import RateLimiterUnavailable, RedisRateLimiter


class FakeRedis:
    """Minimal Redis client stub for deterministic Lua-script contract tests."""

    def __init__(self, result=None, error=None):
        self.result = result if result is not None else [1, 0]
        self.error = error
        self.calls = []

    def eval(self, script, key_count, key, window_ms, limit, member):
        self.calls.append((script, key_count, key, window_ms, limit, member))
        if self.error is not None:
            raise self.error
        return self.result

    def ping(self):
        if self.error is not None:
            raise self.error
        return True


def test_redis_limiter_uses_hashed_identity_and_atomic_script():
    client = FakeRedis()
    limiter = RedisRateLimiter(
        client, namespace="principal", limit=3, window_seconds=12.5
    )

    limiter.check("issuer|subject")

    script, key_count, key, window_ms, limit, member = client.calls[0]
    assert "redis.call('TIME')" in script
    assert key_count == 1
    assert key.startswith("mlac:rate:principal:")
    assert "issuer|subject" not in key
    assert window_ms == 12500
    assert limit == 3
    assert member


def test_redis_limiter_rejects_over_limit_with_retry_after():
    limiter = RedisRateLimiter(
        FakeRedis(result=[0, 4]), namespace="peer", limit=2, window_seconds=10
    )

    with pytest.raises(RuntimeError, match="4"):
        limiter.check("203.0.113.8")


def test_redis_limiter_fails_closed_when_redis_is_unavailable():
    limiter = RedisRateLimiter(
        FakeRedis(error=OSError("connection refused")),
        namespace="principal",
        limit=5,
        window_seconds=30,
    )

    with pytest.raises(RateLimiterUnavailable):
        limiter.check("issuer|subject")
    with pytest.raises(RateLimiterUnavailable):
        limiter.health_check()


def test_redis_limiter_rejects_invalid_configuration_and_responses():
    with pytest.raises(ValueError):
        RedisRateLimiter(FakeRedis(), namespace="", limit=1, window_seconds=1)
    with pytest.raises(ValueError):
        RedisRateLimiter(FakeRedis(), namespace="peer", limit=0, window_seconds=1)
    with pytest.raises(ValueError):
        RedisRateLimiter(FakeRedis(), namespace="peer", limit=1, window_seconds=0)
    limiter = RedisRateLimiter(
        FakeRedis(result=[1]), namespace="peer", limit=1, window_seconds=1
    )
    with pytest.raises(RateLimiterUnavailable):
        limiter.check("peer")
