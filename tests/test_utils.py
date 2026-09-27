import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.utils.cache import TTLCache
from src.utils.rate_limiter import RateLimiter
from src.utils.token_counter import count_tokens


def test_cache_is_bounded():
    cache = TTLCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert len(cache._cache) == 2


def test_rate_limiter_preserves_429():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    asyncio.run(limiter(request))
    with pytest.raises(HTTPException) as error:
        asyncio.run(limiter(request))
    assert error.value.status_code == 429


def test_token_counter_handles_empty_and_text():
    assert count_tokens("") == 0
    assert count_tokens("Hello, world") > 0
