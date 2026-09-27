import time
from typing import Any, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TTLCache:
    """A simple dictionary-based cache with Time-To-Live (TTL) expiration."""
    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 256):
        self._cache = {}
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self.ttl_seconds:
                logger.info("LLM cache hit")
                return entry['value']
            else:
                # Expired
                logger.debug(f"Cache expired for key: {key}")
                del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        if len(self._cache) >= self.max_entries and key not in self._cache:
            oldest = min(self._cache, key=lambda item: self._cache[item]["timestamp"])
            del self._cache[oldest]
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        logger.debug(f"Cache set for key: {key}")

    def clear(self):
        self._cache.clear()

# Global specialized instances can be exported
llm_cache = TTLCache(ttl_seconds=3600)  # 1 hour cache
