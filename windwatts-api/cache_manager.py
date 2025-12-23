from cachetools import cached, TTLCache


class CacheManager:
    def __init__(self, maxsize=100, ttl=300):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def cache_data(self, func):
        return cached(self.cache)(func)
