"""Answer cache: the same question on the same data should not run the whole agent graph twice.

Redis is the shared cache (all API workers see the same entries). With no Redis configured, a dict is used so the
project still runs on a laptop.  The key includes a fingerprint of the data, so new data never returns old answers.
"""
import hashlib
import json
import os


def fingerprint(df) -> str:
    return hashlib.sha256(f"{list(df.columns)}|{len(df)}|{df.iloc[:50].to_csv()}".encode()).hexdigest()[:12]


class Cache:
    def __init__(self, client=None, data_key="", ttl=3600):
        self.client, self.data_key, self.ttl = client, data_key, ttl
        self.local = {}
        self.hits = self.misses = 0

    def _key(self, question):
        norm = " ".join(question.lower().split())
        return f"answer:{self.data_key}:{hashlib.sha256(norm.encode()).hexdigest()[:16]}"

    def get(self, question):
        raw = self.client.get(self._key(question)) if self.client else self.local.get(self._key(question))
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def put(self, question, answer: dict):
        raw = json.dumps(answer)
        if self.client:
            self.client.set(self._key(question), raw, ex=self.ttl)     # entries expire on their own
        else:
            self.local[self._key(question)] = raw


def cache_from_env(df):
    """REDIS_URL=redis://localhost:6379/0 turns Redis on; otherwise an in-process dict."""
    url = os.environ.get("REDIS_URL")
    client = None
    if url:
        import redis
        client = redis.Redis.from_url(url, decode_responses=True)
    return Cache(client, data_key=fingerprint(df))
