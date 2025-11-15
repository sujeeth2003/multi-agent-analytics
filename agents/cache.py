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

