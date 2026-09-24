import os
import sys
import unittest

import fakeredis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient  # noqa: E402

from agents.cache import Cache, fingerprint  # noqa: E402
from api import create_app  # noqa: E402
from agents.tools import sample_sales  # noqa: E402


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.df = sample_sales(600, seed=3)
        self.client = TestClient(create_app(self.df))

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"ok": True})

    def test_ask_answers_from_the_data(self):
        r = self.client.post("/ask", json={"question": "Which region has the highest revenue?"}).json()
        best = self.df.groupby("region").revenue.sum().idxmax()
        self.assertEqual(r["status"], "ok")
        self.assertIn(f"Highest: {best}", r["answer"])

    def test_unanswerable_question_returns_an_honest_failure_not_a_500(self):
        resp = self.client.post("/ask", json={"question": "Which region has the highest profit?"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("could not answer", resp.json()["answer"])

    def test_bad_request_body_is_rejected(self):
        self.assertEqual(self.client.post("/ask", json={"nope": 1}).status_code, 422)

    def test_metrics_reflect_calls(self):
        self.client.post("/ask", json={"question": "Which region has the highest revenue?"})
        self.assertGreaterEqual(self.client.get("/metrics").json()["planner"]["calls"], 1)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.df = sample_sales(600, seed=3)
        self.redis = fakeredis.FakeRedis(decode_responses=True)     # a real Redis server in memory
        self.client = TestClient(create_app(self.df, cache=Cache(self.redis, fingerprint(self.df))))
        self.q = {"question": "Which region has the highest revenue?"}

    def test_second_ask_is_served_from_redis_without_running_the_agents(self):
        first = self.client.post("/ask", json=self.q).json()
        calls = self.client.get("/metrics").json()["planner"]["calls"]
        second = self.client.post("/ask", json=self.q).json()
        self.assertFalse(first["cached"]); self.assertTrue(second["cached"])
        self.assertEqual(first["answer"], second["answer"])
        self.assertEqual(self.client.get("/metrics").json()["planner"]["calls"], calls)   # the planner did not run again
        self.assertEqual(len(self.redis.keys("answer:*")), 1)

    def test_wording_differences_in_case_and_spacing_share_an_entry(self):
        self.client.post("/ask", json=self.q)
        again = self.client.post("/ask", json={"question": "  WHICH region   has the highest REVENUE? "}).json()
        self.assertTrue(again["cached"])

    def test_failures_are_not_cached(self):
        bad = {"question": "Which region has the highest profit?"}
        self.client.post("/ask", json=bad)
        self.assertFalse(self.client.post("/ask", json=bad).json()["cached"])

    def test_different_data_does_not_reuse_answers(self):
        self.client.post("/ask", json=self.q)
        other = sample_sales(600, seed=99)
        c2 = TestClient(create_app(other, cache=Cache(self.redis, fingerprint(other))))
        self.assertFalse(c2.post("/ask", json=self.q).json()["cached"])


if __name__ == "__main__":
    unittest.main()
