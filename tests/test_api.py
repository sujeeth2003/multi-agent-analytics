import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
