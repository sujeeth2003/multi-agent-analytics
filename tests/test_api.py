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

