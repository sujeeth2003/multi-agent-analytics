import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.parallel import answer_many  # noqa: E402
from agents.tools import sample_sales  # noqa: E402


class RayTests(unittest.TestCase):
    def test_answers_come_back_in_order_and_match_the_data(self):
        df = sample_sales(600, seed=4)
        qs = ["Which region has the highest revenue?", "Which product has the highest units?",
              "Which region has the highest profit?"]
        answers, summary = answer_many(df, qs, workers=2)
        self.assertEqual([a["question"] for a in answers], qs)
        self.assertIn(f"Highest: {df.groupby('region').revenue.sum().idxmax()}", answers[0]["answer"])
        self.assertEqual(answers[2]["status"], "replan")                 # unanswerable, still fails honestly in a worker
        self.assertEqual(summary["reporter"]["calls"], 3)                # monitoring merged across both workers


if __name__ == "__main__":
    unittest.main()
