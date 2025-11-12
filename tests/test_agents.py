import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.graph import build_graph  # noqa: E402
from agents.llm import RulePlanner  # noqa: E402
from agents.monitor import Monitor  # noqa: E402
from agents.tools import ToolError, run_tool, sample_sales  # noqa: E402


class ToolTests(unittest.TestCase):
    def setUp(self): self.df = sample_sales(500, seed=1)

    def test_groupby_matches_pandas(self):
        r = run_tool(self.df, "groupby_agg", {"by": "region", "value": "revenue"})
        self.assertAlmostEqual(r["North"], round(self.df[self.df.region == "North"].revenue.sum(), 2), places=1)

    def test_bad_column_suggests_fix_and_arbitrary_tools_rejected(self):
        with self.assertRaises(ToolError) as e: run_tool(self.df, "groupby_agg", {"by": "region", "value": "revenu"})
        self.assertIn("did you mean 'revenue'", str(e.exception))
        with self.assertRaises(ToolError): run_tool(self.df, "os.system", {"cmd": "rm -rf /"})
        with self.assertRaises(ToolError): run_tool(self.df, "correlation", {"a": "region", "b": "units"})


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.df, self.mon = sample_sales(800, seed=2), Monitor()
        self.graph = build_graph(self.df, RulePlanner(), self.mon, max_attempts=3)

    def test_direct_success_takes_one_attempt(self):
        r = self.graph.invoke({"question": "Which region has the highest revenue?"})
        self.assertEqual((r["status"], r["attempts"]), ("ok", 1))
        best = self.df.groupby("region").revenue.sum().idxmax()
        self.assertIn(f"Highest: {best}", r["report"])

    def test_critic_repairs_wrong_column_via_replan(self):
        r = self.graph.invoke({"question": "Which region has the highest sales?"})       # 'sales' is not a column
        self.assertEqual(r["status"], "ok"); self.assertEqual(r["attempts"], 2)
        self.assertIn("revenue", r["report"])

    def test_unanswerable_question_fails_honestly_and_boundedly(self):
        r = self.graph.invoke({"question": "Which region has the highest profit?"})
        self.assertEqual(r["attempts"], 3); self.assertEqual(r["status"], "replan")
        self.assertIn("could not answer", r["report"])

    def test_planner_crash_is_contained_and_monitored(self):
        class Broken:
            def plan(self, *a, **k): raise RuntimeError("model returned garbage")
        mon = Monitor(); g = build_graph(self.df, Broken(), mon, max_attempts=2)
        r = g.invoke({"question": "anything"})
        self.assertIn("could not answer", r["report"]); self.assertGreaterEqual(mon.fail["planner"], 1)

    def test_concurrent_runs_are_isolated(self):
        qs = ["Which region has the highest revenue?", "Which product has the highest units?"] * 8
        with ThreadPoolExecutor(8) as ex: out = list(ex.map(lambda q: self.graph.invoke({"question": q}), qs))
        self.assertTrue(all(o["status"] == "ok" for o in out))
        self.assertEqual(out[0]["report"], out[2]["report"])
        self.assertEqual(self.mon.summary()["reporter"]["calls"], 16)


if __name__ == "__main__":
    unittest.main()
