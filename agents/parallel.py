"""Run many questions in parallel across Ray worker processes.

Threads share one Python interpreter, so CPU-heavy agents (pandas work in the executor) fight over the GIL.
Ray starts separate worker processes instead.  Each worker builds its own copy of the graph once, then answers
questions handed to it; results and per-agent monitoring come back to the caller.
"""
import ray

from .graph import build_graph
from .llm import RulePlanner
from .monitor import Monitor


@ray.remote
class Worker:
    def __init__(self, df):
        self.monitor = Monitor()
        self.graph = build_graph(df, RulePlanner(), self.monitor)      # built once per process, reused for every question

    def answer(self, question):
        r = self.graph.invoke({"question": question})
        return {"question": question, "answer": r["report"], "status": r["status"], "attempts": r["attempts"]}

    def summary(self):
        return self.monitor.summary()


def answer_many(df, questions, workers=4):
    """Returns (answers in the same order as `questions`, monitoring summed over all workers)."""
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        pool = [Worker.remote(df) for _ in range(workers)]
        futures = [pool[i % workers].answer.remote(q) for i, q in enumerate(questions)]   # round-robin
        answers = ray.get(futures)
        merged = {}
        for s in ray.get([w.summary.remote() for w in pool]):
            for node, v in s.items():
                m = merged.setdefault(node, {"calls": 0, "failures": 0})
                m["calls"] += v["calls"]; m["failures"] += v["failures"]
        return answers, merged
    finally:
        ray.shutdown()
