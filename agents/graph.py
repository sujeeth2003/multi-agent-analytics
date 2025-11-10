"""Multi-agent orchestration with LangGraph.

    START -> planner -> executor -> critic --ok--------> reporter -> END
                ^                      |
                +------ replan --------+        (bounded by max_attempts, then reporter says it could not answer)

  planner   turns the question + data schema (+ the critic's feedback) into a tool plan
  executor  runs each planned tool call against the data with validated arguments; failures become data, not crashes
  critic    checks the results are usable and, if not, says exactly what was wrong (e.g. a column-name correction)
  reporter  writes the answer from the results, or an honest 'could not answer' with the reasons

Every node is timed and logged by Monitor, because agents fail in boring ways (wrong column, empty result, malformed plan)
and you cannot fix what you cannot see.
"""
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .monitor import Monitor
from .tools import ToolError, describe, run_tool


class State(TypedDict, total=False):
    question: str
    schema: dict
    plan: list
    results: list
    errors: list
    feedback: dict
    attempts: int
    report: str
    status: str


def build_graph(df, planner, monitor: Monitor, max_attempts=3):
    schema = describe(df)

    def timed(name, fn):
        def wrapper(state):
            t0 = time.perf_counter()
            try:
                out = fn(state)
                monitor.record(name, time.perf_counter() - t0, ok=True)
                return out
            except Exception as e:                       # a crashing node must not take the graph down silently
                monitor.record(name, time.perf_counter() - t0, ok=False, error=repr(e))
                a = state.get("attempts", 0) + (1 if name == "planner" else 0)
                msg = f"{name} crashed: {e!r}"
                return {"errors": [msg], "results": [], "plan": [], "attempts": a, "status": "replan",
                        "feedback": {"message": msg, "fix_columns": {}}}
        return wrapper

    def plan_node(state):
        plan = planner.plan(state["question"], schema, state.get("feedback"))
        return {"plan": plan, "attempts": state.get("attempts", 0) + 1, "errors": [], "results": []}

    def exec_node(state):
        results, errors = [], []
        for step in state["plan"]:
            try:
                results.append({"tool": step["tool"], "args": step["args"], "output": run_tool(df, step["tool"], step["args"])})
            except (ToolError, KeyError) as e:
                errors.append(f"{step.get('tool')}: {e}")
        return {"results": results, "errors": errors}

    def critic_node(state):
        if state.get("errors"):
            fix = {}
            for e in state["errors"]:                    # parse "unknown column 'x' (did you mean 'y'?)" into a correction map
                if "unknown column '" in e and "did you mean '" in e:
                    fix[e.split("unknown column '")[1].split("'")[0]] = e.split("did you mean '")[1].split("'")[0]
            return {"feedback": {"message": "; ".join(state["errors"]), "fix_columns": fix}, "status": "replan"}
        if not state.get("results") or all(not r["output"] for r in state["results"]):
            return {"feedback": {"message": "the plan produced no data", "fix_columns": {}}, "status": "replan"}
        return {"status": "ok"}

