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
