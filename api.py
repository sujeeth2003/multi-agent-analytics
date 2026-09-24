"""HTTP front door for the agent graph.

    uvicorn api:app --port 8000
    curl -X POST localhost:8000/ask -H "content-type: application/json" -d '{"question": "Which region has the highest revenue?"}'

    POST /ask      {"question": "..."}  ->  {"answer", "status", "attempts", "cached"}
    GET  /health   -> {"ok": true}
    GET  /metrics  -> per-agent call counts, failures and latency (the Monitor from agents/monitor.py)
"""
from fastapi import FastAPI
from pydantic import BaseModel

from agents.cache import Cache, cache_from_env
from agents.graph import build_graph
from agents.llm import RulePlanner
from agents.monitor import Monitor
from agents.tools import sample_sales


class Question(BaseModel):
    question: str


def create_app(df=None, planner=None, cache: Cache = None) -> FastAPI:
    df = sample_sales() if df is None else df
    cache = cache or cache_from_env(df)          # Redis when REDIS_URL is set, otherwise an in-process dict
    monitor = Monitor()
    graph = build_graph(df, planner or RulePlanner(), monitor)
    app = FastAPI(title="Multi-agent analytics")

    @app.post("/ask")
    def ask(q: Question):
        hit = cache.get(q.question)
        if hit:
            return {**hit, "cached": True}
        r = graph.invoke({"question": q.question})
        out = {"answer": r["report"], "status": r["status"], "attempts": r["attempts"]}
        if r["status"] == "ok":                  # only good answers are cached: a failure may be temporary
            cache.put(q.question, out)
        return {**out, "cached": False}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/metrics")
    def metrics():
        return {**monitor.summary(), "cache": {"hits": cache.hits, "misses": cache.misses}}

    return app


app = create_app()
