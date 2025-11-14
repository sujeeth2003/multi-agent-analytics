"""HTTP front door for the agent graph.

    uvicorn api:app --port 8000
    curl -X POST localhost:8000/ask -H "content-type: application/json" -d '{"question": "Which region has the highest revenue?"}'

    POST /ask      {"question": "..."}  ->  {"answer", "status", "attempts"}
    GET  /health   -> {"ok": true}
    GET  /metrics  -> per-agent call counts, failures and latency (the Monitor from agents/monitor.py)
"""
from fastapi import FastAPI
from pydantic import BaseModel

from agents.graph import build_graph
from agents.llm import RulePlanner
from agents.monitor import Monitor
from agents.tools import sample_sales

