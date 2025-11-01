"""Planner brains. Every planner implements  plan(question, schema, feedback) -> list of {"tool": ..., "args": {...}}.

RulePlanner   deterministic and offline: keyword rules. It is deliberately naive (it guesses column names from the words
              in the question), so it makes exactly the boring mistakes real LLM agents make and the critic loop has real work to do.
LLMPlanner    asks Claude for a JSON plan (needs `pip install anthropic` and ANTHROPIC_API_KEY); the same graph, tools and critic apply.
"""
import json
import re

WORD_TO_COL = {"sales": "sales", "revenue": "revenue", "income": "income", "units": "units", "quantity": "quantity", "price": "unit_price",
               "region": "region", "product": "product", "date": "order_date", "month": "order_date"}


class RulePlanner:
    def plan(self, question, schema, feedback=None):
        q = question.lower()
        cols = list(schema["columns"])
        if feedback and feedback.get("fix_columns"):          # the critic told us which columns were wrong: use its suggestions
            fix = feedback["fix_columns"]
            q = " ".join(fix.get(w, w) for w in q.split())
            guess = lambda w, default: fix.get(WORD_TO_COL.get(w, default), WORD_TO_COL.get(w, default))
        else:
            guess = lambda w, default: WORD_TO_COL.get(w, default)
        measure = next((guess(w, w) for w in re.findall(r"[a-z_]+", q) if w in ("sales", "revenue", "income", "units", "quantity", "profit", "cost", "margin")), "revenue")
        dim = next((w for w in ("region", "product") if w in q), None)
        n = next((int(x) for x in re.findall(r"\btop (\d+)", q)), None)
        if "trend" in q or "monthly" in q or "over time" in q:
            return [{"tool": "time_trend", "args": {"date": "order_date", "value": measure, "freq": "M"}}]
        if "correlat" in q or "relationship" in q:
            return [{"tool": "correlation", "args": {"a": "unit_price", "b": "units"}}]
        if dim and n:
            return [{"tool": "top_n", "args": {"by": dim, "value": measure, "n": n}}]
        if dim:
            return [{"tool": "groupby_agg", "args": {"by": dim, "value": measure, "agg": "sum"}}]
        return [{"tool": "describe", "args": {}}]

