"""Planner brains. Every planner implements  plan(question, schema, feedback) -> list of {"tool": ..., "args": {...}}.

RulePlanner   deterministic and offline: keyword rules. It is deliberately naive (it guesses column names from the words
              in the question), so it makes exactly the boring mistakes real LLM agents make and the critic loop has real work to do.
LLMPlanner    asks Claude for a JSON plan (needs `pip install anthropic` and ANTHROPIC_API_KEY); the same graph, tools and critic apply.
"""
import json
import re

WORD_TO_COL = {"sales": "sales", "revenue": "revenue", "income": "income", "units": "units", "quantity": "quantity", "price": "unit_price",
               "region": "region", "product": "product", "date": "order_date", "month": "order_date"}


