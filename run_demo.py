"""Run a batch of questions through the agent graph concurrently and print answers plus monitoring.

    python run_demo.py                 # offline rule-based planner
    ANTHROPIC_API_KEY=... python run_demo.py --llm
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

from agents.graph import build_graph
from agents.llm import LLMPlanner, RulePlanner
from agents.monitor import Monitor
from agents.tools import sample_sales

QUESTIONS = [
    "Which region has the highest sales?",                     # 'sales' is not a column: the critic must repair it
    "Show the top 2 product by revenue",
    "What is the monthly revenue trend?",
    "Is there a correlation between price and units?",
    "Which region has the highest profit?",                    # 'profit' does not exist and has no close match: must fail honestly
]

