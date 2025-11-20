"""Run a batch of questions through the agent graph concurrently and print answers plus monitoring.

    python run_demo.py                 # offline rule-based planner, threads
    python run_demo.py --ray           # same questions, spread over Ray worker processes
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--llm", action="store_true"); ap.add_argument("--ray", action="store_true"); a = ap.parse_args()
    df, mon = sample_sales(), Monitor()
    if a.ray:
        from agents.parallel import answer_many
        answers, summary = answer_many(df, QUESTIONS)
        for r in answers:
            print(f"\nQ: {r['question']}\n   attempts={r['attempts']} status={r['status']}\n   A: {r['answer']}")
        print("\nmonitoring (summed over Ray workers):", summary)
        return
    graph = build_graph(df, LLMPlanner() if a.llm else RulePlanner(), mon)
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda q: graph.invoke({"question": q}), QUESTIONS))
    for q, r in zip(QUESTIONS, results):
        print(f"\nQ: {q}\n   attempts={r['attempts']} status={r['status']}\n   A: {r['report']}")
    print("\nmonitoring:")
    for n, s in mon.summary().items():
        print(f"  {n:<9} calls={s['calls']:>3} failures={s['failures']} p50={s['p50_ms']:.2f} ms max={s['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()
