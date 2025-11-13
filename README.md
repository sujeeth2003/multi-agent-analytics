# Multi-Agent Analytics (LangGraph)

A learning project on **multi-agent orchestration**: a question about a dataset flows through a planner, executor, critic and reporter wired as a [LangGraph](https://github.com/langchain-ai/langgraph) state machine, with a **bounded replan loop** and **monitoring on every node**.

```
START -> planner -> executor -> critic --ok--> reporter -> END
            ^                      |
            +------- replan -------+   (at most 3 attempts, then an honest "could not answer")
```
| Agent | Job |
|---|---|
| planner | question + data schema (+ the critic's feedback) -> a JSON list of tool calls |
| executor | runs the calls from a **closed toolbox** (`describe`, `groupby_agg`, `top_n`, `time_trend`, `correlation`); arguments are validated, and the model can never run arbitrary code |
| critic | checks the results and, when something failed, tells the planner exactly what (e.g. "unknown column `sales`, did you mean `revenue`?") |
| reporter | writes the answer from real results, or reports the failure and why |

## What I learned building it
Agents fail in boring ways, so build the failure handling and monitoring first. Running the demo exposed four real bugs, all fixed and now covered by tests:
1. **Wrong column names** (`sales` vs `revenue`): fixed by a critic feedback loop plus a small business glossary.
2. **A library change** (pandas renamed the `"M"` month alias to `"ME"`): the error came back as data, the planner retried the same call, and only the monitor showed the executor failing repeatedly.
3. **A silent wrong answer:** asked for "profit", the naive planner quietly substituted "revenue" and answered confidently. Silent substitution is worse than failure; now unknown measures fail loudly, and the system says it cannot answer.
4. **A crash path that dropped state** (a broken planner left the graph without an attempt counter): every node is wrapped so a crash becomes a `replan` with the error recorded.

## Run
```bash
pip install langgraph pandas numpy
python -m unittest discover -s tests     # 7 tests
python run_demo.py                       # 5 questions, concurrent, with monitoring; the last one is unanswerable on purpose
ANTHROPIC_API_KEY=... python run_demo.py --llm      # swap the planner for Claude (pip install anthropic)
```
Demo output (offline planner):
```
Q: Which region has the highest sales?     attempts=2 status=ok    (critic repaired 'sales' -> 'revenue')
Q: What is the monthly revenue trend?      attempts=1 status=ok
Q: Which region has the highest profit?    attempts=3 status=replan -> "I could not answer after 3 attempts..."
planner calls=8 failures=0 | executor calls=8 p50=7 ms | critic calls=8 | reporter calls=5
```

## Honest scope
- The default planner is **rule-based** so the project runs offline and the graph behaviour is deterministic and testable; it is intentionally naive so the critic has real mistakes to fix. The LLM planner (`--llm`) plugs into the same graph but **was not run here** (no API key).
- Its fuzzy suggestion for "profit" is `product` (string similarity); the loop retries with it, fails on the type check, and gives up honestly. A production critic would also check that a suggested column is semantically plausible.
- Concurrency is threads, one graph invocation per question, verified isolated by a test. Ray, Redis and a FastAPI/Docker wrapper (parts of the original plan) are **not built**; the graph is a plain callable, so wrapping it in FastAPI is a few lines.
