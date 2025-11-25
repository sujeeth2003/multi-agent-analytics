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

## The four pieces around the graph
Each one is small and does one job; each was added as its own commit.
| Tool | Why it is here | Where |
|---|---|---|
| **FastAPI** | turns the graph into a service: `POST /ask`, `GET /health`, `GET /metrics` | `api.py` |
| **Redis** | shared answer cache: asking the same question again skips all four agents. Only successful answers are cached, the key includes a fingerprint of the data, and entries expire after an hour | `agents/cache.py` |
| **Ray** | runs many questions in parallel in separate worker processes (each builds the graph once), instead of threads sharing one GIL | `agents/parallel.py` |
| **Docker** | `docker compose up` starts the API and a Redis together | `Dockerfile`, `docker-compose.yml` |

## Run
```bash
pip install -r requirements.txt
python -m unittest discover -s tests     # 17 tests (graph, API, Redis cache, Ray)
python run_demo.py                       # 5 questions, threads, with monitoring; the last one is unanswerable on purpose
python run_demo.py --ray                 # the same questions over Ray worker processes
uvicorn api:app --port 8000              # the API (set REDIS_URL=redis://localhost:6379/0 to cache in Redis)
docker compose up --build                # API + Redis in containers
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
- The Redis tests run against `fakeredis` (an in-memory Redis-compatible server), the API tests use FastAPI's test client, and the Ray test starts a real local Ray. **The Dockerfile and compose file were written but not built** (no Docker on the machine I used), and the code was not run against a real Redis server.
- Ray here parallelises independent questions; it does not split one question across machines.
