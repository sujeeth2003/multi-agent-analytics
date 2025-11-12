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

