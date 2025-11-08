"""Multi-agent orchestration with LangGraph.

    START -> planner -> executor -> critic --ok--------> reporter -> END
                ^                      |
                +------ replan --------+        (bounded by max_attempts, then reporter says it could not answer)

  planner   turns the question + data schema (+ the critic's feedback) into a tool plan
  executor  runs each planned tool call against the data with validated arguments; failures become data, not crashes
  critic    checks the results are usable and, if not, says exactly what was wrong (e.g. a column-name correction)
  reporter  writes the answer from the results, or an honest 'could not answer' with the reasons

Every node is timed and logged by Monitor, because agents fail in boring ways (wrong column, empty result, malformed plan)
and you cannot fix what you cannot see.
"""
import time
from typing import Any, TypedDict

