"""The executor's toolbox: a small, closed set of pandas operations. The LLM never gets to run arbitrary code;
it can only choose from these tools and supply arguments, and every argument is validated before use.
"""
import difflib

import numpy as np
import pandas as pd


class ToolError(Exception):
    """A recoverable problem (bad column, wrong dtype...) that the critic can report back to the planner."""


# business glossary: words analysts use for columns this dataset calls something else
SYNONYMS = {"sales": "revenue", "income": "revenue", "turnover": "revenue", "quantity": "units", "qty": "units",
            "price": "unit_price", "date": "order_date"}


