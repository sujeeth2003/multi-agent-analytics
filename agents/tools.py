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


def _col(df, name):
    if name not in df.columns:
        syn = SYNONYMS.get(str(name).lower())
        close = [syn] if syn in df.columns else difflib.get_close_matches(str(name), df.columns, n=1, cutoff=0.5)
        raise ToolError(f"unknown column '{name}'" + (f" (did you mean '{close[0]}'?)" if close else f"; available: {list(df.columns)}"))
    return name


def describe(df):
    return {"rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()}}


def groupby_agg(df, by, value, agg="sum"):
    by, value = _col(df, by), _col(df, value)
    if agg not in ("sum", "mean", "count", "max", "min"):
        raise ToolError(f"unsupported aggregation '{agg}'")
    if agg != "count" and not pd.api.types.is_numeric_dtype(df[value]):
        raise ToolError(f"column '{value}' is not numeric")
    return df.groupby(by)[value].agg(agg).sort_values(ascending=False).round(2).to_dict()


