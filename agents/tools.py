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


def top_n(df, by, value, n=3, agg="sum"):
    r = groupby_agg(df, by, value, agg)
    return dict(list(r.items())[: int(n)])


def time_trend(df, date, value, freq="M"):
    date, value = _col(df, date), _col(df, value)
    freq = {"M": "ME", "Y": "YE", "Q": "QE"}.get(freq, freq)          # pandas 2.2+ renamed the period-end aliases
    s = df.set_index(pd.to_datetime(df[date]))[value].resample(freq).sum().round(2)
    return {str(k.date()): float(v) for k, v in s.items()}


def correlation(df, a, b):
    a, b = _col(df, a), _col(df, b)
    if not (pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b])):
        raise ToolError("correlation needs two numeric columns")
    return {"pearson": round(float(df[a].corr(df[b])), 3), "n": int(df[[a, b]].dropna().shape[0])}


TOOLS = {"describe": describe, "groupby_agg": groupby_agg, "top_n": top_n, "time_trend": time_trend, "correlation": correlation}


def run_tool(df, name, args):
    if name not in TOOLS:
        raise ToolError(f"unknown tool '{name}'; available: {sorted(TOOLS)}")
    try:
        return TOOLS[name](df, **args)
    except TypeError as e:
        raise ToolError(f"bad arguments for {name}: {e}")


def sample_sales(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    regions = ["North", "South", "East", "West"]; products = ["Widget", "Gadget", "Gizmo", "Doohickey"]
    df = pd.DataFrame({"order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
                       "region": rng.choice(regions, n, p=[.35, .25, .25, .15]), "product": rng.choice(products, n),
                       "units": rng.integers(1, 20, n)})
    df["unit_price"] = df["product"].map({"Widget": 9.5, "Gadget": 24.0, "Gizmo": 41.0, "Doohickey": 14.0}) * rng.uniform(.9, 1.1, n)
    df["revenue"] = (df["units"] * df["unit_price"]).round(2)
    return df
