from __future__ import annotations
import pandas as pd


def month_ends(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return list of month-end dates between start and end (inclusive)."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    rng = pd.date_range(start=start, end=end, freq="ME")
    return [d.normalize() for d in rng]
