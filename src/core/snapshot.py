from __future__ import annotations

import pandas as pd
import numpy as np


def _latest_fundamental_asof(
    fundamentals: pd.DataFrame,
    ticker: str,
    asof_date: pd.Timestamp,
) -> dict:
    """
    Pick the latest fundamental row that is *known* as of the snapshot date.

    No look-ahead:
      - We only use rows with available_date <= asof_date.
    """
    if fundamentals is None or fundamentals.empty:
        return {}
    if "ticker" not in fundamentals.columns or "available_date" not in fundamentals.columns:
        return {}
    f = fundamentals[fundamentals["ticker"] == ticker]
    f = f[f["available_date"] <= asof_date]
    if f.empty:
        return {}
    row = f.sort_values("available_date").iloc[-1].to_dict()
    return row


def build_monthly_snapshots(
    asof_dates: list[pd.Timestamp],
    tickers: list[str],
    prices_daily: pd.DataFrame,
    fundamentals_quarterly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates a monthly snapshot per ticker containing:
      - asof_date (the *last available trading day <= calendar month_end*, per ticker)
      - month_end (original calendar month end)
      - adj_close at asof_date
      - latest available fundamentals as of asof_date (no look-ahead)
    """
    if prices_daily is None or prices_daily.empty:
        return pd.DataFrame()

    # Normalize dates & keep only needed columns
    prices = prices_daily.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices[["ticker", "date", "adj_close"]].copy()
    prices = prices.dropna(subset=["ticker", "date", "adj_close"])

    # Ensure month_ends are normalized
    month_ends = [pd.to_datetime(d).normalize() for d in asof_dates]

    # Build fast per-ticker lookup arrays
    # dict: ticker -> (dates ndarray[datetime64], adj_close ndarray[float])
    by_ticker: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in tickers:
        p = prices[prices["ticker"] == t].sort_values("date")
        if p.empty:
            continue
        dates = p["date"].to_numpy(dtype="datetime64[ns]")
        vals = p["adj_close"].to_numpy(dtype=float)
        by_ticker[str(t)] = (dates, vals)

    rows = []
    for month_end in month_ends:
        me64 = np.datetime64(month_end.to_datetime64())

        for t in tickers:
            key = str(t)
            if key not in by_ticker:
                continue

            dates, vals = by_ticker[key]

            # last index where dates[idx] <= month_end
            idx = np.searchsorted(dates, me64, side="right") - 1
            if idx < 0:
                continue

            d_use = pd.Timestamp(dates[idx]).normalize()
            px = vals[idx]
            if not np.isfinite(px):
                continue

            fund = _latest_fundamental_asof(fundamentals_quarterly, key, d_use)

            rec = {
                "asof_date": d_use,      # trading day actually used for pricing (<= month_end)
                "month_end": month_end,  # calendar month end (audit)
                "ticker": key,
                "adj_close": float(px),
            }

            # Attach selected fundamental fields (if present)
            for k in [
                "period_end", "available_date",
                "revenue", "net_income", "cogs",
                "total_assets", "total_liabilities", "equity",
                "operating_cash_flow", "capital_expenditures", "free_cash_flow",
            ]:
                if k in fund:
                    rec[k] = fund.get(k)

            rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.normalize()
    df["month_end"] = pd.to_datetime(df["month_end"]).dt.normalize()

    # Guardrail: ensure we never leak future fundamentals into the past.
    if "available_date" in df.columns:
        ad = pd.to_datetime(df["available_date"], errors="coerce")
        bad = df[(ad.notna()) & (ad > df["asof_date"])].head(5)
        if not bad.empty:
            raise RuntimeError(
                "Look-ahead detected: some fundamentals have available_date > asof_date. "
                "Example rows:\n"
                + bad[["ticker", "asof_date", "available_date", "period_end"]].to_string(index=False)
            )

    return df.sort_values(["asof_date", "ticker"]).reset_index(drop=True)
