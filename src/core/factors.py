from __future__ import annotations

import pandas as pd
import numpy as np


def _rolling_max_drawdown(prices: pd.Series, window: int) -> float:
    s = prices.dropna()
    if len(s) < 3:
        return np.nan
    s = s.iloc[-window:] if len(s) > window else s
    running_max = s.cummax()
    dd = s / running_max - 1.0
    return float(dd.min())


def _find_yoy_period_value(
    f_ticker: pd.DataFrame,
    current_period_end: pd.Timestamp,
    field: str,
    target_days: int = 365,
    tolerance_days: int = 120,
) -> float:
    """
    Find the value for the same-ish quarter one year earlier.

    Logic:
      - target = current_period_end - 365d
      - choose the period_end closest to target within +/- tolerance_days
      - return f[field] for that period_end, else NaN
    """
    if pd.isna(current_period_end):
        return np.nan

    target = current_period_end - pd.Timedelta(days=target_days)

    # only consider periods strictly before current
    hist = f_ticker[f_ticker["period_end"] < current_period_end].copy()
    if hist.empty or field not in hist.columns:
        return np.nan

    hist["dist_days"] = (hist["period_end"] - target).abs().dt.days
    best = hist.sort_values("dist_days").iloc[0]

    if best["dist_days"] > tolerance_days:
        return np.nan

    v = best.get(field, np.nan)
    return float(v) if pd.notna(v) else np.nan


def compute_factors_for_snapshots(
    snapshots: pd.DataFrame,
    prices_daily: pd.DataFrame,
    fundamentals_quarterly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produces one row per (asof_date, ticker) with factor values.

    Factors (MVP):
      Momentum:
        - mom_12m_1m
        - mom_6m
        - trend_200dma (0/1)
      Growth:
        - rev_yoy
        - ni_yoy
        - fcf_yoy
      Quality:
        - fcf_margin
        - accruals
        - leverage
        - profitability (net_income / assets)
      Risk:
        - vol_1y
        - maxdd_1y
    """
    snaps = snapshots.copy()
    snaps["asof_date"] = pd.to_datetime(snaps["asof_date"]).dt.normalize()

    prices = prices_daily.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    price_pivot = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()

    # Prepare fundamentals
    f = fundamentals_quarterly.copy()
    if not f.empty:
        f["period_end"] = pd.to_datetime(f["period_end"]).dt.normalize()
        f["available_date"] = pd.to_datetime(f["available_date"]).dt.normalize()
    # We'll filter fundamentals by available_date <= asof_date (point-in-time-ish)
    # and then pick the yoy quarter relative to the snapshot's period_end.

    out_rows = []
    for (asof_date, ticker), g in snaps.groupby(["asof_date", "ticker"], sort=True):
        rec = {"asof_date": asof_date, "ticker": ticker}

        # Price history up to asof_date
        if ticker not in price_pivot.columns:
            continue
        px_series = price_pivot[ticker].loc[:asof_date].dropna()
        if len(px_series) < 260:
            continue

        # Momentum
        r_12m = (px_series.iloc[-1] / px_series.iloc[-252] - 1.0) if len(px_series) >= 252 else np.nan
        r_1m = (px_series.iloc[-1] / px_series.iloc[-21] - 1.0) if len(px_series) >= 21 else np.nan
        rec["mom_12m_1m"] = (r_12m - r_1m) if (pd.notna(r_12m) and pd.notna(r_1m)) else np.nan
        rec["mom_6m"] = (px_series.iloc[-1] / px_series.iloc[-126] - 1.0) if len(px_series) >= 126 else np.nan

        dma200 = px_series.rolling(200).mean().iloc[-1]
        rec["trend_200dma"] = 1.0 if (pd.notna(dma200) and px_series.iloc[-1] > dma200) else 0.0

        # Risk
        daily_ret = px_series.pct_change().dropna()
        rec["vol_1y"] = float(daily_ret.iloc[-252:].std() * np.sqrt(252)) if len(daily_ret) >= 252 else np.nan
        rec["maxdd_1y"] = _rolling_max_drawdown(px_series, window=252)

        # Snapshot fundamentals (latest available already)
        row = g.iloc[0]
        revenue = row.get("revenue", np.nan)
        net_income = row.get("net_income", np.nan)
        fcf = row.get("free_cash_flow", np.nan)
        assets = row.get("total_assets", np.nan)
        liabilities = row.get("total_liabilities", np.nan)
        equity = row.get("equity", np.nan)
        ocf = row.get("operating_cash_flow", np.nan)

        # Quality
        rec["fcf_margin"] = (fcf / revenue) if (pd.notna(fcf) and pd.notna(revenue) and revenue != 0) else np.nan
        rec["leverage"] = (liabilities / equity) if (pd.notna(liabilities) and pd.notna(equity) and equity != 0) else np.nan
        rec["accruals"] = ((net_income - ocf) / assets) if (pd.notna(net_income) and pd.notna(ocf) and pd.notna(assets) and assets != 0) else np.nan
        rec["profitability"] = (net_income / assets) if (pd.notna(net_income) and pd.notna(assets) and assets != 0) else np.nan

        # Growth (YoY) using fundamentals history (point-in-time-ish)
        # Use the snapshot's latest fundamental period_end if present; otherwise skip.
        current_period_end = row.get("period_end", pd.NaT)
        current_period_end = pd.to_datetime(current_period_end).normalize() if pd.notna(current_period_end) else pd.NaT

        if not f.empty and pd.notna(current_period_end):
            f_t = f[(f["ticker"] == ticker) & (f["available_date"] <= asof_date)].copy()

            prev_rev = _find_yoy_period_value(f_t, current_period_end, "revenue")
            prev_ni = _find_yoy_period_value(f_t, current_period_end, "net_income")
            prev_fcf = _find_yoy_period_value(f_t, current_period_end, "free_cash_flow")

            rec["rev_yoy"] = (revenue / prev_rev - 1.0) if (pd.notna(revenue) and pd.notna(prev_rev) and prev_rev != 0) else np.nan
            rec["ni_yoy"] = (net_income / prev_ni - 1.0) if (pd.notna(net_income) and pd.notna(prev_ni) and prev_ni != 0) else np.nan
            rec["fcf_yoy"] = (fcf / prev_fcf - 1.0) if (pd.notna(fcf) and pd.notna(prev_fcf) and prev_fcf != 0) else np.nan
        else:
            rec["rev_yoy"] = np.nan
            rec["ni_yoy"] = np.nan
            rec["fcf_yoy"] = np.nan

        out_rows.append(rec)

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df
    return df.sort_values(["asof_date", "ticker"]).reset_index(drop=True)
