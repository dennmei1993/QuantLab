# src/core/factors.py

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def _safe_div(a: Any, b: Any) -> float:
    try:
        if a is None or b is None:
            return np.nan
        if pd.isna(a) or pd.isna(b):
            return np.nan
        b = float(b)
        if b == 0.0:
            return np.nan
        return float(a) / b
    except Exception:
        return np.nan


def _momentum(prices: pd.Series, lookback: int, skip: int = 0) -> float:
    """
    Momentum = price[t-skip] / price[t-lookback-skip] - 1
    """
    s = prices.dropna()
    if len(s) < lookback + skip + 1:
        return np.nan
    p0 = s.iloc[-1 - skip]
    p1 = s.iloc[-1 - lookback - skip]
    return float(p0 / p1 - 1.0) if p1 and p1 != 0 else np.nan


def _trend_200dma(prices: pd.Series) -> float:
    s = prices.dropna()
    if len(s) < 200:
        return np.nan
    dma = s.rolling(200).mean().iloc[-1]
    return float(s.iloc[-1] / dma - 1.0) if dma and dma != 0 else np.nan


def _volatility_1y(prices: pd.Series) -> float:
    s = prices.dropna()
    if len(s) < 252:
        return np.nan
    rets = s.pct_change().dropna()
    if len(rets) < 50:
        return np.nan
    return float(rets.tail(252).std() * np.sqrt(252))


def _rolling_max_drawdown(prices: pd.Series) -> float:
    """
    Max drawdown over the provided series window.
    """
    s = prices.dropna()
    if len(s) < 10:
        return np.nan
    cummax = s.cummax()
    dd = (s / cummax) - 1.0
    return float(dd.min())


def _pick_available_date_col(fq: pd.DataFrame) -> Optional[str]:
    """
    Choose the best-available 'fundamentals available date' column.
    Prefer available_date, then acceptedDate, filingDate, date.
    """
    for c in ["available_date", "acceptedDate", "filingDate", "date"]:
        if c in fq.columns:
            return c
    return None


def _find_yoy_period_value(
    f_ticker: pd.DataFrame,
    current_period_end: pd.Timestamp,
    field: str,
) -> float:
    """
    Find value around ~1 year before current_period_end.
    Assumes quarterly data; picks closest record within a tolerance.
    """
    if f_ticker.empty or field not in f_ticker.columns:
        return np.nan
    if "period_end" not in f_ticker.columns:
        return np.nan

    # target ~ 1 year ago
    target = pd.to_datetime(current_period_end) - pd.DateOffset(years=1)

    # pick closest by absolute day difference
    f2 = f_ticker.copy()
    f2["__diff"] = (pd.to_datetime(f2["period_end"]) - target).abs()
    f2 = f2.sort_values("__diff")
    if f2.empty:
        return np.nan

    # tolerance: 120 days (quarter drift / reporting schedule)
    best = f2.iloc[0]
    if pd.isna(best["__diff"]) or best["__diff"] > pd.Timedelta(days=120):
        return np.nan

    return float(best.get(field)) if pd.notna(best.get(field)) else np.nan


def _accruals(d: dict) -> float:
    """
    Simplified accruals proxy:
      (net_income - operating_cash_flow) / total_assets
    """
    ni = d.get("net_income", np.nan)
    ocf = d.get("operating_cash_flow", np.nan)
    ta = d.get("total_assets", np.nan)
    return _safe_div((ni - ocf), ta)


def _leverage(d: dict) -> float:
    """
    Leverage = total_debt / total_assets
    """
    debt = d.get("total_debt", np.nan)
    ta = d.get("total_assets", np.nan)
    return _safe_div(debt, ta)


def _yoy_growth(f_ticker: pd.DataFrame, field: str) -> float:
    """
    YoY growth based on the most recent record and a ~1y-ago record.
    Returns (curr - prev) / abs(prev)
    """
    if f_ticker.empty or field not in f_ticker.columns:
        return np.nan
    if "period_end" not in f_ticker.columns:
        return np.nan

    f_ticker = f_ticker.sort_values("period_end")
    current = f_ticker.iloc[-1]
    curr_val = current.get(field, np.nan)
    if pd.isna(curr_val):
        return np.nan

    prev_val = _find_yoy_period_value(
        f_ticker,
        current_period_end=pd.to_datetime(current["period_end"]),
        field=field,
    )
    if pd.isna(prev_val) or prev_val == 0:
        return np.nan

    return _safe_div((curr_val - prev_val), abs(prev_val))


# -----------------------------
# Main
# -----------------------------

def compute_factors_for_snapshots(
    snapshots: pd.DataFrame,
    prices_daily: pd.DataFrame,
    fundamentals_quarterly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Input snapshots: one row per (asof_date, ticker) with price as-of month end,
    plus optional snapshot metadata columns: asset_type, theme.

    Output: one row per (asof_date, ticker) with factor columns:
      Momentum: mom_12m_1m, mom_6m, trend_200dma
      Risk: vol_1y, maxdd_1y
      Quality: fcf_margin, profitability, accruals, leverage   (stocks only)
      Growth: rev_yoy, ni_yoy, fcf_yoy                          (stocks only)

    ETFs: price-only (Momentum + Risk). Fundamental factors are NaN.
    """

    if snapshots is None or snapshots.empty:
        return pd.DataFrame()

    snaps = snapshots.copy()
    snaps["asof_date"] = pd.to_datetime(snaps["asof_date"]).dt.normalize()
    snaps["ticker"] = snaps["ticker"].astype(str).str.upper().str.strip()

    # Prices prep
    px = prices_daily.copy()
    if "date" not in px.columns:
        raise ValueError("prices_daily must contain 'date' column.")
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["ticker"] = px["ticker"].astype(str).str.upper().str.strip()

    price_col = "adj_close" if "adj_close" in px.columns else ("close" if "close" in px.columns else None)
    if not price_col:
        raise ValueError("prices_daily must contain 'adj_close' or 'close' column.")

    # Fundamentals prep
    fq = fundamentals_quarterly.copy() if fundamentals_quarterly is not None else pd.DataFrame()
    if not fq.empty:
        if "ticker" not in fq.columns:
            raise ValueError("fundamentals_quarterly must contain 'ticker' column.")
        fq["ticker"] = fq["ticker"].astype(str).str.upper().str.strip()

        if "period_end" in fq.columns:
            fq["period_end"] = pd.to_datetime(fq["period_end"]).dt.normalize()
        elif "date" in fq.columns:
            # allow older format
            fq["period_end"] = pd.to_datetime(fq["date"]).dt.normalize()

        avail_col = _pick_available_date_col(fq)
        if avail_col is None:
            # fall back: assume period_end is available_date (conservative-ish but still no look-ahead if lagging in snapshot.py)
            fq["available_date"] = fq["period_end"]
            avail_col = "available_date"

        fq[avail_col] = pd.to_datetime(fq[avail_col]).dt.normalize()

    # Build factors
    rows: list[dict[str, Any]] = []

    # Pre-index prices by ticker for speed
    px_by_ticker = {t: g.sort_values("date") for t, g in px.groupby("ticker", sort=False)}

    # Pre-index fundamentals by ticker for speed
    fq_by_ticker = {t: g.sort_values("period_end") for t, g in fq.groupby("ticker", sort=False)} if not fq.empty else {}

    for _, srow in snaps.iterrows():
        asof_date = pd.to_datetime(srow["asof_date"]).normalize()
        ticker = str(srow["ticker"]).upper().strip()

        asset_type = str(srow.get("asset_type", "")).upper().strip() if pd.notna(srow.get("asset_type", np.nan)) else ""
        if asset_type == "":
            # default (if not provided) -> STOCK
            asset_type = "STOCK"
        theme = srow.get("theme", pd.NA)

        # Price history up to asof_date
        pxt = px_by_ticker.get(ticker)
        if pxt is None or pxt.empty:
            continue
        pxt_use = pxt[pxt["date"] <= asof_date]
        if pxt_use.empty:
            continue

        series = pxt_use.set_index("date")[price_col].astype(float).sort_index()

        rec: dict[str, Any] = {
            "asof_date": asof_date,
            "ticker": ticker,
            "asset_type": asset_type,
        }
        if pd.notna(theme):
            rec["theme"] = theme

        # --- Price factors (all assets) ---
        # 12m momentum skipping last 1m (approx 252 trading days, skip ~21)
        rec["mom_12m_1m"] = _momentum(series, lookback=252, skip=21)
        rec["mom_6m"] = _momentum(series, lookback=126, skip=0)
        rec["trend_200dma"] = _trend_200dma(series)
        rec["vol_1y"] = _volatility_1y(series)
        rec["maxdd_1y"] = _rolling_max_drawdown(series.tail(252))

        # --- Fundamental factors (stocks only) ---
        if asset_type != "ETF" and ticker in fq_by_ticker:
            fxt = fq_by_ticker[ticker]
            if not fxt.empty:
                # Use available_date (or chosen column) to avoid look-ahead
                avail_col = _pick_available_date_col(fxt) or "available_date"
                if avail_col not in fxt.columns:
                    # fall back
                    if "available_date" in fxt.columns:
                        avail_col = "available_date"
                    elif "period_end" in fxt.columns:
                        fxt = fxt.copy()
                        fxt["available_date"] = fxt["period_end"]
                        avail_col = "available_date"
                    else:
                        avail_col = None

                if avail_col:
                    fxt_use = fxt[pd.to_datetime(fxt[avail_col]).dt.normalize() <= asof_date].copy()
                else:
                    fxt_use = fxt.copy()

                if not fxt_use.empty:
                    fxt_use = fxt_use.sort_values("period_end")
                    latest = fxt_use.iloc[-1].to_dict()

                    # Map a few possible column variants (keep this forgiving)
                    revenue = latest.get("revenue", latest.get("totalRevenue", np.nan))
                    net_income = latest.get("net_income", latest.get("netIncome", np.nan))
                    fcf = latest.get("fcf", latest.get("freeCashFlow", np.nan))
                    ocf = latest.get("operating_cash_flow", latest.get("operatingCashFlow", np.nan))
                    total_assets = latest.get("total_assets", latest.get("totalAssets", np.nan))
                    total_debt = latest.get("total_debt", latest.get("totalDebt", np.nan))

                    latest_std = {
                        "revenue": revenue,
                        "net_income": net_income,
                        "fcf": fcf,
                        "operating_cash_flow": ocf,
                        "total_assets": total_assets,
                        "total_debt": total_debt,
                    }

                    rec["fcf_margin"] = _safe_div(latest_std["fcf"], latest_std["revenue"])
                    rec["profitability"] = _safe_div(latest_std["net_income"], latest_std["revenue"])
                    rec["accruals"] = _accruals(latest_std)
                    rec["leverage"] = _leverage(latest_std)

                    # YoY growth (use fxt_use so it has history)
                    # Create a simplified view with standardized fields if needed
                    f_for_growth = fxt_use.copy()
                    # standardize field names where possible
                    if "revenue" not in f_for_growth.columns and "totalRevenue" in f_for_growth.columns:
                        f_for_growth["revenue"] = f_for_growth["totalRevenue"]
                    if "net_income" not in f_for_growth.columns and "netIncome" in f_for_growth.columns:
                        f_for_growth["net_income"] = f_for_growth["netIncome"]
                    if "fcf" not in f_for_growth.columns and "freeCashFlow" in f_for_growth.columns:
                        f_for_growth["fcf"] = f_for_growth["freeCashFlow"]

                    rec["rev_yoy"] = _yoy_growth(f_for_growth, "revenue")
                    rec["ni_yoy"] = _yoy_growth(f_for_growth, "net_income")
                    rec["fcf_yoy"] = _yoy_growth(f_for_growth, "fcf")
                else:
                    # no usable fundamentals as of date
                    rec["fcf_margin"] = np.nan
                    rec["profitability"] = np.nan
                    rec["accruals"] = np.nan
                    rec["leverage"] = np.nan
                    rec["rev_yoy"] = np.nan
                    rec["ni_yoy"] = np.nan
                    rec["fcf_yoy"] = np.nan
        else:
            # ETF -> fundamentals intentionally NaN
            rec["fcf_margin"] = np.nan
            rec["profitability"] = np.nan
            rec["accruals"] = np.nan
            rec["leverage"] = np.nan
            rec["rev_yoy"] = np.nan
            rec["ni_yoy"] = np.nan
            rec["fcf_yoy"] = np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["asof_date"] = pd.to_datetime(out["asof_date"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()

    # keep tidy ordering
    key_cols = ["asof_date", "ticker"]
    meta_cols = [c for c in ["asset_type", "theme"] if c in out.columns]
    factor_cols = [c for c in out.columns if c not in (key_cols + meta_cols)]
    ordered = key_cols + meta_cols + sorted(factor_cols)
    ordered = [c for c in ordered if c in out.columns]

    return out[ordered].sort_values(["asof_date", "ticker"]).reset_index(drop=True)
