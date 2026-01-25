from __future__ import annotations

import numpy as np
import pandas as pd

def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _standardize_prices(prices_daily: pd.DataFrame) -> pd.DataFrame:
    df = prices_daily.copy()
    if "date" not in df.columns:
        raise KeyError("prices_daily must contain a 'date' column")
    if "ticker" not in df.columns:
        raise KeyError("prices_daily must contain a 'ticker' column")

    close_col = _first_existing(df, ["close", "adj_close", "c"])
    vol_col = _first_existing(df, ["volume", "v", "vol"])

    if close_col is None:
        raise KeyError(f"prices_daily missing close column. Found: {list(df.columns)}")
    if vol_col is None:
        raise KeyError(f"prices_daily missing volume column. Found: {list(df.columns)}")

    if close_col != "close":
        df = df.rename(columns={close_col: "close"})
    if vol_col != "volume":
        df = df.rename(columns={vol_col: "volume"})

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df[["ticker", "date", "close", "volume"]].dropna(subset=["ticker", "date"])

def _standardize_fundamentals(fund_q: pd.DataFrame) -> pd.DataFrame:
    df = fund_q.copy()
    if "ticker" not in df.columns:
        raise KeyError("fundamentals_quarterly must contain a 'ticker' column")

    period_col = _first_existing(df, ["period_end", "date", "fiscalDateEnding", "reportDate"])
    if period_col is None:
        raise KeyError("fundamentals_quarterly missing a period end date column (e.g., 'period_end').")

    if period_col != "period_end":
        df = df.rename(columns={period_col: "period_end"})

    df["period_end"] = pd.to_datetime(df["period_end"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df

def _compute_quarterly_ttm(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    df = df.sort_values(["ticker", "period_end"]).copy()
    df[out_col] = (
        df.groupby("ticker")[value_col]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return df

def _asof_merge_per_ticker(left_asof: pd.DataFrame, right_ts: pd.DataFrame, right_on: str, cols: list[str]) -> pd.DataFrame:
    out_parts = []
    for t, g_left in left_asof.groupby("ticker", sort=False):
        g_right = right_ts[right_ts["ticker"] == t]
        tmp = g_left.copy()

        if g_right.empty:
            for c in cols:
                tmp[c] = np.nan
            out_parts.append(tmp)
            continue

        g_left = tmp.sort_values("asof_date")
        g_right = g_right.sort_values(right_on)

        merged = pd.merge_asof(
            g_left,
            g_right[[right_on] + cols],
            left_on="asof_date",
            right_on=right_on,
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.drop(columns=[right_on], errors="ignore")
        out_parts.append(merged)

    return pd.concat(out_parts, ignore_index=True)

def build_metrics_monthly(
    asof_dates: list[pd.Timestamp],
    tickers: list[str],
    prices_daily: pd.DataFrame,
    fundamentals_quarterly: pd.DataFrame,
    scores_monthly: pd.DataFrame | None = None,
    rolling_days: int = 20,
) -> pd.DataFrame:
    """
    Month-end UI cache: valuation + liquidity (+ optional AI score).

    Robustness:
      - Always creates standardized columns: shares_outstanding, total_equity, net_income_ttm
        even if fundamentals source does not provide them (filled with NaN).
    """
    if not asof_dates:
        raise ValueError("asof_dates is empty")
    if not tickers:
        raise ValueError("tickers is empty")

    asof_dates = [pd.to_datetime(d).normalize() for d in asof_dates]
    tickers_u = pd.Series(tickers, dtype=str).str.upper().str.strip().unique().tolist()

    grid = pd.DataFrame(
        {
            "asof_date": np.repeat(asof_dates, len(tickers_u)),
            "ticker": np.tile(tickers_u, len(asof_dates)),
        }
    )

    # ---- Daily (price/liquidity) ----
    px = _standardize_prices(prices_daily)
    px = px[px["ticker"].isin(tickers_u)].sort_values(["ticker", "date"]).copy()
    px["dollar_volume"] = px["close"] * px["volume"]
    px["vol_avg_20d"] = (
        px.groupby("ticker")["volume"].rolling(rolling_days, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    px["dvol_avg_20d"] = (
        px.groupby("ticker")["dollar_volume"].rolling(rolling_days, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    daily_cols = ["close", "volume", "vol_avg_20d", "dvol_avg_20d"]
    grid_px = _asof_merge_per_ticker(
        left_asof=grid[["ticker", "asof_date"]],
        right_ts=px.rename(columns={"date": "ts_date"}),
        right_on="ts_date",
        cols=daily_cols,
    )
    out = grid.merge(grid_px, on=["ticker", "asof_date"], how="left")

    # ---- Fundamentals ----
    fund = _standardize_fundamentals(fundamentals_quarterly)
    fund = fund[fund["ticker"].isin(tickers_u)].copy()

    shares_col = _first_existing(
        fund,
        [
            "shares_outstanding",
            "sharesOutstanding",
            "weightedAverageShsOutDil",
            "weightedAverageShsOut",
            "weightedAverageShsOutDiluted",
            "weightedAverageShsOutBasic",
            "commonStockSharesOutstanding",
            "commonStockSharesIssued",
            "numberOfShares",
        ],
    )
    equity_col = _first_existing(
        fund,
        [
            "total_stockholders_equity",
            "totalStockholdersEquity",
            "totalEquity",
            "stockholdersEquity",
            "totalShareholdersEquity",
        ],
    )
    netinc_col = _first_existing(
        fund,
        [
            "net_income",
            "netIncome",
            "netIncomeApplicableToCommonShares",
            "netIncomeCommonStockholders",
            "netIncomeFromContinuingOperations",
        ],
    )

    if netinc_col is not None:
        fund = _compute_quarterly_ttm(fund, netinc_col, "net_income_ttm")
    else:
        fund["net_income_ttm"] = np.nan

    tmp = fund[["ticker", "period_end"]].copy()
    tmp["shares_outstanding"] = pd.to_numeric(fund[shares_col], errors="coerce") if shares_col else np.nan
    tmp["total_equity"] = pd.to_numeric(fund[equity_col], errors="coerce") if equity_col else np.nan
    tmp["net_income_ttm"] = pd.to_numeric(fund["net_income_ttm"], errors="coerce")

    grid_f = _asof_merge_per_ticker(
        left_asof=grid[["ticker", "asof_date"]],
        right_ts=tmp.rename(columns={"period_end": "ts_date"}),
        right_on="ts_date",
        cols=["shares_outstanding", "total_equity", "net_income_ttm"],
    )
    out = out.merge(grid_f, on=["ticker", "asof_date"], how="left")

    # Hard guarantee columns exist (prevents KeyError)
    for c in ["shares_outstanding", "total_equity", "net_income_ttm"]:
        if c not in out.columns:
            out[c] = np.nan

    # ---- Derived ----
    out["market_cap"] = out["close"] * out["shares_outstanding"]

    out["dollar_turnover_20d_avg"] = out["dvol_avg_20d"] / out["market_cap"]
    out["share_turnover_20d_avg"] = out["vol_avg_20d"] / out["shares_outstanding"]

    out["book_value_per_share"] = out["total_equity"] / out["shares_outstanding"]
    out["pb"] = out["close"] / out["book_value_per_share"]

    out["eps_ttm"] = out["net_income_ttm"] / out["shares_outstanding"]
    out["pe_ttm"] = out["close"] / out["eps_ttm"]

    # ---- Optional join scores ----
    if scores_monthly is not None and not scores_monthly.empty:
        sc = scores_monthly.copy()
        sc["asof_date"] = pd.to_datetime(sc["asof_date"]).dt.normalize()
        sc["ticker"] = sc["ticker"].astype(str).str.upper().str.strip()

        keep_cols = [c for c in ["overall_score_ai", "overall_score", "bucket", "rank"] if c in sc.columns]
        if keep_cols:
            sc_small = sc[["asof_date", "ticker"] + keep_cols].drop_duplicates(
                subset=["asof_date", "ticker"], keep="last"
            )
            out = out.merge(sc_small, on=["asof_date", "ticker"], how="left")

    front = [
        "asof_date","ticker",
        "close","volume",
        "market_cap","pe_ttm","pb",
        "vol_avg_20d","dvol_avg_20d",
        "share_turnover_20d_avg","dollar_turnover_20d_avg",
        "shares_outstanding","total_equity","net_income_ttm","eps_ttm","book_value_per_share",
    ]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    return out.sort_values(["asof_date", "ticker"]).reset_index(drop=True)
