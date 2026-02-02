from __future__ import annotations

import numpy as np
import pandas as pd


BUCKETS = ["growth_score", "quality_score", "momentum_score", "risk_score"]
WCOLS = ["w_growth_score", "w_quality_score", "w_momentum_score", "w_risk_score"]


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize()



def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Return Pearson correlation or NaN without emitting divide warnings.
    - Requires at least 2 distinct values in each series after alignment.
    """
    m = x.notna() & y.notna()
    if int(m.sum()) < 2:
        return float("nan")
    xx = x[m].astype(float)
    yy = y[m].astype(float)
    # If either is constant -> corr is undefined (std==0)
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return float("nan")
    return float(xx.corr(yy))


def compute_forward_returns_monthly(
    scores: pd.DataFrame,
    prices_daily: pd.DataFrame,
    horizon_months: int = 1,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """
    Compute forward returns at monthly asof_dates for tickers in `scores`.

    Returns a DataFrame with:
      - asof_date
      - ticker
      - fwd_ret_{horizon_months}m
    """
    if scores.empty or prices_daily.empty:
        return pd.DataFrame(columns=["asof_date", "ticker", f"fwd_ret_{horizon_months}m"])

    s = scores[["asof_date", "ticker"]].copy()
    s["asof_date"] = _norm_date(s["asof_date"])
    s["ticker"] = s["ticker"].astype(str)

    p = prices_daily.copy()
    # accept either 'date' or 'Date'
    if "date" in p.columns:
        p["date"] = _norm_date(p["date"])
        dcol = "date"
    elif "Date" in p.columns:
        p["Date"] = _norm_date(p["Date"])
        dcol = "Date"
    else:
        raise ValueError("prices_daily must contain a 'date' (or 'Date') column")

    if "ticker" not in p.columns:
        raise ValueError("prices_daily must contain a 'ticker' column")
    if price_col not in p.columns:
        raise ValueError(f"prices_daily must contain '{price_col}' column")

    p["ticker"] = p["ticker"].astype(str)

    # Keep only necessary rows for tickers in universe + sort
    tickers = s["ticker"].unique().tolist()
    p = p[p["ticker"].isin(tickers)].sort_values(["ticker", dcol]).copy()

    # Build a monthly price series by taking last available price on each asof_date
    px = p[[dcol, "ticker", price_col]].rename(columns={dcol: "asof_date", price_col: "px"})
    merged = s.merge(px, on=["asof_date", "ticker"], how="left")

    # For robustness: if exact asof_date isn't present in prices, merge_asof backward per ticker.
    miss = merged["px"].isna().mean()
    if miss > 0:
        out_rows = []
        for t, g in s.groupby("ticker"):
            gt = g.sort_values("asof_date").copy()
            pt = (
                p[p["ticker"] == t]
                .sort_values(dcol)[[dcol, price_col]]
                .rename(columns={dcol: "date", price_col: "px"})
            )
            if pt.empty:
                gt["px"] = np.nan
            else:
                gt = pd.merge_asof(
                    gt.sort_values("asof_date"),
                    pt.sort_values("date"),
                    left_on="asof_date",
                    right_on="date",
                    direction="backward",
                ).drop(columns=["date"])
            out_rows.append(gt)
        merged = pd.concat(out_rows, ignore_index=True)

    merged = merged.sort_values(["ticker", "asof_date"]).reset_index(drop=True)
    merged["px_fwd"] = merged.groupby("ticker")["px"].shift(-horizon_months)
    merged[f"fwd_ret_{horizon_months}m"] = (merged["px_fwd"] / merged["px"]) - 1.0

    return merged[["asof_date", "ticker", f"fwd_ret_{horizon_months}m"]]


def compute_forward_returns_by_months(
    scores: pd.DataFrame,
    prices_daily: pd.DataFrame,
    horizon_months: int,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Compute forward returns using a *true* date offset.

    This is required when your `scores` are not strictly monthly (e.g. weekly).
    For each (asof_date, ticker), the forward price is taken on the first
    available trading day **on or after** (asof_date + horizon_months).

    Output columns:
      - asof_date
      - ticker
      - fwd_ret_{horizon_months}m
    """
    if scores.empty or prices_daily.empty:
        return pd.DataFrame(columns=["asof_date", "ticker", f"fwd_ret_{horizon_months}m"])

    s = scores[["asof_date", "ticker"]].copy()
    s["asof_date"] = _norm_date(s["asof_date"])
    s["ticker"] = s["ticker"].astype(str)
    s["target_date"] = s["asof_date"] + pd.DateOffset(months=int(horizon_months))

    p = prices_daily.copy()
    if "date" in p.columns:
        p["date"] = _norm_date(p["date"])
        dcol = "date"
    elif "Date" in p.columns:
        p["Date"] = _norm_date(p["Date"])
        dcol = "Date"
    else:
        raise ValueError("prices_daily must contain a 'date' (or 'Date') column")

    if "ticker" not in p.columns:
        raise ValueError("prices_daily must contain a 'ticker' column")
    if price_col not in p.columns:
        raise ValueError(f"prices_daily must contain '{price_col}' column")
    p["ticker"] = p["ticker"].astype(str)

    tickers = s["ticker"].unique().tolist()
    p = p[p["ticker"].isin(tickers)].sort_values(["ticker", dcol]).copy()

    # Build per-ticker forward merge (direction='forward')
    out_rows = []
    for t, g in s.groupby("ticker"):
        gt = g.sort_values("asof_date").copy()
        pt = (
            p[p["ticker"] == t]
            .sort_values(dcol)[[dcol, price_col]]
            .rename(columns={dcol: "date", price_col: "px"})
        )
        if pt.empty:
            gt["px0"] = np.nan
            gt["px_fwd"] = np.nan
        else:
            # px0: last available on/before asof_date
            px0 = pd.merge_asof(
                gt[["asof_date"]].sort_values("asof_date"),
                pt.sort_values("date"),
                left_on="asof_date",
                right_on="date",
                direction="backward",
            )["px"].to_numpy()

            # px_fwd: first available on/after target_date
            pxf = pd.merge_asof(
                gt[["target_date"]].sort_values("target_date"),
                pt.sort_values("date"),
                left_on="target_date",
                right_on="date",
                direction="forward",
            )["px"].to_numpy()

            gt["px0"] = px0
            gt["px_fwd"] = pxf
        out_rows.append(gt[["asof_date", "ticker", "px0", "px_fwd"]])

    merged = pd.concat(out_rows, ignore_index=True)
    merged[f"fwd_ret_{horizon_months}m"] = (merged["px_fwd"] / merged["px0"]) - 1.0
    return merged[["asof_date", "ticker", f"fwd_ret_{horizon_months}m"]]


def compute_dynamic_bucket_weights_periodic(
    ic_by_period: pd.DataFrame,
    window_periods: int = 52,
    min_weight: float = 0.05,
) -> pd.DataFrame:
    """Periodic version of compute_dynamic_bucket_weights.

    The original function's `window_months` parameter is really a *row window*.
    For weekly engines, pass `window_periods=52` (approx 1 year).
    """
    return compute_dynamic_bucket_weights(
        ic_by_period,
        window_months=int(window_periods),
        min_weight=float(min_weight),
    )


def compute_monthly_ic(
    scores: pd.DataFrame,
    fwd: pd.DataFrame,
    method: str = "spearman",
    min_n: int = 5,
) -> pd.DataFrame:
    """
    Compute cross-sectional IC per month for each bucket.
    Spearman is implemented as Pearson correlation on percentile ranks (no SciPy needed).
    """
    if scores.empty or fwd.empty:
        cols = ["asof_date", "n"] + [f"ic_{b}" for b in BUCKETS]
        return pd.DataFrame(columns=cols)

    s = scores.copy()
    s["asof_date"] = _norm_date(s["asof_date"])
    s["ticker"] = s["ticker"].astype(str)

    f = fwd.copy()
    f["asof_date"] = _norm_date(f["asof_date"])
    f["ticker"] = f["ticker"].astype(str)

    fwd_cols = [c for c in f.columns if c.startswith("fwd_ret_")]
    if not fwd_cols:
        raise ValueError("fwd must contain a forward return column named like 'fwd_ret_1m'")
    ycol = fwd_cols[0]

    df = s.merge(f[["asof_date", "ticker", ycol]], on=["asof_date", "ticker"], how="left")
    df = df.dropna(subset=[ycol])

    # Exclude ETFs from IC estimation (IC should be based on single-name equities only).
    # If asset_type is present, keep only STOCK rows for IC calc.
    if "asset_type" in df.columns:
        df = df[df["asset_type"].astype(str).str.upper() == "STOCK"].copy()


    recs = []
    for d, g in df.groupby("asof_date"):
        y = g[ycol].astype(float)
        rec = {"asof_date": d, "n": int(y.notna().sum())}

        if method.lower() == "spearman":
            ry = y.rank(pct=True)
        else:
            ry = y

        for b in BUCKETS:
            if b not in g.columns:
                rec[f"ic_{b}"] = np.nan
                continue

            x = g[b].astype(float)
            rx = x.rank(pct=True) if method.lower() == "spearman" else x

            m = rx.notna() & ry.notna()
            if int(m.sum()) < int(min_n):
                rec[f"ic_{b}"] = np.nan
            else:
                rec[f"ic_{b}"] = _safe_corr(rx, ry)

        if "overall_score_ai" in g.columns:
            x = g["overall_score_ai"].astype(float)
            rx = x.rank(pct=True) if method.lower() == "spearman" else x
            m = rx.notna() & ry.notna()
            rec["ic_overall_score_ai"] = _safe_corr(rx, ry) if int(m.sum()) >= int(min_n) else np.nan

        recs.append(rec)

    out = pd.DataFrame(recs).sort_values("asof_date").reset_index(drop=True)
    return out


def compute_dynamic_bucket_weights(
    monthly_ic: pd.DataFrame,
    window_months: int = 24,
    min_weight: float = 0.05,
) -> pd.DataFrame:
    """
    Convert monthly IC history into dynamic bucket weights per month.

    - rolling mean IC over last `window_months`
    - clip negative IC to 0
    - normalize to sum=1
    - apply a floor `min_weight` then renormalize
    """
    if monthly_ic.empty:
        return pd.DataFrame(columns=["asof_date"] + WCOLS)

    ic = monthly_ic.copy()
    ic["asof_date"] = _norm_date(ic["asof_date"])
    ic = ic.sort_values("asof_date").reset_index(drop=True)

    # Coerce IC columns to numeric (defensive against NaT/object contamination)
    for b in BUCKETS:
        c = f"ic_{b}"
        ic[c] = pd.to_numeric(ic.get(c, np.nan), errors="coerce")

    roll = (
        ic.set_index("asof_date")[[f"ic_{b}" for b in BUCKETS]]
        .rolling(window=int(window_months), min_periods=3)
        .mean()
        .reset_index()
    )

    w_recs = []
    for _, r in roll.iterrows():
        d = r["asof_date"]
        vals = np.array(
            [pd.to_numeric(r.get(f"ic_{b}", np.nan), errors="coerce") for b in BUCKETS],
            dtype=float,
        )

        vals = np.where(np.isfinite(vals), vals, np.nan)
        vals = np.nan_to_num(vals, nan=0.0)
        vals = np.clip(vals, 0.0, None)

        if vals.sum() <= 0:
            w = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        else:
            w = vals / vals.sum()

        w = np.maximum(w, float(min_weight))
        w = w / w.sum()

        w_recs.append(
            {
                "asof_date": d,
                "w_growth_score": float(w[0]),
                "w_quality_score": float(w[1]),
                "w_momentum_score": float(w[2]),
                "w_risk_score": float(w[3]),
            }
        )

    out = pd.DataFrame(w_recs).sort_values("asof_date").reset_index(drop=True)
    return out


def apply_dynamic_overall_score(
    scores: pd.DataFrame,
    weights_by_month: pd.DataFrame,
    ffill_weights: bool = True,
) -> pd.DataFrame:
    """
    Attach dynamic weights and compute overall_score_ai WITHOUT shrinking the score table.

    - LEFT-merge weights onto scores by asof_date
    - Optionally reindex weights to all score asof_dates and forward-fill missing months
    - If weights missing (early months), fill equal weights

    Fix 2:
      - Compute overall_score_ai in a NaN-safe way:
        renormalize weights over available (non-NaN) bucket scores per row.
    """
    s = scores.copy()
    if s.empty:
        return s

    s["asof_date"] = _norm_date(s["asof_date"])

    # ensure bucket columns exist
    for b in BUCKETS:
        if b not in s.columns:
            s[b] = np.nan

    if weights_by_month is None or weights_by_month.empty:
        if "overall_score" in s.columns:
            s["overall_score_ai"] = s["overall_score"]
        else:
            s["overall_score_ai"] = pd.NA
        for c in WCOLS:
            s[c] = 0.25
        return s

    w = weights_by_month.copy()
    w["asof_date"] = _norm_date(w["asof_date"])
    w = w.sort_values("asof_date").drop_duplicates("asof_date")

    keep = ["asof_date"] + [c for c in WCOLS if c in w.columns]
    w = w[keep].copy()
    for c in WCOLS:
        if c not in w.columns:
            w[c] = np.nan

    if ffill_weights:
        idx = pd.Index(sorted(s["asof_date"].dropna().unique()))
        w = (
            w.set_index("asof_date")
            .reindex(idx)
            .ffill()
            .reset_index()
            .rename(columns={"index": "asof_date"})
        )

    for c in WCOLS:
        w[c] = w[c].astype(float).fillna(0.25)

    out = s.merge(w, on="asof_date", how="left")

    for c in WCOLS:
        out[c] = out[c].astype(float).fillna(0.25)

    # -------- Fix 2: NaN-safe AI overall score (renormalize weights over available buckets) --------
    bmat = out[BUCKETS].astype(float)
    wmat = out[WCOLS].astype(float)

    mask = bmat.notna().astype(float)          # 1 where bucket exists
    w_eff = wmat.values * mask.values          # zero weight where bucket missing
    w_sum = w_eff.sum(axis=1)

    num = (np.nan_to_num(bmat.values, nan=0.0) * w_eff).sum(axis=1)
    overall = np.full(num.shape, np.nan, dtype="float64")
    np.divide(num, w_sum, out=overall, where=(w_sum > 0) & np.isfinite(w_sum))
    out["overall_score_ai"] = overall
    # --------------------------------------------------------------------------------------------

    # (optional) contribution columns (NaN-safe: missing bucket -> 0 contribution)
    for b, wcol in zip(BUCKETS, WCOLS):
        c = f"contrib_ai_{b}"
        if c not in out.columns:
            out[c] = np.nan_to_num(out[b].astype(float), nan=0.0) * out[wcol].astype(float)

    return out
