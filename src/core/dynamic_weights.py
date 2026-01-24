from __future__ import annotations

import numpy as np
import pandas as pd


BUCKETS = ["growth_score", "quality_score", "momentum_score", "risk_score"]
WCOLS = ["w_growth_score", "w_quality_score", "w_momentum_score", "w_risk_score"]


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize()


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
                rec[f"ic_{b}"] = float(rx[m].corr(ry[m]))

        if "overall_score_ai" in g.columns:
            x = g["overall_score_ai"].astype(float)
            rx = x.rank(pct=True) if method.lower() == "spearman" else x
            m = rx.notna() & ry.notna()
            rec["ic_overall_score_ai"] = float(rx[m].corr(ry[m])) if int(m.sum()) >= int(min_n) else np.nan

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

    Option B implementation:
      - Use EXPANDING mean IC (min_periods=3) for early months
      - Switch to ROLLING mean over last `window_months` once enough history exists
      - Clip negative IC to 0
      - Normalize to sum=1
      - Apply floor `min_weight` then renormalize
    """
    if monthly_ic.empty:
        return pd.DataFrame(columns=["asof_date"] + WCOLS)

    ic = monthly_ic.copy()
    ic["asof_date"] = _norm_date(ic["asof_date"])
    ic = ic.sort_values("asof_date").reset_index(drop=True)

    # Coerce IC columns to numeric (defensive against NaT/object contamination)
    ic_cols = []
    for b in BUCKETS:
        c = f"ic_{b}"
        ic[c] = pd.to_numeric(ic.get(c, np.nan), errors="coerce")
        ic_cols.append(c)

    ic_df = ic.set_index("asof_date")[ic_cols].copy()

    # Expanding mean for early periods (min 3 points)
    exp_mean = ic_df.expanding(min_periods=3).mean()

    # Rolling mean for mature periods
    roll_mean = ic_df.rolling(window=int(window_months), min_periods=3).mean()

    # Determine per-column when we have enough VALID points to use rolling
    # Use rolling count (min_periods=1) to measure available observations in last window
    roll_count = ic_df.rolling(window=int(window_months), min_periods=1).count()

    # Use rolling mean when count >= window_months, else expanding mean
    use_roll = roll_count >= float(window_months)
    blended = roll_mean.where(use_roll, exp_mean)

    roll = blended.reset_index()  # columns: asof_date + ic_* rolling/expanding blend

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

        # If all ICs are non-positive / missing, use equal weights
        if vals.sum() <= 0:
            w = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        else:
            w = vals / vals.sum()

        # Enforce minimum weight floor, then renormalize
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
    out["overall_score_ai"] = np.where(w_sum > 0, num / w_sum, np.nan)
    # --------------------------------------------------------------------------------------------

    # (optional) contribution columns (NaN-safe: missing bucket -> 0 contribution)
    for b, wcol in zip(BUCKETS, WCOLS):
        c = f"contrib_ai_{b}"
        if c not in out.columns:
            out[c] = np.nan_to_num(out[b].astype(float), nan=0.0) * out[wcol].astype(float)

    return out
