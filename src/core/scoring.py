from __future__ import annotations

import pandas as pd
import numpy as np


_BUCKETS = {
    "Momentum": ["mom_12m_1m", "mom_6m", "trend_200dma"],
    "Quality": ["fcf_margin", "profitability", "accruals", "leverage"],
    "Growth": ["rev_yoy", "ni_yoy", "fcf_yoy"],
    "Risk": ["vol_1y", "maxdd_1y"],
}

# For these factors, lower is better (so we reverse ranks)
_LOWER_BETTER = {"vol_1y", "maxdd_1y", "leverage", "accruals"}

# Overall weights (alpha-seeking tech tilt)
_WEIGHTS = {"Growth": 0.30, "Quality": 0.30, "Momentum": 0.25, "Risk": 0.15}

_ETF_WEIGHTS = {"Momentum": 0.65, "Risk": 0.35}



def _winsorize(s: pd.Series, p_lo=0.01, p_hi=0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(p_lo)
    hi = s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)


def _rank_0_100(s: pd.Series, higher_better: bool = True) -> pd.Series:
    # rank pct in [0,1], then scale to [0,100]
    r = s.rank(pct=True, method="average")
    if not higher_better:
        r = 1.0 - r
    return r * 100.0

def score_snapshots(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Input: one row per (asof_date,ticker) with factor columns.
    Output: adds per-factor scores, bucket scores, overall_score, data_completeness.
    """
    df = factors.copy()
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.normalize()

    # Per-month cross-sectional scoring
    scored = []
    for asof_date, g in df.groupby("asof_date", sort=True):
        gg = g.copy()

        # factor scores
        for bucket, cols in _BUCKETS.items():
            for c in cols:
                if c not in gg.columns:
                    gg[c] = np.nan
                v = _winsorize(gg[c].astype(float))
                higher_better = (c not in _LOWER_BETTER)
                gg[f"score_{c}"] = _rank_0_100(v, higher_better=higher_better)

        # bucket scores (mean of available factor scores)
        for bucket, cols in _BUCKETS.items():
            score_cols = [f"score_{c}" for c in cols]
            gg[f"{bucket.lower()}_score"] = gg[score_cols].mean(axis=1, skipna=True)

        # data completeness: percent of factor scores present (not NaN)
        
        all_factor_scores = [f"score_{c}" for cols in _BUCKETS.values() for c in cols]
        price_only_factors = _BUCKETS["Momentum"] + _BUCKETS["Risk"]
        price_only_scores = [f"score_{c}" for c in price_only_factors]

        # default completeness = all factors
        present_all = gg[all_factor_scores].notna().sum(axis=1) if all_factor_scores else 0
        den_all = float(len(all_factor_scores))
        comp_all = present_all / den_all if den_all > 0 else np.nan

        # ETF completeness = price-only factors
        price_only_scores = [c for c in price_only_scores if c in gg.columns]
        present_po = gg[price_only_scores].notna().sum(axis=1) if price_only_scores else 0
        den_po = float(len(price_only_scores))
        comp_po = present_po / den_po if den_po > 0 else np.nan

        if "asset_type" in gg.columns:
            is_etf = gg["asset_type"].astype(str).str.upper().eq("ETF")
            gg["data_completeness"] = np.where(is_etf, comp_po, comp_all)
        else:
            gg["data_completeness"] = comp_all

        # ----------------------------
        # overall score (STOCK vs ETF)
        # ----------------------------
        bucket_cols = {bucket: f"{bucket.lower()}_score" for bucket in _BUCKETS.keys()}

        # STOCK overall score: current behavior (fills missing with cross-sectional mean)
        stock_num = 0.0
        stock_den = 0.0
        for bucket, w in _WEIGHTS.items():
            col = bucket_cols.get(bucket)
            if col and col in gg.columns:
                stock_num += w * gg[col].fillna(gg[col].mean())
                stock_den += w
        gg["overall_score_stock"] = stock_num / stock_den if stock_den > 0 else np.nan

        # ETF overall score: use ETF weights and DO NOT fill missing buckets
        etf_num = pd.Series(0.0, index=gg.index, dtype="float64")
        etf_den = pd.Series(0.0, index=gg.index, dtype="float64")
        for bucket, w in _ETF_WEIGHTS.items():
            col = bucket_cols.get(bucket)
            if col and col in gg.columns:
                vals = gg[col].astype(float)
                m = vals.notna()
                etf_num.loc[m] += w * vals.loc[m]
                etf_den.loc[m] += w
        gg["overall_score_etf"] = np.where(etf_den > 0, etf_num / etf_den, np.nan)

        # Choose: ETFs use ETF score, otherwise stock score
        if "asset_type" in gg.columns:
            is_etf = gg["asset_type"].astype(str).str.upper().eq("ETF")
            gg["overall_score"] = np.where(is_etf, gg["overall_score_etf"], gg["overall_score_stock"])
        else:
            gg["overall_score"] = gg["overall_score_stock"]

        scored.append(gg)

    out = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()

    # keep key columns tidy
    keep_cols = (
        ["asof_date", "ticker", "overall_score", "data_completeness"]
        + [f"{b.lower()}_score" for b in _BUCKETS.keys()]
        + [c for c in ["asset_type", "theme"] if c in out.columns]
        + [c for c in ["overall_score_stock", "overall_score_etf"] if c in out.columns]
    )

    # plus raw factors for debugging (only those that exist in the final dataframe)
    raw_factor_cols = [c for c in df.columns if c not in ["asof_date", "ticker"]]
    keep_cols = list(keep_cols) + [c for c in raw_factor_cols if c in out.columns]
    # de-duplicate while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    return (
        out[keep_cols]
        .sort_values(["asof_date", "overall_score"], ascending=[True, False])
        .reset_index(drop=True)
    )