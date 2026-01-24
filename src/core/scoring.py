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
        present = gg[all_factor_scores].notna().sum(axis=1)
        gg["data_completeness"] = present / float(len(all_factor_scores))

        # overall score
        overall = 0.0
        wsum = 0.0
        for bucket, w in _WEIGHTS.items():
            col = f"{bucket.lower()}_score"
            if col in gg.columns:
                overall += w * gg[col].fillna(gg[col].mean())
                wsum += w
        gg["overall_score"] = overall / wsum if wsum > 0 else np.nan

        scored.append(gg)

    out = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    # keep key columns tidy
    keep_cols = ["asof_date", "ticker", "overall_score", "data_completeness"] + \
                [f"{b.lower()}_score" for b in _BUCKETS.keys()]
    # plus raw factors for debugging
    raw_factor_cols = [c for c in df.columns if c not in ["asof_date", "ticker"]]
    keep_cols += raw_factor_cols
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].sort_values(["asof_date", "overall_score"], ascending=[True, False]).reset_index(drop=True)
