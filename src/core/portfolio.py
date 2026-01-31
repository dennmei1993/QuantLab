from __future__ import annotations

import numpy as np
import pandas as pd


def build_monthly_portfolio(
    scores: pd.DataFrame,
    top_n: int,
    benchmark_ticker: str,
    score_col: str = "overall_score_ai",
    ai_min_coverage: float = 0.80,
) -> pd.DataFrame:
    """
    Build a monthly Top-N equal-weight portfolio with trade actions.

    Option 2 (dynamic score selection):
      - Use `score_col` (default overall_score_ai) only if its non-null coverage >= ai_min_coverage for that month.
      - Otherwise fall back to `overall_score`.

    Robust fallback:
      - If the chosen score column doesn't have enough non-null values to form top_n,
        fall back to other usable columns (momentum_score, risk_score, etc).

    Output includes:
      - BUY/HOLD rows for current holdings (weight > 0)
      - SELL rows for positions removed (weight = 0)
      - ai_coverage: month-level coverage of AI score column in the full universe
      - score_col_used: the final column actually used to rank/choose names that month
    """
    df = scores.copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "asof_date",
            "ticker",
            "weight",
            "score_used",
            "score_col_used",
            "index",
        ])

    # Normalize key fields
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)

    # Defensive: coerce potential score columns to numeric (NaT/object -> NaN)
    numeric_candidates = [
        "overall_score",
        score_col,
        "overall_score_ai",
        "growth_score",
        "quality_score",
        "momentum_score",
        "risk_score",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    asof_dates = sorted(df["asof_date"].dropna().unique())
    rows: list[dict] = []

    prev_weights: dict[str, float] = {}

    def _non_null(g: pd.DataFrame, col: str) -> int:
        return int(g[col].notna().sum()) if (col in g.columns) else 0

    for asof_date in asof_dates:
        g_all = df[df["asof_date"] == asof_date].copy()
        if g_all.empty:
            continue

        ai_col = score_col
        static_col = "overall_score"

        # Month-level AI coverage on FULL universe for that month
        if ai_col in g_all.columns:
            ai_coverage = float(g_all[ai_col].notna().mean())
        else:
            ai_coverage = np.nan

        # Option 2: choose preferred column first
        preferred_col = ai_col if (ai_col in g_all.columns and ai_coverage >= float(ai_min_coverage)) else static_col

        # Robust: pick the first column that can actually produce top_n names this month
        # (priority order matters)
        candidates = [
            preferred_col,
            "overall_score_ai",
            "overall_score",
            "momentum_score",
            "risk_score",
            "quality_score",
            "growth_score",
        ]

        needed = int(top_n)

        # Pick a usable column.
        # Prefer a column that can fill top_n, but if none can, take the one with the MOST coverage (>0).
        use_col = None
        best_col = None
        best_non_null = 0

        for c in candidates:
            nn = _non_null(g_all, c)
            if nn >= needed:
                use_col = c
                best_non_null = nn
                break
            if nn > best_non_null:
                best_non_null = nn
                best_col = c

        if use_col is None:
            use_col = best_col  # may still be None

        if use_col is None or best_non_null == 0:
            # Nothing usable this month — skip safely
            continue

        # Build portfolio with as many names as available up to top_n
        n_select = min(needed, best_non_null)

        g = (
            g_all.dropna(subset=[use_col])
            .sort_values(use_col, ascending=False)
            .head(n_select)
            .copy()
        )
        if g.empty:
            continue


        # Equal weights
        w = 1.0 / len(g)
        curr_weights = {str(t): w for t in g["ticker"].astype(str).tolist()}

        prev_set = set(prev_weights.keys())
        curr_set = set(curr_weights.keys())

        buys = curr_set - prev_set
        sells = prev_set - curr_set

        # BUY/HOLD rows
        for _, r in g.iterrows():
            t = str(r["ticker"])
            wt = float(curr_weights.get(t, 0.0))
            prev_wt = float(prev_weights.get(t, 0.0))

            action = "BUY" if t in buys else "HOLD"

            score_val = r.get(use_col)
            rows.append(
                {
                    "asof_date": asof_date,
                    "ticker": t,
                    "action": action,
                    "weight": wt,
                    "prev_weight": prev_wt,
                    "weight_change": wt - prev_wt,
                    "abs_weight_change": abs(wt - prev_wt),
                    "score": float(score_val) if pd.notna(score_val) else pd.NA,
                    "score_col_used": use_col,
                    "ai_coverage": ai_coverage,
                }
            )

        # SELL rows
        for t in sorted(sells):
            prev_wt = float(prev_weights.get(t, 0.0))
            rows.append(
                {
                    "asof_date": asof_date,
                    "ticker": t,
                    "action": "SELL",
                    "weight": 0.0,
                    "prev_weight": prev_wt,
                    "weight_change": -prev_wt,
                    "abs_weight_change": abs(prev_wt),
                    "score": pd.NA,
                    "score_col_used": use_col,
                    "ai_coverage": ai_coverage,
                }
            )

        prev_weights = curr_weights

    port = pd.DataFrame(rows)
    if port.empty:
        return port

    port["asof_date"] = pd.to_datetime(port["asof_date"]).dt.normalize()

    # Turnover estimate per month (sum abs weight changes)
    turnover = (
        port.groupby("asof_date", as_index=False)["abs_weight_change"]
        .sum()
        .rename(columns={"abs_weight_change": "turnover_estimate"})
    )
    port = port.merge(turnover, on="asof_date", how="left")

    # Sort for readability
    action_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    port["action_order"] = port["action"].map(action_order).fillna(9).astype(int)

    port = (
        port.sort_values(["asof_date", "action_order", "ticker"])
        .drop(columns=["action_order"])
        .reset_index(drop=True)
    )

    return port
