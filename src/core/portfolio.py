from __future__ import annotations

import numpy as np
import pandas as pd

def _build_alert_score_map(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw alerts into per-(asof_date,ticker) scoring + counts + type summary.
    Expected alerts_df columns: asof_date, ticker, alert_type, severity
    """
    if alerts_df is None or alerts_df.empty:
        return pd.DataFrame(columns=["asof_date", "ticker", "alert_score", "alert_count", "alert_types"])

    a = alerts_df.copy()
    a["asof_date"] = pd.to_datetime(a["asof_date"]).dt.normalize()
    a["ticker"] = a["ticker"].astype(str)

    # Score weights (tune later)
    type_w = {
        "MA_BREAK_200": 3,
        "MA_BREAK_50": 2,
        "PRICE_MOVE_5D": 2,
        "PRICE_MOVE_1D": 1,
        "VOLUME_SPIKE": 1,
    }
    sev_w = {"high": 2, "med": 1.25, "low": 1.0}

    a["type_w"] = a["alert_type"].map(type_w).fillna(1.0)
    a["sev_w"] = a["severity"].map(sev_w).fillna(1.0)
    a["score"] = a["type_w"] * a["sev_w"]

    # Per ticker per day/month score
    score = (
        a.groupby(["asof_date", "ticker"], as_index=False)["score"]
        .sum()
        .rename(columns={"score": "alert_score"})
    )
    count = (
        a.groupby(["asof_date", "ticker"], as_index=False)
        .size()
        .rename(columns={"size": "alert_count"})
    )
    # compact type list (top 5)
    type_counts = (
        a.groupby(["asof_date", "ticker", "alert_type"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["asof_date", "ticker", "n"], ascending=[True, True, False])
    )
    types = (
        type_counts.groupby(["asof_date", "ticker"])
        .apply(lambda g: ",".join([f"{r.alert_type}:{int(r.n)}" for _, r in g.head(5).iterrows()]))
        .rename("alert_types")
        .reset_index()
    )

    out = score.merge(count, on=["asof_date", "ticker"], how="left").merge(types, on=["asof_date", "ticker"], how="left")
    return out


def apply_alert_overlay_to_portfolio(
    port: pd.DataFrame,
    alerts_df: pd.DataFrame,
    buy_dampen_threshold: float = 3.0,
    buy_dampen_mult: float = 0.5,
    sell_accel_threshold: float = 4.0,
    sell_accel_mult: float = 1.25,
    block_buy_on_ma200: bool = True,
) -> pd.DataFrame:
    """
    Adds alert columns and recommends overlay weights based on alerts.

    Rules:
      - If alert_score >= buy_dampen_threshold and action==BUY: reduce weight_change by buy_dampen_mult.
      - If alert_score >= sell_accel_threshold and action==SELL: increase sell magnitude by sell_accel_mult (bounded).
      - Optionally block BUY if MA_BREAK_200 exists for that ticker on asof_date.

    Returns port with extra columns:
      alert_score, alert_count, alert_types,
      overlay_action, overlay_weight, overlay_weight_change
    """
    if port is None or port.empty:
        return port

    out = port.copy()
    out["asof_date"] = pd.to_datetime(out["asof_date"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str)

    amap = _build_alert_score_map(alerts_df)

    out = out.merge(amap, on=["asof_date", "ticker"], how="left")
    out["alert_score"] = out["alert_score"].fillna(0.0)
    out["alert_count"] = out["alert_count"].fillna(0).astype(int)
    out["alert_types"] = out["alert_types"].fillna("")

    # Base overlay equals original
    out["overlay_weight"] = out["weight"]
    out["overlay_weight_change"] = out["weight_change"]
    out["overlay_action"] = "NO_CHANGE"

    # Optional: block BUY when MA200 break is present that day
    if block_buy_on_ma200 and alerts_df is not None and not alerts_df.empty:
        ma200 = alerts_df.copy()
        ma200["asof_date"] = pd.to_datetime(ma200["asof_date"]).dt.normalize()
        ma200["ticker"] = ma200["ticker"].astype(str)
        ma200 = ma200[ma200["alert_type"] == "MA_BREAK_200"][["asof_date", "ticker"]].drop_duplicates()
        out = out.merge(ma200.assign(_ma200=1), on=["asof_date", "ticker"], how="left")
        out["_ma200"] = out["_ma200"].fillna(0).astype(int)

        mask_block = (out["action"] == "BUY") & (out["_ma200"] == 1)
        if mask_block.any():
            # turn BUY into HOLD (no new position added)
            out.loc[mask_block, "overlay_weight"] = out.loc[mask_block, "prev_weight"]
            out.loc[mask_block, "overlay_weight_change"] = 0.0
            out.loc[mask_block, "overlay_action"] = "BUY_BLOCKED_MA200"

        out = out.drop(columns=["_ma200"])

    # Dampen BUYs in high-alert regime
    mask_buy = (out["action"] == "BUY") & (out["alert_score"] >= buy_dampen_threshold) & (out["overlay_action"] == "NO_CHANGE")
    if mask_buy.any():
        out.loc[mask_buy, "overlay_weight_change"] = out.loc[mask_buy, "weight_change"] * float(buy_dampen_mult)
        out.loc[mask_buy, "overlay_weight"] = out.loc[mask_buy, "prev_weight"] + out.loc[mask_buy, "overlay_weight_change"]
        out.loc[mask_buy, "overlay_action"] = "ADD_DAMPENED"

    # Accelerate SELLS in high-alert regime (bounded so it can't sell more than prev_weight)
    mask_sell = (out["action"] == "SELL") & (out["alert_score"] >= sell_accel_threshold) & (out["overlay_action"] == "NO_CHANGE")
    if mask_sell.any():
        prev_w = out.loc[mask_sell, "prev_weight"].astype(float)
        base_change = out.loc[mask_sell, "weight_change"].astype(float)  # negative
        new_change = base_change * float(sell_accel_mult)               # more negative
        # cap to -prev_weight (can't sell more than you had)
        new_change = np.maximum(new_change, -prev_w)
        out.loc[mask_sell, "overlay_weight_change"] = new_change
        out.loc[mask_sell, "overlay_weight"] = prev_w + new_change
        out.loc[mask_sell, "overlay_action"] = "TRIM_ACCELERATED"

    return out

def build_monthly_portfolio(
    scores: pd.DataFrame,
    top_n: int,
    benchmark_ticker: str,
    score_col: str = "overall_score_ai",
    ai_min_coverage: float = 0.80,
    alerts_df: pd.DataFrame | None = None, 
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

        # --- Optional: overlay alerts onto trade sizing ---
    if alerts_df is not None and not alerts_df.empty:
        port = apply_alert_overlay_to_portfolio(port, alerts_df)

    return port

