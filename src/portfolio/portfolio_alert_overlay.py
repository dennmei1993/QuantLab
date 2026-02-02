import pandas as pd
import numpy as np

def apply_alert_overlay_to_rebalance(
    rebalance_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    max_trim_multiplier: float = 1.5,
    max_add_multiplier: float = 0.7,
) -> pd.DataFrame:
    """
    rebalance_df: must include ['ticker','delta_units','style','weight_pct','pe_ttm'] etc.
    alerts_df: output of compute_daily_alerts()

    Returns rebalance_df with new columns:
      - alert_score
      - overlay_action
      - overlay_delta_units
    """

    out = rebalance_df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()

    if alerts_df is None or alerts_df.empty:
        out["alert_score"] = 0
        out["overlay_action"] = "NO_CHANGE"
        out["overlay_delta_units"] = out["delta_units"]
        return out

    a = alerts_df.copy()
    a["ticker"] = a["ticker"].astype(str).str.upper().str.strip()

    # simple scoring by alert types (tune later)
    score_map = {
        "MA_BREAK_200": 2,
        "MA_BREAK_50": 1,
        "PRICE_MOVE_5D": 2,
        "PRICE_MOVE_1D": 1,
        "VOLUME_SPIKE": 1,
    }
    a["score"] = a["alert_type"].map(score_map).fillna(0)

    score_by_ticker = a.groupby("ticker")["score"].sum().rename("alert_score").reset_index()
    out = out.merge(score_by_ticker, on="ticker", how="left")
    out["alert_score"] = out["alert_score"].fillna(0)

    # overlay rule:
    # - if alert_score high and you're trying to ADD, damp adds (wait for stability)
    # - if alert_score high and you're trying to TRIM (negative delta), allow trim to proceed (or slightly accelerate)
    def overlay_delta(row):
        du = float(row["delta_units"])
        s = float(row["alert_score"])

        if s >= 3:
            if du > 0:
                return du * max_add_multiplier   # dampen adds in risk-off
            if du < 0:
                return du * max_trim_multiplier  # trim a bit faster
        return du

    out["overlay_delta_units"] = out.apply(overlay_delta, axis=1)

    # tag
    def tag(row):
        if row["overlay_delta_units"] == row["delta_units"]:
            return "NO_CHANGE"
        if row["overlay_delta_units"] > row["delta_units"]:
            return "ADD_DAMPENED"
        return "TRIM_ACCELERATED"

    out["overlay_action"] = out.apply(tag, axis=1)

    return out


def roll_alerts_to_month_end(alerts_df: pd.DataFrame, window_trading_days: int = 5) -> pd.DataFrame:
    """
    Roll daily alerts into month-end buckets using the last N TRADING DAYS per month.

    Input alerts_df columns expected:
      - asof_date (date)
      - ticker
      - alert_type
      - severity
      - value (optional)
      - threshold (optional)

    Output columns:
      - asof_date  (month-end date, normalized)
      - ticker
      - alert_score_5d
      - alert_count_5d
      - severe_5d
      - alert_types_5d   (compact top types summary)
    """

    if alerts_df is None or alerts_df.empty:
        return pd.DataFrame(columns=[
            "asof_date", "ticker", "alert_score_5d", "alert_count_5d", "severe_5d", "alert_types_5d"
        ])

    a = alerts_df.copy()
    a["asof_date"] = pd.to_datetime(a["asof_date"]).dt.normalize()
    a["ticker"] = a["ticker"].astype(str).str.upper().str.strip()

    # Map to month-end bucket (calendar month end)
    a["month_end"] = a["asof_date"] + pd.offsets.MonthEnd(0)
    a["month_end"] = pd.to_datetime(a["month_end"]).dt.normalize()

    # --- scoring weights (match your overlay philosophy; tune later)
    type_w = {
        "MA_BREAK_200": 3,
        "MA_BREAK_50": 2,
        "PRICE_MOVE_5D": 2,
        "PRICE_MOVE_1D": 1,
        "VOLUME_SPIKE": 1,
    }
    sev_w = {"high": 2.0, "med": 1.25, "low": 1.0}

    a["type_w"] = a["alert_type"].map(type_w).fillna(1.0)
    a["sev_w"] = a["severity"].map(sev_w).fillna(1.0)
    a["score"] = a["type_w"] * a["sev_w"]

    # Severe count (high + med)
    a["is_severe"] = a["severity"].isin(["high", "med"]).astype(int)

    # Determine last N *trading* days in each month by using the unique asof_date values present
    # (i.e., dates where your pipeline produced alerts; typically trading days)
    trading_days = (
        a[["month_end", "asof_date"]]
        .drop_duplicates()
        .sort_values(["month_end", "asof_date"])
    )

    # take the last N dates per month_end
    last_days = trading_days.groupby("month_end").tail(window_trading_days)

    # filter alerts to only those last days
    a = a.merge(last_days.assign(_keep=1), on=["month_end", "asof_date"], how="inner").drop(columns=["_keep"])

    # Aggregate per (month_end, ticker)
    score = a.groupby(["month_end", "ticker"], as_index=False)["score"].sum().rename(columns={"score": "alert_score_5d"})
    count = a.groupby(["month_end", "ticker"], as_index=False).size().rename(columns={"size": "alert_count_5d"})
    severe = a.groupby(["month_end", "ticker"], as_index=False)["is_severe"].sum().rename(columns={"is_severe": "severe_5d"})

    # top alert types summary
    type_counts = (
        a.groupby(["month_end", "ticker", "alert_type"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["month_end", "ticker", "n"], ascending=[True, True, False])
    )
    top_types = (
        type_counts.groupby(["month_end", "ticker"])
        .apply(lambda g: ",".join([f"{r.alert_type}:{int(r.n)}" for _, r in g.head(5).iterrows()]))
        .rename("alert_types_5d")
        .reset_index()
    )

    out = score.merge(count, on=["month_end", "ticker"], how="outer") \
               .merge(severe, on=["month_end", "ticker"], how="outer") \
               .merge(top_types, on=["month_end", "ticker"], how="outer")

    out = out.fillna({
        "alert_score_5d": 0.0,
        "alert_count_5d": 0,
        "severe_5d": 0,
        "alert_types_5d": ""
    })

    # rename month_end -> asof_date to match portfolio join key
    out = out.rename(columns={"month_end": "asof_date"})
    out["alert_count_5d"] = out["alert_count_5d"].astype(int)
    out["severe_5d"] = out["severe_5d"].astype(int)

    return out

