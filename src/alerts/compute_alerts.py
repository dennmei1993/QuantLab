from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================

@dataclass
class AlertThresholds:
    ret_1d_abs: float = 0.05     # 5%
    ret_5d_abs: float = 0.10     # 10%
    vol_z: float = 3.0
    ma_windows: Tuple[int, int] = (50, 200)

@dataclass
class AlertSuppression:
    suppress_ma_breaks_around_earnings: bool = True
    earnings_window_days: int = 3  # suppress MA breaks ±3 trading days around earnings date
    earnings_calendar_path: str = "data/earnings_calendar.csv"

# =========================
# HELPERS
# =========================

def load_earnings_calendar(path: str) -> pd.DataFrame:
    """
    Expected CSV columns: ticker, earnings_date
    """
    try:
        cal = pd.read_csv(path)
        cal["ticker"] = cal["ticker"].astype(str).str.upper().str.strip()
        cal["earnings_date"] = pd.to_datetime(cal["earnings_date"], errors="coerce")
        cal = cal.dropna(subset=["earnings_date"])
        return cal[["ticker", "earnings_date"]]
    except FileNotFoundError:
        logger.warning(f"[ALERTS][WARN] Earnings calendar not found: {path} (no suppression applied)")
        return pd.DataFrame(columns=["ticker", "earnings_date"])
    except Exception as e:
        logger.warning(f"[ALERTS][WARN] Earnings calendar load failed: {e} (no suppression applied)")
        return pd.DataFrame(columns=["ticker", "earnings_date"])

def apply_suppression_rules(
    alerts: pd.DataFrame,
    asof_date: pd.Timestamp,
    suppression: AlertSuppression = AlertSuppression(),
) -> pd.DataFrame:
    if alerts is None or alerts.empty:
        return alerts

    out = alerts.copy()

    # Earnings-window suppression for MA breaks
    if suppression.suppress_ma_breaks_around_earnings:
        cal = load_earnings_calendar(suppression.earnings_calendar_path)
        if not cal.empty:
            # compute if asof_date is within ±N days of earnings_date
            cal["start"] = cal["earnings_date"] - pd.Timedelta(days=suppression.earnings_window_days)
            cal["end"] = cal["earnings_date"] + pd.Timedelta(days=suppression.earnings_window_days)
            cal["in_window"] = (asof_date >= cal["start"]) & (asof_date <= cal["end"])

            suppress_tickers = set(cal.loc[cal["in_window"], "ticker"].tolist())

            if suppress_tickers:
                mask_ma = out["alert_type"].isin(["MA_BREAK_50", "MA_BREAK_200"])
                mask_ticker = out["ticker"].isin(suppress_tickers)
                suppressed = out[mask_ma & mask_ticker]
                if not suppressed.empty:
                    logger.info(f"[ALERTS] Suppressed {len(suppressed)} MA alerts around earnings for {len(suppress_tickers)} tickers")
                out = out[~(mask_ma & mask_ticker)]

    return out


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

from pathlib import Path

def save_alert_outputs(
    alerts: pd.DataFrame,
    alert_counts: pd.DataFrame,
    asof_date: pd.Timestamp,
    out_dir: str = "data/alerts",
    fmt: str = "parquet",   # "parquet" or "csv"
):
    """
    Save alert events + per-ticker counts side-by-side.

    Files:
      alerts_YYYYMMDD.{fmt}
      alert_counts_YYYYMMDD.{fmt}
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ds = pd.Timestamp(asof_date).strftime("%Y%m%d")

    if fmt == "csv":
        alerts.to_csv(Path(out_dir) / "aalerts_latest.csv", index=False)
        alert_counts.to_csv(Path(out_dir) / "alert_counts_latest.csv", index=False)
    else:
        alerts.to_parquet(Path(out_dir) / "alerts_latest.parquet", index=False)
        alert_counts.to_parquet(Path(out_dir) / "alert_counts_latest.parquet", index=False)

def _severity(x: float) -> str:
    ax = abs(x)
    if ax >= 0.10:
        return "high"
    if ax >= 0.06:
        return "med"
    return "low"


def _mk_alert(
    asof_date,
    ticker,
    alert_type,
    severity,
    value,
    threshold,
    message,
    context,
):
    return dict(
        event_ts=pd.Timestamp.utcnow(),
        asof_date=asof_date,
        ticker=ticker,
        alert_type=alert_type,
        severity=severity,
        value=float(value),
        threshold=float(threshold),
        message=message,
        context=_json(context),
    )


# =========================
# NORMALIZATION
# =========================

def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- date parsing ---
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # --- price selection ---
    if "close" in df.columns and df["close"].notna().any():
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    else:
        df["close"] = pd.to_numeric(df.get("adj_close"), errors="coerce")

    # treat zero / negative prices as invalid
    df.loc[df["close"] <= 0, "close"] = np.nan

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = (
        df[["ticker", "date", "close", "volume"]]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # forward-fill prices within each ticker (trading gaps, holidays)
    df["close"] = df.groupby("ticker")["close"].ffill()

    # drop remaining bad rows (instead of raising)
    df = df.dropna(subset=["close"])

    return df

# =========================
# CORE ALERT ENGINE
# =========================

def compute_daily_alerts(
    prices_daily: pd.DataFrame,
    asof_date: pd.Timestamp,
    thresholds: AlertThresholds = AlertThresholds(),
) -> pd.DataFrame:

    try:
        df = normalize_prices(prices_daily)
    except Exception as e:
        logger.warning(f"[ALERTS][WARN] Normalization failed: {e}")
        return pd.DataFrame()

    df = df[df["date"] <= asof_date]

    if df.empty:
        return pd.DataFrame()

    g = df.groupby("ticker", group_keys=False)

    # returns
    df["ret_1d"] = g["close"].pct_change(1)
    df["ret_5d"] = g["close"].pct_change(5)

    # moving averages
    w1, w2 = thresholds.ma_windows
    df["ma_50"] = g["close"].rolling(w1, min_periods=w1).mean().reset_index(level=0, drop=True)
    df["ma_200"] = g["close"].rolling(w2, min_periods=w2).mean().reset_index(level=0, drop=True)

    # volume z-score
    vol_mean = g["volume"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    vol_std = g["volume"].rolling(20, min_periods=20).std(ddof=0).reset_index(level=0, drop=True)
    df["vol_z20"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    today = df[df["date"] == asof_date]
    alerts: List[Dict] = []

    # --- 1D price move ---
    for _, r in today.loc[today["ret_1d"].abs() >= thresholds.ret_1d_abs].iterrows():
        alerts.append(_mk_alert(
            asof_date,
            r["ticker"],
            "PRICE_MOVE_1D",
            _severity(r["ret_1d"]),
            r["ret_1d"],
            thresholds.ret_1d_abs,
            f"1D move {r['ret_1d']:+.2%}",
            {"ret_1d": r["ret_1d"]},
        ))

    # --- 5D price move ---
    for _, r in today.loc[today["ret_5d"].abs() >= thresholds.ret_5d_abs].iterrows():
        alerts.append(_mk_alert(
            asof_date,
            r["ticker"],
            "PRICE_MOVE_5D",
            _severity(r["ret_5d"]),
            r["ret_5d"],
            thresholds.ret_5d_abs,
            f"5D move {r['ret_5d']:+.2%}",
            {"ret_5d": r["ret_5d"]},
        ))

    # --- volume spike ---
    for _, r in today.loc[today["vol_z20"] >= thresholds.vol_z].iterrows():
        alerts.append(_mk_alert(
            asof_date,
            r["ticker"],
            "VOLUME_SPIKE",
            "med",
            r["vol_z20"],
            thresholds.vol_z,
            f"Volume spike z={r['vol_z20']:.2f}",
            {"vol_z20": r["vol_z20"]},
        ))

    # --- MA breaks ---
    for _, r in today.iterrows():
        if pd.notna(r["ma_50"]) and r["close"] < r["ma_50"]:
            alerts.append(_mk_alert(
                asof_date,
                r["ticker"],
                "MA_BREAK_50",
                "low",
                r["close"],
                r["ma_50"],
                "Price below 50DMA",
                {"close": r["close"], "ma_50": r["ma_50"]},
            ))

        if pd.notna(r["ma_200"]) and r["close"] < r["ma_200"]:
            alerts.append(_mk_alert(
                asof_date,
                r["ticker"],
                "MA_BREAK_200",
                "med",
                r["close"],
                r["ma_200"],
                "Price below 200DMA",
                {"close": r["close"], "ma_200": r["ma_200"]},
            ))

    alerts_df = pd.DataFrame(alerts)

    alerts_df = apply_suppression_rules(
        alerts=alerts_df,
        asof_date=asof_date
    )

    return alerts_df

def summarize_alerts(alerts: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-ticker alert counts + severity breakdown + top alert types.
    """
    if alerts is None or alerts.empty:
        return pd.DataFrame(columns=[
            "ticker", "alert_count", "high", "med", "low", "top_types"
        ])

    sev = (
        alerts.groupby(["ticker", "severity"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for c in ["high", "med", "low"]:
        if c not in sev.columns:
            sev[c] = 0

    total = alerts.groupby("ticker").size().rename("alert_count").reset_index()

    type_counts = (
        alerts.groupby(["ticker", "alert_type"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["ticker", "n"], ascending=[True, False])
    )
    top_types = (
        type_counts.groupby("ticker")
        .apply(lambda g: ",".join(
            [f"{r.alert_type}:{int(r.n)}" for _, r in g.head(5).iterrows()]
        ))
        .rename("top_types")
        .reset_index()
    )

    out = (
        total
        .merge(sev, on="ticker", how="left")
        .merge(top_types, on="ticker", how="left")
        .sort_values(["alert_count", "high", "med"], ascending=False)
        .reset_index(drop=True)
    )

    return out


