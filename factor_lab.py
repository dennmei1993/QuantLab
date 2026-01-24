from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
BACKTESTS_DIR = DATA_DIR / "backtests"
CURATED_DIR = DATA_DIR / "curated"

SCORES_PATH = FEATURES_DIR / "scores_monthly.parquet"
FACTORS_PATH = FEATURES_DIR / "factors_monthly.parquet"
SNAPSHOTS_PATH = FEATURES_DIR / "snapshots_monthly.parquet"
AI_WEIGHTS_PATH = FEATURES_DIR / "ai_weights_monthly.parquet"

PORTFOLIO_PATH = BACKTESTS_DIR / "portfolio_monthly.parquet"
BACKTEST_PATH = BACKTESTS_DIR / "backtest_monthly.parquet"
PRICES_PATH = CURATED_DIR / "prices_daily.parquet"


# =========================
# Bucket config (static)
# =========================
# Must match your scoring weights (static baseline)
BUCKET_WEIGHTS = {
    "growth_score": 0.30,
    "quality_score": 0.30,
    "momentum_score": 0.25,
    "risk_score": 0.15,
}
BUCKET_COLS = list(BUCKET_WEIGHTS.keys())

# AI weight columns (dynamic, from ai_weights_monthly.parquet)
AI_WCOLS = {
    "growth_score": "w_growth_score",
    "quality_score": "w_quality_score",
    "momentum_score": "w_momentum_score",
    "risk_score": "w_risk_score",
}


# =========================
# Streamlit compat dataframe wrapper
# =========================
def st_df(df: pd.DataFrame, *, stretch: bool = True, **kwargs):
    """
    Streamlit compatibility wrapper:
      - Newer Streamlit uses width="stretch"/"content"
      - Older Streamlit uses use_container_width=True/False
    """
    try:
        return st.dataframe(df, width=("stretch" if stretch else "content"), **kwargs)
    except TypeError:
        return st.dataframe(df, use_container_width=stretch, **kwargs)


def unique_cols(cols: list[str]) -> list[str]:
    out, seen = [], set()
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


# =========================
# Health helpers
# =========================
def _human_size(n_bytes: int) -> str:
    if n_bytes is None:
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n_bytes)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.1f} {u}"
        v /= 1024.0
    return f"{n_bytes} B"


def _file_meta(path: Path) -> dict:
    if not path.exists():
        return {
            "file": str(path),
            "exists": False,
            "size": "",
            "modified": "",
        }
    stt = path.stat()
    # local time display
    mtime = datetime.fromtimestamp(stt.st_mtime)
    return {
        "file": str(path),
        "exists": True,
        "size": _human_size(int(stt.st_size)),
        "modified": mtime.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _df_span(df: pd.DataFrame, date_col: str) -> str:
    if df is None or df.empty or date_col not in df.columns:
        return ""
    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{s.min().date()} → {s.max().date()} ({s.nunique()} periods)"


def render_health_panel(
    scores: pd.DataFrame,
    factors: pd.DataFrame,
    snapshots: pd.DataFrame,
    portfolio: pd.DataFrame,
    bt: pd.DataFrame,
    prices: pd.DataFrame,
    ai_w: pd.DataFrame,
) -> None:
    with st.expander("🩺 Health", expanded=True):
        # File-level health (fast, does not require successful reads)
        files = [
            ("scores_monthly", SCORES_PATH),
            ("factors_monthly", FACTORS_PATH),
            ("snapshots_monthly", SNAPSHOTS_PATH),
            ("ai_weights_monthly", AI_WEIGHTS_PATH),
            ("portfolio_monthly", PORTFOLIO_PATH),
            ("backtest_monthly", BACKTEST_PATH),
            ("prices_daily", PRICES_PATH),
        ]
        meta = pd.DataFrame([{**{"name": n}, **_file_meta(p)} for n, p in files])
        st_df(meta[["name", "exists", "size", "modified", "file"]], stretch=True, height=260)

        # Dataset-level health (only if loaded)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Universe tickers", f"{scores['ticker'].nunique():,}" if (not scores.empty and "ticker" in scores.columns) else "")
        with c2:
            st.metric("Scores rows", f"{len(scores):,}" if scores is not None else "")
        with c3:
            st.metric("Backtest months", f"{len(bt):,}" if bt is not None else "")
        with c4:
            st.metric("Portfolio rows", f"{len(portfolio):,}" if portfolio is not None else "")

        spans = []
        spans.append({"table": "scores", "span": _df_span(scores, "asof_date")})
        spans.append({"table": "factors", "span": _df_span(factors, "asof_date")})
        spans.append({"table": "snapshots", "span": _df_span(snapshots, "asof_date")})
        spans.append({"table": "ai_weights", "span": _df_span(ai_w, "asof_date")})
        spans.append({"table": "portfolio", "span": _df_span(portfolio, "asof_date")})
        spans.append({"table": "backtest", "span": _df_span(bt, "trade_date")})
        spans.append({"table": "prices", "span": _df_span(prices, "date")})
        spans_df = pd.DataFrame(spans)
        st_df(spans_df, stretch=True, height=240)

        st.caption(
            "Tip: If the page loads but charts look empty, check file existence + date spans above. "
            "Most issues are missing parquets or a mismatch in date range between scores/backtest."
        )


# =========================
# Charts
# =========================
def line_chart_equity(bt: pd.DataFrame) -> None:
    bt = bt.dropna(subset=["net_return"]).copy()
    if bt.empty:
        st.info("No backtest data to plot.")
        return

    bt = bt.sort_values("trade_date")
    strat_eq = (1.0 + bt["net_return"].astype(float)).cumprod()
    bench_ret = bt["benchmark_return"].astype(float).fillna(0.0)
    bench_eq = (1.0 + bench_ret).cumprod()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bt["next_trade_date"], strat_eq, label="Strategy")
    ax.plot(bt["next_trade_date"], bench_eq, label="Benchmark (QQQ)")
    ax.set_title("Equity Curve (Monthly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (Start=1.0)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def histogram_monthly_returns(bt: pd.DataFrame) -> None:
    if bt.empty:
        st.info("No backtest data to plot.")
        return
    rets = bt["net_return"].astype(float).dropna()
    if rets.empty:
        st.info("No returns to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(rets, bins=20)
    ax.set_title("Strategy Monthly Returns (Histogram)")
    ax.set_xlabel("Monthly return")
    ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)


# =========================
# Summary metrics
# =========================
def compute_summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {}
    rets = bt["net_return"].astype(float).dropna()
    bench = bt["benchmark_return"].astype(float).dropna()
    if rets.empty:
        return {}

    n_months = len(rets)
    n_years = n_months / 12.0
    eq = (1.0 + rets).cumprod()
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else np.nan
    ann_vol = float(rets.std(ddof=1) * np.sqrt(12)) if n_months > 1 else np.nan
    sharpe = float((rets.mean() * 12) / ann_vol) if (pd.notna(ann_vol) and ann_vol > 0) else np.nan

    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    mdd = float(dd.min())

    hit = float((bt["net_return"] > bt["benchmark_return"]).mean())

    bench_cagr = np.nan
    if len(bench) == n_months:
        bench_eq = (1.0 + bench).cumprod()
        bench_cagr = float(bench_eq.iloc[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 else np.nan

    return {
        "Months": n_months,
        "CAGR": cagr,
        "Benchmark CAGR": bench_cagr,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Hit Rate vs Benchmark": hit,
        "Avg Turnover": float(bt["turnover"].mean()) if "turnover" in bt.columns else np.nan,
        "Avg Cost / month": float(bt["cost"].mean()) if "cost" in bt.columns else np.nan,
        "Avg Holdings": float(bt["n_holdings"].mean()) if "n_holdings" in bt.columns else np.nan,
    }


# =========================
# AI score columns (UI fallback)
# =========================
def ensure_ai_columns(scores: pd.DataFrame, ai_w: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the following exist for UI display (compute if missing):
      - overall_score_ai
      - contrib_ai_<bucket>
    Uses ai_weights_monthly.parquet when available.
    If missing, falls back gracefully to static overall_score.
    """
    s = scores.copy()
    s["asof_date"] = pd.to_datetime(s["asof_date"]).dt.normalize()

    if ai_w is None or ai_w.empty or "asof_date" not in ai_w.columns:
        # No AI weights available; provide ai columns as fallback if absent
        if "overall_score_ai" not in s.columns:
            s["overall_score_ai"] = s["overall_score"] if "overall_score" in s.columns else pd.NA
        for b in BUCKET_COLS:
            c = f"contrib_ai_{b}"
            if c not in s.columns and b in s.columns:
                # fallback: use static weight
                s[c] = s[b].astype(float) * float(BUCKET_WEIGHTS[b])
        return s

    w = ai_w.copy()
    w["asof_date"] = pd.to_datetime(w["asof_date"]).dt.normalize()

    # merge weights (only keep weight cols)
    keep = ["asof_date"] + [AI_WCOLS[b] for b in BUCKET_COLS if AI_WCOLS[b] in w.columns]
    w = w[keep].drop_duplicates(subset=["asof_date"])
    df = s.merge(w, on="asof_date", how="left")

    # compute AI contributions if absent
    for b in BUCKET_COLS:
        wcol = AI_WCOLS[b]
        ccol = f"contrib_ai_{b}"
        if ccol not in df.columns and b in df.columns and wcol in df.columns:
            df[ccol] = df[b].astype(float) * df[wcol].astype(float)

    # compute overall_score_ai if absent
    if "overall_score_ai" not in df.columns:
        have_all = all((b in df.columns and AI_WCOLS[b] in df.columns) for b in BUCKET_COLS)
        if have_all:
            df["overall_score_ai"] = (
                df["growth_score"] * df["w_growth_score"]
                + df["quality_score"] * df["w_quality_score"]
                + df["momentum_score"] * df["w_momentum_score"]
                + df["risk_score"] * df["w_risk_score"]
            )
        else:
            df["overall_score_ai"] = df["overall_score"] if "overall_score" in df.columns else pd.NA

    return df


# =========================
# Contributions + deltas (static + AI)
# =========================
def add_contrib_and_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - static per-bucket contributions: contrib_<bucket> (using BUCKET_WEIGHTS)
      - deltas for overall_score, overall_score_ai (if present), bucket scores, and contributions
    """
    s = scores.copy()
    s["asof_date"] = pd.to_datetime(s["asof_date"]).dt.normalize()
    s["ticker"] = s["ticker"].astype(str)

    # static contributions
    for b, w in BUCKET_WEIGHTS.items():
        if b in s.columns and f"contrib_{b}" not in s.columns:
            s[f"contrib_{b}"] = s[b].astype(float) * float(w)

    # choose delta columns
    delta_cols = []
    for c in ["overall_score", "overall_score_ai"]:
        if c in s.columns:
            delta_cols.append(c)

    for b in BUCKET_COLS:
        if b in s.columns:
            delta_cols.append(b)
        sc = f"contrib_{b}"
        if sc in s.columns:
            delta_cols.append(sc)
        ac = f"contrib_ai_{b}"
        if ac in s.columns:
            delta_cols.append(ac)

    # compute deltas per ticker
    s = s.sort_values(["ticker", "asof_date"])
    for c in unique_cols(delta_cols):
        s[f"delta_{c}"] = s.groupby("ticker")[c].diff()

    return s


# =========================
# Main UI
# =========================
def main() -> None:
    st.set_page_config(page_title="Quant Factor MVP Dashboard", layout="wide")
    st.title("Quant Factor Rating — MVP Dashboard")

    # Load datasets
    scores = load_parquet(SCORES_PATH)
    factors = load_parquet(FACTORS_PATH)
    snapshots = load_parquet(SNAPSHOTS_PATH)
    portfolio = load_parquet(PORTFOLIO_PATH)
    bt = load_parquet(BACKTEST_PATH)
    prices = load_parquet(PRICES_PATH)
    ai_w = load_parquet(AI_WEIGHTS_PATH)

    # ✅ Quick Health panel (shows file status + spans)
    render_health_panel(scores, factors, snapshots, portfolio, bt, prices, ai_w)

    # Basic checks
    if scores.empty or portfolio.empty or bt.empty:
        st.warning(
            "Some output files are missing/empty. Run `run_mvp.py` first.\n\n"
            f"- scores: {SCORES_PATH.exists()}\n"
            f"- portfolio: {PORTFOLIO_PATH.exists()}\n"
            f"- backtest: {BACKTEST_PATH.exists()}"
        )
        return

    # Normalize dates
    for df, col in [
        (scores, "asof_date"),
        (factors, "asof_date"),
        (snapshots, "asof_date"),
        (portfolio, "asof_date"),
        (bt, "trade_date"),
        (bt, "next_trade_date"),
        (prices, "date"),
        (ai_w, "asof_date"),
    ]:
        if df is not None and not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.normalize()

    # Ensure AI columns exist for UI (even if pipeline hasn't written them yet)
    scores = ensure_ai_columns(scores, ai_w)

    # Enrich with contributions + deltas (static + AI)
    scores_enriched = add_contrib_and_deltas(scores)

    # Sidebar controls
    st.sidebar.header("Controls")
    asof_dates = sorted(scores_enriched["asof_date"].dropna().unique())
    chosen_date = st.sidebar.selectbox("As-of month-end", asof_dates, index=len(asof_dates) - 1)

    # Score mode toggle
    default_mode = "AI (dynamic weights)" if "overall_score_ai" in scores_enriched.columns else "Static"
    mode = st.sidebar.radio(
        "Rank by",
        ["AI (dynamic weights)", "Static"],
        index=0 if default_mode.startswith("AI") else 1
    )
    rank_col = "overall_score_ai" if mode.startswith("AI") and "overall_score_ai" in scores_enriched.columns else "overall_score"

    top_n = st.sidebar.slider("Top N (display)", min_value=5, max_value=50, value=15, step=1)
    min_data = st.sidebar.slider("Min data completeness", 0.0, 1.0, 0.0, 0.05)

    tickers = sorted(scores_enriched["ticker"].astype(str).unique())
    chosen_ticker = st.sidebar.selectbox("Ticker (drill-down)", ["(none)"] + tickers)

    # AI weights sidebar (current month)
    st.sidebar.subheader("AI weights (current month)")
    if ai_w.empty:
        st.sidebar.caption("No ai_weights_monthly.parquet found. Run run_mvp.py after enabling AI weights.")
    else:
        row = ai_w[ai_w["asof_date"] == chosen_date]
        if row.empty:
            st.sidebar.caption("No weights for selected month.")
        else:
            r = row.iloc[0]
            for k, label in [
                ("w_growth_score", "Growth"),
                ("w_quality_score", "Quality"),
                ("w_momentum_score", "Momentum"),
                ("w_risk_score", "Risk"),
            ]:
                if k in r.index and pd.notna(r[k]):
                    st.sidebar.write(f"{label}: {float(r[k]):.2f}")

    tabs = st.tabs([
        "Overview",
        "This month’s changes",
        "Scores",
        "Portfolio",
        "Backtest",
        "Ticker Drill-down",
        "Data Health"
    ])

    # --- Overview ---
    with tabs[0]:
        st.subheader("Summary")
        summary = compute_summary(bt)
        if summary:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("CAGR", fmt_pct(summary["CAGR"]))
            c2.metric("Benchmark CAGR", fmt_pct(summary["Benchmark CAGR"]))
            c3.metric("Sharpe", f"{summary['Sharpe']:.2f}" if pd.notna(summary["Sharpe"]) else "")
            c4.metric("Max Drawdown", fmt_pct(summary["Max Drawdown"]))
            c5.metric("Hit Rate", fmt_pct(summary["Hit Rate vs Benchmark"]))

            c6, c7, c8 = st.columns(3)
            c6.metric("Avg Turnover", f"{summary['Avg Turnover']:.4f}" if pd.notna(summary["Avg Turnover"]) else "")
            c7.metric("Avg Cost / month", f"{summary['Avg Cost / month']:.6f}" if pd.notna(summary["Avg Cost / month"]) else "")
            c8.metric("Avg Holdings", f"{summary['Avg Holdings']:.1f}" if pd.notna(summary["Avg Holdings"]) else "")

        st.divider()
        st.subheader("Equity curve")
        line_chart_equity(bt)

        st.subheader("AI weights over time")
        if not ai_w.empty:
            wcols = [c for c in ["w_growth_score", "w_quality_score", "w_momentum_score", "w_risk_score"] if c in ai_w.columns]
            if wcols:
                st.line_chart(ai_w.set_index("asof_date")[wcols])

    # --- This month’s changes ---
    with tabs[1]:
        st.subheader("This month’s changes")
        all_dates = sorted(scores_enriched["asof_date"].dropna().unique())
        if chosen_date not in all_dates:
            st.warning("Chosen date not found in scores.")
            return

        idx = all_dates.index(chosen_date)
        if idx == 0:
            st.info("No previous month available to compute deltas.")
        else:
            prev_date = all_dates[idx - 1]
            st.caption(f"Comparing {chosen_date.date()} vs {prev_date.date()} | Mode: {mode}")

            # Portfolio actions
            p_cur = portfolio[portfolio["asof_date"] == chosen_date].copy()

            if "action" in p_cur.columns:
                buys = p_cur[p_cur["action"] == "BUY"].copy()
                sells = p_cur[p_cur["action"] == "SELL"].copy()

                c1, c2 = st.columns(2)

                with c1:
                    st.write("**Buys**")
                    if buys.empty:
                        st.write("None")
                    else:
                        sc = scores_enriched[scores_enriched["asof_date"] == chosen_date][
                            ["ticker", rank_col, f"delta_{rank_col}"]
                            + [f"delta_{b}" for b in BUCKET_COLS if f"delta_{b}" in scores_enriched.columns]
                        ].rename(columns={rank_col: "score", f"delta_{rank_col}": "delta_score"})

                        buys2 = buys.merge(sc, on="ticker", how="left", suffixes=("_port", "_snap"))

                        # Prefer snapshot score/delta (aligned to AI/static toggle)
                        if "score_snap" in buys2.columns:
                            buys2["score"] = buys2["score_snap"]
                        elif "score" not in buys2.columns and "score_port" in buys2.columns:
                            buys2["score"] = buys2["score_port"]

                        if "delta_score" not in buys2.columns and "delta_score_snap" in buys2.columns:
                            buys2["delta_score"] = buys2["delta_score_snap"]

                        show_cols = ["ticker", "weight", "score", "delta_score"] + [
                            f"delta_{b}" for b in BUCKET_COLS if f"delta_{b}" in buys2.columns
                        ]
                        show_cols = [c for c in show_cols if c in buys2.columns]

                        # Safe sort if delta_score exists
                        if "delta_score" in buys2.columns:
                            buys2 = buys2.sort_values("delta_score", ascending=False)

                        st_df(buys2[show_cols], stretch=True)

                with c2:
                    st.write("**Sells**")
                    if sells.empty:
                        st.write("None")
                    else:
                        sc_prev = scores_enriched[scores_enriched["asof_date"] == prev_date][
                            ["ticker", rank_col] + BUCKET_COLS
                        ].rename(columns={rank_col: "prev_score"})

                        sells2 = sells.merge(sc_prev, on="ticker", how="left")
                        show_cols = ["ticker", "prev_weight", "prev_score"] + [c for c in BUCKET_COLS if c in sells2.columns]
                        show_cols = [c for c in show_cols if c in sells2.columns]
                        if "prev_score" in sells2.columns:
                            sells2 = sells2.sort_values("prev_score", ascending=False)
                        st_df(sells2[show_cols], stretch=True)
            else:
                st.info("No action column found in portfolio. (Re-run after adding BUY/SELL/HOLD.)")

            st.divider()

            # Top movers by selected score
            movers = scores_enriched[scores_enriched["asof_date"] == chosen_date].copy()
            dcol = f"delta_{rank_col}"
            movers = movers.dropna(subset=[dcol]) if dcol in movers.columns else movers

            n_show = st.slider("Show top movers", 10, 50, 20, 5)

            col_up, col_dn = st.columns(2)
            with col_up:
                st.write("**Biggest score increases**")
                if dcol in movers.columns:
                    up = movers.sort_values(dcol, ascending=False).head(n_show)
                    cols = ["ticker", rank_col, dcol] + [f"delta_{b}" for b in BUCKET_COLS if f"delta_{b}" in movers.columns]
                    cols = [c for c in cols if c in up.columns]
                    st_df(up[cols], stretch=True)
                else:
                    st.info(f"No delta column found for {rank_col}.")

            with col_dn:
                st.write("**Biggest score decreases**")
                if dcol in movers.columns:
                    dn = movers.sort_values(dcol, ascending=True).head(n_show)
                    cols = ["ticker", rank_col, dcol] + [f"delta_{b}" for b in BUCKET_COLS if f"delta_{b}" in movers.columns]
                    cols = [c for c in cols if c in dn.columns]
                    st_df(dn[cols], stretch=True)
                else:
                    st.info(f"No delta column found for {rank_col}.")

            st.divider()

            # Contribution deltas for holdings
            st.write("**Factor contribution deltas (holdings)**")
            held = portfolio[(portfolio["asof_date"] == chosen_date)].copy()
            if "weight" in held.columns:
                held = held[held["weight"] > 0].copy()

            sc_cur = scores_enriched[scores_enriched["asof_date"] == chosen_date].copy()

            sc_cols = ["ticker", rank_col, f"delta_{rank_col}"]
            for b in BUCKET_COLS:
                for c in [b, f"delta_{b}", f"contrib_{b}", f"delta_contrib_{b}", f"contrib_ai_{b}", f"delta_contrib_ai_{b}"]:
                    if c in sc_cur.columns:
                        sc_cols.append(c)

            sc_cols = [c for c in unique_cols(sc_cols) if c in sc_cur.columns]
            held2 = held.merge(sc_cur[sc_cols], on="ticker", how="left")

            if f"delta_{rank_col}" in held2.columns:
                held2 = held2.sort_values(f"delta_{rank_col}", ascending=False)

            cols_raw = ["action", "ticker", "weight"] + sc_cols
            cols = [c for c in unique_cols(cols_raw) if c in held2.columns]
            st_df(held2[cols], stretch=True)

            st.caption("Static contrib = bucket_score × static_weight. AI contrib = bucket_score × dynamic_weight. Deltas are MoM per ticker.")

    # --- Scores ---
    with tabs[2]:
        st.subheader("Ranked Scores (month-end snapshot)")
        view = scores_enriched[scores_enriched["asof_date"] == chosen_date].copy()
        if "data_completeness" in view.columns:
            view = view[view["data_completeness"] >= min_data]

        view = view.sort_values(rank_col, ascending=False).head(top_n)

        cols = ["ticker", rank_col]
        cols += [b for b in BUCKET_COLS if b in view.columns]
        if "data_completeness" in view.columns:
            cols.append("data_completeness")

        cols = [c for c in cols if c in view.columns]
        st_df(view[cols], stretch=True)
        st.caption("Use the sidebar 'Rank by' toggle to switch between AI and Static.")

    # --- Portfolio ---
    with tabs[3]:
        st.subheader("Portfolio Holdings (Top-N selection)")
        port_view = portfolio[portfolio["asof_date"] == chosen_date].copy()

        if "action" in port_view.columns:
            action_filter = st.multiselect("Show actions", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
            port_view = port_view[port_view["action"].isin(action_filter)]

        # Sort in BUY/HOLD/SELL order (not alphabetic)
        if "action" in port_view.columns:
            order = {"BUY": 0, "HOLD": 1, "SELL": 2}
            port_view["__action_order"] = port_view["action"].map(order).fillna(9).astype(int)
            if "weight" in port_view.columns:
                port_view = port_view.sort_values(["__action_order", "weight"], ascending=[True, False])
            else:
                port_view = port_view.sort_values(["__action_order", "ticker"])
            port_view = port_view.drop(columns=["__action_order"])

        # ✅ Portfolio score column compatibility:
        # New portfolio.py uses "score"; older versions used "overall_score"
        score_col = "score" if "score" in port_view.columns else ("overall_score" if "overall_score" in port_view.columns else None)

        base_cols = ["action", "ticker", "weight", "prev_weight", "weight_change"]
        if score_col:
            base_cols.append(score_col)
        base_cols += ["score_col_used", "turnover_estimate"]

        show_cols = [c for c in base_cols if c in port_view.columns]
        st_df(port_view[show_cols], stretch=True)

        if "turnover_estimate" in port_view.columns and not port_view.empty:
            st.caption(f"Turnover estimate (sum abs weight changes): {port_view['turnover_estimate'].iloc[0]:.4f}")

    # --- Backtest ---
    with tabs[4]:
        st.subheader("Backtest (monthly)")
        st.write("This is the monthly rebalance series produced by `run_mvp.py`.")
        bt_cols = ["trade_date", "next_trade_date", "net_return", "benchmark_return", "turnover", "cost", "n_holdings"]
        bt_cols = [c for c in bt_cols if c in bt.columns]
        st_df(bt[bt_cols].sort_values("trade_date"), stretch=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            line_chart_equity(bt)
        with c2:
            histogram_monthly_returns(bt)

    # --- Ticker drill-down ---
    with tabs[5]:
        st.subheader("Ticker drill-down")
        if chosen_ticker == "(none)":
            st.info("Select a ticker in the sidebar to see its history.")
        else:
            t = chosen_ticker
            s_hist = scores_enriched[scores_enriched["ticker"] == t].sort_values("asof_date").copy()

            st.write("Score + contribution history")
            cols = [
                "asof_date",
                "overall_score",
                "delta_overall_score",
                "overall_score_ai",
                "delta_overall_score_ai",
                "data_completeness",
            ]
            cols += BUCKET_COLS
            cols += [f"delta_{b}" for b in BUCKET_COLS]
            cols += [f"contrib_{b}" for b in BUCKET_COLS]
            cols += [f"delta_contrib_{b}" for b in BUCKET_COLS]
            cols += [f"contrib_ai_{b}" for b in BUCKET_COLS]
            cols += [f"delta_contrib_ai_{b}" for b in BUCKET_COLS]

            cols = [c for c in unique_cols(cols) if c in s_hist.columns]
            st_df(s_hist[cols], stretch=True)

            # Price chart
            if not prices.empty and "ticker" in prices.columns and "date" in prices.columns:
                p = prices[prices["ticker"].astype(str) == t].sort_values("date")
                if not p.empty and "adj_close" in p.columns:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(p["date"], p["adj_close"])
                    ax.set_title(f"{t} Adjusted Close")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Adj Close")
                    fig.tight_layout()
                    st.pyplot(fig)

    # --- Data health ---
    with tabs[6]:
        st.subheader("Data health")
        st.write("These checks help you understand missingness and coverage.")

        if not snapshots.empty:
            snap_latest = snapshots[snapshots["asof_date"] == chosen_date].copy()
            fund_cols = [c for c in ["revenue", "net_income", "free_cash_flow", "total_assets", "equity"] if c in snap_latest.columns]
            if fund_cols:
                coverage = (snap_latest[fund_cols].notna().mean()).sort_values(ascending=False)
                st.write("Fundamental field coverage (latest asof)")
                st_df(coverage.rename("coverage").to_frame(), stretch=True)

        if "data_completeness" in scores_enriched.columns:
            d = scores_enriched[scores_enriched["asof_date"] == chosen_date]["data_completeness"].dropna()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(d, bins=20)
            ax.set_title("Data Completeness Distribution (selected month)")
            ax.set_xlabel("data_completeness")
            ax.set_ylabel("count")
            fig.tight_layout()
            st.pyplot(fig)

        st.caption("If completeness is low, it’s usually fundamentals gaps. Paid fundamentals data will improve this.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        st.error("App crashed. Full traceback:")
        st.code(traceback.format_exc())
        raise
