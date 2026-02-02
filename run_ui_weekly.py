from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Paths (weekly pipeline)
# =========================
DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
CURATED_DIR = DATA_DIR / "curated"

# Weekly artifacts (created by run_weekly_update.py)
SCORES_WEEKLY_PATH = CURATED_DIR / "scores_weekly.parquet"
RANKS_LATEST_PATH = CURATED_DIR / "ranks_latest.parquet"
PORTFOLIO_TARGET_PATH = CURATED_DIR / "portfolio_target.parquet"
WATCHLIST_CHANGES_PATH = CURATED_DIR / "watchlist_changes.parquet"
ALERTS_PATH = CURATED_DIR / "alerts.parquet"
STALE_TICKERS_PATH = CURATED_DIR / "stale_tickers_weekly.parquet"

# Supporting features
AI_WEIGHTS_WEEKLY_PATH = FEATURES_DIR / "ai_weights_weekly.parquet"
WEEKLY_IC_PATH = FEATURES_DIR / "weekly_ic.parquet"

# Prices (daily)
PRICES_PATH = CURATED_DIR / "prices_daily.parquet"

# Universe metadata (ticker -> STOCK/ETF + optional theme)
UNIVERSE_CSV_PATH = DATA_DIR / "universe" / "us_tickers.csv"


# =========================
# Bucket config (must match your pipeline buckets)
# =========================
BUCKET_COLS = ["growth_score", "quality_score", "momentum_score", "risk_score"]
AI_WCOLS = {
    "growth_score": "w_growth_score",
    "quality_score": "w_quality_score",
    "momentum_score": "w_momentum_score",
    "risk_score": "w_risk_score",
}


# =========================
# Streamlit compat dataframe wrapper
# =========================

# =========================
# Helpers
# =========================
def _unique_list(seq):
    """Preserve order but drop duplicates."""
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _make_unique_columns(cols):
    """Make column names unique by appending __2, __3... to duplicates."""
    counts = {}
    new_cols = []
    for c in cols:
        if c not in counts:
            counts[c] = 1
            new_cols.append(c)
        else:
            counts[c] += 1
            new_cols.append(f"{c}__{counts[c]}")
    return new_cols


def ticker_url(ticker: str, provider: str = "tradingview") -> str:
    """Return a browser URL for a ticker (US default)."""
    t = str(ticker).strip().upper()
    if provider == "yahoo":
        return f"https://finance.yahoo.com/quote/{t}"
    if provider == "finviz":
        return f"https://finviz.com/quote.ashx?t={t}"
    # default: tradingview
    # Note: For most US equities/ETFs, NASDAQ/NYSE/AMEX is unknown here; TradingView resolves many with just symbol,
    # but the most reliable is to prefix with NASDAQ/NYSE. We'll use "NASDAQ" as a best-effort default.
    return f"https://www.tradingview.com/symbols/{t}/"


def with_ticker_url(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    if df is None or df.empty or ticker_col not in df.columns:
        return df
    df = df.copy()
    df["url"] = df[ticker_col].astype(str).map(lambda x: ticker_url(x, LINK_PROVIDER))
    return df


import streamlit.components.v1 as components

def newtab_link(label: str, url: str):
    """Render a safe HTML link that opens in a new browser tab."""
    html = f'''
    <a href="{url}" target="_blank" rel="noopener noreferrer" style="text-decoration:none">
      <button style="padding:0.4rem 0.7rem; border-radius:0.4rem; border:1px solid #ccc; cursor:pointer;">
        {label}
      </button>
    </a>
    '''
    st.markdown(html, unsafe_allow_html=True)

def newtab_open(url: str):
    """Attempt to open a new tab via JS (works only on user interaction).""
    js = f"<script>window.open('{url}', '_blank').focus();</script>"
    components.html(js, height=0, width=0)
def st_df(df: pd.DataFrame, *, stretch: bool = True, **kwargs):
    """
    Streamlit compatibility wrapper:
      - Newer Streamlit uses width="stretch"/"content"
      - Older Streamlit uses use_container_width=True/False
    """
    # PyArrow (used by st.dataframe) does not allow duplicate column names.
    # This can happen after merges or if show_cols contains duplicates.
    if hasattr(df, 'columns'):
        cols = list(df.columns)
        if len(cols) != len(set(cols)):
            # Rename duplicates deterministically so display never crashes.
            df = df.copy()
            df.columns = _make_unique_columns(cols)
    try:
        return st.dataframe(df, width=("stretch" if stretch else "content"), **kwargs)
    except TypeError:
        return st.dataframe(df, use_container_width=stretch, **kwargs)


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_universe_csv(path: Path) -> pd.DataFrame:
    """Load data/universe/us_tickers.csv with columns: ticker, asset_type (STOCK|ETF), optional theme."""
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "asset_type", "theme"])
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return pd.DataFrame(columns=["ticker", "asset_type", "theme"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    if "asset_type" not in df.columns:
        df["asset_type"] = "STOCK"
    df["asset_type"] = df["asset_type"].astype(str).str.strip().str.upper()
    df = df[df["asset_type"].isin(["STOCK", "ETF"])]

    if "theme" not in df.columns:
        df["theme"] = ""
    df["theme"] = df["theme"].fillna("").astype(str).str.strip()

    df = df[df["ticker"].notna() & (df["ticker"] != "")]
    df = df.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return df[["ticker", "asset_type", "theme"]]


def apply_universe_meta(df: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """Left-join asset_type/theme onto any dataframe with a ticker column."""
    if df is None or df.empty or "ticker" not in df.columns or uni is None or uni.empty:
        return df
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    return out.merge(uni, on="ticker", how="left")


def fmt_pct(x, digits: int = 1) -> str:
    """Format a decimal (e.g., 0.274) as percent string (e.g., 27.4%)."""
    try:
        if x is None:
            return "—"
        import math
        if isinstance(x, float) and math.isnan(x):
            return "—"
    except Exception:
        pass
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "—"


def _normalize_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.normalize()
    return out


def _severity_order(s: str) -> int:
    s = str(s).lower().strip()
    return {"high": 0, "med": 1, "low": 2}.get(s, 9)


def _short_ctx(x: str, max_len: int = 140) -> str:
    x = "" if x is None else str(x)
    x = x.replace("\n", " ").strip()
    return x if len(x) <= max_len else x[: max_len - 1] + "…"


# =========================
# Charts
# =========================
def line_chart_price(prices: pd.DataFrame, ticker: str) -> None:
    if prices is None or prices.empty:
        st.info("No prices_daily.parquet available.")
        return
    p = prices[prices["ticker"].astype(str).str.upper() == str(ticker).upper()].copy()
    if p.empty:
        st.info("No price history for this ticker.")
        return
    p = p.sort_values("date")
    ycol = "adj_close" if "adj_close" in p.columns else ("close" if "close" in p.columns else None)
    if ycol is None:
        st.info("Price file missing adj_close/close columns.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(p["date"], pd.to_numeric(p[ycol], errors="coerce"))
    ax.set_title(f"{ticker} price ({ycol})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Quant Factor Rating — Weekly Dashboard", layout="wide")

# Sidebar: link provider for opening in browser
LINK_PROVIDER = st.sidebar.selectbox('Open links with', ['tradingview','yahoo','finviz'], index=0)
st.sidebar.caption('Links open in a new tab when you click them. Some browsers block pop-ups unless you click the button.')
    st.title("Quant Factor Rating — Weekly Decision Dashboard")

    # --- Load datasets ---
    universe = load_universe_csv(UNIVERSE_CSV_PATH)

    scores = load_parquet(SCORES_WEEKLY_PATH)
    ranks_latest = load_parquet(RANKS_LATEST_PATH)
    portfolio_target = load_parquet(PORTFOLIO_TARGET_PATH)
    watchlist = load_parquet(WATCHLIST_CHANGES_PATH)
    alerts = load_parquet(ALERTS_PATH)
    stale = load_parquet(STALE_TICKERS_PATH)

    ai_w = load_parquet(AI_WEIGHTS_WEEKLY_PATH)
    weekly_ic = load_parquet(WEEKLY_IC_PATH)
    prices = load_parquet(PRICES_PATH)

    # Attach universe meta
    scores = apply_universe_meta(scores, universe)
    ranks_latest = apply_universe_meta(ranks_latest, universe)
    portfolio_target = apply_universe_meta(portfolio_target, universe)
    watchlist = apply_universe_meta(watchlist, universe)
    alerts = apply_universe_meta(alerts, universe)

    # Normalize dates
    scores = _normalize_dates(scores, ["asof_date"])
    ranks_latest = _normalize_dates(ranks_latest, ["week_end"])
    portfolio_target = _normalize_dates(portfolio_target, ["week_end"])
    watchlist = _normalize_dates(watchlist, ["week_end"])
    alerts = _normalize_dates(alerts, ["asof_date", "event_ts"])
    stale = _normalize_dates(stale, ["last_available_week_end"])
    ai_w = _normalize_dates(ai_w, ["asof_date"])
    weekly_ic = _normalize_dates(weekly_ic, ["asof_date"])
    prices = _normalize_dates(prices, ["date"])

    # --- Basic checks ---
    missing = []
    for path, name in [
        (SCORES_WEEKLY_PATH, "curated/scores_weekly.parquet"),
        (RANKS_LATEST_PATH, "curated/ranks_latest.parquet"),
        (PORTFOLIO_TARGET_PATH, "curated/portfolio_target.parquet"),
        (ALERTS_PATH, "curated/alerts.parquet"),
    ]:
        if not path.exists():
            missing.append(name)

    if missing:
        st.warning(
            "Some weekly artifacts are missing. Run the pipeline first:\n\n"
            "- Daily: `python run_daily_update.py`\n"
            "- Weekly: `python run_weekly_update.py`\n\n"
            "Missing:\n- " + "\n- ".join(missing)
        )

    # =========================
    # Sidebar controls
    # =========================
    st.sidebar.header("Controls")

    # Asset type / theme filters
    asset_types = ["STOCK", "ETF"]
    if "asset_type" in scores.columns and not scores.empty:
        present = sorted([str(x).strip().upper() for x in scores["asset_type"].dropna().unique().tolist()])
        present = [x for x in present if x in ["STOCK", "ETF"]]
        if present:
            asset_types = present

    chosen_asset_types = st.sidebar.multiselect(
        "Asset type",
        options=["STOCK", "ETF"],
        default=asset_types,
    )

    themes = []
    if "theme" in scores.columns and not scores.empty:
        themes = sorted([t for t in scores["theme"].dropna().unique().tolist() if str(t).strip() != ""])
    chosen_themes = []
    if themes:
        chosen_themes = st.sidebar.multiselect("Theme (optional)", options=themes, default=[])

    # Decide the weekly "as-of" list based on scores_weekly (authoritative)
    week_ends = sorted(scores["asof_date"].dropna().unique()) if (scores is not None and not scores.empty and "asof_date" in scores.columns) else []
    default_week = week_ends[-1] if week_ends else None
    chosen_week = st.sidebar.selectbox("Week-end (Friday)", week_ends, index=(len(week_ends) - 1) if week_ends else 0)

    # Score mode toggle (AI vs Static)
    default_mode = "AI (dynamic weights)" if "overall_score_ai" in scores.columns else "Static"
    mode = st.sidebar.radio(
        "Rank by",
        ["AI (dynamic weights)", "Static"],
        index=0 if default_mode.startswith("AI") else 1,
    )
    rank_col = "overall_score_ai" if mode.startswith("AI") and "overall_score_ai" in scores.columns else "overall_score"

    top_n = st.sidebar.slider("Top N (display)", min_value=5, max_value=100, value=25, step=5)
    min_data = st.sidebar.slider("Min data completeness", 0.0, 1.0, 0.0, 0.05)

    # Ticker drill-down
    tickers = sorted(scores["ticker"].astype(str).unique()) if (scores is not None and not scores.empty and "ticker" in scores.columns) else []
    chosen_ticker = st.sidebar.selectbox("Ticker (drill-down)", ["(none)"] + tickers)

    # AI weights sidebar
    st.sidebar.subheader("AI weights (selected week)")
    if ai_w.empty:
        st.sidebar.caption("No features/ai_weights_weekly.parquet found yet.")
    else:
        w_row = ai_w[ai_w["asof_date"] == chosen_week] if (chosen_week is not None and "asof_date" in ai_w.columns) else pd.DataFrame()
        if w_row.empty:
            st.sidebar.caption("No weights for selected week.")
        else:
            r = w_row.iloc[0]
            for b in BUCKET_COLS:
                wcol = AI_WCOLS.get(b)
                if wcol and wcol in r.index and pd.notna(r[wcol]):
                    st.sidebar.write(f"{b.replace('_score','').title()}: {float(r[wcol]):.2f}")

    # =========================
    # Tabs
    # =========================
    tabs = st.tabs([
        "Overview",
        "Ranks (latest)",
        "Scores (weekly)",
        "Portfolio target",
        "Alerts",
        "Watchlist changes",
        "Ticker drill-down",
        "Data health",
    ])

    # --- Overview ---
    with tabs[0]:
        st.subheader("What’s in the latest weekly run?")

        # latest ranks file is top_n already; show basic stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe tickers (scores)", int(scores["ticker"].nunique()) if not scores.empty else 0)
        c2.metric("Weeks in history", int(scores["asof_date"].nunique()) if ("asof_date" in scores.columns and not scores.empty) else 0)
        c3.metric("Ranks_latest rows", int(len(ranks_latest)) if not ranks_latest.empty else 0)
        c4.metric("Alerts (total)", int(len(alerts)) if not alerts.empty else 0)

        st.divider()

        st.subheader("Weekly IC (sanity check)")
        if weekly_ic.empty or "asof_date" not in weekly_ic.columns:
            st.info("weekly_ic.parquet not found or empty yet.")
        else:
            # show latest few rows; chart if present
            ic_view = weekly_ic.sort_values("asof_date").copy()
            st_df(ic_view.tail(12), stretch=True)

            # Quick chart if there are bucket IC columns
            ic_cols = [c for c in ic_view.columns if c.endswith("_ic") or c.endswith("_ic_spearman") or c.startswith("ic_")]
            ic_cols = [c for c in ic_cols if c not in ["asof_date"]]
            if ic_cols:
                st.line_chart(ic_view.set_index("asof_date")[ic_cols])

    # --- Ranks (latest) ---
    with tabs[1]:
        st.subheader("Latest Top-N ranks (curated/ranks_latest.parquet)")

        view = ranks_latest.copy()
        if not view.empty:
            # Apply asset_type/theme filters
            if "asset_type" in view.columns and chosen_asset_types:
                view = view[view["asset_type"].isin(chosen_asset_types)].copy()
            if "theme" in view.columns and chosen_themes:
                view = view[view["theme"].isin(chosen_themes)].copy()

            # Sort by rank if present
            if "rank" in view.columns:
                view = view.sort_values("rank")
            else:
                view = view.sort_values(rank_col, ascending=False)

            show_cols = ["week_end", "rank", "ticker", rank_col, "overall_score", "overall_score_ai"] + BUCKET_COLS + ["asset_type", "theme"]
            show_cols = [c for c in show_cols if c in view.columns]
            show_cols = _unique_list(show_cols)
            view = view.loc[:, ~pd.Index(view.columns).duplicated()]
            view = with_ticker_url(view)
            if 'url' in view.columns and 'url' not in show_cols: show_cols.append('url')
            st_df(view[show_cols], stretch=True, column_config={'url': st.column_config.LinkColumn('url')})

        else:
            st.info("No ranks_latest.parquet found yet (or it is empty). Run weekly update.")

    # --- Scores (weekly) ---
    with tabs[2]:
        st.subheader("Ranked scores for selected week")
        if scores.empty:
            st.info("No scores_weekly.parquet yet.")
        else:
            v = scores[scores["asof_date"] == chosen_week].copy()

            # Apply filters
            if "asset_type" in v.columns and chosen_asset_types:
                v = v[v["asset_type"].isin(chosen_asset_types)].copy()
            if "theme" in v.columns and chosen_themes:
                v = v[v["theme"].isin(chosen_themes)].copy()
            if "data_completeness" in v.columns:
                v = v[pd.to_numeric(v["data_completeness"], errors="coerce").fillna(0.0) >= float(min_data)].copy()

            # Sort & show top N
            v[rank_col] = pd.to_numeric(v.get(rank_col), errors="coerce")
            v = v.dropna(subset=[rank_col]).sort_values(rank_col, ascending=False).head(int(top_n))

            cols = ["asof_date", "ticker"]
            if "asset_type" in v.columns:
                cols.append("asset_type")
            if "theme" in v.columns:
                cols.append("theme")
            cols.append(rank_col)
            cols += [b for b in BUCKET_COLS if b in v.columns]
            if "data_completeness" in v.columns:
                cols.append("data_completeness")

            cols = [c for c in cols if c in v.columns]
            st_df(v[cols], stretch=True)
            st.caption("This view reads curated/scores_weekly.parquet (the full weekly universe).")

    # --- Portfolio target ---
    with tabs[3]:
        st.subheader("Portfolio target (weekly actions)")
        if portfolio_target.empty:
            st.info("No portfolio_target.parquet yet.")
        else:
            pv = portfolio_target[portfolio_target["week_end"] == chosen_week].copy() if ("week_end" in portfolio_target.columns and chosen_week is not None) else portfolio_target.copy()

            # Apply filters
            if "asset_type" in pv.columns and chosen_asset_types:
                pv = pv[pv["asset_type"].isin(chosen_asset_types)].copy()
            if "theme" in pv.columns and chosen_themes:
                pv = pv[pv["theme"].isin(chosen_themes)].copy()

            # Optional action filter
            if "action" in pv.columns:
                action_filter = st.multiselect("Show actions", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
                pv = pv[pv["action"].isin(action_filter)]

            # Sort in BUY/HOLD/SELL order
            if "action" in pv.columns:
                order = {"BUY": 0, "HOLD": 1, "SELL": 2}
                pv["__action_order"] = pv["action"].map(order).fillna(9).astype(int)
                # prefer target_weight or weight
                wcol = "target_weight" if "target_weight" in pv.columns else ("weight" if "weight" in pv.columns else None)
                if wcol:
                    pv[wcol] = pd.to_numeric(pv[wcol], errors="coerce")
                    pv = pv.sort_values(["__action_order", wcol], ascending=[True, False])
                else:
                    pv = pv.sort_values(["__action_order", "ticker"])
                pv = pv.drop(columns=["__action_order"])

            # Column set
            cols = ["week_end", "action", "ticker"]
            for c in ["target_weight", "weight", "prev_weight", "weight_change", "score_used", "score_col_used", "ai_coverage", "turnover_estimate"]:
                if c in pv.columns:
                    cols.append(c)
            if "alert_score" in pv.columns:
                cols += ["alert_score", "alert_count", "alert_types"]
            cols += [c for c in ["asset_type", "theme"] if c in pv.columns]
            cols = [c for c in cols if c in pv.columns]

            st_df(pv[cols], stretch=True)

            if "turnover_estimate" in pv.columns and not pv.empty:
                st.caption(f"Turnover estimate (sum abs weight changes): {pv['turnover_estimate'].iloc[0]:.4f}")

    # --- Alerts ---
    with tabs[4]:
        st.subheader("Alerts feed (daily + weekly rank-change alerts)")
        if alerts.empty:
            st.info("No alerts.parquet yet.")
        else:
            # choose alert date
            alert_dates = sorted(alerts["asof_date"].dropna().unique()) if "asof_date" in alerts.columns else []
            chosen_alert_date = st.selectbox("Alert date", alert_dates, index=len(alert_dates) - 1 if alert_dates else 0)

            av = alerts[alerts["asof_date"] == chosen_alert_date].copy() if chosen_alert_date is not None else alerts.copy()

            # filters
            if "asset_type" in av.columns and chosen_asset_types:
                av = av[av["asset_type"].isin(chosen_asset_types)].copy()
            if "theme" in av.columns and chosen_themes:
                av = av[av["theme"].isin(chosen_themes)].copy()

            alert_types = sorted(av["alert_type"].dropna().unique()) if "alert_type" in av.columns else []
            chosen_types = st.multiselect("Alert types", options=alert_types, default=alert_types)
            if chosen_types and "alert_type" in av.columns:
                av = av[av["alert_type"].isin(chosen_types)].copy()

            severities = ["high", "med", "low"]
            chosen_sev = st.multiselect("Severities", options=severities, default=severities)
            if chosen_sev and "severity" in av.columns:
                av = av[av["severity"].astype(str).str.lower().isin(chosen_sev)].copy()

            # sort: high -> low, then by value magnitude
            if "severity" in av.columns:
                av["__sev"] = av["severity"].map(_severity_order).fillna(9).astype(int)
            else:
                av["__sev"] = 9

            if "value" in av.columns:
                av["__absval"] = pd.to_numeric(av["value"], errors="coerce").abs()
            else:
                av["__absval"] = 0.0

            av = av.sort_values(["__sev", "__absval"], ascending=[True, False]).drop(columns=["__sev", "__absval"])

            # shorten context for UI readability
            if "context" in av.columns:
                av["context"] = av["context"].apply(_short_ctx)

            cols = ["event_ts", "asof_date", "ticker", "alert_type", "severity", "message", "value", "threshold", "asset_type", "theme", "context"]
            cols = [c for c in cols if c in av.columns]
            st_df(av[cols], stretch=True)

            st.divider()
            st.subheader("Alert concentration (top tickers)")
            if "ticker" in av.columns:
                top = (
                    av.groupby(["ticker"], as_index=False)
                      .size()
                      .rename(columns={"size": "n_alerts"})
                      .sort_values("n_alerts", ascending=False)
                      .head(20)
                )
                top = apply_universe_meta(top, universe)
                st_df(top, stretch=True)

    # --- Watchlist changes ---
    with tabs[5]:
        st.subheader("Watchlist changes (Top-N ENTER/EXIT)")
        if watchlist.empty:
            st.info("No watchlist_changes.parquet yet.")
        else:
            wv = watchlist.copy()
            if "week_end" in wv.columns and chosen_week is not None:
                wv = wv[wv["week_end"] == chosen_week].copy()

            # apply filters
            if "asset_type" in wv.columns and chosen_asset_types:
                wv = wv[wv["asset_type"].isin(chosen_asset_types)].copy()
            if "theme" in wv.columns and chosen_themes:
                wv = wv[wv["theme"].isin(chosen_themes)].copy()

            cols = ["week_end", "ticker", "change", "rank_prev", "rank_now", "asset_type", "theme"]
            cols = [c for c in cols if c in wv.columns]
            st_df(wv[cols].sort_values(["change", "ticker"]), stretch=True)

    # --- Ticker drill-down ---
    with tabs[6]:
        st.subheader("Ticker drill-down (scores + alerts + price)")
        if chosen_ticker == "(none)":
            st.info("Select a ticker in the sidebar.")
        else:
            t = str(chosen_ticker).upper()

            # Score history
            st.write("Weekly score history")
            sh = scores[scores["ticker"].astype(str).str.upper() == t].sort_values("asof_date").copy() if not scores.empty else pd.DataFrame()
            if not sh.empty:
                cols = ["asof_date", "overall_score", "overall_score_ai"] + BUCKET_COLS + ["data_completeness", "asset_type", "theme"]
                cols = [c for c in cols if c in sh.columns]
                st_df(sh[cols], stretch=True)

                # simple line chart for overall_score_ai if present
                if "overall_score_ai" in sh.columns and sh["overall_score_ai"].notna().any():
                    st.line_chart(sh.set_index("asof_date")[["overall_score_ai"]])
                elif "overall_score" in sh.columns and sh["overall_score"].notna().any():
                    st.line_chart(sh.set_index("asof_date")[["overall_score"]])
            else:
                st.info("No score history for this ticker in scores_weekly.parquet.")

            st.divider()

            # Alerts history (last 60 days)
            st.write("Recent alerts (last 60 days)")
            ah = alerts[alerts["ticker"].astype(str).str.upper() == t].copy() if not alerts.empty else pd.DataFrame()
            if not ah.empty:
                ah = ah.sort_values("asof_date", ascending=False)
                cols = ["event_ts", "asof_date", "alert_type", "severity", "message", "value", "threshold", "context"]
                cols = [c for c in cols if c in ah.columns]
                if "context" in ah.columns:
                    ah["context"] = ah["context"].apply(_short_ctx)
                st_df(ah.head(100)[cols], stretch=True)
            else:
                st.info("No alerts for this ticker yet.")

            st.divider()
            st.write("Price chart")
            line_chart_price(prices, t)

    # --- Data health ---
    with tabs[7]:
        st.subheader("Data health / coverage")
        c1, c2 = st.columns(2)

        with c1:
            st.write("Scores coverage by week")
            if scores.empty:
                st.info("No scores.")
            else:
                cov = (
                    scores.groupby("asof_date")["ticker"]
                    .nunique()
                    .rename("n_tickers")
                    .reset_index()
                    .sort_values("asof_date")
                )
                st_df(cov.tail(20), stretch=True)
                st.line_chart(cov.set_index("asof_date")[["n_tickers"]])

        with c2:
            st.write("Stale tickers (vs latest week)")
            if stale.empty:
                st.caption("No stale_tickers_weekly.parquet (or no stale tickers).")
            else:
                sv = stale.sort_values(["weeks_stale", "ticker"], ascending=[False, True]).head(50)
                st_df(sv, stretch=True)


if __name__ == "__main__":
    main()