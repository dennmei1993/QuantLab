from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# =========================
# Paths (weekly pipeline)
# =========================
DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
CURATED_DIR = DATA_DIR / "curated"

SCORES_WEEKLY_PATH = CURATED_DIR / "scores_weekly.parquet"
RANKS_LATEST_PATH = CURATED_DIR / "ranks_latest.parquet"
PORTFOLIO_TARGET_PATH = CURATED_DIR / "portfolio_target.parquet"
WATCHLIST_CHANGES_PATH = CURATED_DIR / "watchlist_changes.parquet"
ALERTS_PATH = CURATED_DIR / "alerts.parquet"
STALE_TICKERS_PATH = CURATED_DIR / "stale_tickers_weekly.parquet"

AI_WEIGHTS_WEEKLY_PATH = FEATURES_DIR / "ai_weights_weekly.parquet"
WEEKLY_IC_PATH = FEATURES_DIR / "weekly_ic.parquet"
PRICES_PATH = CURATED_DIR / "prices_daily.parquet"

UNIVERSE_CSV_PATH = DATA_DIR / "universe" / "us_tickers.csv"


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


def _dedupe_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated columns (keeps first) and ensure remaining names are unique."""
    if df is None or df.empty:
        return df
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        df.columns = _make_unique_columns(cols)
    return df


def st_df(df: pd.DataFrame, *, stretch: bool = True, **kwargs):
    """
    Streamlit compatibility wrapper:
      - Newer Streamlit uses width="stretch"/"content"
      - Older Streamlit uses use_container_width=True/False
    Also defuses duplicate column name crashes (pyarrow).
    """
    df = _dedupe_df_columns(df)
    try:
        return st.dataframe(df, width=("stretch" if stretch else "content"), **kwargs)
    except TypeError:
        return st.dataframe(df, use_container_width=stretch, **kwargs)


def ticker_url(ticker: str, provider: str = "tradingview") -> str:
    t = str(ticker).strip().upper()
    if provider == "yahoo":
        return f"https://finance.yahoo.com/quote/{t}"
    if provider == "finviz":
        return f"https://finviz.com/quote.ashx?t={t}"
    # TradingView resolves many symbols without exchange prefix via /symbols/{T}/
    return f"https://www.tradingview.com/symbols/{t}/"


def newtab_link(label: str, url: str):
    """HTML link that opens in a new browser tab."""
    # No triple-quoted strings to avoid parsing issues.
    html = (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">'
        f'<button style="padding:0.40rem 0.70rem; border-radius:0.40rem; border:1px solid #ccc; cursor:pointer;">'
        f'{label}'
        f"</button></a>"
    )
    st.markdown(html, unsafe_allow_html=True)


def newtab_open(url: str):
    """Attempt to open a new tab via JS. Must be triggered by a user click."""
    js = f"<script>window.open('{url}', '_blank');</script>"
    components.html(js, height=0, width=0)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_all():
    data = {
        "scores_weekly": _read_parquet(SCORES_WEEKLY_PATH),
        "ranks_latest": _read_parquet(RANKS_LATEST_PATH),
        "portfolio_target": _read_parquet(PORTFOLIO_TARGET_PATH),
        "watchlist_changes": _read_parquet(WATCHLIST_CHANGES_PATH),
        "alerts": _read_parquet(ALERTS_PATH),
        "stale_tickers": _read_parquet(STALE_TICKERS_PATH),
        "ai_weights_weekly": _read_parquet(AI_WEIGHTS_WEEKLY_PATH),
        "weekly_ic": _read_parquet(WEEKLY_IC_PATH),
        "prices_daily": _read_parquet(PRICES_PATH),
    }
    # Universe CSV optional
    if UNIVERSE_CSV_PATH.exists():
        uni = pd.read_csv(UNIVERSE_CSV_PATH)
        data["universe"] = uni
    else:
        data["universe"] = pd.DataFrame()
    # Dedupe columns everywhere (safe display + safe merges)
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            data[k] = _dedupe_df_columns(v)
    return data


def _pick_asof_col(df: pd.DataFrame) -> str | None:
    for c in ["week_end", "asof", "asof_date", "date"]:
        if c in df.columns:
            return c
    return None


def fmt_pct(x) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return str(x)


def plot_price(prices: pd.DataFrame, ticker: str):
    if prices.empty:
        st.info("prices_daily.parquet not found or empty.")
        return
    df = prices.copy()
    # Try common column names
    tcol = "ticker" if "ticker" in df.columns else None
    dcol = "date" if "date" in df.columns else ("day" if "day" in df.columns else None)
    ccol = "close" if "close" in df.columns else ("adj_close" if "adj_close" in df.columns else None)
    if not (tcol and dcol and ccol):
        st.warning("prices_daily schema not recognized (need ticker/date/close).")
        return
    sub = df[df[tcol].astype(str).str.upper() == str(ticker).upper()].copy()
    if sub.empty:
        st.info("No price rows for this ticker.")
        return
    sub[dcol] = pd.to_datetime(sub[dcol])
    sub = sub.sort_values(dcol).tail(400)
    fig = plt.figure()
    plt.plot(sub[dcol], sub[ccol])
    plt.title(f"{ticker} price (last ~400 trading days)")
    plt.xticks(rotation=30)
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Quant Pipeline – Weekly UI", layout="wide")

    st.title("Quant Pipeline – Weekly UI (clean)")

    # Sidebar controls
    LINK_PROVIDER = st.sidebar.selectbox("Open links with", ["tradingview", "yahoo", "finviz"], index=0)
    st.sidebar.caption("Links open in a new tab when you click them. Chrome may block popups unless you use the 'Force new tab' button.")
    topn = st.sidebar.slider("Top N", min_value=5, max_value=200, value=50, step=5)

    data = load_all()

    ranks_latest = data["ranks_latest"]
    scores_weekly = data["scores_weekly"]
    portfolio_target = data["portfolio_target"]
    alerts = data["alerts"]
    watch_changes = data["watchlist_changes"]
    stale = data["stale_tickers"]
    weekly_ic = data["weekly_ic"]
    prices = data["prices_daily"]
    universe = data["universe"]

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

    # ---- Overview
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ranks_latest rows", 0 if ranks_latest.empty else len(ranks_latest))
        c2.metric("scores_weekly rows", 0 if scores_weekly.empty else len(scores_weekly))
        c3.metric("portfolio_target rows", 0 if portfolio_target.empty else len(portfolio_target))
        c4.metric("alerts rows", 0 if alerts.empty else len(alerts))

        if not weekly_ic.empty:
            st.subheader("Weekly IC (sanity)")
            st_df(weekly_ic.tail(100))
        else:
            st.info("weekly_ic.parquet not found (optional).")

    # ---- Ranks latest
    with tabs[1]:
        st.subheader("Ranks (latest week)")
        if ranks_latest.empty:
            st.warning("Missing data/curated/ranks_latest.parquet")
        else:
            df = ranks_latest.copy()

            # Optional universe join
            if not universe.empty and "ticker" in df.columns and "ticker" in universe.columns:
                df = df.merge(universe, on="ticker", how="left", suffixes=("", "__u"))

            # Pick score column
            score_col = None
            for c in ["overall_score_ai", "overall_score", "score", "overall_score_ai__2"]:
                if c in df.columns:
                    score_col = c
                    break

            if score_col:
                df = df.sort_values(score_col, ascending=False)

            show_cols = []
            for c in ["week_end", "asof", "asof_date"]:
                if c in df.columns:
                    show_cols.append(c)
                    break
            for c in ["rank", "ticker", "theme", "sector", "industry", score_col, "growth_score", "quality_score", "momentum_score", "risk_score"]:
                if c and c in df.columns:
                    show_cols.append(c)

            show_cols = _unique_list([c for c in show_cols if c])
            df = df.head(topn)

            # Add URL column for clicking
            if "ticker" in df.columns:
                df = df.copy()
                df["url"] = df["ticker"].astype(str).map(lambda x: ticker_url(x, LINK_PROVIDER))
                if "url" not in show_cols:
                    show_cols.append("url")

            st_df(df[show_cols], column_config={"url": st.column_config.LinkColumn("url")})

    # ---- Scores weekly
    with tabs[2]:
        st.subheader("Scores (weekly history – pick week)")
        if scores_weekly.empty:
            st.warning("Missing data/curated/scores_weekly.parquet")
        else:
            df = scores_weekly.copy()
            asof_col = _pick_asof_col(df)
            if asof_col:
                df[asof_col] = pd.to_datetime(df[asof_col]).dt.date
                weeks = sorted(df[asof_col].dropna().unique())
                sel = st.selectbox("Week end", weeks, index=len(weeks) - 1)
                df = df[df[asof_col] == sel].copy()

            # Pick score col
            score_col = None
            for c in ["overall_score_ai", "overall_score", "score"]:
                if c in df.columns:
                    score_col = c
                    break
            if score_col:
                df = df.sort_values(score_col, ascending=False)

            show_cols = [c for c in [asof_col, "ticker", score_col, "growth_score", "quality_score", "momentum_score", "risk_score", "theme", "rank"] if c and c in df.columns]
            show_cols = _unique_list(show_cols)

            df = df.head(topn)
            if "ticker" in df.columns:
                df["url"] = df["ticker"].astype(str).map(lambda x: ticker_url(x, LINK_PROVIDER))
                if "url" not in show_cols:
                    show_cols.append("url")

            st_df(df[show_cols], column_config={"url": st.column_config.LinkColumn("url")})

    # ---- Portfolio target
    with tabs[3]:
        st.subheader("Portfolio target (weekly)")
        if portfolio_target.empty:
            st.warning("Missing data/curated/portfolio_target.parquet")
        else:
            df = portfolio_target.copy()
            asof_col = _pick_asof_col(df)
            if asof_col:
                df[asof_col] = pd.to_datetime(df[asof_col]).dt.date
                weeks = sorted(df[asof_col].dropna().unique())
                sel = st.selectbox("Portfolio week end", weeks, index=len(weeks) - 1)
                df = df[df[asof_col] == sel].copy()

            weight_col = None
            for c in ["target_weight", "weight", "w_target", "portfolio_weight"]:
                if c in df.columns:
                    weight_col = c
                    break

            if weight_col:
                df = df.sort_values(weight_col, ascending=False)

            show_cols = [c for c in [asof_col, "ticker", weight_col, "rank", "overall_score_ai", "overall_score", "theme", "sector"] if c and c in df.columns]
            show_cols = _unique_list(show_cols)

            if "ticker" in df.columns:
                df["url"] = df["ticker"].astype(str).map(lambda x: ticker_url(x, LINK_PROVIDER))
                if "url" not in show_cols:
                    show_cols.append("url")

            st_df(df[show_cols], column_config={"url": st.column_config.LinkColumn("url")})

    # ---- Alerts
    with tabs[4]:
        st.subheader("Alerts feed")
        if alerts.empty:
            st.warning("Missing data/curated/alerts.parquet")
        else:
            df = alerts.copy()
            # Normalize date
            dcol = "date" if "date" in df.columns else ("asof_date" if "asof_date" in df.columns else None)
            if dcol:
                df[dcol] = pd.to_datetime(df[dcol]).dt.date

            # Filters
            c1, c2, c3 = st.columns(3)
            if dcol:
                dates = sorted(df[dcol].dropna().unique())
                sel_date = c1.selectbox("Date", dates, index=len(dates) - 1)
                df = df[df[dcol] == sel_date].copy()
            if "alert_type" in df.columns:
                types = ["(all)"] + sorted([x for x in df["alert_type"].dropna().unique()])
                sel_type = c2.selectbox("Type", types, index=0)
                if sel_type != "(all)":
                    df = df[df["alert_type"] == sel_type]
            if "severity" in df.columns:
                sevs = ["(all)"] + sorted([x for x in df["severity"].dropna().unique()])
                sel_sev = c3.selectbox("Severity", sevs, index=0)
                if sel_sev != "(all)":
                    df = df[df["severity"] == sel_sev]

            # Sort by severity/value if present
            if "severity_score" in df.columns:
                df = df.sort_values("severity_score", ascending=False)
            elif "value" in df.columns:
                df = df.sort_values("value", ascending=False)

            show_cols = [c for c in [dcol, "ticker", "alert_type", "severity", "direction", "value", "message"] if c and c in df.columns]
            show_cols = _unique_list(show_cols)

            if "ticker" in df.columns:
                df["url"] = df["ticker"].astype(str).map(lambda x: ticker_url(x, LINK_PROVIDER))
                if "url" not in show_cols:
                    show_cols.append("url")

            st_df(df.head(500)[show_cols], column_config={"url": st.column_config.LinkColumn("url")})

    # ---- Watchlist changes
    with tabs[5]:
        st.subheader("Watchlist changes (enter/exit)")
        if watch_changes.empty:
            st.warning("Missing data/curated/watchlist_changes.parquet")
        else:
            df = watch_changes.copy()
            asof_col = _pick_asof_col(df)
            if asof_col:
                df[asof_col] = pd.to_datetime(df[asof_col]).dt.date
                weeks = sorted(df[asof_col].dropna().unique())
                sel = st.selectbox("Week end", weeks, index=len(weeks) - 1)
                df = df[df[asof_col] == sel].copy()

            show_cols = [c for c in [asof_col, "ticker", "change_type", "from_rank", "to_rank", "reason"] if c and c in df.columns]
            show_cols = _unique_list(show_cols)

            if "ticker" in df.columns:
                df["url"] = df["ticker"].astype(str).map(lambda x: ticker_url(x, LINK_PROVIDER))
                if "url" not in show_cols:
                    show_cols.append("url")

            st_df(df[show_cols], column_config={"url": st.column_config.LinkColumn("url")})

    # ---- Ticker drill-down
    with tabs[6]:
        st.subheader("Ticker drill-down")
        ticker = st.text_input("Ticker", value="")
        if not ticker and not ranks_latest.empty and "ticker" in ranks_latest.columns:
            ticker = str(ranks_latest["ticker"].iloc[0])

        ticker = str(ticker).strip().upper()
        if ticker:
            url = ticker_url(ticker, LINK_PROVIDER)
            c1, c2 = st.columns([1, 2])
            with c1:
                newtab_link("Open quote in new tab", url)
                if st.button("Force new tab (popup)", key=f"force_{ticker}"):
                    newtab_open(url)
            with c2:
                st.write(url)

            # Scores history (weekly)
            if not scores_weekly.empty and "ticker" in scores_weekly.columns:
                df = scores_weekly.copy()
                df["ticker"] = df["ticker"].astype(str).str.upper()
                sub = df[df["ticker"] == ticker].copy()
                asof_col = _pick_asof_col(sub)
                if not sub.empty and asof_col:
                    sub[asof_col] = pd.to_datetime(sub[asof_col])
                    score_col = None
                    for c in ["overall_score_ai", "overall_score", "score"]:
                        if c in sub.columns:
                            score_col = c
                            break
                    if score_col:
                        sub = sub.sort_values(asof_col).tail(60)
                        fig = plt.figure()
                        plt.plot(sub[asof_col], sub[score_col])
                        plt.title(f"{ticker} weekly score ({score_col})")
                        plt.xticks(rotation=30)
                        st.pyplot(fig)

            # Recent alerts
            if not alerts.empty and "ticker" in alerts.columns:
                a = alerts.copy()
                a["ticker"] = a["ticker"].astype(str).str.upper()
                suba = a[a["ticker"] == ticker].copy()
                if not suba.empty:
                    dcol = "date" if "date" in suba.columns else ("asof_date" if "asof_date" in suba.columns else None)
                    if dcol:
                        suba[dcol] = pd.to_datetime(suba[dcol])
                        suba = suba.sort_values(dcol, ascending=False).head(50)
                    st.caption("Recent alerts (max 50)")
                    cols = [c for c in [dcol, "alert_type", "severity", "direction", "value", "message"] if c and c in suba.columns]
                    st_df(suba[cols])

            # Price chart
            plot_price(prices, ticker)

    # ---- Data health
    with tabs[7]:
        st.subheader("Data health")
        if not stale.empty:
            st.caption("Stale tickers (weekly)")
            st_df(stale.head(500))
        else:
            st.info("stale_tickers_weekly.parquet not found (optional).")

        # Coverage check
        if not scores_weekly.empty and "ticker" in scores_weekly.columns:
            st.caption("Weekly coverage (tickers per week)")
            df = scores_weekly.copy()
            asof_col = _pick_asof_col(df)
            if asof_col:
                df[asof_col] = pd.to_datetime(df[asof_col]).dt.date
                cov = df.groupby(asof_col)["ticker"].nunique().reset_index(name="n_tickers")
                st_df(cov.tail(50))

    st.caption("If a tab is empty, it usually means the corresponding parquet file is missing. Run your daily/weekly pipeline first.")


if __name__ == "__main__":
    main()
