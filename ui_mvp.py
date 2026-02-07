# app_overview_mvp.py
# Portfolio Copilot — MVP Overview (Stable Price Convention)

from pathlib import Path
from dataclasses import dataclass
from turtle import color
from typing import Dict

import pandas as pd
import streamlit as st

# =========================================================
# Session State
# =========================================================

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None


# =========================================================
# Config
# =========================================================

APP_TITLE = "Portfolio Copilot"
DATA_DIR = Path(__file__).parent / "data" / "curated"
FEATURE_DIR = Path(__file__).parent / "data" / "features"

DATASETS = {
    "prices_daily": DATA_DIR / "prices_daily.parquet",
    "portfolio_target": DATA_DIR / "portfolio_target.parquet",
    "scores_weekly": DATA_DIR / "scores_weekly.parquet",
    "ranks_latest": DATA_DIR / "ranks_latest.parquet",
    "metrics_monthly": DATA_DIR / "metrics_monthly.parquet",
    "watchlist_changes": DATA_DIR / "watchlist_changes.parquet",
    "company_overview": DATA_DIR / "company_overview.parquet",
    "market_regime": DATA_DIR / "market_regime_weekly.parquet",
    "ai_weights_weekly": FEATURE_DIR / "ai_weights_weekly.parquet",
}

STATE_DIR = Path(__file__).parent / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

USER_PORTFOLIO_PATH = STATE_DIR / "user_portfolio.csv"

# =========================================================
# Utilities
# =========================================================

def render_style_panel(info):

    style = compute_style_leadership(info)

    st.write(info.get("ai_weights_weekly", pd.DataFrame()).head())

    if not style:
        st.info("Style leadership data unavailable.")
        return

    st.subheader("Factor Leadership")

    c1, c2 = st.columns(2)

    c1.metric(
        "Leading Factor",
        style["leader_bucket"].capitalize()
    )

    c2.metric(
        "Weight Strength",
        f"{style['leader_weight']:.2f}"
    )

   



def compute_style_leadership(info):

    weights = info.get("ai_weights_weekly", pd.DataFrame())

    if weights.empty:
        return None

    if "asof_date" not in weights.columns:
        return None

    latest = weights["asof_date"].max()
    df = weights[weights["asof_date"] == latest]

    if df.empty:
        return None

    row = df.iloc[0].drop("asof_date")

    if row.empty:
        return None

    leader_bucket = row.idxmax()
    leader_weight = row.max()

    return {
        "leader_bucket": leader_bucket,
        "leader_weight": leader_weight,
        "weights": row
    }



def render_alignment_gauge(info, snapshot):

    regime_df = info.get("market_regime", pd.DataFrame())
    portfolio = compute_portfolio_positioning(snapshot)

    if regime_df.empty or not portfolio:
        return

    strong_pct = regime_df.iloc[-1]["strong_pct"]
    growth_weight = portfolio["growth_weight"]

    st.subheader("Market vs Portfolio Alignment")

    c1, c2 = st.columns(2)

    c1.metric(
        "Market Strength",
        f"{strong_pct*100:.0f}% Strong"
    )

    c2.metric(
        "Portfolio Growth Exposure",
        f"{growth_weight*100:.0f}%"
    )

    diff = growth_weight - strong_pct

    if abs(diff) < 0.1:
        st.success("Portfolio exposure closely aligned with market strength.")
    elif diff > 0:
        st.warning("Portfolio more aggressive than market breadth.")
    else:
        st.info("Portfolio more defensive than market breadth.")

def render_regime_history(info):
    regime_df = info.get("market_regime", pd.DataFrame())

    if regime_df.empty:
        st.info("No regime history available.")
        return

    df = regime_df.copy()
    df["week_end"] = pd.to_datetime(df["week_end"])

    st.subheader("Regime History")

    chart_df = df.sort_values("week_end")

    st.line_chart(
        chart_df.set_index("week_end")[["strong_pct", "weak_pct"]],
        height=200
    )

def compute_portfolio_positioning(snapshot):
    table = snapshot.get("table")

    if table is None or table.empty:
        return {}

    growth_weight = table[table["weight"] > 0.05]["weight"].sum()
    top3_concentration = table.head(3)["weight"].sum()
    total_value = table["position_value"].sum()
    total_pnl = table["unrealized_pnl"].sum()

    return {
        "growth_weight": growth_weight,
        "top3_concentration": top3_concentration,
        "total_value": total_value,
        "total_pnl": total_pnl,
    }

USER_PORTFOLIO_PATH = Path("data/state/user_portfolio.csv")

def load_user_portfolio():
    if not USER_PORTFOLIO_PATH.exists():
        st.warning("user_portfolio.csv not found.")
        return pd.DataFrame(columns=["ticker", "units", "avg_cost"])

    # Try safe encoding
    try:
        df = pd.read_csv(USER_PORTFOLIO_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(USER_PORTFOLIO_PATH, encoding="latin1")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Detect required columns
    required_cols = ["ticker", "units", "avg_cost"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.write("Detected columns:", list(df.columns))
        return pd.DataFrame(columns=required_cols)

    # Clean data
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0.0)
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce").fillna(0.0)

    # Remove invalid rows
    df = df[df["units"] > 0].copy()

    return df[required_cols]


def save_user_portfolio(df):
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    total = df["weight"].sum()
    if total > 0:
        df["weight"] = df["weight"] / total

    df.to_csv(USER_PORTFOLIO_PATH, index=False)

def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()


def load_datasets() -> Dict[str, pd.DataFrame]:
    return {k: safe_read_parquet(v) for k, v in DATASETS.items()}


def format_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"


# =========================================================
# Copilot Cards
# =========================================================

@dataclass
class CopilotCards:
    market_mode: tuple[str, str]
    portfolio_stance: tuple[str, str]
    key_watch: tuple[str, str]
    copilot_note: tuple[str, str]


def build_copilot_cards(info, snapshot):

    portfolio = compute_portfolio_positioning(snapshot)
    regime_df = info.get("market_regime", pd.DataFrame())

    if regime_df.empty or not portfolio:
        return CopilotCards(
            market_mode=("—", ""),
            portfolio_stance=("—", ""),
            key_watch=("—", ""),
            copilot_note=("No data available", "")
        )

    row = regime_df.iloc[-1]

    regime = row.get("regime", "—")
    avg_score = row.get("avg_score", None)
    strong_pct = row.get("strong_pct", None)
    weak_pct = row.get("weak_pct", None)

    # -----------------------
    # Portfolio stance logic
    # -----------------------
    growth_weight = portfolio["growth_weight"]
    top3 = portfolio["top3_concentration"]

    if growth_weight > 0.6:
        stance = "Growth-Heavy"
    elif growth_weight < 0.4:
        stance = "Defensive-Leaning"
    else:
        stance = "Balanced"

    # -----------------------
    # Alignment logic
    # -----------------------
    if regime == "Risk-On" and growth_weight > 0.6:
        alignment = "Positioning aligned with Risk-On regime"
    elif regime == "Risk-Off" and growth_weight < 0.4:
        alignment = "Positioning aligned with Risk-Off regime"
    else:
        alignment = "Portfolio positioning diverges from regime"

    return CopilotCards(
        market_mode=(
            regime,
            f"Avg score {avg_score:.1f} | "
            f"Strong {strong_pct*100:.0f}% | Weak {weak_pct*100:.0f}%"
        ),
        portfolio_stance=(
            stance,
            f"Top 3 concentration {top3*100:.0f}%"
        ),
        key_watch=(
            "Concentration Risk" if top3 > 0.4 else "Diversified",
            f"Top 3 = {top3*100:.0f}%"
        ),
        copilot_note=(
            alignment,
            f"Total value ${portfolio['total_value']:,.0f} | "
            f"Unrealized P&L ${portfolio['total_pnl']:,.0f}"
        ),
    )


# =========================================================
# Portfolio Snapshot (price-safe)
# =========================================================

def compute_portfolio_snapshot(info):
    portfolio = load_user_portfolio()
    prices = info.get("prices_daily", pd.DataFrame())

    if portfolio.empty or prices.empty:
        return {"table": pd.DataFrame(), "asof": None}

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    latest = prices["date"].max()

    # Get latest available price per ticker
    latest_prices = (
        prices[prices["date"] <= latest]
        .sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "adj_close"]]
        .rename(columns={"adj_close": "price"})
    )

    merged = portfolio.merge(latest_prices, on="ticker", how="left")

    # Calculate metrics
    merged["position_value"] = merged["units"] * merged["price"]
    total_value = merged["position_value"].sum()

    if total_value > 0:
        merged["weight"] = merged["position_value"] / total_value
    else:
        merged["weight"] = 0.0

    merged["unrealized_pnl"] = (
        (merged["price"] - merged["avg_cost"]) * merged["units"]
    )

    return {
        "table": merged.sort_values("weight", ascending=False),
        "asof": latest
    }



def compute_portfolio_snapshot_for_parquet(info):

    portfolio = load_user_portfolio()

    st.write("Portfolio columns:", list(portfolio.columns))

    prices = info.get("prices_daily", pd.DataFrame())

    if portfolio.empty:
        return {"table": pd.DataFrame(), "asof": None}

    # 🔑 FIX: isolate current portfolio
    # 🔍 Detect portfolio date column
    date_col = next(
        (c for c in ["asof_date", "date", "rebalance_date", "month"] if c in portfolio.columns),
        None
    )

    if date_col is None:
        st.warning("No date column found in portfolio_target — using all rows.")
    else:
        portfolio[date_col] = pd.to_datetime(portfolio[date_col])
        latest = portfolio[date_col].max()

        portfolio = portfolio[
            (portfolio[date_col] == latest) &
            (portfolio["weight"] > 0)
        ].copy()

    # --- weights ---
    weight_col = next((c for c in ["weight", "target_weight", "allocation", "pct"] if c in portfolio), None)
    portfolio = portfolio.copy()

    if weight_col is None:
        st.warning("No weight column found in portfolio_target.")
        return {"table": pd.DataFrame(), "asof": None}

    # 🔐 Force numeric weights
    #portfolio["weight"] = pd.to_numeric(portfolio["weight"], errors="coerce").fillna(0.0)

    # ✅ FINAL FILTER — THIS IS THE IMPORTANT LINE
    portfolio = portfolio[portfolio["weight"] > 0].copy()

    if portfolio.empty:
        st.warning("No active holdings (all weights are zero).")
        return {"table": portfolio, "asof": None}

    # --- prices ---
    if prices.empty or not {"ticker", "adj_close", "date"}.issubset(prices.columns):
        return {"table": portfolio, "asof": None}

    latest_prices = (
        prices.sort_values("date")
        .groupby("ticker")
        .tail(1)[["ticker", "adj_close", "date"]]
        .rename(columns={"adj_close": "price"})
    )

    merged = portfolio.merge(latest_prices, on="ticker", how="left")

    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce")

    wsum = merged["weight"].sum()
    if wsum > 0:
        merged["weight"] /= wsum

    merged["value"] = merged["weight"]  # placeholder, no price here

    return {
        "table": merged.sort_values("weight", ascending=False),
        "asof": merged["date"].max(),
    }


# =========================================================
# Portfolio Returns (SAME price convention)
# =========================================================

def compute_portfolio_returns(snapshot, prices):
    table = snapshot.get("table")

    if table is None or table.empty:
        return {"MTD": None, "YTD": None}

    if prices.empty or not {"ticker", "adj_close", "date"}.issubset(prices.columns):
        return {"MTD": None, "YTD": None}

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    def prices_asof(asof):
        return (
            prices[prices["date"] <= asof]
            .sort_values("date")
            .groupby("ticker", as_index=False)
            .tail(1)
            .rename(columns={"adj_close": "price"})
        )

    def portfolio_value(asof_date):
        df = prices_asof(asof_date)

        if df.empty:
            return None

        m = table.merge(
            df[["ticker", "price"]],
            on="ticker",
            how="left",
            validate="many_to_one"
        )

        # ⬇️ GUARANTEE price exists
        if "price" not in m.columns:
            return None

        m["weight"] = pd.to_numeric(m["weight"], errors="coerce")
        m["price"] = pd.to_numeric(m["price"], errors="coerce")

        return (m["weight"] * m["price"]).sum()

    latest = prices["date"].max()
    mtd_start = latest.replace(day=1)
    ytd_start = latest.replace(month=1, day=1)

    cur = portfolio_value(latest)
    mtd = portfolio_value(mtd_start)
    ytd = portfolio_value(ytd_start)

    return {
        "MTD": (cur / mtd - 1) if cur and mtd else None,
        "YTD": (cur / ytd - 1) if cur and ytd else None,
    }




# =========================================================
# Portfolio Holdings (Clickable)
# =========================================================

def render_portfolio_holdings(snapshot):
    st.subheader("Portfolio Holdings")

    table = snapshot.get("table")
    asof = snapshot.get("asof")

    if table is None or table.empty:
        st.info("Portfolio is empty.")
        return

    st.caption(f"Prices as of {asof.date()}")

    display_cols = [
        "ticker",
        "units",
        "avg_cost",
        "price",
        "position_value",
        "weight",
        "unrealized_pnl",
    ]

    formatted = table[display_cols].copy()

    formatted["price"] = formatted["price"].round(2)
    formatted["position_value"] = formatted["position_value"].round(2)
    formatted["weight"] = (formatted["weight"] * 100).round(2)
    formatted["unrealized_pnl"] = formatted["unrealized_pnl"].round(2)

    def highlight_pnl(val):
        color = "green" if val > 0 else "red"
        return f"color: {color}"

    st.dataframe(
        formatted.style.applymap(highlight_pnl, subset=["unrealized_pnl"]),
        use_container_width=True
    )


# =========================================================
# Watchlist
# =========================================================

def compute_watchlist_highlights(info):
    ranks = info.get("ranks_latest", pd.DataFrame())
    if ranks.empty:
        return pd.DataFrame({"Ticker": ["—"]})
    return ranks[["ticker"]].rename(columns={"ticker": "Ticker"}).head(5)


def render_watchlist_highlights(info):
    st.subheader("Watchlist Highlights")

    df = compute_watchlist_highlights(info)
    for _, row in df.iterrows():
        if st.button(row["Ticker"], key=f"watch_{row['Ticker']}"):
            st.session_state.selected_ticker = row["Ticker"]


# =========================================================
# Detail Panel
# =========================================================

def render_ticker_detail(info):
    ticker = st.session_state.get("selected_ticker")
    if not ticker:
        return

    st.divider()
    st.subheader(f"{ticker} — Detail")

    prices = info.get("prices_daily", pd.DataFrame())
    overview = info.get("company_overview", pd.DataFrame())
    scores = info.get("scores_weekly", pd.DataFrame())

    if not prices.empty and {"ticker", "adj_close", "date"}.issubset(prices.columns):
        df = prices[prices["ticker"] == ticker].sort_values("date")
        st.line_chart(df.set_index("date")["adj_close"])

    if not overview.empty:
        r = overview[overview["ticker"] == ticker]
        if not r.empty:
            st.write("Sector:", r.iloc[0].get("sector", "—"))

    if "overall_score_ai" in scores:
        r = scores[scores["ticker"] == ticker]
        if not r.empty:
            st.metric("Factor Score", f"{r.iloc[-1]['overall_score_ai']:.2f}")

    if st.button("Close"):
        st.session_state.selected_ticker = None

    st.divider()
    st.subheader("Add / Update Position")


    current_portfolio = load_user_portfolio()
    existing_weight = 0.0

    if ticker in current_portfolio["ticker"].values:
        existing_weight = current_portfolio.loc[
            current_portfolio["ticker"] == ticker, "weight"
        ].iloc[0]

    new_weight = st.number_input(
        "Target Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(existing_weight),
        step=0.01
    )

    if st.button("Save Position"):
        df = current_portfolio.copy()

        if ticker in df["ticker"].values:
            df.loc[df["ticker"] == ticker, "weight"] = new_weight
        else:
            df = pd.concat(
                [df, pd.DataFrame([{"ticker": ticker, "weight": new_weight}])],
                ignore_index=True
            )

        save_user_portfolio(df)
        st.success("Position saved.")
        st.rerun()

    if st.button("Remove Position"):
        df = current_portfolio[current_portfolio["ticker"] != ticker]
        save_user_portfolio(df)
        st.success("Position removed.")
        st.rerun()



# =========================================================
# Main
# =========================================================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    info = load_datasets()

    lh, rh = st.columns([0.7, 0.3])
    with lh:
        st.title("Portfolio Copilot")
        st.caption("Overview")


    with rh:
        prices = info.get("prices_daily", pd.DataFrame())
        if not prices.empty:
            st.caption(f"Data as of {prices['date'].max().date()}")

    snap = compute_portfolio_snapshot(info)
    cards = build_copilot_cards(info, snap)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Regime", cards.market_mode[0], cards.market_mode[1])
    c2.metric("Portfolio Position", cards.portfolio_stance[0], cards.portfolio_stance[1])
    c3.metric("Key Watch", cards.key_watch[0], cards.key_watch[1])
    c4.write(cards.copilot_note[0])
    st.caption(cards.copilot_note[1])

    render_regime_history(info)

    render_alignment_gauge(info, snap)

    render_style_panel(info)


    left, right = st.columns([0.6, 0.4])

    #with left:
    

    with right:
        snap = compute_portfolio_snapshot(info)
        returns = compute_portfolio_returns(snap, info.get("prices_daily", pd.DataFrame()))
        st.metric("MTD", format_pct(returns["MTD"]))
        st.metric("YTD", format_pct(returns["YTD"]))
        render_portfolio_holdings(snap)

    render_watchlist_highlights(info)
    render_ticker_detail(info)


if __name__ == "__main__":
    main()
