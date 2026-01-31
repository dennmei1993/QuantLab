from __future__ import annotations

import pandas as pd
import numpy as np


def _month_trade_dates(asof_dates: list[pd.Timestamp], prices_idx: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Map each month-end signal date -> next trading date (close-to-close execution simplified).
    Returns list of (signal_date_used, trade_date_used).
    """
    out = []
    for d in asof_dates:
        # ensure signal date is a trading day (or prior trading day)
        eligible = prices_idx[prices_idx <= d]
        if len(eligible) == 0:
            continue
        signal = eligible[-1]

        # trade next trading day
        future = prices_idx[prices_idx > signal]
        if len(future) == 0:
            continue
        trade = future[0]
        out.append((signal.normalize(), trade.normalize()))
    return out


def run_backtest(
    portfolio: pd.DataFrame,
    prices_daily: pd.DataFrame,
    benchmark_ticker: str = "QQQ",
    trading_cost_bps: float = 10.0,
    trade_at: str = "next_close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monthly rebalance backtest:
      - Signal at month-end (asof_date)
      - Execute at next trading day close (trade_at=next_close)
      - Hold until next rebalance

    Costs:
      - simple bps cost applied to turnover each rebalance (both buy + sell)
    """
    if trade_at != "next_close":
        raise NotImplementedError("Only trade_at='next_close' implemented in MVP scaffold.")

    prices = prices_daily.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    px = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()

    # Ignore SELL rows (weight == 0) – UI only, not tradable

    # Portfolio file may include BUY/HOLD/SELL rows for UI.
    # Backtest should only consider active positions (weight > 0).

    
    port = portfolio.copy()

    # --- Guard rails: empty portfolio / missing columns ---
    required_cols = {"asof_date", "ticker"}
    if port is None or port.empty:
        raise RuntimeError(
            "Portfolio is empty — cannot backtest. "
            "This usually means scoring produced no investable tickers for the chosen window."
        )
    missing = required_cols - set(port.columns)
    if missing:
        raise RuntimeError(
            f"Portfolio missing required columns: {sorted(missing)}. "
            f"Columns present: {list(port.columns)}"
        )

    if "weight" in port.columns:
        port = port[port["weight"] > 0.0].copy()

    port["asof_date"] = pd.to_datetime(port["asof_date"]).dt.normalize()

    asof_dates = sorted(port["asof_date"].unique())
    trade_pairs = _month_trade_dates(asof_dates, px.index)

    # Build positions at each rebalance date, executed on trade_date close
    # Then compute next period return from trade_date to next trade_date.
    results = []
    prev_weights = {}  # ticker -> weight

    for i in range(len(trade_pairs) - 1):
        signal_date, trade_date = trade_pairs[i]
        _, next_trade_date = trade_pairs[i + 1]

        # portfolio weights for this signal month
        w_df = port[port["asof_date"] == signal_date][["ticker", "weight"]].copy()
        if w_df.empty:
            continue
        weights = dict(zip(w_df["ticker"], w_df["weight"]))

        # compute turnover vs previous weights
        tickers = sorted(set(prev_weights.keys()) | set(weights.keys()))
        turnover = 0.0
        for t in tickers:
            turnover += abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0))
        # turnover is sum abs changes; round-trip cost per turnover dollar:
        # approximate cost = turnover * cost_per_side (bps)  (already includes both buys/sells via turnover)
        cost = turnover * (trading_cost_bps / 10000.0)

        # compute holding period returns close-to-close
        # strategy return = sum w_i * (P_next / P_trade - 1)
        ret = 0.0
        missing = 0
        for t, w in weights.items():
            if t not in px.columns:
                missing += 1
                continue
            p0 = px.at[trade_date, t] if trade_date in px.index else np.nan
            p1 = px.at[next_trade_date, t] if next_trade_date in px.index else np.nan
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                missing += 1
                continue
            ret += w * (p1 / p0 - 1.0)

        # benchmark return
        b0 = px.at[trade_date, benchmark_ticker]
        b1 = px.at[next_trade_date, benchmark_ticker]
        bench_ret = (b1 / b0 - 1.0) if (pd.notna(b0) and pd.notna(b1) and b0 != 0) else np.nan

        net_ret = ret - cost

        results.append({
            "asof_date": signal_date,
            "trade_date": trade_date,
            "next_trade_date": next_trade_date,
            "gross_return": ret,
            "cost": cost,
            "net_return": net_ret,
            "benchmark_return": bench_ret,
            "turnover": turnover,
            "missing_assets": missing,
            "n_holdings": len(weights),
        })

        prev_weights = weights

    bt = pd.DataFrame(results)
    if bt.empty:
        return bt, pd.DataFrame()

    bt = bt.sort_values("trade_date").reset_index(drop=True)

    # equity curve (monthly points)
    eq = []
    strat = 1.0
    bench = 1.0
    for _, r in bt.iterrows():
        strat *= (1.0 + float(r["net_return"]))
        bench *= (1.0 + float(r["benchmark_return"])) if pd.notna(r["benchmark_return"]) else 1.0
        eq.append({"date": r["next_trade_date"], "strategy_equity": strat, "benchmark_equity": bench})
    equity = pd.DataFrame(eq)

    return bt, equity
