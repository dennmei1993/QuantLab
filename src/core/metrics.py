from __future__ import annotations

import pandas as pd
import numpy as np


def _cagr(equity_end: float, n_years: float) -> float:
    if n_years <= 0 or equity_end <= 0:
        return float("nan")
    return equity_end ** (1.0 / n_years) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    s = equity.astype(float)
    running_max = s.cummax()
    dd = s / running_max - 1.0
    return float(dd.min())


def summarize_backtest(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"error": "empty backtest"}

    # Using monthly steps
    rets = bt["net_return"].astype(float).dropna()
    bench = bt["benchmark_return"].astype(float).dropna()

    n_months = len(rets)
    n_years = n_months / 12.0

    strat_equity = (1.0 + rets).cumprod()
    bench_equity = (1.0 + bench).cumprod() if len(bench) == n_months else None

    cagr = _cagr(float(strat_equity.iloc[-1]), n_years)
    vol = float(rets.std(ddof=1) * np.sqrt(12)) if n_months > 1 else float("nan")
    sharpe = float((rets.mean() * 12) / vol) if (pd.notna(vol) and vol > 0) else float("nan")
    mdd = _max_drawdown(strat_equity)

    hit = float((rets > bt["benchmark_return"]).mean())

    avg_turnover = float(bt["turnover"].mean())
    avg_cost = float(bt["cost"].mean())
    avg_holdings = float(bt["n_holdings"].mean())

    out = {
        "months": n_months,
        "CAGR": round(cagr, 4),
        "AnnVol": round(vol, 4),
        "Sharpe": round(sharpe, 4),
        "MaxDrawdown": round(mdd, 4),
        "HitRate_vs_Benchmark": round(hit, 4),
        "AvgTurnover": round(avg_turnover, 4),
        "AvgCost_per_month": round(avg_cost, 6),
        "AvgHoldings": round(avg_holdings, 2),
    }

    if bench_equity is not None:
        bench_cagr = _cagr(float(bench_equity.iloc[-1]), n_years)
        out["Benchmark_CAGR"] = round(bench_cagr, 4)

    return out
