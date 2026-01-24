from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(data_dir: Path) -> None:
    (data_dir / "raw" / "yahoo").mkdir(parents=True, exist_ok=True)
    (data_dir / "curated").mkdir(parents=True, exist_ok=True)
    (data_dir / "features").mkdir(parents=True, exist_ok=True)
    (data_dir / "backtests").mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_png(equity_curve: pd.DataFrame, path: Path) -> None:
    """
    equity_curve columns: date, strategy_equity, benchmark_equity
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(equity_curve["date"], equity_curve["strategy_equity"], label="Strategy")
    ax.plot(equity_curve["date"], equity_curve["benchmark_equity"], label="QQQ")
    ax.set_title("Equity Curve (Monthly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (Start=1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
