from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# --- yfinance import: provide a clearer runtime message if missing ---
try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    yf = None
    _YF_IMPORT_ERROR = e


# Canonical ticker -> Yahoo ticker mapping
# Keep universe/pipeline canonical (e.g. BRK.B) and map ONLY when calling Yahoo.
_TICKER_ALIASES_YAHOO: Dict[str, str] = {
    "BRK.B": "BRK-B",
    # If you ever need it:
    # "BF.B": "BF-B",
}


def _map_tickers_to_yahoo(canonical_tickers: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
      yahoo_tickers: list of symbols passed to yfinance
      yahoo_to_canon: mapping for converting yfinance symbols back to canonical symbols
    """
    yahoo_tickers: List[str] = []
    yahoo_to_canon: Dict[str, str] = {}

    for t in canonical_tickers:
        canon = str(t).strip().upper()
        yh = _TICKER_ALIASES_YAHOO.get(canon, canon)
        yahoo_tickers.append(yh)
        yahoo_to_canon[yh] = canon

    return yahoo_tickers, yahoo_to_canon


@dataclass
class YahooAdapter:
    cache_dir: Path

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_prices_daily(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Fetch daily price history (Adjusted Close + Volume) via yfinance.

        Returns DataFrame with columns:
            ticker, date, adj_close, volume

        NOTE: Supports canonical tickers like 'BRK.B' by mapping to Yahoo 'BRK-B'
        while keeping output 'ticker' as canonical.
        """
        if yf is None:  # pragma: no cover
            raise ImportError(
                "yfinance is not installed (or not available in this environment). "
                "Activate your venv then run: pip install yfinance\n"
                f"Original import error: {_YF_IMPORT_ERROR}"
            )

        canonical = [str(t).strip().upper() for t in tickers]
        yahoo_tickers, yahoo_to_canon = _map_tickers_to_yahoo(canonical)

        df = yf.download(
            tickers=yahoo_tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )

        if df is None or df.empty:
            raise RuntimeError("yfinance returned empty price data.")

        def to_long(frame: pd.DataFrame, field: str, out_name: str) -> pd.DataFrame:
            """
            Convert yfinance wide format to long format safely.
            Output columns: date, ticker, <out_name>
            """
            if isinstance(frame.columns, pd.MultiIndex):
                if field not in frame.columns.get_level_values(0):
                    raise KeyError(f"Expected field '{field}' not found in yfinance MultiIndex columns.")
                sub = frame[field].copy()
            else:
                if field not in frame.columns:
                    raise KeyError(f"Expected column '{field}' not found.")
                # Single-ticker fallback
                sub = frame[[field]].copy()
                sub.columns = yahoo_tickers[:1]

            long_df = sub.stack().rename(out_name).reset_index()
            cols = list(long_df.columns)
            long_df = long_df.rename(columns={cols[0]: "date", cols[1]: "ticker"})
            long_df["date"] = pd.to_datetime(long_df["date"]).dt.normalize()

            # Map Yahoo symbol back to canonical symbol
            long_df["ticker"] = (
                long_df["ticker"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(lambda x: yahoo_to_canon.get(x, x))
            )
            return long_df

        adj_long = to_long(df, "Adj Close", "adj_close")
        vol_long = to_long(df, "Volume", "volume")

        merged = adj_long.merge(vol_long, on=["date", "ticker"], how="inner")
        merged = merged.dropna(subset=["adj_close"])
        merged["volume"] = merged["volume"].fillna(0.0).astype(float)

        return merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    def fetch_fundamentals_quarterly(
        self,
        tickers: list[str],
        availability_lag_days: int = 45
    ) -> pd.DataFrame:
        """
        Best-effort quarterly fundamentals using yfinance statements.

        IMPORTANT:
        - yfinance is NOT point-in-time accurate
        - availability_date is approximated as period_end + availability_lag_days

        NOTE: Supports canonical tickers like 'BRK.B' by mapping to Yahoo 'BRK-B'
        while keeping output 'ticker' as canonical.
        """
        if yf is None:  # pragma: no cover
            raise ImportError(
                "yfinance is not installed (or not available in this environment). "
                "Activate your venv then run: pip install yfinance\n"
                f"Original import error: {_YF_IMPORT_ERROR}"
            )

        canonical = [str(t).strip().upper() for t in tickers]
        yahoo_tickers, yahoo_to_canon = _map_tickers_to_yahoo(canonical)

        rows: list[dict] = []

        for i, yh_symbol in enumerate(yahoo_tickers, start=1):
            canon_symbol = yahoo_to_canon.get(yh_symbol, yh_symbol)

            try:
                tk = yf.Ticker(yh_symbol)

                q_is = tk.quarterly_financials
                q_bs = tk.quarterly_balance_sheet
                q_cf = tk.quarterly_cashflow

                periods = set()

                def collect_periods(df_: pd.DataFrame | None) -> None:
                    if df_ is not None and not df_.empty:
                        for c in df_.columns:
                            periods.add(pd.to_datetime(c).normalize())

                collect_periods(q_is)
                collect_periods(q_bs)
                collect_periods(q_cf)

                for period_end in sorted(periods):
                    rec = {
                        "ticker": canon_symbol,  # ✅ keep canonical
                        "period_end": period_end,
                        "filing_date": pd.NaT,
                        "available_date": period_end + pd.Timedelta(days=availability_lag_days),
                        "revenue": np.nan,
                        "net_income": np.nan,
                        "cogs": np.nan,
                        "total_assets": np.nan,
                        "total_liabilities": np.nan,
                        "equity": np.nan,
                        "operating_cash_flow": np.nan,
                        "capital_expenditures": np.nan,
                        "free_cash_flow": np.nan,
                    }

                    def get_cell(df_: pd.DataFrame | None, labels: list[str]) -> float:
                        if df_ is None or df_.empty:
                            return np.nan
                        for col in df_.columns:
                            if pd.to_datetime(col).normalize() == period_end:
                                for lab in labels:
                                    if lab in df_.index:
                                        val = df_.loc[lab, col]
                                        return float(val) if pd.notna(val) else np.nan
                        return np.nan

                    rec["revenue"] = get_cell(q_is, ["Total Revenue", "Revenue"])
                    rec["net_income"] = get_cell(q_is, ["Net Income"])
                    rec["cogs"] = get_cell(q_is, ["Cost Of Revenue"])

                    rec["total_assets"] = get_cell(q_bs, ["Total Assets"])
                    rec["total_liabilities"] = get_cell(q_bs, ["Total Liab", "Total Liabilities"])
                    rec["equity"] = get_cell(q_bs, ["Total Stockholder Equity", "Stockholders Equity"])

                    rec["operating_cash_flow"] = get_cell(
                        q_cf, ["Total Cash From Operating Activities"]
                    )
                    rec["capital_expenditures"] = get_cell(
                        q_cf, ["Capital Expenditures"]
                    )

                    if pd.notna(rec["operating_cash_flow"]) and pd.notna(rec["capital_expenditures"]):
                        rec["free_cash_flow"] = (
                            rec["operating_cash_flow"] - rec["capital_expenditures"]
                        )

                    rows.append(rec)

                if i % 25 == 0:
                    time.sleep(1.0)

            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["period_end"] = pd.to_datetime(df["period_end"]).dt.normalize()
        df["available_date"] = pd.to_datetime(df["available_date"]).dt.normalize()
        return df.sort_values(["ticker", "period_end"]).reset_index(drop=True)
