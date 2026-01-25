from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


@dataclass
class FMPAdapter:
    """Financial Modeling Prep (FMP) fundamentals adapter.

    Output contract matches the rest of the pipeline:
        ticker, period_end, filing_date, available_date,
        revenue, net_income, cogs,
        total_assets, total_liabilities, equity,
        operating_cash_flow, capital_expenditures, free_cash_flow

    Env var required:
        FMP_API_KEY

    Optional env vars:
        FMP_LIMIT   (quarters to request per statement; default 24 ~ 6 years buffer)
        FMP_RPM     (requests per minute; overrides requests_per_minute)

    Notes:
    - FMP provides filing/accepted dates for many tickers; we use those as point-in-time 'available_date'.
    - If filing/accepted dates are missing, we fall back to period_end + availability_lag_days.
    """

    cache_dir: Path
    api_key_env: str = "FMP_API_KEY"
    base_url: str = "https://financialmodelingprep.com/stable"


    # Throttle (can override via env FMP_RPM)
    requests_per_minute: int = 50

    # If your plan only supports ~5y history, keep this ~24 quarters (6y buffer)
    statement_limit: int = 24

    availability_lag_days: int = 45

    max_retries: int = 8

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Optional overrides
        rpm_env = os.getenv("FMP_RPM")
        if rpm_env:
            try:
                self.requests_per_minute = max(1, int(rpm_env))
            except ValueError:
                pass

        limit_env = os.getenv("FMP_LIMIT")
        if limit_env:
            try:
                self.statement_limit = max(1, int(limit_env))
            except ValueError:
                pass

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing {self.api_key_env}. Set it as an environment variable."
            )
        return key

    def _throttle(self) -> None:
        if self.requests_per_minute <= 0:
            return
        time.sleep(60.0 / float(self.requests_per_minute))

    
    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        params = dict(params)
        params["apikey"] = self.api_key

        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")

        last_text = ""
        for attempt in range(1, int(getattr(self, "max_retries", 8)) + 1):
            self._throttle()
            resp = requests.get(url, params=params, timeout=60)
            last_text = resp.text[:300]

            if resp.status_code == 200:
                payload = resp.json()
                return payload if isinstance(payload, list) else []

            # Rate limit: exponential backoff
            if resp.status_code == 429:
                wait = min(60, 2 ** attempt)
                print(f"[FMP] 429 rate limit. Backing off {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue
        # Temporary server/network issues: retry
            if resp.status_code in (500, 502, 503, 504):
                wait = min(30, 2 ** attempt)
                print(f"[FMP] {resp.status_code} server error. Retrying in {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue
        # Hard failure: raise
            raise RuntimeError(f"FMP API error {resp.status_code}: {last_text}")

        raise RuntimeError(f"FMP API retry exhausted. Last response: {last_text}")


    @staticmethod
    def _to_float(x: Any) -> float:
        try:
            if x is None or (isinstance(x, str) and not x.strip()):
                return float("nan")
            return float(x)
        except Exception:
            return float("nan")

    @staticmethod
    def _pick_date(rec: dict[str, Any]) -> pd.Timestamp | None:
        # FMP often provides 'fillingDate' (sic) and/or 'acceptedDate'.
        for k in ("fillingDate", "filingDate", "acceptedDate"):
            v = rec.get(k)
            if v:
                try:
                    return pd.to_datetime(v).normalize()
                except Exception:
                    pass
        return pd.NaT

    def _fetch_quarterly_statements(
        self, ticker: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        # API paths use /v3; keep base_url '/api' and include /v3 prefix.
        common = {"symbol": ticker, "period": "quarter", "limit": int(self.statement_limit)}
        inc = self._get("income-statement", common)
        bs  = self._get("balance-sheet-statement", common)
        cf  = self._get("cash-flow-statement", common)

        return inc, bs, cf

    def fetch_fundamentals_quarterly(self, tickers: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        errors = 0

        for t in tickers:
            try:
                inc, bs, cf = self._fetch_quarterly_statements(t)

                # Index by period_end ("date" field)
                def index_by_date(
                    lst: list[dict[str, Any]]
                ) -> dict[pd.Timestamp, dict[str, Any]]:
                    out: dict[pd.Timestamp, dict[str, Any]] = {}
                    for r in lst:
                        d = r.get("date")
                        if not d:
                            continue
                        try:
                            dt = pd.to_datetime(d).normalize()
                        except Exception:
                            continue
                        out[dt] = r
                    return out

                inc_i = index_by_date(inc)
                bs_i = index_by_date(bs)
                cf_i = index_by_date(cf)

                all_periods = sorted(set(inc_i) | set(bs_i) | set(cf_i))
                for pe in all_periods:
                    r_inc = inc_i.get(pe, {})
                    r_bs = bs_i.get(pe, {})
                    r_cf = cf_i.get(pe, {})

                    filing_date = self._pick_date(r_inc) if r_inc else self._pick_date(r_bs)
                    if pd.isna(filing_date):
                        filing_date = self._pick_date(r_cf)

                    available_date = (
                        filing_date
                        if pd.notna(filing_date)
                        else pe + pd.Timedelta(days=int(self.availability_lag_days))
                    )

                    revenue = self._to_float(r_inc.get("revenue"))
                    net_income = self._to_float(r_inc.get("netIncome"))
                    cogs = self._to_float(r_inc.get("costOfRevenue"))

                    total_assets = self._to_float(r_bs.get("totalAssets"))
                    total_liabilities = self._to_float(r_bs.get("totalLiabilities"))
                    equity = self._to_float(r_bs.get("totalStockholdersEquity"))

                    operating_cf = self._to_float(r_cf.get("operatingCashFlow"))
                    capex = self._to_float(r_cf.get("capitalExpenditure"))

                    free_cf = float("nan")
                    if np.isfinite(operating_cf) and np.isfinite(capex):
                        # FMP capex is usually negative; keep sign-consistent with your pipeline:
                        # free_cash_flow = OCF - capex
                        free_cf = operating_cf - capex

                    rows.append(
                        {
                            "ticker": str(t),
                            "period_end": pe,
                            "filing_date": filing_date,
                            "available_date": pd.to_datetime(available_date).normalize(),
                            "revenue": revenue,
                            "net_income": net_income,
                            "cogs": cogs,
                            "total_assets": total_assets,
                            "total_liabilities": total_liabilities,
                            "equity": equity,
                            "operating_cash_flow": operating_cf,
                            "capital_expenditures": capex,
                            "free_cash_flow": free_cf,
                        }
                    )

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"[FMP] {t}: ERROR {type(e).__name__}: {e}")
                continue

        if not rows:
            raise RuntimeError(
                "FMP fundamentals returned 0 rows (all tickers failed). "
                "Check FMP_API_KEY, endpoint access, and plan limits. "
                "First errors were printed above."
            )

        df = pd.DataFrame(rows)
        df["period_end"] = pd.to_datetime(df["period_end"]).dt.normalize()
        df["available_date"] = pd.to_datetime(df["available_date"]).dt.normalize()
        if "filing_date" in df.columns:
            df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.normalize()

        # Deduplicate if any duplicates due to statement overlap
        df = df.sort_values(["ticker", "period_end", "available_date"]).drop_duplicates(
            subset=["ticker", "period_end"], keep="last"
        )

        return df.sort_values(["ticker", "period_end"]).reset_index(drop=True)
