from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
import json
import hashlib
from typing import Any

import pandas as pd
import requests

def normalize_fmp_symbol(ticker: str) -> str:
    # Convert dot-class to dash-class or parent
    t = ticker.upper()
    if t == "BRK.B":
        return "BRK-A"   # or "BRK.A" depending on FMP endpoint
    if t == "BF.B":
        return "BF-A"
    return t


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
        FMP_LIMIT               (quarters to request per statement; default 24 ~ 6 years buffer)
        FMP_RPM                 (requests per minute; overrides requests_per_minute)
        FMP_USE_HTTP_CACHE      (0/1, default 1) cache raw HTTP JSON responses to disk
        FMP_HTTP_CACHE_DAYS     (default 7) max age for cached HTTP responses
    """

    cache_dir: Path
    api_key_env: str = "FMP_API_KEY"
    base_url: str = "https://financialmodelingprep.com/stable"

    # Throttle (can override via env FMP_RPM)
    requests_per_minute: int = 50

    # ~24 quarters = 6y buffer
    statement_limit: int = 24

    availability_lag_days: int = 45
    max_retries: int = 8

    # HTTP response cache settings (disk)
    use_http_cache: bool = True
    http_cache_days: int = 7

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "http_cache").mkdir(parents=True, exist_ok=True)

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

        use_cache_env = os.getenv("FMP_USE_HTTP_CACHE")
        if use_cache_env is not None:
            self.use_http_cache = bool(int(use_cache_env))

        cache_days_env = os.getenv("FMP_HTTP_CACHE_DAYS")
        if cache_days_env:
            try:
                self.http_cache_days = max(0, int(cache_days_env))
            except ValueError:
                pass

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing {self.api_key_env}. Set it as an environment variable.")
        return key

    def _throttle(self) -> None:
        if self.requests_per_minute <= 0:
            return
        time.sleep(60.0 / float(self.requests_per_minute))

    # -------------------------
    # Disk cache helpers
    # -------------------------
    def _cache_key(self, url: str, params: dict[str, Any]) -> str:
        # Stable hash so filenames are safe and short
        payload = {"url": url, "params": params}
        s = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / "http_cache" / f"{key}.json"

    def _cache_is_fresh(self, p: Path) -> bool:
        if not p.exists():
            return False
        if self.http_cache_days <= 0:
            return False
        age_seconds = time.time() - p.stat().st_mtime
        return age_seconds <= (self.http_cache_days * 86400)

    def _cache_read(self, p: Path) -> list[dict[str, Any]]:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    def _cache_write(self, p: Path, payload: Any) -> None:
        try:
            p.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            # cache failures should never crash the pipeline
            pass

    # -------------------------
    # HTTP
    # -------------------------
    def _get(self, path: str, params: dict[str, Any], *, force_refresh: bool = False) -> list[dict[str, Any]]:
        params = dict(params)
        params["apikey"] = self.api_key

        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")

        # Disk cache (raw JSON list)
        if self.use_http_cache and not force_refresh:
            key = self._cache_key(url, params)
            cp = self._cache_path(key)
            if self._cache_is_fresh(cp):
                return self._cache_read(cp)

        last_text = ""
        for attempt in range(1, int(getattr(self, "max_retries", 8)) + 1):
            self._throttle()
            resp = requests.get(url, params=params, timeout=60)
            last_text = resp.text[:300]

            if resp.status_code == 200:
                payload = resp.json()
                payload_list = payload if isinstance(payload, list) else []
                # write cache even on forced refresh so later runs are fast
                if self.use_http_cache:
                    key = self._cache_key(url, params)
                    self._cache_write(self._cache_path(key), payload_list)
                return payload_list

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

            # Hard failure
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
        self,
        ticker: str,
        statement_limit: int | None = None,
        *,
        force_refresh: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        limit = int(statement_limit) if statement_limit is not None else int(self.statement_limit)
        symbol = normalize_fmp_symbol(ticker)
        common = {"symbol": symbol, "period": "quarter", "limit": limit}

        inc = self._get("income-statement", common, force_refresh=force_refresh)
        bs  = self._get("balance-sheet-statement", common, force_refresh=force_refresh)
        cf  = self._get("cash-flow-statement", common, force_refresh=force_refresh)
        return inc, bs, cf

    def fetch_fundamentals_quarterly(
        self,
        tickers: list[str],
        statement_limit: int | None = None,
        *,
        force_refresh: bool = False,
        progress_every: int = 50,
    ) -> pd.DataFrame:
        """Fetch quarterly fundamentals for tickers.

        force_refresh=True bypasses disk HTTP cache (still throttles / respects API limits).
        """
        rows: list[dict[str, Any]] = []
        errors = 0

        for i, t in enumerate(tickers, start=1):
            if progress_every and (i == 1 or i % progress_every == 0):
                print(f"[FMP] fundamentals: {i}/{len(tickers)} tickers...")

            try:
                inc, bs, cf = self._fetch_quarterly_statements(
                    t, statement_limit=statement_limit, force_refresh=force_refresh
                )

                # Index by period_end ("date" field)
                def index_by_date(lst: list[dict[str, Any]]) -> dict[pd.Timestamp, dict[str, Any]]:
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
                    fcf = operating_cf - capex if pd.notna(operating_cf) and pd.notna(capex) else float("nan")

                    rows.append(
                        dict(
                            ticker=str(t).upper(),
                            period_end=pe,
                            filing_date=filing_date,
                            available_date=available_date,
                            revenue=revenue,
                            net_income=net_income,
                            cogs=cogs,
                            total_assets=total_assets,
                            total_liabilities=total_liabilities,
                            equity=equity,
                            operating_cash_flow=operating_cf,
                            capital_expenditures=capex,
                            free_cash_flow=fcf,
                        )
                    )
            except Exception as e:
                errors += 1
                if errors <= 15:
                    print(f"[FMP] {t}: fundamentals fetch failed: {e}")
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Basic cleanup / ordering
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["period_end"] = pd.to_datetime(df["period_end"]).dt.normalize()
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.normalize()
        df["available_date"] = pd.to_datetime(df["available_date"], errors="coerce").dt.normalize()

        df = df.sort_values(["ticker", "period_end"]).reset_index(drop=True)
        return df
