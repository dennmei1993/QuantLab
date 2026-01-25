from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


def normalize_fmp_symbol(ticker: str) -> str:
    """
    Normalize tickers for FMP endpoints.

    Strategy:
      1) Try dash-class for dot tickers: BRK.B -> BRK-B
      2) Special-case known class shares
    """
    t = ticker.strip().upper()
    special = {
        "BRK.B": "BRK-B",
        "BRK.A": "BRK-A",
        "BF.B": "BF-B",
        "BF.A": "BF-A",
    }
    if t in special:
        return special[t]
    if "." in t:
        return t.replace(".", "-")
    return t


@dataclass
class FMPProfileResult:
    ticker: str          # your canonical ticker (e.g., BRK.B)
    fmp_symbol: str      # symbol used for API call (e.g., BRK-B)
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    status_code: int | None = None


def _request(url: str, params: dict[str, Any], timeout: int = 30) -> requests.Response:
    # Do NOT print the URL (apikey in query string)
    return requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "quant-factor-rating/1.0"})


def fetch_fmp_profile(
    ticker: str,
    api_key: str,
    *,
    max_retries: int = 3,
    sleep_s: float = 0.4,
) -> FMPProfileResult:
    """
    Fetch company overview/profile for a ticker using *stable* endpoint.
    Returns a structured result that never raises for expected HTTP issues.
    """
    t = ticker.strip().upper()
    sym = normalize_fmp_symbol(t)

    # Use STABLE endpoint (recommended by FMP docs)
    base_url = "https://financialmodelingprep.com/stable/profile"

    def _attempt(symbol: str) -> FMPProfileResult:
        params = {"symbol": symbol, "apikey": api_key}
        last_err = None
        for k in range(1, max_retries + 1):
            try:
                resp = _request(base_url, params=params, timeout=30)
                code = resp.status_code

                # Rate limit / transient errors: retry
                if code in (429, 500, 502, 503, 504):
                    time.sleep(sleep_s * k)
                    continue

                # 403: invalid/missing API key OR endpoint not allowed for tier
                if code == 403:
                    return FMPProfileResult(
                        ticker=t, fmp_symbol=symbol, ok=False,
                        error="FMP_403_FORBIDDEN (check API key / endpoint tier)",
                        status_code=code
                    )

                # 402: tier limitation
                if code == 402:
                    return FMPProfileResult(
                        ticker=t, fmp_symbol=symbol, ok=False,
                        error="FMP_402_PAYMENT_REQUIRED (plan limitation for this symbol/endpoint)",
                        status_code=code
                    )

                if 400 <= code < 500:
                    return FMPProfileResult(
                        ticker=t, fmp_symbol=symbol, ok=False,
                        error=f"FMP_{code}_CLIENT_ERROR",
                        status_code=code
                    )

                resp.raise_for_status()
                js = resp.json()
                row = js[0] if isinstance(js, list) and js else {}
                return FMPProfileResult(ticker=t, fmp_symbol=symbol, ok=True, data=row, status_code=code)

            except Exception as e:
                last_err = str(e)
                time.sleep(sleep_s * k)

        return FMPProfileResult(
            ticker=t, fmp_symbol=symbol, ok=False,
            error=f"RETRY_EXHAUSTED: {last_err}",
            status_code=None
        )

    res = _attempt(sym)

    # Fallback for BRK.B / BF.B: try A-share if blocked
    if (not res.ok) and res.status_code in (402, 403) and t in ("BRK.B", "BF.B"):
        fallback = {"BRK.B": "BRK-A", "BF.B": "BF-A"}[t]
        res2 = _attempt(fallback)
        if res2.ok:
            return res2

    return res


def build_company_overview_cache(
    tickers: list[str],
    fmp_api_key: str,
    existing: pd.DataFrame | None = None,
    chunk_size: int = 100,
    sleep_s: float = 0.15,
    refresh_days: int = 30,
) -> pd.DataFrame:
    """
    Build/update a slow-changing company overview cache for theme tagging / UI.

    - Only fetches missing tickers by default
    - Optionally refreshes stale rows older than refresh_days (if last_updated exists)
    - Never crashes the run on per-ticker errors (records error/status_code)
    """
    tickers_u = pd.Series(tickers, dtype=str).str.upper().str.strip().unique().tolist()

    existing_df = None
    if existing is not None and not existing.empty:
        existing_df = existing.copy()
        existing_df["ticker"] = existing_df["ticker"].astype(str).str.upper().str.strip()

    done = set()
    stale = set()

    if existing_df is not None and "ticker" in existing_df.columns:
        done = set(existing_df["ticker"].tolist())
        if "last_updated" in existing_df.columns:
            lu = pd.to_datetime(existing_df["last_updated"], errors="coerce")
            cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(refresh_days))
            stale = set(existing_df.loc[lu < cutoff, "ticker"].tolist())

    need = [t for t in tickers_u if (t not in done) or (t in stale)]
    print(f"[OVERVIEW] need fetch: {len(need)} (missing={len([t for t in tickers_u if t not in done])}, stale={len(stale)})")

    rows = []
    for i in range(0, len(need), chunk_size):
        chunk = need[i:i + chunk_size]
        print(f"[OVERVIEW] chunk {i//chunk_size + 1}/{(len(need) + chunk_size - 1)//chunk_size}: {len(chunk)} tickers")
        for t in chunk:
            res = fetch_fmp_profile(t, fmp_api_key)
            if res.ok and res.data is not None:
                row = res.data
                rows.append({
                    "ticker": res.ticker,
                    "fmp_symbol": res.fmp_symbol,
                    "company_name": row.get("companyName") or row.get("name"),
                    "sector": row.get("sector"),
                    "industry": row.get("industry"),
                    "exchange": row.get("exchangeShortName") or row.get("exchange"),
                    "country": row.get("country"),
                    "currency": row.get("currency"),
                    "website": row.get("website"),
                    "employees": row.get("fullTimeEmployees"),
                    "market_cap": row.get("mktCap") or row.get("marketCap"),
                    "description": row.get("description"),
                    "ceo": row.get("ceo"),
                    "error": None,
                    "status_code": res.status_code,
                    "last_updated": pd.Timestamp.utcnow(),
                    "source": "fmp",
                })
            else:
                rows.append({
                    "ticker": res.ticker,
                    "fmp_symbol": res.fmp_symbol,
                    "company_name": None,
                    "sector": None,
                    "industry": None,
                    "exchange": None,
                    "country": None,
                    "currency": None,
                    "website": None,
                    "employees": None,
                    "market_cap": None,
                    "description": None,
                    "ceo": None,
                    "error": res.error,
                    "status_code": res.status_code,
                    "last_updated": pd.Timestamp.utcnow(),
                    "source": "fmp",
                })
            time.sleep(sleep_s)

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker"])
    if existing_df is None or existing_df.empty:
        merged = new_df
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    merged = merged.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return merged
