from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
from typing import Any, Optional

import time
import requests
import pandas as pd

# FMP moved many users off legacy endpoints after Aug 31, 2025. This module uses the
# "Stable API" base URL which remains available to current subscriptions.
FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"


# ---------------------------
# Helpers
# ---------------------------

def _norm_ticker(t: str) -> str:
    return str(t).upper().strip()


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return int(float(x))
    except Exception:
        return None


@dataclass
class FMPFetchResult:
    ticker: str
    fmp_symbol: str
    ok: bool
    status_code: int
    data: Optional[dict[str, Any]]
    error: Optional[str]


# ---------------------------
# FMP API calls (Stable)
# ---------------------------

def _map_to_fmp_symbol(ticker: str) -> str:
    # FMP uses "-" for class shares (e.g., BRK-B) while Yahoo uses "."
    t = _norm_ticker(ticker)
    return t.replace(".", "-")


def _fmp_get_json(
    path: str,
    api_key: str,
    params: dict[str, Any] | None = None,
    timeout: int = 60,
) -> tuple[int, Any, str | None]:
    params = dict(params or {})
    params["apikey"] = api_key
    url = f"{FMP_STABLE_BASE}{path}"
    try:
        resp = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "quant-factor-lab/1.0"},
        )
    except Exception as e:
        return 0, None, f"request_error: {e}"

    status = int(resp.status_code)
    if status >= 400:
        txt = (resp.text or "")[:300]
        return status, None, txt

    try:
        return status, resp.json(), None
    except Exception as e:
        return status, None, f"json_error: {e}"


def fetch_fmp_profile(ticker: str, api_key: str) -> FMPFetchResult:
    """
    Stable API profile:
      GET /stable/profile?symbol=XXX

    Returns the first object in the list (FMP usually responds with a list).
    """
    t = _norm_ticker(ticker)
    sym = _map_to_fmp_symbol(t)
    status, payload, err = _fmp_get_json("/profile", api_key, params={"symbol": sym})
    if err is not None or payload is None:
        return FMPFetchResult(
            ticker=t, fmp_symbol=sym, ok=False, status_code=status or 0, data=None, error=err
        )

    row = None
    if isinstance(payload, list) and payload:
        row = payload[0]
    elif isinstance(payload, dict):
        row = payload

    if not isinstance(row, dict):
        return FMPFetchResult(
            ticker=t, fmp_symbol=sym, ok=False, status_code=status, data=None, error="empty_profile"
        )

    return FMPFetchResult(ticker=t, fmp_symbol=sym, ok=True, status_code=status, data=row, error=None)


def fetch_fmp_quote(ticker: str, api_key: str) -> FMPFetchResult:
    """
    Stable API quote:
      GET /stable/quote?symbol=XXX

    Returns the first object in the list (FMP usually responds with a list).
    """
    t = _norm_ticker(ticker)
    sym = _map_to_fmp_symbol(t)
    status, payload, err = _fmp_get_json("/quote", api_key, params={"symbol": sym})
    if err is not None or payload is None:
        return FMPFetchResult(
            ticker=t, fmp_symbol=sym, ok=False, status_code=status or 0, data=None, error=err
        )

    row = None
    if isinstance(payload, list) and payload:
        row = payload[0]
    elif isinstance(payload, dict):
        row = payload

    if not isinstance(row, dict):
        return FMPFetchResult(
            ticker=t, fmp_symbol=sym, ok=False, status_code=status, data=None, error="empty_quote"
        )

    return FMPFetchResult(ticker=t, fmp_symbol=sym, ok=True, status_code=status, data=row, error=None)


# ---------------------------
# Public: overview cache builder
# ---------------------------

def build_company_overview(
    tickers: list[str],
    fmp_api_key: str,
    existing: pd.DataFrame | None = None,
    asset_type_map: dict[str, str] | None = None,
    chunk_size: int = 100,
    sleep_s: float = 0.15,
    refresh_days: int = 30,
) -> pd.DataFrame:
    """
    Build/update a slow-changing company overview cache for UI and production capping.

    Enriches profile with quote fields commonly used for:
      - production universe cap (marketCap, avgVolume)
      - quick 'company review' cards (price, beta, pe, eps, dividendYield)

    Notes:
      - Only fetches missing tickers by default.
      - Optionally refreshes stale rows older than refresh_days (if last_updated exists).
      - Never crashes the run on per-ticker errors (records error/status_code).
      - asset_type_map (optional) lets you tag STOCK vs ETF (or other).
    """
    tickers_u = pd.Series(tickers, dtype=str).map(_norm_ticker).unique().tolist()

    existing_df = None
    if existing is not None and not existing.empty:
        existing_df = existing.copy()
        if "ticker" in existing_df.columns:
            existing_df["ticker"] = existing_df["ticker"].astype(str).map(_norm_ticker)

    done: set[str] = set()
    stale: set[str] = set()

    if existing_df is not None and "ticker" in existing_df.columns:
        done = set(existing_df["ticker"].tolist())
        if "last_updated" in existing_df.columns:
            lu = pd.to_datetime(existing_df["last_updated"], errors="coerce")
            cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=int(refresh_days))
            stale = set(existing_df.loc[lu < cutoff, "ticker"].tolist())

    missing = [t for t in tickers_u if t not in done]
    need = [t for t in tickers_u if (t not in done) or (t in stale)]
    print(f"[OVERVIEW] need fetch: {len(need)} (missing={len(missing)}, stale={len(stale)})")

    # Normalize asset_type map once
    atm: dict[str, str] = {}
    if asset_type_map:
        atm = {_norm_ticker(k): str(v).upper().strip() for k, v in asset_type_map.items()}

    rows: list[dict[str, Any]] = []
    total_chunks = (len(need) + int(chunk_size) - 1) // int(chunk_size) if need else 0

    for ci, i in enumerate(range(0, len(need), int(chunk_size)), start=1):
        chunk = need[i : i + int(chunk_size)]
        print(f"[OVERVIEW] chunk {ci}/{total_chunks}: {len(chunk)} tickers")

        for t in chunk:
            prof = fetch_fmp_profile(t, fmp_api_key)
            qt = fetch_fmp_quote(t, fmp_api_key)

            profile_row = prof.data if (prof.ok and isinstance(prof.data, dict)) else {}
            quote_row = qt.data if (qt.ok and isinstance(qt.data, dict)) else {}

            # Liquidity/size fields (support multiple possible key spellings)
            mktcap = (
                profile_row.get("mktCap")
                or profile_row.get("marketCap")
                or quote_row.get("marketCap")
                or quote_row.get("mktCap")
            )
            avgv = (
                quote_row.get("avgVolume")
                or quote_row.get("avgVol")
                or quote_row.get("volumeAvg")
                or quote_row.get("volAvg")
            )

            price = quote_row.get("price") or quote_row.get("previousClose")
            beta = quote_row.get("beta")
            pe = quote_row.get("pe") or quote_row.get("peRatio")
            eps = quote_row.get("eps") or quote_row.get("epsTTM")
            dy = quote_row.get("dividendYield")
            exchange = (
                profile_row.get("exchangeShortName")
                or profile_row.get("exchange")
                or quote_row.get("exchange")
            )

            ok_any = bool(prof.ok or qt.ok)
            status_code = int(prof.status_code or qt.status_code or 0)
            err = None if ok_any else (prof.error or qt.error)

            rows.append(
                {
                    "ticker": _norm_ticker(t),
                    "fmp_symbol": _map_to_fmp_symbol(t),
                    "asset_type": atm.get(_norm_ticker(t)),
                    "company_name": profile_row.get("companyName")
                    or profile_row.get("name")
                    or quote_row.get("name"),
                    "sector": profile_row.get("sector"),
                    "industry": profile_row.get("industry"),
                    "exchange": exchange,
                    "country": profile_row.get("country"),
                    "currency": profile_row.get("currency") or quote_row.get("currency"),
                    "website": profile_row.get("website"),
                    "employees": _safe_int(profile_row.get("fullTimeEmployees")),
                    # ---- Liquidity / size (for capping) ----
                    "marketCap": _safe_float(mktcap),
                    "avgVolume": _safe_float(avgv),
                    # Back-compat snake_case (older scripts used market_cap / avg_volume)
                    "market_cap": _safe_float(mktcap),
                    "avg_volume": _safe_float(avgv),
                    # ---- Quick review fields ----
                    "price": _safe_float(price),
                    "beta": _safe_float(beta),
                    "pe": _safe_float(pe),
                    "eps": _safe_float(eps),
                    "dividendYield": _safe_float(dy),
                    "description": profile_row.get("description"),
                    "ceo": profile_row.get("ceo"),
                    # status
                    "error": err,
                    "status_code": status_code,
                    "last_updated": pd.Timestamp.utcnow(),
                    "source": "fmp_stable",
                }
            )

            time.sleep(float(sleep_s))

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker"])

    if existing_df is None or existing_df.empty:
        merged = new_df
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    merged = merged.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)

    # Ensure expected columns exist even if empty (prevents KeyError downstream)
    for col in ["marketCap", "avgVolume", "market_cap", "avg_volume", "asset_type"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged


# -------------------------------
# Backwards-compatible wrapper
# -------------------------------
def build_company_overview_cache(
    tickers: list[str],
    data_dir: str | Path,
    fmp_api_key: str,
    asset_type_map: dict[str, str] | None = None,
    chunk_size: int = 100,
    sleep_s: float = 0.15,
    refresh_days: int = 30,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Compatibility API used by older runners (e.g. run_bootstrap.py).

    Reads/writes: <data_dir>/curated/company_overview.parquet

    Parameters are intentionally permissive; extra args should be passed by name.
    """
    data_dir = Path(data_dir)
    curated_path = data_dir / "curated" / "company_overview.parquet"
    curated_path.parent.mkdir(parents=True, exist_ok=True)

    existing = None
    if curated_path.exists() and not force_refresh:
        try:
            existing = pd.read_parquet(curated_path)
            # normalize ticker just in case
            if "ticker" in existing.columns:
                existing["ticker"] = existing["ticker"].astype(str)
        except Exception:
            existing = None

    updated = build_company_overview(
        tickers=[str(t) for t in tickers],
        fmp_api_key=fmp_api_key,
        existing=existing,
        asset_type_map=asset_type_map,
        chunk_size=int(chunk_size),
        sleep_s=float(sleep_s),
        refresh_days=int(refresh_days),
    )

    try:
        updated.to_parquet(curated_path, index=False)
    except Exception:
        # don't hard-fail a run if parquet write fails
        pass

    return updated
