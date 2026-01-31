from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
import re
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

from datetime import datetime






@dataclass
class PolygonAdapter:
    """
    Polygon.io market data adapter (daily aggregates) with robust caching.

    Key improvements vs previous version:
    - Uses existing range-keyed parquet cache files:
        {TICKER}__{START}__{END}__adj.parquet
    - If exact cache isn't present, reuses any cached file that COVERS the requested range.
      (e.g. 2015-01-01..2026-01-23 cache can satisfy 2016-01-01..2025-12-31 request)
    - Only calls API if no covering cache exists.
    - Handles 429 rate limits with exponential backoff + retries.
    - Throttles based on requests_per_minute (env: POLYGON_RPM).
    """

    cache_dir: Path
    api_key_env: str = "POLYGON_API_KEY"
    base_url: str = "https://api.polygon.io"
    requests_per_minute: int = 2
    max_retries: int = 10

    def _record_empty_ticker(self, ticker: str) -> None:
        """Append ticker to data/universe/empty_tickers.csv if not already present."""
        universe_dir = self.cache_dir.parents[1] / "universe"  # data/universe
        universe_dir.mkdir(parents=True, exist_ok=True)
        path = universe_dir / "empty_tickers.csv"

        t = str(ticker).strip().upper()
        existing: set[str] = set()

        if os.path.exists(path):
            try:
                df0 = pd.read_csv(path)
                if "ticker" in df0.columns:
                    existing = set(df0["ticker"].astype(str).str.strip().str.upper())
            except Exception:
                # If the file is malformed, we treat as empty and keep going.
                existing = set()

        if t and t not in existing:
            pd.DataFrame({"ticker": [t]}).to_csv(
                path,
                mode="a" if os.path.exists(path) else "w",
                header=not os.path.exists(path),
                index=False,
            )

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        rpm_env = os.getenv("POLYGON_RPM")
        if rpm_env:
            try:
                self.requests_per_minute = max(1, int(rpm_env))
            except ValueError:
                pass

        retries_env = os.getenv("POLYGON_MAX_RETRIES")
        if retries_env:
            try:
                self.max_retries = max(1, int(retries_env))
            except ValueError:
                pass

        self._min_interval_s = 60.0 / float(self.requests_per_minute) if self.requests_per_minute > 0 else 0.0
        self._last_call_ts = 0.0
        self._session = requests.Session()

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key.strip():
            raise RuntimeError(f"Missing {self.api_key_env}. Set it as an environment variable.")
        return key.strip()

    def _rate_limit_wait(self) -> None:
        if self._min_interval_s <= 0:
            return
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with retry/backoff on 429."""
        url = self.base_url.rstrip("/") + path
        params = dict(params)
        params["apiKey"] = self.api_key

        last_text: Optional[str] = None

        for attempt in range(self.max_retries):
            self._rate_limit_wait()

            resp = self._session.get(url, params=params, timeout=60)
            self._last_call_ts = time.time()
            last_text = resp.text

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                wait = min(60, (2 ** attempt) * 2)
                print(f"[Polygon] 429 rate limit. Backing off {wait}s (attempt {attempt+1}/{self.max_retries})")
                time.sleep(wait)
                continue

            raise RuntimeError(f"Polygon API error {resp.status_code}: {resp.text[:300]}")

        raise RuntimeError(f"Polygon API retry exhausted. Last response: {(last_text or '')[:300]}")

    def _cache_suffix(self, adjusted: bool) -> str:
        return "__adj.parquet" if adjusted else ".parquet"

    def _exact_cache_path(self, ticker: str, start_s: str, end_s: str, adjusted: bool) -> Path:
        # Matches your existing convention: ZS__2015-01-01__2026-01-23__adj.parquet
        return self.cache_dir / f"{ticker}__{start_s}__{end_s}{self._cache_suffix(adjusted)}"

    def _find_covering_cache(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjusted: bool,
    ) -> Optional[Path]:
        """Find the tightest cached file that covers [start, end]."""
        suffix = self._cache_suffix(adjusted)
        pat = re.compile(rf"^{re.escape(ticker)}__(\d{{4}}-\d{{2}}-\d{{2}})__(\d{{4}}-\d{{2}}-\d{{2}}){re.escape(suffix)}$")

        best: Optional[tuple[int, Path]] = None
        for p in self.cache_dir.glob(f"{ticker}__*{suffix}"):
            m = pat.match(p.name)
            if not m:
                continue
            c_start = pd.to_datetime(m.group(1)).normalize()
            c_end = pd.to_datetime(m.group(2)).normalize()
            if c_start <= start.normalize() and c_end >= end.normalize():
                span = int((c_end - c_start).days)
                if best is None or span < best[0]:
                    best = (span, p)

        return None if best is None else best[1]

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = df["ticker"].astype(str)
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0).astype(float)
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def _fetch_one_ticker_range(
        self,
        ticker: str,
        start_s: str,
        end_s: str,
        adjusted: bool,
    ) -> pd.DataFrame:
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_s}/{end_s}"
        payload = self._get(
            path,
            params={
                "adjusted": "true" if adjusted else "false",
                "sort": "asc",
                "limit": 50000,
            },
        )

        results = payload.get("results") or []
        rows: list[dict[str, Any]] = []

        for r in results:
            dt = pd.to_datetime(r.get("t"), unit="ms").normalize()
            o = r.get("o")
            h = r.get("h")
            l = r.get("l")
            c = r.get("c")
            v = r.get("v")

            if c is None:
                continue

            rows.append(
                {
                    "ticker": str(ticker),
                    "date": dt,
                    "open": float(o) if o is not None else float("nan"),
                    "high": float(h) if h is not None else float("nan"),
                    "low": float(l) if l is not None else float("nan"),
                    "close": float(c),
                    # When adjusted=true, Polygon returns adjusted prices for OHLC.
                    "adj_close": float(c),
                    "volume": float(v) if v is not None else 0.0,
                }
            )

        if not rows:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])

        return self._normalize_df(pd.DataFrame(rows))

    def fetch_prices_daily(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        adjusted: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV for tickers, using cache whenever possible.

        Cache behavior:
        - Prefer exact cache file for (ticker,start,end,adjusted)
        - If missing, reuse any cached file that covers requested range and slice
        - Only call API if no covering cache exists
        - When API is called, save exact-range cache file
        """
        start = pd.to_datetime(start).normalize()
        end = pd.to_datetime(end).normalize()
        # Clamp end date to today to avoid querying future ranges (Polygon returns empty / errors).
        today = pd.Timestamp.today().normalize()
        end_was_future = bool(end > today)
        if end_was_future:
            print(f"[Polygon] End date {end.date()} is in the future; clamping to today {today.date()}.")
            end = today
        if start > end:
            raise RuntimeError(f"Requested price range is entirely in the future: {start.date()}..{end.date()}")
        range_days = int((end - start).days)

        start_s = start.strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")

        frames: list[pd.DataFrame] = []

        for t in tickers:
            # 1) exact cache
            exact = self._exact_cache_path(t, start_s, end_s, adjusted)
            use_file = exact

            # 2) covering cache (superset)
            if use_cache and not use_file.exists():
                covering = self._find_covering_cache(t, start, end, adjusted)
                if covering is not None:
                    use_file = covering

            # 3) read cache if found
            if use_cache and use_file.exists():
                df = pd.read_parquet(use_file)
                df = self._normalize_df(df)
                m = (df["date"] >= start) & (df["date"] <= end)
                df = df.loc[m].reset_index(drop=True)
                if not df.empty:
                    frames.append(df)
                continue

            # 4) otherwise call API and save exact cache
            df_new = self._fetch_one_ticker_range(t, start_s, end_s, adjusted)

            if df_new.empty:
                print(f"[Polygon] {t}: empty response for {start_s}..{end_s}")
                # Avoid falsely flagging 'empty' tickers when the requested range includes future dates
                # or when the range is very short (new listings, holidays, etc.).
                if (not end_was_future) and (range_days >= 20):
                    self._record_empty_ticker(t)
                else:
                    if end_was_future:
                        print(f"[Polygon] {t}: skipped empty_tickers.csv because end date was future-clamped.")
                    else:
                        print(f"[Polygon] {t}: skipped empty_tickers.csv because range_days={range_days} < 20.")
                continue
            else:
                print(f"[Polygon] {t}: fetched {len(df_new):,} rows for {start_s}..{end_s}")
            
            if use_cache:
                try:
                    df_new.to_parquet(exact, index=False)
                except Exception as e:
                    print(f"[Polygon] {t}: failed to write cache {exact}: {e}")

            frames.append(df_new)

        if not frames:
            raise RuntimeError("Polygon returned empty price data (no cached files matched and API returned no rows).")

        out = pd.concat(frames, ignore_index=True)
        out = self._normalize_df(out)
        out = out.dropna(subset=["adj_close"])
        return out