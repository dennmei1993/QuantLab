from __future__ import annotations

from pathlib import Path
import pandas as pd


# --- Ticker hygiene rules ---
def _is_allowed_ticker(t: str) -> bool:
    """
    Exclude dual-class / dot tickers and similar class-format tickers.

    - Exclude: BRK.B, BF.B (dot tickers)
    - Also exclude: BRK-B, BF-B (often dot-normalized forms)
    """
    if not t:
        return False
    t = t.strip().upper()
    if t in {"NAN", "NONE"}:
        return False
    # exclude dot tickers and dash-class tickers
    if "." in t:
        return False
    if "-" in t:
        return False
    # optional: exclude slashes (rare US share-class formats)
    if "/" in t:
        return False
    return True


def _dedup_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _load_universe_csv(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"Universe CSV missing 'ticker' column: {path}")
    tickers = df["ticker"].astype(str).str.strip().str.upper().tolist()
    tickers = [t for t in tickers if _is_allowed_ticker(t)]
    return _dedup_preserve_order(tickers)


def _load_from_wikipedia() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if "ticker" in cols:
            tick_col = t.columns[cols.index("ticker")]
            tickers = t[tick_col].astype(str).str.strip().str.upper().tolist()
            tickers = [x for x in tickers if _is_allowed_ticker(x)]
            return _dedup_preserve_order(tickers)
    raise ValueError("No 'Ticker' column found in Wikipedia tables.")


def _repo_root() -> Path:
    # Your original code used parents[2] :contentReference[oaicite:2]{index=2}
    # Keep it consistent with your repo layout.
    return Path(__file__).resolve().parents[2]


def get_nasdaq100_tickers() -> list[str]:
    """
    Universe source priority (robust, deterministic):
      1) Local CSV: data/universe/nasdaq100.csv  (RECOMMENDED)
      2) Wikipedia scrape (optional fallback)
      3) If neither works: raise error
    """
    root = _repo_root()
    csv_path = root / "data" / "universe" / "nasdaq100.csv"

    # 1) Local CSV first
    if csv_path.exists():
        tickers = _load_universe_csv(csv_path)
        if len(tickers) < 80:
            raise RuntimeError(
                f"Universe CSV seems too small ({len(tickers)}). Check file: {csv_path}"
            )
        return tickers

    # 2) Wikipedia fallback (optional)
    try:
        tickers = _load_from_wikipedia()
        if len(tickers) < 80:
            raise RuntimeError("Wikipedia universe scrape returned too few tickers.")
        return tickers
    except Exception as e:
        raise FileNotFoundError(
            f"Universe CSV not found at {csv_path} and Wikipedia scrape failed: {e}"
        )


def get_sp500_tickers() -> list[str]:
    """
    Requires local CSV: data/universe/sp500.csv with column 'ticker'
    """
    root = _repo_root()
    csv_path = root / "data" / "universe" / "sp500.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Build it first (we can generate from FMP)."
        )
    tickers = _load_universe_csv(csv_path)
    if len(tickers) < 400:
        raise RuntimeError(f"S&P 500 CSV seems too small ({len(tickers)}). Check file: {csv_path}")
    return tickers


def get_us_liquid_tickers() -> list[str]:
    """
    Requires local CSV: data/universe/us_liquid.csv with column 'ticker'
    This is your 'US tradable excluding small/illiquid' list.
    """
    root = _repo_root()
    csv_path = root / "data" / "universe" / "us_liquid.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Build it first (we can generate from FMP screener)."
        )
    tickers = _load_universe_csv(csv_path)
    if len(tickers) < 300:
        raise RuntimeError(f"US liquid CSV seems too small ({len(tickers)}). Check file: {csv_path}")
    return tickers


def get_universe(name: str) -> list[str]:
    """
    Convenience router.
    name: 'nasdaq100' | 'sp500' | 'us_liquid' | 'sp500_plus'
    """
    name = (name or "").strip().lower()
    if name == "nasdaq100":
        return get_nasdaq100_tickers()
    if name == "sp500":
        return get_sp500_tickers()
    if name == "us_liquid":
        return get_us_liquid_tickers()
    if name == "sp500_plus":
        # Ensure NASDAQ100 "stays": union them in
        n100 = get_nasdaq100_tickers()
        spx = get_sp500_tickers()
        extra = get_us_liquid_tickers()
        return _dedup_preserve_order(spx + n100 + extra)

    raise ValueError(f"Unknown universe name: {name}")
