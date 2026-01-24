# Canonical universe ticker → per-provider symbol

TICKER_ALIASES = {
    "BRK.B": {
        "yahoo": "BRK-B",
        "polygon": "BRK.B",
        "fmp": "BRK.B",
    },
}

def map_ticker(ticker: str, source: str) -> str:
    """
    Map canonical ticker to provider-specific ticker.
    source: 'yahoo' | 'polygon' | 'fmp'
    """
    return TICKER_ALIASES.get(ticker, {}).get(source, ticker)
