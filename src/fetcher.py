"""
fetcher.py — Fetch OHLCV candle data from Binance public API.
No API key required for market data.
"""

import requests
import pandas as pd
from datetime import datetime
from config import BINANCE_BASE_URL


def normalize_symbol(symbol: str) -> str:
    """Convert 'ETH' or 'eth' → 'ETHUSDT', keep 'ETHUSDT' as-is."""
    s = symbol.strip().upper()
    if not s.endswith("USDT") and not s.endswith("BTC") and not s.endswith("BNB"):
        s = s + "USDT"
    return s


def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch candlestick data from Binance.

    Args:
        symbol   : e.g. 'ETHUSDT'
        interval : '1h', '4h', '1d'
        limit    : number of candles (max 1000)

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 400:
            raise ValueError(f"Symbol '{symbol}' not found on Binance. "
                             f"Check the symbol name (e.g. BTCUSDT, ETHUSDT).")
        raise e

    raw = resp.json()
    df = pd.DataFrame(raw, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df.set_index('timestamp', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def get_current_price(symbol: str) -> float:
    """Get the latest ticker price for a symbol."""
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/price"
    resp = requests.get(url, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["price"])


def fetch_all_timeframes(symbol: str) -> dict:
    """
    Fetch 1h / 4h / 1d data for the last ~30 days.
    Returns dict with keys '1h', '4h', '1d' → DataFrames.
    """
    limits = {"1h": 720, "4h": 180, "1d": 30}
    data = {}
    for interval, limit in limits.items():
        data[interval] = get_klines(symbol, interval, limit)
    return data
