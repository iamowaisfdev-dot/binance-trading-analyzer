"""
fetcher.py — Fetch OHLCV candle data from Binance public API.
No API key required for market data.
"""
import requests
import pandas as pd
from datetime import datetime
from config import BINANCE_BASE_URL, BINANCE_FUTURES_URL


def normalize_symbol(symbol: str) -> str:
    """Convert 'ETH' or 'eth' → 'ETHUSDT', keep 'ETHUSDT' as-is."""
    s = symbol.strip().upper()
    if not s.endswith("USDT") and not s.endswith("BTC"):
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
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
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
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price"
    resp = requests.get(url, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "price" not in data:
        raise ValueError(f"Symbol '{symbol}' not available on Binance Futures.")
    return float(data["price"])


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


def get_funding_rate(symbol: str) -> dict:
    """Fetch current funding rate for a futures symbol."""
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
    try:
        resp = requests.get(url, params={"symbol": symbol, "limit": 1}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            rate = float(data[0]['fundingRate']) * 100  # Convert to percentage
            return {
                "rate"      : round(rate, 4),
                "sentiment" : "OVERLEVERAGED LONG"  if rate >  0.05 else
                              "OVERLEVERAGED SHORT" if rate < -0.05 else
                              "NEUTRAL"
            }
    except Exception:
        pass
    return {"rate": 0, "sentiment": "NEUTRAL"}

def get_open_interest(symbol: str) -> dict:
    """Fetch current open interest for a futures symbol."""
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/openInterest"
    try:
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        oi = float(data['openInterest'])

        # Get OI history to check trend (last 10 periods)
        hist_url = f"{BINANCE_FUTURES_URL}/futures/data/openInterestHist"
        hist_resp = requests.get(hist_url, params={
            "symbol": symbol, "period": "1h", "limit": 10
        }, timeout=10)
        hist_resp.raise_for_status()
        hist = hist_resp.json()

        if len(hist) >= 2:
            oi_old = float(hist[0]['sumOpenInterest'])
            oi_new = float(hist[-1]['sumOpenInterest'])
            change_pct = round((oi_new - oi_old) / oi_old * 100, 2)
        else:
            change_pct = 0

        trend = "INCREASING" if change_pct > 1 else \
                "DECREASING" if change_pct < -1 else "STABLE"

        return {
            "current"    : round(oi, 2),
            "change_pct" : change_pct,
            "trend"      : trend,
            "signal"     : "BEARISH CONFIRMATION"  if trend == "INCREASING" else
                           "TREND WEAKENING"        if trend == "DECREASING" else
                           "NEUTRAL"
        }
    except Exception:
        pass
    return {"current": 0, "change_pct": 0, "trend": "UNKNOWN", "signal": "NEUTRAL"}