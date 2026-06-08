"""
indicators.py — Pure-pandas technical indicators.
No pandas-ta or ta-lib dependency needed.
"""

import pandas as pd
import numpy as np


# ─── Basic Indicators ────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line   = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period=20, std_mult=2):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std_mult * sigma, mid, mid - std_mult * sigma   # upper, mid, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c  = df['high'], df['low'], df['close']
    prev_c   = c.shift(1)
    tr       = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def volume_sma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(period).mean()


# ─── Support / Resistance ─────────────────────────────────────────────────────

def find_levels(df: pd.DataFrame, n: int = 5, current_price: float = None) -> dict:
    """
    Find the 3 nearest support levels below and 3 resistance levels above
    current price, using recent swing highs/lows.
    """
    highs = df['high'].values
    lows  = df['low'].values

    def is_swing_high(i):
        if i < n or i >= len(highs) - n:
            return False
        return all(highs[i] >= highs[i - j] for j in range(1, n + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, n + 1))

    def is_swing_low(i):
        if i < n or i >= len(lows) - n:
            return False
        return all(lows[i] <= lows[i - j] for j in range(1, n + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, n + 1))

    res_levels = sorted(set(round(highs[i], 4)
                            for i in range(len(highs)) if is_swing_high(i)))
    sup_levels = sorted(set(round(lows[i], 4)
                            for i in range(len(lows))  if is_swing_low(i)))

    if current_price:
        res_levels = [r for r in res_levels if r > current_price][:3]
        sup_levels = [s for s in reversed(sup_levels) if s < current_price][:3]

    return {"resistance": res_levels, "support": sup_levels}


# ─── Trend Label ─────────────────────────────────────────────────────────────

def trend_label(close: pd.Series) -> str:
    e9   = ema(close, 9).iloc[-1]
    e21  = ema(close, 21).iloc[-1]
    e50  = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]
    price = close.iloc[-1]

    if price > e9 > e21 > e50 > e200:
        return "STRONG BULLISH"
    elif price > e21 > e50:
        return "BULLISH"
    elif price < e9 < e21 < e50 < e200:
        return "STRONG BEARISH"
    elif price < e21 < e50:
        return "BEARISH"
    else:
        return "NEUTRAL / RANGING"


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> dict:
    """
    Detect RSI divergence — bullish or bearish.
    
    Bullish divergence: Price making lower lows but RSI making higher lows
    Bearish divergence: Price making higher highs but RSI making lower highs
    """
    close  = df['close']
    rsi_series = rsi(close)

    # Get last N candles
    prices = close.tail(lookback).values
    rsi_vals = rsi_series.tail(lookback).values

    # Remove NaN
    valid = ~(np.isnan(prices) | np.isnan(rsi_vals))
    prices   = prices[valid]
    rsi_vals = rsi_vals[valid]

    if len(prices) < 6:
        return {"type": "NONE", "strength": "NONE", "signal": "NO DIVERGENCE DETECTED"}

    # Find last 2 swing lows (for bullish divergence)
    def find_swing_lows(arr, n=3):
        lows = []
        for i in range(n, len(arr) - n):
            if all(arr[i] <= arr[i-j] for j in range(1, n+1)) and \
               all(arr[i] <= arr[i+j] for j in range(1, n+1)):
                lows.append((i, arr[i]))
        return lows[-2:] if len(lows) >= 2 else []

    def find_swing_highs(arr, n=3):
        highs = []
        for i in range(n, len(arr) - n):
            if all(arr[i] >= arr[i-j] for j in range(1, n+1)) and \
               all(arr[i] >= arr[i+j] for j in range(1, n+1)):
                highs.append((i, arr[i]))
        return highs[-2:] if len(highs) >= 2 else []

    price_lows  = find_swing_lows(prices)
    price_highs = find_swing_highs(prices)

    # Bullish divergence check
    if len(price_lows) == 2:
        p1_idx, p1_val = price_lows[0]
        p2_idx, p2_val = price_lows[1]
        r1_val = rsi_vals[p1_idx]
        r2_val = rsi_vals[p2_idx]

        # Price lower low but RSI higher low = bullish divergence
        if p2_val < p1_val and r2_val > r1_val:
            diff = round(r2_val - r1_val, 2)
            strength = "STRONG" if diff > 5 else "MODERATE" if diff > 2 else "WEAK"
            return {
                "type"    : "BULLISH",
                "strength": strength,
                "signal"  : f"BULLISH DIVERGENCE ({strength}) — Potential reversal UP. RSI +{diff} vs lower price low."
            }

    # Bearish divergence check
    if len(price_highs) == 2:
        p1_idx, p1_val = price_highs[0]
        p2_idx, p2_val = price_highs[1]
        r1_val = rsi_vals[p1_idx]
        r2_val = rsi_vals[p2_idx]

        # Price higher high but RSI lower high = bearish divergence
        if p2_val > p1_val and r2_val < r1_val:
            diff = round(r1_val - r2_val, 2)
            strength = "STRONG" if diff > 5 else "MODERATE" if diff > 2 else "WEAK"
            return {
                "type"    : "BEARISH",
                "strength": strength,
                "signal"  : f"BEARISH DIVERGENCE ({strength}) — Potential reversal DOWN. RSI -{diff} vs higher price high."
            }

    return {"type": "NONE", "strength": "NONE", "signal": "NO DIVERGENCE DETECTED"}




# ─── Full Summary for One Timeframe ──────────────────────────────────────────

def analyze_timeframe(df: pd.DataFrame, current_price: float) -> dict:
    """Return a dict of all indicator values for the latest candle."""
    close = df['close']

    # EMAs
    e9   = round(ema(close, 9).iloc[-1],   4)
    e21  = round(ema(close, 21).iloc[-1],  4)
    e50  = round(ema(close, 50).iloc[-1],  4)
    e200 = round(ema(close, 200).iloc[-1], 4)

    # RSI
    rsi_val = round(rsi(close).iloc[-1], 2)

    # MACD
    ml, sl, hist = macd(close)
    macd_val  = round(ml.iloc[-1],   4)
    sig_val   = round(sl.iloc[-1],   4)
    hist_val  = round(hist.iloc[-1], 4)

    # Bollinger Bands
    bb_up, bb_mid, bb_low = bollinger_bands(close)
    bb_upper = round(bb_up.iloc[-1],  4)
    bb_lower = round(bb_low.iloc[-1], 4)
    bb_mid_v = round(bb_mid.iloc[-1], 4)

    # ATR
    atr_val = round(atr(df).iloc[-1], 4)

    # Volume
    vol_cur = round(df['volume'].iloc[-1], 2)
    vol_avg = round(volume_sma(df['volume']).iloc[-1], 2)
    vol_ratio = round(vol_cur / vol_avg, 2) if vol_avg else 1.0

    # Last 5 candles (for AI context)
    tail = df.tail(50)[['open', 'high', 'low', 'close', 'volume']].round(2)
    last5 = [
        f"{row.Index.strftime('%m-%d %H:%M')} O:{row.open} H:{row.high} L:{row.low} C:{row.close} V:{round(row.volume)}"
        for row in tail.itertuples()
    ]

    # Support / Resistance
    levels = find_levels(df, current_price=current_price)
    divergence = detect_rsi_divergence(df)
    return {
        "trend"      : trend_label(close),
        "ema"        : {"9": e9, "21": e21, "50": e50, "200": e200},
        "rsi"        : rsi_val,
        "macd"       : {"macd": macd_val, "signal": sig_val, "histogram": hist_val},
        "bollinger"  : {"upper": bb_upper, "mid": bb_mid_v, "lower": bb_lower},
        "atr"        : atr_val,
        "volume"     : {"current": vol_cur, "avg_20": vol_avg, "ratio_vs_avg": vol_ratio},
        "levels"     : levels,
        "last_5_candles": last5,
        "rsi_divergence": divergence,
    }