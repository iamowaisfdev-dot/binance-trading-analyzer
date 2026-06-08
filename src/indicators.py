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


def volume_profile(df: pd.DataFrame, bins: int = 10) -> dict:
    """
    Calculate Volume Profile — price levels with highest trading volume.
    High volume nodes = strong support/resistance.
    """
    if len(df) < 10:
        return {"high_volume_nodes": [], "poc": None, "signal": "INSUFFICIENT DATA"}

    price_min = df['low'].min()
    price_max = df['high'].max()
    bin_size  = (price_max - price_min) / bins

    # Build volume buckets
    buckets = {}
    for _, row in df.iterrows():
        bucket = int((row['close'] - price_min) / bin_size)
        bucket = min(bucket, bins - 1)  # cap at last bucket
        buckets[bucket] = buckets.get(bucket, 0) + row['volume']

    if not buckets:
        return {"high_volume_nodes": [], "poc": None, "signal": "NO DATA"}

    # Point of Control (POC) — highest volume price level
    poc_bucket   = max(buckets, key=buckets.get)
    poc_price    = round(price_min + (poc_bucket + 0.5) * bin_size, 4)

    # High Volume Nodes — top 3 buckets
    sorted_buckets = sorted(buckets.items(), key=lambda x: x[1], reverse=True)[:3]
    hvn_prices = [
        round(price_min + (b + 0.5) * bin_size, 4)
        for b, _ in sorted_buckets
    ]

    current_price = df['close'].iloc[-1]

    # Signal based on POC position
    if current_price > poc_price * 1.01:
        signal = f"PRICE ABOVE POC ({poc_price}) — POC acts as support"
    elif current_price < poc_price * 0.99:
        signal = f"PRICE BELOW POC ({poc_price}) — POC acts as resistance"
    else:
        signal = f"PRICE AT POC ({poc_price}) — High liquidity zone, breakout likely"

    return {
        "poc"              : poc_price,
        "high_volume_nodes": hvn_prices,
        "signal"           : signal
    }

def market_structure(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detect market structure — Higher Highs/Lower Lows and Break of Structure.
    
    Bullish structure: Higher Highs + Higher Lows
    Bearish structure: Lower Highs + Lower Lows
    Break of Structure: When price breaks previous swing high/low
    """
    if len(df) < lookback:
        return {"structure": "UNKNOWN", "bos": "NONE", "signal": "INSUFFICIENT DATA"}

    highs  = df['high'].values[-lookback:]
    lows   = df['low'].values[-lookback:]
    closes = df['close'].values[-lookback:]

    # Find swing highs and lows
    def get_swings(arr, mode='high', n=3):
        swings = []
        for i in range(n, len(arr) - n):
            if mode == 'high':
                if all(arr[i] >= arr[i-j] for j in range(1, n+1)) and \
                   all(arr[i] >= arr[i+j] for j in range(1, n+1)):
                    swings.append((i, arr[i]))
            else:
                if all(arr[i] <= arr[i-j] for j in range(1, n+1)) and \
                   all(arr[i] <= arr[i+j] for j in range(1, n+1)):
                    swings.append((i, arr[i]))
        return swings

    swing_highs = get_swings(highs, 'high')
    swing_lows  = get_swings(lows,  'low')

    # Determine structure
    structure = "NEUTRAL"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1][1] > swing_highs[-2][1]   # Higher High
        hl = swing_lows[-1][1]  > swing_lows[-2][1]    # Higher Low
        lh = swing_highs[-1][1] < swing_highs[-2][1]   # Lower High
        ll = swing_lows[-1][1]  < swing_lows[-2][1]    # Lower Low

        if hh and hl:
            structure = "BULLISH (HH + HL)"
        elif lh and ll:
            structure = "BEARISH (LH + LL)"
        elif hh and ll:
            structure = "MIXED — Expanding"
        elif lh and hl:
            structure = "MIXED — Contracting"

    # Break of Structure (BOS)
    bos = "NONE"
    bos_signal = ""
    current_price = closes[-1]

    if len(swing_highs) >= 1 and len(swing_lows) >= 1:
        last_high = swing_highs[-1][1]
        last_low  = swing_lows[-1][1]

        if current_price > last_high:
            bos = "BULLISH BOS"
            bos_signal = f"Price broke above last swing high ({round(last_high, 4)}) — BULLISH structure shift"
        elif current_price < last_low:
            bos = "BEARISH BOS"
            bos_signal = f"Price broke below last swing low ({round(last_low, 4)}) — BEARISH structure shift"

    # Overall signal
    if bos == "BULLISH BOS" or structure == "BULLISH (HH + HL)":
        signal = f"BULLISH STRUCTURE — {bos_signal or structure}"
    elif bos == "BEARISH BOS" or structure == "BEARISH (LH + LL)":
        signal = f"BEARISH STRUCTURE — {bos_signal or structure}"
    else:
        signal = f"NEUTRAL STRUCTURE — {structure}"

    return {
        "structure": structure,
        "bos"      : bos,
        "signal"   : signal,
        "last_swing_high": round(swing_highs[-1][1], 4) if swing_highs else None,
        "last_swing_low" : round(swing_lows[-1][1],  4) if swing_lows  else None,
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """
    Detect high-accuracy candlestick patterns.
    Uses last 3 candles for multi-candle patterns.
    """
    if len(df) < 5:
        return {"patterns": [], "signal": "INSUFFICIENT DATA", "bias": "NEUTRAL"}

    c = df.tail(5)
    o = c['open'].values
    h = c['high'].values
    l = c['low'].values
    cl = c['close'].values

    # Current candle (index -1), previous (-2), two back (-3)
    o1, h1, l1, c1 = o[-1], h[-1], l[-1], cl[-1]   # current
    o2, h2, l2, c2 = o[-2], h[-2], l[-2], cl[-2]   # previous
    o3, h3, l3, c3 = o[-3], h[-3], l[-3], cl[-3]   # two back

    body1  = abs(c1 - o1)
    body2  = abs(c2 - o2)
    body3  = abs(c3 - o3)
    range1 = h1 - l1
    range2 = h2 - l2
    range3 = h3 - l3

    upper_wick1 = h1 - max(o1, c1)
    lower_wick1 = min(o1, c1) - l1
    upper_wick2 = h2 - max(o2, c2)
    lower_wick2 = min(o2, c2) - l2

    bull1 = c1 > o1
    bear1 = c1 < o1
    bull2 = c2 > o2
    bear2 = c2 < o2
    bull3 = c3 > o3
    bear3 = c3 < o3
    patterns = []

    # ── SINGLE CANDLE PATTERNS ──────────────────────────────────────────────

    # Hammer (Bullish Reversal)
    if (lower_wick1 >= body1 * 2 and
        upper_wick1 <= body1 * 0.5 and
        range1 > 0 and body1 > 0):
        patterns.append({"name": "HAMMER", "type": "BULLISH", "strength": "STRONG",
                          "desc": "Long lower wick — buyers rejected sellers strongly"})

    # Inverted Hammer (Bullish Reversal)
    if (upper_wick1 >= body1 * 2 and
        lower_wick1 <= body1 * 0.5 and
        range1 > 0 and body1 > 0 and bull1):
        patterns.append({"name": "INVERTED HAMMER", "type": "BULLISH", "strength": "MODERATE",
                          "desc": "Long upper wick after downtrend — potential reversal"})

    # Shooting Star (Bearish Reversal)
    if (upper_wick1 >= body1 * 2 and
        lower_wick1 <= body1 * 0.5 and
        range1 > 0 and body1 > 0 and bear1):
        patterns.append({"name": "SHOOTING STAR", "type": "BEARISH", "strength": "STRONG",
                          "desc": "Long upper wick — sellers rejected buyers at highs"})

    # Hanging Man (Bearish Reversal)
    if (lower_wick1 >= body1 * 2 and
        upper_wick1 <= body1 * 0.5 and
        range1 > 0 and bear1):
        patterns.append({"name": "HANGING MAN", "type": "BEARISH", "strength": "MODERATE",
                          "desc": "Hammer shape after uptrend — bearish warning"})

    # Doji (Indecision)
    if body1 <= range1 * 0.1 and range1 > 0:
        patterns.append({"name": "DOJI", "type": "NEUTRAL", "strength": "MODERATE",
                          "desc": "Open ≈ Close — market indecision, watch for breakout"})

    # Dragonfly Doji (Bullish)
    if (body1 <= range1 * 0.1 and
        lower_wick1 >= range1 * 0.6 and
        upper_wick1 <= range1 * 0.1):
        patterns.append({"name": "DRAGONFLY DOJI", "type": "BULLISH", "strength": "STRONG",
                          "desc": "Long lower wick doji — strong bullish reversal signal"})

    # Gravestone Doji (Bearish)
    if (body1 <= range1 * 0.1 and
        upper_wick1 >= range1 * 0.6 and
        lower_wick1 <= range1 * 0.1):
        patterns.append({"name": "GRAVESTONE DOJI", "type": "BEARISH", "strength": "STRONG",
                          "desc": "Long upper wick doji — strong bearish reversal signal"})

    # Marubozu Bullish (Strong Bull)
    if (bull1 and
        upper_wick1 <= body1 * 0.05 and
        lower_wick1 <= body1 * 0.05 and
        body1 >= range1 * 0.9):
        patterns.append({"name": "BULLISH MARUBOZU", "type": "BULLISH", "strength": "STRONG",
                          "desc": "Full body candle — bulls in complete control"})

    # Marubozu Bearish (Strong Bear)
    if (bear1 and
        upper_wick1 <= body1 * 0.05 and
        lower_wick1 <= body1 * 0.05 and
        body1 >= range1 * 0.9):
        patterns.append({"name": "BEARISH MARUBOZU", "type": "BEARISH", "strength": "STRONG",
                          "desc": "Full body candle — bears in complete control"})

    # Spinning Top (Indecision)
    if (body1 <= range1 * 0.3 and
        upper_wick1 >= body1 and
        lower_wick1 >= body1 and
        range1 > 0):
        patterns.append({"name": "SPINNING TOP", "type": "NEUTRAL", "strength": "WEAK",
                          "desc": "Small body with equal wicks — indecision"})

    # ── TWO CANDLE PATTERNS ──────────────────────────────────────────────────

    # Bullish Engulfing
    if (bear2 and bull1 and
        o1 <= c2 and c1 >= o2 and
        body1 > body2):
        patterns.append({"name": "BULLISH ENGULFING", "type": "BULLISH", "strength": "STRONG",
                          "desc": "Bull candle engulfs previous bear — strong reversal"})

    # Bearish Engulfing
    if (bull2 and bear1 and
        o1 >= c2 and c1 <= o2 and
        body1 > body2):
        patterns.append({"name": "BEARISH ENGULFING", "type": "BEARISH", "strength": "STRONG",
                          "desc": "Bear candle engulfs previous bull — strong reversal"})

    # Bullish Harami
    if (bear2 and bull1 and
        o1 > c2 and c1 < o2 and
        body1 < body2 * 0.5):
        patterns.append({"name": "BULLISH HARAMI", "type": "BULLISH", "strength": "MODERATE",
                          "desc": "Small bull inside large bear — reversal possible"})

    # Bearish Harami
    if (bull2 and bear1 and
        o1 < c2 and c1 > o2 and
        body1 < body2 * 0.5):
        patterns.append({"name": "BEARISH HARAMI", "type": "BEARISH", "strength": "MODERATE",
                          "desc": "Small bear inside large bull — reversal possible"})

    # Tweezer Bottom (Bullish)
    if (bear2 and bull1 and
        abs(l1 - l2) <= range1 * 0.02):
        patterns.append({"name": "TWEEZER BOTTOM", "type": "BULLISH", "strength": "MODERATE",
                          "desc": "Equal lows — double support rejection"})

    # Tweezer Top (Bearish)
    if (bull2 and bear1 and
        abs(h1 - h2) <= range1 * 0.02):
        patterns.append({"name": "TWEEZER TOP", "type": "BEARISH", "strength": "MODERATE",
                          "desc": "Equal highs — double resistance rejection"})

    # Piercing Line (Bullish)
    if (bear2 and bull1 and
        o1 < l2 and
        c1 > (o2 + c2) / 2 and c1 < o2):
        patterns.append({"name": "PIERCING LINE", "type": "BULLISH", "strength": "STRONG",
                          "desc": "Bull closes above midpoint of previous bear — reversal"})

    # Dark Cloud Cover (Bearish)
    if (bull2 and bear1 and
        o1 > h2 and
        c1 < (o2 + c2) / 2 and c1 > o2):
        patterns.append({"name": "DARK CLOUD COVER", "type": "BEARISH", "strength": "STRONG",
                          "desc": "Bear closes below midpoint of previous bull — reversal"})

    # ── THREE CANDLE PATTERNS ────────────────────────────────────────────────

    # Morning Star (Bullish)
    if (bear3 and
        body2 <= range2 * 0.3 and
        bull1 and
        c1 > (o3 + c3) / 2):
        patterns.append({"name": "MORNING STAR", "type": "BULLISH", "strength": "STRONG",
                          "desc": "Bear + Doji + Bull — classic bottom reversal"})

    # Evening Star (Bearish)
    if (bull3 and
        body2 <= range2 * 0.3 and
        bear1 and
        c1 < (o3 + c3) / 2):
        patterns.append({"name": "EVENING STAR", "type": "BEARISH", "strength": "STRONG",
                          "desc": "Bull + Doji + Bear — classic top reversal"})

    # Three White Soldiers (Bullish)
    if (bull1 and bull2 and bull3 and
        o1 > o2 and o2 > o3 and
        c1 > c2 and c2 > c3 and
        body1 > range1 * 0.6 and
        body2 > range2 * 0.6 and
        body3 > range3 * 0.6):
        patterns.append({"name": "THREE WHITE SOLDIERS", "type": "BULLISH", "strength": "STRONG",
                          "desc": "3 consecutive strong bull candles — powerful uptrend"})

    # Three Black Crows (Bearish)
    if (bear1 and bear2 and not bull3 and
        o1 < o2 and o2 < o3 and
        c1 < c2 and c2 < c3 and
        body1 > range1 * 0.6 and
        body2 > range2 * 0.6 and
        body3 > range3 * 0.6):
        patterns.append({"name": "THREE BLACK CROWS", "type": "BEARISH", "strength": "STRONG",
                          "desc": "3 consecutive strong bear candles — powerful downtrend"})

    # Three Inside Up (Bullish)
    if (bear3 and bull2 and bull1 and
        o2 > c3 and c2 < o3 and
        c1 > o3):
        patterns.append({"name": "THREE INSIDE UP", "type": "BULLISH", "strength": "STRONG",
                          "desc": "Harami followed by confirmation — bullish reversal confirmed"})

    # Three Inside Down (Bearish)
    if (bull3 and bear2 and bear1 and
        o2 < c3 and c2 > o3 and
        c1 < o3):
        patterns.append({"name": "THREE INSIDE DOWN", "type": "BEARISH", "strength": "STRONG",
                          "desc": "Harami followed by confirmation — bearish reversal confirmed"})

    # ── Determine Overall Bias ───────────────────────────────────────────────

    bullish = sum(1 for p in patterns if p['type'] == 'BULLISH')
    bearish = sum(1 for p in patterns if p['type'] == 'BEARISH')

    strong_bull = sum(1 for p in patterns if p['type'] == 'BULLISH' and p['strength'] == 'STRONG')
    strong_bear = sum(1 for p in patterns if p['type'] == 'BEARISH' and p['strength'] == 'STRONG')

    if strong_bull >= 2 or (strong_bull >= 1 and bullish > bearish):
        bias = "STRONG BULLISH"
        signal = f"BULLISH patterns detected: {', '.join(p['name'] for p in patterns if p['type'] == 'BULLISH')}"
    elif strong_bear >= 2 or (strong_bear >= 1 and bearish > bullish):
        bias = "STRONG BEARISH"
        signal = f"BEARISH patterns detected: {', '.join(p['name'] for p in patterns if p['type'] == 'BEARISH')}"
    elif bullish > bearish:
        bias = "BULLISH"
        signal = f"Bullish patterns: {', '.join(p['name'] for p in patterns if p['type'] == 'BULLISH')}"
    elif bearish > bullish:
        bias = "BEARISH"
        signal = f"Bearish patterns: {', '.join(p['name'] for p in patterns if p['type'] == 'BEARISH')}"
    else:
        bias = "NEUTRAL"
        signal = "No clear pattern or mixed signals" if not patterns else \
                 f"Neutral patterns: {', '.join(p['name'] for p in patterns)}"

    return {
        "patterns": [p['name'] for p in patterns],
        "details" : patterns,
        "bias"    : bias,
        "signal"  : signal
    }


def stochastic_rsi(series: pd.Series, rsi_period: int = 14,
                   stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> dict:
    """
    Stochastic RSI — RSI ka RSI.
    K line aur D line — overbought/oversold aur crossovers detect karta hai.
    """
    rsi_series = rsi(series, rsi_period)

    rsi_min = rsi_series.rolling(stoch_period).min()
    rsi_max = rsi_series.rolling(stoch_period).max()
    rsi_range = rsi_max - rsi_min

    stoch_k_raw = ((rsi_series - rsi_min) / rsi_range.replace(0, np.nan)) * 100
    k_line = stoch_k_raw.rolling(smooth_k).mean()
    d_line = k_line.rolling(smooth_d).mean()

    k = round(k_line.iloc[-1], 2) if not np.isnan(k_line.iloc[-1]) else 50.0
    d = round(d_line.iloc[-1], 2) if not np.isnan(d_line.iloc[-1]) else 50.0

    k_prev = k_line.iloc[-2] if len(k_line) > 1 else k
    d_prev = d_line.iloc[-2] if len(d_line) > 1 else d

    # Crossover detection
    bullish_cross = bool(k_prev < d_prev and k > d)
    bearish_cross = bool(k_prev > d_prev and k < d)

    if k < 20 and d < 20:
        zone = "OVERSOLD"
        signal = "OVERSOLD — Strong LONG opportunity"
    elif k > 80 and d > 80:
        zone = "OVERBOUGHT"
        signal = "OVERBOUGHT — Strong SHORT opportunity"
    elif bullish_cross and k < 50:
        zone = "BULLISH CROSS"
        signal = f"BULLISH CROSSOVER at {k} — K crossed above D in low zone"
    elif bearish_cross and k > 50:
        zone = "BEARISH CROSS"
        signal = f"BEARISH CROSSOVER at {k} — K crossed below D in high zone"
    else:
        zone = "NEUTRAL"
        signal = f"Neutral zone — K:{k} D:{d}"

    return {
        "k"     : k,
        "d"     : d,
        "zone"  : zone,
        "signal": signal,
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
    }


def ema_crossover(series: pd.Series) -> dict:
    """
    EMA Crossover signals.
    Golden Cross: EMA50 crosses above EMA200 = strong bullish
    Death Cross: EMA50 crosses below EMA200 = strong bearish
    Short term: EMA9 crosses EMA21
    """
    e9   = ema(series, 9)
    e21  = ema(series, 21)
    e50  = ema(series, 50)
    e200 = ema(series, 200)

    # Current and previous values
    e9_c,   e9_p   = e9.iloc[-1],   e9.iloc[-2]
    e21_c,  e21_p  = e21.iloc[-1],  e21.iloc[-2]
    e50_c,  e50_p  = e50.iloc[-1],  e50.iloc[-2]
    e200_c, e200_p = e200.iloc[-1], e200.iloc[-2]

    signals = []

    # Golden Cross (EMA50 crosses above EMA200)
    if e50_p < e200_p and e50_c > e200_c:
        signals.append({
            "type"    : "GOLDEN CROSS",
            "bias"    : "BULLISH",
            "strength": "STRONG",
            "desc"    : "EMA50 crossed above EMA200 — major bullish signal"
        })

    # Death Cross (EMA50 crosses below EMA200)
    if e50_p > e200_p and e50_c < e200_c:
        signals.append({
            "type"    : "DEATH CROSS",
            "bias"    : "BEARISH",
            "strength": "STRONG",
            "desc"    : "EMA50 crossed below EMA200 — major bearish signal"
        })

    # Bullish Short Term (EMA9 crosses above EMA21)
    if e9_p < e21_p and e9_c > e21_c:
        signals.append({
            "type"    : "BULLISH EMA CROSS",
            "bias"    : "BULLISH",
            "strength": "MODERATE",
            "desc"    : "EMA9 crossed above EMA21 — short term bullish momentum"
        })

    # Bearish Short Term (EMA9 crosses below EMA21)
    if e9_p > e21_p and e9_c < e21_c:
        signals.append({
            "type"    : "BEARISH EMA CROSS",
            "bias"    : "BEARISH",
            "strength": "MODERATE",
            "desc"    : "EMA9 crossed below EMA21 — short term bearish momentum"
        })

    # EMA alignment signal
    if e9_c > e21_c > e50_c > e200_c:
        alignment = "FULL BULLISH ALIGNMENT — All EMAs stacked bullish"
    elif e9_c < e21_c < e50_c < e200_c:
        alignment = "FULL BEARISH ALIGNMENT — All EMAs stacked bearish"
    elif e9_c > e21_c > e50_c:
        alignment = "BULLISH ALIGNMENT (short-mid term)"
    elif e9_c < e21_c < e50_c:
        alignment = "BEARISH ALIGNMENT (short-mid term)"
    else:
        alignment = "MIXED ALIGNMENT — No clear trend"

    crossover_names = [s['type'] for s in signals]

    return {
        "crossovers" : crossover_names,
        "details"    : signals,
        "alignment"  : alignment,
        "signal"     : f"{alignment}" + (f" | {', '.join(crossover_names)}" if crossover_names else "")
    }

def confluence_score(analysis: dict) -> dict:
    """
    Confluence Score — kitne indicators ek direction mein agree kar rahe hain.
    Score: 0-100
    0-30   = STRONG BEARISH
    31-45  = BEARISH
    46-54  = NEUTRAL
    55-70  = BULLISH
    71-100 = STRONG BULLISH
    """
    bull_score = 0
    bear_score = 0
    reasons    = []

    # ── 1. Trend Label (Weight: 15) ──────────────────────────────────────────
    trend = analysis.get("trend", "")
    if "STRONG BULLISH" in trend:
        bull_score += 15
        reasons.append("Trend: STRONG BULLISH (+15)")
    elif "BULLISH" in trend:
        bull_score += 10
        reasons.append("Trend: BULLISH (+10)")
    elif "STRONG BEARISH" in trend:
        bear_score += 15
        reasons.append("Trend: STRONG BEARISH (-15)")
    elif "BEARISH" in trend:
        bear_score += 10
        reasons.append("Trend: BEARISH (-10)")

    # ── 2. EMA Alignment (Weight: 15) ────────────────────────────────────────
    ema_cross = analysis.get("ema_crossover", {})
    alignment = ema_cross.get("alignment", "")
    if "FULL BULLISH" in alignment:
        bull_score += 15
        reasons.append("EMA: Full bullish alignment (+15)")
    elif "BULLISH ALIGNMENT" in alignment:
        bull_score += 8
        reasons.append("EMA: Bullish alignment (+8)")
    elif "FULL BEARISH" in alignment:
        bear_score += 15
        reasons.append("EMA: Full bearish alignment (-15)")
    elif "BEARISH ALIGNMENT" in alignment:
        bear_score += 8
        reasons.append("EMA: Bearish alignment (-8)")

    crossovers = ema_cross.get("crossovers", [])
    if "GOLDEN CROSS" in crossovers:
        bull_score += 10
        reasons.append("EMA: Golden Cross (+10)")
    if "DEATH CROSS" in crossovers:
        bear_score += 10
        reasons.append("EMA: Death Cross (-10)")
    if "BULLISH EMA CROSS" in crossovers:
        bull_score += 5
        reasons.append("EMA: Bullish short-term cross (+5)")
    if "BEARISH EMA CROSS" in crossovers:
        bear_score += 5
        reasons.append("EMA: Bearish short-term cross (-5)")

    # ── 3. MACD (Weight: 10) ─────────────────────────────────────────────────
    macd_data = analysis.get("macd", {})
    hist = macd_data.get("histogram", 0)
    macd_line = macd_data.get("macd", 0)
    if hist > 0 and macd_line > 0:
        bull_score += 10
        reasons.append("MACD: Bullish histogram (+10)")
    elif hist > 0:
        bull_score += 5
        reasons.append("MACD: Histogram positive (+5)")
    elif hist < 0 and macd_line < 0:
        bear_score += 10
        reasons.append("MACD: Bearish histogram (-10)")
    elif hist < 0:
        bear_score += 5
        reasons.append("MACD: Histogram negative (-5)")

    # ── 4. RSI (Weight: 10) ──────────────────────────────────────────────────
    rsi_val = analysis.get("rsi", 50)
    if rsi_val < 30:
        bull_score += 10
        reasons.append(f"RSI: Oversold {rsi_val} (+10)")
    elif rsi_val < 45:
        bull_score += 5
        reasons.append(f"RSI: Low {rsi_val} (+5)")
    elif rsi_val > 70:
        bear_score += 10
        reasons.append(f"RSI: Overbought {rsi_val} (-10)")
    elif rsi_val > 55:
        bear_score += 5
        reasons.append(f"RSI: High {rsi_val} (-5)")

    # ── 5. Stochastic RSI (Weight: 10) ───────────────────────────────────────
    stoch = analysis.get("stochastic_rsi", {})
    zone  = stoch.get("zone", "NEUTRAL")
    if zone == "OVERSOLD":
        bull_score += 10
        reasons.append("StochRSI: Oversold (+10)")
    elif zone == "BULLISH CROSS":
        bull_score += 7
        reasons.append("StochRSI: Bullish cross (+7)")
    elif zone == "OVERBOUGHT":
        bear_score += 10
        reasons.append("StochRSI: Overbought (-10)")
    elif zone == "BEARISH CROSS":
        bear_score += 7
        reasons.append("StochRSI: Bearish cross (-7)")

    # ── 6. RSI Divergence (Weight: 10) ───────────────────────────────────────
    div = analysis.get("rsi_divergence", {})
    div_type     = div.get("type", "NONE")
    div_strength = div.get("strength", "NONE")
    if div_type == "BULLISH":
        pts = 10 if div_strength == "STRONG" else 7 if div_strength == "MODERATE" else 4
        bull_score += pts
        reasons.append(f"RSI Divergence: BULLISH {div_strength} (+{pts})")
    elif div_type == "BEARISH":
        pts = 10 if div_strength == "STRONG" else 7 if div_strength == "MODERATE" else 4
        bear_score += pts
        reasons.append(f"RSI Divergence: BEARISH {div_strength} (-{pts})")

    # ── 7. Market Structure (Weight: 10) ─────────────────────────────────────
    ms  = analysis.get("market_structure", {})
    bos = ms.get("bos", "NONE")
    struct = ms.get("structure", "NEUTRAL")
    if bos == "BULLISH BOS":
        bull_score += 10
        reasons.append("Structure: Bullish BOS (+10)")
    elif "BULLISH" in struct:
        bull_score += 6
        reasons.append("Structure: Bullish HH+HL (+6)")
    elif bos == "BEARISH BOS":
        bear_score += 10
        reasons.append("Structure: Bearish BOS (-10)")
    elif "BEARISH" in struct:
        bear_score += 6
        reasons.append("Structure: Bearish LH+LL (-6)")

    # ── 8. Candlestick Patterns (Weight: 8) ──────────────────────────────────
    cp   = analysis.get("candlestick_patterns", {})
    bias = cp.get("bias", "NEUTRAL")
    if bias == "STRONG BULLISH":
        bull_score += 8
        reasons.append("Candles: Strong bullish patterns (+8)")
    elif bias == "BULLISH":
        bull_score += 5
        reasons.append("Candles: Bullish patterns (+5)")
    elif bias == "STRONG BEARISH":
        bear_score += 8
        reasons.append("Candles: Strong bearish patterns (-8)")
    elif bias == "BEARISH":
        bear_score += 5
        reasons.append("Candles: Bearish patterns (-5)")

    # ── 9. Volume (Weight: 6) ────────────────────────────────────────────────
    vol = analysis.get("volume", {})
    vol_ratio = vol.get("ratio_vs_avg", 1.0)
    if vol_ratio >= 1.5:
        bull_score += 3
        bear_score += 3
        reasons.append(f"Volume: High conviction {vol_ratio}x (+3 both)")
    elif vol_ratio < 0.7:
        bear_score += 4
        reasons.append(f"Volume: Low conviction {vol_ratio}x (-4)")

    # ── 10. Volume Profile POC (Weight: 6) ───────────────────────────────────
    vp  = analysis.get("volume_profile", {})
    vp_signal = vp.get("signal", "")
    if "ABOVE POC" in vp_signal:
        bull_score += 6
        reasons.append("Vol Profile: Price above POC (+6)")
    elif "BELOW POC" in vp_signal:
        bear_score += 6
        reasons.append("Vol Profile: Price below POC (-6)")
    elif "AT POC" in vp_signal:
        bull_score += 2
        bear_score += 2
        reasons.append("Vol Profile: Price at POC (+2 both)")

    # ── Final Score ───────────────────────────────────────────────────────────
    total      = bull_score + bear_score
    net_bull   = bull_score - bear_score
    # Normalize to 0-100 (50 = neutral)
    if total == 0:
        score = 50
    else:
        score = round(50 + (net_bull / max(total, 30)) * 50)
        score = max(0, min(100, score))

    if score >= 71:
        label     = "STRONG BULLISH"
        direction = "LONG"
    elif score >= 55:
        label     = "BULLISH"
        direction = "LONG"
    elif score <= 29:
        label     = "STRONG BEARISH"
        direction = "SHORT"
    elif score <= 45:
        label     = "BEARISH"
        direction = "SHORT"
    else:
        label     = "NEUTRAL"
        direction = "WAIT"

    return {
        "score"    : score,
        "label"    : label,
        "direction": direction,
        "bull_pts" : bull_score,
        "bear_pts" : bear_score,
        "reasons"  : reasons[:5],  # Top 5 reasons
        "signal"   : f"CONFLUENCE {score}/100 — {label} | Suggested: {direction}"
    }

def vwap(df: pd.DataFrame) -> dict:
    """
    VWAP — Volume Weighted Average Price.
    Institutions is level pe buy/sell karte hain.
    Price above VWAP = bullish, below = bearish.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cum_vol       = df['volume'].cumsum()
    cum_tp_vol    = (typical_price * df['volume']).cumsum()

    vwap_series   = cum_tp_vol / cum_vol
    vwap_val      = round(vwap_series.iloc[-1], 4)
    current_price = df['close'].iloc[-1]

    distance_pct  = round((current_price - vwap_val) / vwap_val * 100, 2)

    if current_price > vwap_val * 1.02:
        signal = f"PRICE ABOVE VWAP ({vwap_val}) — Bullish bias, VWAP is support"
        bias   = "BULLISH"
    elif current_price < vwap_val * 0.98:
        signal = f"PRICE BELOW VWAP ({vwap_val}) — Bearish bias, VWAP is resistance"
        bias   = "BEARISH"
    else:
        signal = f"PRICE AT VWAP ({vwap_val}) — At equilibrium, watch for breakout"
        bias   = "NEUTRAL"

    return {
        "vwap"        : vwap_val,
        "distance_pct": distance_pct,
        "bias"        : bias,
        "signal"      : signal
    }


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict:
    """
    Supertrend Indicator — ATR based trend following.
    Very popular in crypto — clear BUY/SELL signals.
    """
    hl2    = (df['high'] + df['low']) / 2
    atr_s  = atr(df, period)

    upper_band = hl2 + multiplier * atr_s
    lower_band = hl2 - multiplier * atr_s

    supertrend_vals = pd.Series(index=df.index, dtype=float)
    direction_vals  = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i < period:
            supertrend_vals.iloc[i] = upper_band.iloc[i]
            direction_vals.iloc[i]  = -1
            continue

        prev_st  = supertrend_vals.iloc[i - 1]
        prev_dir = direction_vals.iloc[i - 1]
        close    = df['close'].iloc[i]

        # Calculate final upper and lower bands
        final_upper = upper_band.iloc[i]
        final_lower = lower_band.iloc[i]

        if lower_band.iloc[i] > lower_band.iloc[i - 1] or df['close'].iloc[i - 1] < prev_st:
            pass
        else:
            final_lower = lower_band.iloc[i - 1]

        if upper_band.iloc[i] < upper_band.iloc[i - 1] or df['close'].iloc[i - 1] > prev_st:
            pass
        else:
            final_upper = upper_band.iloc[i - 1]

        if prev_dir == -1 and close < final_lower:
            direction_vals.iloc[i]  = 1
            supertrend_vals.iloc[i] = final_upper
        elif prev_dir == 1 and close > final_upper:
            direction_vals.iloc[i]  = -1
            supertrend_vals.iloc[i] = final_lower
        elif prev_dir == -1:
            direction_vals.iloc[i]  = -1
            supertrend_vals.iloc[i] = final_lower
        else:
            direction_vals.iloc[i]  = 1
            supertrend_vals.iloc[i] = final_upper

    st_val    = round(supertrend_vals.iloc[-1], 4)
    direction = direction_vals.iloc[-1]
    prev_dir  = direction_vals.iloc[-2] if len(direction_vals) > 1 else direction

    # Signal
    if direction == -1 and prev_dir == 1:
        signal = f"SUPERTREND BUY SIGNAL — Flipped bullish at {st_val}"
        bias   = "STRONG BULLISH"
    elif direction == 1 and prev_dir == -1:
        signal = f"SUPERTREND SELL SIGNAL — Flipped bearish at {st_val}"
        bias   = "STRONG BEARISH"
    elif direction == -1:
        signal = f"SUPERTREND BULLISH — Price above {st_val}"
        bias   = "BULLISH"
    else:
        signal = f"SUPERTREND BEARISH — Price below {st_val}"
        bias   = "BEARISH"

    return {
        "value"    : st_val,
        "direction": "BULLISH" if direction == -1 else "BEARISH",
        "bias"     : bias,
        "signal"   : signal,
        "flipped"  : bool(direction != prev_dir)
    }

def adx(df: pd.DataFrame, period: int = 14) -> dict:
    """
    ADX — Average Directional Index.
    Trend strength measure karta hai (not direction).
    ADX > 25 = Strong trend
    ADX > 40 = Very strong trend
    ADX < 20 = Sideways / No trend
    +DI > -DI = Bullish direction
    -DI > +DI = Bearish direction
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move,   0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)

    # Smooth with EWM
    atr_s     = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean()  / atr_s
    minus_di  = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr_s

    # DX and ADX
    di_sum  = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx      = (di_diff / di_sum.replace(0, np.nan)) * 100
    adx_series = dx.ewm(com=period - 1, adjust=False).mean()

    adx_val  = round(adx_series.iloc[-1], 2)
    plus_val = round(plus_di.iloc[-1],    2)
    minus_val= round(minus_di.iloc[-1],   2)

    # Trend strength label
    if adx_val >= 40:
        strength = "VERY STRONG TREND"
    elif adx_val >= 25:
        strength = "STRONG TREND"
    elif adx_val >= 20:
        strength = "DEVELOPING TREND"
    else:
        strength = "WEAK / SIDEWAYS"

    # Direction
    if plus_val > minus_val:
        direction = "BULLISH"
        signal = f"ADX {adx_val} ({strength}) — +DI {plus_val} > -DI {minus_val} — Bullish direction"
    else:
        direction = "BEARISH"
        signal = f"ADX {adx_val} ({strength}) — -DI {minus_val} > +DI {plus_val} — Bearish direction"

    # Trade recommendation
    if adx_val >= 25 and plus_val > minus_val:
        trade_signal = "STRONG BULLISH TREND — Good for LONG"
    elif adx_val >= 25 and minus_val > plus_val:
        trade_signal = "STRONG BEARISH TREND — Good for SHORT"
    elif adx_val < 20:
        trade_signal = "SIDEWAYS MARKET — Avoid trend trades"
    else:
        trade_signal = "WEAK TREND — Wait for confirmation"

    return {
        "adx"         : adx_val,
        "plus_di"     : plus_val,
        "minus_di"    : minus_val,
        "strength"    : strength,
        "direction"   : direction,
        "signal"      : signal,
        "trade_signal": trade_signal
    }

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
    vol_profile = volume_profile(df)
    mkt_structure = market_structure(df)
    candle_patterns = detect_candlestick_patterns(df)
    stoch_rsi     = stochastic_rsi(close)
    ema_cross     = ema_crossover(close)
    conf_score = confluence_score({
        "trend"              : trend_label(close),
        "ema_crossover"      : ema_cross,
        "macd"               : {"macd": macd_val, "signal": sig_val, "histogram": hist_val},
        "rsi"                : rsi_val,
        "stochastic_rsi"     : stoch_rsi,
        "rsi_divergence"     : divergence,
        "market_structure"   : mkt_structure,
        "candlestick_patterns": candle_patterns,
        "volume"             : {"ratio_vs_avg": vol_ratio},
        "volume_profile"     : vol_profile,
    })
    vwap_data      = vwap(df)
    supertrend_data = supertrend(df)
    adx_data = adx(df)
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
        "volume_profile" : vol_profile,
        "market_structure": mkt_structure,
        "candlestick_patterns": candle_patterns,
        "stochastic_rsi"      : stoch_rsi,
        "ema_crossover"       : ema_cross,
        "confluence_score"    : conf_score,
        "vwap"             : vwap_data,
        "supertrend"       : supertrend_data,
        "adx"              : adx_data,
    }