"""
ai_analyst.py — Send full market context to Claude and get a structured trade decision.
"""

import json
import numpy as np
import anthropic
from config import ANTHROPIC_API_KEY


class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy floats/ints to plain Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def build_prompt(
    symbol: str,
    current_price: float,
    coin_analysis: dict,   # {"1h": {...}, "4h": {...}, "1d": {...}}
    btc_price: float,
    btc_analysis: dict,
    news_text: str,
) -> str:

    def fmt(d):
        return json.dumps(d, indent=2, cls=_NumpyEncoder)

    return f"""You are a professional crypto technical analyst. Your job is to analyze market data and decide whether a trade is worth taking.

STRICT RULES:
- DO NOT force a trade signal. If market conditions are unclear, say NO TRADE.
- If BTC is in a strong downtrend, be very cautious about LONG signals on any coin.
- Leverage should be conservative: max 5x for clear trends, 1-2x for uncertain conditions.
- Risk score: 0 = minimal risk (very clear setup), 100 = extremely risky (avoid).
- Entry should be at current price or a nearby key level.
- Stop Loss must be below a clear support (for LONG) or above resistance (for SHORT).
- Target must have at least 1.5:1 reward/risk ratio.

═══════════════════════════════════
COIN: {symbol}
Current Price: {current_price}
═══════════════════════════════════

── 1H ANALYSIS ──
{fmt(coin_analysis['1h'])}

── 4H ANALYSIS ──
{fmt(coin_analysis['4h'])}

── 1D ANALYSIS ──
{fmt(coin_analysis['1d'])}

═══════════════════════════════════
BTC (Market Context)
BTC Price: {btc_price}
═══════════════════════════════════

── BTC 1H ──
{fmt(btc_analysis['1h'])}

── BTC 4H ──
{fmt(btc_analysis['4h'])}

── BTC 1D ──
{fmt(btc_analysis['1d'])}

═══════════════════════════════════
RECENT NEWS:
{news_text}
═══════════════════════════════════

Based on ALL the above data, provide your analysis.

Respond ONLY with valid JSON, no markdown, no explanation outside JSON:
{{
  "trade": true or false,
  "direction": "LONG" or "SHORT" or null,
  "entry_price": number or null,
  "target_price": number or null,
  "stop_loss": number or null,
  "leverage": number or null,
  "risk_score": number (0-100),
  "btc_trend": "BULLISH" or "BEARISH" or "NEUTRAL",
  "news_sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL",
  "key_reasons": ["reason 1", "reason 2", "reason 3"],
  "expected_tp_hours": number or null,
  "no_trade_reason": "explanation if trade is false, else null",
  "analysis_summary": "2-3 sentence overall market read"
}}"""


def get_trade_signal(
    symbol: str,
    current_price: float,
    coin_analysis: dict,
    btc_price: float,
    btc_analysis: dict,
    news_text: str,
) -> dict:
    """
    Call Claude API with all market data.
    Returns parsed dict with trade decision.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = build_prompt(symbol, current_price, coin_analysis,
                          btc_price, btc_analysis, news_text)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return a NO TRADE with raw text
        result = {
            "trade"           : False,
            "no_trade_reason" : "AI response could not be parsed.",
            "analysis_summary": raw[:300],
            "risk_score"      : 99,
            "btc_trend"       : "UNKNOWN",
            "news_sentiment"  : "NEUTRAL",
            "key_reasons"     : [],
        }

    return result
