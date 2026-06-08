"""
ai_analyst.py — Send full market context to Claude and get a structured trade decision.
"""

import json
import numpy as np
import anthropic
from google import genai
from config import GEMINI_API_KEY
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
    coin_analysis: dict,
    btc_price: float,
    btc_analysis: dict,
    news_text: str,
    funding_rate: dict = None,
    open_interest: dict = None,
    fear_greed: dict = None,
    btc_dominance: dict = None,
) -> str:

    def fmt(d):
        return json.dumps(d, indent=2, cls=_NumpyEncoder)

    return f"""You are a professional crypto technical analyst. Your job is to analyze market data and decide whether a trade is worth taking.

STRICT RULES:
- Only give a signal when at least 2 out of 3 timeframes (1h, 4h, 1d) agree on direction.
- Entry price MUST be current market price or the predictive price which will hit soon like support or resistance price.
- Stop Loss must be based on ATR — not tighter than 1x ATR from entry.
- Minimum Risk:Reward ratio is 2:1 — reject any setup below this.
- Maximum risk score allowed is 55 — if setup scores higher, say NO TRADE.
- Leverage maximum 10x, only if trend is confirmed on 4h and 1d both.
- In sideways or choppy market (RSI between 45-55 on all timeframes), say NO TRADE.
- BTC bearish: SHORT signals only if coin also shows clear weakness on 4h chart.
- BTC bullish: LONG signals only if coin shows clear strength on 4h chart.
- If volume is below average on 1h and 4h, say NO TRADE — no conviction in move.
- Expected TP time minimum 6 hours — please avoid very short scalps.
- Only give a signal when at least 2 out of 3 timeframes (1h, 4h, 1d) agree on direction.
- RSI Divergence is a strong signal — BULLISH divergence on 4h or 1d = consider LONG even in downtrend.
- BEARISH divergence on 4h or 1d = consider SHORT even in uptrend.
- STRONG divergence overrides volume rule — give signal even if volume is slightly below average.
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

── 1W ANALYSIS (Macro Trend) ──
{fmt(coin_analysis['1w'])}

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

── BTC 1W (Macro Trend) ──
{fmt(btc_analysis['1w'])}

═══════════════════════════════════════
FUNDING RATE:
Rate: {funding_rate.get('rate', 0) if funding_rate else 'N/A'}%
Sentiment: {funding_rate.get('sentiment', 'NEUTRAL') if funding_rate else 'N/A'}

Note: Positive funding = longs paying shorts (market overleveraged long = SHORT opportunity)
Negative funding = shorts paying longs (market overleveraged short = LONG opportunity)
═══════════════════════════════════════
OPEN INTEREST:
Current OI: {open_interest.get('current', 'N/A') if open_interest else 'N/A'}
Change (10h): {open_interest.get('change_pct', 0) if open_interest else 'N/A'}%
Trend: {open_interest.get('trend', 'UNKNOWN') if open_interest else 'N/A'}
Signal: {open_interest.get('signal', 'NEUTRAL') if open_interest else 'N/A'}

Note: Rising OI + falling price = strong downtrend. Rising OI + rising price = strong uptrend.
Falling OI = trend losing strength, possible reversal.
═══════════════════════════════════════
FEAR & GREED INDEX:
Value: {fear_greed.get('value', 50) if fear_greed else 'N/A'}/100
Label: {fear_greed.get('label', 'Neutral') if fear_greed else 'N/A'}
Signal: {fear_greed.get('signal', 'NEUTRAL') if fear_greed else 'N/A'}
Note: 0-25 Extreme Fear = LONG opportunity. 75-100 Extreme Greed = SHORT opportunity.
═══════════════════════════════════
BTC DOMINANCE:
Dominance: {btc_dominance.get('dominance', 50) if btc_dominance else 'N/A'}%
Signal: {btc_dominance.get('signal', 'NEUTRAL') if btc_dominance else 'N/A'}
Note: Dominance >55% = alts weak, prefer BTC trades or SHORT alts.
Dominance <45% = altseason, LONG alts opportunity.
═══════════════════════════════════════
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
    funding_rate: dict = None,
    open_interest: dict = None,
    fear_greed: dict = None,
    btc_dominance: dict = None,
) -> dict:
    """
    Call Claude API with all market data.
    Returns parsed dict with trade decision.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = build_prompt(symbol, current_price, coin_analysis,
                          btc_price, btc_analysis, news_text, funding_rate, open_interest, fear_greed, btc_dominance)

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

def get_trade_signal_gemini(
    symbol: str,
    current_price: float,
    coin_analysis: dict,
    btc_price: float,
    btc_analysis: dict,
    news_text: str,
    funding_rate: dict = None,
    open_interest: dict = None,
    fear_greed: dict = None,
    btc_dominance: dict = None,
) -> dict:
    """Call Google Gemini API with all market data."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY missing. Add it to your .env file.")
    prompt = build_prompt(symbol, current_price, coin_analysis,
                          btc_price, btc_analysis, news_text, funding_rate, open_interest, fear_greed, btc_dominance)
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "trade"           : False,
            "no_trade_reason" : "Gemini response could not be parsed.",
            "analysis_summary": raw[:300],
            "risk_score"      : 99,
            "btc_trend"       : "UNKNOWN",
            "news_sentiment"  : "NEUTRAL",
            "key_reasons"     : [],
        }

    return result