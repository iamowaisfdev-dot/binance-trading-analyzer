"""
monitor.py — Smart 24/7 Monitor
- Every 5 minutes: pure logic scan (no AI)
- Signal found: WhatsApp immediately + AI confirmation (max 10/day)
- No signal: WhatsApp with reason why
- Run separately from scheduler.py

Usage:
    python monitor.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import json
from datetime import datetime, date
import pytz

from src.fetcher    import normalize_symbol, fetch_all_timeframes, get_current_price
from src.fetcher    import get_funding_rate, get_open_interest, get_btc_dominance
from src.indicators import analyze_timeframe
from src.notifier   import send_whatsapp, format_signal_message
from src.ai_analyst import get_trade_signal, get_trade_signal_gemini
from config         import ANTHROPIC_API_KEY, GEMINI_API_KEY, CALLMEBOT_PHONE

PKT                   = pytz.timezone("Asia/Karachi")
SCAN_INTERVAL_SECONDS = 300        # 5 minutes
MAX_AI_CALLS_PER_DAY  = 10
AI_CALLS_FILE         = "ai_calls_today.json"


# ── AI Call Counter ────────────────────────────────────────────────────────────

def get_ai_calls_today() -> int:
    try:
        with open(AI_CALLS_FILE, "r") as f:
            data = json.load(f)
            if data.get("date") == str(date.today()):
                return data.get("count", 0)
    except Exception:
        pass
    return 0


def increment_ai_calls():
    count = get_ai_calls_today() + 1
    with open(AI_CALLS_FILE, "w") as f:
        json.dump({"date": str(date.today()), "count": count}, f)
    return count


# ── Pure Logic Signal Detection ────────────────────────────────────────────────

def check_signal(symbol: str, coin_analysis: dict,
                 btc_analysis: dict, btc_dominance: dict,
                 coin_price: float) -> dict:
    """
    Pure indicator logic — no AI needed.
    Returns signal dict or no-signal with reasons.
    """
    reasons_no_signal = []

    # ── BTC Dominance check ─────────────────────────────────────────────────
    dominance = btc_dominance.get("dominance", 50)
    if dominance > 62:
        reasons_no_signal.append(f"BTC Dominance too high ({dominance}%) — alts weak")

    # ── BTC Trend check ─────────────────────────────────────────────────────
    btc_4h = btc_analysis.get("4h", {})
    btc_1d = btc_analysis.get("1d", {})
    btc_trend_4h = btc_4h.get("trend", "NEUTRAL")
    btc_trend_1d = btc_1d.get("trend", "NEUTRAL")

    btc_bearish = "BEARISH" in btc_trend_4h and "BEARISH" in btc_trend_1d
    btc_bullish = "BULLISH" in btc_trend_4h and "BULLISH" in btc_trend_1d

    # ── Get coin indicators for 1h, 4h, 1d ─────────────────────────────────
    tf_1h = coin_analysis.get("1h", {})
    tf_4h = coin_analysis.get("4h", {})
    tf_1d = coin_analysis.get("1d", {})

    # Confluence scores
    conf_1h = tf_1h.get("confluence_score", {}).get("score", 50)
    conf_4h = tf_4h.get("confluence_score", {}).get("score", 50)
    conf_1d = tf_1d.get("confluence_score", {}).get("score", 50)

    # Supertrend
    st_4h = tf_4h.get("supertrend", {}).get("direction", "NEUTRAL")
    st_1d = tf_1d.get("supertrend", {}).get("direction", "NEUTRAL")

    # ADX
    adx_4h = tf_4h.get("adx", {}).get("adx", 0)
    adx_1d = tf_1d.get("adx", {}).get("adx", 0)

    # RSI
    rsi_1h = tf_1h.get("rsi", 50)
    rsi_4h = tf_4h.get("rsi", 50)

    # Volume
    vol_1h = tf_1h.get("volume", {}).get("ratio_vs_avg", 1.0)
    vol_4h = tf_4h.get("volume", {}).get("ratio_vs_avg", 1.0)

    # EMA alignment
    ema_align_4h = tf_4h.get("ema_crossover", {}).get("alignment", "")
    ema_align_1d = tf_1d.get("ema_crossover", {}).get("alignment", "")

    # ATR for SL/TP
    atr_4h = tf_4h.get("atr", 0)

    # ── ADX check ───────────────────────────────────────────────────────────
    if adx_4h < 18 and adx_1d < 18:
        reasons_no_signal.append(f"ADX too low ({adx_4h}) — sideways market, no trend")

    # ── Volume check ────────────────────────────────────────────────────────
    if vol_1h < 0.5 and vol_4h < 0.5:
        reasons_no_signal.append(f"Volume too low ({vol_4h}x avg) — no conviction")

    # ── LONG signal logic ────────────────────────────────────────────────────
    long_conditions = []
    short_conditions = []

    # LONG conditions
    if conf_4h >= 60:
        long_conditions.append(f"4H Confluence bullish ({conf_4h})")
    if conf_1d >= 58:
        long_conditions.append(f"1D Confluence bullish ({conf_1d})")
    if st_4h == "BULLISH":
        long_conditions.append("Supertrend 4H bullish")
    if st_1d == "BULLISH":
        long_conditions.append("Supertrend 1D bullish")
    if "BULLISH" in ema_align_4h:
        long_conditions.append("EMA aligned bullish 4H")
    if rsi_4h < 65 and rsi_4h > 40:
        long_conditions.append(f"RSI healthy ({rsi_4h})")
    # 1H confirmation
    if conf_1h >= 58:
        long_conditions.append(f"1H Confluence bullish ({conf_1h})")
    st_1h = tf_1h.get("supertrend", {}).get("direction", "NEUTRAL")
    if st_1h == "BULLISH":
        long_conditions.append("Supertrend 1H bullish")
    ema_align_1h = tf_1h.get("ema_crossover", {}).get("alignment", "")
    if "BULLISH" in ema_align_1h:
        long_conditions.append("EMA aligned bullish 1H")
    if btc_bullish:
        long_conditions.append("BTC trend bullish")

    # SHORT conditions
    if conf_4h <= 40:
        short_conditions.append(f"4H Confluence bearish ({conf_4h})")
    if conf_1d <= 42:
        short_conditions.append(f"1D Confluence bearish ({conf_1d})")
    if st_4h == "BEARISH":
        short_conditions.append("Supertrend 4H bearish")
    if st_1d == "BEARISH":
        short_conditions.append("Supertrend 1D bearish")
    if "BEARISH" in ema_align_4h:
        short_conditions.append("EMA aligned bearish 4H")
    if rsi_4h > 35 and rsi_4h < 60:
        short_conditions.append(f"RSI healthy ({rsi_4h})")
    # 1H confirmation
    if conf_1h <= 42:
        short_conditions.append(f"1H Confluence bearish ({conf_1h})")
    st_1h = tf_1h.get("supertrend", {}).get("direction", "NEUTRAL")
    if st_1h == "BEARISH":
        short_conditions.append("Supertrend 1H bearish")
    ema_align_1h = tf_1h.get("ema_crossover", {}).get("alignment", "")
    if "BEARISH" in ema_align_1h:
        short_conditions.append("EMA aligned bearish 1H")
    if btc_bearish:
        short_conditions.append("BTC trend bearish")

    # ── Decision ────────────────────────────────────────────────────────────
    # Need minimum 4 conditions
    long_score  = len(long_conditions)
    short_score = len(short_conditions)

    # Block longs if BTC/dominance bearish
    if dominance > 62 or btc_bearish:
        long_score = 0

    # Block shorts if BTC bullish
    if btc_bullish:
        short_score = 0

    if long_score >= 4 and long_score > short_score:
        # Calculate entry/TP/SL
        entry = coin_price
        sl    = round(entry - (atr_4h * 1.5), 6)
        tp    = round(entry + (atr_4h * 3.0), 6)
        rr    = round((tp - entry) / (entry - sl), 2) if entry != sl else 0

        if rr >= 1.8:
            return {
                "signal"    : True,
                "direction" : "LONG",
                "entry"     : entry,
                "tp"        : tp,
                "sl"        : sl,
                "rr"        : rr,
                "leverage"  : 5 if adx_4h >= 25 else 3,
                "conditions": long_conditions,
                "conf_score": conf_4h,
            }
        else:
            reasons_no_signal.append(f"LONG setup but RR too low ({rr})")

    elif short_score >= 4 and short_score > long_score:
        entry = coin_price
        sl    = round(entry + (atr_4h * 1.5), 6)
        tp    = round(entry - (atr_4h * 3.0), 6)
        rr    = round((entry - tp) / (sl - entry), 2) if entry != sl else 0

        if rr >= 1.8:
            return {
                "signal"    : True,
                "direction" : "SHORT",
                "entry"     : entry,
                "tp"        : tp,
                "sl"        : sl,
                "rr"        : rr,
                "leverage"  : 5 if adx_4h >= 25 else 3,
                "conditions": short_conditions,
                "conf_score": conf_4h,
            }
        else:
            reasons_no_signal.append(f"SHORT setup but RR too low ({rr})")
    else:
        if long_score > 0 or short_score > 0:
            reasons_no_signal.append(
                f"Conditions insufficient (LONG:{long_score}/4, SHORT:{short_score}/4)"
            )
        else:
            if 45 <= rsi_4h <= 55:
                reasons_no_signal.append(f"RSI neutral ({rsi_4h}) — no clear direction")
            if not btc_bullish and not btc_bearish:
                reasons_no_signal.append("BTC trend unclear")

    return {
        "signal" : False,
        "reasons": reasons_no_signal if reasons_no_signal else ["No clear setup on any indicator"]
    }


# ── Format WhatsApp Messages ───────────────────────────────────────────────────

def signal_msg(symbol: str, sig: dict, ai_result: dict = None) -> str:
    icon = "📈" if sig["direction"] == "LONG" else "📉"
    now  = datetime.now(PKT).strftime("%I:%M %p PKT")

    ai_section = ""
    if ai_result:
        if ai_result.get("trade"):
            ai_section = f"\n\n✅ *AI CONFIRMED*\n{ai_result.get('analysis_summary', '')[:150]}"
        else:
            reason = ai_result.get("no_trade_reason", "No reason given")[:150]
            ai_section = f"\n\n❌ *AI: Not Confirmed*\n{reason}"

    conditions = "\n".join(f"• {c}" for c in sig.get("conditions", [])[:4])

    return f"""{icon} *SIGNAL: {symbol} {sig['direction']}*
⏰ {now}

💰 Entry:    {sig['entry']:,.4f} USDT
✅ Target:   {sig['tp']:,.4f} USDT
❌ SL:       {sig['sl']:,.4f} USDT
⚖️ R:R:      1 : {sig['rr']}
🔧 Leverage: {sig['leverage']}x
📊 Conf:     {sig['conf_score']}/100

📋 *Reasons:*
{conditions}{ai_section}"""


def no_signal_msg(reasons: list, scanned: int, btc_dominance: float) -> str:
    now     = datetime.now(PKT).strftime("%I:%M %p PKT")
    reasons_str = "\n".join(f"• {r}" for r in reasons[:5])
    return f"""🔍 *No Signal — {now}*
Scanned: {scanned} coins | BTC Dom: {btc_dominance}%

{reasons_str}"""


# ── Main Scan ──────────────────────────────────────────────────────────────────

def run_scan():
    try:
        with open("coins.txt", "r") as f:
            coins = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        send_whatsapp("❌ Monitor Error: coins.txt not found")
        return

    total = len(coins)
    now   = datetime.now(PKT).strftime("%Y-%m-%d %I:%M %p PKT")
    print(f"\n[{now}] Scanning {total} coins...")

    # ── Fetch global data once ───────────────────────────────────────────────
    btc_dominance_data = get_btc_dominance()
    dominance          = btc_dominance_data.get("dominance", 50)

    try:
        btc_tf       = fetch_all_timeframes("BTCUSDT")
        btc_price    = get_current_price("BTCUSDT")
        btc_analysis = {tf: analyze_timeframe(df, btc_price) for tf, df in btc_tf.items()}
    except Exception as e:
        send_whatsapp(f"❌ BTC fetch error: {str(e)[:80]}")
        return

    all_reasons = []
    signal_found = False

    for i, coin in enumerate(coins, 1):
        symbol = normalize_symbol(coin)
        print(f"  [{i}/{total}] {symbol}...", end="\r")
        time.sleep(3)

        try:
            coin_tf    = fetch_all_timeframes(symbol)
            coin_price = get_current_price(symbol)
            coin_analysis = {tf: analyze_timeframe(df, coin_price) for tf, df in coin_tf.items()}

            result = check_signal(symbol, coin_analysis, btc_analysis,
                                   btc_dominance_data, coin_price)

            if result.get("signal"):
                signal_found = True
                print(f"\n  ✅ SIGNAL: {symbol} {result['direction']}")

                # Step 1: Immediate WhatsApp (no AI yet)
                msg = signal_msg(symbol, result)
                send_whatsapp(msg)

                # Step 2: AI confirmation (if calls remaining)
                ai_calls = get_ai_calls_today()
                if ai_calls < MAX_AI_CALLS_PER_DAY:
                    print(f"  🤖 Sending to AI for confirmation ({ai_calls+1}/{MAX_AI_CALLS_PER_DAY})...")
                    try:
                        from src.fetcher import get_funding_rate, get_open_interest, get_fear_greed
                        funding_rate  = get_funding_rate(symbol)
                        open_interest = get_open_interest(symbol)
                        fear_greed    = get_fear_greed()

                        has_gemini = bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")
                        has_claude = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here")

                        if has_gemini and not has_claude:
                            ai_result = get_trade_signal_gemini(
                                symbol, coin_price, coin_analysis,
                                btc_price, btc_analysis, "N/A",
                                funding_rate, open_interest, fear_greed, btc_dominance_data
                            )
                        else:
                            ai_result = get_trade_signal(
                                symbol, coin_price, coin_analysis,
                                btc_price, btc_analysis, "N/A",
                                funding_rate, open_interest, fear_greed, btc_dominance_data
                            )

                        increment_ai_calls()

                        # Send AI confirmation WhatsApp
                        confirm_msg = signal_msg(symbol, result, ai_result)
                        send_whatsapp(confirm_msg)

                    except Exception as e:
                        send_whatsapp(f"⚠️ AI confirmation failed: {str(e)[:80]}")
                else:
                    send_whatsapp(f"⚠️ AI limit reached ({MAX_AI_CALLS_PER_DAY}/day) — no AI confirmation for {symbol}")

                return  # Stop after first signal

            else:
                # Collect reasons
                for r in result.get("reasons", []):
                    if r not in all_reasons:
                        all_reasons.append(r)

        except Exception as e:
            print(f"\n  [{i}/{total}] {symbol} error: {str(e)[:50]}")
            continue

    # No signal found — send reason message
    if not signal_found:
        print("  No signal found.")
        top_reasons = all_reasons[:5] if all_reasons else ["All indicators neutral"]
        msg = no_signal_msg(top_reasons, total, dominance)
        send_whatsapp(msg)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(PKT).strftime("%Y-%m-%d %I:%M %p PKT")
    print(f"\n  🚀 Smart Monitor Started (24/7)")
    print(f"  🕐 PKT Time: {now}")
    print(f"  ⏱ Interval: every {SCAN_INTERVAL_SECONDS // 60} minutes")
    print(f"  🤖 AI calls limit: {MAX_AI_CALLS_PER_DAY}/day")
    print(f"  📱 WhatsApp: {'ON' if CALLMEBOT_PHONE else 'OFF'}")
    print(f"\n  NOTE: Stop scheduler.py before running this!\n")

    send_whatsapp(f"🚀 Smart Monitor Started\n⏱ Scanning every 5 min\n🤖 AI limit: {MAX_AI_CALLS_PER_DAY}/day")

    while True:
        try:
            run_scan()
        except Exception as e:
            print(f"\n  ✗ Scan error: {e}")
            send_whatsapp(f"❌ Monitor error: {str(e)[:100]}")

        print(f"\n  ⏳ Next scan in {SCAN_INTERVAL_SECONDS // 60} min...")
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()