"""
scheduler.py — Runs coin scan at 1PM and 9PM Pakistan Time, Monday-Friday.
Deploy this on Railway: worker: python scheduler.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron       import CronTrigger
from datetime                        import datetime
import pytz

from src.indicators  import analyze_timeframe
from src.news        import fetch_news, news_summary
from src.ai_analyst  import get_trade_signal, get_trade_signal_gemini
from src.notifier    import send_whatsapp, format_signal_message
from src.fetcher import normalize_symbol, fetch_all_timeframes, get_current_price, get_funding_rate, get_open_interest, get_fear_greed
from config          import ANTHROPIC_API_KEY, GEMINI_API_KEY, CALLMEBOT_PHONE

PKT = pytz.timezone("Asia/Karachi")


def run_scheduled_scan():
    """Called automatically at scheduled times."""
    now = datetime.now(PKT).strftime("%Y-%m-%d %I:%M %p PKT")
    print(f"\n{'='*50}")
    print(f"  ⏰ Scheduled scan started: {now}")
    print(f"{'='*50}\n")

    # Read coins
    try:
        with open("coins.txt", "r") as f:
            coins = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("  ✗ coins.txt not found!")
        return

    total         = len(coins)
    signals_found = 0
    max_signals   = 3

    has_claude = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here")
    has_gemini = bool(GEMINI_API_KEY    and GEMINI_API_KEY    != "your_gemini_api_key_here")

    print(f"  📋 Scanning {total} coins...\n")

    for i, coin in enumerate(coins, 1):
        symbol = normalize_symbol(coin)
        print(f"  [{i}/{total}] Analyzing {symbol}...", end="\r")

        try:
            coin_tf    = fetch_all_timeframes(symbol)
            coin_price = get_current_price(symbol)
            btc_tf     = fetch_all_timeframes("BTCUSDT")
            btc_price  = get_current_price("BTCUSDT")

            coin_analysis = {tf: analyze_timeframe(df, coin_price) for tf, df in coin_tf.items()}
            btc_analysis  = {tf: analyze_timeframe(df, btc_price)  for tf, df in btc_tf.items()}

            news_items = fetch_news(coin)
            news_txt   = news_summary(news_items)
            funding_rate = get_funding_rate(symbol)
            open_interest = get_open_interest(symbol)
            fear_greed = get_fear_greed()
            if has_gemini and not has_claude:
                result = get_trade_signal_gemini(symbol, coin_price, coin_analysis,
                                                  btc_price, btc_analysis, news_txt, funding_rate, open_interest, fear_greed)
            else:
                result = get_trade_signal(symbol, coin_price, coin_analysis,
                                           btc_price, btc_analysis, news_txt, funding_rate, open_interest, fear_greed)

            if result.get("trade"):
                # Quality filter
                risk_ok  = result.get("risk_score", 99) <= 55
                rr_ok   = False
                rr      = 0
                try:
                    entry     = result["entry_price"]
                    target    = result["target_price"]
                    sl        = result["stop_loss"]
                    direction = result.get("direction")
                    if direction == "LONG":
                        rr = (target - entry) / (entry - sl)
                    else:
                        rr = (entry - target) / (sl - entry)
                    rr_ok = rr >= 1.8
                except Exception:
                    pass

                if not risk_ok or not rr_ok:
                    print(" " * 60, end="\r")
                    print(f"  [{i}/{total}] {symbol} — rejected (RR:{round(rr,2)} Risk:{result.get('risk_score')})")
                    continue

                signals_found += 1
                print(" " * 60, end="\r")
                print(f"\n  ✅ Signal {signals_found}/{max_signals}: {symbol}")
                print(f"     {result.get('direction')} | Entry: {result['entry_price']} | TP: {result['target_price']} | SL: {result['stop_loss']}")

                # WhatsApp
                msg  = format_signal_message(symbol, coin_price, result)
                sent = send_whatsapp(msg)
                print(f"     📱 WhatsApp: {'Sent ✓' if sent else 'Failed ✗'}")

                if signals_found >= max_signals:
                    print(f"\n  3 signals complete. Scan finished.\n")
                    return

        except Exception as e:
            print(" " * 60, end="\r")
            print(f"  [{i}/{total}] {symbol} — skipped ({str(e)[:50]})")
            continue

    print(" " * 60, end="\r")

    if signals_found == 0:
        print("\n  No trade signals found today.\n")
        send_whatsapp("🔍 Scan complete — No valid trade signals found.")


# ── Scheduler Setup ───────────────────────────────────────────────────────────

scheduler = BlockingScheduler(timezone=PKT)

# 1 PM PKT — Monday to Friday
scheduler.add_job(
    run_scheduled_scan,
    trigger=CronTrigger(hour=13, minute=0, day_of_week="mon-fri", timezone=PKT)
)

# 9 PM PKT — Monday to Friday
scheduler.add_job(
    run_scheduled_scan,
    trigger=CronTrigger(hour=21, minute=0, day_of_week="mon-fri", timezone=PKT)
)

# 5 PM PKT — Monday to Friday
scheduler.add_job(
    run_scheduled_scan,
    trigger=CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone=PKT)
)

if __name__ == "__main__":
    now = datetime.now(PKT).strftime("%Y-%m-%d %I:%M %p PKT")
    print(f"\n  🚀 Crypto Scheduler Started")
    print(f"  🕐 Current PKT Time: {now}")
    print(f"  📅 Schedule: Mon-Fri  |  1:00 PM + 9:00 PM PKT")
    print(f"  📱 WhatsApp notifications: {'ON' if CALLMEBOT_PHONE else 'OFF'}")
    print(f"  🔄 Running immediate scan on startup...\n")
    run_scheduled_scan()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n  Scheduler stopped.")
