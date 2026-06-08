"""
main.py — Crypto Trade Analyzer
Run: python main.py ETHUSDT   OR   python main.py ETH
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from colorama import init, Fore, Style, Back

# ── Local imports ─────────────────────────────────────────────────────────────
from src.indicators  import analyze_timeframe
from src.news        import fetch_news, news_summary
from src.fetcher import normalize_symbol, fetch_all_timeframes, get_current_price, get_funding_rate, get_open_interest, get_fear_greed
from src.ai_analyst  import get_trade_signal, get_trade_signal_gemini
from config          import ANTHROPIC_API_KEY, GEMINI_API_KEY
from src.notifier import send_whatsapp, format_signal_message
init(autoreset=True)  # colorama


# ─── Print Helpers ────────────────────────────────────────────────────────────

W  = Style.BRIGHT + Fore.WHITE
DIM = Style.DIM + Fore.WHITE
G  = Style.BRIGHT + Fore.GREEN
R  = Style.BRIGHT + Fore.RED
Y  = Style.BRIGHT + Fore.YELLOW
C  = Style.BRIGHT + Fore.CYAN
M  = Style.BRIGHT + Fore.MAGENTA
RESET = Style.RESET_ALL

LINE = W + "═" * 52 + RESET


def header(text):
    pad = (50 - len(text)) // 2
    print(LINE)
    print(W + "║" + " " * pad + C + text + " " * (50 - pad - len(text)) + W + "║" + RESET)
    print(LINE)


def row(label, value, color=W):
    label_str = (DIM + label).ljust(26)
    print(f"  {label_str} {color}{value}{RESET}")


def sep():
    print(DIM + "─" * 52 + RESET)


# ─── Risk Score Color ─────────────────────────────────────────────────────────

def risk_color(score):
    if score <= 30:
        return G + f"{score}/100  [LOW RISK]"
    elif score <= 60:
        return Y + f"{score}/100  [MEDIUM RISK]"
    else:
        return R + f"{score}/100  [HIGH RISK]"


def btc_trend_color(t):
    t = t.upper()
    if "BULL" in t:   return G + f"▲ {t}"
    if "BEAR" in t:   return R + f"▼ {t}"
    return Y + f"→ {t}"


def news_color(s):
    s = s.upper()
    if s == "POSITIVE": return G + "↑ POSITIVE"
    if s == "NEGATIVE": return R + "↓ NEGATIVE"
    return Y + "→ NEUTRAL"


# ─── Main Logic ───────────────────────────────────────────────────────────────

def run(symbol_input: str):
    symbol = normalize_symbol(symbol_input)
    base   = symbol.replace("USDT", "")

    print()
    print(C + f"  🔍 Analyzing {symbol}..." + RESET)
    print(DIM + "  Fetching Binance candles (1h / 4h / 1d) for last 30 days..." + RESET)

    # ── Fetch coin data ───────────────────────────────────────────────────────
    try:
        coin_tf   = fetch_all_timeframes(symbol)
        coin_price = get_current_price(symbol)
    except ValueError as e:
        print(R + f"\n  ✗ Error: {e}" + RESET)
        return

    print(DIM + "  Fetching BTC context..." + RESET)

    # ── Fetch BTC data ────────────────────────────────────────────────────────
    btc_tf    = fetch_all_timeframes("BTCUSDT")
    btc_price = get_current_price("BTCUSDT")

    print(DIM + "  Calculating indicators..." + RESET)

    # ── Compute indicators ────────────────────────────────────────────────────
    coin_analysis = {tf: analyze_timeframe(df, coin_price) for tf, df in coin_tf.items()}
    btc_analysis  = {tf: analyze_timeframe(df, btc_price)  for tf, df in btc_tf.items()}

    print(DIM + "  Fetching news..." + RESET)

    # ── News ──────────────────────────────────────────────────────────────────
    news_items   = fetch_news(base)
    news_txt     = news_summary(news_items)
    funding_rate = get_funding_rate(symbol)
    open_interest = get_open_interest(symbol)
    fear_greed = get_fear_greed()
     # ── AI Selection ─────────────────────────────────────────────────────────
    has_claude  = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here")
    has_gemini  = bool(GEMINI_API_KEY    and GEMINI_API_KEY    != "your_gemini_api_key_here")

    ai_choice = "C"  # default

    if has_claude and has_gemini:
        print()
        print(C + "  Both AI keys detected." + RESET)
        print(W + "  Which AI to use?  Claude (C)  /  Gemini (G)" + RESET)
        ai_choice = input(C + "  Enter C or G: " + RESET).strip().upper()
        if ai_choice not in ("C", "G"):
            ai_choice = "C"
        print()
    elif has_gemini and not has_claude:
        ai_choice = "G"
    elif has_claude and not has_gemini:
        ai_choice = "C"
    else:
        print(R + "\n  ✗ No AI API key found. Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env\n" + RESET)
        return

    if ai_choice == "G":
        print(DIM + "  Asking Gemini to analyze everything...\n" + RESET)
        result = get_trade_signal_gemini(
            symbol, coin_price, coin_analysis,
            btc_price, btc_analysis, news_txt, funding_rate, open_interest, fear_greed
        )
    else:
        print(DIM + "  Asking Claude to analyze everything...\n" + RESET)
        result = get_trade_signal(
            symbol, coin_price, coin_analysis,
            btc_price, btc_analysis, news_txt, funding_rate, open_interest, fear_greed
        )

    # ── Print Output ──────────────────────────────────────────────────────────
    now = datetime.now().astimezone().strftime("%Y-%m-%d  %I:%M %p %Z")
    header(f"TRADE ANALYSIS  :  {symbol}")

    row("Coin",          W  + symbol)
    row("Current Price", G  + f"${coin_price:,.4f}")
    row("Analysis Time", DIM + now)

    sep()
    row("BTC Price",  W + f"${btc_price:,.2f}")
    row("BTC Trend",  btc_trend_color(result.get("btc_trend", "NEUTRAL")))

    sep()
    row("News Sentiment", news_color(result.get("news_sentiment", "NEUTRAL")))

    sep()

    trade = result.get("trade", False)

    if trade:
        direction = result.get("direction", "?")
        dir_color = G if direction == "LONG" else R
        dir_icon  = "📈" if direction == "LONG" else "📉"

        print()
        print(W + f"  {dir_icon}  SIGNAL FOUND" + RESET)
        print()

        row("Direction",    dir_color + direction)
        row("Entry Price",  W + f"${result['entry_price']:,.4f}")
        row("Target Price", G + f"${result['target_price']:,.4f}")
        row("Stop Loss",    R + f"${result['stop_loss']:,.4f}")

        # R:R ratio
        entry  = result['entry_price']
        target = result['target_price']
        sl     = result['stop_loss']
        try:
            if direction == "LONG":
                rr = round((target - entry) / (entry - sl), 2)
            else:
                rr = round((entry - target) / (sl - entry), 2)
            row("Risk:Reward",  (G if rr >= 2 else Y) + f"1 : {rr}")
        except Exception:
            pass

        # Expected TP Time
        tp_hours = result.get("expected_tp_hours")
        if tp_hours:
            days  = int(tp_hours // 24)
            hours = int(tp_hours % 24)
            if days > 0:
                tp_str = f"{days}d {hours}h"
            else:
                tp_str = f"{hours}h"
            row("Expected TP Time", C + tp_str)

        row("Leverage",    Y + f"{result.get('leverage', '?')}x")
        row("Risk Score",  risk_color(result.get("risk_score", 99)))

        sep()
        print(f"\n  {DIM}Key Reasons:{RESET}")
        for reason in result.get("key_reasons", []):
            print(f"    {G}•{RESET} {reason}")

    else:
        print()
        print(R + "  ⛔  NO TRADE — CONDITIONS NOT MET" + RESET)
        print()
        reason = result.get("no_trade_reason", "Market conditions are unfavorable.")
        # Word-wrap reason
        words = reason.split()
        line, lines = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 46:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))
        for l in lines:
            print(f"  {DIM}{l}{RESET}")

        row("\n  Risk Score", risk_color(result.get("risk_score", 99)))

    sep()
    print(f"\n  {DIM}Analysis Summary:{RESET}")
    summary = result.get("analysis_summary", "")
    words = summary.split()
    line, lines = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 46:
            lines.append(" ".join(line[:-1]))
            line = [w]
    if line:
        lines.append(" ".join(line))
    for l in lines:
        print(f"  {l}")


def run_scan(filepath: str):
    """Scan all coins from file, stop at first trade signal."""
    try:
        with open(filepath, 'r') as f:
            coins = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(R + f"\n  ✗ File not found: {filepath}" + RESET)
        return

    total = len(coins)
    print()
    print(C + f"  📋 Scanning {total} coins for trade signals..." + RESET)
    print(DIM + "  Will stop after 3 signals.\n" + RESET)
    signals_found = 0
    max_signals = 3

    for i, coin in enumerate(coins, 1):
        symbol = normalize_symbol(coin)
        print(DIM + f"  [{i}/{total}] Analyzing {symbol}..." + RESET, end='\r')

        try:
            coin_tf    = fetch_all_timeframes(symbol)
            coin_price = get_current_price(symbol)
            btc_tf     = fetch_all_timeframes("BTCUSDT")
            btc_price  = get_current_price("BTCUSDT")

            coin_analysis = {tf: analyze_timeframe(df, coin_price) for tf, df in coin_tf.items()}
            btc_analysis  = {tf: analyze_timeframe(df, btc_price)  for tf, df in btc_tf.items()}

            news_items = fetch_news(coin)
            news_txt   = news_summary(news_items)

            has_claude = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here")
            has_gemini = bool(GEMINI_API_KEY    and GEMINI_API_KEY    != "your_gemini_api_key_here")
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
                # Quality filters — reject weak signals
                rr_ok    = False
                risk_ok  = result.get("risk_score", 99) <= 55
                try:
                    entry  = result['entry_price']
                    target = result['target_price']
                    sl     = result['stop_loss']
                    direction = result.get("direction")
                    if direction == "LONG":
                        rr = (target - entry) / (entry - sl)
                    else:
                        rr = (entry - target) / (sl - entry)
                    rr_ok = rr >= 1.8
                except Exception:
                    rr_ok = False

                if not risk_ok or not rr_ok:
                    print(" " * 60, end='\r')
                    print(DIM + f"  [{i}/{total}] {symbol} — signal rejected (RR:{round(rr,2) if rr_ok is False else round(rr,2)} / Risk:{result.get('risk_score')})" + RESET)
                    continue

                signals_found += 1
                print(" " * 60, end='\r')
                print(G + f"\n  ✅ Signal {signals_found}/3 found on {symbol}!\n" + RESET)
                print_signal_only(symbol, coin_price, result)
                # WhatsApp notification
                msg = format_signal_message(symbol, coin_price, result)
                sent = send_whatsapp(msg)
                if sent:
                    print(G + "  📱 WhatsApp notification sent!" + RESET)
                else:
                    print(DIM + "  📱 WhatsApp not configured or failed." + RESET)
                if signals_found >= max_signals:
                    print(G + "  3 signals complete. Scan finished.\n" + RESET)
                    return
                print(DIM + f"  Continuing scan for signal {signals_found + 1}/3...\n" + RESET)

        except Exception as e:
            print(" " * 60, end='\r')
            print(DIM + f"  [{i}/{total}] {symbol} — skipped ({str(e)[:40]})" + RESET)
            continue

    print(" " * 60, end='\r')
    print()
    print(Y + "  No trade signal found across all coins." + RESET)
    print()


def print_signal_only(symbol: str, coin_price: float, result: dict):
    """Print only the trade signal — no news, no summary, no analysis."""
    now = datetime.now().astimezone().strftime("%Y-%m-%d  %I:%M %p %Z")

    direction = result.get("direction", "?")
    dir_color = G if direction == "LONG" else R
    dir_icon  = "📈" if direction == "LONG" else "📉"

    header(f"{dir_icon} TRADE SIGNAL  :  {symbol}")

    row("Coin",          W  + symbol)
    row("Current Price", G  + f"${coin_price:,.4f}")
    row("Signal Time",   DIM + now)
    sep()

    row("Direction",    dir_color + direction)
    row("Entry Price",  W + f"${result['entry_price']:,.4f}")
    row("Target Price", G + f"${result['target_price']:,.4f}")
    row("Stop Loss",    R + f"${result['stop_loss']:,.4f}")

    try:
        entry  = result['entry_price']
        target = result['target_price']
        sl     = result['stop_loss']
        if direction == "LONG":
            rr = round((target - entry) / (entry - sl), 2)
        else:
            rr = round((entry - target) / (sl - entry), 2)
        row("Risk:Reward", (G if rr >= 2 else Y) + f"1 : {rr}")
    except Exception:
        pass

    tp_hours = result.get("expected_tp_hours")
    if tp_hours:
        days  = int(tp_hours // 24)
        hours = int(tp_hours % 24)
        tp_str = f"{days}d {hours}h" if days > 0 else f"{hours}h"
        row("Expected TP Time", C + tp_str)

    row("Leverage",   Y + f"{result.get('leverage', '?')}x")
    row("Risk Score", risk_color(result.get("risk_score", 99)))
    sep()
    row("BTC Trend",  btc_trend_color(result.get("btc_trend", "NEUTRAL")))

    print()
    print(LINE)
    print()



# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print()
        print(C + "  Crypto Trade Analyzer" + RESET)
        print(DIM + "  Usage: python main.py <SYMBOL>" + RESET)
        print(DIM + "  Examples:" + RESET)
        print(DIM + "    python main.py ETH" + RESET)
        print(DIM + "    python main.py SOLUSDT" + RESET)
        print(DIM + "    python main.py BNB" + RESET)
        print()
        sym = input(C + "  Enter coin symbol: " + RESET).strip()
        if sym:
            run(sym)
    elif sys.argv[1] == "--scan" and len(sys.argv) >= 3:
        run_scan(sys.argv[2])
    else:
        run(sys.argv[1])
