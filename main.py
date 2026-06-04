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
from src.fetcher     import normalize_symbol, fetch_all_timeframes, get_current_price
from src.indicators  import analyze_timeframe
from src.news        import fetch_news, news_summary
from src.ai_analyst  import get_trade_signal

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
    news_items = fetch_news(base)
    news_txt   = news_summary(news_items)

    print(DIM + "  Asking Claude to analyze everything...\n" + RESET)

    # ── AI Analysis ───────────────────────────────────────────────────────────
    result = get_trade_signal(
        symbol, coin_price, coin_analysis,
        btc_price, btc_analysis, news_txt
    )

    # ── Print Output ──────────────────────────────────────────────────────────
    now = datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")
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

    print()
    print(LINE)
    print(DIM + "  ⚠  This tool is for informational purposes only." + RESET)
    print(DIM + "     Always manage your own risk. Never risk more than" + RESET)
    print(DIM + "     1-2% of your portfolio on any single trade." + RESET)
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
    else:
        run(sys.argv[1])
