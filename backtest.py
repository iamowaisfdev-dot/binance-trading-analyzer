"""
backtest.py — Backtest trading strategy on 30 days of historical data.

Key design:
- Simulates EXACT same schedule: Mon-Fri, 1PM & 9PM PKT
- Uses ONLY ONE AI call for all time slots (token efficient)
- TP/SL outcome checked from actual price data (no AI needed)
- $1000 starting balance, 25% margin per trade

Usage:
    python backtest.py ADA
    python backtest.py SOL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import pandas as pd
from datetime import datetime, timedelta
import pytz
from colorama import init, Fore, Style

from src.fetcher    import normalize_symbol, get_klines
from src.indicators import ema, rsi, macd, atr
from config         import GEMINI_API_KEY, ANTHROPIC_API_KEY

init(autoreset=True)
PKT = pytz.timezone("Asia/Karachi")

W     = Style.BRIGHT + Fore.WHITE
G     = Style.BRIGHT + Fore.GREEN
R     = Style.BRIGHT + Fore.RED
Y     = Style.BRIGHT + Fore.YELLOW
C     = Style.BRIGHT + Fore.CYAN
DIM   = Style.DIM   + Fore.WHITE
RESET = Style.RESET_ALL
LINE  = W + "═" * 60 + RESET


# ── 1. Generate Mon-Fri 1PM/9PM PKT slots ─────────────────────────────────────

def get_slots(days_back: int = 30) -> list:
    now    = datetime.now(PKT)
    slots  = []
    for d in range(days_back, 0, -1):
        day = now - timedelta(days=d)
        if day.weekday() < 5:                    # Mon=0 … Fri=4
            for hour in [13, 21]:                # 1 PM, 9 PM
                slot = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                if slot < now:
                    slots.append(slot)
    return slots


# ── 2. Build compact snapshot for one slot ────────────────────────────────────

def _quick(df: pd.DataFrame) -> str:
    """Return one-line indicator summary for a dataframe slice."""
    c     = df['close']
    price = round(c.iloc[-1], 6)
    e9    = round(ema(c, 9).iloc[-1],  6)
    e21   = round(ema(c, 21).iloc[-1], 6)
    e50   = round(ema(c, 50).iloc[-1], 6)
    rsi_v = round(rsi(c).iloc[-1], 1)
    _, _, hist = macd(c)
    macd_v = round(hist.iloc[-1], 8)
    atr_v  = round(atr(df).iloc[-1], 8)
    vol_r  = round(df['volume'].iloc[-1] / df['volume'].tail(20).mean(), 2)

    if price > e9 > e21 > e50:   trend = "BULL"
    elif price < e9 < e21 < e50: trend = "BEAR"
    else:                         trend = "NEUT"

    return (f"P:{price}|T:{trend}|RSI:{rsi_v}|"
            f"MACD:{macd_v}|ATR:{atr_v}|VOL:{vol_r}x|"
            f"E9:{e9}|E21:{e21}|E50:{e50}")


def build_snapshot(df_coin: pd.DataFrame, df_btc: pd.DataFrame,
                   slot: datetime, symbol: str) -> str | None:
    """Slice data up to slot time and return compact string."""
    utc = slot.astimezone(pytz.utc).replace(tzinfo=None)

    coin_sl = df_coin[df_coin.index <= utc].tail(200)
    btc_sl  = df_btc [df_btc.index  <= utc].tail(200)

    if len(coin_sl) < 50 or len(btc_sl) < 50:
        return None

    t = slot.strftime("%m-%d %I:%M%p")
    return f"{t}|{symbol}:{_quick(coin_sl)}|BTC:{_quick(btc_sl)}"


# ── 3. ONE AI call for all snapshots ──────────────────────────────────────────

def batch_ai_analysis(snapshots: list, symbol: str) -> list:
    n = len(snapshots)
    snap_text = "\n".join(f"SLOT_{i+1}: {s}" for i, s in enumerate(snapshots))

    prompt = f"""You are a crypto futures trading analyst. Backtest a strategy on {symbol}.

Below are {n} market snapshots taken at 1PM and 9PM PKT, Mon-Fri.
Each line: timestamp | coin indicators | BTC indicators
Indicators: Price, Trend(BULL/BEAR/NEUT), RSI, MACD-histogram, ATR, Volume-ratio, EMA9/21/50

SIGNAL RULES:
- LONG: Trend=BULL, RSI>55, MACD>0, Price>EMA9>EMA21, BTC also BULL, Volume>1.0x
- SHORT: Trend=BEAR, RSI<45, MACD<0, Price<EMA9<EMA21, BTC also BEAR, Volume>1.0x
- Entry = current Price
- SL = 1.5 × ATR from entry (LONG: entry-1.5xATR, SHORT: entry+1.5xATR)
- TP = 3.0 × ATR from entry (LONG: entry+3xATR, SHORT: entry-3xATR)  
- Leverage: 5x if trend strong, 3x if moderate
- If conditions unclear = trade:false

SNAPSHOTS:
{snap_text}

Reply ONLY with a JSON array of exactly {n} objects, one per slot, in order:
[
  {{"slot":1,"trade":false,"direction":null,"entry":null,"tp":null,"sl":null,"leverage":null}},
  {{"slot":2,"trade":true,"direction":"LONG","entry":0.542,"tp":0.566,"sl":0.530,"leverage":5}},
  ...
]
No markdown, no explanation, just the JSON array."""

    raw = ""

    if GEMINI_API_KEY and GEMINI_API_KEY not in ("", "your_gemini_api_key_here"):
        import google.genai as genai
        client   = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        raw = response.text.strip()

    elif ANTHROPIC_API_KEY and ANTHROPIC_API_KEY not in ("", "your_anthropic_api_key_here"):
        import anthropic
        client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()

    else:
        raise ValueError("No AI API key found in .env")

    # Clean markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


# ── 4. Check TP/SL outcome from real price data ───────────────────────────────

def check_outcome(df: pd.DataFrame, entry: float, tp: float, sl: float,
                  direction: str, signal_time: datetime,
                  max_candles: int = 168) -> dict:
    """Scan forward candles — whichever hits first (TP or SL) wins."""
    utc    = signal_time.astimezone(pytz.utc).replace(tzinfo=None)
    future = df[df.index > utc].head(max_candles)

    for i, (ts, row) in enumerate(future.iterrows()):
        if direction == "LONG":
            if row['low']  <= sl: return {"result":"SL","exit":sl, "hours":i+1}
            if row['high'] >= tp: return {"result":"TP","exit":tp, "hours":i+1}
        else:
            if row['high'] >= sl: return {"result":"SL","exit":sl, "hours":i+1}
            if row['low']  <= tp: return {"result":"TP","exit":tp, "hours":i+1}

    last = future['close'].iloc[-1] if len(future) else entry
    return {"result":"OPEN","exit":round(last,6),"hours":max_candles}


# ── 5. P&L calculation ────────────────────────────────────────────────────────

def calc_pnl(balance: float, entry: float, exit_price: float,
             direction: str, leverage: int) -> tuple:
    margin = balance * 0.25
    pct    = ((exit_price - entry) / entry) if direction == "LONG" \
             else ((entry - exit_price) / entry)
    pnl    = margin * leverage * pct
    return round(pnl, 2), round(margin, 2)


# ── 6. Print report ───────────────────────────────────────────────────────────

def print_report(symbol: str, trades: list, start_bal: float):
    if not trades:
        print(Y + "\n  No trade signals found in this period.\n" + RESET)
        return

    tp_list   = [t for t in trades if t['result'] == 'TP']
    sl_list   = [t for t in trades if t['result'] == 'SL']
    op_list   = [t for t in trades if t['result'] == 'OPEN']
    final_bal = trades[-1]['bal']
    total_pnl = final_bal - start_bal
    win_rate  = len(tp_list) / len(trades) * 100
    pnls      = [t['pnl'] for t in trades]

    print()
    print(LINE)
    print(W + f"║{'  BACKTEST REPORT — ' + symbol:^58}║" + RESET)
    print(LINE)

    def row(label, val, color=W):
        print(f"  {DIM}{label:<18}{RESET}{color}{val}{RESET}")

    print()
    row("Period",        "Last 30 days  (Mon-Fri, 1PM & 9PM PKT)")
    row("Starting Bal",  f"${start_bal:,.2f}")
    row("Final Bal",     f"${final_bal:,.2f}", G if total_pnl >= 0 else R)
    row("Total PnL",     f"${total_pnl:+,.2f}  ({total_pnl/start_bal*100:+.1f}%)",
        G if total_pnl >= 0 else R)

    print(f"\n  {DIM}{'─'*55}{RESET}")
    row("Total Signals", str(len(trades)))
    row("TP Hit",        f"{len(tp_list)}  ({win_rate:.0f}%)", G)
    row("SL Hit",        str(len(sl_list)), R)
    row("Still Open",    str(len(op_list)), Y)
    row("Best Trade",    f"${max(pnls):+,.2f}", G)
    row("Worst Trade",   f"${min(pnls):+,.2f}", R)

    avg_h = sum(t['hours'] for t in trades) / len(trades)
    row("Avg Hold Time", f"{avg_h:.0f}h")

    print(f"\n  {DIM}{'─'*55}{RESET}")
    print(f"\n  {DIM}{'#':<4}{'Date/Time':<16}{'Dir':<7}{'Entry':<11}{'Exit':<11}"
          f"{'Res':<6}{'Lev':<6}{'PnL':<12}{'Balance'}{RESET}")
    print(f"  {DIM}{'─'*80}{RESET}")

    for t in trades:
        rc = G if t['result']=='TP' else (R if t['result']=='SL' else Y)
        pc = G if t['pnl'] >= 0 else R
        print(f"  {t['#']:<4}{t['time']:<16}{t['dir']:<7}"
              f"${t['entry']:<10}${t['exit']:<10}"
              f"{rc}{t['result']:<6}{RESET}{t['lev']:<6}x"
              f"{pc}${t['pnl']:>+8,.2f}{RESET}   ${t['bal']:,.2f}")

    print()
    print(LINE)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backtest(symbol_input: str):
    symbol = normalize_symbol(symbol_input)

    print()
    print(C + f"  📊 Backtesting {symbol} — Last 30 Days" + RESET)
    print(DIM + "  Mon-Fri | 1PM & 9PM PKT | $1,000 start | 25% margin/trade\n" + RESET)

    # Fetch ~2 months of 1h data for point-in-time slicing
    print(DIM + "  Fetching historical candles..." + RESET)
    df_coin = get_klines(symbol,    "1h", limit=1500)
    df_btc  = get_klines("BTCUSDT", "1h", limit=1500)

    slots = get_slots(days_back=30)
    print(DIM + f"  Total time slots: {len(slots)}" + RESET)

    # Build snapshots
    print(DIM + "  Building market snapshots..." + RESET)
    snaps, valid_slots = [], []
    for slot in slots:
        s = build_snapshot(df_coin, df_btc, slot, symbol)
        if s:
            snaps.append(s)
            valid_slots.append(slot)

    print(DIM + f"  Valid snapshots: {len(snaps)}" + RESET)
    print(DIM + f"  Calling AI (1 API call for all {len(snaps)} slots)..." + RESET)

    ai_results = batch_ai_analysis(snaps, symbol)

    # Pad/trim if AI returned wrong count
    while len(ai_results) < len(valid_slots):
        ai_results.append({"trade": False})

    print(G + f"  ✓ AI done. Processing outcomes...\n" + RESET)

    balance = 1000.0
    trades  = []
    num     = 0

    for slot, ai in zip(valid_slots, ai_results):
        if not ai.get("trade"):
            continue

        try:
            entry     = float(ai['entry'])
            tp        = float(ai['tp'])
            sl        = float(ai['sl'])
            direction = ai['direction']
            leverage  = int(ai.get('leverage') or 5)
        except (KeyError, TypeError, ValueError):
            continue

        outcome     = check_outcome(df_coin, entry, tp, sl, direction, slot)
        pnl, margin = calc_pnl(balance, entry, outcome['exit'], direction, leverage)
        balance     = max(0.0, balance + pnl)
        num        += 1

        trades.append({
            "#"     : num,
            "time"  : slot.strftime("%m-%d %I:%M%p"),
            "dir"   : direction,
            "entry" : round(entry, 5),
            "exit"  : round(outcome['exit'], 5),
            "result": outcome['result'],
            "hours" : outcome['hours'],
            "lev"   : leverage,
            "pnl"   : pnl,
            "bal"   : round(balance, 2),
        })

    print_report(symbol, trades, 1000.0)


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else input("  Coin symbol (e.g. ADA): ").strip()
    run_backtest(sym)
