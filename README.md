# 🔍 Crypto Trade Analyzer

Personal trading analysis tool — enter any coin symbol, get a structured AI-powered
trade signal (or a clear "NO TRADE" if conditions aren't right).

---

## What It Does

- Fetches **1h / 4h / 1d** candle data from Binance (free, no key needed)
- Calculates full TA: EMA 9/21/50/200, RSI, MACD, Bollinger Bands, ATR, Support/Resistance
- **Always checks BTC** as market context alongside your coin
- Fetches **recent news** via CryptoPanic (optional, free API)
- Sends everything to **Claude AI** for analysis
- Returns a clean signal **only when conditions are genuinely favorable**:

```
Entry Price:   $2,340.50
Direction:     LONG
Leverage:      3x
Risk Score:    22/100  [LOW RISK]
Target Price:  $2,520.00
Stop Loss:     $2,200.00
Risk:Reward:   1 : 2.4
```

---

## Setup (Step by Step)

### 1. Clone / Download this folder

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get API Keys

#### Anthropic Claude API (Required)
- Go to https://console.anthropic.com
- Create account → API Keys → Create Key
- Cost: very cheap (each analysis = ~1000 tokens, fractions of a cent)

#### CryptoPanic API (Optional — for news)
- Go to https://cryptopanic.com/developers/api/
- Register free → get your API token

### 4. Create your .env file
```bash
cp .env.example .env
```
Then edit `.env` and paste your keys.

### 5. Run
```bash
python main.py ETH
python main.py SOLUSDT
python main.py BNB
```

Or run without argument and it will ask you:
```bash
python main.py
```

---

## Important Notes

- **Binance data** is fetched in real-time — no delay
- **30 days of data** is analyzed per timeframe
- The AI will say **NO TRADE** if conditions are not clear — this is by design
- Never risk more than **1-2%** of your total capital on any single trade
- This tool is for **personal informational use** only

---

## File Structure

```
crypto-analyzer/
├── main.py              ← Run this
├── requirements.txt
├── .env                 ← Your API keys (create from .env.example)
├── config.py
└── src/
    ├── fetcher.py       ← Binance data
    ├── indicators.py    ← TA calculations
    ├── news.py          ← CryptoPanic news
    └── ai_analyst.py   ← Claude AI analysis
```
