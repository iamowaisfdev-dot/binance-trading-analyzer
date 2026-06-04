"""
notifier.py — WhatsApp notifications via CallMeBot
"""

import requests
from urllib.parse import quote
from config import CALLMEBOT_PHONE, CALLMEBOT_APIKEY


def send_whatsapp(message: str) -> bool:
    if not CALLMEBOT_PHONE or not CALLMEBOT_APIKEY:
        return False
    try:
        url = f"https://api.callmebot.com/whatsapp.php"
        params = {
            "phone"  : CALLMEBOT_PHONE,
            "text"   : message,
            "apikey" : CALLMEBOT_APIKEY
        }
        resp = requests.get(url, params=params, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def format_signal_message(symbol: str, price: float, result: dict) -> str:
    direction = result.get("direction", "?")
    icon      = "📈" if direction == "LONG" else "📉"

    tp_hours = result.get("expected_tp_hours")
    tp_str   = ""
    if tp_hours:
        days  = int(tp_hours // 24)
        hours = int(tp_hours % 24)
        tp_str = f"{days}d {hours}h" if days > 0 else f"{hours}h"

    try:
        entry  = result['entry_price']
        target = result['target_price']
        sl     = result['stop_loss']
        if direction == "LONG":
            rr = round((target - entry) / (entry - sl), 2)
        else:
            rr = round((entry - target) / (sl - entry), 2)
        rr_str = f"1 : {rr}"
    except Exception:
        rr_str = "N/A"

    msg = f"""{icon} *TRADE SIGNAL — {symbol}*

💰 Price:     {price:,.4f}
📊 Direction: {direction}
🎯 Entry:     {result['entry_price']:,.4f}
✅ Target:    {result['target_price']:,.4f}
❌ SL:        {result['stop_loss']:,.4f}
⚖️ R:R:       {rr_str}
⏱ TP Time:   {tp_str}
🔧 Leverage:  {result.get('leverage', '?')}x
⚠️ Risk:      {result.get('risk_score')}/100
📈 BTC:       {result.get('btc_trend')}"""

    return msg