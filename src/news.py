"""
news.py — Fetch recent crypto news from CryptoPanic (free API).
Falls back gracefully if no key is provided.
"""

import requests
from config import CRYPTOPANIC_API_KEY


def fetch_news(symbol: str, max_items: int = 8) -> list[dict]:
    """
    Fetch recent news for a given coin.
    symbol : base coin symbol e.g. 'ETH', 'BNB'  (not ETHUSDT)

    Returns list of dicts: [{"title": ..., "sentiment": ..., "url": ...}, ...]
    Returns empty list if no key / request fails.
    """
    if not CRYPTOPANIC_API_KEY or CRYPTOPANIC_API_KEY == "your_cryptopanic_api_key_here":
        return []

    # Strip USDT from symbol if present
    base = symbol.replace("USDT", "").replace("usdt", "").upper()

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token" : CRYPTOPANIC_API_KEY,
        "currencies" : base,
        "kind"       : "news",
        "public"     : "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results = []
    for item in data.get("results", [])[:max_items]:
        results.append({
            "title"    : item.get("title", ""),
            "sentiment": item.get("votes", {}).get("positive", 0) - item.get("votes", {}).get("negative", 0),
            "url"      : item.get("url", ""),
        })

    return results


def news_summary(news_items: list[dict]) -> str:
    """Turn news list into a compact string for the AI prompt."""
    if not news_items:
        return "No news data available."
    lines = []
    for i, n in enumerate(news_items, 1):
        sent = "↑" if n["sentiment"] > 0 else ("↓" if n["sentiment"] < 0 else "→")
        lines.append(f"{i}. [{sent}] {n['title']}")
    return "\n".join(lines)
