"""
news.py — Fetch crypto news from CoinDesk + CoinTelegraph FREE RSS feeds.
No API key required.
"""

import requests
import xml.etree.ElementTree as ET


FEEDS = {
    "CoinDesk"      : "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph" : "https://cointelegraph.com/rss",
}


def fetch_news(symbol: str, max_items: int = 8) -> list[dict]:
    """
    Fetch recent news from CoinDesk + CoinTelegraph RSS.
    Filters by coin symbol if possible, else returns latest headlines.
    """
    base = symbol.replace("USDT", "").replace("usdt", "").upper()
    results = []

    for source, url in FEEDS.items():
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            items = root.findall(".//item")
            for item in items:
                title = item.findtext("title", "").strip()
                link  = item.findtext("link", "").strip()
                desc  = item.findtext("description", "").strip()

                # Filter: prefer coin-related news, fallback to general crypto
                combined = (title + desc).upper()
                relevant = base in combined or any(
                    k in combined for k in ["BITCOIN", "CRYPTO", "BTC", "ALTCOIN", "MARKET"]
                )

                if relevant and title:
                    results.append({
                        "title"    : title,
                        "source"   : source,
                        "sentiment": 0,  # RSS has no vote data, AI will judge from text
                        "url"      : link,
                    })

        except Exception:
            continue  # If one feed fails, other still works

    # Deduplicate by title, return latest max_items
    seen, unique = set(), []
    for item in results:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)

    return unique[:max_items]


def news_summary(news_items: list[dict]) -> str:
    """Turn news list into a compact string for the AI prompt."""
    if not news_items:
        return "No news data available."
    lines = []
    for i, n in enumerate(news_items, 1):
        lines.append(f"{i}. [{n['source']}] {n['title']}")
    return "\n".join(lines)