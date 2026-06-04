import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
BINANCE_BASE_URL     = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CALLMEBOT_PHONE  = os.getenv("CALLMEBOT_PHONE", "")
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY", "")