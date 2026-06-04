import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
BINANCE_BASE_URL     = "https://api.binance.com"
