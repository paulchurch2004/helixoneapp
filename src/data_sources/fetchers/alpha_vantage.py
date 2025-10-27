
import requests
from .base import BaseFetcher

class AlphaVantageFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, ticker: str):
        try:
            url = f"https://www.alphavantage.co/query?function=RSI&symbol={ticker}&interval=daily&time_period=14&series_type=close&apikey={self.api_key}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            rsi_data = data.get("Technical Analysis: RSI")
            last_entry = next(reversed(rsi_data.values()))
            return {"alpha_vantage": {"rsi": round(float(last_entry["RSI"]), 2)}}
        except Exception as e:
            return {"alpha_vantage": {"error": str(e)}}
