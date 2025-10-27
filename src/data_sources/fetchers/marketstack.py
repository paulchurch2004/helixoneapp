
import requests
from .base import BaseFetcher

class MarketStackFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, ticker: str):
        try:
            url = f"http://api.marketstack.com/v1/eod/latest?access_key={self.api_key}&symbols={ticker}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()["data"][0]
            return {"marketstack": {
                "open": data.get("open"),
                "close": data.get("close"),
                "volume": data.get("volume")
            }}
        except Exception as e:
            return {"marketstack": {"error": str(e)}}
