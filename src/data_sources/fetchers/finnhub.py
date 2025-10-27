
import requests
from .base import BaseFetcher

class FinnhubFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, ticker: str):
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.api_key}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return {
                "finnhub": {
                    "current_price": data["c"],
                    "high": data["h"],
                    "low": data["l"],
                    "open": data["o"],
                    "previous_close": data["pc"]
                }
            }
        except Exception as e:
            return {"finnhub": {"error": str(e)}}
