
import requests
from .base import BaseFetcher

class TiingoFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def supports(self, ticker: str) -> bool:
        return not ticker.endswith(".PA")  # Tiingo mostly for US

    def fetch(self, ticker: str):
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            headers = {"Content-Type": "application/json", "Authorization": f"Token {self.api_key}"}
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()[0]
            return {"tiingo": {
                "open": data.get("open"),
                "close": data.get("close"),
                "volume": data.get("volume")
            }}
        except Exception as e:
            return {"tiingo": {"error": str(e)}}
