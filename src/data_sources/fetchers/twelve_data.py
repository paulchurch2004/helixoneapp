
import requests
from .base import BaseFetcher

class TwelveDataFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, ticker: str):
        try:
            url = f"https://api.twelvedata.com/price?symbol={ticker}&apikey={self.api_key}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return {"twelve_data": {"price": round(float(data["price"]), 2)}}
        except Exception as e:
            return {"twelve_data": {"error": str(e)}}
