
import requests
from datetime import date, timedelta
from .base import BaseFetcher

class PolygonFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, ticker: str):
        try:
            yesterday = date.today() - timedelta(days=1)
            url = f"https://api.polygon.io/v1/open-close/{ticker}/{yesterday}?adjusted=true&apiKey={self.api_key}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return {"polygon": {
                "open": data.get("open"),
                "close": data.get("close"),
                "volume": data.get("volume")
            }}
        except Exception as e:
            return {"polygon": {"error": str(e)}}
