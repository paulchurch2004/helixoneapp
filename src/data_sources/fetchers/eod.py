
import requests
from .base import BaseFetcher

class EODFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def supports(self, ticker: str) -> bool:
        return True  # large international support

    def fetch(self, ticker: str):
        try:
            url = f"https://eodhistoricaldata.com/api/eod/{ticker}?api_token={self.api_key}&fmt=json&limit=1"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()[0]
            return {"eod": {
                "open": data.get("open"),
                "close": data.get("close"),
                "volume": data.get("volume")
            }}
        except Exception as e:
            return {"eod": {"error": str(e)}}
