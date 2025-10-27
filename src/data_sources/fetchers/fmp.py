
import requests
from .base import BaseFetcher

class FMPFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, ticker: str):
        try:
            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={self.api_key}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()[0]
            return {"fmp": {
                "pe_ratio": data.get("peRatioTTM"),
                "eps": data.get("epsTTM"),
                "roa": data.get("returnOnAssetsTTM")
            }}
        except Exception as e:
            return {"fmp": {"error": str(e)}}
