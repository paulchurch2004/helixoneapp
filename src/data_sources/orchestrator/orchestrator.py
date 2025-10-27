from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from .cache import SimpleCache
from data_sources.fetchers.base import BaseFetcher
from src.ticker_resolver import resolve_ticker  # 👈 Ajouté

class APIOrchestratorWithCache:
    def __init__(self, fetchers: List[BaseFetcher], cache: SimpleCache):
        self.fetchers = fetchers
        self.cache = cache

    def get_all_data(self, ticker: str) -> Dict:
        # 👇 Résolution automatique nom/ticker
        ticker = resolve_ticker(ticker, "reference_data/tickers_full.json")

        cached = self.cache.get(ticker)
        if cached:
            return {"source": "cache", "data": cached}

        results = {}
        with ThreadPoolExecutor(max_workers=len(self.fetchers)) as executor:
            futures = {executor.submit(f.fetch, ticker): f for f in self.fetchers}
            for future in as_completed(futures):
                try:
                    data = future.result()
                    results.update(data)
                except Exception as e:
                    results[type(futures[future]).__name__] = {"error": str(e)}

        self.cache.set(ticker, results)
        return {"source": "live", "data": results}
