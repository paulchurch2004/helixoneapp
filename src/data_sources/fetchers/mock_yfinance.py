
import random
from .base import BaseFetcher

class MockYFinanceFetcher(BaseFetcher):
    def fetch(self, ticker: str):
        return {
            "yfinance": {
                "price": round(random.uniform(100, 500), 2),
                "volume": random.randint(100000, 5000000)
            }
        }
