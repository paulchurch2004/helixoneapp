
import time
from typing import Dict, Optional

class SimpleCache:
    def __init__(self, ttl_seconds=900):
        self.ttl = ttl_seconds
        self.store = {}

    def get(self, ticker: str) -> Optional[Dict]:
        entry = self.store.get(ticker)
        if entry:
            timestamp, data = entry
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.store[ticker]
        return None

    def set(self, ticker: str, data: Dict):
        self.store[ticker] = (time.time(), data)
