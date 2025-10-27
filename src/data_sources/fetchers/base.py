
from abc import ABC, abstractmethod
from typing import Dict

class BaseFetcher(ABC):
    @abstractmethod
    def fetch(self, ticker: str) -> Dict:
        pass
