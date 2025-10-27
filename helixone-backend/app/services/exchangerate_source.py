"""
ExchangeRate-API Data Source
Documentation: https://www.exchangerate-api.com/docs/

Free Tier:
- 1,500 requests/month FREE
- No credit card required
- API key required (free signup)

Coverage:
- 160+ currencies
- Real-time exchange rates
- Historical data
- Currency conversion

Use Cases:
- Forex exchange rates
- Multi-currency support
- Currency conversion for international trades
- Alternative to paid forex APIs
"""

import requests
import time
import os
from typing import Dict, List, Optional
from datetime import datetime


class ExchangeRateSource:
    """
    ExchangeRate-API collector for forex data

    Free: 1,500 requests/month
    Coverage: 160+ currencies
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ExchangeRate-API source

        Args:
            api_key: API key (optional, reads from env if not provided)
        """
        self.api_key = api_key or os.getenv('EXCHANGERATE_API_KEY')
        self.base_url = "https://v6.exchangerate-api.com/v6"

        # Rate limiting: 1500/month ≈ 50/day ≈ 2/hour
        # Be very conservative
        self.min_request_interval = 2.0
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str) -> Dict:
        """Make API request"""
        if not self.api_key:
            raise Exception("ExchangeRate API key required. Set EXCHANGERATE_API_KEY in .env or get one at https://www.exchangerate-api.com/")

        self._rate_limit()

        url = f"{self.base_url}/{self.api_key}/{endpoint}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if data.get('result') != 'success':
                error = data.get('error-type', 'Unknown error')
                raise Exception(f"ExchangeRate-API error: {error}")

            return data

        except requests.exceptions.RequestException as e:
            raise Exception(f"ExchangeRate-API request failed: {str(e)}")

    def get_latest_rates(self, base: str = 'USD') -> Dict:
        """
        Get latest exchange rates for base currency

        Args:
            base: Base currency code (e.g., 'USD', 'EUR', 'GBP')

        Returns:
            Dictionary with rates and metadata

        Example:
            >>> rates = exchangerate.get_latest_rates('USD')
            >>> eur_rate = rates['conversion_rates']['EUR']
            >>> print(f"1 USD = {eur_rate} EUR")
        """
        return self._make_request(f'latest/{base}')

    def get_pair_conversion(self, base: str, target: str, amount: float = 1.0) -> Dict:
        """
        Convert amount from base to target currency

        Args:
            base: Base currency code
            target: Target currency code
            amount: Amount to convert (default 1.0)

        Returns:
            Conversion result

        Example:
            >>> result = exchangerate.get_pair_conversion('USD', 'EUR', 100)
            >>> print(f"100 USD = {result['conversion_result']} EUR")
        """
        endpoint = f'pair/{base}/{target}/{amount}'
        return self._make_request(endpoint)

    def get_supported_currencies(self) -> List[str]:
        """
        Get list of supported currency codes

        Returns:
            List of 3-letter currency codes

        Example:
            >>> currencies = exchangerate.get_supported_currencies()
            >>> print(f"{len(currencies)} currencies supported")
        """
        # Get rates for USD (contains all currencies)
        data = self.get_latest_rates('USD')
        return list(data['conversion_rates'].keys())

    # === Convenience Methods ===

    def get_exchange_rate(self, base: str, target: str) -> float:
        """
        Get simple exchange rate

        Args:
            base: Base currency
            target: Target currency

        Returns:
            Exchange rate as float

        Example:
            >>> rate = exchangerate.get_exchange_rate('USD', 'EUR')
            >>> print(f"1 USD = {rate} EUR")
        """
        data = self.get_latest_rates(base)
        return data['conversion_rates'][target]

    def get_multiple_rates(self, base: str, targets: List[str]) -> Dict[str, float]:
        """
        Get exchange rates for multiple target currencies

        Args:
            base: Base currency
            targets: List of target currencies

        Returns:
            Dictionary {currency: rate}

        Example:
            >>> rates = exchangerate.get_multiple_rates('USD', ['EUR', 'GBP', 'JPY'])
            >>> for curr, rate in rates.items():
            ...     print(f"1 USD = {rate} {curr}")
        """
        data = self.get_latest_rates(base)
        all_rates = data['conversion_rates']

        return {target: all_rates[target] for target in targets if target in all_rates}

    def convert_currency(self, amount: float, base: str, target: str) -> float:
        """
        Convert currency amount

        Args:
            amount: Amount to convert
            base: Source currency
            target: Target currency

        Returns:
            Converted amount

        Example:
            >>> eur_amount = exchangerate.convert_currency(100, 'USD', 'EUR')
            >>> print(f"100 USD = {eur_amount:.2f} EUR")
        """
        result = self.get_pair_conversion(base, target, amount)
        return result['conversion_result']

    def get_major_pairs(self, base: str = 'USD') -> Dict[str, float]:
        """
        Get rates for major currency pairs

        Args:
            base: Base currency (default USD)

        Returns:
            Dictionary with major pairs

        Example:
            >>> major = exchangerate.get_major_pairs('USD')
        """
        major_currencies = ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'CNY']
        return self.get_multiple_rates(base, major_currencies)


# === Singleton Pattern ===

_exchangerate_instance = None

def get_exchangerate_collector(api_key: Optional[str] = None) -> ExchangeRateSource:
    """
    Get or create ExchangeRate collector instance (singleton pattern)

    Args:
        api_key: Optional API key override

    Returns:
        ExchangeRateSource instance
    """
    global _exchangerate_instance

    if _exchangerate_instance is None:
        _exchangerate_instance = ExchangeRateSource(api_key=api_key)

    return _exchangerate_instance
