"""
Kraken API Data Source
Documentation: https://docs.kraken.com/rest/

Free Tier:
- UNLIMITED and FREE for public endpoints
- No API key required for market data
- Rate limit: 1 request/second for public data

Coverage:
- 200+ cryptocurrency pairs
- EUR, GBP, JPY, CAD pairs (not just USD)
- Institutional-grade exchange
- Real-time prices, orderbooks, OHLC data

Use Cases:
- European crypto prices (EUR pairs)
- Multi-currency crypto data
- Price comparison and arbitrage
- Alternative to Binance/Coinbase
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime


class KrakenSource:
    """
    Kraken API collector for cryptocurrency data

    Free: Unlimited public data
    Coverage: 200+ pairs, multi-currency (EUR, GBP, JPY, CAD)
    """

    def __init__(self):
        """Initialize Kraken API source"""
        self.base_url = "https://api.kraken.com/0/public"

        # Rate limiting: 1 req/sec for public endpoints
        self.min_request_interval = 1.0
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request"""
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if data.get('error') and len(data['error']) > 0:
                raise Exception(f"Kraken API error: {data['error']}")

            return data.get('result', {})

        except requests.exceptions.RequestException as e:
            raise Exception(f"Kraken API request failed: {str(e)}")

    def get_server_time(self) -> Dict:
        """Get server time"""
        return self._make_request('Time')

    def get_system_status(self) -> Dict:
        """Get system status"""
        return self._make_request('SystemStatus')

    def get_asset_info(self, assets: Optional[List[str]] = None) -> Dict:
        """
        Get asset information

        Args:
            assets: List of assets (e.g., ['XBT', 'ETH']) or None for all

        Returns:
            Asset info dictionary
        """
        params = {}
        if assets:
            params['asset'] = ','.join(assets)

        return self._make_request('Assets', params)

    def get_tradable_pairs(self, pairs: Optional[List[str]] = None) -> Dict:
        """
        Get tradable asset pairs

        Args:
            pairs: List of pairs or None for all

        Returns:
            Pair info dictionary

        Example:
            >>> pairs = kraken.get_tradable_pairs(['XXBTZUSD', 'XETHZUSD'])
        """
        params = {}
        if pairs:
            params['pair'] = ','.join(pairs)

        return self._make_request('AssetPairs', params)

    def get_ticker(self, pairs: List[str]) -> Dict:
        """
        Get ticker information

        Args:
            pairs: List of pairs (e.g., ['XXBTZUSD', 'XETHZUSD'])

        Returns:
            Ticker data for pairs

        Example:
            >>> ticker = kraken.get_ticker(['XXBTZUSD'])
            >>> btc_data = ticker['XXBTZUSD']
            >>> last_price = btc_data['c'][0]  # Last trade price
        """
        params = {'pair': ','.join(pairs)}
        return self._make_request('Ticker', params)

    def get_ohlc(
        self,
        pair: str,
        interval: int = 60,
        since: Optional[int] = None
    ) -> Dict:
        """
        Get OHLC (candlestick) data

        Args:
            pair: Asset pair
            interval: Time frame in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Return data since given timestamp

        Returns:
            OHLC data array

        Example:
            >>> ohlc = kraken.get_ohlc('XXBTZUSD', interval=60)
            >>> for candle in ohlc['XXBTZUSD'][:5]:
            ...     time, open, high, low, close, vwap, volume, count = candle
        """
        params = {
            'pair': pair,
            'interval': interval
        }

        if since:
            params['since'] = since

        return self._make_request('OHLC', params)

    def get_orderbook(self, pair: str, count: int = 10) -> Dict:
        """
        Get order book

        Args:
            pair: Asset pair
            count: Maximum number of asks/bids (default 10, max 500)

        Returns:
            Order book with asks and bids

        Example:
            >>> book = kraken.get_orderbook('XXBTZUSD', count=5)
            >>> asks = book['XXBTZUSD']['asks']  # [[price, volume, timestamp], ...]
            >>> bids = book['XXBTZUSD']['bids']
        """
        params = {
            'pair': pair,
            'count': min(count, 500)
        }

        return self._make_request('Depth', params)

    def get_recent_trades(self, pair: str, since: Optional[int] = None) -> Dict:
        """
        Get recent trades

        Args:
            pair: Asset pair
            since: Return trades since given timestamp

        Returns:
            Recent trades

        Example:
            >>> trades = kraken.get_recent_trades('XXBTZUSD')
            >>> for trade in trades['XXBTZUSD'][:5]:
            ...     price, volume, time, buy_sell, order_type, misc = trade
        """
        params = {'pair': pair}

        if since:
            params['since'] = since

        return self._make_request('Trades', params)

    def get_spread(self, pair: str, since: Optional[int] = None) -> Dict:
        """
        Get recent spread data

        Args:
            pair: Asset pair
            since: Return data since given timestamp

        Returns:
            Spread data
        """
        params = {'pair': pair}

        if since:
            params['since'] = since

        return self._make_request('Spread', params)

    # === Convenience Methods ===

    def get_crypto_price(self, symbol: str, quote: str = 'USD') -> float:
        """
        Get simple crypto price

        Args:
            symbol: Crypto symbol (use 'XBT' for Bitcoin, 'ETH', etc.)
            quote: Quote currency ('USD', 'EUR', 'GBP', etc.)

        Returns:
            Current price

        Example:
            >>> btc_usd = kraken.get_crypto_price('XBT', 'USD')
            >>> btc_eur = kraken.get_crypto_price('XBT', 'EUR')
        """
        # Kraken uses specific pair naming
        if symbol == 'BTC':
            symbol = 'XBT'

        # Build pair name (Kraken format)
        pair = f"X{symbol}Z{quote}"

        ticker = self.get_ticker([pair])

        # Get last trade price
        for pair_name, data in ticker.items():
            return float(data['c'][0])  # c = [price, lot_volume]

        raise Exception(f"Price not found for {symbol}/{quote}")

    def get_crypto_prices_multi_currency(self, symbol: str) -> Dict[str, float]:
        """
        Get crypto price in multiple currencies

        Args:
            symbol: Crypto symbol (e.g., 'XBT', 'ETH')

        Returns:
            Dictionary {currency: price}

        Example:
            >>> btc_prices = kraken.get_crypto_prices_multi_currency('XBT')
            >>> print(f"BTC USD: ${btc_prices['USD']:,.2f}")
            >>> print(f"BTC EUR: €{btc_prices['EUR']:,.2f}")
        """
        if symbol == 'BTC':
            symbol = 'XBT'

        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD']
        pairs = [f"X{symbol}Z{curr}" for curr in currencies]

        try:
            ticker = self.get_ticker(pairs)

            result = {}
            for pair_name, data in ticker.items():
                # Extract currency from pair name
                for curr in currencies:
                    if curr in pair_name:
                        result[curr] = float(data['c'][0])
                        break

            return result
        except:
            # Fallback to just USD
            return {'USD': self.get_crypto_price(symbol, 'USD')}


# === Singleton Pattern ===

_kraken_instance = None

def get_kraken_collector() -> KrakenSource:
    """
    Get or create Kraken collector instance (singleton pattern)

    Returns:
        KrakenSource instance
    """
    global _kraken_instance

    if _kraken_instance is None:
        _kraken_instance = KrakenSource()

    return _kraken_instance
