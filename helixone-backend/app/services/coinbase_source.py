"""
Coinbase Pro API Data Source
Documentation: https://docs.cloud.coinbase.com/exchange/docs

Free Tier:
- UNLIMITED and FREE for public endpoints
- No API key required for market data
- Rate limit: 10 requests/second (public)

Coverage:
- Major cryptocurrency pairs
- Institutional-grade data
- Real-time prices, orderbooks
- Historical data

Use Cases:
- Institutional crypto prices
- US-based exchange data
- Price comparison with Binance
- Arbitrage opportunities
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime


class CoinbaseSource:
    """
    Coinbase Pro API collector for cryptocurrency data

    Free: Unlimited public data
    Coverage: Major crypto pairs, institutional data
    """

    def __init__(self):
        """Initialize Coinbase Pro API source"""
        self.base_url = "https://api.exchange.coinbase.com"

        # Rate limiting: 10/sec
        self.min_request_interval = 0.1
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

        headers = {
            'Accept': 'application/json',
            'User-Agent': 'HelixOne/1.0'
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Coinbase API request failed: {str(e)}")

    def get_products(self) -> List[Dict]:
        """
        Get all available trading pairs

        Returns:
            List of products with trading info
        """
        return self._make_request('products')

    def get_product(self, product_id: str) -> Dict:
        """
        Get single product information

        Args:
            product_id: Product ID (e.g., 'BTC-USD')

        Returns:
            Product information
        """
        return self._make_request(f'products/{product_id}')

    def get_product_ticker(self, product_id: str) -> Dict:
        """
        Get product ticker (latest trade)

        Args:
            product_id: Product ID (e.g., 'BTC-USD')

        Returns:
            Ticker with price, volume, time

        Example:
            >>> ticker = coinbase.get_product_ticker('BTC-USD')
            >>> print(f"BTC: ${ticker['price']}")
        """
        return self._make_request(f'products/{product_id}/ticker')

    def get_product_stats(self, product_id: str) -> Dict:
        """
        Get 24hr product statistics

        Args:
            product_id: Product ID

        Returns:
            24hr stats (open, high, low, volume, etc.)

        Example:
            >>> stats = coinbase.get_product_stats('BTC-USD')
            >>> print(f"24h Volume: ${stats['volume']}")
        """
        return self._make_request(f'products/{product_id}/stats')

    def get_product_orderbook(self, product_id: str, level: int = 1) -> Dict:
        """
        Get product order book

        Args:
            product_id: Product ID
            level: Detail level (1=best bid/ask, 2=top 50, 3=full)

        Returns:
            Order book data

        Example:
            >>> book = coinbase.get_product_orderbook('BTC-USD', level=2)
            >>> best_bid = book['bids'][0]  # [price, size, num_orders]
        """
        params = {'level': level}
        return self._make_request(f'products/{product_id}/book', params)

    def get_product_trades(self, product_id: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades

        Args:
            product_id: Product ID
            limit: Max trades to return (default 100, max 1000)

        Returns:
            List of recent trades

        Example:
            >>> trades = coinbase.get_product_trades('BTC-USD', limit=10)
        """
        # Note: Coinbase uses pagination, limit is handled via headers
        return self._make_request(f'products/{product_id}/trades')

    def get_product_candles(
        self,
        product_id: str,
        granularity: int = 3600,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[List]:
        """
        Get historic rates (candles/klines)

        Args:
            product_id: Product ID
            granularity: Candle size in seconds (60, 300, 900, 3600, 21600, 86400)
            start: Start time (ISO 8601)
            end: End time (ISO 8601)

        Returns:
            List of candles: [time, low, high, open, close, volume]

        Example:
            >>> candles = coinbase.get_product_candles('BTC-USD', granularity=3600)
            >>> for candle in candles[:5]:
            ...     time, low, high, open, close, volume = candle
            ...     print(f"{datetime.fromtimestamp(time)}: ${close:,.2f}")
        """
        params = {'granularity': granularity}

        if start:
            params['start'] = start
        if end:
            params['end'] = end

        return self._make_request(f'products/{product_id}/candles', params)

    def get_server_time(self) -> Dict:
        """
        Get server time

        Returns:
            Server time info
        """
        return self._make_request('time')

    # === Convenience Methods ===

    def get_crypto_price(self, symbol: str, quote: str = 'USD') -> float:
        """
        Get simple crypto price

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            quote: Quote currency (default 'USD')

        Returns:
            Current price as float

        Example:
            >>> btc_price = coinbase.get_crypto_price('BTC')
            >>> print(f"BTC: ${btc_price:,.2f}")
        """
        product_id = f"{symbol}-{quote}"
        ticker = self.get_product_ticker(product_id)
        return float(ticker['price'])

    def get_crypto_prices(self, symbols: List[str], quote: str = 'USD') -> Dict[str, float]:
        """
        Get multiple crypto prices

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH', 'LTC'])
            quote: Quote currency

        Returns:
            Dictionary {symbol: price}

        Example:
            >>> prices = coinbase.get_crypto_prices(['BTC', 'ETH', 'LTC'])
        """
        result = {}
        for symbol in symbols:
            try:
                price = self.get_crypto_price(symbol, quote)
                result[symbol] = price
            except:
                result[symbol] = None

        return result

    def get_market_summary(self, quote: str = 'USD') -> List[Dict]:
        """
        Get market summary for all products with given quote

        Args:
            quote: Quote currency (e.g., 'USD', 'EUR', 'BTC')

        Returns:
            List of product summaries with prices and volumes
        """
        products = self.get_products()

        # Filter by quote currency
        filtered_products = [
            p for p in products
            if p['quote_currency'] == quote and p['status'] == 'online'
        ]

        summaries = []
        for product in filtered_products[:20]:  # Limit to avoid rate limits
            try:
                product_id = product['id']
                ticker = self.get_product_ticker(product_id)
                stats = self.get_product_stats(product_id)

                summaries.append({
                    'product_id': product_id,
                    'base': product['base_currency'],
                    'quote': product['quote_currency'],
                    'price': float(ticker['price']),
                    'volume_24h': float(stats.get('volume', 0)),
                    'open_24h': float(stats.get('open', 0)),
                    'high_24h': float(stats.get('high', 0)),
                    'low_24h': float(stats.get('low', 0))
                })
            except:
                continue

        # Sort by volume
        summaries.sort(key=lambda x: x['volume_24h'], reverse=True)

        return summaries


# === Singleton Pattern ===

_coinbase_instance = None

def get_coinbase_collector() -> CoinbaseSource:
    """
    Get or create Coinbase collector instance (singleton pattern)

    Returns:
        CoinbaseSource instance
    """
    global _coinbase_instance

    if _coinbase_instance is None:
        _coinbase_instance = CoinbaseSource()

    return _coinbase_instance
