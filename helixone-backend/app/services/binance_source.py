"""
Binance API Data Source
Documentation: https://binance-docs.github.io/apidocs/spot/en/

Free Tier:
- UNLIMITED and FREE for public endpoints
- No API key required for market data
- Rate limit: 1200 requests/minute (public)

Coverage:
- 350+ cryptocurrency pairs
- Real-time prices, volumes, orderbooks
- Largest crypto exchange by volume
- Historical klines (candlestick data)
- 24hr ticker statistics

Use Cases:
- Real-time crypto prices
- Trading volume analysis
- Order book depth
- Price history
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class BinanceSource:
    """
    Binance API collector for cryptocurrency data

    Free: Unlimited public data
    Coverage: 350+ crypto pairs, real-time prices
    """

    def __init__(self):
        """Initialize Binance API source"""
        self.base_url = "https://api.binance.com/api/v3"

        # Rate limiting: 1200/min = 20/sec
        # Be conservative: 10/sec
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
        """
        Make API request

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Binance API request failed: {str(e)}")

    def ping(self) -> bool:
        """
        Test connectivity to Binance API

        Returns:
            True if API is reachable
        """
        try:
            self._make_request('ping')
            return True
        except:
            return False

    def get_server_time(self) -> int:
        """
        Get Binance server time

        Returns:
            Server timestamp in milliseconds
        """
        result = self._make_request('time')
        return result['serverTime']

    def get_ticker_price(self, symbol: Optional[str] = None) -> Dict:
        """
        Get latest price for symbol(s)

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT') or None for all

        Returns:
            Price data

        Example:
            >>> # Single symbol
            >>> btc = binance.get_ticker_price('BTCUSDT')
            >>> print(f"BTC: ${btc['price']}")

            >>> # All symbols
            >>> all_prices = binance.get_ticker_price()
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('ticker/price', params)

    def get_ticker_24hr(self, symbol: Optional[str] = None) -> Dict:
        """
        Get 24hr ticker statistics

        Args:
            symbol: Trading pair or None for all

        Returns:
            24hr statistics including:
            - priceChange, priceChangePercent
            - weightedAvgPrice
            - lastPrice, volume
            - highPrice, lowPrice
            - openPrice, closeTime

        Example:
            >>> stats = binance.get_ticker_24hr('BTCUSDT')
            >>> print(f"24h Change: {stats['priceChangePercent']}%")
            >>> print(f"Volume: ${stats['quoteVolume']}")
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('ticker/24hr', params)

    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book depth

        Args:
            symbol: Trading pair
            limit: Depth (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Order book with bids and asks

        Example:
            >>> book = binance.get_orderbook('BTCUSDT', limit=10)
            >>> best_bid = book['bids'][0]  # [price, quantity]
            >>> best_ask = book['asks'][0]
        """
        params = {
            'symbol': symbol,
            'limit': limit
        }

        return self._make_request('depth', params)

    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get recent trades

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)

        Returns:
            List of recent trades

        Example:
            >>> trades = binance.get_recent_trades('BTCUSDT', limit=100)
            >>> for trade in trades[:5]:
            ...     print(f"{trade['price']} x {trade['qty']}")
        """
        params = {
            'symbol': symbol,
            'limit': min(limit, 1000)
        }

        return self._make_request('trades', params)

    def get_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        Get candlestick/kline data

        Args:
            symbol: Trading pair
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M)
            limit: Number of candles (max 1000)
            start_time: Start timestamp (milliseconds)
            end_time: End timestamp (milliseconds)

        Returns:
            List of klines: [
                openTime, open, high, low, close, volume,
                closeTime, quoteVolume, trades, takerBuyBase,
                takerBuyQuote, ignore
            ]

        Example:
            >>> # Last 24 hours, 1h candles
            >>> klines = binance.get_klines('BTCUSDT', '1h', limit=24)
            >>> for k in klines:
            ...     open_time, o, h, l, c, v = k[:6]
            ...     print(f"{datetime.fromtimestamp(open_time/1000)}: {c}")
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._make_request('klines', params)

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Get exchange information (symbols, trading rules)

        Args:
            symbol: Specific symbol or None for all

        Returns:
            Exchange info including symbols and trading rules

        Example:
            >>> info = binance.get_exchange_info('BTCUSDT')
            >>> symbol_info = info['symbols'][0]
            >>> print(f"Base: {symbol_info['baseAsset']}")
            >>> print(f"Quote: {symbol_info['quoteAsset']}")
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        return self._make_request('exchangeInfo', params)

    # === Convenience Methods ===

    def get_top_pairs_by_volume(self, quote: str = 'USDT', limit: int = 10) -> List[Dict]:
        """
        Get top trading pairs by 24h volume

        Args:
            quote: Quote currency (USDT, BTC, ETH)
            limit: Number of pairs to return

        Returns:
            List of top pairs sorted by volume
        """
        # Get all 24hr tickers
        all_tickers = self.get_ticker_24hr()

        # Filter by quote currency
        filtered = [
            t for t in all_tickers
            if t['symbol'].endswith(quote)
        ]

        # Sort by quote volume (descending)
        sorted_tickers = sorted(
            filtered,
            key=lambda x: float(x['quoteVolume']),
            reverse=True
        )

        return sorted_tickers[:limit]

    def get_crypto_price_simple(self, symbols: List[str], quote: str = 'USDT') -> Dict[str, float]:
        """
        Get simple prices for multiple cryptos

        Args:
            symbols: List of base symbols (e.g., ['BTC', 'ETH', 'BNB'])
            quote: Quote currency (default USDT)

        Returns:
            Dictionary {symbol: price}

        Example:
            >>> prices = binance.get_crypto_price_simple(['BTC', 'ETH', 'BNB'])
            >>> print(f"BTC: ${prices['BTC']:,.2f}")
        """
        all_prices = self.get_ticker_price()

        result = {}
        for symbol in symbols:
            pair = f"{symbol}{quote}"
            for price_data in all_prices:
                if price_data['symbol'] == pair:
                    result[symbol] = float(price_data['price'])
                    break

        return result

    def get_market_overview(self) -> Dict:
        """
        Get overall market overview

        Returns:
            Dictionary with market statistics
        """
        # Get top pairs
        top_usdt = self.get_top_pairs_by_volume('USDT', limit=5)

        # Calculate total volume (top 100 USDT pairs)
        all_usdt = self.get_top_pairs_by_volume('USDT', limit=100)
        total_volume = sum(float(t['quoteVolume']) for t in all_usdt)

        # Get BTC dominance indicator (BTC volume vs total)
        btc_ticker = self.get_ticker_24hr('BTCUSDT')
        btc_volume = float(btc_ticker['quoteVolume'])

        return {
            'total_24h_volume_top100': total_volume,
            'btc_24h_volume': btc_volume,
            'btc_volume_dominance': (btc_volume / total_volume * 100) if total_volume > 0 else 0,
            'top_5_pairs': [
                {
                    'symbol': t['symbol'],
                    'price': float(t['lastPrice']),
                    'volume_24h': float(t['quoteVolume']),
                    'change_24h': float(t['priceChangePercent'])
                }
                for t in top_usdt[:5]
            ]
        }


# === Singleton Pattern ===

_binance_instance = None

def get_binance_collector() -> BinanceSource:
    """
    Get or create Binance collector instance (singleton pattern)

    Returns:
        BinanceSource instance
    """
    global _binance_instance

    if _binance_instance is None:
        _binance_instance = BinanceSource()

    return _binance_instance
