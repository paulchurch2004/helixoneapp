"""
CoinCap API Data Source
Documentation: https://docs.coincap.io/

Free Tier:
- UNLIMITED and FREE
- No API key required
- Rate limit: 200 requests/minute (very generous)

Coverage:
- 2000+ cryptocurrencies
- Real-time prices aggregated from multiple exchanges
- Market data, historical data
- Exchange rankings

Use Cases:
- Comprehensive crypto coverage
- Average prices across exchanges
- Alternative to single exchange data
- Market overview and rankings
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime


class CoinCapSource:
    """
    CoinCap API collector for cryptocurrency data

    Free: Unlimited, 200 req/min
    Coverage: 2000+ cryptos, aggregated exchange data
    """

    def __init__(self):
        """Initialize CoinCap API source"""
        self.base_url = "https://api.coincap.io/v2"

        # Rate limiting: 200/min = 3.3/sec
        # Be conservative: 1/sec
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

        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip'
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"CoinCap API request failed: {str(e)}")

    def get_assets(
        self,
        search: Optional[str] = None,
        ids: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get assets (cryptocurrencies)

        Args:
            search: Search by asset symbol or name
            ids: Filter by asset IDs (e.g., ['bitcoin', 'ethereum'])
            limit: Number of results (max 2000)
            offset: Pagination offset

        Returns:
            List of assets with market data

        Example:
            >>> # Top 10 by market cap
            >>> top10 = coincap.get_assets(limit=10)
            >>> for asset in top10:
            ...     print(f"{asset['name']}: ${asset['priceUsd']}")
        """
        params = {
            'limit': min(limit, 2000),
            'offset': offset
        }

        if search:
            params['search'] = search

        if ids:
            params['ids'] = ','.join(ids)

        result = self._make_request('assets', params)
        return result.get('data', [])

    def get_asset(self, asset_id: str) -> Dict:
        """
        Get single asset by ID

        Args:
            asset_id: Asset ID (e.g., 'bitcoin', 'ethereum')

        Returns:
            Asset data

        Example:
            >>> btc = coincap.get_asset('bitcoin')
            >>> print(f"BTC: ${btc['priceUsd']}")
        """
        result = self._make_request(f'assets/{asset_id}')
        return result.get('data', {})

    def get_asset_history(
        self,
        asset_id: str,
        interval: str = 'd1',
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Dict]:
        """
        Get asset price history

        Args:
            asset_id: Asset ID
            interval: Time interval (m1, m5, m15, m30, h1, h2, h6, h12, d1)
            start: Start timestamp (milliseconds)
            end: End timestamp (milliseconds)

        Returns:
            List of price history points

        Example:
            >>> # Last 7 days
            >>> history = coincap.get_asset_history('bitcoin', interval='d1')
        """
        params = {'interval': interval}

        if start:
            params['start'] = start
        if end:
            params['end'] = end

        result = self._make_request(f'assets/{asset_id}/history', params)
        return result.get('data', [])

    def get_asset_markets(self, asset_id: str, limit: int = 20) -> List[Dict]:
        """
        Get markets for an asset

        Args:
            asset_id: Asset ID
            limit: Number of markets to return

        Returns:
            List of markets trading this asset

        Example:
            >>> markets = coincap.get_asset_markets('bitcoin', limit=10)
            >>> for market in markets:
            ...     print(f"{market['exchangeId']}: ${market['priceUsd']}")
        """
        params = {'limit': limit}

        result = self._make_request(f'assets/{asset_id}/markets', params)
        return result.get('data', [])

    def get_rates(self, ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Get exchange rates

        Args:
            ids: Filter by rate IDs

        Returns:
            List of exchange rates

        Example:
            >>> rates = coincap.get_rates()
            >>> # Find USD rate
            >>> usd_rate = next(r for r in rates if r['id'] == 'united-states-dollar')
        """
        params = {}
        if ids:
            params['ids'] = ','.join(ids)

        result = self._make_request('rates', params)
        return result.get('data', [])

    def get_rate(self, rate_id: str) -> Dict:
        """Get single exchange rate"""
        result = self._make_request(f'rates/{rate_id}')
        return result.get('data', {})

    def get_exchanges(self, limit: int = 100) -> List[Dict]:
        """
        Get exchanges

        Args:
            limit: Number of exchanges

        Returns:
            List of exchanges with volume data

        Example:
            >>> exchanges = coincap.get_exchanges(limit=10)
            >>> for ex in exchanges:
            ...     print(f"{ex['name']}: ${ex['volumeUsd']} (24h)")
        """
        params = {'limit': limit}

        result = self._make_request('exchanges', params)
        return result.get('data', [])

    def get_exchange(self, exchange_id: str) -> Dict:
        """Get single exchange data"""
        result = self._make_request(f'exchanges/{exchange_id}')
        return result.get('data', {})

    def get_markets(
        self,
        exchange_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get markets

        Args:
            exchange_id: Filter by exchange
            limit: Number of markets

        Returns:
            List of trading markets

        Example:
            >>> # Binance markets
            >>> binance_markets = coincap.get_markets(exchange_id='binance', limit=10)
        """
        params = {'limit': limit}

        if exchange_id:
            params['exchangeId'] = exchange_id

        result = self._make_request('markets', params)
        return result.get('data', [])

    # === Convenience Methods ===

    def get_crypto_price(self, symbol: str) -> float:
        """
        Get simple crypto price by symbol

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')

        Returns:
            Current price in USD

        Example:
            >>> btc_price = coincap.get_crypto_price('BTC')
        """
        # Search by symbol
        assets = self.get_assets(search=symbol, limit=1)

        if assets and len(assets) > 0:
            return float(assets[0]['priceUsd'])

        raise Exception(f"Asset not found: {symbol}")

    def get_crypto_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get multiple crypto prices

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH', 'ADA'])

        Returns:
            Dictionary {symbol: price}

        Example:
            >>> prices = coincap.get_crypto_prices(['BTC', 'ETH', 'ADA'])
        """
        # Map common symbols to CoinCap IDs
        symbol_to_id = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binance-coin',
            'ADA': 'cardano',
            'SOL': 'solana',
            'XRP': 'ripple',
            'DOT': 'polkadot',
            'DOGE': 'dogecoin',
            'AVAX': 'avalanche',
            'MATIC': 'polygon',
            'LTC': 'litecoin',
            'LINK': 'chainlink'
        }

        ids = [symbol_to_id.get(s, s.lower()) for s in symbols]
        assets = self.get_assets(ids=ids)

        result = {}
        for asset in assets:
            # Find matching symbol
            for symbol in symbols:
                if asset['symbol'].upper() == symbol.upper():
                    result[symbol] = float(asset['priceUsd'])
                    break

        return result

    def get_market_summary(self, limit: int = 10) -> Dict:
        """
        Get market summary

        Args:
            limit: Number of top coins

        Returns:
            Market summary with top coins

        Example:
            >>> summary = coincap.get_market_summary(limit=10)
            >>> print(f"Total market cap: ${summary['total_market_cap']:,.0f}")
        """
        top_assets = self.get_assets(limit=limit)

        total_market_cap = sum(float(a['marketCapUsd'] or 0) for a in top_assets)
        total_volume_24h = sum(float(a['volumeUsd24Hr'] or 0) for a in top_assets)

        return {
            'total_market_cap': total_market_cap,
            'total_volume_24h': total_volume_24h,
            'top_assets': [
                {
                    'rank': int(a['rank']),
                    'symbol': a['symbol'],
                    'name': a['name'],
                    'price_usd': float(a['priceUsd']),
                    'market_cap_usd': float(a['marketCapUsd'] or 0),
                    'volume_24h_usd': float(a['volumeUsd24Hr'] or 0),
                    'change_24h': float(a['changePercent24Hr'] or 0)
                }
                for a in top_assets
            ]
        }


# === Singleton Pattern ===

_coincap_instance = None

def get_coincap_collector() -> CoinCapSource:
    """
    Get or create CoinCap collector instance (singleton pattern)

    Returns:
        CoinCapSource instance
    """
    global _coincap_instance

    if _coincap_instance is None:
        _coincap_instance = CoinCapSource()

    return _coincap_instance
