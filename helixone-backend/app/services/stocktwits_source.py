"""
StockTwits API Data Source - Trading Sentiment
Documentation: https://api.stocktwits.com/developers/docs/api

Features:
- Real-time trading sentiment (Bullish/Bearish)
- Ticker-specific message streams
- Trending tickers
- User watchlists
- Community sentiment gauge
- Free tier: ~200 requests/hour (no API key required)

Coverage:
- 3M+ active traders
- Real-time messages with sentiment tags
- Professional traders & institutions
- Direct ticker tagging ($TICKER)

Use Cases:
- Real-time sentiment tracking
- Bullish/Bearish ratio analysis
- Trending stock detection
- Community mood gauge
- Complement to Reddit sentiment
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter


class StockTwitsSource:
    """
    StockTwits API collector for trading sentiment

    Free: ~200 req/hour (no API key required)
    Coverage: 3M+ traders, real-time messages
    Data: Messages, sentiment (Bullish/Bearish), trends
    """

    def __init__(self):
        """Initialize StockTwits API source"""
        self.base_url = "https://api.stocktwits.com/api/2"

        # Rate limiting (conservative: ~200 req/hour = 1 req per 18 seconds)
        self.min_request_interval = 2.0  # 2 seconds between requests
        self.last_request_time = 0

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HelixOne/1.0.0 (Financial Data Aggregator)',
            'Accept': 'application/json'
        })

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make API request with error handling

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters

        Returns:
            JSON response or None on error
        """
        self._rate_limit()

        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("⚠️  StockTwits rate limit reached")
                return None
            else:
                print(f"⚠️  StockTwits API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"⚠️  StockTwits request error: {str(e)[:60]}")
            return None

    # === TICKER SENTIMENT ===

    def get_ticker_stream(
        self,
        symbol: str,
        limit: int = 30
    ) -> List[Dict]:
        """
        Get message stream for a specific ticker

        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'TSLA')
            limit: Number of messages (max 30)

        Returns:
            List of messages with sentiment

        Example:
            >>> st = StockTwitsSource()
            >>> messages = st.get_ticker_stream('AAPL', limit=20)
            >>> for msg in messages[:5]:
            ...     print(f"{msg['user']}: {msg['sentiment']} - {msg['body'][:50]}")
        """
        endpoint = f"streams/symbol/{symbol.upper()}.json"
        params = {'limit': min(limit, 30)}

        data = self._make_request(endpoint, params)

        if not data:
            print(f"⚠️  No data returned for {symbol}")
            return []

        if 'messages' not in data:
            print(f"⚠️  No 'messages' key in response for {symbol}")
            print(f"   Keys in response: {list(data.keys()) if data else 'None'}")
            return []

        messages = []
        for msg in data['messages']:
            try:
                # Sentiment extraction (peut être dans entities.sentiment ou directement dans le message)
                sentiment = None
                if 'entities' in msg and msg['entities'] and 'sentiment' in msg['entities']:
                    if msg['entities']['sentiment']:
                        sentiment = msg['entities']['sentiment'].get('basic')

                messages.append({
                    'id': msg.get('id'),
                    'body': msg.get('body', ''),
                    'created_at': datetime.strptime(msg['created_at'], '%Y-%m-%dT%H:%M:%SZ') if 'created_at' in msg else datetime.now(),
                    'user': msg.get('user', {}).get('username', 'unknown'),
                    'user_followers': msg.get('user', {}).get('followers', 0),
                    'sentiment': sentiment,
                    'likes': msg.get('likes', {}).get('total', 0) if isinstance(msg.get('likes'), dict) else 0,
                    'symbols': [s['symbol'] for s in msg.get('symbols', [])]
                })
            except Exception as e:
                # Skip messages that fail to parse
                print(f"⚠️  Skipping message: {str(e)[:50]}")
                continue

        return messages

    def get_ticker_sentiment(
        self,
        symbol: str,
        limit: int = 30
    ) -> Dict:
        """
        Get sentiment analysis for a specific ticker

        Args:
            symbol: Stock ticker
            limit: Number of messages to analyze

        Returns:
            {
                'symbol': str,
                'total_messages': int,
                'bullish': int,
                'bearish': int,
                'neutral': int,
                'bullish_pct': float,
                'bearish_pct': float,
                'sentiment_ratio': float,  # bullish / bearish
                'sentiment': str,  # 'bullish', 'bearish', 'neutral'
                'avg_likes': float,
                'top_message': dict
            }

        Example:
            >>> sentiment = st.get_ticker_sentiment('TSLA')
            >>> print(f"TSLA: {sentiment['sentiment']} ({sentiment['bullish_pct']:.1f}% bullish)")
        """
        messages = self.get_ticker_stream(symbol, limit)

        if not messages:
            return {
                'symbol': symbol.upper(),
                'total_messages': 0,
                'bullish': 0,
                'bearish': 0,
                'neutral': 0,
                'bullish_pct': 0,
                'bearish_pct': 0,
                'sentiment_ratio': 0,
                'sentiment': 'neutral',
                'avg_likes': 0,
                'top_message': None
            }

        # Count sentiment
        bullish = len([m for m in messages if m['sentiment'] == 'Bullish'])
        bearish = len([m for m in messages if m['sentiment'] == 'Bearish'])
        neutral = len(messages) - bullish - bearish

        total = len(messages)
        bullish_pct = (bullish / total) * 100 if total > 0 else 0
        bearish_pct = (bearish / total) * 100 if total > 0 else 0

        # Sentiment ratio (avoid division by zero)
        sentiment_ratio = bullish / bearish if bearish > 0 else (bullish if bullish > 0 else 1)

        # Overall sentiment
        if bullish > bearish * 1.5:
            sentiment = 'bullish'
        elif bearish > bullish * 1.5:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        # Average likes
        avg_likes = sum(m['likes'] for m in messages) / total if total > 0 else 0

        # Top message (most likes)
        top_message = max(messages, key=lambda x: x['likes']) if messages else None

        return {
            'symbol': symbol.upper(),
            'total_messages': total,
            'bullish': bullish,
            'bearish': bearish,
            'neutral': neutral,
            'bullish_pct': bullish_pct,
            'bearish_pct': bearish_pct,
            'sentiment_ratio': sentiment_ratio,
            'sentiment': sentiment,
            'avg_likes': avg_likes,
            'top_message': {
                'user': top_message['user'],
                'body': top_message['body'],
                'sentiment': top_message['sentiment'],
                'likes': top_message['likes']
            } if top_message else None
        }

    # === TRENDING ===

    def get_trending_tickers(self, limit: int = 30) -> List[Dict]:
        """
        Get trending tickers on StockTwits

        Args:
            limit: Number of tickers (max 30)

        Returns:
            List of trending tickers with metadata

        Example:
            >>> trending = st.get_trending_tickers(limit=10)
            >>> for ticker in trending[:5]:
            ...     print(f"{ticker['symbol']}: {ticker['watchlist_count']} watchers")
        """
        endpoint = "trending/symbols.json"
        params = {'limit': min(limit, 30)}

        data = self._make_request(endpoint, params)

        if not data or 'symbols' not in data:
            return []

        trending = []
        for symbol in data['symbols']:
            trending.append({
                'symbol': symbol['symbol'],
                'title': symbol['title'],
                'watchlist_count': symbol.get('watchlist_count', 0),
                'exchange': symbol.get('exchange'),
                'sector': symbol.get('sector'),
                'industry': symbol.get('industry')
            })

        return trending

    def get_trending_with_sentiment(self, limit: int = 10) -> List[Dict]:
        """
        Get trending tickers with their sentiment analysis

        Args:
            limit: Number of tickers to analyze

        Returns:
            List of tickers with sentiment data

        Example:
            >>> trending = st.get_trending_with_sentiment(5)
            >>> for t in trending:
            ...     print(f"{t['symbol']}: {t['sentiment']} ({t['bullish_pct']:.1f}% bullish)")
        """
        trending_tickers = self.get_trending_tickers(limit)

        results = []
        for ticker in trending_tickers:
            sentiment = self.get_ticker_sentiment(ticker['symbol'], limit=30)

            results.append({
                'symbol': ticker['symbol'],
                'title': ticker['title'],
                'watchlist_count': ticker['watchlist_count'],
                'sentiment': sentiment['sentiment'],
                'bullish_pct': sentiment['bullish_pct'],
                'bearish_pct': sentiment['bearish_pct'],
                'total_messages': sentiment['total_messages']
            })

        return results

    # === WATCHLIST ===

    def get_multiple_sentiments(
        self,
        symbols: List[str],
        limit_per_symbol: int = 30
    ) -> Dict[str, Dict]:
        """
        Get sentiment for multiple tickers

        Args:
            symbols: List of ticker symbols
            limit_per_symbol: Messages per ticker

        Returns:
            {symbol: sentiment_data}

        Example:
            >>> symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD']
            >>> sentiments = st.get_multiple_sentiments(symbols)
            >>> for symbol, data in sentiments.items():
            ...     print(f"{symbol}: {data['sentiment']} ({data['bullish_pct']:.1f}%)")
        """
        results = {}

        for symbol in symbols:
            sentiment = self.get_ticker_sentiment(symbol, limit_per_symbol)
            results[symbol.upper()] = sentiment

        return results

    # === MARKET OVERVIEW ===

    def get_market_sentiment_overview(self) -> Dict:
        """
        Get overall market sentiment from trending tickers

        Returns:
            {
                'trending_tickers': list,
                'overall_bullish_pct': float,
                'overall_bearish_pct': float,
                'overall_sentiment': str,
                'most_bullish': dict,
                'most_bearish': dict,
                'timestamp': datetime
            }

        Example:
            >>> overview = st.get_market_sentiment_overview()
            >>> print(f"Market: {overview['overall_sentiment']} ({overview['overall_bullish_pct']:.1f}% bullish)")
        """
        trending = self.get_trending_with_sentiment(limit=10)

        if not trending:
            return {
                'trending_tickers': [],
                'overall_bullish_pct': 0,
                'overall_bearish_pct': 0,
                'overall_sentiment': 'neutral',
                'most_bullish': None,
                'most_bearish': None,
                'timestamp': datetime.now()
            }

        # Calculate overall sentiment
        total_bullish = sum(t['bullish_pct'] for t in trending)
        total_bearish = sum(t['bearish_pct'] for t in trending)
        count = len(trending)

        avg_bullish = total_bullish / count
        avg_bearish = total_bearish / count

        # Overall sentiment
        if avg_bullish > avg_bearish * 1.3:
            overall_sentiment = 'bullish'
        elif avg_bearish > avg_bullish * 1.3:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'

        # Most bullish/bearish
        most_bullish = max(trending, key=lambda x: x['bullish_pct'])
        most_bearish = max(trending, key=lambda x: x['bearish_pct'])

        return {
            'trending_tickers': trending,
            'overall_bullish_pct': avg_bullish,
            'overall_bearish_pct': avg_bearish,
            'overall_sentiment': overall_sentiment,
            'most_bullish': most_bullish,
            'most_bearish': most_bearish,
            'timestamp': datetime.now()
        }


# === SINGLETON PATTERN ===

_stocktwits_instance = None

def get_stocktwits_collector() -> StockTwitsSource:
    """
    Get or create StockTwits collector instance (singleton)

    Returns:
        StockTwitsSource instance
    """
    global _stocktwits_instance

    if _stocktwits_instance is None:
        _stocktwits_instance = StockTwitsSource()

    return _stocktwits_instance
