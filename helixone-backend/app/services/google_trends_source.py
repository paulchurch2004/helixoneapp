"""
Google Trends API Data Source
Documentation: https://pypi.org/project/pytrends/

Features:
- Search interest over time
- Related queries
- Trending searches
- Regional interest
- Interest by category
- 100% FREE - No API key required

Coverage:
- All search terms
- All regions/countries
- Historical data (2004+)
- Real-time trending

Use Cases:
- Retail interest tracking
- Hype cycle detection
- Geographic sentiment
- Search momentum
- Public interest gauge
"""

from pytrends.request import TrendReq
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd


class GoogleTrendsSource:
    """
    Google Trends collector for search interest data

    Free: Unlimited (rate limited by Google)
    Coverage: All keywords, all regions
    Data: Interest over time, related queries, trending
    """

    def __init__(self):
        """Initialize Google Trends API"""
        # Initialize pytrends
        # Note: No API key required!
        self.pytrends = TrendReq(hl='en-US', tz=360)

        # Rate limiting
        self.min_request_interval = 2.0  # Be respectful
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    # === INTEREST OVER TIME ===

    def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = 'today 3-m',
        geo: str = ''
    ) -> pd.DataFrame:
        """
        Get search interest over time

        Args:
            keywords: List of keywords (max 5)
            timeframe: Time range ('today 1-m', 'today 3-m', 'today 12-m', 'today 5-y', 'all')
            geo: Country code ('' for worldwide, 'US', 'GB', etc.)

        Returns:
            DataFrame with interest over time (0-100 scale)

        Example:
            >>> trends = GoogleTrendsSource()
            >>> df = trends.get_interest_over_time(['Bitcoin', 'Ethereum'], 'today 3-m')
            >>> print(df.tail())
        """
        self._rate_limit()

        try:
            # Build payload
            self.pytrends.build_payload(
                keywords,
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop=''
            )

            # Get data
            df = self.pytrends.interest_over_time()

            # Remove 'isPartial' column if exists
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])

            return df

        except Exception as e:
            print(f"Google Trends error: {str(e)[:50]}")
            return pd.DataFrame()

    def get_current_interest(
        self,
        keywords: List[str],
        geo: str = ''
    ) -> Dict[str, float]:
        """
        Get current search interest (last data point)

        Args:
            keywords: List of keywords
            geo: Country code

        Returns:
            {keyword: interest_score}

        Example:
            >>> interest = trends.get_current_interest(['Tesla', 'Apple'])
            >>> print(f"Tesla interest: {interest['Tesla']}/100")
        """
        df = self.get_interest_over_time(keywords, 'now 7-d', geo)

        if df.empty:
            return {k: 0 for k in keywords}

        # Get last row
        latest = df.iloc[-1]

        return {k: float(latest[k]) if k in latest else 0 for k in keywords}

    # === RELATED QUERIES ===

    def get_related_queries(
        self,
        keyword: str,
        geo: str = ''
    ) -> Dict[str, pd.DataFrame]:
        """
        Get related queries for a keyword

        Args:
            keyword: Search keyword
            geo: Country code

        Returns:
            {
                'top': DataFrame of top related queries,
                'rising': DataFrame of rising related queries
            }

        Example:
            >>> related = trends.get_related_queries('Bitcoin')
            >>> print("Top related:")
            >>> print(related['top'].head())
            >>> print("\\nRising:")
            >>> print(related['rising'].head())
        """
        self._rate_limit()

        try:
            self.pytrends.build_payload([keyword], geo=geo, timeframe='today 3-m')
            related = self.pytrends.related_queries()

            return {
                'top': related[keyword]['top'] if keyword in related else pd.DataFrame(),
                'rising': related[keyword]['rising'] if keyword in related else pd.DataFrame()
            }

        except Exception as e:
            print(f"Related queries error: {str(e)[:50]}")
            return {'top': pd.DataFrame(), 'rising': pd.DataFrame()}

    # === TRENDING SEARCHES ===

    def get_trending_searches(self, country: str = 'united_states') -> List[str]:
        """
        Get current trending searches in a country

        Args:
            country: Country name ('united_states', 'united_kingdom', 'japan', etc.)

        Returns:
            List of trending search terms

        Example:
            >>> trending = trends.get_trending_searches('united_states')
            >>> for i, term in enumerate(trending[:10], 1):
            ...     print(f"{i}. {term}")
        """
        self._rate_limit()

        try:
            df = self.pytrends.trending_searches(pn=country)
            return df[0].tolist() if not df.empty else []

        except Exception as e:
            print(f"Trending searches error: {str(e)[:50]}")
            return []

    # === REGIONAL INTEREST ===

    def get_interest_by_region(
        self,
        keyword: str,
        resolution: str = 'COUNTRY',
        inc_low_vol: bool = False
    ) -> pd.DataFrame:
        """
        Get interest by geographic region

        Args:
            keyword: Search keyword
            resolution: 'COUNTRY', 'REGION', 'CITY', 'DMA'
            inc_low_vol: Include low volume regions

        Returns:
            DataFrame with interest by region

        Example:
            >>> interest = trends.get_interest_by_region('Tesla', 'COUNTRY')
            >>> print(interest.sort_values(ascending=False).head(10))
        """
        self._rate_limit()

        try:
            self.pytrends.build_payload([keyword], timeframe='today 3-m')
            df = self.pytrends.interest_by_region(
                resolution=resolution,
                inc_low_vol=inc_low_vol,
                inc_geo_code=False
            )

            return df

        except Exception as e:
            print(f"Interest by region error: {str(e)[:50]}")
            return pd.DataFrame()

    # === STOCK/CRYPTO SPECIFIC ===

    def get_stock_interest(
        self,
        ticker: str,
        company_name: str,
        timeframe: str = 'today 3-m'
    ) -> pd.DataFrame:
        """
        Get search interest for a stock

        Args:
            ticker: Stock ticker (e.g., 'TSLA')
            company_name: Company name (e.g., 'Tesla')
            timeframe: Time range

        Returns:
            DataFrame with interest for ticker and company name

        Example:
            >>> df = trends.get_stock_interest('AAPL', 'Apple')
            >>> print(df.tail())
        """
        keywords = [ticker, company_name]
        return self.get_interest_over_time(keywords, timeframe)

    def get_crypto_interest(
        self,
        crypto_name: str,
        timeframe: str = 'today 3-m'
    ) -> pd.DataFrame:
        """
        Get search interest for a cryptocurrency

        Args:
            crypto_name: Crypto name (e.g., 'Bitcoin', 'Ethereum')
            timeframe: Time range

        Returns:
            DataFrame with crypto interest

        Example:
            >>> df = trends.get_crypto_interest('Bitcoin', 'today 12-m')
            >>> max_interest = df['Bitcoin'].max()
            >>> print(f"Peak interest: {max_interest}/100")
        """
        return self.get_interest_over_time([crypto_name], timeframe)

    def compare_cryptos(
        self,
        crypto_names: List[str],
        timeframe: str = 'today 3-m'
    ) -> pd.DataFrame:
        """
        Compare search interest for multiple cryptos

        Args:
            crypto_names: List of crypto names (max 5)
            timeframe: Time range

        Returns:
            DataFrame comparing interest

        Example:
            >>> df = trends.compare_cryptos(['Bitcoin', 'Ethereum', 'Solana'])
            >>> print(df.tail())
        """
        if len(crypto_names) > 5:
            crypto_names = crypto_names[:5]

        return self.get_interest_over_time(crypto_names, timeframe)

    # === HYPE DETECTION ===

    def detect_hype_cycle(
        self,
        keyword: str,
        threshold: float = 50.0
    ) -> Dict:
        """
        Detect if a keyword is in a hype cycle

        Args:
            keyword: Search term
            threshold: Interest threshold to consider "hype"

        Returns:
            {
                'keyword': str,
                'current_interest': float,
                'avg_interest': float,
                'max_interest': float,
                'is_trending': bool,
                'trend_direction': str ('up', 'down', 'stable')
            }

        Example:
            >>> hype = trends.detect_hype_cycle('GameStop')
            >>> if hype['is_trending']:
            ...     print(f"{hype['keyword']} is trending {hype['trend_direction']}!")
        """
        df = self.get_interest_over_time([keyword], 'today 3-m')

        if df.empty or keyword not in df.columns:
            return {
                'keyword': keyword,
                'current_interest': 0,
                'avg_interest': 0,
                'max_interest': 0,
                'is_trending': False,
                'trend_direction': 'stable'
            }

        current = float(df[keyword].iloc[-1])
        avg = float(df[keyword].mean())
        max_val = float(df[keyword].max())

        # Check last 7 days trend
        last_week = df[keyword].iloc[-7:] if len(df) >= 7 else df[keyword]
        trend = 'stable'

        if len(last_week) > 1:
            if last_week.iloc[-1] > last_week.iloc[0] * 1.2:
                trend = 'up'
            elif last_week.iloc[-1] < last_week.iloc[0] * 0.8:
                trend = 'down'

        is_trending = current >= threshold

        return {
            'keyword': keyword,
            'current_interest': current,
            'avg_interest': avg,
            'max_interest': max_val,
            'is_trending': is_trending,
            'trend_direction': trend,
            'hype_score': (current / avg) if avg > 0 else 0
        }

    # === SUMMARY ===

    def get_market_interest_summary(
        self,
        tickers: List[str],
        crypto_names: List[str]
    ) -> Dict:
        """
        Get overall market interest summary

        Args:
            tickers: List of stock tickers
            crypto_names: List of crypto names

        Returns:
            Summary dict

        Example:
            >>> summary = trends.get_market_interest_summary(
            ...     ['TSLA', 'AAPL'],
            ...     ['Bitcoin', 'Ethereum']
            ... )
        """
        # Get interest for stocks
        stock_interest = {}
        for ticker in tickers[:3]:  # Limit to avoid rate limits
            interest = self.get_current_interest([ticker])
            stock_interest[ticker] = interest.get(ticker, 0)

        # Get interest for crypto
        crypto_interest = {}
        for crypto in crypto_names[:3]:
            interest = self.get_current_interest([crypto])
            crypto_interest[crypto] = interest.get(crypto, 0)

        return {
            'stock_interest': stock_interest,
            'crypto_interest': crypto_interest,
            'timestamp': datetime.now()
        }


# === SINGLETON PATTERN ===

_trends_instance = None

def get_google_trends_collector() -> GoogleTrendsSource:
    """
    Get or create Google Trends collector instance (singleton)

    Returns:
        GoogleTrendsSource instance
    """
    global _trends_instance

    if _trends_instance is None:
        _trends_instance = GoogleTrendsSource()

    return _trends_instance
