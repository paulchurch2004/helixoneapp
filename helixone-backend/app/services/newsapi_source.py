"""
NewsAPI.org Data Source
Documentation: https://newsapi.org/docs

Free Tier Limits:
- 100 requests per day
- 1000 requests per month
- News up to 1 month old
- 80,000+ sources worldwide

Paid Tiers:
- Developer: $449/month - 250 req/day, news up to 1 year old
- Business: Custom pricing

API Key: Required (free registration)
Coverage: 80,000+ news sources, 150+ countries, multiple languages
"""

import os
import time
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from functools import lru_cache


class NewsAPISource:
    """
    NewsAPI.org collector for financial and general news

    Free Tier: 100 requests/day
    Coverage: 80,000+ sources, business news, technology, finance
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI source

        Args:
            api_key: NewsAPI key (optional, will read from env if not provided)
        """
        self.api_key = api_key or os.getenv('NEWSAPI_API_KEY')

        if not self.api_key:
            raise ValueError("NewsAPI API key is required. Get one at https://newsapi.org/register")

        self.base_url = "https://newsapi.org/v2"

        # Rate limiting: 100 requests/day = ~4 req/hour = 1 req/15min
        # We'll be conservative: 1 request every 60 seconds
        self.min_request_interval = 60.0
        self.last_request_time = 0

        # Default parameters
        self.default_language = 'en'
        self.default_page_size = 20
        self.max_page_size = 100

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make API request with rate limiting

        Args:
            endpoint: API endpoint (e.g., 'top-headlines', 'everything')
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        # Add API key to params
        params['apiKey'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if data.get('status') != 'ok':
                error_message = data.get('message', 'Unknown error')
                error_code = data.get('code', 'unknown')
                raise Exception(f"NewsAPI error ({error_code}): {error_message}")

            return data

        except requests.exceptions.RequestException as e:
            raise Exception(f"NewsAPI request failed: {str(e)}")

    def get_top_headlines(
        self,
        category: Optional[str] = None,
        sources: Optional[List[str]] = None,
        q: Optional[str] = None,
        country: str = 'us',
        page_size: int = 20,
        page: int = 1
    ) -> Dict:
        """
        Get top headlines

        Args:
            category: business, entertainment, general, health, science, sports, technology
            sources: List of source IDs (e.g., ['bloomberg', 'reuters'])
            q: Keywords/phrase to search for in article title and body
            country: 2-letter ISO 3166-1 code (us, gb, de, fr, etc.)
            page_size: Number of results per page (max 100)
            page: Page number

        Returns:
            Dictionary with articles and metadata

        Example:
            >>> news = newsapi.get_top_headlines(category='business', country='us')
            >>> for article in news['articles']:
            ...     print(article['title'])
        """
        params = {
            'pageSize': min(page_size, self.max_page_size),
            'page': page
        }

        if category:
            params['category'] = category

        if sources:
            # Can't combine sources with country/category
            params['sources'] = ','.join(sources)
            params.pop('category', None)
        else:
            params['country'] = country

        if q:
            params['q'] = q

        return self._make_request('top-headlines', params)

    def get_everything(
        self,
        q: Optional[str] = None,
        sources: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page_size: int = 20,
        page: int = 1
    ) -> Dict:
        """
        Search all articles (more comprehensive than top-headlines)

        Args:
            q: Keywords to search for (supports AND/OR/NOT operators)
            sources: List of source IDs
            domains: List of domains to restrict search (e.g., ['bloomberg.com', 'reuters.com'])
            exclude_domains: Domains to exclude
            from_date: Oldest article date (YYYY-MM-DD or datetime)
            to_date: Newest article date (YYYY-MM-DD or datetime)
            language: 2-letter ISO-639-1 code (en, es, fr, de, etc.)
            sort_by: 'relevancy', 'popularity', 'publishedAt'
            page_size: Results per page (max 100)
            page: Page number

        Returns:
            Dictionary with articles and metadata

        Example:
            >>> # Search for Bitcoin news from last 7 days
            >>> news = newsapi.get_everything(
            ...     q='Bitcoin OR BTC',
            ...     from_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            ...     sort_by='relevancy'
            ... )
        """
        params = {
            'pageSize': min(page_size, self.max_page_size),
            'page': page,
            'language': language,
            'sortBy': sort_by
        }

        if q:
            params['q'] = q

        if sources:
            params['sources'] = ','.join(sources)

        if domains:
            params['domains'] = ','.join(domains)

        if exclude_domains:
            params['excludeDomains'] = ','.join(exclude_domains)

        if from_date:
            if isinstance(from_date, datetime):
                params['from'] = from_date.strftime('%Y-%m-%d')
            else:
                params['from'] = from_date

        if to_date:
            if isinstance(to_date, datetime):
                params['to'] = to_date.strftime('%Y-%m-%d')
            else:
                params['to'] = to_date

        return self._make_request('everything', params)

    def get_sources(
        self,
        category: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None
    ) -> Dict:
        """
        Get available news sources

        Args:
            category: business, entertainment, general, health, science, sports, technology
            language: 2-letter ISO-639-1 code
            country: 2-letter ISO 3166-1 code

        Returns:
            Dictionary with sources list
        """
        params = {}

        if category:
            params['category'] = category
        if language:
            params['language'] = language
        if country:
            params['country'] = country

        return self._make_request('top-headlines/sources', params)

    # === Convenience Methods for HelixOne ===

    def get_financial_news(
        self,
        days_back: int = 7,
        page_size: int = 50
    ) -> List[Dict]:
        """
        Get recent financial/business news

        Args:
            days_back: Number of days to look back
            page_size: Number of articles to return

        Returns:
            List of article dictionaries
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        result = self.get_everything(
            q='stock OR market OR finance OR economy OR trading',
            domains=[
                'bloomberg.com',
                'reuters.com',
                'cnbc.com',
                'marketwatch.com',
                'wsj.com',
                'ft.com'
            ],
            from_date=from_date,
            sort_by='publishedAt',
            page_size=page_size
        )

        return result.get('articles', [])

    def get_company_news(
        self,
        company_name: str,
        ticker: Optional[str] = None,
        days_back: int = 30,
        page_size: int = 20
    ) -> List[Dict]:
        """
        Get news for specific company

        Args:
            company_name: Company name (e.g., 'Apple', 'Tesla')
            ticker: Stock ticker (optional, e.g., 'AAPL')
            days_back: Number of days to look back
            page_size: Number of articles

        Returns:
            List of article dictionaries
        """
        # Build search query
        if ticker:
            query = f'"{company_name}" OR {ticker}'
        else:
            query = f'"{company_name}"'

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        result = self.get_everything(
            q=query,
            from_date=from_date,
            sort_by='relevancy',
            page_size=page_size
        )

        return result.get('articles', [])

    def get_crypto_news(
        self,
        crypto_name: Optional[str] = None,
        days_back: int = 7,
        page_size: int = 30
    ) -> List[Dict]:
        """
        Get cryptocurrency news

        Args:
            crypto_name: Specific crypto (e.g., 'Bitcoin', 'Ethereum') or None for all
            days_back: Number of days to look back
            page_size: Number of articles

        Returns:
            List of article dictionaries
        """
        if crypto_name:
            query = f'cryptocurrency AND {crypto_name}'
        else:
            query = 'cryptocurrency OR bitcoin OR ethereum OR crypto'

        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        result = self.get_everything(
            q=query,
            from_date=from_date,
            sort_by='publishedAt',
            page_size=page_size
        )

        return result.get('articles', [])

    def get_sector_news(
        self,
        sector: str,
        days_back: int = 7,
        page_size: int = 20
    ) -> List[Dict]:
        """
        Get news for specific sector

        Args:
            sector: Sector name (e.g., 'technology', 'healthcare', 'energy')
            days_back: Days to look back
            page_size: Number of articles

        Returns:
            List of article dictionaries
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        result = self.get_everything(
            q=f'{sector} AND (stock OR market OR company)',
            from_date=from_date,
            sort_by='publishedAt',
            page_size=page_size
        )

        return result.get('articles', [])

    def get_trending_topics(
        self,
        category: str = 'business',
        country: str = 'us',
        page_size: int = 10
    ) -> List[Dict]:
        """
        Get trending news topics

        Args:
            category: business, technology, etc.
            country: Country code
            page_size: Number of articles

        Returns:
            List of trending articles
        """
        result = self.get_top_headlines(
            category=category,
            country=country,
            page_size=page_size
        )

        return result.get('articles', [])


# === Singleton Pattern ===

_newsapi_instance = None

def get_newsapi_collector(api_key: Optional[str] = None) -> NewsAPISource:
    """
    Get or create NewsAPI collector instance (singleton pattern)

    Args:
        api_key: Optional API key override

    Returns:
        NewsAPISource instance
    """
    global _newsapi_instance

    if _newsapi_instance is None:
        _newsapi_instance = NewsAPISource(api_key=api_key)

    return _newsapi_instance
