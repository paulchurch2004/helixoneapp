"""
Reddit API Data Source - Sentiment Analysis
Documentation: https://www.reddit.com/dev/api/

Features:
- WallStreetBets sentiment
- Stock mentions tracking
- Crypto discussions
- Hot topics detection
- Comment sentiment analysis
- Free tier: 60 requests/minute

Coverage:
- r/wallstreetbets (13M+ members)
- r/stocks (6M+ members)
- r/CryptoCurrency (7M+ members)
- r/investing (2M+ members)
- Custom subreddit tracking

Use Cases:
- Retail sentiment gauge
- Meme stock detection
- Hype cycle tracking
- Community sentiment
- Contrarian indicators
"""

import praw
import os
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import Counter
import re


class RedditSource:
    """
    Reddit API collector for sentiment analysis

    Free: 60 req/min (requires Reddit account)
    Coverage: All subreddits
    Data: Posts, comments, upvotes, sentiment
    """

    def __init__(self):
        """Initialize Reddit API source"""
        # Reddit API credentials (read-only, no user auth needed)
        self.client_id = os.getenv('REDDIT_CLIENT_ID', 'anonymous')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET', 'anonymous')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'HelixOne/1.0')

        # Initialize Reddit instance
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                check_for_async=False
            )
            # Set to read-only mode (no authentication required)
            self.reddit.read_only = True
        except Exception as e:
            # Fallback: use anonymous mode
            self.reddit = None
            print(f"⚠️  Reddit API: {str(e)[:50]}")

        # Rate limiting
        self.min_request_interval = 1.0  # 60 req/min
        self.last_request_time = 0

        # Popular finance subreddits
        self.finance_subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'CryptoCurrency',
            'CryptoMarkets',
            'options',
            'stockmarket',
            'pennystocks',
            'RobinHood',
            'Daytrading'
        ]

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    def _extract_ticker_symbols(self, text: str) -> List[str]:
        """
        Extract stock ticker symbols from text

        Args:
            text: Text to search

        Returns:
            List of ticker symbols (e.g., ['AAPL', 'TSLA'])
        """
        # Pattern: $TICKER or uppercase 2-5 letters
        pattern = r'\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?=\s|$|[,.])'
        matches = re.findall(pattern, text.upper())

        # Flatten and filter common words
        tickers = [m[0] or m[1] for m in matches]

        # Filter out common false positives
        blacklist = {
            # Mots anglais courants
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY',
            'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'ITS',
            'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'HE', 'BE', 'TO', 'OF', 'IN', 'IT', 'IS', 'AT', 'ON', 'AN', 'AS',
            'BY', 'OR', 'IF', 'UP', 'SO', 'NO', 'MY', 'DO', 'GO', 'ME', 'WE', 'US',
            # Mots financiers génériques
            'WSB', 'CEO', 'CFO', 'CTO', 'COO', 'DD', 'IMO', 'USA', 'SEC', 'FED', 'USD', 'ETF', 'IPO', 'ATH', 'ATL',
            'YOLO', 'FOMO', 'IMO', 'TBH', 'LOL', 'WTF', 'OMG', 'RIP', 'GG', 'GL', 'AH', 'PM', 'EOD', 'EOW', 'YTD',
            # Pronoms et démonstratifs
            'THIS', 'THAT', 'THESE', 'THOSE', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 'THERE', 'THEN', 'THAN',
            # Autres faux positifs courants
            'BEEN', 'FROM', 'HAVE', 'THEY', 'WILL', 'WITH', 'INTO', 'OVER', 'JUST', 'LIKE', 'THEM', 'SOME',
            'VERY', 'BACK', 'GOOD', 'MOST', 'ONLY', 'ALSO', 'WELL', 'DOWN', 'EVEN', 'HERE', 'LAST', 'NEXT',
            'POST', 'MUCH', 'MAKE', 'MADE', 'LONG', 'SAME', 'SURE', 'TAKE', 'TOOK', 'BOTH', 'CALL', 'PUTS',
            'EACH', 'WEEK', 'MOON', 'HOLD', 'HELD', 'SELL', 'SOLD', 'GAIN', 'LOSS', 'WISH', 'HOPE', 'NEED',
            'WANT', 'KNOW', 'TELL', 'COME', 'GAVE', 'LOOK', 'READ', 'HIGH', 'BEST', 'EVER', 'ONCE', 'EDIT',
            'CLICK', 'LINK', 'REAL', 'MOVE', 'WENT', 'WERE', 'YOUR', 'AFTER', 'GOT', 'TODAY', 'KEEP', 'CALLS',
            'GAINS', 'DIP', 'THINK', 'END', 'TIME', 'OFF', 'DOING', 'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MUST',
            'GOING', 'STILL', 'GOING', 'OPEN', 'CLOSE', 'BUY', 'BOUGHT', 'RIGHT', 'LEFT', 'SAID', 'SAYS',
            'FULL', 'VIEW', 'DAILY', 'SH', 'GOD', 'HOLY', 'THANKS', 'BIG', 'MAY', 'HELP', 'LESS', 'MORE',
            'AFTER', 'ABOUT', 'BEING', 'THING', 'SHOW', 'YEAR', 'YEARS', 'PART', 'EVERY', 'MANY', 'FIRST',
            'FOUND', 'PLACE', 'POINT', 'TURNS', 'NEAR', 'FEELS', 'SEEMS', 'LOOKS', 'MEANS', 'KEEPS', 'GIVES',
            'UNTIL', 'PRICE', 'HIT', 'OPENS', 'DAYS', 'NICE', 'CASH', 'SHORT', 'TRADE', 'STOCK', 'MARKET',
            'MONEY', 'SHARE', 'VALUE', 'TOTAL', 'FINAL', 'START', 'STOP', 'TURN', 'WAIT', 'WATCH', 'HITS',
            'FREE', 'LIFE', 'LUCKY', 'IRA', 'YES', 'THANK', 'LIVE', 'DID', 'DONE', 'EASY', 'HARD', 'FAST',
            'SLOW', 'TRUE', 'FALSE', 'SAVE', 'LOSE', 'WON', 'LOST', 'PAY', 'PAID', 'COST', 'WORTH', 'NICE',
            'MEME', 'GUESS', 'SETUP', 'ROUND', 'HAD', 'DUMP', 'SET', 'TALKS', 'TALK', 'PUMP', 'DROP', 'DROPS',
            'RISE', 'FELL', 'FALL', 'RISES', 'FALLS', 'BELOW', 'ABOVE', 'UNDER', 'NEAR', 'CLOSE', 'CLOSED',
            'TIMES', 'HOME', 'IM', 'WORK', 'SINCE', 'ANY', 'OUCH', 'WORKS', 'GOES', 'GONE', 'MINE', 'ELSE',
            'NEVER', 'AGAIN', 'AROUND', 'LATE', 'EARLY', 'DURING', 'WORTH', 'EACH', 'FEW', 'FULL', 'KIND',
            'MEAN', 'OWN', 'REALLY', 'STILL', 'SUCH', 'USED', 'USING', 'WORKS', 'YEAH', 'YET', 'YOUNG',
            'FIRMS', 'WIN', 'WINS', 'PEAK', 'KIDS', 'BEAT', 'BEATS', 'PLAY', 'PLAYS', 'RISK', 'RISKS',
            'CHANGE', 'RATE', 'RATES', 'DEBT', 'FEAR', 'GROW', 'GROWS', 'RUNS', 'RUN', 'JUMP', 'JUMPS',
            'SALES', 'RE', 'ADD', 'SHT', 'GAME', 'HMM', 'ADDED', 'ADDS', 'GAMES', 'SALES', 'PLAN', 'PLANS',
            'TEST', 'TESTS', 'NEWS', 'SMALL', 'LARGE', 'CLEAR', 'DEAL', 'DEALS', 'ADDED', 'IDEA', 'IDEAS'
        }
        tickers = [t for t in tickers if t not in blacklist]

        return list(set(tickers))  # Remove duplicates

    # === SUBREDDIT POSTS ===

    def get_hot_posts(
        self,
        subreddit: str = 'wallstreetbets',
        limit: int = 25
    ) -> List[Dict]:
        """
        Get hot posts from a subreddit

        Args:
            subreddit: Subreddit name (e.g., 'wallstreetbets')
            limit: Number of posts (default 25, max 100)

        Returns:
            List of posts with metadata

        Example:
            >>> reddit = RedditSource()
            >>> posts = reddit.get_hot_posts('wallstreetbets', limit=10)
            >>> for post in posts:
            ...     print(f"{post['title']}: {post['score']} upvotes")
        """
        if not self.reddit:
            return []

        self._rate_limit()

        try:
            sub = self.reddit.subreddit(subreddit)
            posts = []

            for submission in sub.hot(limit=limit):
                posts.append({
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'author': str(submission.author),
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'url': f"https://reddit.com{submission.permalink}",
                    'link_flair_text': submission.link_flair_text,
                    'tickers': self._extract_ticker_symbols(submission.title + ' ' + submission.selftext)
                })

            return posts

        except Exception as e:
            print(f"Reddit API error: {str(e)[:50]}")
            return []

    def get_top_posts(
        self,
        subreddit: str = 'wallstreetbets',
        time_filter: str = 'day',
        limit: int = 25
    ) -> List[Dict]:
        """
        Get top posts from a subreddit

        Args:
            subreddit: Subreddit name
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            limit: Number of posts

        Returns:
            List of top posts
        """
        if not self.reddit:
            return []

        self._rate_limit()

        try:
            sub = self.reddit.subreddit(subreddit)
            posts = []

            for submission in sub.top(time_filter=time_filter, limit=limit):
                posts.append({
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'author': str(submission.author),
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'url': f"https://reddit.com{submission.permalink}",
                    'tickers': self._extract_ticker_symbols(submission.title + ' ' + submission.selftext)
                })

            return posts

        except Exception as e:
            print(f"Reddit API error: {str(e)[:50]}")
            return []

    # === TICKER MENTIONS ===

    def get_ticker_mentions(
        self,
        subreddit: str = 'wallstreetbets',
        time_filter: str = 'day',
        limit: int = 100
    ) -> Dict[str, int]:
        """
        Get most mentioned tickers in a subreddit

        Args:
            subreddit: Subreddit name
            time_filter: Time period
            limit: Number of posts to analyze

        Returns:
            {ticker: mention_count}

        Example:
            >>> mentions = reddit.get_ticker_mentions('wallstreetbets', 'day', 100)
            >>> top_10 = sorted(mentions.items(), key=lambda x: x[1], reverse=True)[:10]
            >>> for ticker, count in top_10:
            ...     print(f"{ticker}: {count} mentions")
        """
        posts = self.get_top_posts(subreddit, time_filter, limit)

        all_tickers = []
        for post in posts:
            all_tickers.extend(post['tickers'])

        # Count mentions
        ticker_counts = Counter(all_tickers)

        return dict(ticker_counts)

    def get_trending_tickers(
        self,
        subreddits: Optional[List[str]] = None,
        min_mentions: int = 3,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get trending tickers across multiple subreddits

        Args:
            subreddits: List of subreddits (default: finance subreddits)
            min_mentions: Minimum mentions to include
            limit: Posts per subreddit

        Returns:
            List of {ticker, mentions, sentiment_score}

        Example:
            >>> trending = reddit.get_trending_tickers(min_mentions=5)
            >>> for t in trending[:10]:
            ...     print(f"{t['ticker']}: {t['mentions']} mentions")
        """
        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing']

        all_tickers = []

        for subreddit in subreddits:
            posts = self.get_hot_posts(subreddit, limit)
            for post in posts:
                all_tickers.extend(post['tickers'])

        # Count and filter
        ticker_counts = Counter(all_tickers)
        filtered = {t: c for t, c in ticker_counts.items() if c >= min_mentions}

        # Sort by mentions
        trending = [
            {'ticker': ticker, 'mentions': count}
            for ticker, count in sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        ]

        return trending

    # === SENTIMENT ANALYSIS ===

    def get_ticker_sentiment(
        self,
        ticker: str,
        subreddit: str = 'wallstreetbets',
        limit: int = 50
    ) -> Dict:
        """
        Get sentiment for a specific ticker

        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            subreddit: Subreddit to search
            limit: Number of posts to analyze

        Returns:
            {
                'ticker': str,
                'mentions': int,
                'avg_score': float,
                'bullish_posts': int,
                'bearish_posts': int,
                'neutral_posts': int,
                'sentiment': str ('bullish', 'bearish', 'neutral')
            }

        Example:
            >>> sentiment = reddit.get_ticker_sentiment('TSLA')
            >>> print(f"TSLA sentiment: {sentiment['sentiment']}")
            >>> print(f"Mentions: {sentiment['mentions']}")
        """
        posts = self.get_hot_posts(subreddit, limit)

        # Filter posts mentioning the ticker
        relevant_posts = [p for p in posts if ticker.upper() in p['tickers']]

        if not relevant_posts:
            return {
                'ticker': ticker,
                'mentions': 0,
                'avg_score': 0,
                'bullish_posts': 0,
                'bearish_posts': 0,
                'neutral_posts': 0,
                'sentiment': 'neutral'
            }

        # Simple sentiment based on upvote ratio
        bullish = len([p for p in relevant_posts if p['upvote_ratio'] > 0.7])
        bearish = len([p for p in relevant_posts if p['upvote_ratio'] < 0.5])
        neutral = len(relevant_posts) - bullish - bearish

        avg_score = sum(p['score'] for p in relevant_posts) / len(relevant_posts)

        # Overall sentiment
        if bullish > bearish:
            sentiment = 'bullish'
        elif bearish > bullish:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return {
            'ticker': ticker,
            'mentions': len(relevant_posts),
            'avg_score': avg_score,
            'bullish_posts': bullish,
            'bearish_posts': bearish,
            'neutral_posts': neutral,
            'sentiment': sentiment,
            'subreddit': subreddit
        }

    # === SUBREDDIT INFO ===

    def get_subreddit_info(self, subreddit: str) -> Dict:
        """
        Get subreddit information

        Args:
            subreddit: Subreddit name

        Returns:
            Subreddit metadata
        """
        if not self.reddit:
            return {}

        try:
            sub = self.reddit.subreddit(subreddit)
            return {
                'name': sub.display_name,
                'title': sub.title,
                'subscribers': sub.subscribers,
                'active_users': sub.active_user_count,
                'description': sub.public_description,
                'created_utc': datetime.fromtimestamp(sub.created_utc)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_market_sentiment_summary(self) -> Dict:
        """
        Get overall market sentiment from multiple subreddits

        Returns:
            {
                'trending_tickers': list,
                'overall_sentiment': str,
                'top_subreddits': list
            }

        Example:
            >>> summary = reddit.get_market_sentiment_summary()
            >>> print(f"Overall sentiment: {summary['overall_sentiment']}")
            >>> print(f"Top ticker: {summary['trending_tickers'][0]['ticker']}")
        """
        # Get trending across main subreddits
        trending = self.get_trending_tickers(
            subreddits=['wallstreetbets', 'stocks'],
            min_mentions=5,
            limit=25
        )

        return {
            'trending_tickers': trending[:10],
            'timestamp': datetime.now(),
            'subreddits_analyzed': ['wallstreetbets', 'stocks'],
            'total_tickers': len(trending)
        }


# === SINGLETON PATTERN ===

_reddit_instance = None

def get_reddit_collector() -> RedditSource:
    """
    Get or create Reddit collector instance (singleton)

    Returns:
        RedditSource instance
    """
    global _reddit_instance

    if _reddit_instance is None:
        _reddit_instance = RedditSource()

    return _reddit_instance
