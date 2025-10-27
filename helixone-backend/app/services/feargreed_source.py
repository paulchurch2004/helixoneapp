"""
Crypto Fear & Greed Index Data Source
Documentation: https://alternative.me/crypto/fear-and-greed-index/

Free Tier Limits:
- UNLIMITED and FREE
- No API key required
- No rate limiting

Coverage:
- Crypto market sentiment indicator (0-100 scale)
- Historical data available
- Updated every 8 hours

Interpretation:
- 0-24: Extreme Fear
- 25-49: Fear
- 50-74: Greed
- 75-100: Extreme Greed
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class FearGreedSource:
    """
    Crypto Fear & Greed Index collector

    Free: Unlimited, no API key required
    Use case: Crypto market sentiment analysis
    """

    def __init__(self):
        """Initialize Fear & Greed Index source"""
        self.base_url = "https://api.alternative.me"

        # No rate limiting needed (unlimited), but be courteous
        self.min_request_interval = 1.0
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce courtesy rate limiting"""
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
            raise Exception(f"Fear & Greed Index request failed: {str(e)}")

    def get_current(self) -> Dict:
        """
        Get current Fear & Greed Index value

        Returns:
            Dictionary with:
            {
                'value': '50',
                'value_classification': 'Neutral',
                'timestamp': '1234567890',
                'time_until_update': '12345'
            }

        Example:
            >>> fg = feargreed.get_current()
            >>> print(f"Current index: {fg['value']} ({fg['value_classification']})")
        """
        result = self._make_request('fng/')

        if result.get('data') and len(result['data']) > 0:
            return result['data'][0]
        else:
            raise Exception("No data returned from Fear & Greed API")

    def get_historical(self, limit: int = 30) -> List[Dict]:
        """
        Get historical Fear & Greed Index values

        Args:
            limit: Number of historical points (1-365+)

        Returns:
            List of dictionaries, each with:
            {
                'value': '50',
                'value_classification': 'Neutral',
                'timestamp': '1234567890'
            }

        Example:
            >>> history = feargreed.get_historical(limit=7)
            >>> for point in history:
            ...     date = datetime.fromtimestamp(int(point['timestamp']))
            ...     print(f"{date.date()}: {point['value']} ({point['value_classification']})")
        """
        params = {
            'limit': limit
        }

        result = self._make_request('fng/', params)

        if result.get('data'):
            return result['data']
        else:
            raise Exception("No historical data returned")

    def get_index_with_interpretation(self) -> Dict:
        """
        Get current index with detailed interpretation

        Returns:
            Dictionary with:
            {
                'value': 50,
                'classification': 'Neutral',
                'timestamp': datetime,
                'interpretation': 'Detailed interpretation...',
                'trading_advice': 'Suggested trading strategy...'
            }
        """
        current = self.get_current()

        value = int(current['value'])
        classification = current['value_classification']
        timestamp = datetime.fromtimestamp(int(current['timestamp']))

        # Detailed interpretation
        if value <= 24:
            interpretation = "The market is in EXTREME FEAR. Investors are very worried, which often indicates a buying opportunity."
            trading_advice = "Consider buying - fear often creates opportunities for long-term investors."
        elif value <= 49:
            interpretation = "The market is experiencing FEAR. Sentiment is negative but not extreme."
            trading_advice = "Look for quality assets at discounted prices. Good time to accumulate."
        elif value <= 74:
            interpretation = "The market shows GREED. Investors are optimistic and buying."
            trading_advice = "Exercise caution. Consider taking some profits or wait for pullbacks."
        else:
            interpretation = "The market is in EXTREME GREED. Investors are overly optimistic, often a warning sign."
            trading_advice = "Consider selling or reducing positions. Market may be overbought."

        return {
            'value': value,
            'classification': classification,
            'timestamp': timestamp,
            'interpretation': interpretation,
            'trading_advice': trading_advice
        }

    def get_trend(self, days: int = 7) -> Dict:
        """
        Analyze Fear & Greed trend over recent days

        Args:
            days: Number of days to analyze (1-30)

        Returns:
            Dictionary with trend analysis:
            {
                'current_value': 50,
                'previous_value': 45,
                'change': 5,
                'change_percent': 11.11,
                'trend': 'Increasing Fear',
                'average': 47.5
            }
        """
        history = self.get_historical(limit=days)

        if len(history) < 2:
            raise Exception("Not enough historical data for trend analysis")

        current_value = int(history[0]['value'])
        previous_value = int(history[-1]['value'])
        change = current_value - previous_value
        change_percent = (change / previous_value) * 100 if previous_value != 0 else 0

        # Calculate average
        values = [int(point['value']) for point in history]
        average = sum(values) / len(values)

        # Determine trend description
        if change > 10:
            trend = "Rapidly Increasing Greed"
        elif change > 0:
            trend = "Gradually Increasing Greed"
        elif change < -10:
            trend = "Rapidly Increasing Fear"
        elif change < 0:
            trend = "Gradually Increasing Fear"
        else:
            trend = "Stable"

        return {
            'current_value': current_value,
            'previous_value': previous_value,
            'change': change,
            'change_percent': round(change_percent, 2),
            'trend': trend,
            'average': round(average, 2),
            'days_analyzed': days
        }

    def is_extreme_sentiment(self, threshold_fear: int = 25, threshold_greed: int = 75) -> Dict:
        """
        Check if current sentiment is extreme

        Args:
            threshold_fear: Value below which is considered extreme fear (default 25)
            threshold_greed: Value above which is considered extreme greed (default 75)

        Returns:
            Dictionary with:
            {
                'is_extreme': True,
                'type': 'fear' or 'greed',
                'value': 20,
                'message': 'Extreme fear detected!'
            }
        """
        current = self.get_current()
        value = int(current['value'])

        if value <= threshold_fear:
            return {
                'is_extreme': True,
                'type': 'fear',
                'value': value,
                'message': f"Extreme fear detected! Index at {value} (threshold: {threshold_fear})"
            }
        elif value >= threshold_greed:
            return {
                'is_extreme': True,
                'type': 'greed',
                'value': value,
                'message': f"Extreme greed detected! Index at {value} (threshold: {threshold_greed})"
            }
        else:
            return {
                'is_extreme': False,
                'type': 'neutral',
                'value': value,
                'message': f"Sentiment is neutral at {value}"
            }

    def get_statistics(self, days: int = 30) -> Dict:
        """
        Get statistical summary over recent period

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with statistics:
            {
                'min': 20,
                'max': 80,
                'average': 50,
                'median': 52,
                'std_dev': 15.5,
                'days_in_fear': 15,
                'days_in_greed': 15
            }
        """
        history = self.get_historical(limit=days)
        values = [int(point['value']) for point in history]

        if not values:
            raise Exception("No data available for statistics")

        import statistics

        days_in_fear = sum(1 for v in values if v < 50)
        days_in_greed = sum(1 for v in values if v >= 50)

        return {
            'min': min(values),
            'max': max(values),
            'average': round(statistics.mean(values), 2),
            'median': statistics.median(values),
            'std_dev': round(statistics.stdev(values), 2) if len(values) > 1 else 0,
            'days_in_fear': days_in_fear,
            'days_in_greed': days_in_greed,
            'total_days': len(values)
        }


# === Singleton Pattern ===

_feargreed_instance = None

def get_feargreed_collector() -> FearGreedSource:
    """
    Get or create Fear & Greed Index collector instance (singleton pattern)

    Returns:
        FearGreedSource instance
    """
    global _feargreed_instance

    if _feargreed_instance is None:
        _feargreed_instance = FearGreedSource()

    return _feargreed_instance
