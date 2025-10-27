"""
Sentiment Aggregator - Analyse avancée du sentiment multi-sources
Détecte les tendances, les changements brusques, et calcule des scores de confiance

Analyse:
- Évolution temporelle du sentiment (7 derniers jours)
- Détection de changements brusques (alertes)
- Corrélation entre sources
- Force du signal (volume + sentiment)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from statistics import mean, stdev

from app.services.reddit_source import get_reddit_collector
from app.services.stocktwits_source import get_stocktwits_collector
from app.services.newsapi_source import NewsAPISource

logger = logging.getLogger(__name__)


@dataclass
class SentimentTrend:
    """Tendance de sentiment sur une période"""
    ticker: str
    period_days: int

    # Tendance globale
    current_sentiment: str  # 'bullish', 'bearish', 'neutral'
    trend_direction: str  # 'improving', 'deteriorating', 'stable'
    trend_strength: float  # 0-100

    # Scores actuels
    current_bullish_score: float  # 0-100
    current_bearish_score: float  # 0-100
    current_neutral_score: float  # 0-100

    # Changements
    sentiment_change_7d: float  # Changement sur 7 jours (-100 à +100)
    momentum: str  # 'accelerating', 'decelerating', 'stable'

    # Volume et activité
    total_mentions: int
    mentions_change_7d: float  # % changement volume mentions
    engagement_score: float  # 0-100 (volume × sentiment)

    # Consensus entre sources
    source_consensus: float  # 0-100 (accord entre Reddit, StockTwits, News)
    confidence_level: str  # 'very_high', 'high', 'medium', 'low'

    # Alertes
    alerts: List[str]  # Liste d'alertes détectées

    updated_at: datetime = None


@dataclass
class SentimentSignal:
    """Signal d'action basé sur le sentiment"""
    ticker: str
    signal_type: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    confidence: float  # 0-100
    reasons: List[str]  # Raisons du signal
    risk_level: str  # 'low', 'medium', 'high'
    timeframe: str  # 'short_term', 'medium_term', 'long_term'


class SentimentAggregator:
    """
    Agrégateur avancé de sentiment
    Analyse les tendances et génère des signaux d'action
    """

    def __init__(self):
        self.reddit = get_reddit_collector()
        self.stocktwits = get_stocktwits_collector()
        self.newsapi = NewsAPISource()

        logger.info("SentimentAggregator initialisé")

    def analyze_sentiment_trend(
        self,
        ticker: str,
        lookback_days: int = 7
    ) -> SentimentTrend:
        """
        Analyse la tendance de sentiment sur une période

        Args:
            ticker: Ticker de l'action
            lookback_days: Nombre de jours à analyser

        Returns:
            SentimentTrend avec analyse complète
        """
        logger.info(f"📊 Analyse tendance sentiment pour {ticker} ({lookback_days}j)")

        alerts = []

        # Collecter données actuelles
        reddit_current = self._get_reddit_sentiment(ticker)
        stocktwits_current = self._get_stocktwits_sentiment(ticker)
        news_current = self._get_news_sentiment(ticker)

        # Calculer scores actuels (moyenne pondérée)
        bullish_score, bearish_score, neutral_score = self._calculate_weighted_scores(
            reddit_current,
            stocktwits_current,
            news_current
        )

        # Déterminer sentiment actuel
        if bullish_score > bearish_score + 20:
            current_sentiment = 'bullish'
        elif bearish_score > bullish_score + 20:
            current_sentiment = 'bearish'
        else:
            current_sentiment = 'neutral'

        # Calculer volume total mentions
        total_mentions = (
            reddit_current.get('mentions', 0) +
            stocktwits_current.get('messages', 0) +
            news_current.get('count', 0)
        )

        # Consensus entre sources (similarité des scores)
        source_consensus = self._calculate_consensus(
            reddit_current,
            stocktwits_current,
            news_current
        )

        # Niveau de confiance
        if source_consensus > 80 and total_mentions > 50:
            confidence_level = 'very_high'
        elif source_consensus > 60 and total_mentions > 20:
            confidence_level = 'high'
        elif source_consensus > 40 or total_mentions > 10:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'

        # Analyser tendance (simplifié pour v1)
        # TODO: Implémenter collecte historique pour vraie tendance 7j
        trend_direction = 'stable'
        trend_strength = 50.0
        sentiment_change_7d = 0.0
        mentions_change_7d = 0.0
        momentum = 'stable'

        # Détection d'alertes
        if bullish_score > 80:
            alerts.append("Sentiment extrêmement bullish (>80%) - Possible euphorie")
        elif bearish_score > 80:
            alerts.append("Sentiment extrêmement bearish (>80%) - Possible panique")

        if total_mentions > 100 and bullish_score > 70:
            alerts.append("Volume mentions élevé + sentiment très positif - Buzz fort")

        if source_consensus < 30:
            alerts.append("Consensus faible entre sources - Signaux contradictoires")

        # Calculer engagement score (volume × sentiment)
        engagement_score = min(total_mentions / 10, 50) + (bullish_score - bearish_score) / 2
        engagement_score = max(0, min(100, engagement_score))

        trend = SentimentTrend(
            ticker=ticker,
            period_days=lookback_days,
            current_sentiment=current_sentiment,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            current_bullish_score=bullish_score,
            current_bearish_score=bearish_score,
            current_neutral_score=neutral_score,
            sentiment_change_7d=sentiment_change_7d,
            momentum=momentum,
            total_mentions=total_mentions,
            mentions_change_7d=mentions_change_7d,
            engagement_score=engagement_score,
            source_consensus=source_consensus,
            confidence_level=confidence_level,
            alerts=alerts,
            updated_at=datetime.now()
        )

        logger.info(
            f"✅ {ticker}: {current_sentiment.upper()} "
            f"(Bull:{bullish_score:.0f}% Bear:{bearish_score:.0f}%) "
            f"Confiance:{confidence_level}"
        )

        return trend

    def generate_sentiment_signal(
        self,
        ticker: str,
        trend: SentimentTrend
    ) -> SentimentSignal:
        """
        Génère un signal d'action basé sur l'analyse de sentiment

        Args:
            ticker: Ticker
            trend: Tendance de sentiment

        Returns:
            SentimentSignal avec recommandation
        """
        reasons = []
        risk_level = 'medium'

        # Analyser pour générer signal
        bull_score = trend.current_bullish_score
        bear_score = trend.current_bearish_score
        consensus = trend.source_consensus
        mentions = trend.total_mentions

        # Logique de décision
        if bull_score > 70 and consensus > 60 and mentions > 20:
            signal_type = 'strong_buy' if bull_score > 80 else 'buy'
            confidence = min(95, bull_score + consensus / 2)
            reasons.append(f"Sentiment très bullish ({bull_score:.0f}%)")
            reasons.append(f"Consensus élevé entre sources ({consensus:.0f}%)")
            if mentions > 50:
                reasons.append(f"Volume de mentions élevé ({mentions})")
            risk_level = 'low' if consensus > 70 else 'medium'

        elif bear_score > 70 and consensus > 60 and mentions > 20:
            signal_type = 'strong_sell' if bear_score > 80 else 'sell'
            confidence = min(95, bear_score + consensus / 2)
            reasons.append(f"Sentiment très bearish ({bear_score:.0f}%)")
            reasons.append(f"Consensus élevé entre sources ({consensus:.0f}%)")
            risk_level = 'high'

        elif bull_score > bear_score + 15:
            signal_type = 'buy'
            confidence = 50 + (bull_score - bear_score) / 2
            reasons.append(f"Sentiment modérément bullish (Bull:{bull_score:.0f}% vs Bear:{bear_score:.0f}%)")
            risk_level = 'medium'

        elif bear_score > bull_score + 15:
            signal_type = 'sell'
            confidence = 50 + (bear_score - bull_score) / 2
            reasons.append(f"Sentiment modérément bearish (Bear:{bear_score:.0f}% vs Bull:{bull_score:.0f}%)")
            risk_level = 'medium'

        else:
            signal_type = 'hold'
            confidence = 50
            reasons.append("Sentiment neutre ou mixte")
            risk_level = 'medium'

        # Ajuster confiance selon consensus
        if consensus < 40:
            confidence *= 0.7
            reasons.append("⚠️ Consensus faible réduit la confiance")

        # Ajuster selon volume mentions
        if mentions < 10:
            confidence *= 0.8
            reasons.append("⚠️ Volume de mentions faible")

        # Timeframe basé sur la force du signal
        if confidence > 75:
            timeframe = 'short_term'  # Signal fort, agir rapidement
        elif confidence > 60:
            timeframe = 'medium_term'
        else:
            timeframe = 'long_term'

        return SentimentSignal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            reasons=reasons,
            risk_level=risk_level,
            timeframe=timeframe
        )

    # ========================================================================
    # MÉTHODES PRIVÉES
    # ========================================================================

    def _get_reddit_sentiment(self, ticker: str) -> Dict:
        """Récupère sentiment Reddit"""
        try:
            data = self.reddit.get_ticker_sentiment(ticker, subreddit='wallstreetbets', limit=50)
            return {
                'sentiment': data.get('sentiment', 'neutral'),
                'mentions': data.get('mentions', 0),
                'bullish': data.get('bullish_posts', 0),
                'bearish': data.get('bearish_posts', 0),
                'neutral': data.get('neutral_posts', 0)
            }
        except Exception as e:
            logger.debug(f"Reddit sentiment erreur pour {ticker}: {e}")
            return {'sentiment': 'neutral', 'mentions': 0, 'bullish': 0, 'bearish': 0, 'neutral': 0}

    def _get_stocktwits_sentiment(self, ticker: str) -> Dict:
        """Récupère sentiment StockTwits"""
        try:
            data = self.stocktwits.get_ticker_sentiment(ticker, limit=30)
            return {
                'sentiment': data.get('sentiment', 'neutral'),
                'messages': data.get('total_messages', 0),
                'bullish_pct': data.get('bullish_pct', 0),
                'bearish_pct': data.get('bearish_pct', 0),
                'bullish': data.get('bullish', 0),
                'bearish': data.get('bearish', 0)
            }
        except Exception as e:
            logger.debug(f"StockTwits sentiment erreur pour {ticker}: {e}")
            return {'sentiment': 'neutral', 'messages': 0, 'bullish_pct': 0, 'bearish_pct': 0}

    def _get_news_sentiment(self, ticker: str) -> Dict:
        """Récupère sentiment News"""
        try:
            articles = self.newsapi.search_stock_news(ticker, days_back=7, max_articles=20)

            if not articles:
                return {'sentiment': 'neutral', 'count': 0, 'score': 0}

            # Analyse basique
            positive = 0
            negative = 0
            for article in articles:
                title = article.get('title', '').lower()
                if any(word in title for word in ['surge', 'gain', 'profit', 'beat', 'high', 'up', 'bull']):
                    positive += 1
                elif any(word in title for word in ['fall', 'loss', 'miss', 'low', 'down', 'bear', 'crash']):
                    negative += 1

            total = len(articles)
            score = (positive - negative) / total if total > 0 else 0

            if score > 0.2:
                sentiment = 'positive'
            elif score < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return {
                'sentiment': sentiment,
                'count': total,
                'score': score,
                'positive': positive,
                'negative': negative
            }
        except Exception as e:
            logger.debug(f"News sentiment erreur pour {ticker}: {e}")
            return {'sentiment': 'neutral', 'count': 0, 'score': 0}

    def _calculate_weighted_scores(
        self,
        reddit: Dict,
        stocktwits: Dict,
        news: Dict
    ) -> Tuple[float, float, float]:
        """
        Calcule scores pondérés bullish/bearish/neutral

        Returns:
            (bullish_score, bearish_score, neutral_score) en %
        """
        bullish_scores = []
        bearish_scores = []
        weights = []

        # Reddit (poids: 25%)
        if reddit['mentions'] > 0:
            total = reddit['bullish'] + reddit['bearish'] + reddit['neutral']
            if total > 0:
                bull_pct = (reddit['bullish'] / total) * 100
                bear_pct = (reddit['bearish'] / total) * 100
                bullish_scores.append(bull_pct)
                bearish_scores.append(bear_pct)
                weights.append(0.25)

        # StockTwits (poids: 50% - plus fiable)
        if stocktwits['messages'] > 0:
            bullish_scores.append(stocktwits['bullish_pct'])
            bearish_scores.append(stocktwits['bearish_pct'])
            weights.append(0.50)

        # News (poids: 25%)
        if news['count'] > 0:
            # Convertir score -1/1 en pourcentages
            if news['score'] > 0:
                bull_pct = news['score'] * 100
                bear_pct = 0
            elif news['score'] < 0:
                bull_pct = 0
                bear_pct = abs(news['score']) * 100
            else:
                bull_pct = 0
                bear_pct = 0

            bullish_scores.append(bull_pct)
            bearish_scores.append(bear_pct)
            weights.append(0.25)

        if not bullish_scores:
            return 0.0, 0.0, 100.0

        # Calculer moyenne pondérée
        total_weight = sum(weights)
        bullish_avg = sum(b * w for b, w in zip(bullish_scores, weights)) / total_weight
        bearish_avg = sum(b * w for b, w in zip(bearish_scores, weights)) / total_weight
        neutral_avg = 100 - bullish_avg - bearish_avg

        return bullish_avg, bearish_avg, max(0, neutral_avg)

    def _calculate_consensus(
        self,
        reddit: Dict,
        stocktwits: Dict,
        news: Dict
    ) -> float:
        """
        Calcule le consensus entre les sources (0-100)
        100 = toutes les sources d'accord, 0 = totalement en désaccord
        """
        sentiments = []

        # Mapper les sentiments en scores numériques
        sentiment_map = {
            'bullish': 100, 'positive': 100,
            'neutral': 50,
            'bearish': 0, 'negative': 0
        }

        if reddit['mentions'] > 0:
            sentiments.append(sentiment_map.get(reddit['sentiment'], 50))

        if stocktwits['messages'] > 0:
            sentiments.append(sentiment_map.get(stocktwits['sentiment'], 50))

        if news['count'] > 0:
            sentiments.append(sentiment_map.get(news['sentiment'], 50))

        if len(sentiments) < 2:
            return 50.0  # Pas assez de sources pour calculer consensus

        # Calculer écart-type (moins d'écart = plus de consensus)
        try:
            std = stdev(sentiments)
            # Normaliser: écart-type de 0 = consensus 100, écart-type de 50 = consensus 0
            consensus = max(0, 100 - (std * 2))
            return consensus
        except:
            return 50.0


# Singleton
_sentiment_aggregator_instance = None

def get_sentiment_aggregator() -> SentimentAggregator:
    """Retourne l'instance singleton"""
    global _sentiment_aggregator_instance
    if _sentiment_aggregator_instance is None:
        _sentiment_aggregator_instance = SentimentAggregator()
    return _sentiment_aggregator_instance
