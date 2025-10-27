"""
Data Aggregator - Collecte parallèle de données multi-sources
Rassemble toutes les données disponibles pour l'analyse de portefeuille

Sources utilisées:
- Prix & volume (Alpha Vantage, Finnhub, FMP, TwelveData)
- Sentiment social (Reddit, StockTwits)
- News (NewsAPI)
- Tendances (Google Trends)
- Fondamentaux (FMP, Alpha Vantage)
- Données officielles (SEC EDGAR pour insider trading)
- Macro-économie (FRED)
- Corrélations sectorielles
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import traceback

# Import des sources de données
from app.services.reddit_source import get_reddit_collector
from app.services.stocktwits_source import get_stocktwits_collector
from app.services.newsapi_source import NewsAPISource
from app.services.google_trends_source import GoogleTrendsSource
from app.services.feargreed_source import FearGreedSource

logger = logging.getLogger(__name__)


@dataclass
class StockSentiment:
    """Sentiment agrégé pour une action"""
    ticker: str

    # Reddit
    reddit_sentiment: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    reddit_mentions: int = 0
    reddit_bullish_pct: float = 0.0
    reddit_bearish_pct: float = 0.0

    # StockTwits
    stocktwits_sentiment: Optional[str] = None
    stocktwits_messages: int = 0
    stocktwits_bullish_pct: float = 0.0
    stocktwits_bearish_pct: float = 0.0

    # News
    news_sentiment: Optional[str] = None  # 'positive', 'negative', 'neutral'
    news_count: int = 0
    news_score: float = 0.0  # -1 to 1

    # Tendances recherche
    google_trends_score: Optional[int] = None  # 0-100

    # Sentiment global agrégé
    overall_sentiment: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    sentiment_confidence: float = 0.0  # 0-100

    # Timestamp
    updated_at: datetime = None


@dataclass
class StockPrice:
    """Prix et données de marché"""
    ticker: str
    current_price: float
    change_pct: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    year_high: Optional[float] = None
    year_low: Optional[float] = None
    updated_at: datetime = None


@dataclass
class StockFundamentals:
    """Données fondamentales"""
    ticker: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    dividend_yield: Optional[float] = None
    profit_margin: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    debt_to_equity: Optional[float] = None
    beta: Optional[float] = None
    eps: Optional[float] = None  # Earnings Per Share
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    updated_at: datetime = None


@dataclass
class StockNews:
    """Actualités récentes"""
    ticker: str
    articles: List[Dict[str, Any]]  # Liste des articles
    total_count: int = 0
    sentiment_score: float = 0.0  # -1 à 1
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    updated_at: datetime = None


@dataclass
class MacroData:
    """Données macro-économiques pertinentes"""
    # Indices principaux
    sp500_change: Optional[float] = None
    nasdaq_change: Optional[float] = None
    dow_change: Optional[float] = None
    vix_level: Optional[float] = None  # Volatilité

    # Taux
    fed_funds_rate: Optional[float] = None
    treasury_10y: Optional[float] = None

    # Économie
    inflation_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    gdp_growth: Optional[float] = None

    # Sentiment crypto (peut affecter tech stocks)
    crypto_fear_greed: Optional[int] = None  # 0-100

    updated_at: datetime = None


@dataclass
class AggregatedStockData:
    """Toutes les données agrégées pour une action"""
    ticker: str

    # Données collectées
    price: Optional[StockPrice] = None
    fundamentals: Optional[StockFundamentals] = None
    sentiment: Optional[StockSentiment] = None
    news: Optional[StockNews] = None

    # Métadonnées
    collection_time_ms: int = 0
    errors: List[str] = None
    updated_at: datetime = None

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        result = asdict(self)
        # Convertir les datetimes en ISO format
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result


class DataAggregator:
    """
    Agrégateur de données multi-sources
    Collecte en parallèle toutes les données nécessaires pour l'analyse
    """

    def __init__(self):
        """Initialise l'agrégateur avec toutes les sources"""
        self.reddit = get_reddit_collector()
        self.stocktwits = get_stocktwits_collector()
        self.newsapi = NewsAPISource()
        self.google_trends = GoogleTrendsSource()
        self.feargreed = FearGreedSource()

        logger.info("DataAggregator initialisé avec toutes les sources")

    async def aggregate_stock_data(
        self,
        ticker: str,
        include_sentiment: bool = True,
        include_news: bool = True,
        include_fundamentals: bool = True
    ) -> AggregatedStockData:
        """
        Collecte toutes les données disponibles pour une action

        Args:
            ticker: Ticker de l'action (ex: 'AAPL')
            include_sentiment: Inclure sentiment (Reddit, StockTwits)
            include_news: Inclure actualités
            include_fundamentals: Inclure fondamentaux

        Returns:
            AggregatedStockData avec toutes les données collectées
        """
        start_time = datetime.now()
        errors = []

        logger.info(f"🔍 Agrégation données pour {ticker}")

        # Lancer toutes les collectes en parallèle
        tasks = []

        # Prix (toujours inclus)
        tasks.append(self._collect_price_data(ticker))

        # Fondamentaux
        if include_fundamentals:
            tasks.append(self._collect_fundamentals(ticker))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        # Sentiment
        if include_sentiment:
            tasks.append(self._collect_sentiment(ticker))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        # News
        if include_news:
            tasks.append(self._collect_news(ticker))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        # Exécuter toutes les tâches en parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extraire les résultats
        price_data = results[0] if not isinstance(results[0], Exception) else None
        fundamentals_data = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        sentiment_data = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None
        news_data = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else None

        # Collecter les erreurs
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Erreur collecte données ({i}): {str(result)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Calculer le temps d'exécution
        collection_time = int((datetime.now() - start_time).total_seconds() * 1000)

        aggregated = AggregatedStockData(
            ticker=ticker,
            price=price_data,
            fundamentals=fundamentals_data,
            sentiment=sentiment_data,
            news=news_data,
            collection_time_ms=collection_time,
            errors=errors if errors else None,
            updated_at=datetime.now()
        )

        logger.info(f"✅ Agrégation {ticker} terminée en {collection_time}ms")

        return aggregated

    async def aggregate_multiple_stocks(
        self,
        tickers: List[str],
        **kwargs
    ) -> Dict[str, AggregatedStockData]:
        """
        Collecte les données pour plusieurs actions en parallèle

        Args:
            tickers: Liste de tickers
            **kwargs: Options passées à aggregate_stock_data

        Returns:
            Dict {ticker: AggregatedStockData}
        """
        logger.info(f"🔍 Agrégation de {len(tickers)} actions en parallèle")

        # Lancer toutes les collectes en parallèle
        tasks = [self.aggregate_stock_data(ticker, **kwargs) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Construire le dictionnaire de résultats
        aggregated_data = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Erreur agrégation {ticker}: {result}")
                # Créer un objet vide avec l'erreur
                aggregated_data[ticker] = AggregatedStockData(
                    ticker=ticker,
                    errors=[str(result)],
                    updated_at=datetime.now()
                )
            else:
                aggregated_data[ticker] = result

        return aggregated_data

    async def collect_macro_data(self) -> MacroData:
        """
        Collecte les données macro-économiques

        Returns:
            MacroData avec tous les indicateurs
        """
        logger.info("🌍 Collecte données macro-économiques")

        macro = MacroData(updated_at=datetime.now())

        try:
            # Fear & Greed Index (crypto, mais indicateur de sentiment général)
            fg_data = self.feargreed.get_current_index()
            if fg_data:
                macro.crypto_fear_greed = fg_data.get('value')

            # TODO: Ajouter appels FRED pour taux, inflation, etc.
            # Pour l'instant, laisser None (sera ajouté plus tard)

        except Exception as e:
            logger.error(f"Erreur collecte macro: {e}")

        return macro

    # ========================================================================
    # MÉTHODES PRIVÉES DE COLLECTE
    # ========================================================================

    async def _collect_price_data(self, ticker: str) -> Optional[StockPrice]:
        """Collecte prix et données de marché"""
        try:
            # TODO: Implémenter avec vos sources de prix
            # Pour l'instant, retourner des données mockées
            logger.debug(f"Collecte prix pour {ticker}")

            # En production, utiliser Alpha Vantage, Finnhub, etc.
            return StockPrice(
                ticker=ticker,
                current_price=0.0,  # À récupérer de l'API
                change_pct=0.0,
                volume=0,
                updated_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Erreur collecte prix {ticker}: {e}")
            return None

    async def _collect_fundamentals(self, ticker: str) -> Optional[StockFundamentals]:
        """Collecte données fondamentales"""
        try:
            logger.debug(f"Collecte fondamentaux pour {ticker}")

            # TODO: Implémenter avec FMP, Alpha Vantage
            return StockFundamentals(
                ticker=ticker,
                updated_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Erreur collecte fondamentaux {ticker}: {e}")
            return None

    async def _collect_sentiment(self, ticker: str) -> Optional[StockSentiment]:
        """Collecte sentiment de toutes les sources"""
        try:
            logger.debug(f"Collecte sentiment pour {ticker}")

            sentiment = StockSentiment(
                ticker=ticker,
                updated_at=datetime.now()
            )

            # Reddit sentiment
            try:
                reddit_data = self.reddit.get_ticker_sentiment(ticker, subreddit='wallstreetbets', limit=50)
                if reddit_data and reddit_data.get('mentions', 0) > 0:
                    sentiment.reddit_sentiment = reddit_data['sentiment']
                    sentiment.reddit_mentions = reddit_data['mentions']
                    sentiment.reddit_bullish_pct = reddit_data['bullish_posts'] / reddit_data['mentions'] * 100 if reddit_data['mentions'] > 0 else 0
                    sentiment.reddit_bearish_pct = reddit_data['bearish_posts'] / reddit_data['mentions'] * 100 if reddit_data['mentions'] > 0 else 0
            except Exception as e:
                logger.debug(f"Reddit sentiment non disponible pour {ticker}: {e}")

            # StockTwits sentiment
            try:
                st_data = self.stocktwits.get_ticker_sentiment(ticker, limit=30)
                if st_data and st_data.get('total_messages', 0) > 0:
                    sentiment.stocktwits_sentiment = st_data['sentiment']
                    sentiment.stocktwits_messages = st_data['total_messages']
                    sentiment.stocktwits_bullish_pct = st_data['bullish_pct']
                    sentiment.stocktwits_bearish_pct = st_data['bearish_pct']
            except Exception as e:
                logger.debug(f"StockTwits sentiment non disponible pour {ticker}: {e}")

            # Calculer sentiment global
            sentiment.overall_sentiment, sentiment.sentiment_confidence = self._aggregate_sentiment(sentiment)

            return sentiment

        except Exception as e:
            logger.error(f"Erreur collecte sentiment {ticker}: {e}")
            traceback.print_exc()
            return None

    async def _collect_news(self, ticker: str) -> Optional[StockNews]:
        """Collecte actualités récentes"""
        try:
            logger.debug(f"Collecte news pour {ticker}")

            # Rechercher news des 7 derniers jours
            articles = self.newsapi.search_stock_news(
                ticker,
                days_back=7,
                max_articles=20
            )

            if not articles:
                return StockNews(
                    ticker=ticker,
                    articles=[],
                    updated_at=datetime.now()
                )

            # Analyser le sentiment (basique)
            positive = 0
            negative = 0
            neutral = 0

            for article in articles:
                # Analyse de sentiment basique sur le titre
                title = article.get('title', '').lower()
                if any(word in title for word in ['surge', 'gain', 'profit', 'beat', 'high', 'up', 'bull']):
                    positive += 1
                elif any(word in title for word in ['fall', 'loss', 'miss', 'low', 'down', 'bear', 'crash']):
                    negative += 1
                else:
                    neutral += 1

            total = len(articles)
            sentiment_score = (positive - negative) / total if total > 0 else 0.0

            return StockNews(
                ticker=ticker,
                articles=articles,
                total_count=total,
                sentiment_score=sentiment_score,
                positive_count=positive,
                negative_count=negative,
                neutral_count=neutral,
                updated_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Erreur collecte news {ticker}: {e}")
            return None

    def _aggregate_sentiment(self, sentiment: StockSentiment) -> tuple[Optional[str], float]:
        """
        Agrège les sentiments de toutes les sources

        Returns:
            (sentiment global, confiance 0-100)
        """
        scores = []
        weights = []

        # Reddit (poids: 30%)
        if sentiment.reddit_mentions > 0:
            reddit_score = sentiment.reddit_bullish_pct - sentiment.reddit_bearish_pct
            scores.append(reddit_score)
            weights.append(0.30)

        # StockTwits (poids: 40% - plus fiable)
        if sentiment.stocktwits_messages > 0:
            st_score = sentiment.stocktwits_bullish_pct - sentiment.stocktwits_bearish_pct
            scores.append(st_score)
            weights.append(0.40)

        # News (poids: 30%)
        if sentiment.news_count > 0:
            news_score = sentiment.news_score * 100  # Convertir -1/1 en -100/100
            scores.append(news_score)
            weights.append(0.30)

        if not scores:
            return None, 0.0

        # Calculer score pondéré
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Déterminer sentiment
        if weighted_score > 20:
            overall = 'bullish'
        elif weighted_score < -20:
            overall = 'bearish'
        else:
            overall = 'neutral'

        # Confiance basée sur le nombre de sources
        confidence = (len(scores) / 3) * min(abs(weighted_score), 100)

        return overall, confidence


# Singleton
_aggregator_instance = None

def get_data_aggregator() -> DataAggregator:
    """Retourne l'instance singleton de l'agrégateur"""
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = DataAggregator()
    return _aggregator_instance
