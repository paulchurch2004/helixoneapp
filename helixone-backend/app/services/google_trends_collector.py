"""
Google Trends Data Collector
Source: Google Trends (via pytrends library)

GRATUIT - ILLIMITÉ - Pas de clé API requise
Données: Search volume, interest over time, related queries, trending searches

Author: HelixOne Team
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

# pytrends sera importé dynamiquement
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("⚠️  pytrends non installé. Run: pip install pytrends")


class GoogleTrendsCollector:
    """
    Collecteur de données Google Trends
    GRATUIT et ILLIMITÉ (avec rate limiting raisonnable)
    """

    def __init__(self):
        """Initialiser le collecteur Google Trends"""
        if not PYTRENDS_AVAILABLE:
            raise ImportError("pytrends library not installed. Run: pip install pytrends")

        # Initialiser pytrends avec paramètres par défaut
        self.pytrends = TrendReq(
            hl='en-US',  # Langue
            tz=360       # Timezone offset
        )

        logger.info("✅ Google Trends Collector initialisé (GRATUIT - ILLIMITÉ)")

    # ========================================================================
    # INTEREST OVER TIME
    # ========================================================================

    def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = 'today 12-m',
        geo: str = 'US'
    ) -> pd.DataFrame:
        """
        Récupérer l'évolution de l'intérêt pour des mots-clés

        Args:
            keywords: Liste de mots-clés (max 5)
            timeframe: Période ('today 12-m', 'today 5-y', 'all', etc.)
            geo: Code pays (US, FR, GB, etc.) ou '' pour mondial

        Returns:
            DataFrame avec l'intérêt au fil du temps
        """
        logger.info(f"📈 Google Trends: Interest over time pour {keywords}")

        if len(keywords) > 5:
            logger.warning(f"⚠️  Max 5 keywords. Tronqué à {keywords[:5]}")
            keywords = keywords[:5]

        try:
            # Construire la requête
            self.pytrends.build_payload(
                kw_list=keywords,
                timeframe=timeframe,
                geo=geo
            )

            # Récupérer les données
            data = self.pytrends.interest_over_time()

            logger.info(f"✅ {len(data)} points de données récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur interest over time: {e}")
            raise

    def get_ticker_interest(
        self,
        ticker: str,
        timeframe: str = 'today 12-m',
        geo: str = 'US'
    ) -> pd.DataFrame:
        """
        Récupérer l'intérêt de recherche pour un ticker spécifique

        Args:
            ticker: Symbole du ticker (ex: AAPL, TSLA)
            timeframe: Période
            geo: Code pays

        Returns:
            DataFrame avec l'intérêt
        """
        logger.info(f"📊 Google Trends: Ticker interest pour {ticker}")

        # Essayer plusieurs variantes du ticker
        keywords = [
            ticker,
            f"{ticker} stock",
            f"{ticker} share price"
        ]

        return self.get_interest_over_time(keywords[:5], timeframe, geo)

    # ========================================================================
    # INTEREST BY REGION
    # ========================================================================

    def get_interest_by_region(
        self,
        keywords: List[str],
        timeframe: str = 'today 12-m',
        resolution: str = 'COUNTRY'
    ) -> pd.DataFrame:
        """
        Récupérer l'intérêt par région géographique

        Args:
            keywords: Liste de mots-clés (max 5)
            timeframe: Période
            resolution: 'COUNTRY', 'REGION', 'CITY', 'DMA'

        Returns:
            DataFrame avec l'intérêt par région
        """
        logger.info(f"🌍 Google Trends: Interest by region pour {keywords}")

        if len(keywords) > 5:
            keywords = keywords[:5]

        try:
            # Construire la requête
            self.pytrends.build_payload(
                kw_list=keywords,
                timeframe=timeframe
            )

            # Récupérer les données
            data = self.pytrends.interest_by_region(
                resolution=resolution,
                inc_low_vol=True,
                inc_geo_code=True
            )

            logger.info(f"✅ {len(data)} régions récupérées")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur interest by region: {e}")
            raise

    # ========================================================================
    # RELATED QUERIES
    # ========================================================================

    def get_related_queries(
        self,
        keywords: List[str],
        timeframe: str = 'today 12-m',
        geo: str = 'US'
    ) -> Dict:
        """
        Récupérer les requêtes associées

        Args:
            keywords: Liste de mots-clés (max 5)
            timeframe: Période
            geo: Code pays

        Returns:
            Dict avec 'top' et 'rising' queries pour chaque keyword
        """
        logger.info(f"🔍 Google Trends: Related queries pour {keywords}")

        if len(keywords) > 5:
            keywords = keywords[:5]

        try:
            # Construire la requête
            self.pytrends.build_payload(
                kw_list=keywords,
                timeframe=timeframe,
                geo=geo
            )

            # Récupérer les queries associées
            data = self.pytrends.related_queries()

            logger.info(f"✅ Related queries récupérées")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur related queries: {e}")
            raise

    # ========================================================================
    # TRENDING SEARCHES
    # ========================================================================

    def get_trending_searches(self, country: str = 'united_states') -> pd.DataFrame:
        """
        Récupérer les recherches tendances du jour

        Args:
            country: Pays ('united_states', 'france', 'united_kingdom', etc.)

        Returns:
            DataFrame des trending searches
        """
        logger.info(f"🔥 Google Trends: Trending searches pour {country}")

        try:
            data = self.pytrends.trending_searches(pn=country)
            logger.info(f"✅ {len(data)} trending searches récupérées")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur trending searches: {e}")
            raise

    def get_realtime_trending(self, country: str = 'US', category: str = 'all') -> pd.DataFrame:
        """
        Récupérer les tendances en temps réel

        Args:
            country: Code pays (US, FR, GB, etc.)
            category: Catégorie ('all', 'b' pour business, etc.)

        Returns:
            DataFrame des trending now
        """
        logger.info(f"⚡ Google Trends: Realtime trending pour {country}")

        try:
            data = self.pytrends.realtime_trending_searches(country_code=country)
            logger.info(f"✅ {len(data)} realtime trends récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur realtime trending: {e}")
            raise

    # ========================================================================
    # SUGGESTIONS
    # ========================================================================

    def get_suggestions(self, keyword: str) -> List[Dict]:
        """
        Récupérer les suggestions pour un mot-clé

        Args:
            keyword: Mot-clé

        Returns:
            Liste de suggestions
        """
        logger.info(f"💡 Google Trends: Suggestions pour '{keyword}'")

        try:
            data = self.pytrends.suggestions(keyword=keyword)
            logger.info(f"✅ {len(data)} suggestions récupérées")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur suggestions: {e}")
            raise

    # ========================================================================
    # CATEGORIES
    # ========================================================================

    def get_categories(self) -> Dict:
        """
        Récupérer la liste des catégories disponibles

        Returns:
            Dict des catégories
        """
        logger.info("📂 Google Trends: Récupération catégories")

        try:
            data = self.pytrends.categories()
            logger.info(f"✅ Catégories récupérées")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur catégories: {e}")
            raise

    # ========================================================================
    # UTILS
    # ========================================================================

    def compare_tickers(
        self,
        tickers: List[str],
        timeframe: str = 'today 12-m',
        geo: str = 'US'
    ) -> pd.DataFrame:
        """
        Comparer l'intérêt de recherche pour plusieurs tickers

        Args:
            tickers: Liste de tickers (max 5)
            timeframe: Période
            geo: Code pays

        Returns:
            DataFrame comparatif
        """
        logger.info(f"📊 Google Trends: Compare tickers {tickers}")

        if len(tickers) > 5:
            logger.warning(f"⚠️  Max 5 tickers. Tronqué à {tickers[:5]}")
            tickers = tickers[:5]

        return self.get_interest_over_time(tickers, timeframe, geo)

    def get_stock_sentiment_score(
        self,
        ticker: str,
        timeframe: str = 'today 3-m'
    ) -> Dict:
        """
        Calculer un score de sentiment basé sur les tendances de recherche

        Args:
            ticker: Symbole du ticker
            timeframe: Période

        Returns:
            Dict avec score et métadonnées
        """
        logger.info(f"🎯 Google Trends: Sentiment score pour {ticker}")

        try:
            # Récupérer interest over time
            data = self.get_ticker_interest(ticker, timeframe)

            if data.empty or ticker not in data.columns:
                return {
                    'ticker': ticker,
                    'sentiment_score': 0,
                    'trend': 'neutral',
                    'error': 'No data available'
                }

            # Calculer le score
            values = data[ticker].dropna()
            if len(values) < 2:
                return {
                    'ticker': ticker,
                    'sentiment_score': 0,
                    'trend': 'neutral',
                    'error': 'Insufficient data'
                }

            # Score basé sur la tendance récente vs moyenne
            recent_avg = values[-7:].mean()  # Dernière semaine
            overall_avg = values.mean()

            # Score normalisé (-100 à +100)
            if overall_avg > 0:
                sentiment_score = ((recent_avg - overall_avg) / overall_avg) * 100
            else:
                sentiment_score = 0

            # Déterminer la tendance
            if sentiment_score > 20:
                trend = 'bullish'
            elif sentiment_score < -20:
                trend = 'bearish'
            else:
                trend = 'neutral'

            return {
                'ticker': ticker,
                'sentiment_score': round(sentiment_score, 2),
                'trend': trend,
                'recent_avg': round(recent_avg, 2),
                'overall_avg': round(overall_avg, 2),
                'max_interest': int(values.max()),
                'min_interest': int(values.min()),
                'current_interest': int(values.iloc[-1])
            }

        except Exception as e:
            logger.error(f"❌ Erreur sentiment score: {e}")
            return {
                'ticker': ticker,
                'sentiment_score': 0,
                'trend': 'neutral',
                'error': str(e)
            }


# Singleton
_google_trends_collector_instance = None

def get_google_trends_collector() -> GoogleTrendsCollector:
    """Obtenir l'instance singleton du Google Trends collector"""
    global _google_trends_collector_instance

    if _google_trends_collector_instance is None:
        _google_trends_collector_instance = GoogleTrendsCollector()

    return _google_trends_collector_instance
