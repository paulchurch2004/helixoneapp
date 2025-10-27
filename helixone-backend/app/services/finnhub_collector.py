"""
Service de collecte de données via Finnhub API
Fournit des news, sentiment et données de marché en temps réel (60 req/min gratuit)
"""

import finnhub
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import os
import time

logger = logging.getLogger(__name__)

# Clé API Finnhub (gratuit: https://finnhub.io/register)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "your_finnhub_api_key_here")


class FinnhubCollector:
    """
    Collecteur de données Finnhub

    Features:
    - News en temps réel
    - Analyse de sentiment
    - Données fondamentales basiques
    - Recommandations d'analystes
    - Earnings calendar

    Limites gratuites:
    - 60 requêtes/minute
    - API key gratuite
    """

    def __init__(self, api_key: str = FINNHUB_API_KEY):
        """
        Initialiser le collecteur Finnhub

        Args:
            api_key: Clé API Finnhub
        """
        self.api_key = api_key
        self.client = finnhub.Client(api_key=api_key)

        # Rate limiting: 60 requêtes/minute
        self.requests_per_minute = 60
        self.request_times = []

        logger.info(f"✅ FinnhubCollector initialisé (clé: {api_key[:8]}...)")

    def _rate_limit(self):
        """Respecter les limites de taux (60 req/min)"""
        now = time.time()

        # Nettoyer les requêtes de plus d'une minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        # Si on a atteint la limite, attendre
        if len(self.request_times) >= self.requests_per_minute:
            oldest = self.request_times[0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.debug(f"⏳ Rate limiting: attente {wait_time:.1f}s")
                time.sleep(wait_time)
                # Nettoyer à nouveau
                now = time.time()
                self.request_times = [t for t in self.request_times if now - t < 60]

        # Enregistrer cette requête
        self.request_times.append(now)

    def get_company_news(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Récupérer les news d'une entreprise

        Args:
            symbol: Symbole du ticker
            from_date: Date de début au format YYYY-MM-DD (défaut: 7 jours en arrière)
            to_date: Date de fin au format YYYY-MM-DD (défaut: aujourd'hui)

        Returns:
            Liste de news avec titre, résumé, source, URL, timestamp
        """
        try:
            self._rate_limit()

            # Dates par défaut
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            if not from_date:
                from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            logger.info(f"📰 Finnhub: News pour {symbol}")

            # Les dates sont déjà des strings au format YYYY-MM-DD
            from_str = from_date
            to_str = to_date

            # Appeler l'API
            news = self.client.company_news(symbol, _from=from_str, to=to_str)

            logger.info(f"✅ {symbol}: {len(news)} articles récupérés")

            return news

        except Exception as e:
            logger.error(f"❌ Erreur news Finnhub {symbol}: {e}")
            raise

    def get_market_news(
        self,
        category: str = 'general',
        min_id: int = 0
    ) -> List[Dict]:
        """
        Récupérer les news de marché générales

        Args:
            category: 'general', 'forex', 'crypto', 'merger'
            min_id: ID minimum (pour pagination)

        Returns:
            Liste de news
        """
        try:
            self._rate_limit()

            logger.info(f"📰 Finnhub: Market news ({category})")

            news = self.client.general_news(category, min_id=min_id)

            logger.info(f"✅ {len(news)} articles de marché récupérés")

            return news

        except Exception as e:
            logger.error(f"❌ Erreur market news Finnhub: {e}")
            raise

    def get_news_sentiment(self, symbol: str) -> Dict:
        """
        Récupérer le sentiment des news pour un symbole

        Args:
            symbol: Symbole du ticker

        Returns:
            Dict avec sentiment score, buzz, et détails
        """
        try:
            self._rate_limit()

            logger.info(f"😊 Finnhub: Sentiment pour {symbol}")

            sentiment = self.client.news_sentiment(symbol)

            logger.info(f"✅ {symbol}: Sentiment score = {sentiment.get('companyNewsScore', 0)}")

            return sentiment

        except Exception as e:
            logger.error(f"❌ Erreur sentiment Finnhub {symbol}: {e}")
            raise

    def get_quote(self, symbol: str) -> Dict:
        """
        Récupérer la quote en temps réel

        Args:
            symbol: Symbole du ticker

        Returns:
            Dict avec prix current, high, low, open, volume
        """
        try:
            self._rate_limit()

            logger.info(f"💹 Finnhub: Quote pour {symbol}")

            quote = self.client.quote(symbol)

            logger.info(f"✅ {symbol}: ${quote.get('c', 0):.2f}")

            return quote

        except Exception as e:
            logger.error(f"❌ Erreur quote Finnhub {symbol}: {e}")
            raise

    def get_recommendation_trends(self, symbol: str) -> List[Dict]:
        """
        Récupérer les tendances de recommandations d'analystes

        Args:
            symbol: Symbole du ticker

        Returns:
            Liste avec recommandations (buy, hold, sell, strong buy, strong sell)
        """
        try:
            self._rate_limit()

            logger.info(f"📈 Finnhub: Recommandations pour {symbol}")

            recommendations = self.client.recommendation_trends(symbol)

            logger.info(f"✅ {symbol}: {len(recommendations)} périodes de recommandations")

            return recommendations

        except Exception as e:
            logger.error(f"❌ Erreur recommandations Finnhub {symbol}: {e}")
            raise

    def get_price_target(self, symbol: str) -> Dict:
        """
        Récupérer les objectifs de prix des analystes

        Args:
            symbol: Symbole du ticker

        Returns:
            Dict avec target high, low, mean, median
        """
        try:
            self._rate_limit()

            logger.info(f"🎯 Finnhub: Price target pour {symbol}")

            target = self.client.price_target(symbol)

            logger.info(f"✅ {symbol}: Target moyen = ${target.get('targetMean', 0):.2f}")

            return target

        except Exception as e:
            logger.error(f"❌ Erreur price target Finnhub {symbol}: {e}")
            raise

    def get_earnings_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        Récupérer le calendrier des publications de résultats

        Args:
            from_date: Date de début au format YYYY-MM-DD
            to_date: Date de fin au format YYYY-MM-DD
            symbol: Symbole spécifique (optionnel)

        Returns:
            Dict avec calendrier des earnings
        """
        try:
            self._rate_limit()

            logger.info("📅 Finnhub: Earnings calendar")

            # Dates par défaut (30 jours)
            if not from_date:
                from_date = datetime.now().strftime('%Y-%m-%d')
            if not to_date:
                to_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

            from_str = from_date
            to_str = to_date

            calendar = self.client.earnings_calendar(
                _from=from_str,
                to=to_str,
                symbol=symbol
            )

            count = len(calendar.get('earningsCalendar', []))
            logger.info(f"✅ {count} événements earnings")

            return calendar

        except Exception as e:
            logger.error(f"❌ Erreur earnings calendar Finnhub: {e}")
            raise

    def get_basic_financials(self, symbol: str, metric: str = 'all') -> Dict:
        """
        Récupérer les données financières basiques

        Args:
            symbol: Symbole du ticker
            metric: Type de métriques ('all', 'margin', 'growth', etc.)

        Returns:
            Dict avec métriques financières
        """
        try:
            self._rate_limit()

            logger.info(f"💰 Finnhub: Basic financials pour {symbol}")

            financials = self.client.company_basic_financials(symbol, metric)

            metrics_count = len(financials.get('metric', {}))
            logger.info(f"✅ {symbol}: {metrics_count} métriques financières")

            return financials

        except Exception as e:
            logger.error(f"❌ Erreur basic financials Finnhub {symbol}: {e}")
            raise

    def get_company_profile(self, symbol: str) -> Dict:
        """
        Récupérer le profil d'entreprise

        Args:
            symbol: Symbole du ticker

        Returns:
            Dict avec nom, industrie, logo, etc.
        """
        try:
            self._rate_limit()

            logger.info(f"🏢 Finnhub: Profile pour {symbol}")

            profile = self.client.company_profile2(symbol=symbol)

            logger.info(f"✅ {symbol}: {profile.get('name', 'N/A')}")

            return profile

        except Exception as e:
            logger.error(f"❌ Erreur profile Finnhub {symbol}: {e}")
            raise

    def get_social_sentiment(
        self,
        symbol: str,
        from_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Récupérer le sentiment des réseaux sociaux (Reddit, Twitter)

        Args:
            symbol: Symbole du ticker
            from_date: Date de début au format YYYY-MM-DD (défaut: hier)

        Returns:
            Liste de sentiment par plateforme
        """
        try:
            self._rate_limit()

            logger.info(f"📱 Finnhub: Social sentiment pour {symbol}")

            if not from_date:
                from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

            from_str = from_date

            sentiment = self.client.social_sentiment(symbol, _from=from_str)

            reddit_count = len(sentiment.get('reddit', []))
            twitter_count = len(sentiment.get('twitter', []))

            logger.info(f"✅ {symbol}: Reddit={reddit_count}, Twitter={twitter_count}")

            return sentiment

        except Exception as e:
            logger.error(f"❌ Erreur social sentiment Finnhub {symbol}: {e}")
            raise

    def get_market_sentiment(self) -> Dict:
        """
        Récupérer le sentiment général du marché

        Returns:
            Dict avec indices de sentiment
        """
        try:
            self._rate_limit()

            logger.info("📊 Finnhub: Market sentiment")

            # Note: Finnhub free tier peut ne pas avoir accès à certaines données
            # On utilise les indices disponibles
            sentiment = {}

            # Essayer de récupérer le VIX (indice de volatilité)
            try:
                vix = self.get_quote('^VIX')
                sentiment['vix'] = vix
            except:
                pass

            logger.info("✅ Market sentiment récupéré")

            return sentiment

        except Exception as e:
            logger.error(f"❌ Erreur market sentiment Finnhub: {e}")
            raise

    def get_usage_stats(self) -> Dict:
        """
        Obtenir les statistiques d'utilisation

        Returns:
            Dict avec nombre de requêtes
        """
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 60]

        return {
            'requests_last_minute': len(recent_requests),
            'max_requests_per_minute': self.requests_per_minute,
            'remaining': self.requests_per_minute - len(recent_requests),
            'percentage_used': (len(recent_requests) / self.requests_per_minute) * 100
        }


# Instance globale pour réutilisation
_finnhub_collector = None


def get_finnhub_collector(api_key: str = None) -> FinnhubCollector:
    """
    Obtenir l'instance du collecteur Finnhub (singleton)

    Args:
        api_key: Clé API (optionnel, utilise variable d'environnement par défaut)

    Returns:
        Instance FinnhubCollector
    """
    global _finnhub_collector

    if _finnhub_collector is None:
        _finnhub_collector = FinnhubCollector(api_key=api_key or FINNHUB_API_KEY)

    return _finnhub_collector
