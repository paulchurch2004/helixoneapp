"""
IEX Cloud Data Collector
Source: https://iexcloud.io/

Tier gratuit: 50,000 messages/mois
Données: Marché USA temps réel, fondamentaux, news

Author: HelixOne Team
"""

import os
import requests
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

IEX_API_KEY = os.getenv('IEX_CLOUD_API_KEY', '')
IEX_BASE_URL = "https://cloud.iexapis.com/stable"


class IEXCloudCollector:
    """
    Collecteur de données IEX Cloud
    50,000 messages/mois GRATUIT
    """

    def __init__(self, api_key: str = IEX_API_KEY):
        """
        Initialiser le collecteur IEX Cloud

        Args:
            api_key: Clé API IEX Cloud
        """
        if not api_key:
            logger.warning("⚠️  IEX Cloud API key non configurée")

        self.api_key = api_key
        self.base_url = IEX_BASE_URL

        # Rate limiting & usage
        self.messages_used = 0
        self.max_messages_per_month = 50000
        self.last_request_time = time.time()
        self.min_request_interval = 0.1

        logger.info("✅ IEX Cloud Collector initialisé (50k messages/mois)")

    def _rate_limit(self):
        """Rate limiting automatique"""
        if self.messages_used >= self.max_messages_per_month:
            logger.warning(f"⚠️  Limite mensuelle IEX Cloud atteinte ({self.max_messages_per_month})")
            raise Exception(f"Limite mensuelle IEX Cloud atteinte: {self.max_messages_per_month} messages")

        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> any:
        """
        Faire une requête à l'API IEX Cloud

        Args:
            endpoint: Endpoint
            params: Paramètres

        Returns:
            Réponse JSON
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}
        params['token'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Compter les messages utilisés
            if 'iexcloud-messages-used' in response.headers:
                self.messages_used += int(response.headers['iexcloud-messages-used'])

            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête IEX {endpoint}: {e}")
            raise

    # ========================================================================
    # DONNÉES DE MARCHÉ
    # ========================================================================

    def get_quote(self, symbol: str) -> Dict:
        """
        Récupérer la quote temps réel

        Args:
            symbol: Symbole du ticker

        Returns:
            Quote
        """
        logger.info(f"📊 IEX: Quote pour {symbol}")

        endpoint = f"stock/{symbol}/quote"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: ${data.get('latestPrice', 0)}")

        return data

    def get_ohlc(self, symbol: str) -> Dict:
        """
        Récupérer OHLC du jour

        Args:
            symbol: Symbole du ticker

        Returns:
            OHLC
        """
        logger.info(f"📊 IEX: OHLC pour {symbol}")

        endpoint = f"stock/{symbol}/ohlc"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: OHLC récupéré")

        return data

    def get_historical_prices(
        self,
        symbol: str,
        range_period: str = "1m"
    ) -> List[Dict]:
        """
        Récupérer les prix historiques

        Ranges: 5d, 1m, 3m, 6m, ytd, 1y, 2y, 5y, max

        Args:
            symbol: Symbole du ticker
            range_period: Période

        Returns:
            Prix historiques
        """
        logger.info(f"📈 IEX: Historical prices {symbol} ({range_period})")

        endpoint = f"stock/{symbol}/chart/{range_period}"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: {len(data)} jours de données")

        return data

    def get_intraday_prices(
        self,
        symbol: str
    ) -> List[Dict]:
        """
        Récupérer les prix intraday

        Args:
            symbol: Symbole du ticker

        Returns:
            Prix intraday
        """
        logger.info(f"📊 IEX: Intraday prices pour {symbol}")

        endpoint = f"stock/{symbol}/intraday-prices"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: {len(data)} points intraday")

        return data

    # ========================================================================
    # DONNÉES FONDAMENTALES
    # ========================================================================

    def get_company_info(self, symbol: str) -> Dict:
        """
        Récupérer les informations de l'entreprise

        Args:
            symbol: Symbole du ticker

        Returns:
            Company info
        """
        logger.info(f"🏢 IEX: Company info pour {symbol}")

        endpoint = f"stock/{symbol}/company"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: {data.get('companyName', 'N/A')}")

        return data

    def get_stats(self, symbol: str) -> Dict:
        """
        Récupérer les statistiques clés

        Args:
            symbol: Symbole du ticker

        Returns:
            Stats
        """
        logger.info(f"📊 IEX: Stats pour {symbol}")

        endpoint = f"stock/{symbol}/stats"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: Stats récupérées")

        return data

    def get_dividends(
        self,
        symbol: str,
        range_period: str = "1y"
    ) -> List[Dict]:
        """
        Récupérer les dividendes

        Args:
            symbol: Symbole du ticker
            range_period: Période

        Returns:
            Dividendes
        """
        logger.info(f"💰 IEX: Dividendes pour {symbol}")

        endpoint = f"stock/{symbol}/dividends/{range_period}"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: {len(data)} dividendes")

        return data

    # ========================================================================
    # NEWS
    # ========================================================================

    def get_news(self, symbol: str, last: int = 10) -> List[Dict]:
        """
        Récupérer les news

        Args:
            symbol: Symbole du ticker
            last: Nombre de news

        Returns:
            News
        """
        logger.info(f"📰 IEX: News pour {symbol}")

        endpoint = f"stock/{symbol}/news/last/{last}"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: {len(data)} articles")

        return data

    # ========================================================================
    # MARKET DATA
    # ========================================================================

    def get_market_volume(self) -> List[Dict]:
        """
        Récupérer le volume du marché

        Returns:
            Volume par exchange
        """
        logger.info("📊 IEX: Market volume")

        endpoint = "stock/market/volume"

        data = self._make_request(endpoint)

        logger.info(f"✅ Volume market récupéré")

        return data

    def get_sectors_performance(self) -> List[Dict]:
        """
        Récupérer la performance des secteurs

        Returns:
            Performance secteurs
        """
        logger.info("📊 IEX: Sectors performance")

        endpoint = "stock/market/sector-performance"

        data = self._make_request(endpoint)

        logger.info(f"✅ {len(data)} secteurs")

        return data

    # ========================================================================
    # UTILS
    # ========================================================================

    def get_usage_stats(self) -> Dict:
        """
        Obtenir les statistiques d'utilisation

        Returns:
            Stats d'utilisation
        """
        return {
            "messages_used": self.messages_used,
            "max_messages_per_month": self.max_messages_per_month,
            "messages_remaining": self.max_messages_per_month - self.messages_used,
            "usage_percentage": (self.messages_used / self.max_messages_per_month) * 100
        }


# Singleton
_iex_cloud_collector_instance = None

def get_iex_cloud_collector() -> IEXCloudCollector:
    """Obtenir l'instance singleton du IEX Cloud collector"""
    global _iex_cloud_collector_instance

    if _iex_cloud_collector_instance is None:
        _iex_cloud_collector_instance = IEXCloudCollector()

    return _iex_cloud_collector_instance
