"""
CoinGecko API Data Collector
Source: https://www.coingecko.com/en/api

GRATUIT - 10-50 req/minute - Pas de clé API requise (démo)
Données: 13,000+ cryptos, prix, volume, market cap, historique

Author: HelixOne Team
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoSource:
    """
    Collecteur de données CoinGecko
    GRATUIT - 10-50 requêtes/minute
    Meilleure API crypto gratuite au monde
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialiser le collecteur CoinGecko

        Args:
            api_key: Clé API (optionnelle - démo gratuit sans clé)
        """
        self.base_url = COINGECKO_BASE_URL
        self.api_key = api_key
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'HelixOne/1.0'
        }

        if api_key:
            self.headers['x-cg-demo-api-key'] = api_key

        # Rate limiting: ~10-50 req/min
        self.min_request_interval = 1.2  # 1.2s entre requêtes = 50/min
        self.last_request_time = time.time()

        logger.info("✅ CoinGecko Collector initialisé (GRATUIT - 10-50 req/min)")

    def _rate_limit(self):
        """Rate limiting automatique"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Faire une requête à l'API CoinGecko

        Args:
            endpoint: Endpoint de l'API
            params: Paramètres de requête

        Returns:
            Réponse JSON
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête CoinGecko {endpoint}: {e}")
            raise

    # ========================================================================
    # MARKET DATA
    # ========================================================================

    def get_coin_markets(
        self,
        vs_currency: str = 'usd',
        ids: Optional[List[str]] = None,
        order: str = 'market_cap_desc',
        per_page: int = 100,
        page: int = 1
    ) -> List[Dict]:
        """
        Récupérer la liste des cryptos avec prix et market data

        Args:
            vs_currency: Devise de cotation (usd, eur, btc, etc.)
            ids: Liste d'IDs cryptos (bitcoin, ethereum, etc.)
            order: Ordre de tri (market_cap_desc, volume_desc, etc.)
            per_page: Nombre par page (max 250)
            page: Numéro de page

        Returns:
            Liste de cryptos avec données de marché
        """
        logger.info(f"📊 CoinGecko: Markets (vs {vs_currency})")

        params = {
            'vs_currency': vs_currency,
            'order': order,
            'per_page': per_page,
            'page': page,
            'sparkline': False
        }

        if ids:
            params['ids'] = ','.join(ids)

        data = self._make_request('coins/markets', params)

        logger.info(f"✅ {len(data)} cryptos récupérées")
        return data

    def get_coin_details(self, coin_id: str, localization: bool = False) -> Dict:
        """
        Récupérer les détails complets d'une crypto

        Args:
            coin_id: ID de la crypto (bitcoin, ethereum, etc.)
            localization: Inclure traductions

        Returns:
            Détails complets de la crypto
        """
        logger.info(f"🪙 CoinGecko: Détails pour {coin_id}")

        params = {
            'localization': str(localization).lower(),
            'tickers': True,
            'market_data': True,
            'community_data': True,
            'developer_data': True
        }

        data = self._make_request(f'coins/{coin_id}', params)

        logger.info(f"✅ Détails {coin_id} récupérés")
        return data

    def get_coin_price(
        self,
        ids: List[str],
        vs_currencies: List[str] = None,
        include_market_cap: bool = True,
        include_24h_vol: bool = True,
        include_24h_change: bool = True
    ) -> Dict:
        """
        Récupérer le prix simple d'une ou plusieurs cryptos

        Args:
            ids: Liste d'IDs cryptos
            vs_currencies: Devises de cotation
            include_market_cap: Inclure market cap
            include_24h_vol: Inclure volume 24h
            include_24h_change: Inclure variation 24h

        Returns:
            Dict avec prix par crypto et devise
        """
        logger.info(f"💰 CoinGecko: Prix pour {len(ids)} cryptos")

        if vs_currencies is None:
            vs_currencies = ['usd']

        params = {
            'ids': ','.join(ids),
            'vs_currencies': ','.join(vs_currencies),
            'include_market_cap': str(include_market_cap).lower(),
            'include_24hr_vol': str(include_24h_vol).lower(),
            'include_24hr_change': str(include_24h_change).lower()
        }

        data = self._make_request('simple/price', params)

        logger.info(f"✅ Prix récupérés")
        return data

    def get_coin_market_chart(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: int = 30,
        interval: str = 'daily'
    ) -> Dict:
        """
        Récupérer l'historique des prix d'une crypto

        Args:
            coin_id: ID de la crypto
            vs_currency: Devise de cotation
            days: Nombre de jours (1, 7, 14, 30, 90, 180, 365, max)
            interval: Intervalle (daily si > 90 jours, sinon auto)

        Returns:
            Historique des prix, market cap et volume
        """
        logger.info(f"📈 CoinGecko: Historique {coin_id} ({days} jours)")

        params = {
            'vs_currency': vs_currency,
            'days': days
        }

        if days > 90:
            params['interval'] = 'daily'

        data = self._make_request(f'coins/{coin_id}/market_chart', params)

        # Format des données:
        # {
        #   'prices': [[timestamp_ms, price], ...],
        #   'market_caps': [[timestamp_ms, market_cap], ...],
        #   'total_volumes': [[timestamp_ms, volume], ...]
        # }

        logger.info(f"✅ Historique récupéré: {len(data.get('prices', []))} points")
        return data

    # ========================================================================
    # GLOBAL DATA
    # ========================================================================

    def get_global_data(self) -> Dict:
        """
        Récupérer les données globales du marché crypto

        Returns:
            Données globales (market cap total, volume, dominance, etc.)
        """
        logger.info("🌍 CoinGecko: Données globales marché crypto")

        data = self._make_request('global')

        if 'data' in data:
            logger.info(f"✅ Données globales récupérées")
            return data['data']

        return data

    def get_global_defi_data(self) -> Dict:
        """
        Récupérer les données globales DeFi

        Returns:
            Données DeFi (TVL, volumes, etc.)
        """
        logger.info("💎 CoinGecko: Données globales DeFi")

        data = self._make_request('global/decentralized_finance_defi')

        if 'data' in data:
            logger.info(f"✅ Données DeFi récupérées")
            return data['data']

        return data

    # ========================================================================
    # EXCHANGES
    # ========================================================================

    def get_exchanges(self, per_page: int = 100, page: int = 1) -> List[Dict]:
        """
        Récupérer la liste des exchanges

        Args:
            per_page: Nombre par page (max 250)
            page: Numéro de page

        Returns:
            Liste des exchanges avec volumes
        """
        logger.info(f"🏦 CoinGecko: Liste exchanges")

        params = {
            'per_page': per_page,
            'page': page
        }

        data = self._make_request('exchanges', params)

        logger.info(f"✅ {len(data)} exchanges récupérés")
        return data

    def get_exchange_details(self, exchange_id: str) -> Dict:
        """
        Récupérer les détails d'un exchange

        Args:
            exchange_id: ID de l'exchange (binance, coinbase, etc.)

        Returns:
            Détails de l'exchange
        """
        logger.info(f"🏦 CoinGecko: Détails exchange {exchange_id}")

        data = self._make_request(f'exchanges/{exchange_id}')

        logger.info(f"✅ Détails {exchange_id} récupérés")
        return data

    # ========================================================================
    # TRENDING
    # ========================================================================

    def get_trending(self) -> Dict:
        """
        Récupérer les cryptos tendances (top 7)

        Returns:
            Liste des cryptos tendances
        """
        logger.info("🔥 CoinGecko: Cryptos tendances")

        data = self._make_request('search/trending')

        if 'coins' in data:
            logger.info(f"✅ {len(data['coins'])} cryptos tendances")

        return data

    # ========================================================================
    # CATEGORIES
    # ========================================================================

    def get_categories(self, order: str = 'market_cap_desc') -> List[Dict]:
        """
        Récupérer les catégories de cryptos avec market data

        Args:
            order: Ordre de tri

        Returns:
            Liste des catégories
        """
        logger.info("📂 CoinGecko: Catégories cryptos")

        params = {
            'order': order
        }

        data = self._make_request('coins/categories', params)

        logger.info(f"✅ {len(data)} catégories récupérées")
        return data

    # ========================================================================
    # UTILS
    # ========================================================================

    def search_coins(self, query: str) -> Dict:
        """
        Rechercher des cryptos par nom ou symbole

        Args:
            query: Terme de recherche

        Returns:
            Résultats de recherche
        """
        logger.info(f"🔍 CoinGecko: Recherche '{query}'")

        params = {
            'query': query
        }

        data = self._make_request('search', params)

        coins_found = len(data.get('coins', []))
        logger.info(f"✅ {coins_found} résultats trouvés")

        return data

    def ping(self) -> bool:
        """
        Vérifier que l'API est accessible

        Returns:
            True si API accessible
        """
        try:
            data = self._make_request('ping')
            return 'gecko_says' in data
        except:
            return False


# Singleton
_coingecko_instance = None

def get_coingecko_collector(api_key: Optional[str] = None) -> CoinGeckoSource:
    """Obtenir l'instance singleton du CoinGecko collector"""
    global _coingecko_instance

    if _coingecko_instance is None:
        _coingecko_instance = CoinGeckoSource(api_key)

    return _coingecko_instance
