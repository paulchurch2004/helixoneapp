"""
Twelve Data Collector
Source: https://twelvedata.com/

Tier gratuit: 800 requêtes/jour
Données: Marché global (stocks, forex, crypto), indicateurs techniques

Author: HelixOne Team
"""

import os
import requests
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY', '')
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"


class TwelveDataCollector:
    """
    Collecteur de données Twelve Data
    """

    def __init__(self, api_key: str = TWELVEDATA_API_KEY):
        """
        Initialiser le collecteur Twelve Data

        Args:
            api_key: Clé API Twelve Data
        """
        if not api_key:
            logger.warning("⚠️  Twelve Data API key non configurée")

        self.api_key = api_key
        self.base_url = TWELVEDATA_BASE_URL

        # Rate limiting: 800 req/jour (gratuit)
        self.requests_today = 0
        self.max_requests_per_day = 800
        self.last_request_time = time.time()
        self.min_request_interval = 1.0  # 1 seconde entre requêtes

        logger.info("✅ Twelve Data Collector initialisé")

    def _rate_limit(self):
        """Rate limiting automatique"""
        if self.requests_today >= self.max_requests_per_day:
            logger.warning(f"⚠️  Limite quotidienne Twelve Data atteinte ({self.max_requests_per_day})")
            raise Exception(f"Limite quotidienne Twelve Data atteinte: {self.max_requests_per_day} requêtes")

        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        self.last_request_time = time.time()
        self.requests_today += 1

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Faire une requête à l'API Twelve Data

        Args:
            endpoint: Endpoint de l'API
            params: Paramètres

        Returns:
            Réponse JSON
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}
        params['apikey'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Vérifier erreur
            if isinstance(data, dict) and 'code' in data and data['code'] >= 400:
                raise Exception(f"Twelve Data Error: {data.get('message', 'Unknown error')}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête Twelve Data {endpoint}: {e}")
            raise

    # ========================================================================
    # DONNÉES DE MARCHÉ
    # ========================================================================

    def get_quote(self, symbol: str, interval: str = "1day") -> Dict:
        """
        Récupérer la quote temps réel

        Args:
            symbol: Symbole du ticker
            interval: Intervalle (1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month)

        Returns:
            Quote temps réel
        """
        logger.info(f"📊 Twelve Data: Quote pour {symbol}")

        endpoint = "quote"
        params = {'symbol': symbol, 'interval': interval}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: ${data.get('close', 0)}")

        return data

    def get_time_series(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Récupérer les données de séries temporelles (OHLCV)

        Args:
            symbol: Symbole du ticker
            interval: Intervalle (1min à 1month)
            outputsize: Nombre de points (max 5000)
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)

        Returns:
            Séries temporelles OHLCV
        """
        logger.info(f"📈 Twelve Data: Time Series pour {symbol} ({interval})")

        endpoint = "time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize
        }

        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        data = self._make_request(endpoint, params)

        if 'values' in data:
            logger.info(f"✅ {symbol}: {len(data['values'])} points de données")

        return data

    def get_eod(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Récupérer les données End of Day (EOD)

        Args:
            symbol: Symbole du ticker
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)

        Returns:
            Données EOD
        """
        logger.info(f"📊 Twelve Data: EOD pour {symbol}")

        endpoint = "eod"
        params = {'symbol': symbol}

        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: EOD récupéré")

        return data

    # ========================================================================
    # INDICATEURS TECHNIQUES
    # ========================================================================

    def get_rsi(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 14
    ) -> Dict:
        """
        Récupérer le RSI (Relative Strength Index)

        Args:
            symbol: Symbole du ticker
            interval: Intervalle
            time_period: Période RSI (défaut 14)

        Returns:
            RSI
        """
        logger.info(f"📊 Twelve Data: RSI pour {symbol}")

        endpoint = "rsi"
        params = {
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period
        }

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: RSI récupéré")

        return data

    def get_macd(
        self,
        symbol: str,
        interval: str = "1day"
    ) -> Dict:
        """
        Récupérer le MACD (Moving Average Convergence Divergence)

        Args:
            symbol: Symbole du ticker
            interval: Intervalle

        Returns:
            MACD
        """
        logger.info(f"📊 Twelve Data: MACD pour {symbol}")

        endpoint = "macd"
        params = {'symbol': symbol, 'interval': interval}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: MACD récupéré")

        return data

    def get_ema(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 9
    ) -> Dict:
        """
        Récupérer l'EMA (Exponential Moving Average)

        Args:
            symbol: Symbole du ticker
            interval: Intervalle
            time_period: Période (défaut 9)

        Returns:
            EMA
        """
        logger.info(f"📊 Twelve Data: EMA pour {symbol}")

        endpoint = "ema"
        params = {
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period
        }

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: EMA récupéré")

        return data

    def get_bbands(
        self,
        symbol: str,
        interval: str = "1day",
        time_period: int = 20
    ) -> Dict:
        """
        Récupérer les Bollinger Bands

        Args:
            symbol: Symbole du ticker
            interval: Intervalle
            time_period: Période (défaut 20)

        Returns:
            Bollinger Bands
        """
        logger.info(f"📊 Twelve Data: Bollinger Bands pour {symbol}")

        endpoint = "bbands"
        params = {
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period
        }

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: Bollinger Bands récupéré")

        return data

    # ========================================================================
    # FOREX
    # ========================================================================

    def get_forex_pair(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 30
    ) -> Dict:
        """
        Récupérer les données Forex

        Args:
            symbol: Paire forex (ex: EUR/USD)
            interval: Intervalle
            outputsize: Nombre de points

        Returns:
            Données Forex
        """
        logger.info(f"💱 Twelve Data: Forex pour {symbol}")

        endpoint = "time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize
        }

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: Forex récupéré")

        return data

    def get_currency_conversion(
        self,
        symbol: str,
        amount: float = 1
    ) -> Dict:
        """
        Convertir une devise

        Args:
            symbol: Paire (ex: USD/EUR)
            amount: Montant à convertir

        Returns:
            Conversion
        """
        logger.info(f"💱 Twelve Data: Conversion {symbol}")

        endpoint = "currency_conversion"
        params = {'symbol': symbol, 'amount': amount}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {amount} = {data.get('amount', 0)}")

        return data

    # ========================================================================
    # CRYPTO
    # ========================================================================

    def get_crypto_quote(
        self,
        symbol: str,
        interval: str = "1day"
    ) -> Dict:
        """
        Récupérer la quote crypto

        Args:
            symbol: Symbole crypto (ex: BTC/USD)
            interval: Intervalle

        Returns:
            Quote crypto
        """
        logger.info(f"₿ Twelve Data: Crypto quote pour {symbol}")

        endpoint = "quote"
        params = {'symbol': symbol, 'interval': interval}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: ${data.get('close', 0)}")

        return data

    # ========================================================================
    # RÉFÉRENCE
    # ========================================================================

    def get_symbol_search(self, query: str) -> Dict:
        """
        Rechercher un symbole

        Args:
            query: Requête de recherche

        Returns:
            Résultats de recherche
        """
        logger.info(f"🔍 Twelve Data: Recherche '{query}'")

        endpoint = "symbol_search"
        params = {'symbol': query}

        data = self._make_request(endpoint, params)

        if 'data' in data:
            logger.info(f"✅ {len(data['data'])} résultats trouvés")

        return data

    def get_exchanges(self) -> Dict:
        """
        Récupérer la liste des exchanges

        Returns:
            Liste des exchanges
        """
        logger.info("🏦 Twelve Data: Liste des exchanges")

        endpoint = "exchanges"
        params = {}

        data = self._make_request(endpoint, params)

        if 'data' in data:
            logger.info(f"✅ {len(data['data'])} exchanges")

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
            "requests_today": self.requests_today,
            "max_requests_per_day": self.max_requests_per_day,
            "requests_remaining": self.max_requests_per_day - self.requests_today,
            "usage_percentage": (self.requests_today / self.max_requests_per_day) * 100
        }


# Singleton
_twelvedata_collector_instance = None

def get_twelvedata_collector() -> TwelveDataCollector:
    """Obtenir l'instance singleton du Twelve Data collector"""
    global _twelvedata_collector_instance

    if _twelvedata_collector_instance is None:
        _twelvedata_collector_instance = TwelveDataCollector()

    return _twelvedata_collector_instance
