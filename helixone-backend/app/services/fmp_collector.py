"""
Financial Modeling Prep (FMP) Data Collector
Source: https://financialmodelingprep.com/developer/docs/

Tier gratuit: 250 requêtes/jour
Données: États financiers, ratios, dividendes, ownership, insider trading

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

FMP_API_KEY = os.getenv('FMP_API_KEY', '')
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPCollector:
    """
    Collecteur de données Financial Modeling Prep
    """

    def __init__(self, api_key: str = FMP_API_KEY):
        """
        Initialiser le collecteur FMP

        Args:
            api_key: Clé API FMP
        """
        if not api_key:
            logger.warning("⚠️  FMP API key non configurée")

        self.api_key = api_key
        self.base_url = FMP_BASE_URL

        # Rate limiting: 250 req/jour
        self.requests_today = 0
        self.max_requests_per_day = 250
        self.last_request_time = time.time()
        self.min_request_interval = 0.5  # 2 requêtes/seconde max

        logger.info("✅ FMP Collector initialisé")

    def _rate_limit(self):
        """Rate limiting automatique"""
        # Vérifier limite quotidienne
        if self.requests_today >= self.max_requests_per_day:
            logger.warning(f"⚠️  Limite quotidienne FMP atteinte ({self.max_requests_per_day})")
            raise Exception(f"Limite quotidienne FMP atteinte: {self.max_requests_per_day} requêtes")

        # Rate limiting entre requêtes
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        self.last_request_time = time.time()
        self.requests_today += 1

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Faire une requête à l'API FMP

        Args:
            endpoint: Endpoint de l'API (ex: "income-statement/AAPL")
            params: Paramètres additionnels

        Returns:
            Réponse JSON
        """
        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"

        # Ajouter la clé API
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Vérifier erreur FMP
            if isinstance(data, dict) and 'Error Message' in data:
                raise Exception(f"FMP Error: {data['Error Message']}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête FMP {endpoint}: {e}")
            raise

    # ========================================================================
    # MARKET DATA & QUOTES
    # ========================================================================

    def get_quote(self, symbol: str) -> List[Dict]:
        """
        Récupérer la quote temps réel pour un symbole

        Args:
            symbol: Symbole du ticker (ex: AAPL)

        Returns:
            Liste contenant la quote (1 élément)
        """
        logger.info(f"💵 FMP: Quote pour {symbol}")

        endpoint = f"quote/{symbol}"
        data = self._make_request(endpoint)

        if data:
            logger.info(f"✅ {symbol}: Quote récupérée - ${data[0].get('price', 'N/A')}")

        return data

    def get_real_time_price(self, symbol: str) -> Dict:
        """
        Récupérer le prix temps réel simple

        Args:
            symbol: Symbole du ticker

        Returns:
            Dict avec le prix
        """
        logger.info(f"💰 FMP: Prix temps réel pour {symbol}")

        endpoint = f"quote-short/{symbol}"
        data = self._make_request(endpoint)

        if data:
            logger.info(f"✅ {symbol}: Prix ${data[0].get('price', 'N/A')}")

        return data[0] if data else {}

    # ========================================================================
    # ÉTATS FINANCIERS
    # ========================================================================

    def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict]:
        """
        Récupérer le compte de résultat (Income Statement)

        Args:
            symbol: Symbole du ticker
            period: "annual" ou "quarter"
            limit: Nombre de périodes (max 10 en gratuit)

        Returns:
            Liste de comptes de résultat
        """
        logger.info(f"📄 FMP: Income Statement pour {symbol} ({period})")

        endpoint = f"income-statement/{symbol}"
        params = {'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {len(data)} income statements")

        return data

    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict]:
        """
        Récupérer le bilan (Balance Sheet)

        Args:
            symbol: Symbole du ticker
            period: "annual" ou "quarter"
            limit: Nombre de périodes

        Returns:
            Liste de bilans
        """
        logger.info(f"📄 FMP: Balance Sheet pour {symbol} ({period})")

        endpoint = f"balance-sheet-statement/{symbol}"
        params = {'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {len(data)} balance sheets")

        return data

    def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict]:
        """
        Récupérer le tableau de flux de trésorerie (Cash Flow Statement)

        Args:
            symbol: Symbole du ticker
            period: "annual" ou "quarter"
            limit: Nombre de périodes

        Returns:
            Liste de cash flow statements
        """
        logger.info(f"📄 FMP: Cash Flow pour {symbol} ({period})")

        endpoint = f"cash-flow-statement/{symbol}"
        params = {'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {len(data)} cash flow statements")

        return data

    # ========================================================================
    # RATIOS FINANCIERS
    # ========================================================================

    def get_financial_ratios(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict]:
        """
        Récupérer tous les ratios financiers calculés (50+ ratios)

        Ratios inclus:
        - Profitabilité: ROE, ROA, marges
        - Liquidité: Current ratio, Quick ratio
        - Solvabilité: Debt/Equity, Interest coverage
        - Efficacité: Asset turnover, Inventory turnover
        - Valorisation: P/E, P/B, P/S, EV/EBITDA

        Args:
            symbol: Symbole du ticker
            period: "annual" ou "quarter"
            limit: Nombre de périodes

        Returns:
            Liste de ratios financiers
        """
        logger.info(f"📊 FMP: Ratios financiers pour {symbol}")

        endpoint = f"ratios/{symbol}"
        params = {'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {len(data)} périodes de ratios")

        return data

    def get_key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict]:
        """
        Récupérer les métriques clés (market cap, P/E, EPS, etc.)

        Args:
            symbol: Symbole du ticker
            period: "annual" ou "quarter"
            limit: Nombre de périodes

        Returns:
            Liste de key metrics
        """
        logger.info(f"📊 FMP: Key Metrics pour {symbol}")

        endpoint = f"key-metrics/{symbol}"
        params = {'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {len(data)} périodes de metrics")

        return data

    def get_financial_growth(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict]:
        """
        Récupérer les taux de croissance (revenue growth, EPS growth, etc.)

        Args:
            symbol: Symbole du ticker
            period: "annual" ou "quarter"
            limit: Nombre de périodes

        Returns:
            Liste de financial growth metrics
        """
        logger.info(f"📈 FMP: Financial Growth pour {symbol}")

        endpoint = f"financial-growth/{symbol}"
        params = {'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {len(data)} périodes de growth")

        return data

    # ========================================================================
    # COMPANY PROFILE
    # ========================================================================

    def get_company_profile(self, symbol: str) -> List[Dict]:
        """
        Récupérer le profil complet de l'entreprise

        Args:
            symbol: Symbole du ticker

        Returns:
            Profil de l'entreprise
        """
        logger.info(f"🏢 FMP: Company Profile pour {symbol}")

        endpoint = f"profile/{symbol}"

        data = self._make_request(endpoint)

        if data and len(data) > 0:
            company = data[0]
            logger.info(f"✅ {symbol}: {company.get('companyName', 'N/A')}")

        return data

    # ========================================================================
    # DIVIDENDES
    # ========================================================================

    def get_dividends_historical(self, symbol: str) -> List[Dict]:
        """
        Récupérer l'historique complet des dividendes

        Args:
            symbol: Symbole du ticker

        Returns:
            Liste des dividendes historiques
        """
        logger.info(f"💰 FMP: Dividendes historiques pour {symbol}")

        endpoint = f"historical-price-full/stock_dividend/{symbol}"

        data = self._make_request(endpoint)

        if 'historical' in data:
            dividends = data['historical']
            logger.info(f"✅ {symbol}: {len(dividends)} dividendes historiques")
            return dividends

        return []

    def get_dividends_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Récupérer le calendrier des dividendes

        Args:
            from_date: Date de début (YYYY-MM-DD)
            to_date: Date de fin (YYYY-MM-DD)

        Returns:
            Calendrier des dividendes
        """
        logger.info("💰 FMP: Calendrier dividendes")

        endpoint = "stock_dividend_calendar"

        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {len(data)} dividendes prévus")

        return data

    # ========================================================================
    # OWNERSHIP & INSIDER TRADING
    # ========================================================================

    def get_insider_trading(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Récupérer les transactions d'initiés (insider trading)

        Args:
            symbol: Symbole du ticker
            limit: Nombre de transactions

        Returns:
            Liste des insider trades
        """
        logger.info(f"👤 FMP: Insider Trading pour {symbol}")

        endpoint = f"insider-trading"
        params = {'symbol': symbol, 'limit': limit}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {len(data)} insider transactions")

        return data

    def get_institutional_holders(self, symbol: str) -> List[Dict]:
        """
        Récupérer les détenteurs institutionnels

        Args:
            symbol: Symbole du ticker

        Returns:
            Liste des institutional holders
        """
        logger.info(f"🏦 FMP: Institutional Holders pour {symbol}")

        endpoint = f"institutional-holder/{symbol}"

        data = self._make_request(endpoint)

        logger.info(f"✅ {symbol}: {len(data)} institutional holders")

        return data

    # ========================================================================
    # EARNINGS & ESTIMATES
    # ========================================================================

    def get_earnings_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Récupérer le calendrier des publications de résultats

        Args:
            from_date: Date de début (YYYY-MM-DD)
            to_date: Date de fin (YYYY-MM-DD)

        Returns:
            Calendrier earnings
        """
        logger.info("📅 FMP: Earnings Calendar")

        endpoint = "earning_calendar"

        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {len(data)} earnings prévus")

        return data

    def get_analyst_estimates(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict]:
        """
        Récupérer les estimations des analystes

        Args:
            symbol: Symbole du ticker
            period: "annual" ou "quarter"
            limit: Nombre de périodes

        Returns:
            Estimations analystes
        """
        logger.info(f"📊 FMP: Analyst Estimates pour {symbol}")

        endpoint = f"analyst-estimates/{symbol}"
        params = {'period': period, 'limit': limit}

        data = self._make_request(endpoint, params)

        logger.info(f"✅ {symbol}: {len(data)} périodes d'estimations")

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


# Singleton pour partager l'instance
_fmp_collector_instance = None

def get_fmp_collector() -> FMPCollector:
    """Obtenir l'instance singleton du FMP collector"""
    global _fmp_collector_instance

    if _fmp_collector_instance is None:
        _fmp_collector_instance = FMPCollector()

    return _fmp_collector_instance
