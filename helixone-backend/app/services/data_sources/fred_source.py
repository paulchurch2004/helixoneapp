"""
Source de données FRED (Federal Reserve Economic Data)
API gratuite: ILLIMITÉ!
Documentation: https://fred.stlouisfed.org/docs/api/fred/
"""

import logging
from typing import Optional, List, Dict
from datetime import date, datetime
from decimal import Decimal
import aiohttp

from app.services.data_sources.base import BaseDataSource, DataUnavailableError
from app.core.config import settings

logger = logging.getLogger(__name__)


class FREDSource(BaseDataSource):
    """
    Source de données FRED (Federal Reserve)

    Capacités:
    - Données macro-économiques officielles
    - Taux d'intérêt (Fed Funds Rate, Treasury Yields)
    - Inflation (CPI, PCE)
    - PIB, Chômage, etc.
    - Données historiques complètes (plusieurs décennies)

    Limites:
    - AUCUNE limite de requêtes! 🎉
    - Données de très haute qualité (source officielle)
    - Pas de données d'actions individuelles (uniquement macro)
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Séries importantes pour l'analyse macro
    IMPORTANT_SERIES = {
        'fed_funds_rate': 'DFF',  # Federal Funds Effective Rate
        'treasury_10y': 'DGS10',  # 10-Year Treasury Constant Maturity Rate
        'treasury_2y': 'DGS2',    # 2-Year Treasury
        'cpi': 'CPIAUCSL',        # Consumer Price Index
        'core_cpi': 'CPILFESL',   # Core CPI (excluding food & energy)
        'unemployment': 'UNRATE',  # Unemployment Rate
        'gdp': 'GDP',             # Gross Domestic Product
        'sp500': 'SP500',         # S&P 500 Index
        'vix': 'VIXCLS',          # CBOE Volatility Index
        'dollar_index': 'DTWEXBGS',  # Trade Weighted U.S. Dollar Index
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client FRED

        Args:
            api_key: Clé API FRED (ou depuis settings)
        """
        self.api_key = api_key or settings.FRED_API_KEY

        if not self.api_key or self.api_key == "your_key_here":
            logger.warning("FRED API key not configured")

    @property
    def name(self) -> str:
        return "FRED"

    def is_available(self) -> bool:
        """Vérifie si la source est disponible"""
        return bool(self.api_key and self.api_key != "your_key_here")

    async def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Effectue une requête HTTP vers l'API FRED

        Args:
            endpoint: Endpoint de l'API
            params: Paramètres supplémentaires

        Returns:
            Réponse JSON
        """
        if params is None:
            params = {}

        params['api_key'] = self.api_key
        params['file_type'] = 'json'

        url = f"{self.BASE_URL}/{endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise DataUnavailableError(
                        self.name,
                        f"HTTP {response.status}: {await response.text()}"
                    )

                return await response.json()

    async def get_series(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Récupère une série temporelle FRED

        Args:
            series_id: ID de la série (ex: 'DFF' pour Fed Funds Rate)
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)

        Returns:
            Liste de points de données {date, value}
        """
        if not self.is_available():
            return []

        try:
            params = {'series_id': series_id}

            if start_date:
                params['observation_start'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['observation_end'] = end_date.strftime('%Y-%m-%d')

            data = await self._make_request('series/observations', params)

            if not data or 'observations' not in data:
                return []

            results = []
            for obs in data['observations']:
                # FRED retourne '.' pour les valeurs manquantes
                if obs['value'] != '.':
                    results.append({
                        'date': datetime.strptime(obs['date'], '%Y-%m-%d').date(),
                        'value': Decimal(obs['value'])
                    })

            logger.info(f"✅ Série FRED {series_id} récupérée: {len(results)} observations")
            return results

        except Exception as e:
            logger.error(f"FRED.get_series for {series_id} failed: {e}")
            return []

    async def get_macro_snapshot(self, as_of_date: Optional[date] = None) -> Dict:
        """
        Récupère un snapshot des principaux indicateurs macro

        Args:
            as_of_date: Date de référence (None = dernière valeur disponible)

        Returns:
            Dictionnaire avec les principaux indicateurs
        """
        if not self.is_available():
            return {}

        try:
            # Récupérer les dernières valeurs de chaque série importante
            snapshot = {}

            for name, series_id in self.IMPORTANT_SERIES.items():
                try:
                    observations = await self.get_series(
                        series_id,
                        start_date=as_of_date - timedelta(days=90) if as_of_date else None,
                        end_date=as_of_date
                    )

                    if observations:
                        # Prendre la dernière observation
                        latest = observations[-1]
                        snapshot[name] = {
                            'value': latest['value'],
                            'date': latest['date'],
                            'series_id': series_id
                        }
                except Exception as e:
                    logger.warning(f"Failed to get {name} ({series_id}): {e}")
                    continue

            logger.info(f"✅ Snapshot macro récupéré: {len(snapshot)} indicateurs")
            return snapshot

        except Exception as e:
            logger.error(f"FRED.get_macro_snapshot failed: {e}")
            return {}

    async def get_interest_rates(self) -> Dict[str, Decimal]:
        """
        Récupère les principaux taux d'intérêt

        Returns:
            Dictionnaire {nom: taux}
        """
        if not self.is_available():
            return {}

        rates = {}

        try:
            # Fed Funds Rate
            fed_funds = await self.get_series('DFF')
            if fed_funds:
                rates['fed_funds_rate'] = fed_funds[-1]['value']

            # Treasury Yields
            for maturity, series_id in [
                ('2y', 'DGS2'),
                ('5y', 'DGS5'),
                ('10y', 'DGS10'),
                ('30y', 'DGS30')
            ]:
                data = await self.get_series(series_id)
                if data:
                    rates[f'treasury_{maturity}'] = data[-1]['value']

            logger.info(f"✅ Taux d'intérêt récupérés: {len(rates)} taux")
            return rates

        except Exception as e:
            logger.error(f"FRED.get_interest_rates failed: {e}")
            return {}

    async def get_inflation_data(self) -> Dict:
        """
        Récupère les données d'inflation

        Returns:
            Dictionnaire avec CPI actuel et variations
        """
        if not self.is_available():
            return {}

        try:
            # CPI (last 12 months)
            from datetime import timedelta
            end_date = date.today()
            start_date = end_date - timedelta(days=365)

            cpi_data = await self.get_series('CPIAUCSL', start_date, end_date)

            if not cpi_data or len(cpi_data) < 2:
                return {}

            latest = cpi_data[-1]
            year_ago = cpi_data[0]

            # Calculer l'inflation YoY
            inflation_yoy = ((latest['value'] - year_ago['value']) / year_ago['value']) * 100

            result = {
                'current_cpi': latest['value'],
                'date': latest['date'],
                'inflation_yoy': inflation_yoy,
                'previous_year_cpi': year_ago['value']
            }

            logger.info(f"✅ Données d'inflation: CPI={latest['value']}, YoY={inflation_yoy:.2f}%")
            return result

        except Exception as e:
            logger.error(f"FRED.get_inflation_data failed: {e}")
            return {}

    # Note: FRED ne fournit pas de données d'actions individuelles
    # Les méthodes get_quote, get_historical, get_fundamentals ne sont pas implémentées
    # car FRED est une source de données macro uniquement

    async def get_quote(self, ticker: str) -> None:
        """FRED ne fournit pas de quotes d'actions"""
        return None

    async def get_historical(self, ticker: str, start_date: date, end_date: date, interval: str = "1d") -> None:
        """FRED ne fournit pas de données historiques d'actions"""
        return None

    async def get_fundamentals(self, ticker: str) -> None:
        """FRED ne fournit pas de fondamentaux d'actions"""
        return None


# Importer timedelta qui manquait
from datetime import timedelta
