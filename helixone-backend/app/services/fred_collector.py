"""
Service de collecte de données macroéconomiques via FRED API
Federal Reserve Economic Data - Données macro USA de qualité institutionnelle (GRATUIT et ILLIMITÉ)
"""

from fredapi import Fred
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

# Clé API FRED (gratuit: https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = os.getenv("FRED_API_KEY", "your_fred_api_key_here")  # Remplacer par votre clé

# Séries économiques importantes (codes FRED)
FRED_SERIES = {
    # Taux d'intérêt
    'FED_FUNDS_RATE': 'DFF',  # Federal Funds Effective Rate
    'TREASURY_10Y': 'DGS10',  # 10-Year Treasury Constant Maturity Rate
    'TREASURY_2Y': 'DGS2',    # 2-Year Treasury Constant Maturity Rate
    'TREASURY_30Y': 'DGS30',  # 30-Year Treasury Constant Maturity Rate
    'TREASURY_1Y': 'DGS1',    # 1-Year Treasury Constant Maturity Rate

    # Inflation
    'CPI': 'CPIAUCSL',        # Consumer Price Index
    'CORE_CPI': 'CPILFESL',   # Core CPI (sans food & energy)
    'PCE': 'PCE',             # Personal Consumption Expenditures
    'CORE_PCE': 'PCEPILFE',   # Core PCE
    'PPI': 'PPIACO',          # Producer Price Index

    # PIB
    'GDP': 'GDP',             # Gross Domestic Product
    'REAL_GDP': 'GDPC1',      # Real GDP
    'GDP_GROWTH': 'A191RL1Q225SBEA',  # Real GDP Growth Rate

    # Emploi
    'UNEMPLOYMENT': 'UNRATE',          # Unemployment Rate
    'NONFARM_PAYROLLS': 'PAYEMS',      # All Employees, Total Nonfarm
    'JOBLESS_CLAIMS': 'ICSA',          # Initial Jobless Claims
    'PARTICIPATION_RATE': 'CIVPART',    # Labor Force Participation Rate

    # Marché immobilier
    'HOUSING_STARTS': 'HOUST',         # Housing Starts
    'HOME_SALES': 'HSN1F',             # New Home Sales
    'CASE_SHILLER': 'CSUSHPISA',       # S&P/Case-Shiller Home Price Index

    # Ventes et consommation
    'RETAIL_SALES': 'RSXFS',           # Retail Sales
    'CONSUMER_SENTIMENT': 'UMCSENT',   # University of Michigan Consumer Sentiment

    # Production
    'INDUSTRIAL_PRODUCTION': 'INDPRO', # Industrial Production Index
    'CAPACITY_UTILIZATION': 'TCU',     # Capacity Utilization
    'ISM_MANUFACTURING': 'MANEMP',     # ISM Manufacturing Employment Index

    # Monnaie et crédit
    'M1': 'M1SL',                      # M1 Money Supply
    'M2': 'M2SL',                      # M2 Money Supply
    'TOTAL_CREDIT': 'TOTLL',           # Total Consumer Credit Outstanding

    # Indices boursiers (FRED)
    'SP500': 'SP500',                  # S&P 500
    'NASDAQ': 'NASDAQCOM',             # NASDAQ Composite Index
    'VIX': 'VIXCLS',                   # CBOE Volatility Index

    # Dette
    'FEDERAL_DEBT': 'GFDEBTN',         # Federal Debt Total Public Debt
    'DEBT_GDP_RATIO': 'GFDEGDQ188S',   # Federal Debt to GDP Ratio

    # Commerce international
    'TRADE_BALANCE': 'BOPGSTB',        # Trade Balance
    'EXPORTS': 'EXPGS',                # Exports of Goods and Services
    'IMPORTS': 'IMPGS',                # Imports of Goods and Services
}


class FREDCollector:
    """
    Collecteur de données macroéconomiques FRED

    Features:
    - 500,000+ séries économiques disponibles
    - Données historiques jusqu'à 100+ ans
    - Qualité institutionnelle (Federal Reserve)
    - Gratuit et illimité

    Catégories:
    - Taux d'intérêt (Fed Funds, Treasury yields)
    - Inflation (CPI, PCE, PPI)
    - PIB et croissance économique
    - Emploi (chômage, non-farm payrolls)
    - Marché immobilier
    - Production industrielle
    - Monnaie et crédit
    """

    def __init__(self, api_key: str = FRED_API_KEY):
        """
        Initialiser le collecteur FRED

        Args:
            api_key: Clé API FRED (gratuit)
        """
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)
        self.series_cache = {}

        logger.info(f"✅ FREDCollector initialisé (clé: {api_key[:8]}...)")

    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Récupérer une série économique

        Args:
            series_id: Code de la série FRED (ex: 'DFF', 'CPIAUCSL')
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)

        Returns:
            pandas Series avec les données
        """
        try:
            logger.info(f"📊 FRED: Collecte série {series_id}")

            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )

            logger.info(f"✅ {series_id}: {len(data)} observations collectées")

            return data

        except Exception as e:
            logger.error(f"❌ Erreur collecte FRED {series_id}: {e}")
            raise

    def get_series_info(self, series_id: str) -> Dict:
        """
        Récupérer les métadonnées d'une série

        Args:
            series_id: Code de la série FRED

        Returns:
            Dict avec title, units, frequency, etc.
        """
        try:
            logger.info(f"ℹ️  FRED: Métadonnées pour {series_id}")

            info = self.fred.get_series_info(series_id)

            metadata = {
                'id': info['id'],
                'title': info['title'],
                'units': info['units'],
                'frequency': info['frequency'],
                'seasonal_adjustment': info.get('seasonal_adjustment', 'Not Seasonally Adjusted'),
                'last_updated': info['last_updated'],
                'popularity': info.get('popularity', 0),
                'notes': info.get('notes', '')
            }

            logger.info(f"✅ {series_id}: {metadata['title']}")

            return metadata

        except Exception as e:
            logger.error(f"❌ Erreur métadonnées FRED {series_id}: {e}")
            raise

    def get_all_key_indicators(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.Series]:
        """
        Récupérer tous les indicateurs clés prédéfinis

        Args:
            start_date: Date de début (défaut: 10 ans en arrière)
            end_date: Date de fin (défaut: aujourd'hui)

        Returns:
            Dict {nom_indicateur: Series}
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=3650)  # 10 ans

        if end_date is None:
            end_date = datetime.now()

        indicators = {}

        for name, series_id in FRED_SERIES.items():
            try:
                data = self.get_series(series_id, start_date, end_date)
                indicators[name] = data
                logger.info(f"✅ {name}: {len(data)} observations")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de collecter {name}: {e}")
                continue

        logger.info(f"📊 {len(indicators)}/{len(FRED_SERIES)} indicateurs collectés")

        return indicators

    def get_interest_rates(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Récupérer tous les taux d'intérêt

        Returns:
            DataFrame avec Fed Funds, Treasury yields
        """
        rates = {
            'fed_funds': self.get_series('DFF', start_date, end_date),
            'treasury_1y': self.get_series('DGS1', start_date, end_date),
            'treasury_2y': self.get_series('DGS2', start_date, end_date),
            'treasury_10y': self.get_series('DGS10', start_date, end_date),
            'treasury_30y': self.get_series('DGS30', start_date, end_date),
        }

        df = pd.DataFrame(rates)
        logger.info(f"✅ Taux d'intérêt: {len(df)} observations")

        return df

    def get_inflation_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Récupérer toutes les données d'inflation

        Returns:
            DataFrame avec CPI, PCE, PPI
        """
        inflation = {
            'cpi': self.get_series('CPIAUCSL', start_date, end_date),
            'core_cpi': self.get_series('CPILFESL', start_date, end_date),
            'pce': self.get_series('PCE', start_date, end_date),
            'core_pce': self.get_series('PCEPILFE', start_date, end_date),
            'ppi': self.get_series('PPIACO', start_date, end_date),
        }

        df = pd.DataFrame(inflation)
        logger.info(f"✅ Données d'inflation: {len(df)} observations")

        return df

    def get_employment_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Récupérer toutes les données d'emploi

        Returns:
            DataFrame avec unemployment, payrolls, etc.
        """
        employment = {
            'unemployment_rate': self.get_series('UNRATE', start_date, end_date),
            'nonfarm_payrolls': self.get_series('PAYEMS', start_date, end_date),
            'jobless_claims': self.get_series('ICSA', start_date, end_date),
            'participation_rate': self.get_series('CIVPART', start_date, end_date),
        }

        df = pd.DataFrame(employment)
        logger.info(f"✅ Données d'emploi: {len(df)} observations")

        return df

    def get_gdp_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Récupérer les données de PIB

        Returns:
            DataFrame avec GDP nominal, real, growth
        """
        gdp = {
            'gdp': self.get_series('GDP', start_date, end_date),
            'real_gdp': self.get_series('GDPC1', start_date, end_date),
            'gdp_growth': self.get_series('A191RL1Q225SBEA', start_date, end_date),
        }

        df = pd.DataFrame(gdp)
        logger.info(f"✅ Données de PIB: {len(df)} observations")

        return df

    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """
        Rechercher des séries FRED par mots-clés

        Args:
            search_text: Texte de recherche
            limit: Nombre de résultats (défaut: 10)

        Returns:
            DataFrame avec les séries trouvées
        """
        try:
            logger.info(f"🔍 FRED: Recherche '{search_text}'")

            results = self.fred.search(search_text, limit=limit)

            logger.info(f"✅ {len(results)} séries trouvées")

            return results

        except Exception as e:
            logger.error(f"❌ Erreur recherche FRED: {e}")
            raise

    def get_yield_curve(self, date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Récupérer la courbe des taux (yield curve) pour une date donnée

        Args:
            date: Date (défaut: dernière date disponible)

        Returns:
            Dict {maturity: yield}
        """
        maturities = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '3Y': 'DGS3',
            '5Y': 'DGS5',
            '7Y': 'DGS7',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30',
        }

        yield_curve = {}

        for maturity, series_id in maturities.items():
            try:
                data = self.get_series(series_id, start_date=date, end_date=date)
                if len(data) > 0:
                    yield_curve[maturity] = data.iloc[-1]
            except:
                continue

        logger.info(f"✅ Yield curve: {len(yield_curve)} maturités")

        return yield_curve

    def calculate_yield_spread(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Calculer le spread 10Y-2Y (indicateur de récession)

        Un spread négatif (inversion de la courbe) précède souvent une récession

        Returns:
            Series avec le spread 10Y-2Y
        """
        treasury_10y = self.get_series('DGS10', start_date, end_date)
        treasury_2y = self.get_series('DGS2', start_date, end_date)

        spread = treasury_10y - treasury_2y

        logger.info(f"✅ Yield spread 10Y-2Y: {len(spread)} observations")
        logger.info(f"📊 Spread actuel: {spread.iloc[-1]:.2f}%")

        if spread.iloc[-1] < 0:
            logger.warning("⚠️ ALERTE: Courbe inversée (spread négatif) - Risque de récession!")

        return spread


# Instance globale pour réutilisation
_fred_collector = None

def get_fred_collector(api_key: str = None) -> FREDCollector:
    """
    Obtenir l'instance du collecteur FRED (singleton)

    Args:
        api_key: Clé API FRED (optionnel)

    Returns:
        Instance FREDCollector
    """
    global _fred_collector

    if _fred_collector is None:
        _fred_collector = FREDCollector(api_key=api_key or FRED_API_KEY)

    return _fred_collector
