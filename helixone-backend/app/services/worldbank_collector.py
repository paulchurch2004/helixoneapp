"""
World Bank Data Collector
Source: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

GRATUIT - ILLIMITÉ - Pas de clé API requise
Données: Macro global, PIB, population, indicateurs de développement (200+ pays)

Author: HelixOne Team
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

WORLDBANK_BASE_URL = "https://api.worldbank.org/v2"


class WorldBankCollector:
    """
    Collecteur de données World Bank
    GRATUIT et ILLIMITÉ
    """

    def __init__(self):
        """Initialiser le collecteur World Bank"""
        self.base_url = WORLDBANK_BASE_URL
        logger.info("✅ World Bank Collector initialisé (GRATUIT - ILLIMITÉ)")

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Faire une requête à l'API World Bank

        Args:
            endpoint: Endpoint
            params: Paramètres

        Returns:
            Réponse JSON
        """
        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}

        # Format JSON par défaut
        params['format'] = 'json'
        params['per_page'] = params.get('per_page', 100)

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête World Bank {endpoint}: {e}")
            raise

    # ========================================================================
    # INDICATEURS ÉCONOMIQUES
    # ========================================================================

    def get_indicator(
        self,
        indicator: str,
        country: str = "all",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> List[Dict]:
        """
        Récupérer un indicateur économique

        Indicateurs populaires:
        - NY.GDP.MKTP.CD: PIB (USD courant)
        - NY.GDP.PCAP.CD: PIB par habitant
        - FP.CPI.TOTL.ZG: Inflation CPI
        - SL.UEM.TOTL.ZS: Taux de chômage
        - SP.POP.TOTL: Population totale
        - GC.DOD.TOTL.GD.ZS: Dette publique (% PIB)

        Args:
            indicator: Code indicateur
            country: Code pays (ISO2) ou "all"
            start_year: Année de début
            end_year: Année de fin

        Returns:
            Données de l'indicateur
        """
        logger.info(f"🌍 World Bank: {indicator} pour {country}")

        endpoint = f"country/{country}/indicator/{indicator}"

        params = {}
        if start_year:
            params['date'] = f"{start_year}:{end_year or datetime.now().year}"

        data = self._make_request(endpoint, params)

        # data[0] = metadata, data[1] = actual data
        if len(data) > 1 and data[1]:
            logger.info(f"✅ {len(data[1])} points de données récupérés")
            return data[1]

        return []

    def get_gdp(
        self,
        country: str = "USA",
        start_year: int = 2000
    ) -> List[Dict]:
        """Récupérer le PIB"""
        return self.get_indicator("NY.GDP.MKTP.CD", country, start_year)

    def get_gdp_per_capita(
        self,
        country: str = "USA",
        start_year: int = 2000
    ) -> List[Dict]:
        """Récupérer le PIB par habitant"""
        return self.get_indicator("NY.GDP.PCAP.CD", country, start_year)

    def get_inflation(
        self,
        country: str = "USA",
        start_year: int = 2000
    ) -> List[Dict]:
        """Récupérer l'inflation CPI"""
        return self.get_indicator("FP.CPI.TOTL.ZG", country, start_year)

    def get_unemployment(
        self,
        country: str = "USA",
        start_year: int = 2000
    ) -> List[Dict]:
        """Récupérer le taux de chômage"""
        return self.get_indicator("SL.UEM.TOTL.ZS", country, start_year)

    def get_population(
        self,
        country: str = "USA",
        start_year: int = 2000
    ) -> List[Dict]:
        """Récupérer la population totale"""
        return self.get_indicator("SP.POP.TOTL", country, start_year)

    def get_public_debt(
        self,
        country: str = "USA",
        start_year: int = 2000
    ) -> List[Dict]:
        """Récupérer la dette publique (% PIB)"""
        return self.get_indicator("GC.DOD.TOTL.GD.ZS", country, start_year)

    # ========================================================================
    # PAYS
    # ========================================================================

    def get_countries(self) -> List[Dict]:
        """
        Récupérer la liste de tous les pays

        Returns:
            Liste des pays
        """
        logger.info("🌍 World Bank: Liste des pays")

        endpoint = "country"
        params = {'per_page': 300}

        data = self._make_request(endpoint, params)

        if len(data) > 1 and data[1]:
            logger.info(f"✅ {len(data[1])} pays récupérés")
            return data[1]

        return []

    def get_country_info(self, country: str) -> Dict:
        """
        Récupérer les informations d'un pays

        Args:
            country: Code pays (ISO2)

        Returns:
            Informations du pays
        """
        logger.info(f"🌍 World Bank: Info pour {country}")

        endpoint = f"country/{country}"

        data = self._make_request(endpoint)

        if len(data) > 1 and data[1] and len(data[1]) > 0:
            logger.info(f"✅ {data[1][0].get('name', country)}")
            return data[1][0]

        return {}

    # ========================================================================
    # INDICATEURS MULTIPLES
    # ========================================================================

    def get_multiple_indicators(
        self,
        indicators: List[str],
        country: str = "USA",
        start_year: int = 2000
    ) -> Dict[str, List[Dict]]:
        """
        Récupérer plusieurs indicateurs à la fois

        Args:
            indicators: Liste de codes indicateurs
            country: Code pays
            start_year: Année de début

        Returns:
            Dict avec données pour chaque indicateur
        """
        logger.info(f"🌍 World Bank: {len(indicators)} indicateurs pour {country}")

        results = {}

        for indicator in indicators:
            try:
                data = self.get_indicator(indicator, country, start_year)
                results[indicator] = data
            except Exception as e:
                logger.warning(f"Erreur indicateur {indicator}: {e}")
                results[indicator] = []

        logger.info(f"✅ {len(results)} indicateurs récupérés")

        return results

    def get_economic_dashboard(
        self,
        country: str = "USA",
        start_year: int = 2010
    ) -> Dict:
        """
        Récupérer un dashboard économique complet pour un pays

        Args:
            country: Code pays
            start_year: Année de début

        Returns:
            Dashboard économique
        """
        logger.info(f"🌍 World Bank: Dashboard économique pour {country}")

        indicators = {
            'gdp': 'NY.GDP.MKTP.CD',
            'gdp_per_capita': 'NY.GDP.PCAP.CD',
            'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
            'inflation': 'FP.CPI.TOTL.ZG',
            'unemployment': 'SL.UEM.TOTL.ZS',
            'population': 'SP.POP.TOTL',
            'public_debt': 'GC.DOD.TOTL.GD.ZS',
            'exports': 'NE.EXP.GNFS.ZS',
            'imports': 'NE.IMP.GNFS.ZS',
            'fdi': 'BX.KLT.DINV.WD.GD.ZS'
        }

        dashboard = {}

        for name, code in indicators.items():
            try:
                data = self.get_indicator(code, country, start_year)
                dashboard[name] = data
            except Exception as e:
                logger.warning(f"Erreur {name}: {e}")
                dashboard[name] = []

        logger.info(f"✅ Dashboard complet pour {country}")

        return dashboard


# Singleton
_worldbank_collector_instance = None

def get_worldbank_collector() -> WorldBankCollector:
    """Obtenir l'instance singleton du World Bank collector"""
    global _worldbank_collector_instance

    if _worldbank_collector_instance is None:
        _worldbank_collector_instance = WorldBankCollector()

    return _worldbank_collector_instance
