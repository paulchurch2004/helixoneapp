"""
Eurostat Data Collector
Source: https://ec.europa.eu/eurostat/

GRATUIT - ILLIMITÉ - Pas de clé API requise
Données: Statistiques européennes (économie, démographie, société, environnement)

Author: HelixOne Team
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


class EurostatCollector:
    """
    Collecteur de données Eurostat
    GRATUIT et ILLIMITÉ
    """

    def __init__(self):
        """Initialiser le collecteur Eurostat"""
        self.base_url = EUROSTAT_BASE_URL
        logger.info("✅ Eurostat Collector initialisé (GRATUIT - ILLIMITÉ)")

    def _make_request(self, dataset: str, params: Dict = None) -> Dict:
        """
        Faire une requête à l'API Eurostat

        Args:
            dataset: Code du dataset (nama_10_gdp, prc_hicp_midx, etc.)
            params: Paramètres de requête

        Returns:
            Réponse JSON
        """
        url = f"{self.base_url}/{dataset}"

        if params is None:
            params = {}

        # Format JSON par défaut
        if 'format' not in params:
            params['format'] = 'JSON'

        # Langue par défaut EN
        if 'lang' not in params:
            params['lang'] = 'en'

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête Eurostat {dataset}: {e}")
            raise

    # ========================================================================
    # GDP & NATIONAL ACCOUNTS
    # ========================================================================

    def get_gdp(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020',
        end_period: str = '2024'
    ) -> Dict:
        """
        Récupérer le PIB

        Args:
            country: Code géo (EU27_2020, FR, DE, IT, ES, etc.)
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Données de PIB
        """
        logger.info(f"📊 Eurostat: PIB pour {country}")

        # nama_10_gdp = National accounts aggregates
        dataset = "nama_10_gdp"

        params = {
            'geo': country,
            'na_item': 'B1GQ',  # Gross domestic product at market prices
            'unit': 'CP_MEUR',  # Current prices, million euro
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ PIB {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur PIB: {e}")
            raise

    def get_gdp_growth(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020',
        end_period: str = '2024'
    ) -> Dict:
        """
        Récupérer la croissance du PIB

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Croissance PIB
        """
        logger.info(f"📈 Eurostat: Croissance PIB pour {country}")

        dataset = "nama_10_gdp"

        params = {
            'geo': country,
            'na_item': 'B1GQ',
            'unit': 'CLV10_MEUR',  # Chain linked volumes
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Croissance PIB {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur croissance PIB: {e}")
            raise

    # ========================================================================
    # INFLATION (HICP)
    # ========================================================================

    def get_inflation_hicp(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer l'inflation HICP

        Args:
            country: Code géo
            start_period: Période de début (YYYY-MM)
            end_period: Période de fin (YYYY-MM)

        Returns:
            Inflation HICP
        """
        logger.info(f"📈 Eurostat: Inflation HICP pour {country}")

        # prc_hicp_midx = HICP monthly index
        dataset = "prc_hicp_midx"

        params = {
            'geo': country,
            'coicop': 'CP00',  # All-items HICP
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Inflation HICP {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur inflation: {e}")
            raise

    def get_inflation_annual_rate(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer le taux d'inflation annuel

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Taux d'inflation annuel
        """
        logger.info(f"📈 Eurostat: Taux inflation annuel pour {country}")

        # prc_hicp_manr = HICP annual rate of change
        dataset = "prc_hicp_manr"

        params = {
            'geo': country,
            'coicop': 'CP00',
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Taux inflation annuel {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux inflation: {e}")
            raise

    # ========================================================================
    # UNEMPLOYMENT
    # ========================================================================

    def get_unemployment_rate(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer le taux de chômage

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Taux de chômage
        """
        logger.info(f"💼 Eurostat: Taux chômage pour {country}")

        # une_rt_m = Unemployment rate, monthly
        dataset = "une_rt_m"

        params = {
            'geo': country,
            's_adj': 'SA',  # Seasonally adjusted
            'age': 'Y_GE15',  # 15 years or over
            'sex': 'T',  # Total
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Taux chômage {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux chômage: {e}")
            raise

    # ========================================================================
    # INDUSTRIAL PRODUCTION
    # ========================================================================

    def get_industrial_production(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer la production industrielle

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Production industrielle
        """
        logger.info(f"🏭 Eurostat: Production industrielle pour {country}")

        # sts_inpr_m = Production in industry, monthly
        dataset = "sts_inpr_m"

        params = {
            'geo': country,
            's_adj': 'SA',  # Seasonally adjusted
            'nace_r2': 'B-D',  # Industry
            'indic_bt': 'PROD',  # Production
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Production industrielle {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur production industrielle: {e}")
            raise

    # ========================================================================
    # BUSINESS & CONSUMER SURVEYS
    # ========================================================================

    def get_business_confidence(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer l'indicateur de confiance des entreprises

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Confiance des entreprises
        """
        logger.info(f"📊 Eurostat: Confiance entreprises pour {country}")

        # ei_bssi_m_r2 = Business and consumer surveys
        dataset = "ei_bssi_m_r2"

        params = {
            'geo': country,
            's_adj': 'SA',
            'indic': 'BS-ICI',  # Industrial confidence indicator
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Confiance entreprises {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur confiance entreprises: {e}")
            raise

    def get_consumer_confidence(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer l'indicateur de confiance des consommateurs

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Confiance des consommateurs
        """
        logger.info(f"📊 Eurostat: Confiance consommateurs pour {country}")

        dataset = "ei_bsco_m"

        params = {
            'geo': country,
            's_adj': 'SA',
            'indic': 'BS-CSMCI-BAL',  # Consumer confidence indicator
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Confiance consommateurs {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur confiance consommateurs: {e}")
            raise

    # ========================================================================
    # TRADE
    # ========================================================================

    def get_trade_balance(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer la balance commerciale

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Balance commerciale
        """
        logger.info(f"📦 Eurostat: Balance commerciale pour {country}")

        # ext_st_eu27_2020sitc = External trade
        dataset = "ext_lt_maineu"

        params = {
            'reporter': country,
            'partner': 'EXT_EU27_2020',  # Extra EU27
            'product': 'TOTAL',
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Balance commerciale {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur balance commerciale: {e}")
            raise

    # ========================================================================
    # POPULATION & DEMOGRAPHY
    # ========================================================================

    def get_population(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020',
        end_period: str = '2024'
    ) -> Dict:
        """
        Récupérer la population

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Population
        """
        logger.info(f"👥 Eurostat: Population pour {country}")

        # demo_pjan = Population on 1 January
        dataset = "demo_pjan"

        params = {
            'geo': country,
            'age': 'TOTAL',
            'sex': 'T',
            'sinceTimePeriod': start_period,
            'untilTimePeriod': end_period
        }

        try:
            data = self._make_request(dataset, params)
            logger.info(f"✅ Population {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur population: {e}")
            raise

    # ========================================================================
    # UTILS
    # ========================================================================

    def get_economic_dashboard(
        self,
        country: str = 'EU27_2020',
        start_period: str = '2020',
        end_period: str = '2024'
    ) -> Dict:
        """
        Récupérer un tableau de bord économique européen

        Args:
            country: Code géo
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Dashboard économique
        """
        logger.info(f"📊 Eurostat: Dashboard économique pour {country}")

        dashboard = {}

        # Convertir en périodes mensuelles pour certains indicateurs
        start_month = f"{start_period}-01"
        end_month = f"{end_period}-12"

        try:
            # PIB
            try:
                dashboard['gdp'] = self.get_gdp(country, start_period, end_period)
            except:
                dashboard['gdp'] = None

            # Croissance PIB
            try:
                dashboard['gdp_growth'] = self.get_gdp_growth(country, start_period, end_period)
            except:
                dashboard['gdp_growth'] = None

            # Inflation
            try:
                dashboard['inflation'] = self.get_inflation_annual_rate(country, start_month, end_month)
            except:
                dashboard['inflation'] = None

            # Chômage
            try:
                dashboard['unemployment'] = self.get_unemployment_rate(country, start_month, end_month)
            except:
                dashboard['unemployment'] = None

            # Production industrielle
            try:
                dashboard['industrial_production'] = self.get_industrial_production(country, start_month, end_month)
            except:
                dashboard['industrial_production'] = None

            # Confiance consommateurs
            try:
                dashboard['consumer_confidence'] = self.get_consumer_confidence(country, start_month, end_month)
            except:
                dashboard['consumer_confidence'] = None

            # Confiance entreprises
            try:
                dashboard['business_confidence'] = self.get_business_confidence(country, start_month, end_month)
            except:
                dashboard['business_confidence'] = None

            # Population
            try:
                dashboard['population'] = self.get_population(country, start_period, end_period)
            except:
                dashboard['population'] = None

            logger.info(f"✅ Dashboard économique {country} assemblé")
            return dashboard

        except Exception as e:
            logger.error(f"❌ Erreur dashboard: {e}")
            raise


# Singleton
_eurostat_collector_instance = None

def get_eurostat_collector() -> EurostatCollector:
    """Obtenir l'instance singleton du Eurostat collector"""
    global _eurostat_collector_instance

    if _eurostat_collector_instance is None:
        _eurostat_collector_instance = EurostatCollector()

    return _eurostat_collector_instance
