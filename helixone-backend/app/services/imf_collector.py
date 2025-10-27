"""
IMF (International Monetary Fund) Data Collector
Source: http://dataservices.imf.org/

GRATUIT - ILLIMITÉ - Pas de clé API requise
Données: Macro global, balance des paiements, finances publiques, indicateurs financiers

Author: HelixOne Team
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# API migrée de dataservices.imf.org vers sdmxcentral.imf.org en 2024-2025
# SDMX 2.1 API
IMF_BASE_URL = "https://sdmxcentral.imf.org/ws/public/sdmxapi/rest"


class IMFCollector:
    """
    Collecteur de données IMF
    GRATUIT et ILLIMITÉ
    """

    def __init__(self):
        """Initialiser le collecteur IMF"""
        self.base_url = IMF_BASE_URL
        logger.info("✅ IMF Collector initialisé (GRATUIT - ILLIMITÉ)")

    def _make_request(self, endpoint: str) -> Dict:
        """
        Faire une requête à l'API IMF

        Args:
            endpoint: Endpoint de l'API

        Returns:
            Réponse JSON
        """
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête IMF {endpoint}: {e}")
            raise

    # ========================================================================
    # INTERNATIONAL FINANCIAL STATISTICS (IFS)
    # ========================================================================

    def get_exchange_rates(
        self,
        country: str = 'US',
        start_period: str = '2020',
        end_period: str = '2025'
    ) -> Dict:
        """
        Récupérer les taux de change

        Args:
            country: Code pays (US, FR, GB, etc.)
            start_period: Période de début (YYYY)
            end_period: Période de fin (YYYY)

        Returns:
            Données de taux de change
        """
        logger.info(f"💱 IMF IFS: Taux de change pour {country}")

        # IFS = International Financial Statistics
        # ENDA_XDC_USD_RATE = End of period exchange rate (LCU per USD)
        database = "IFS"
        indicator = "ENDA_XDC_USD_RATE"
        frequency = "M"  # Monthly
        area = country

        endpoint = f"CompactData/{database}/{frequency}.{area}.{indicator}?startPeriod={start_period}&endPeriod={end_period}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ Taux de change {country} récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux de change: {e}")
            raise

    def get_inflation_rate(
        self,
        country: str = 'US',
        start_period: str = '2020',
        end_period: str = '2025'
    ) -> Dict:
        """
        Récupérer le taux d'inflation

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Données d'inflation
        """
        logger.info(f"📈 IMF IFS: Inflation pour {country}")

        database = "IFS"
        indicator = "PCPI_IX"  # Consumer Price Index
        frequency = "M"
        area = country

        endpoint = f"CompactData/{database}/{frequency}.{area}.{indicator}?startPeriod={start_period}&endPeriod={end_period}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ Inflation {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur inflation: {e}")
            raise

    def get_gdp(
        self,
        country: str = 'US',
        start_period: str = '2020',
        end_period: str = '2025'
    ) -> Dict:
        """
        Récupérer le PIB

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Données de PIB
        """
        logger.info(f"📊 IMF IFS: PIB pour {country}")

        database = "IFS"
        indicator = "NGDP_XDC"  # GDP in national currency
        frequency = "Q"  # Quarterly
        area = country

        endpoint = f"CompactData/{database}/{frequency}.{area}.{indicator}?startPeriod={start_period}&endPeriod={end_period}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ PIB {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur PIB: {e}")
            raise

    def get_interest_rates(
        self,
        country: str = 'US',
        start_period: str = '2020',
        end_period: str = '2025'
    ) -> Dict:
        """
        Récupérer les taux d'intérêt

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Données de taux d'intérêt
        """
        logger.info(f"💰 IMF IFS: Taux d'intérêt pour {country}")

        database = "IFS"
        indicator = "FIGB_PA"  # Government bond yield
        frequency = "M"
        area = country

        endpoint = f"CompactData/{database}/{frequency}.{area}.{indicator}?startPeriod={start_period}&endPeriod={end_period}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ Taux d'intérêt {country} récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux d'intérêt: {e}")
            raise

    # ========================================================================
    # BALANCE OF PAYMENTS (BOP)
    # ========================================================================

    def get_current_account(
        self,
        country: str = 'US',
        start_period: str = '2020',
        end_period: str = '2025'
    ) -> Dict:
        """
        Récupérer la balance courante

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Balance courante
        """
        logger.info(f"💵 IMF BOP: Balance courante pour {country}")

        database = "BOP"
        indicator = "BCA_BP6_USD"  # Current account balance
        frequency = "Q"
        area = country

        endpoint = f"CompactData/{database}/{frequency}.{area}.{indicator}?startPeriod={start_period}&endPeriod={end_period}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ Balance courante {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur balance courante: {e}")
            raise

    def get_trade_balance(
        self,
        country: str = 'US',
        start_period: str = '2020',
        end_period: str = '2025'
    ) -> Dict:
        """
        Récupérer la balance commerciale

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Balance commerciale
        """
        logger.info(f"📦 IMF BOP: Balance commerciale pour {country}")

        database = "BOP"
        indicator = "BGS_BP6_USD"  # Goods and services balance
        frequency = "Q"
        area = country

        endpoint = f"CompactData/{database}/{frequency}.{area}.{indicator}?startPeriod={start_period}&endPeriod={end_period}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ Balance commerciale {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur balance commerciale: {e}")
            raise

    # ========================================================================
    # WORLD ECONOMIC OUTLOOK (WEO)
    # ========================================================================

    def get_weo_gdp_growth(
        self,
        country: str = 'US',
        start_year: int = 2020
    ) -> Dict:
        """
        Récupérer les prévisions de croissance du PIB (WEO)

        Note: WEO a un endpoint différent

        Args:
            country: Code pays ISO (USA, FRA, etc.)
            start_year: Année de début

        Returns:
            Prévisions WEO
        """
        logger.info(f"📈 IMF WEO: Prévisions croissance PIB pour {country}")

        # Note: WEO endpoint est différent, utilise un autre format
        # Endpoint simplifié pour l'exemple
        try:
            # Pour WEO, nous pourrions utiliser l'API principale mais avec des limitations
            logger.warning("⚠️  WEO endpoint nécessite un accès différent")
            return {
                'note': 'WEO data requires specific endpoint access',
                'alternative': 'Use IFS GDP data instead'
            }

        except Exception as e:
            logger.error(f"❌ Erreur WEO: {e}")
            raise

    # ========================================================================
    # FINANCIAL SOUNDNESS INDICATORS (FSI)
    # ========================================================================

    def get_banking_soundness(
        self,
        country: str = 'US',
        start_period: str = '2020',
        end_period: str = '2025'
    ) -> Dict:
        """
        Récupérer les indicateurs de solidité bancaire

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Indicateurs FSI
        """
        logger.info(f"🏦 IMF FSI: Indicateurs bancaires pour {country}")

        database = "FSI"
        indicator = "FSCAPR_PA"  # Regulatory Tier 1 capital to risk-weighted assets
        frequency = "Q"
        area = country

        endpoint = f"CompactData/{database}/{frequency}.{area}.{indicator}?startPeriod={start_period}&endPeriod={end_period}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ Indicateurs bancaires {country} récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur FSI: {e}")
            raise

    # ========================================================================
    # MÉTADONNÉES
    # ========================================================================

    def get_countries_list(self, database: str = "IFS") -> Dict:
        """
        Récupérer la liste des pays disponibles

        Args:
            database: Base de données (IFS, BOP, FSI, etc.)

        Returns:
            Liste des pays
        """
        logger.info(f"🌍 IMF: Liste des pays pour {database}")

        endpoint = f"CodeList/{database}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ Liste des pays récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur liste pays: {e}")
            raise

    def get_indicators_list(self, database: str = "IFS") -> Dict:
        """
        Récupérer la liste des indicateurs disponibles

        Args:
            database: Base de données

        Returns:
            Liste des indicateurs
        """
        logger.info(f"📊 IMF: Liste des indicateurs pour {database}")

        endpoint = f"CodeList/{database}"

        try:
            data = self._make_request(endpoint)
            logger.info(f"✅ Liste des indicateurs récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur liste indicateurs: {e}")
            raise

    # ========================================================================
    # UTILS
    # ========================================================================

    def get_macro_dashboard(
        self,
        country: str = 'US',
        start_period: str = '2020',
        end_period: str = '2025'
    ) -> Dict:
        """
        Récupérer un tableau de bord macro complet

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Dashboard macro
        """
        logger.info(f"📊 IMF: Dashboard macro pour {country}")

        dashboard = {}

        try:
            # Tenter de récupérer plusieurs indicateurs
            try:
                dashboard['gdp'] = self.get_gdp(country, start_period, end_period)
            except:
                dashboard['gdp'] = None

            try:
                dashboard['inflation'] = self.get_inflation_rate(country, start_period, end_period)
            except:
                dashboard['inflation'] = None

            try:
                dashboard['interest_rates'] = self.get_interest_rates(country, start_period, end_period)
            except:
                dashboard['interest_rates'] = None

            try:
                dashboard['exchange_rates'] = self.get_exchange_rates(country, start_period, end_period)
            except:
                dashboard['exchange_rates'] = None

            logger.info(f"✅ Dashboard macro {country} assemblé")
            return dashboard

        except Exception as e:
            logger.error(f"❌ Erreur dashboard: {e}")
            raise


# Singleton
_imf_collector_instance = None

def get_imf_collector() -> IMFCollector:
    """Obtenir l'instance singleton du IMF collector"""
    global _imf_collector_instance

    if _imf_collector_instance is None:
        _imf_collector_instance = IMFCollector()

    return _imf_collector_instance
