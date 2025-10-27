"""
BIS (Bank for International Settlements) Data Collector
Source: https://data.bis.org/

GRATUIT - ILLIMITÉ - Pas de clé API requise
Données: Statistiques bancaires, marchés financiers, dérivés, dette, forex

Author: HelixOne Team
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# API a migré de data.bis.org vers stats.bis.org en 2024-2025
BIS_BASE_URL = "https://stats.bis.org/api/v1"


class BISCollector:
    """
    Collecteur de données BIS
    GRATUIT et ILLIMITÉ
    """

    def __init__(self):
        """Initialiser le collecteur BIS"""
        self.base_url = BIS_BASE_URL
        logger.info("✅ BIS Collector initialisé (GRATUIT - ILLIMITÉ)")

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Faire une requête à l'API BIS

        Args:
            endpoint: Endpoint de l'API
            params: Paramètres de requête

        Returns:
            Réponse JSON
        """
        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}

        # SDMX 2.1 API utilise Accept headers au lieu de format parameter
        headers = {
            'Accept': 'application/vnd.sdmx.data+json;version=1.0.0',
            'User-Agent': 'HelixOne/1.0 (Financial Data Platform)'
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête BIS {endpoint}: {e}")
            raise

    # ========================================================================
    # CREDIT STATISTICS
    # ========================================================================

    def get_credit_to_gdp(
        self,
        country: str = 'US',
        start_period: str = '2020-Q1',
        end_period: str = '2024-Q4'
    ) -> Dict:
        """
        Récupérer le ratio crédit/PIB

        Args:
            country: Code pays (US, FR, GB, etc.)
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Ratio crédit/PIB
        """
        logger.info(f"📊 BIS: Crédit/PIB pour {country}")

        # WEBSTATS_TOTAL_CREDIT_DATAFLOW
        # Q:US:P:A = Quarterly, US, Private non-financial sector, All sectors
        dataset = "WEBSTATS_CREDIT_DATAFLOW"
        key = f"Q.{country}.P.A"

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Crédit/PIB {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur crédit/PIB: {e}")
            raise

    def get_total_credit(
        self,
        country: str = 'US',
        start_period: str = '2020-Q1',
        end_period: str = '2024-Q4'
    ) -> Dict:
        """
        Récupérer le crédit total

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Crédit total
        """
        logger.info(f"💰 BIS: Crédit total pour {country}")

        dataset = "WEBSTATS_LONG_DATAFLOW"
        key = f"Q.{country}.N.628"  # Total credit to private non-financial sector

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Crédit total {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur crédit total: {e}")
            raise

    # ========================================================================
    # DEBT SECURITIES STATISTICS
    # ========================================================================

    def get_debt_securities(
        self,
        country: str = 'US',
        start_period: str = '2020-Q1',
        end_period: str = '2024-Q4'
    ) -> Dict:
        """
        Récupérer les titres de dette

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Titres de dette
        """
        logger.info(f"📜 BIS: Titres de dette pour {country}")

        dataset = "WEBSTATS_DEBTSEC_DATAFLOW"
        key = f"Q.{country}.S.A.A.A.770.A"  # All issuers, all currencies

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Titres de dette {country} récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur titres dette: {e}")
            raise

    # ========================================================================
    # EXCHANGE RATES
    # ========================================================================

    def get_effective_exchange_rate(
        self,
        country: str = 'US',
        rate_type: str = 'R',  # R=Real, N=Nominal
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer le taux de change effectif

        Args:
            country: Code pays
            rate_type: 'R' pour réel, 'N' pour nominal
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Taux de change effectif
        """
        logger.info(f"💱 BIS: Taux de change effectif {rate_type} pour {country}")

        dataset = "WEBSTATS_EER_DATAFLOW"
        key = f"M.{country}.{rate_type}.N"  # Monthly, Country, Type, Narrow

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Taux change effectif {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux change: {e}")
            raise

    # ========================================================================
    # PROPERTY PRICES
    # ========================================================================

    def get_property_prices(
        self,
        country: str = 'US',
        start_period: str = '2020-Q1',
        end_period: str = '2024-Q4'
    ) -> Dict:
        """
        Récupérer les prix de l'immobilier

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Prix immobilier
        """
        logger.info(f"🏠 BIS: Prix immobilier pour {country}")

        dataset = "WEBSTATS_RPPI_DATAFLOW"
        key = f"Q.{country}.N"  # Quarterly, Country, Nominal

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Prix immobilier {country} récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur prix immobilier: {e}")
            raise

    # ========================================================================
    # DERIVATIVES STATISTICS
    # ========================================================================

    def get_otc_derivatives(
        self,
        start_period: str = '2020-12',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer les statistiques sur les dérivés OTC

        Args:
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Statistiques dérivés OTC
        """
        logger.info(f"📊 BIS: Dérivés OTC")

        dataset = "WEBSTATS_OTC_DERIV_DATAFLOW"
        key = "H.A.A.A.A.A"  # Half-yearly, All markets

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Dérivés OTC récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur dérivés OTC: {e}")
            raise

    # ========================================================================
    # CENTRAL BANK POLICY RATES
    # ========================================================================

    def get_policy_rates(
        self,
        country: str = 'US',
        start_period: str = '2020-01',
        end_period: str = '2024-12'
    ) -> Dict:
        """
        Récupérer les taux directeurs des banques centrales

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Taux directeurs
        """
        logger.info(f"💰 BIS: Taux directeurs pour {country}")

        dataset = "WEBSTATS_CBPOL_DATAFLOW"
        key = f"M.{country}.0"  # Monthly, Country, Policy rate

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Taux directeurs {country} récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux directeurs: {e}")
            raise

    # ========================================================================
    # GLOBAL LIQUIDITY
    # ========================================================================

    def get_global_liquidity(
        self,
        start_period: str = '2020-Q1',
        end_period: str = '2024-Q4'
    ) -> Dict:
        """
        Récupérer les indicateurs de liquidité globale

        Args:
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Liquidité globale
        """
        logger.info(f"💧 BIS: Liquidité globale")

        dataset = "WEBSTATS_GLI_DATAFLOW"
        key = "Q.F.A.A.A.5J.A"  # Quarterly, All currencies

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Liquidité globale récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur liquidité globale: {e}")
            raise

    # ========================================================================
    # BANKING STATISTICS
    # ========================================================================

    def get_banking_stats(
        self,
        country: str = 'US',
        start_period: str = '2020-Q1',
        end_period: str = '2024-Q4'
    ) -> Dict:
        """
        Récupérer les statistiques bancaires

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Statistiques bancaires
        """
        logger.info(f"🏦 BIS: Statistiques bancaires pour {country}")

        dataset = "WEBSTATS_CBS_DATAFLOW"
        key = f"Q.{country}.A.A.A.A.A.A"  # Consolidated banking statistics

        params = {
            'startPeriod': start_period,
            'endPeriod': end_period
        }

        try:
            data = self._make_request(f"data/{dataset}/{key}", params)
            logger.info(f"✅ Stats bancaires {country} récupérées")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur stats bancaires: {e}")
            raise

    # ========================================================================
    # UTILS
    # ========================================================================

    def get_financial_dashboard(
        self,
        country: str = 'US',
        start_period: str = '2020-Q1',
        end_period: str = '2024-Q4'
    ) -> Dict:
        """
        Récupérer un tableau de bord financier complet

        Args:
            country: Code pays
            start_period: Période de début
            end_period: Période de fin

        Returns:
            Dashboard financier
        """
        logger.info(f"📊 BIS: Dashboard financier pour {country}")

        dashboard = {}

        try:
            # Crédit/PIB
            try:
                dashboard['credit_to_gdp'] = self.get_credit_to_gdp(country, start_period, end_period)
            except:
                dashboard['credit_to_gdp'] = None

            # Crédit total
            try:
                dashboard['total_credit'] = self.get_total_credit(country, start_period, end_period)
            except:
                dashboard['total_credit'] = None

            # Titres de dette
            try:
                dashboard['debt_securities'] = self.get_debt_securities(country, start_period, end_period)
            except:
                dashboard['debt_securities'] = None

            # Prix immobilier
            try:
                dashboard['property_prices'] = self.get_property_prices(country, start_period, end_period)
            except:
                dashboard['property_prices'] = None

            # Taux directeurs
            try:
                start_month = start_period[:7] if '-Q' in start_period else start_period
                end_month = end_period[:7] + '-12' if '-Q' in end_period else end_period
                dashboard['policy_rates'] = self.get_policy_rates(country, start_month, end_month)
            except:
                dashboard['policy_rates'] = None

            # Taux de change effectif
            try:
                start_month = start_period[:7] if '-Q' in start_period else start_period
                end_month = end_period[:7] + '-12' if '-Q' in end_period else end_period
                dashboard['exchange_rate'] = self.get_effective_exchange_rate(country, 'R', start_month, end_month)
            except:
                dashboard['exchange_rate'] = None

            logger.info(f"✅ Dashboard financier {country} assemblé")
            return dashboard

        except Exception as e:
            logger.error(f"❌ Erreur dashboard: {e}")
            raise


# Singleton
_bis_collector_instance = None

def get_bis_collector() -> BISCollector:
    """Obtenir l'instance singleton du BIS collector"""
    global _bis_collector_instance

    if _bis_collector_instance is None:
        _bis_collector_instance = BISCollector()

    return _bis_collector_instance
