"""
OECD (Organisation for Economic Co-operation and Development) Data Collector
Source: https://stats.oecd.org/

GRATUIT - ILLIMITÉ - Pas de clé API requise
Données: Indicateurs économiques OCDE, développement, emploi, éducation, santé

Author: HelixOne Team
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

OECD_BASE_URL = "https://stats.oecd.org/sdmx-json/data"


class OECDCollector:
    """
    Collecteur de données OECD
    GRATUIT et ILLIMITÉ
    """

    def __init__(self):
        """Initialiser le collecteur OECD"""
        self.base_url = OECD_BASE_URL
        logger.info("✅ OECD Collector initialisé (GRATUIT - ILLIMITÉ)")

    def _make_request(self, dataset: str, filter_exp: str = "all", params: Dict = None) -> Dict:
        """
        Faire une requête à l'API OECD

        Args:
            dataset: Dataset OECD (QNA, MEI, SNA_TABLE1, etc.)
            filter_exp: Expression de filtre (ex: "USA.GDP.VOBARSA")
            params: Paramètres additionnels

        Returns:
            Réponse JSON
        """
        url = f"{self.base_url}/{dataset}/{filter_exp}"

        if params is None:
            params = {}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête OECD {dataset}/{filter_exp}: {e}")
            raise

    # ========================================================================
    # GDP & NATIONAL ACCOUNTS
    # ========================================================================

    def get_gdp(
        self,
        country: str = 'USA',
        start_time: str = '2020',
        end_time: str = '2024'
    ) -> Dict:
        """
        Récupérer le PIB

        Args:
            country: Code pays (USA, FRA, GBR, DEU, etc.)
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Données de PIB
        """
        logger.info(f"📊 OECD: PIB pour {country}")

        # QNA = Quarterly National Accounts
        # B1_GE = GDP
        dataset = "QNA"
        filter_exp = f"{country}.B1_GE.VBARSA.Q"  # Volume, seasonally adjusted

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ PIB {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur PIB: {e}")
            raise

    def get_gdp_growth(
        self,
        country: str = 'USA',
        start_time: str = '2020',
        end_time: str = '2024'
    ) -> Dict:
        """
        Récupérer la croissance du PIB

        Args:
            country: Code pays
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Taux de croissance PIB
        """
        logger.info(f"📈 OECD: Croissance PIB pour {country}")

        dataset = "QNA"
        filter_exp = f"{country}.B1_GE.GPSA.Q"  # Growth rate, seasonally adjusted

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ Croissance PIB {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur croissance PIB: {e}")
            raise

    # ========================================================================
    # MAIN ECONOMIC INDICATORS (MEI)
    # ========================================================================

    def get_unemployment_rate(
        self,
        country: str = 'USA',
        start_time: str = '2020-01',
        end_time: str = '2024-12'
    ) -> Dict:
        """
        Récupérer le taux de chômage

        Args:
            country: Code pays
            start_time: Période de début (YYYY-MM)
            end_time: Période de fin (YYYY-MM)

        Returns:
            Taux de chômage
        """
        logger.info(f"💼 OECD MEI: Taux de chômage pour {country}")

        dataset = "MEI"
        filter_exp = f"{country}.LRUNTTTT.STSA.M"  # Unemployment rate, seasonally adjusted

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ Taux de chômage {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux chômage: {e}")
            raise

    def get_inflation_cpi(
        self,
        country: str = 'USA',
        start_time: str = '2020-01',
        end_time: str = '2024-12'
    ) -> Dict:
        """
        Récupérer l'inflation CPI

        Args:
            country: Code pays
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Inflation CPI
        """
        logger.info(f"📈 OECD MEI: Inflation CPI pour {country}")

        dataset = "MEI"
        filter_exp = f"{country}.CPALTT01.IXOB.M"  # CPI All items

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ Inflation CPI {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur inflation: {e}")
            raise

    def get_interest_rates(
        self,
        country: str = 'USA',
        start_time: str = '2020-01',
        end_time: str = '2024-12'
    ) -> Dict:
        """
        Récupérer les taux d'intérêt

        Args:
            country: Code pays
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Taux d'intérêt
        """
        logger.info(f"💰 OECD MEI: Taux d'intérêt pour {country}")

        dataset = "MEI"
        filter_exp = f"{country}.IR3TIB01.ST.M"  # 3-month interbank rate

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ Taux d'intérêt {country} récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux intérêt: {e}")
            raise

    def get_industrial_production(
        self,
        country: str = 'USA',
        start_time: str = '2020-01',
        end_time: str = '2024-12'
    ) -> Dict:
        """
        Récupérer la production industrielle

        Args:
            country: Code pays
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Production industrielle
        """
        logger.info(f"🏭 OECD MEI: Production industrielle pour {country}")

        dataset = "MEI"
        filter_exp = f"{country}.PRINTO01.IXOB.M"  # Industrial production index

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ Production industrielle {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur production industrielle: {e}")
            raise

    # ========================================================================
    # COMPOSITE LEADING INDICATORS (CLI)
    # ========================================================================

    def get_cli(
        self,
        country: str = 'USA',
        start_time: str = '2020-01',
        end_time: str = '2024-12'
    ) -> Dict:
        """
        Récupérer les indicateurs avancés composites (CLI)

        Args:
            country: Code pays
            start_time: Période de début
            end_time: Période de fin

        Returns:
            CLI
        """
        logger.info(f"📊 OECD MEI: CLI pour {country}")

        dataset = "MEI"
        filter_exp = f"{country}.LOLITOAA.STSA.M"  # CLI, amplitude adjusted

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ CLI {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur CLI: {e}")
            raise

    # ========================================================================
    # TRADE & BALANCE OF PAYMENTS
    # ========================================================================

    def get_trade_balance(
        self,
        country: str = 'USA',
        start_time: str = '2020-01',
        end_time: str = '2024-12'
    ) -> Dict:
        """
        Récupérer la balance commerciale

        Args:
            country: Code pays
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Balance commerciale
        """
        logger.info(f"📦 OECD MEI: Balance commerciale pour {country}")

        dataset = "MEI"
        filter_exp = f"{country}.XTIMVA01.CXMLSA.M"  # Exports of goods and services

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ Balance commerciale {country} récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur balance commerciale: {e}")
            raise

    # ========================================================================
    # LABOR MARKET
    # ========================================================================

    def get_employment_rate(
        self,
        country: str = 'USA',
        start_time: str = '2020-Q1',
        end_time: str = '2024-Q4'
    ) -> Dict:
        """
        Récupérer le taux d'emploi

        Args:
            country: Code pays
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Taux d'emploi
        """
        logger.info(f"💼 OECD: Taux d'emploi pour {country}")

        dataset = "QNA"
        filter_exp = f"{country}.B1_GE.EMPSA.Q"  # Employment, seasonally adjusted

        params = {
            'startTime': start_time,
            'endTime': end_time
        }

        try:
            data = self._make_request(dataset, filter_exp, params)
            logger.info(f"✅ Taux d'emploi {country} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux emploi: {e}")
            raise

    # ========================================================================
    # UTILS
    # ========================================================================

    def get_economic_dashboard(
        self,
        country: str = 'USA',
        start_time: str = '2020',
        end_time: str = '2024'
    ) -> Dict:
        """
        Récupérer un tableau de bord économique complet

        Args:
            country: Code pays
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Dashboard économique
        """
        logger.info(f"📊 OECD: Dashboard économique pour {country}")

        dashboard = {}

        # Convertir les années en périodes mensuelles pour MEI
        start_month = f"{start_time}-01"
        end_month = f"{end_time}-12"

        try:
            # PIB
            try:
                dashboard['gdp'] = self.get_gdp(country, start_time, end_time)
            except:
                dashboard['gdp'] = None

            # Croissance PIB
            try:
                dashboard['gdp_growth'] = self.get_gdp_growth(country, start_time, end_time)
            except:
                dashboard['gdp_growth'] = None

            # Chômage
            try:
                dashboard['unemployment'] = self.get_unemployment_rate(country, start_month, end_month)
            except:
                dashboard['unemployment'] = None

            # Inflation
            try:
                dashboard['inflation'] = self.get_inflation_cpi(country, start_month, end_month)
            except:
                dashboard['inflation'] = None

            # Taux d'intérêt
            try:
                dashboard['interest_rates'] = self.get_interest_rates(country, start_month, end_month)
            except:
                dashboard['interest_rates'] = None

            # Production industrielle
            try:
                dashboard['industrial_production'] = self.get_industrial_production(country, start_month, end_month)
            except:
                dashboard['industrial_production'] = None

            # CLI
            try:
                dashboard['cli'] = self.get_cli(country, start_month, end_month)
            except:
                dashboard['cli'] = None

            logger.info(f"✅ Dashboard économique {country} assemblé")
            return dashboard

        except Exception as e:
            logger.error(f"❌ Erreur dashboard: {e}")
            raise

    def get_country_comparison(
        self,
        countries: List[str],
        indicator: str = 'gdp',
        start_time: str = '2020',
        end_time: str = '2024'
    ) -> Dict:
        """
        Comparer un indicateur pour plusieurs pays

        Args:
            countries: Liste de codes pays
            indicator: Indicateur ('gdp', 'unemployment', 'inflation', etc.)
            start_time: Période de début
            end_time: Période de fin

        Returns:
            Comparaison multi-pays
        """
        logger.info(f"🌍 OECD: Comparaison {indicator} pour {countries}")

        comparison = {}

        for country in countries:
            try:
                if indicator == 'gdp':
                    comparison[country] = self.get_gdp(country, start_time, end_time)
                elif indicator == 'unemployment':
                    comparison[country] = self.get_unemployment_rate(country, f"{start_time}-01", f"{end_time}-12")
                elif indicator == 'inflation':
                    comparison[country] = self.get_inflation_cpi(country, f"{start_time}-01", f"{end_time}-12")
                else:
                    comparison[country] = None
            except Exception as e:
                logger.warning(f"⚠️  Erreur pour {country}: {e}")
                comparison[country] = None

        logger.info(f"✅ Comparaison {indicator} assemblée")
        return comparison


# Singleton
_oecd_collector_instance = None

def get_oecd_collector() -> OECDCollector:
    """Obtenir l'instance singleton du OECD collector"""
    global _oecd_collector_instance

    if _oecd_collector_instance is None:
        _oecd_collector_instance = OECDCollector()

    return _oecd_collector_instance
