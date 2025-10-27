"""
ECB (European Central Bank) Data Collector
Source: https://data.ecb.europa.eu/

GRATUIT - ILLIMITÉ - Pas de clé API requise
Données: Macro Europe, taux BCE, inflation zone euro, PIB

Author: HelixOne Team
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data"


class ECBCollector:
    """
    Collecteur de données ECB
    GRATUIT et ILLIMITÉ
    """

    def __init__(self):
        """Initialiser le collecteur ECB"""
        self.base_url = ECB_BASE_URL
        logger.info("✅ ECB Collector initialisé (GRATUIT - ILLIMITÉ)")

    def _make_request(self, flow: str, key: str, params: Dict = None) -> str:
        """
        Faire une requête à l'API ECB

        Args:
            flow: Dataflow (ex: EXR, FM, ICP)
            key: Clé (dimensions séparées par .)
            params: Paramètres

        Returns:
            XML data
        """
        url = f"{self.base_url}/{flow}/{key}"

        if params is None:
            params = {}

        # Format par défaut
        if 'format' not in params:
            params['format'] = 'jsondata'

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()

            # Retourner JSON ou XML selon format
            if params['format'] == 'jsondata':
                return response.json()
            else:
                return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête ECB {flow}/{key}: {e}")
            raise

    # ========================================================================
    # TAUX D'INTÉRÊT
    # ========================================================================

    def get_key_interest_rates(self) -> Dict:
        """
        Récupérer les taux d'intérêt clés de la BCE

        Returns:
            Taux directeurs BCE
        """
        logger.info("💰 ECB: Taux d'intérêt clés")

        # FM = Financial Market Data
        # B.U2.EUR.4F.KR.MRR_FR.LEV = Main refinancing operations
        flow = "FM"
        key = "B.U2.EUR.4F.KR.MRR_FR.LEV"

        try:
            data = self._make_request(flow, key)
            logger.info("✅ Taux BCE récupérés")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux BCE: {e}")
            raise

    def get_euro_exchange_rates(self, currency: str = "USD") -> Dict:
        """
        Récupérer les taux de change EUR/XXX

        Args:
            currency: Code devise (USD, GBP, JPY, etc.)

        Returns:
            Taux de change
        """
        logger.info(f"💱 ECB: Taux de change EUR/{currency}")

        # EXR = Exchange Rates
        # D = Daily
        # SP00 = Spot
        # A = Average
        flow = "EXR"
        key = f"D.{currency}.EUR.SP00.A"

        try:
            data = self._make_request(flow, key)
            logger.info(f"✅ EUR/{currency} récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur taux change: {e}")
            raise

    # ========================================================================
    # INFLATION
    # ========================================================================

    def get_hicp_inflation(self) -> Dict:
        """
        Récupérer l'inflation HICP (Harmonised Index of Consumer Prices)

        Returns:
            Inflation zone euro
        """
        logger.info("📈 ECB: Inflation HICP zone euro")

        # ICP = Index of Consumer Prices
        flow = "ICP"
        key = "M.U2.N.000000.4.ANR"  # Monthly, Euro area, All items, Annual rate

        try:
            data = self._make_request(flow, key)
            logger.info("✅ Inflation HICP récupérée")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur inflation: {e}")
            raise

    # ========================================================================
    # AGRÉGATS MONÉTAIRES
    # ========================================================================

    def get_m3_money_supply(self) -> Dict:
        """
        Récupérer M3 (masse monétaire)

        Returns:
            M3 zone euro
        """
        logger.info("💵 ECB: Masse monétaire M3")

        # BSI = Balance Sheet Items
        # M = Monthly, U2 = Euro area, M3 aggregate
        flow = "BSI"
        key = "M.U2.Y.V.M30.X.1.U2.2300.Z01.E"

        try:
            data = self._make_request(flow, key)
            logger.info("✅ M3 récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur M3: {e}")
            raise

    # ========================================================================
    # PIB
    # ========================================================================

    def get_gdp_euro_area(self) -> Dict:
        """
        Récupérer le PIB de la zone euro

        Returns:
            PIB zone euro
        """
        logger.info("📊 ECB: PIB zone euro")

        # MNA = Main National Accounts
        flow = "MNA"
        key = "Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N"

        try:
            data = self._make_request(flow, key)
            logger.info("✅ PIB zone euro récupéré")
            return data

        except Exception as e:
            logger.error(f"❌ Erreur PIB: {e}")
            raise


# Singleton
_ecb_collector_instance = None

def get_ecb_collector() -> ECBCollector:
    """Obtenir l'instance singleton du ECB collector"""
    global _ecb_collector_instance

    if _ecb_collector_instance is None:
        _ecb_collector_instance = ECBCollector()

    return _ecb_collector_instance
