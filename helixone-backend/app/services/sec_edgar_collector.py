"""
SEC Edgar Data Collector
Source: https://www.sec.gov/edgar

GRATUIT - ILLIMITÉ - Pas de clé API requise
Données: Filings (10-K, 10-Q, 8-K), Company facts, insider transactions

Author: HelixOne Team
"""

import requests
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

SEC_BASE_URL = "https://data.sec.gov"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"


class SECEdgarCollector:
    """
    Collecteur de données SEC Edgar
    GRATUIT et ILLIMITÉ (avec User-Agent requis)
    """

    def __init__(self):
        """Initialiser le collecteur SEC Edgar"""
        self.base_url = SEC_BASE_URL
        self.submissions_url = SEC_SUBMISSIONS_URL

        # User-Agent OBLIGATOIRE par SEC
        # Note: Pas de Host header - laissons requests le gérer automatiquement
        self.headers = {
            'User-Agent': 'HelixOne Financial Platform contact@helixone.com',
            'Accept-Encoding': 'gzip, deflate'
        }

        # Rate limiting: 10 requêtes/seconde max (recommandé)
        self.min_request_interval = 0.1
        self.last_request_time = time.time()

        logger.info("✅ SEC Edgar Collector initialisé (GRATUIT - ILLIMITÉ)")

    def _rate_limit(self):
        """Rate limiting automatique (10 req/sec max)"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str) -> Dict:
        """
        Faire une requête à l'API SEC

        Args:
            url: URL complète

        Returns:
            Réponse JSON
        """
        self._rate_limit()

        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête SEC {url}: {e}")
            raise

    # ========================================================================
    # COMPANY INFORMATION
    # ========================================================================

    def get_company_tickers(self) -> List[Dict]:
        """
        Récupérer la liste de toutes les entreprises avec leurs CIK

        Returns:
            Liste des entreprises
        """
        logger.info("🏢 SEC: Liste des company tickers")

        # Note: company_tickers.json est sur www.sec.gov, pas data.sec.gov
        url = "https://www.sec.gov/files/company_tickers.json"

        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            companies = list(data.values())
            logger.info(f"✅ {len(companies)} entreprises récupérées")

            return companies

        except Exception as e:
            logger.error(f"❌ Erreur company tickers: {e}")
            raise

    def get_cik_by_ticker(self, ticker: str) -> Optional[str]:
        """
        Trouver le CIK d'une entreprise par son ticker

        Args:
            ticker: Symbole du ticker

        Returns:
            CIK (Central Index Key)
        """
        logger.info(f"🔍 SEC: Recherche CIK pour {ticker}")

        companies = self.get_company_tickers()

        for company in companies:
            if company.get('ticker', '').upper() == ticker.upper():
                cik = str(company.get('cik_str')).zfill(10)
                logger.info(f"✅ {ticker}: CIK {cik}")
                return cik

        logger.warning(f"⚠️  CIK non trouvé pour {ticker}")
        return None

    def get_company_submissions(self, cik: str) -> Dict:
        """
        Récupérer tous les filings d'une entreprise

        Args:
            cik: CIK de l'entreprise (10 chiffres)

        Returns:
            Métadonnées entreprise + liste filings
        """
        logger.info(f"📄 SEC: Submissions pour CIK {cik}")

        # CIK doit être 10 chiffres
        cik = str(cik).zfill(10)

        url = f"{self.submissions_url}/CIK{cik}.json"

        data = self._make_request(url)

        if 'filings' in data and 'recent' in data['filings']:
            filings_count = len(data['filings']['recent'].get('accessionNumber', []))
            logger.info(f"✅ {filings_count} filings récents pour {data.get('name', cik)}")

        return data

    # ========================================================================
    # FILINGS
    # ========================================================================

    def get_filings_by_type(
        self,
        cik: str,
        form_type: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Récupérer les filings d'un type spécifique

        Form types populaires:
        - 10-K: Rapport annuel
        - 10-Q: Rapport trimestriel
        - 8-K: Événements majeurs
        - 4: Insider transactions
        - 13F-HR: Institutional holdings
        - DEF 14A: Proxy statement

        Args:
            cik: CIK de l'entreprise
            form_type: Type de formulaire
            limit: Nombre de filings

        Returns:
            Liste des filings
        """
        logger.info(f"📄 SEC: Filings {form_type} pour CIK {cik}")

        submissions = self.get_company_submissions(cik)

        if 'filings' not in submissions or 'recent' not in submissions['filings']:
            return []

        recent = submissions['filings']['recent']

        # Extraire les filings du type demandé
        filings = []
        for i in range(len(recent.get('accessionNumber', []))):
            if recent['form'][i] == form_type:
                filing = {
                    'accessionNumber': recent['accessionNumber'][i],
                    'filingDate': recent['filingDate'][i],
                    'reportDate': recent.get('reportDate', [''])[i],
                    'acceptanceDateTime': recent.get('acceptanceDateTime', [''])[i],
                    'primaryDocument': recent.get('primaryDocument', [''])[i],
                    'primaryDocDescription': recent.get('primaryDocDescription', [''])[i],
                }
                filings.append(filing)

                if len(filings) >= limit:
                    break

        logger.info(f"✅ {len(filings)} filings {form_type} trouvés")

        return filings

    def get_10k_filings(self, cik: str, limit: int = 5) -> List[Dict]:
        """Récupérer les rapports annuels 10-K"""
        return self.get_filings_by_type(cik, '10-K', limit)

    def get_10q_filings(self, cik: str, limit: int = 8) -> List[Dict]:
        """Récupérer les rapports trimestriels 10-Q"""
        return self.get_filings_by_type(cik, '10-Q', limit)

    def get_8k_filings(self, cik: str, limit: int = 10) -> List[Dict]:
        """Récupérer les 8-K (événements majeurs)"""
        return self.get_filings_by_type(cik, '8-K', limit)

    def get_insider_transactions(self, cik: str, limit: int = 20) -> List[Dict]:
        """Récupérer les Form 4 (insider transactions)"""
        return self.get_filings_by_type(cik, '4', limit)

    # ========================================================================
    # COMPANY FACTS (XBRL)
    # ========================================================================

    def get_company_facts(self, cik: str) -> Dict:
        """
        Récupérer les faits financiers XBRL d'une entreprise

        Contient:
        - Données financières structurées
        - Historique complet
        - US-GAAP taxonomy

        Args:
            cik: CIK de l'entreprise

        Returns:
            Facts XBRL
        """
        logger.info(f"📊 SEC: Company Facts pour CIK {cik}")

        cik = str(cik).zfill(10)

        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json"

        data = self._make_request(url)

        if 'facts' in data:
            logger.info(f"✅ Facts récupérés pour {data.get('entityName', cik)}")

        return data

    def get_revenue_history(self, cik: str) -> List[Dict]:
        """
        Récupérer l'historique des revenus via XBRL

        Args:
            cik: CIK de l'entreprise

        Returns:
            Historique revenus
        """
        logger.info(f"💰 SEC: Revenue history pour CIK {cik}")

        facts = self.get_company_facts(cik)

        if 'facts' not in facts:
            return []

        # Chercher Revenues dans US-GAAP
        us_gaap = facts['facts'].get('us-gaap', {})

        # Plusieurs noms possibles pour revenue
        revenue_concepts = [
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'SalesRevenueNet'
        ]

        for concept in revenue_concepts:
            if concept in us_gaap:
                revenue_data = us_gaap[concept]
                if 'units' in revenue_data and 'USD' in revenue_data['units']:
                    revenues = revenue_data['units']['USD']
                    logger.info(f"✅ {len(revenues)} périodes de revenus")
                    return revenues

        logger.warning("⚠️  Revenus non trouvés dans XBRL")
        return []

    # ========================================================================
    # UTILS
    # ========================================================================

    def get_filing_url(self, cik: str, accession_number: str, primary_document: str) -> str:
        """
        Construire l'URL d'un filing

        Args:
            cik: CIK
            accession_number: Accession number (ex: 0001628280-21-017455)
            primary_document: Document principal

        Returns:
            URL du filing
        """
        cik = str(cik).zfill(10)
        # Enlever les tirets de l'accession number pour l'URL
        acc_no_dashes = accession_number.replace('-', '')

        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dashes}/{primary_document}"

        return url


# Singleton
_sec_edgar_collector_instance = None

def get_sec_edgar_collector() -> SECEdgarCollector:
    """Obtenir l'instance singleton du SEC Edgar collector"""
    global _sec_edgar_collector_instance

    if _sec_edgar_collector_instance is None:
        _sec_edgar_collector_instance = SECEdgarCollector()

    return _sec_edgar_collector_instance
