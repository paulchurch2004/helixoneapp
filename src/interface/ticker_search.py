"""
🔍 TICKER SEARCH - Système de recherche intelligent avec autocomplete

Permet de rechercher des tickers par:
- Symbole (AAPL, MSFT, etc.)
- Nom complet (Apple, Microsoft, etc.)
- Recherche partielle (App → Apple, Mic → Microsoft)

Features:
- Base de données de 100+ tickers populaires
- Autocomplete avec suggestions
- Recherche fuzzy (tolérante aux fautes)
"""

from typing import List, Dict, Tuple, Optional
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

# Base de données des tickers populaires avec noms complets
POPULAR_TICKERS = {
    # Tech Giants
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'GOOG': 'Alphabet Inc. Class C',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc. (Facebook)',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'AMD': 'Advanced Micro Devices Inc.',
    'INTC': 'Intel Corporation',
    'NFLX': 'Netflix Inc.',
    'ADBE': 'Adobe Inc.',
    'CRM': 'Salesforce Inc.',
    'ORCL': 'Oracle Corporation',
    'CSCO': 'Cisco Systems Inc.',

    # Finance
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corp.',
    'WFC': 'Wells Fargo & Company',
    'GS': 'Goldman Sachs Group Inc.',
    'MS': 'Morgan Stanley',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.',
    'AXP': 'American Express Company',
    'BLK': 'BlackRock Inc.',
    'SCHW': 'Charles Schwab Corporation',

    # Consumer
    'WMT': 'Walmart Inc.',
    'HD': 'Home Depot Inc.',
    'NKE': 'Nike Inc.',
    'MCD': "McDonald's Corporation",
    'SBUX': 'Starbucks Corporation',
    'DIS': 'Walt Disney Company',
    'COST': 'Costco Wholesale Corporation',
    'TGT': 'Target Corporation',
    'LOW': "Lowe's Companies Inc.",
    'TJX': 'TJX Companies Inc.',

    # Healthcare
    'JNJ': 'Johnson & Johnson',
    'UNH': 'UnitedHealth Group Inc.',
    'PFE': 'Pfizer Inc.',
    'ABBV': 'AbbVie Inc.',
    'TMO': 'Thermo Fisher Scientific Inc.',
    'ABT': 'Abbott Laboratories',
    'MRK': 'Merck & Co. Inc.',
    'LLY': 'Eli Lilly and Company',
    'CVS': 'CVS Health Corporation',
    'AMGN': 'Amgen Inc.',

    # Energy
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger NV',
    'EOG': 'EOG Resources Inc.',

    # Industrial
    'BA': 'Boeing Company',
    'CAT': 'Caterpillar Inc.',
    'GE': 'General Electric Company',
    'MMM': '3M Company',
    'HON': 'Honeywell International Inc.',
    'UPS': 'United Parcel Service Inc.',
    'LMT': 'Lockheed Martin Corporation',
    'RTX': 'RTX Corporation (Raytheon)',

    # Telecom
    'T': 'AT&T Inc.',
    'VZ': 'Verizon Communications Inc.',
    'TMUS': 'T-Mobile US Inc.',

    # Automotive
    'F': 'Ford Motor Company',
    'GM': 'General Motors Company',
    'TM': 'Toyota Motor Corporation',

    # Crypto/Blockchain
    'COIN': 'Coinbase Global Inc.',
    'MARA': 'Marathon Digital Holdings',
    'RIOT': 'Riot Platforms Inc.',

    # Semiconductors
    'TSM': 'Taiwan Semiconductor Manufacturing',
    'ASML': 'ASML Holding NV',
    'AVGO': 'Broadcom Inc.',
    'QCOM': 'QUALCOMM Inc.',
    'TXN': 'Texas Instruments Inc.',
    'MU': 'Micron Technology Inc.',

    # Social Media / Communication
    'SNAP': 'Snap Inc.',
    'PINS': 'Pinterest Inc.',
    'TWTR': 'Twitter Inc. (X)',
    'SPOT': 'Spotify Technology SA',

    # E-commerce / Retail
    'SHOP': 'Shopify Inc.',
    'ETSY': 'Etsy Inc.',
    'EBAY': 'eBay Inc.',
    'BABA': 'Alibaba Group',
    'JD': 'JD.com Inc.',

    # Entertainment / Gaming
    'EA': 'Electronic Arts Inc.',
    'ATVI': 'Activision Blizzard Inc.',
    'TTWO': 'Take-Two Interactive',
    'RBLX': 'Roblox Corporation',
    'U': 'Unity Software Inc.',

    # Cloud / Software
    'SNOW': 'Snowflake Inc.',
    'PLTR': 'Palantir Technologies Inc.',
    'DDOG': 'Datadog Inc.',
    'CRWD': 'CrowdStrike Holdings Inc.',
    'ZM': 'Zoom Video Communications',
    'DOCU': 'DocuSign Inc.',

    # Electric Vehicles
    'NIO': 'NIO Inc.',
    'RIVN': 'Rivian Automotive Inc.',
    'LCID': 'Lucid Group Inc.',

    # Index ETFs
    'SPY': 'SPDR S&P 500 ETF Trust',
    'QQQ': 'Invesco QQQ Trust',
    'IWM': 'iShares Russell 2000 ETF',
    'DIA': 'SPDR Dow Jones Industrial Average ETF',
    'VOO': 'Vanguard S&P 500 ETF',
    'VTI': 'Vanguard Total Stock Market ETF',
}


class TickerSearchEngine:
    """Moteur de recherche intelligent pour tickers"""

    def __init__(self):
        self.tickers_db = POPULAR_TICKERS.copy()
        # Créer un mapping inversé nom → ticker pour recherche rapide
        self.name_to_ticker = {name.lower(): ticker for ticker, name in self.tickers_db.items()}
        logger.info(f"🔍 TickerSearchEngine initialisé avec {len(self.tickers_db)} tickers")

    def search(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Recherche intelligente de tickers

        Args:
            query: Texte de recherche (peut être ticker ou nom)
            limit: Nombre maximum de résultats

        Returns:
            Liste de dicts avec 'ticker', 'name', 'score'
        """
        if not query or len(query) < 1:
            return []

        query = query.strip().upper()
        results = []

        # 1. Recherche exacte par ticker
        if query in self.tickers_db:
            results.append({
                'ticker': query,
                'name': self.tickers_db[query],
                'score': 100,
                'match_type': 'exact_ticker'
            })
            return results

        # 2. Recherche par préfixe de ticker (A → AAPL, AMZN, etc.)
        prefix_matches = []
        for ticker, name in self.tickers_db.items():
            if ticker.startswith(query):
                prefix_matches.append({
                    'ticker': ticker,
                    'name': name,
                    'score': 95,
                    'match_type': 'prefix_ticker'
                })

        if prefix_matches:
            # Trier par longueur de ticker (plus court = plus pertinent)
            prefix_matches.sort(key=lambda x: len(x['ticker']))
            results.extend(prefix_matches[:limit])

        # 3. Recherche fuzzy dans les noms
        query_lower = query.lower()
        name_matches = []

        for ticker, name in self.tickers_db.items():
            name_lower = name.lower()

            # Recherche de sous-chaîne dans le nom
            if query_lower in name_lower:
                # Score basé sur la position (plus tôt = meilleur)
                position = name_lower.index(query_lower)
                score = 90 - (position * 2)  # Pénaliser les matches tardifs
                name_matches.append({
                    'ticker': ticker,
                    'name': name,
                    'score': max(score, 70),
                    'match_type': 'substring_name'
                })
            else:
                # Fuzzy matching avec rapidfuzz
                ratio = fuzz.partial_ratio(query_lower, name_lower)
                if ratio > 60:  # Seuil de similarité
                    name_matches.append({
                        'ticker': ticker,
                        'name': name,
                        'score': ratio,
                        'match_type': 'fuzzy_name'
                    })

        # Trier par score
        name_matches.sort(key=lambda x: x['score'], reverse=True)
        results.extend(name_matches[:limit - len(results)])

        # Dédupliquer et limiter
        seen_tickers = set()
        unique_results = []
        for result in results:
            if result['ticker'] not in seen_tickers:
                seen_tickers.add(result['ticker'])
                unique_results.append(result)
                if len(unique_results) >= limit:
                    break

        logger.info(f"🔍 Recherche '{query}': {len(unique_results)} résultats trouvés")
        return unique_results

    def get_ticker_info(self, ticker: str) -> Optional[Dict[str, str]]:
        """Récupère les infos d'un ticker spécifique"""
        ticker = ticker.upper().strip()
        if ticker in self.tickers_db:
            return {
                'ticker': ticker,
                'name': self.tickers_db[ticker]
            }
        return None

    def add_ticker(self, ticker: str, name: str):
        """Ajoute un nouveau ticker à la base de données"""
        ticker = ticker.upper().strip()
        self.tickers_db[ticker] = name
        self.name_to_ticker[name.lower()] = ticker
        logger.info(f"✅ Ticker ajouté: {ticker} - {name}")


# Instance singleton
_search_engine_instance = None

def get_search_engine() -> TickerSearchEngine:
    """Retourne l'instance singleton du moteur de recherche"""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = TickerSearchEngine()
    return _search_engine_instance


# Fonctions utilitaires
def search_ticker(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """Raccourci pour rechercher un ticker"""
    engine = get_search_engine()
    return engine.search(query, limit)


def get_ticker_name(ticker: str) -> str:
    """Récupère le nom complet d'un ticker"""
    engine = get_search_engine()
    info = engine.get_ticker_info(ticker)
    return info['name'] if info else ticker
