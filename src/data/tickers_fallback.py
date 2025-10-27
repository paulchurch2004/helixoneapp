"""
Base de données de tickers étendue pour HelixOne
Remplace le fichier JSON manquant par une base complète
"""

TICKERS_DATABASE = {
    # === INDICES MAJEURS ===
    "^FCHI": {"name": "CAC 40", "exchange": "PA", "type": "index"},
    "^GSPC": {"name": "S&P 500", "exchange": "US", "type": "index"},
    "^IXIC": {"name": "NASDAQ Composite", "exchange": "US", "type": "index"},
    "^DJI": {"name": "Dow Jones", "exchange": "US", "type": "index"},
    "^RUT": {"name": "Russell 2000", "exchange": "US", "type": "index"},
    
    # === CAC 40 ===
    "MC.PA": {"name": "LVMH", "exchange": "PA", "sector": "Consumer Discretionary"},
    "OR.PA": {"name": "L'Oréal", "exchange": "PA", "sector": "Consumer Staples"},
    "SAN.PA": {"name": "Sanofi", "exchange": "PA", "sector": "Healthcare"},
    "TTE.PA": {"name": "TotalEnergies", "exchange": "PA", "sector": "Energy"},
    "ASML.AS": {"name": "ASML", "exchange": "AS", "sector": "Technology"},
    "AI.PA": {"name": "Air Liquide", "exchange": "PA", "sector": "Materials"},
    "BNP.PA": {"name": "BNP Paribas", "exchange": "PA", "sector": "Financial Services"},
    "SAF.PA": {"name": "Safran", "exchange": "PA", "sector": "Industrials"},
    "RMS.PA": {"name": "Hermès", "exchange": "PA", "sector": "Consumer Discretionary"},
    "KER.PA": {"name": "Kering", "exchange": "PA", "sector": "Consumer Discretionary"},
    "EL.PA": {"name": "EssilorLuxottica", "exchange": "PA", "sector": "Healthcare"},
    "STLAM.MI": {"name": "Stellantis", "exchange": "MI", "sector": "Consumer Discretionary"},
    "CAP.PA": {"name": "Capgemini", "exchange": "PA", "sector": "Technology"},
    "DG.PA": {"name": "Vinci", "exchange": "PA", "sector": "Industrials"},
    "SU.PA": {"name": "Schneider Electric", "exchange": "PA", "sector": "Industrials"},
    "BN.PA": {"name": "Danone", "exchange": "PA", "sector": "Consumer Staples"},
    "AXA.PA": {"name": "AXA", "exchange": "PA", "sector": "Financial Services"},
    "GLE.PA": {"name": "Société Générale", "exchange": "PA", "sector": "Financial Services"},
    "VIE.PA": {"name": "Veolia", "exchange": "PA", "sector": "Utilities"},
    "ENGI.PA": {"name": "Engie", "exchange": "PA", "sector": "Utilities"},
    
    # === NASDAQ 100 / TECH US ===
    "AAPL": {"name": "Apple Inc.", "exchange": "NASDAQ", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corporation", "exchange": "NASDAQ", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc. Class A", "exchange": "NASDAQ", "sector": "Communication Services"},
    "GOOG": {"name": "Alphabet Inc. Class C", "exchange": "NASDAQ", "sector": "Communication Services"},
    "AMZN": {"name": "Amazon.com Inc.", "exchange": "NASDAQ", "sector": "Consumer Discretionary"},
    "NVDA": {"name": "NVIDIA Corporation", "exchange": "NASDAQ", "sector": "Technology"},
    "TSLA": {"name": "Tesla Inc.", "exchange": "NASDAQ", "sector": "Consumer Discretionary"},
    "META": {"name": "Meta Platforms Inc.", "exchange": "NASDAQ", "sector": "Communication Services"},
    "AVGO": {"name": "Broadcom Inc.", "exchange": "NASDAQ", "sector": "Technology"},
    "COST": {"name": "Costco Wholesale Corporation", "exchange": "NASDAQ", "sector": "Consumer Staples"},
    "NFLX": {"name": "Netflix Inc.", "exchange": "NASDAQ", "sector": "Communication Services"},
    "AMD": {"name": "Advanced Micro Devices", "exchange": "NASDAQ", "sector": "Technology"},
    "PYPL": {"name": "PayPal Holdings Inc.", "exchange": "NASDAQ", "sector": "Financial Services"},
    "INTC": {"name": "Intel Corporation", "exchange": "NASDAQ", "sector": "Technology"},
    "QCOM": {"name": "QUALCOMM Incorporated", "exchange": "NASDAQ", "sector": "Technology"},
    "ADBE": {"name": "Adobe Inc.", "exchange": "NASDAQ", "sector": "Technology"},
    "TXN": {"name": "Texas Instruments", "exchange": "NASDAQ", "sector": "Technology"},
    "INTU": {"name": "Intuit Inc.", "exchange": "NASDAQ", "sector": "Technology"},
    "ISRG": {"name": "Intuitive Surgical", "exchange": "NASDAQ", "sector": "Healthcare"},
    "CMCSA": {"name": "Comcast Corporation", "exchange": "NASDAQ", "sector": "Communication Services"},
    
    # === S&P 500 LEADERS ===
    "JNJ": {"name": "Johnson & Johnson", "exchange": "NYSE", "sector": "Healthcare"},
    "UNH": {"name": "UnitedHealth Group", "exchange": "NYSE", "sector": "Healthcare"},
    "PG": {"name": "Procter & Gamble", "exchange": "NYSE", "sector": "Consumer Staples"},
    "HD": {"name": "Home Depot", "exchange": "NYSE", "sector": "Consumer Discretionary"},
    "MA": {"name": "Mastercard", "exchange": "NYSE", "sector": "Financial Services"},
    "V": {"name": "Visa Inc.", "exchange": "NYSE", "sector": "Financial Services"},
    "BAC": {"name": "Bank of America", "exchange": "NYSE", "sector": "Financial Services"},
    "JPM": {"name": "JPMorgan Chase", "exchange": "NYSE", "sector": "Financial Services"},
    "WMT": {"name": "Walmart Inc.", "exchange": "NYSE", "sector": "Consumer Staples"},
    "DIS": {"name": "Walt Disney", "exchange": "NYSE", "sector": "Communication Services"},
    "ADBE": {"name": "Adobe Inc.", "exchange": "NASDAQ", "sector": "Technology"},
    "CRM": {"name": "Salesforce", "exchange": "NYSE", "sector": "Technology"},
    "NFLX": {"name": "Netflix", "exchange": "NASDAQ", "sector": "Communication Services"},
    "KO": {"name": "Coca-Cola", "exchange": "NYSE", "sector": "Consumer Staples"},
    "PFE": {"name": "Pfizer Inc.", "exchange": "NYSE", "sector": "Healthcare"},
    "MRK": {"name": "Merck & Co", "exchange": "NYSE", "sector": "Healthcare"},
    "ABBV": {"name": "AbbVie Inc.", "exchange": "NYSE", "sector": "Healthcare"},
    "TMO": {"name": "Thermo Fisher Scientific", "exchange": "NYSE", "sector": "Healthcare"},
    "ACN": {"name": "Accenture", "exchange": "NYSE", "sector": "Technology"},
    "ORCL": {"name": "Oracle Corporation", "exchange": "NYSE", "sector": "Technology"},
    
    # === CRYPTO / NEW ECONOMY ===
    "COIN": {"name": "Coinbase Global", "exchange": "NASDAQ", "sector": "Financial Services"},
    "HOOD": {"name": "Robinhood Markets", "exchange": "NASDAQ", "sector": "Financial Services"},
    "SQ": {"name": "Block Inc.", "exchange": "NYSE", "sector": "Financial Services"},
    "ROKU": {"name": "Roku Inc.", "exchange": "NASDAQ", "sector": "Communication Services"},
    "UBER": {"name": "Uber Technologies", "exchange": "NYSE", "sector": "Technology"},
    "LYFT": {"name": "Lyft Inc.", "exchange": "NASDAQ", "sector": "Technology"},
    "ABNB": {"name": "Airbnb Inc.", "exchange": "NASDAQ", "sector": "Consumer Discretionary"},
    "DASH": {"name": "DoorDash Inc.", "exchange": "NYSE", "sector": "Consumer Discretionary"},
    "SNOW": {"name": "Snowflake Inc.", "exchange": "NYSE", "sector": "Technology"},
    "PLTR": {"name": "Palantir Technologies", "exchange": "NYSE", "sector": "Technology"},
    
    # === INTERNATIONAL ===
    "NVO": {"name": "Novo Nordisk", "exchange": "NYSE", "sector": "Healthcare"},
    "ASML": {"name": "ASML Holding", "exchange": "NASDAQ", "sector": "Technology"},
    "UL": {"name": "Unilever", "exchange": "NYSE", "sector": "Consumer Staples"},
    "NVS": {"name": "Novartis AG", "exchange": "NYSE", "sector": "Healthcare"},
    "SAP": {"name": "SAP SE", "exchange": "NYSE", "sector": "Technology"},
    "SHEL": {"name": "Shell plc", "exchange": "NYSE", "sector": "Energy"},
    "BP": {"name": "BP p.l.c.", "exchange": "NYSE", "sector": "Energy"},
    "GSK": {"name": "GSK plc", "exchange": "NYSE", "sector": "Healthcare"},
    
    # === ETFs POPULAIRES ===
    "SPY": {"name": "SPDR S&P 500 ETF", "exchange": "NYSE", "type": "ETF"},
    "QQQ": {"name": "Invesco QQQ Trust", "exchange": "NASDAQ", "type": "ETF"},
    "IWM": {"name": "iShares Russell 2000 ETF", "exchange": "NYSE", "type": "ETF"},
    "VTI": {"name": "Vanguard Total Stock Market ETF", "exchange": "NYSE", "type": "ETF"},
    "VOO": {"name": "Vanguard S&P 500 ETF", "exchange": "NYSE", "type": "ETF"},
    "ARKK": {"name": "ARK Innovation ETF", "exchange": "NYSE", "type": "ETF"},
    
    # === COMMODITIES ===
    "GLD": {"name": "SPDR Gold Shares", "exchange": "NYSE", "type": "ETF"},
    "SLV": {"name": "iShares Silver Trust", "exchange": "NYSE", "type": "ETF"},
    "USO": {"name": "United States Oil Fund", "exchange": "NYSE", "type": "ETF"},
    "UNG": {"name": "United States Natural Gas Fund", "exchange": "NYSE", "type": "ETF"},
}

def create_search_index():
    """Crée un index de recherche optimisé"""
    search_index = {}
    
    for symbol, data in TICKERS_DATABASE.items():
        name = data["name"].lower()
        
        # Index par symbole
        search_index[symbol.lower()] = symbol
        search_index[symbol.replace(".", "").lower()] = symbol
        
        # Index par nom complet
        search_index[name] = symbol
        
        # Index par mots du nom
        words = name.replace(".", " ").replace(",", " ").split()
        for word in words:
            if len(word) > 2:  # Ignorer les mots trop courts
                if word not in search_index:
                    search_index[word] = symbol
        
        # Index par nom sans extensions (Inc., Corp., etc.)
        clean_name = name
        for suffix in [" inc", " corp", " corporation", " company", " ltd", " plc"]:
            clean_name = clean_name.replace(suffix, "")
        
        if clean_name != name:
            search_index[clean_name] = symbol
    
    return search_index

# Index de recherche global
SEARCH_INDEX = create_search_index()

def find_ticker(query: str) -> str:
    """Trouve un ticker par recherche exacte ou floue"""
    if not query:
        return None
    
    query = query.strip().lower()
    
    # Recherche exacte
    if query in SEARCH_INDEX:
        return SEARCH_INDEX[query]
    
    # Recherche floue avec rapidfuzz si disponible
    try:
        from rapidfuzz import fuzz, process
        
        # Recherche dans les noms
        names = [(name, symbol) for name, symbol in SEARCH_INDEX.items() if len(name) > 2]
        result = process.extractOne(
            query, 
            [name for name, _ in names], 
            scorer=fuzz.token_sort_ratio,
            score_cutoff=70
        )
        
        if result:
            match, score, _ = result
            # Trouver le symbole correspondant
            for name, symbol in names:
                if name == match:
                    return symbol
                    
    except ImportError:
        # Fallback : recherche partielle
        for name, symbol in SEARCH_INDEX.items():
            if query in name or name in query:
                return symbol
    
    return None

def get_ticker_suggestions(query: str, limit: int = 5) -> list:
    """Retourne des suggestions de tickers"""
    if not query:
        return []
    
    query = query.strip().lower()
    suggestions = []
    
    # Recherche dans l'index
    for name, symbol in list(SEARCH_INDEX.items())[:500]:  # Limiter pour performance
        if query in name:
            if symbol not in suggestions:
                suggestions.append(symbol)
            if len(suggestions) >= limit:
                break
    
    return suggestions

def get_ticker_info(symbol: str) -> dict:
    """Retourne les informations d'un ticker"""
    return TICKERS_DATABASE.get(symbol, {})

def get_all_symbols() -> list:
    """Retourne tous les symboles disponibles"""
    return list(TICKERS_DATABASE.keys())

def get_symbols_by_sector(sector: str) -> list:
    """Retourne les symboles d'un secteur"""
    return [symbol for symbol, data in TICKERS_DATABASE.items() 
            if data.get("sector") == sector]

def get_symbols_by_exchange(exchange: str) -> list:
    """Retourne les symboles d'une bourse"""
    return [symbol for symbol, data in TICKERS_DATABASE.items() 
            if data.get("exchange") == exchange]

# Interface de compatibilité avec l'ancien système
def charger_tickers():
    """Interface de compatibilité"""
    return SEARCH_INDEX

# Export pour utilisation dans data_manager
actions_db = SEARCH_INDEX

if __name__ == "__main__":
    # Tests
    print(f"Base de données: {len(TICKERS_DATABASE)} tickers")
    print(f"Index de recherche: {len(SEARCH_INDEX)} entrées")
    
    # Test de recherche
    test_queries = ["apple", "microsoft", "lvmh", "tesla", "aapl"]
    for query in test_queries:
        result = find_ticker(query)
        print(f"'{query}' -> {result}")
    
    # Test suggestions
    suggestions = get_ticker_suggestions("app")
    print(f"Suggestions pour 'app': {suggestions}")