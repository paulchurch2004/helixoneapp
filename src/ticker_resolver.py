from rapidfuzz import process
import json
import unicodedata

def normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").lower().strip()

def load_ticker_data(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_ticker(query: str, data_path: str) -> str | None:
    query_normalized = normalize(query)
    data = load_ticker_data(data_path)
    
    # Recherche par ticker direct
    for ticker, name in data.items():
        if normalize(ticker) == query_normalized:
            return ticker

    # Recherche par nom d'entreprise
    for ticker, name in data.items():
        if query_normalized in normalize(name):
            return ticker
    
    return None


import json

from rapidfuzz import process

def autocomplete_ticker(query, data, limit=5):
    """
    Retourne une liste de suggestions (ticker + nom) proches du texte tapé.
    data : list de dicts déjà chargée depuis le JSON
    """
    query = query.lower()

    candidates = [
        (entry['symbol'], entry['name']) for entry in data
        if 'symbol' in entry and 'name' in entry
    ]

    search_strings = [f"{symbol} - {name}" for symbol, name in candidates]

    matches = process.extract(query, search_strings, limit=limit, score_cutoff=50)

    return [match[0] for match in matches]
