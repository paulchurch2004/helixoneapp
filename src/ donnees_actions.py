import json
import os
from unidecode import unidecode

actions_db = {}

def normalize(text):
    """ Nettoyage pour recherche flexible """
    return unidecode(text.lower().strip().replace(".", "").replace(",", "").replace("&", "and"))

def charger_tickers():
    chemin = os.path.join("src", "data_sources", "reference-data", "tickers_full.json")
    with open(chemin, "r", encoding="utf-8") as f:
        tickers = json.load(f)

    for item in tickers:
        symbol = item.get("symbol", "").upper()
        name = item.get("name", "").strip()

        if not symbol or not name:
            continue

        keys = set()

        keys.add(normalize(symbol))  # "aapl"
        keys.add(normalize(name))    # "apple inc"
        if " inc" in name.lower():
            keys.add(normalize(name.replace("Inc.", "").replace("inc", "")))  # "apple"

        for key in keys:
            if key and key not in actions_db:
                actions_db[key] = symbol

    return actions_db

actions_db = charger_tickers()
