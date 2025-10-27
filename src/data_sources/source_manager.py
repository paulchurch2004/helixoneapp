# source_manager.py

from data_sources.scraper_source import get_data as scraper_get_data
from data_sources.scraper_finviz import get_finviz_data
from data_sources.scraper_helpers import get_boursorama_data

from data_sources.finnhub_source import get_fundamentaux_finnhub
import yfinance as yf

# 🔍 SCRAPER SELECTOR
def get_best_data(query: str) -> dict:
    sources = [
        ("Boursorama", get_boursorama_data),
        ("Finviz", get_finviz_data),
        ("Scraper générique", scraper_get_data),
    ]

    for name, source_func in sources:
        print(f"[🔍] Tentative {name} pour {query}")
        try:
            data = source_func(query)
            if data and data.get("prix"):
                print(f"[✅] {name} a fourni les données.")
                return data
            else:
                print(f"[⚠️] {name} a retourné un prix vide.")
        except Exception as e:
            print(f"[❌] Erreur {name} : {e}")

    print(f"[⛔] Aucune source n’a pu fournir les données pour {query}")
    return {}

# 📊 API FUSION ENGINE

def get_fondamentaux_yf(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "croissance": f"{info.get('revenueGrowth', 0) * 100:.2f}%",
            "marge": f"{info.get('profitMargins', 0) * 100:.2f}%",
            "dette": f"{info.get('debtToEquity', 0):.2f}",
            "PE": f"{info.get('trailingPE', 'N/A')}",
            "ROE": f"{info.get('returnOnEquity', 0) * 100:.2f}%"
        }
    except:
        return {
            "croissance": "N/A", "marge": "N/A", "dette": "N/A", "PE": "N/A", "ROE": "N/A"
        }

def choisir_valeur(*sources):
    for val in sources:
        if val and val != "N/A":
            return val
    return "N/A"

def get_fondamentaux_unifies(ticker: str) -> dict:
    yf_data = get_fondamentaux_yf(ticker)
    finnhub_data = get_fundamentaux_finnhub(ticker)

    return {
        "croissance": choisir_valeur(yf_data["croissance"], finnhub_data["croissance"]),
        "marge": choisir_valeur(yf_data["marge"], finnhub_data["marge"]),
        "dette": choisir_valeur(yf_data["dette"], finnhub_data["dette"]),
        "PE": choisir_valeur(yf_data["PE"], finnhub_data["PE"]),
        "ROE": choisir_valeur(yf_data["ROE"], finnhub_data["ROE"]),
    }
