# data_sources/finnhub_source.py

import requests
import json
import os

def load_api_key():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("finnhub_api_key")
    except Exception as e:
        print(f"[⚠️] Impossible de charger la clé API Finnhub : {e}")
        return None

API_KEY = load_api_key()

def get_data(ticker: str) -> dict:
    if not API_KEY:
        raise ValueError("Clé API Finnhub non trouvée.")

    url_profile = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={API_KEY}"
    url_ratios = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={API_KEY}"

    try:
        profile = requests.get(url_profile).json()
        ratios = requests.get(url_ratios).json().get("metric", {})

        prix = float(ratios.get("52WeekHigh")) if "52WeekHigh" in ratios else None

        return {
            "prix": prix,
            "pe_ratio": float(ratios.get("peTTM")) if "peTTM" in ratios else None,
            "roe": float(ratios.get("roeTTM")) if "roeTTM" in ratios else None,
            "debt": float(ratios.get("totalDebt")) if "totalDebt" in ratios else None,
            "revenue": float(ratios.get("revenueTTM")) if "revenueTTM" in ratios else None,
            "free_cash_flow": float(ratios.get("freeCashFlowTTM")) if "freeCashFlowTTM" in ratios else None,
            "dividende": float(ratios.get("dividendYieldIndicatedAnnual")) if "dividendYieldIndicatedAnnual" in ratios else None,
            "beta": float(ratios.get("beta")) if "beta" in ratios else None,
            "sector": profile.get("finnhubIndustry"),
            "shortName": profile.get("name")
        }

    except Exception as e:
        print(f"[❌] Erreur lors de l’appel Finnhub : {e}")
        return {}

def get_fundamentaux_finnhub(ticker: str) -> dict:
    try:
        raw = get_data(ticker)
        return {
            "croissance": f"{(raw['revenue'] or 0) / 1_000_000_000:.2f}%",
            "marge": "N/A",
            "dette": f"{raw['debt']:.2f}" if raw.get("debt") else "N/A",
            "PE": f"{raw['pe_ratio']:.2f}" if raw.get("pe_ratio") else "N/A",
            "ROE": f"{raw['roe']:.2f}%" if raw.get("roe") else "N/A"
        }
    except Exception as e:
        print(f"[❌] Erreur format fondamentaux Finnhub : {e}")
        return {
            "croissance": "N/A", "marge": "N/A", "dette": "N/A", "PE": "N/A", "ROE": "N/A"
        }

def get_esg_score_finnhub(ticker: str) -> dict:
    try:
        if not API_KEY:
            raise ValueError("API key absente pour Finnhub")

        url = f"https://finnhub.io/api/v1/esg/scores?symbol={ticker}&token={API_KEY}"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")

        data = resp.json()

        return {
            "total": data.get("esgScore", "N/A"),
            "environment": data.get("environmentScore", "N/A"),
            "social": data.get("socialScore", "N/A"),
            "governance": data.get("governanceScore", "N/A"),
            "grade": data.get("grade", "N/A")
        }

    except Exception as e:
        print(f"[❌] Erreur ESG Finnhub : {e}")
        return {
            "total": "N/A",
            "environment": "N/A",
            "social": "N/A",
            "governance": "N/A",
            "grade": "N/A"
        }
