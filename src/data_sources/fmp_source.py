import requests
import json
import os

def load_api_key():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("fmp_api_key")
    except Exception as e:
        print(f"[⚠️] Erreur chargement clé FMP : {e}")
        return None

API_KEY = load_api_key()

def get_data(ticker: str) -> dict:
    base_url = "https://financialmodelingprep.com/api/v3"
    try:
        # Quote (prix actuel)
        quote_url = f"{base_url}/quote/{ticker}?apikey={API_KEY}"
        quote_resp = requests.get(quote_url).json()
        quote = quote_resp[0] if quote_resp else {}

        # Ratios
        ratios_url = f"{base_url}/ratios-ttm/{ticker}?apikey={API_KEY}"
        ratios_resp = requests.get(ratios_url).json()
        ratios = ratios_resp[0] if ratios_resp else {}

        # Profile (secteur, beta)
        profile_url = f"{base_url}/profile/{ticker}?apikey={API_KEY}"
        profile_resp = requests.get(profile_url).json()
        profile = profile_resp[0] if profile_resp else {}

        return {
            "prix": float(quote.get("price", 0)) or None,
            "pe_ratio": float(ratios.get("peRatioTTM", 0)) or None,
            "roe": float(ratios.get("returnOnEquityTTM", 0)) or None,
            "debt": float(ratios.get("totalDebtTTM", 0)) or None,
            "revenue": float(ratios.get("revenueTTM", 0)) or None,
            "free_cash_flow": float(ratios.get("freeCashFlowTTM", 0)) or None,
            "dividende": float(ratios.get("dividendYieldTTM", 0)) or None,
            "beta": float(profile.get("beta", 0)) or None,
            "sector": profile.get("sector"),
            "shortName": profile.get("companyName")
        }

    except Exception as e:
        print(f"[❌] Erreur appel FMP : {e}")
        return {}

def get_fundamentaux_fmp(ticker: str) -> dict:
    try:
        raw = get_data(ticker)
        return {
            "croissance": f"{(raw.get('revenue') or 0) / 1_000_000_000:.2f}%",  # approximation
            "marge": "N/A",  # non disponible ici
            "dette": f"{raw['debt']:.2f}" if raw.get("debt") else "N/A",
            "PE": f"{raw['pe_ratio']:.2f}" if raw.get("pe_ratio") else "N/A",
            "ROE": f"{raw['roe']:.2f}%" if raw.get("roe") else "N/A"
        }
    except Exception as e:
        print(f"[❌] Erreur fondamentaux FMP : {e}")
        return {
            "croissance": "N/A",
            "marge": "N/A",
            "dette": "N/A",
            "PE": "N/A",
            "ROE": "N/A"
        }
