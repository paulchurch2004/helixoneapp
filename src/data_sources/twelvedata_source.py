import requests
import pandas as pd
import json
import os

def load_api_key():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("twelve_data_api_key")
    except Exception as e:
        print(f"[⚠️] Erreur chargement clé Twelve Data : {e}")
        return None

API_KEY = load_api_key()

def get_price(ticker: str):
    url = f"https://api.twelvedata.com/price?symbol={ticker}&apikey={API_KEY}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return float(r.json().get("price"))
    except Exception as e:
        print(f"[❌] Erreur Twelve Data prix : {e}")
        return None

def get_history(ticker: str, interval="1day", outputsize=180):
    url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval={interval}&outputsize={outputsize}&apikey={API_KEY}&format=JSON"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json().get("values", [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception as e:
        print(f"[❌] Erreur Twelve Data historique : {e}")
        return pd.DataFrame()

def get_data(ticker: str) -> dict:
    prix = get_price(ticker)
    return {"prix": prix} if prix is not None else {}
