
import aiohttp
import pandas as pd
import os
import json

def load_api_key():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("eod_api_key")
    except Exception as e:
        print(f"[⚠️] Erreur chargement clé EOD : {e}")
        return None

API_KEY = load_api_key()

async def get_history(ticker: str, days=180):
    try:
        url = f"https://eodhistoricaldata.com/api/eod/{ticker}?from=2022-01-01&api_token={API_KEY}&period=d&fmt=json"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                if r.status != 200:
                    print(f"[❌] Erreur HTTP {r.status} sur {url}")
                    return pd.DataFrame(), {}

                try:
                    data = await r.json()
                except Exception as e:
                    text = await r.text()
                    print(f"[❌] Erreur parsing JSON : {e}")
                    print(f"[❓] Contenu retourné :\n{text[:300]}\n--- FIN CONTENU ---")
                    return pd.DataFrame(), {}

        if not data or not isinstance(data, list):
            print("[❌] Données historiques vides ou mal formatées")
            return pd.DataFrame(), {}

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)

        fondamentaux = {
            "PER": 15.2,
            "rendement": 2.1
        }

        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float), fondamentaux

    except Exception as e:
        print(f"[❌] Erreur historique EOD : {e}")
        return pd.DataFrame(), {}
