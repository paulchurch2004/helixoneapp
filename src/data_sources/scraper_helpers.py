import json
import os
import aiohttp
from bs4 import BeautifulSoup
from typing import Optional
from data_sources.schemas import BoursoramaStockData

CODES_FILE = os.path.join(os.path.dirname(__file__), "../../codes_boursorama.json")


def get_boursorama_code(ticker: str) -> Optional[str]:
    try:
        with open(CODES_FILE, "r", encoding="utf-8") as f:
            codes = json.load(f)
        return codes.get(ticker)
    except Exception as e:
        print(f"[❌] Erreur chargement code Boursorama : {e}")
        return None


async def get_boursorama_data(ticker: str, session: aiohttp.ClientSession) -> Optional[BoursoramaStockData]:
    print(f"[Boursorama] 📈 Scraping simple pour {ticker}")
    code = get_boursorama_code(ticker)
    if not code:
        print(f"[Boursorama] ❌ Aucun code trouvé pour {ticker}")
        return None

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        url = f"https://www.boursorama.com/cours/{code}/"
        async with session.get(url, headers=headers, timeout=10) as response:
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")

        nom = soup.find("h1").text.strip().replace("Cours", "").strip() if soup.find("h1") else None
        price_tag = soup.find("span", {"class": "c-instrument c-instrument--last"})
        cours = float(price_tag.text.replace(",", ".").replace("€", "").replace(" ", "").strip()) if price_tag else None
        variation_tag = soup.find("span", {"class": lambda x: x and "c-instrument--variation" in x})
        variation = variation_tag.text.strip() if variation_tag else None

        data_map = {
            "per": None,
            "dividende": None,
            "volume": None,
            "capitalisation": None,
            "objectif_cours": None,
            "recommandation": None
        }

        rows = soup.find_all("tr", class_="c-table__row")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            print(f"[Label trouvé] {label} => {value}")

            if "per" in label:
                data_map["per"] = value.replace(",", ".")
            elif "dividende" in label:
                montant = value.split(" ")[0]
                data_map["dividende"] = montant.replace(",", ".")
            elif "volume" in label:
                data_map["volume"] = value.replace(" ", "").replace("\u202f", "")
            elif "capitalisation" in label:
                data_map["capitalisation"] = value
            elif "objectif" in label:
                data_map["objectif_cours"] = value
            elif "recommandation" in label:
                data_map["recommandation"] = value

        dividende_valide = (
            data_map["dividende"]
            and data_map["dividende"].replace(".", "", 1).isdigit()
            and cours
        )

        return BoursoramaStockData(
            nom=nom,
            code=code,
            cours=cours,
            variation=variation,
            volume=int(data_map["volume"]) if data_map["volume"] and data_map["volume"].isdigit() else None,
            per=float(data_map["per"]) if data_map["per"] else None,
            dividende=float(data_map["dividende"]) if dividende_valide else None,
            rendement=(
                round(float(data_map["dividende"]) / cours * 100, 2)
                if dividende_valide
                else None
            ),
            capitalisation=data_map["capitalisation"],
            objectif_cours=data_map["objectif_cours"],
            recommandation=data_map["recommandation"],
            secteur=None,
            activite=None,
            isin=None,
            devise=None,
            raison_sociale=None,
            effectif=None,
            nombre_titres=None,
            dernier_coupon=None
        )

    except Exception as e:
        print(f"[❌] Erreur scraping Boursorama : {e}")
        return None
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

def clean_number(text: str):
    try:
        return float(text.replace(",", "").replace("$", "").strip())
    except:
        return None

def get_macrotrends_data(ticker: str, company_name: str) -> dict:
    """
    Scrape MacroTrends for the most recent net income and free cash flow values.

    Args:
        ticker (str): Stock ticker, e.g., "AAPL"
        company_name (str): Company name in URL, e.g., "apple"

    Returns:
        dict: {
            "net_income": float or None,
            "free_cash_flow": float or None
        }
    """
    base = f"https://www.macrotrends.net/stocks/charts/{ticker}/{company_name}"
    result = {}

    urls = {
        "net_income": f"{base}/net-income",
        "free_cash_flow": f"{base}/free-cash-flow"
    }

    for key, url in urls.items():
        try:
            print(f"[MacroTrends] Scraping {key} from {url}")
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            table = soup.find("table", class_="historical_data_table")
            if not table:
                print(f"[MacroTrends ❌] Tableau non trouvé pour {key}")
                continue

            tbody = table.find("tbody")
            if not tbody:
                print(f"[MacroTrends ❌] Aucun contenu dans le tableau {key}")
                continue

            first_row = tbody.find("tr")
            if not first_row:
                print(f"[MacroTrends ❌] Pas de ligne de données pour {key}")
                continue

            cols = first_row.find_all("td")
            if len(cols) >= 2:
                raw_value = cols[1].text
                result[key] = clean_number(raw_value)
            else:
                print(f"[MacroTrends ❌] Pas assez de colonnes pour {key}")

        except Exception as e:
            print(f"[MacroTrends ❌] Erreur scraping {key} : {e}")

    return result
