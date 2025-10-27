import requests
from bs4 import BeautifulSoup

def get_macro_data():
    print("[SCRAPER] 🌍 Récupération des données macroéconomiques...")

    try:
        fed_rate = scrape_fed_rate()
        inflation = scrape_inflation()
        gdp = scrape_gdp()

        macro = {
            "fed_rate": fed_rate,
            "inflation_us": inflation,
            "gdp_us": gdp
        }

        print(f"[🌍] Données macro récupérées via scraping : {macro}")
        if all(v is None for v in macro.values()):
            return None
        return macro

    except Exception as e:
        print(f"[SCRAPER] ❌ Erreur macro_scraper : {e}")
        return None

def scrape_fed_rate():
    try:
        url = "https://www.cnbc.com/quotes/.FEDRATE"
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        value = soup.find("span", class_="QuoteStrip-lastPrice").text
        return float(value.strip("%"))
    except Exception:
        print("[FED] ⚠️ Élément introuvable.")
        return None

def scrape_inflation():
    try:
        url = "https://tradingeconomics.com/united-states/inflation-cpi"
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        div = soup.find("div", class_="table-responsive")
        value = div.find("td").text.strip().replace("%", "")
        return float(value)
    except Exception:
        print("[Inflation] ⚠️ Élément introuvable.")
        return None

def scrape_gdp():
    try:
        url = "https://tradingeconomics.com/united-states/gdp-growth"
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        div = soup.find("div", class_="table-responsive")
        value = div.find("td").text.strip().replace("%", "")
        return float(value)
    except Exception:
        print("[GDP] ⚠️ Élément introuvable.")
        return None
