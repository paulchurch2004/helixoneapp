import json
import os
import re
import aiohttp
from bs4 import BeautifulSoup

CODES_FILE = os.path.join(os.path.dirname(__file__), "../../codes_boursorama.json")

# ✅ Fonction de récupération du code Boursorama depuis fichier local
def get_boursorama_code(ticker):
    try:
        with open(CODES_FILE, "r", encoding="utf-8") as f:
            codes = json.load(f)
        return codes.get(ticker)
    except:
        return None

# ✅ Scraper principal AVEC session aiohttp passée en argument
async def get_boursorama_data(ticker, session):
    print(f"[Boursorama] 📈 Scraping données pour {ticker}")
    code = get_boursorama_code(ticker)
    if not code:
        print(f"[Boursorama] ❌ Aucun code trouvé pour {ticker}")
        return {}

    url = f"https://www.boursorama.com/cours/{code}/"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")

        title = soup.find("h1")
        nom = title.text.strip().replace("Cours", "").strip() if title else None

        price_tag = soup.find("span", {"class": "c-instrument c-instrument--last"})
        try:
            prix = float(price_tag.text.replace(",", ".").replace("€", "").replace(" ", "").strip()) if price_tag else None
        except:
            prix = None

        variation_tag = soup.find("span", class_="c-instrument--variation")
        variation = variation_tag.text.strip() if variation_tag else None

        per = dividende = None
        rows = soup.find_all("tr", class_="c-table__row")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            label = cells[0].get_text(separator=" ", strip=True).lower()
            if "per" in label:
                per = cells[2].get_text(strip=True)
            elif "dividende" in label:
                dividende = cells[2].get_text(strip=True)

        capitalisation = None
        try:
            cap_block = soup.find_all("div", class_="c-list-info__item")
            for block in cap_block:
                label = block.find("span", class_="c-list-info__label")
                value = block.find("span", class_="c-list-info__value")
                if label and value and "capitalisation" in label.text.lower():
                    capitalisation = value.text.strip()
                    break
        except:
            pass

        isin = secteur = None
        details_block = soup.find("ul", class_="c-list-info")
        if details_block:
            for item in details_block.find_all("li"):
                label = item.find("span", class_="c-list-info__label")
                value = item.find("span", class_="c-list-info__value")
                if label and value:
                    text = label.text.lower()
                    if "isin" in text:
                        isin = value.text.strip()
                    elif "secteur" in text:
                        secteur = value.text.strip()

        return {
            "nom": nom,
            "prix": prix,
            "variation": variation,
            "per": per,
            "dividende": dividende,
            "capitalisation": capitalisation,
            "isin": isin,
            "secteur": secteur
        }

    except Exception as e:
        print(f"[Boursorama] ❌ Erreur scraping : {e}")
        return {}

# 🔧 Placeholder générique pour les sources non encore faites
def make_placeholder_scraper(name):
    async def placeholder(ticker, session=None):
        print(f"[{name}] 📱 (placeholder) scraping pour {ticker}")
        return {}
    return placeholder

# Placeholders pour toutes les autres sources
get_morningstar_data = make_placeholder_scraper('Morningstar')
get_zacks_data = make_placeholder_scraper('Zacks')
get_macrotrendsnet_data = make_placeholder_scraper('Macrotrendsnet')
get_seeking_alpha_data = make_placeholder_scraper('SeekingAlpha')
get_investingcom_data = make_placeholder_scraper('Investingcom')
get_reuters_data = make_placeholder_scraper('Reuters')
get_benzinga_data = make_placeholder_scraper('Benzinga')
get_gurufocus_data = make_placeholder_scraper('GuruFocus')
get_motley_fool_data = make_placeholder_scraper('MotleyFool')
get_bloomberg_data = make_placeholder_scraper('Bloomberg')
get_cnbc_data = make_placeholder_scraper('CNBC')
get_marketscreener_data = make_placeholder_scraper('MarketScreener')
get_finbox_data = make_placeholder_scraper('Finbox')
get_simplywallst_data = make_placeholder_scraper('SimplyWallst')
get_stockanalysiscom_data = make_placeholder_scraper('StockAnalysis.com')
get_wsj_data = make_placeholder_scraper('WSJ')
get_thestreet_data = make_placeholder_scraper('TheStreet')
get_cnn_business_data = make_placeholder_scraper('CNNBusiness')
get_finpedia_data = make_placeholder_scraper('Finpedia')
get_msn_money_data = make_placeholder_scraper('MSNMoney')
get_companiesmarketcap_data = make_placeholder_scraper('CompaniesMarketCap')
get_alt_yahoo_html_data = make_placeholder_scraper('AltYahooHTML')
