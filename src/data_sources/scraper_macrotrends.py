import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# 🔍 Convertir un tableau HTML en DataFrame propre
def extract_table_to_dataframe(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    data = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) == 2:
            year = cols[0].text.strip()
            value = cols[1].text.strip().replace("$", "").replace(",", "")
            try:
                value = float(value)
                data.append((int(year), value))
            except:
                continue

    df = pd.DataFrame(data, columns=["Année", "Valeur (M USD)"])
    return df.sort_values("Année", ascending=False).reset_index(drop=True)

# 🌐 Scraper la page MacroTrends pour un ticker, une entreprise, et une métrique (ex: revenue)
async def scrape_macrotrends(ticker: str, company_name: str, metric: str):
    url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{company_name}/{metric}"
    print(f"[MacroTrends] Scraping {metric} for {ticker}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # 👁 mode visible pour debug
        page = await browser.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        except Exception as e:
            print(f"❌ Timeout ou erreur de chargement : {e}")
            await browser.close()
            return

        table = await page.query_selector("table")
        if table:
            html = await table.inner_html()
            df = extract_table_to_dataframe(html)
            print("✅ Données extraites :\n")
            print(df)
        else:
            print("❌ Aucun tableau trouvé sur la page.")

        await browser.close()

# 🔧 Fonction réutilisable pour récupérer la dernière valeur d'une métrique
async def get_macrotrends_data(ticker: str, company_name: str, metric: str) -> tuple[float, pd.DataFrame]:
    url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{company_name}/{metric}"
    print(f"[MacroTrends] 📈 {metric.upper()} pour {ticker}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=90000)
            table = await page.query_selector("table")
            if not table:
                raise ValueError("❌ Aucun tableau trouvé")
            html = await table.inner_html()
        finally:
            await browser.close()

    df = extract_table_to_dataframe(html)
    if df.empty:
        raise ValueError("❌ Aucune donnée extractible")

    last_value = df.iloc[0]["Valeur (M USD)"]
    return last_value, df

# 📊 Test manuel
if __name__ == "__main__":
    asyncio.run(scrape_macrotrends("AAPL", "apple", "revenue"))

__all__ = ["get_macrotrends_data", "enrich_with_macrotrends"]

async def enrich_with_macrotrends(ticker: str, company_name: str) -> dict:
    try:
        revenue, _ = await get_macrotrends_data(ticker, company_name, "revenue")
        net_income, _ = await get_macrotrends_data(ticker, company_name, "net-income")
        return {
            "revenue": revenue,
            "net_income": net_income
        }
    except Exception as e:
        print(f"[MacroTrends ❌] Erreur enrichissement {ticker}: {e}")
        return {}
