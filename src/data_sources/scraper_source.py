import asyncio
from data_sources.scraper_finviz import get_finviz_data
from data_sources.scraper_macrotrends import enrich_with_macrotrends

# ❌ Ne pas appeler get_boursorama_data ici, il est géré dans data_provider

# 🔍 Scraping multi-sources pour enrichissement partiel
async def get_data(ticker: str) -> dict:
    print(f"[SCRAPER-UNIV] 🔍 Scraping multi-sources pour {ticker}")
    result = {}

    # Scraper Finviz
    try:
        finviz = get_finviz_data(ticker)
        if finviz:
            result.update({
                "cours": finviz.cours,
                "per": finviz.price_to_earnings,
                "capitalisation": finviz.capitalisation,
                "beta": finviz.beta,
                "secteur": finviz.secteur,
            })
    except Exception as e:
        print(f"[Finviz ❌] Erreur pour {ticker}: {e}")

    # Scraper MacroTrends (revenue, net_income...)
    try:
        # Remplacer "apple" par un mapping plus tard
        data = await enrich_with_macrotrends(ticker, "apple")
        result.update(data)
    except Exception as e:
        print(f"[MacroTrends ❌] Erreur enrichissement {ticker}: {e}")

    return result
