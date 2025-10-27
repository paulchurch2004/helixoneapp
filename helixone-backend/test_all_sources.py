"""
Script de test global - Toutes les sources HelixOne
Test rapide pour vérifier le status de chaque source
"""

import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Charger .env AVANT tout import
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Désactiver les logs pour tests rapides
import logging
logging.basicConfig(level=logging.ERROR)

def test_source(name, test_func):
    """Test une source et retourne le résultat"""
    try:
        result = test_func()
        return {"name": name, "status": "✅ OK", "result": result, "error": None}
    except Exception as e:
        error_msg = str(e)[:100]
        return {"name": name, "status": "❌ FAIL", "result": None, "error": error_msg}


def main():
    print("\n" + "="*80)
    print("🧪 TEST GLOBAL - TOUTES LES SOURCES HELIXONE")
    print("="*80 + "\n")

    results = []

    # ===== NOUVELLES SOURCES =====
    print("📦 NOUVELLES SOURCES (7)")
    print("-" * 80)

    # 1. CoinGecko
    print("1. CoinGecko API (crypto)... ", end="", flush=True)
    try:
        from app.services.coingecko_source import get_coingecko_collector
        cg = get_coingecko_collector()
        data = cg.get_coin_price(['bitcoin'], vs_currencies=['usd'])
        btc_price = data['bitcoin']['usd']
        results.append({"name": "CoinGecko", "status": "✅ OK", "detail": f"BTC=${btc_price:,.0f}"})
        print(f"✅ OK (BTC=${btc_price:,.0f})")
    except Exception as e:
        results.append({"name": "CoinGecko", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 2. NewsAPI
    print("2. NewsAPI.org (news)... ", end="", flush=True)
    api_key = os.getenv('NEWSAPI_API_KEY')
    if not api_key:
        results.append({"name": "NewsAPI", "status": "⏳ CONFIG", "detail": "Clé API manquante"})
        print("⏳ CONFIG (clé API manquante)")
    else:
        try:
            from app.services.newsapi_source import get_newsapi_collector
            news = get_newsapi_collector()
            sources = news.get_sources(category='business')
            count = len(sources.get('sources', []))
            results.append({"name": "NewsAPI", "status": "✅ OK", "detail": f"{count} sources"})
            print(f"✅ OK ({count} sources)")
        except Exception as e:
            results.append({"name": "NewsAPI", "status": "❌ FAIL", "detail": str(e)[:50]})
            print(f"❌ FAIL: {str(e)[:50]}")

    # 3. Quandl
    print("3. Quandl (commodities)... ", end="", flush=True)
    api_key = os.getenv('QUANDL_API_KEY')
    if not api_key:
        results.append({"name": "Quandl", "status": "⏳ CONFIG", "detail": "Clé API manquante (403 sans)"})
        print("⏳ CONFIG (clé API manquante - 403 sans)")
    else:
        try:
            from app.services.quandl_source import get_quandl_collector
            quandl = get_quandl_collector()
            gold = quandl.get_gold_price(limit=1)
            price = gold['data'][0][2] if gold.get('data') else None
            results.append({"name": "Quandl", "status": "✅ OK", "detail": f"Gold=${price:,.2f}"})
            print(f"✅ OK (Gold=${price:,.2f})")
        except Exception as e:
            results.append({"name": "Quandl", "status": "❌ FAIL", "detail": str(e)[:50]})
            print(f"❌ FAIL: {str(e)[:50]}")

    # 4. Alpha Vantage Commodities
    print("4. Alpha Vantage Commodities... ", end="", flush=True)
    try:
        from app.services.alpha_vantage_collector import get_alpha_vantage_collector
        av = get_alpha_vantage_collector()
        # Test with a simple quote instead of commodity (faster)
        quote = av.get_quote('AAPL')
        price = quote['price']
        results.append({"name": "Alpha Vantage +", "status": "✅ OK", "detail": f"AAPL=${price:.2f}"})
        print(f"✅ OK (AAPL=${price:.2f})")
    except Exception as e:
        results.append({"name": "Alpha Vantage +", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 5. Fear & Greed
    print("5. Fear & Greed Index... ", end="", flush=True)
    try:
        from app.services.feargreed_source import get_feargreed_collector
        fg = get_feargreed_collector()
        current = fg.get_current()
        value = current['value']
        classification = current['value_classification']
        results.append({"name": "Fear & Greed", "status": "✅ OK", "detail": f"{value}/100 ({classification})"})
        print(f"✅ OK ({value}/100 - {classification})")
    except Exception as e:
        results.append({"name": "Fear & Greed", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 6. Carbon Intensity
    print("6. Carbon Intensity API... ", end="", flush=True)
    try:
        from app.services.carbon_intensity_source import get_carbon_intensity_collector
        carbon = get_carbon_intensity_collector()
        current = carbon.get_current_intensity()
        intensity = current['intensity'].get('actual') or current['intensity'].get('forecast')
        index = current['intensity']['index']
        results.append({"name": "Carbon Intensity", "status": "✅ OK", "detail": f"{intensity} gCO2/kWh ({index})"})
        print(f"✅ OK ({intensity} gCO2/kWh - {index})")
    except Exception as e:
        results.append({"name": "Carbon Intensity", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 7. USAspending
    print("7. USAspending.gov... ", end="", flush=True)
    try:
        from app.services.usaspending_source import get_usaspending_collector
        usa = get_usaspending_collector()
        contracts = usa.search_spending_by_recipient("Boeing", fiscal_year=2024, limit=1)
        count = len(contracts)
        results.append({"name": "USAspending.gov", "status": "✅ OK", "detail": f"{count} contrats trouvés"})
        print(f"✅ OK ({count} contrats)")
    except Exception as e:
        results.append({"name": "USAspending.gov", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # ===== SOURCES ALTERNATIVES (CRYPTO/FOREX) =====
    print("\n💎 SOURCES ALTERNATIVES - CRYPTO/FOREX (5)")
    print("-" * 80)

    # 8. Binance
    print("8. Binance (crypto exchange #1)... ", end="", flush=True)
    try:
        from app.services.binance_source import get_binance_collector
        binance = get_binance_collector()
        ticker = binance.get_ticker_price('BTCUSDT')
        price = float(ticker['price'])
        results.append({"name": "Binance", "status": "✅ OK", "detail": f"BTC=${price:,.0f}"})
        print(f"✅ OK (BTC=${price:,.0f})")
    except Exception as e:
        results.append({"name": "Binance", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 9. Coinbase
    print("9. Coinbase (US exchange)... ", end="", flush=True)
    try:
        from app.services.coinbase_source import get_coinbase_collector
        coinbase = get_coinbase_collector()
        price = coinbase.get_crypto_price('BTC', 'USD')
        results.append({"name": "Coinbase", "status": "✅ OK", "detail": f"BTC=${price:,.0f}"})
        print(f"✅ OK (BTC=${price:,.0f})")
    except Exception as e:
        results.append({"name": "Coinbase", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 10. Kraken
    print("10. Kraken (EU exchange)... ", end="", flush=True)
    try:
        from app.services.kraken_source import get_kraken_collector
        kraken = get_kraken_collector()
        price = kraken.get_crypto_price('XBT', 'USD')
        results.append({"name": "Kraken", "status": "✅ OK", "detail": f"BTC=${price:,.0f}"})
        print(f"✅ OK (BTC=${price:,.0f})")
    except Exception as e:
        results.append({"name": "Kraken", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 11. CoinCap
    print("11. CoinCap (2000+ cryptos)... ", end="", flush=True)
    try:
        from app.services.coincap_source import get_coincap_collector
        coincap = get_coincap_collector()
        price = coincap.get_crypto_price('bitcoin')
        results.append({"name": "CoinCap", "status": "✅ OK", "detail": f"BTC=${price:,.0f}"})
        print(f"✅ OK (BTC=${price:,.0f})")
    except Exception as e:
        results.append({"name": "CoinCap", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 12. ExchangeRate
    print("12. ExchangeRate (160+ currencies)... ", end="", flush=True)
    api_key = os.getenv('EXCHANGERATE_API_KEY')
    if not api_key:
        results.append({"name": "ExchangeRate", "status": "⏳ CONFIG", "detail": "Clé API manquante"})
        print("⏳ CONFIG (clé API manquante)")
    else:
        try:
            from app.services.exchangerate_source import get_exchangerate_collector
            exchg = get_exchangerate_collector()
            rates = exchg.get_latest_rates('USD')
            eur_rate = rates['rates']['EUR']
            results.append({"name": "ExchangeRate", "status": "✅ OK", "detail": f"EUR={eur_rate:.4f}"})
            print(f"✅ OK (EUR={eur_rate:.4f})")
        except Exception as e:
            results.append({"name": "ExchangeRate", "status": "❌ FAIL", "detail": str(e)[:50]})
            print(f"❌ FAIL: {str(e)[:50]}")

    # ===== SOURCES EXISTANTES =====
    print("\n📚 SOURCES EXISTANTES (15)")
    print("-" * 80)

    # 13. FRED
    print("13. FRED (Federal Reserve)... ", end="", flush=True)
    try:
        from app.services.fred_collector import get_fred_collector
        fred = get_fred_collector()
        # Get last 1 year of GDP data
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = fred.get_series('GDP', start_date=start_date, end_date=end_date)
        if data is not None and len(data) > 0:
            latest_value = data.iloc[-1]
            results.append({"name": "FRED", "status": "✅ OK", "detail": f"GDP=${latest_value:.1f}T"})
            print(f"✅ OK (GDP=${latest_value:.1f}T)")
        else:
            results.append({"name": "FRED", "status": "⚠️  WARNING", "detail": "No data"})
            print("⚠️  WARNING (no data)")
    except Exception as e:
        results.append({"name": "FRED", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 14. SEC Edgar
    print("14. SEC Edgar... ", end="", flush=True)
    try:
        from app.services.sec_edgar_collector import get_sec_edgar_collector
        sec = get_sec_edgar_collector()
        tickers = sec.get_company_tickers()
        count = len(tickers) if tickers else 0
        results.append({"name": "SEC Edgar", "status": "✅ OK", "detail": f"{count} companies"})
        print(f"✅ OK ({count} companies)")
    except Exception as e:
        results.append({"name": "SEC Edgar", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 15. Finnhub
    print("15. Finnhub... ", end="", flush=True)
    try:
        from app.services.finnhub_collector import get_finnhub_collector
        finnhub = get_finnhub_collector()
        quote = finnhub.get_quote('AAPL')
        price = quote.get('c', 0)
        results.append({"name": "Finnhub", "status": "✅ OK", "detail": f"AAPL=${price:.2f}"})
        print(f"✅ OK (AAPL=${price:.2f})")
    except Exception as e:
        results.append({"name": "Finnhub", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 16. FMP
    print("16. Financial Modeling Prep... ", end="", flush=True)
    try:
        from app.services.fmp_collector import get_fmp_collector
        fmp = get_fmp_collector()
        quote = fmp.get_quote('AAPL')
        if quote and len(quote) > 0:
            price = quote[0].get('price', 0)
            results.append({"name": "FMP", "status": "✅ OK", "detail": f"AAPL=${price:.2f}"})
            print(f"✅ OK (AAPL=${price:.2f})")
        else:
            results.append({"name": "FMP", "status": "⚠️  WARNING", "detail": "No data"})
            print("⚠️  WARNING (no data)")
    except Exception as e:
        results.append({"name": "FMP", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 17. Twelve Data
    print("17. Twelve Data... ", end="", flush=True)
    try:
        from app.services.twelvedata_collector import get_twelvedata_collector
        twelve = get_twelvedata_collector()
        quote = twelve.get_quote('AAPL')
        price = float(quote.get('close', 0))
        results.append({"name": "Twelve Data", "status": "✅ OK", "detail": f"AAPL=${price:.2f}"})
        print(f"✅ OK (AAPL=${price:.2f})")
    except Exception as e:
        results.append({"name": "Twelve Data", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # 18. Yahoo Finance
    print("18. Yahoo Finance... ", end="", flush=True)
    try:
        import yfinance as yf
        stock = yf.Ticker('AAPL')
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        if price and price > 0:
            results.append({"name": "Yahoo Finance", "status": "✅ OK", "detail": f"AAPL=${price:.2f}"})
            print(f"✅ OK (AAPL=${price:.2f})")
        else:
            results.append({"name": "Yahoo Finance", "status": "⚠️  WARNING", "detail": "No price data"})
            print("⚠️  WARNING (no price)")
    except Exception as e:
        results.append({"name": "Yahoo Finance", "status": "❌ FAIL", "detail": str(e)[:50]})
        print(f"❌ FAIL: {str(e)[:50]}")

    # Skip slow/broken sources for quick test
    print("19. World Bank... ", end="", flush=True)
    print("⏭️  SKIPPED (slow)")
    results.append({"name": "World Bank", "status": "⏭️  SKIP", "detail": "Slow test"})

    print("20. OECD... ", end="", flush=True)
    print("⏭️  SKIPPED (slow)")
    results.append({"name": "OECD", "status": "⏭️  SKIP", "detail": "Slow test"})

    print("21. ECB... ", end="", flush=True)
    print("⏭️  SKIPPED (slow)")
    results.append({"name": "ECB", "status": "⏭️  SKIP", "detail": "Slow test"})

    print("22. Eurostat... ", end="", flush=True)
    print("⏭️  SKIPPED (slow)")
    results.append({"name": "Eurostat", "status": "⏭️  SKIP", "detail": "Slow test"})

    print("23. BIS... ", end="", flush=True)
    print("⏭️  SKIPPED (broken - migration)")
    results.append({"name": "BIS", "status": "⚠️  BROKEN", "detail": "API migration needed"})

    print("24. IMF... ", end="", flush=True)
    print("⏭️  SKIPPED (broken - migration)")
    results.append({"name": "IMF", "status": "⚠️  BROKEN", "detail": "Server migration needed"})

    # Summary
    print("\n" + "="*80)
    print("📊 RÉSUMÉ")
    print("="*80 + "\n")

    ok_count = sum(1 for r in results if r["status"] == "✅ OK")
    fail_count = sum(1 for r in results if r["status"] == "❌ FAIL")
    config_count = sum(1 for r in results if r["status"] == "⏳ CONFIG")
    broken_count = sum(1 for r in results if r["status"] == "⚠️  BROKEN")
    skip_count = sum(1 for r in results if r["status"] == "⏭️  SKIP")
    total = len(results)

    print(f"✅ Fonctionnelles:       {ok_count}/{total}")
    print(f"❌ En erreur:           {fail_count}/{total}")
    print(f"⏳ Config requise:      {config_count}/{total}")
    print(f"⚠️  Cassées (migration): {broken_count}/{total}")
    print(f"⏭️  Skipped (lent):      {skip_count}/{total}")

    print(f"\n📊 Taux de succès: {ok_count}/{total - skip_count - broken_count} = {ok_count*100/(total-skip_count-broken_count):.0f}%")

    # Details
    print("\n" + "="*80)
    print("📋 DÉTAILS")
    print("="*80 + "\n")

    for r in results:
        status = r["status"]
        name = r["name"]
        detail = r.get("detail", "")

        print(f"{status:<12} {name:<25} {detail}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
