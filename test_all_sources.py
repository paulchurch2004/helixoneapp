#!/usr/bin/env python3
"""
Script de test complet pour toutes les sources de données HelixOne
Teste chaque source individuellement puis l'aggregator
"""

import sys
import os
import asyncio
from datetime import date, timedelta

# Configuration du chemin
sys.path.insert(0, "/Users/macintosh/Desktop/helixone/helixone-backend")
os.chdir("/Users/macintosh/Desktop/helixone/helixone-backend")

from app.services.data_sources.yahoo_finance import YahooFinanceSource
from app.services.data_sources.finnhub_source import FinnhubSource
from app.services.data_sources.fmp_source import FMPSource
from app.services.data_sources.alphavantage_source import AlphaVantageSource
from app.services.data_sources.twelvedata_source import TwelveDataSource
from app.services.data_sources.fred_source import FREDSource
from app.services.data_sources.aggregator import DataAggregator


async def test_source(source, ticker="AAPL"):
    """Teste une source individuelle"""

    print(f"\n{'='*70}")
    print(f"🧪 TEST: {source.name}")
    print(f"{'='*70}")

    if not source.is_available():
        print(f"⚠️  {source.name} n'est pas disponible (clé API manquante)")
        return {
            'available': False,
            'quote': False,
            'historical': False,
            'fundamentals': False,
            'esg': False,
            'news': False
        }

    print(f"✅ {source.name} est disponible\n")

    results = {'available': True}

    # Test 1: Quote
    print(f"📊 Test Quote ({ticker})...")
    try:
        quote = await source.get_quote(ticker)
        if quote:
            print(f"   ✅ Prix: ${quote.price}")
            print(f"   Variation: {quote.change} ({quote.change_percent}%)")
            results['quote'] = True
        else:
            print(f"   ⚠️  Aucun résultat")
            results['quote'] = False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        results['quote'] = False

    # Test 2: Historical
    print(f"\n📈 Test Données historiques...")
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        historical = await source.get_historical(ticker, start_date, end_date)

        if historical and historical.prices:
            print(f"   ✅ {len(historical.prices)} jours de données")
            print(f"   Premier: {historical.prices[0].date} - ${historical.prices[0].close}")
            print(f"   Dernier: {historical.prices[-1].date} - ${historical.prices[-1].close}")
            results['historical'] = True
        else:
            print(f"   ⚠️  Aucun résultat")
            results['historical'] = False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        results['historical'] = False

    # Test 3: Fundamentals
    print(f"\n💼 Test Fondamentaux...")
    try:
        fundamentals = await source.get_fundamentals(ticker)

        if fundamentals:
            print(f"   ✅ Fondamentaux récupérés")
            if fundamentals.market_cap:
                print(f"   Market Cap: ${fundamentals.market_cap:,.0f}")
            if fundamentals.pe_ratio:
                print(f"   P/E Ratio: {fundamentals.pe_ratio}")
            if fundamentals.sector:
                print(f"   Secteur: {fundamentals.sector}")
            results['fundamentals'] = True
        else:
            print(f"   ⚠️  Aucun résultat")
            results['fundamentals'] = False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        results['fundamentals'] = False

    # Test 4: ESG (si disponible)
    if hasattr(source, 'get_esg_scores'):
        print(f"\n🌍 Test ESG...")
        try:
            esg = await source.get_esg_scores(ticker)

            if esg:
                print(f"   ✅ Scores ESG récupérés")
                print(f"   Score Total: {esg.total_score}")
                print(f"   Note: {esg.grade}")
                results['esg'] = True
            else:
                print(f"   ⚠️  Aucun résultat")
                results['esg'] = False
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['esg'] = False
    else:
        results['esg'] = None

    # Test 5: News (si disponible)
    if hasattr(source, 'get_news'):
        print(f"\n📰 Test Actualités...")
        try:
            news = await source.get_news(ticker, limit=3)

            if news:
                print(f"   ✅ {len(news)} articles récupérés")
                if len(news) > 0:
                    print(f"   Dernier: {news[0].title[:60]}...")
                results['news'] = True
            else:
                print(f"   ⚠️  Aucun résultat")
                results['news'] = False
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['news'] = False
    else:
        results['news'] = None

    return results


async def test_fred():
    """Test spécial pour FRED (données macro)"""
    print(f"\n{'='*70}")
    print(f"🧪 TEST: FRED (Données Macro)")
    print(f"{'='*70}")

    fred = FREDSource()

    if not fred.is_available():
        print(f"⚠️  FRED n'est pas disponible (clé API manquante)")
        return {'available': False}

    print(f"✅ FRED est disponible\n")

    results = {'available': True}

    # Test 1: Taux d'intérêt
    print(f"💰 Test Taux d'intérêt...")
    try:
        rates = await fred.get_interest_rates()

        if rates:
            print(f"   ✅ {len(rates)} taux récupérés")
            for name, rate in list(rates.items())[:3]:
                print(f"   {name}: {rate}%")
            results['rates'] = True
        else:
            print(f"   ⚠️  Aucun résultat")
            results['rates'] = False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        results['rates'] = False

    # Test 2: Inflation
    print(f"\n📊 Test Données d'inflation...")
    try:
        inflation = await fred.get_inflation_data()

        if inflation:
            print(f"   ✅ Données d'inflation récupérées")
            print(f"   CPI actuel: {inflation.get('current_cpi')}")
            print(f"   Inflation YoY: {inflation.get('inflation_yoy'):.2f}%")
            results['inflation'] = True
        else:
            print(f"   ⚠️  Aucun résultat")
            results['inflation'] = False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        results['inflation'] = False

    # Test 3: Snapshot macro
    print(f"\n🌐 Test Snapshot macro...")
    try:
        snapshot = await fred.get_macro_snapshot()

        if snapshot:
            print(f"   ✅ {len(snapshot)} indicateurs récupérés")
            for name, data in list(snapshot.items())[:3]:
                print(f"   {name}: {data['value']}")
            results['snapshot'] = True
        else:
            print(f"   ⚠️  Aucun résultat")
            results['snapshot'] = False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        results['snapshot'] = False

    return results


async def test_aggregator():
    """Teste l'aggregator avec fallback"""
    print(f"\n{'='*70}")
    print(f"🎯 TEST: DataAggregator (Avec Fallback)")
    print(f"{'='*70}")

    aggregator = DataAggregator()

    print(f"\n✅ Aggregator initialisé avec {len(aggregator.available_sources)} sources:")
    for source in aggregator.available_sources:
        print(f"   - {source.name}")

    if not aggregator.available_sources:
        print(f"\n⚠️  Aucune source disponible!")
        return

    ticker = "AAPL"

    # Test 1: Quote avec fallback
    print(f"\n📊 Test Quote avec fallback ({ticker})...")
    try:
        quote = await aggregator.get_quote(ticker)
        print(f"   ✅ Prix récupéré: ${quote.price}")
        print(f"   Source: {quote.source}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 2: Fondamentaux fusionnés
    print(f"\n💼 Test Fondamentaux fusionnés...")
    try:
        fundamentals = await aggregator.get_fundamentals_merged(ticker)

        if fundamentals:
            print(f"   ✅ Fondamentaux fusionnés depuis: {fundamentals.source}")
            if fundamentals.market_cap:
                print(f"   Market Cap: ${fundamentals.market_cap:,.0f}")
            if fundamentals.pe_ratio:
                print(f"   P/E Ratio: {fundamentals.pe_ratio}")
        else:
            print(f"   ⚠️  Aucun résultat")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 3: ESG
    print(f"\n🌍 Test ESG...")
    try:
        esg = await aggregator.get_esg_scores(ticker)

        if esg:
            print(f"   ✅ ESG récupéré via {esg.source}")
            print(f"   Score Total: {esg.total_score}")
            print(f"   Note: {esg.grade}")
        else:
            print(f"   ⚠️  Aucune source ESG disponible")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 4: News agrégées
    print(f"\n📰 Test News agrégées...")
    try:
        news = await aggregator.get_news(ticker, limit=5)

        print(f"   ✅ {len(news)} articles uniques agrégés")
        if news:
            print(f"   Dernier: {news[0].title[:60]}...")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")


async def main():
    """Point d'entrée principal"""

    print("="*70)
    print("🚀 TEST COMPLET DES SOURCES DE DONNÉES HELIXONE")
    print("="*70)

    # Dictionnaire pour stocker tous les résultats
    all_results = {}

    # Test de chaque source
    sources_to_test = [
        YahooFinanceSource(),
        FinnhubSource(),
        FMPSource(),
        AlphaVantageSource(),
        TwelveDataSource(),
    ]

    for source in sources_to_test:
        results = await test_source(source)
        all_results[source.name] = results

    # Test FRED séparément (données macro)
    fred_results = await test_fred()
    all_results['FRED'] = fred_results

    # Test aggregator
    await test_aggregator()

    # Afficher le résumé
    print(f"\n\n{'='*70}")
    print(f"📊 RÉSUMÉ DES TESTS")
    print(f"{'='*70}\n")

    print(f"{'Source':<15} {'Disponible':<12} {'Quote':<8} {'Historique':<12} {'Fondamentaux':<14} {'ESG':<6} {'News':<6}")
    print(f"{'-'*70}")

    for name, results in all_results.items():
        if name == 'FRED':
            # Format spécial pour FRED
            available = '✅' if results['available'] else '❌'
            rates = '✅' if results.get('rates') else ('⚠️' if results.get('rates') is False else '-')
            inflation = '✅' if results.get('inflation') else ('⚠️' if results.get('inflation') is False else '-')
            snapshot = '✅' if results.get('snapshot') else ('⚠️' if results.get('snapshot') is False else '-')
            print(f"{name:<15} {available:<12} {rates:<8} {inflation:<12} {snapshot:<14} {'-':<6} {'-':<6}")
        else:
            available = '✅' if results['available'] else '❌'
            quote = '✅' if results.get('quote') else ('⚠️' if results.get('quote') is False else '-')
            historical = '✅' if results.get('historical') else ('⚠️' if results.get('historical') is False else '-')
            fundamentals = '✅' if results.get('fundamentals') else ('⚠️' if results.get('fundamentals') is False else '-')
            esg = '✅' if results.get('esg') else ('⚠️' if results.get('esg') is False else ('-' if results.get('esg') is None else '-'))
            news = '✅' if results.get('news') else ('⚠️' if results.get('news') is False else ('-' if results.get('news') is None else '-'))

            print(f"{name:<15} {available:<12} {quote:<8} {historical:<12} {fundamentals:<14} {esg:<6} {news:<6}")

    print(f"\n{'='*70}")
    print(f"✅ TEST TERMINÉ!")
    print(f"{'='*70}")

    # Compter les sources disponibles
    available_count = sum(1 for r in all_results.values() if r.get('available'))
    print(f"\n💡 {available_count}/{len(all_results)} sources sont disponibles")

    if available_count < len(all_results):
        print(f"\n⚠️  Sources manquantes: Configurez les clés API dans .env")
        print(f"   Voir: OBTENIR_CLES_API.md pour obtenir les clés gratuites")


if __name__ == "__main__":
    asyncio.run(main())
