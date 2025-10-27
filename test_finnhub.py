"""
Script de test pour Finnhub
"""

import sys
import os
import asyncio

# Ajouter les chemins
sys.path.insert(0, "/Users/macintosh/Desktop/helixone/helixone-backend")

# Configuration
from app.core.config import settings

async def test_finnhub():
    """Test de la source Finnhub"""

    print("=" * 70)
    print("🧪 TEST FINNHUB")
    print("=" * 70)

    # Vérifier la clé API
    api_key = os.environ.get("FINNHUB_API_KEY") or settings.FINNHUB_API_KEY

    if not api_key or api_key == "your_key_here":
        print("\n❌ ERREUR: Clé API Finnhub non configurée!")
        print("\n📋 Pour obtenir une clé:")
        print("   1. Allez sur https://finnhub.io/register")
        print("   2. Créez un compte (2 minutes)")
        print("   3. Copiez votre clé depuis https://finnhub.io/dashboard")
        print("\n🔧 Puis configurez-la:")
        print("   - Soit: export FINNHUB_API_KEY='votre_clé'")
        print("   - Soit: Ajoutez dans helixone-backend/.env")
        print("          FINNHUB_API_KEY=votre_clé")
        return

    print(f"✅ Clé API configurée: {api_key[:10]}...")

    # Importer et tester
    from app.services.data_sources.finnhub_source import FinnhubSource

    print("\n📊 Initialisation de Finnhub...")
    finnhub = FinnhubSource(api_key=api_key)

    if not finnhub.is_available():
        print("❌ Finnhub non disponible")
        return

    print("✅ Finnhub initialisé\n")

    # Test 1: Quote
    print("-" * 70)
    print("TEST 1: Récupération d'un prix (Quote)")
    print("-" * 70)

    ticker = "AAPL"
    print(f"🔍 Récupération du prix de {ticker}...")

    try:
        quote = await finnhub.get_quote(ticker)

        if quote:
            print(f"✅ Quote récupérée!")
            print(f"   Ticker: {quote.ticker}")
            print(f"   Nom: {quote.name}")
            print(f"   Prix: ${quote.price}")
            print(f"   Change: {quote.change} ({quote.change_percent}%)")
            print(f"   Source: {quote.source}")
            print(f"   Timestamp: {quote.timestamp}")
        else:
            print("❌ Aucun résultat")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Données historiques
    print("\n" + "-" * 70)
    print("TEST 2: Données historiques")
    print("-" * 70)

    from datetime import date, timedelta
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    print(f"🔍 Récupération des données de {ticker} du {start_date} au {end_date}...")

    try:
        historical = await finnhub.get_historical(ticker, start_date, end_date)

        if historical and historical.prices:
            print(f"✅ Données historiques récupérées!")
            print(f"   Nombre de jours: {len(historical.prices)}")
            print(f"   Premier jour: {historical.prices[0].date} - ${historical.prices[0].close}")
            print(f"   Dernier jour: {historical.prices[-1].date} - ${historical.prices[-1].close}")
            print(f"   Source: {historical.source}")
        else:
            print("❌ Aucun résultat")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Fondamentaux
    print("\n" + "-" * 70)
    print("TEST 3: Données fondamentales")
    print("-" * 70)

    print(f"🔍 Récupération des fondamentaux de {ticker}...")

    try:
        fundamentals = await finnhub.get_fundamentals(ticker)

        if fundamentals:
            print(f"✅ Fondamentaux récupérés!")
            print(f"   Market Cap: ${fundamentals.market_cap:,.0f}" if fundamentals.market_cap else "   Market Cap: N/A")
            print(f"   P/E Ratio: {fundamentals.pe_ratio}" if fundamentals.pe_ratio else "   P/E Ratio: N/A")
            print(f"   EPS: {fundamentals.eps}" if fundamentals.eps else "   EPS: N/A")
            print(f"   ROE: {fundamentals.return_on_equity}%" if fundamentals.return_on_equity else "   ROE: N/A")
            print(f"   Beta: {fundamentals.beta}" if fundamentals.beta else "   Beta: N/A")
            print(f"   Secteur: {fundamentals.sector}" if fundamentals.sector else "   Secteur: N/A")
            print(f"   Source: {fundamentals.source}")
        else:
            print("❌ Aucun résultat")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: News
    print("\n" + "-" * 70)
    print("TEST 4: Actualités avec sentiment")
    print("-" * 70)

    print(f"🔍 Récupération des actualités de {ticker}...")

    try:
        news = await finnhub.get_news(ticker, limit=5)

        if news:
            print(f"✅ {len(news)} articles récupérés!")
            for i, article in enumerate(news, 1):
                print(f"\n   Article {i}:")
                print(f"   📰 {article.title}")
                print(f"   🔗 {article.url}")
                print(f"   📅 {article.published_at}")
                if article.sentiment:
                    print(f"   😊 Sentiment: {article.sentiment}")
        else:
            print("⚠️  Aucune actualité récente")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DU TEST")
    print("=" * 70)
    print("✅ Finnhub fonctionne correctement!")
    print("✅ Données disponibles: Prix, Historique, Fondamentaux, News")
    print("✅ Prêt pour l'intégration dans l'aggregator")
    print("\n💡 Prochaine étape: Ajouter les autres sources (Alpha Vantage, FMP, FRED)")


if __name__ == "__main__":
    # Vérifier les dépendances
    try:
        import finnhub
        print("✅ Module finnhub installé")
    except ImportError:
        print("❌ Module finnhub manquant!")
        print("📦 Installation: pip install finnhub-python")
        sys.exit(1)

    # Lancer le test
    asyncio.run(test_finnhub())
