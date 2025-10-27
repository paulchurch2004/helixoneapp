"""
Test script pour Google Trends Data Collector
"""

import sys
import logging
from app.services.google_trends_collector import get_google_trends_collector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_google_trends():
    """Tester les fonctionnalités Google Trends"""

    print("\n" + "="*70)
    print("🔍 TEST GOOGLE TRENDS DATA COLLECTOR")
    print("="*70 + "\n")

    trends = get_google_trends_collector()

    # Test 1: Interest over time pour un ticker
    print("\n📈 Test 1: Interest over time pour AAPL")
    print("-" * 70)
    try:
        data = trends.get_ticker_interest('AAPL', timeframe='today 3-m', geo='US')
        print(f"✅ Interest over time récupéré")
        print(f"   Période: {data.index[0]} à {data.index[-1]}")
        print(f"   Points de données: {len(data)}")
        if 'AAPL' in data.columns:
            print(f"   Valeur actuelle: {data['AAPL'].iloc[-1]}")
            print(f"   Valeur max: {data['AAPL'].max()}")
            print(f"   Valeur moyenne: {data['AAPL'].mean():.2f}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test interest over time")

    # Test 2: Compare plusieurs tickers
    print("\n📊 Test 2: Comparaison AAPL vs MSFT vs GOOGL")
    print("-" * 70)
    try:
        data = trends.compare_tickers(['AAPL', 'MSFT', 'GOOGL'], timeframe='today 3-m', geo='US')
        print(f"✅ Comparaison récupérée")
        print(f"   Période: {data.index[0]} à {data.index[-1]}")
        print(f"   Points de données: {len(data)}")
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            if ticker in data.columns:
                print(f"   {ticker}: actuel={data[ticker].iloc[-1]}, max={data[ticker].max()}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test comparaison")

    # Test 3: Sentiment score
    print("\n🎯 Test 3: Sentiment score pour TSLA")
    print("-" * 70)
    try:
        score = trends.get_stock_sentiment_score('TSLA', timeframe='today 3-m')
        print(f"✅ Sentiment score calculé")
        print(f"   Ticker: {score['ticker']}")
        print(f"   Score: {score['sentiment_score']}")
        print(f"   Trend: {score['trend']}")
        if 'recent_avg' in score:
            print(f"   Recent avg: {score['recent_avg']}")
            print(f"   Overall avg: {score['overall_avg']}")
            print(f"   Current interest: {score['current_interest']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test sentiment")

    # Test 4: Related queries
    print("\n🔍 Test 4: Related queries pour 'Tesla stock'")
    print("-" * 70)
    try:
        queries = trends.get_related_queries(['Tesla'], timeframe='today 1-m', geo='US')
        print(f"✅ Related queries récupérées")
        for keyword, data in queries.items():
            print(f"   Keyword: {keyword}")
            if data['top'] is not None and not data['top'].empty:
                print(f"      Top queries: {len(data['top'])} trouvées")
                top_3 = data['top'].head(3)
                for idx, row in top_3.iterrows():
                    print(f"         - {row['query']} (value: {row['value']})")
            if data['rising'] is not None and not data['rising'].empty:
                print(f"      Rising queries: {len(data['rising'])} trouvées")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test related queries")

    # Test 5: Trending searches
    print("\n🔥 Test 5: Trending searches (United States)")
    print("-" * 70)
    try:
        trending = trends.get_trending_searches(country='united_states')
        print(f"✅ Trending searches récupérées")
        print(f"   Nombre: {len(trending)}")
        print(f"   Top 5:")
        for i, search in enumerate(trending.head(5)[0], 1):
            print(f"      {i}. {search}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test trending")

    # Test 6: Suggestions
    print("\n💡 Test 6: Suggestions pour 'Apple'")
    print("-" * 70)
    try:
        suggestions = trends.get_suggestions('Apple')
        print(f"✅ Suggestions récupérées")
        print(f"   Nombre: {len(suggestions)}")
        print(f"   Top 5:")
        for i, sugg in enumerate(suggestions[:5], 1):
            print(f"      {i}. {sugg['title']} (type: {sugg['type']})")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test suggestions")

    # Test 7: Interest by region
    print("\n🌍 Test 7: Interest by region pour NVDA")
    print("-" * 70)
    try:
        regions = trends.get_interest_by_region(['NVDA'], timeframe='today 3-m')
        print(f"✅ Interest by region récupéré")
        print(f"   Régions: {len(regions)}")
        # Afficher top 5 régions
        top_regions = regions.sort_values('NVDA', ascending=False).head(5)
        print(f"   Top 5 régions:")
        for idx, row in top_regions.iterrows():
            print(f"      {idx}: {row['NVDA']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test interest by region")

    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST GOOGLE TRENDS")
    print("="*70)
    print("✅ Source: Google Trends (pytrends)")
    print("✅ Limite: ILLIMITÉ (avec rate limiting)")
    print("✅ Clé API: Pas requise")
    print("✅ Données disponibles:")
    print("   - 📈 Interest over time (évolution de l'intérêt)")
    print("   - 📊 Compare tickers (comparaison)")
    print("   - 🎯 Sentiment score (score de sentiment)")
    print("   - 🔍 Related queries (requêtes associées)")
    print("   - 🔥 Trending searches (tendances)")
    print("   - 💡 Suggestions (suggestions)")
    print("   - 🌍 Interest by region (intérêt géographique)")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_google_trends()
    except Exception as e:
        logger.exception("Erreur globale test Google Trends")
        sys.exit(1)
