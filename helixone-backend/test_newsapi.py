"""
Script de test pour NewsAPI.org
Test des actualités - GRATUIT (100 req/jour)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.newsapi_source import get_newsapi_collector
from datetime import datetime, timedelta


def test_newsapi():
    """Tester NewsAPI.org"""

    print("\n" + "="*70)
    print("📰 TEST NEWSAPI.ORG - NEWS AGGREGATOR")
    print("GRATUIT - 100 req/jour - Clé API requise")
    print("="*70 + "\n")

    # Check if API key is configured
    api_key = os.getenv('NEWSAPI_API_KEY')

    if not api_key:
        print("❌ Clé API NewsAPI non configurée!\n")
        print("🔑 Pour obtenir une clé GRATUITE:")
        print("   1. Aller sur: https://newsapi.org/register")
        print("   2. S'inscrire avec votre email (2 minutes)")
        print("   3. Copier votre clé API")
        print("   4. Ajouter au .env:")
        print("      NEWSAPI_API_KEY=votre_clé_ici")
        print("\n" + "="*70 + "\n")
        return

    try:
        newsapi = get_newsapi_collector()
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}\n")
        return

    # Test 1: Get Available Sources
    print("📚 Test 1: Sources disponibles (business)")
    print("-" * 70)
    try:
        sources_result = newsapi.get_sources(category='business')
        sources = sources_result.get('sources', [])

        if sources:
            print(f"\n✅ {len(sources)} sources business trouvées\n")
            print(f"{'ID':<20} {'Nom':<30} {'Pays':<5}")
            print("-" * 70)

            # Show top 10
            for source in sources[:10]:
                sid = source.get('id', 'N/A')
                name = source.get('name', 'Unknown')[:29]
                country = source.get('country', '').upper()

                print(f"{sid:<20} {name:<30} {country:<5}")

            print()
        else:
            print("⚠️  Aucune source trouvée\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 2: Top Headlines - Business
    print("📊 Test 2: Top Headlines - Business (US)")
    print("-" * 70)
    try:
        headlines = newsapi.get_top_headlines(
            category='business',
            country='us',
            page_size=5
        )

        articles = headlines.get('articles', [])
        total_results = headlines.get('totalResults', 0)

        if articles:
            print(f"\n✅ {total_results} articles trouvés (affichage de 5)\n")

            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No title')
                source_name = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]
                url = article.get('url', '')

                print(f"{i}. [{source_name}] {title}")
                print(f"   📅 {published}")
                print(f"   🔗 {url[:60]}...")
                print()

        else:
            print("⚠️  Aucun article trouvé\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 3: Search Everything - Bitcoin
    print("🔍 Test 3: Recherche 'Bitcoin' (7 derniers jours)")
    print("-" * 70)
    try:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        bitcoin_news = newsapi.get_everything(
            q='Bitcoin OR BTC',
            from_date=from_date,
            sort_by='relevancy',
            page_size=5
        )

        articles = bitcoin_news.get('articles', [])
        total = bitcoin_news.get('totalResults', 0)

        if articles:
            print(f"\n✅ {total} articles Bitcoin trouvés (affichage de 5)\n")

            for i, article in enumerate(articles, 1):
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]
                description = article.get('description', 'No description')[:100]

                print(f"{i}. [{source}] {title}")
                print(f"   📅 {published}")
                print(f"   📝 {description}...")
                print()

        else:
            print("⚠️  Aucun article Bitcoin trouvé\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 4: Financial News (Convenience Method)
    print("💰 Test 4: Actualités financières (méthode pratique)")
    print("-" * 70)
    try:
        financial_articles = newsapi.get_financial_news(
            days_back=3,
            page_size=5
        )

        if financial_articles:
            print(f"\n✅ {len(financial_articles)} articles financiers récents\n")

            for i, article in enumerate(financial_articles, 1):
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]

                print(f"{i}. [{source}] {title}")
                print(f"   📅 {published}")
                print()

        else:
            print("⚠️  Aucun article financier trouvé\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 5: Company News - Apple
    print("🍎 Test 5: Actualités Apple (AAPL)")
    print("-" * 70)
    try:
        apple_news = newsapi.get_company_news(
            company_name='Apple',
            ticker='AAPL',
            days_back=14,
            page_size=5
        )

        if apple_news:
            print(f"\n✅ {len(apple_news)} articles Apple trouvés\n")

            for i, article in enumerate(apple_news, 1):
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]

                print(f"{i}. [{source}] {title}")
                print(f"   📅 {published}")
                print()

        else:
            print("⚠️  Aucun article Apple trouvé\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 6: Crypto News
    print("🪙 Test 6: Actualités Crypto (général)")
    print("-" * 70)
    try:
        crypto_news = newsapi.get_crypto_news(
            days_back=5,
            page_size=5
        )

        if crypto_news:
            print(f"\n✅ {len(crypto_news)} articles crypto trouvés\n")

            for i, article in enumerate(crypto_news, 1):
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]

                print(f"{i}. [{source}] {title}")
                print(f"   📅 {published}")
                print()

        else:
            print("⚠️  Aucun article crypto trouvé\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 7: Sector News - Technology
    print("💻 Test 7: Actualités Secteur Tech")
    print("-" * 70)
    try:
        tech_news = newsapi.get_sector_news(
            sector='technology',
            days_back=7,
            page_size=5
        )

        if tech_news:
            print(f"\n✅ {len(tech_news)} articles tech trouvés\n")

            for i, article in enumerate(tech_news, 1):
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown')
                published = article.get('publishedAt', '')[:10]

                print(f"{i}. [{source}] {title}")
                print(f"   📅 {published}")
                print()

        else:
            print("⚠️  Aucun article tech trouvé\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Summary
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST NEWSAPI.ORG")
    print("="*70)
    print("✅ Tous les tests exécutés")
    print("📰 NewsAPI.org est maintenant intégré dans HelixOne!")
    print("\nCaractéristiques:")
    print("  - ✅ GRATUIT (100 req/jour)")
    print("  - ✅ 80,000+ sources de news")
    print("  - ✅ 150+ pays")
    print("  - ✅ Recherche avancée")
    print("  - ✅ Filtres par date, source, langue")
    print("  - ✅ Business, tech, finance, crypto")
    print("\nDonnées disponibles:")
    print("  - 📰 Top headlines par pays/catégorie")
    print("  - 🔍 Recherche complète avec opérateurs")
    print("  - 🏢 Actualités par entreprise")
    print("  - 🪙 Actualités crypto")
    print("  - 📊 Actualités par secteur")
    print("  - 📈 Actualités financières filtrées")
    print("\nLimites Free Tier:")
    print("  - 100 requêtes/jour")
    print("  - News jusqu'à 1 mois dans le passé")
    print("  - Pas de données historiques > 1 mois")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_newsapi()
