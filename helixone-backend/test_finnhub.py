"""
Script de test pour Finnhub API
Test des fonctionnalités principales: news, sentiment, analystes
"""

import sys
import os
from datetime import datetime, timedelta

# Ajouter le chemin du backend au path Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.services.finnhub_collector import get_finnhub_collector

def test_finnhub():
    """Tester toutes les fonctionnalités Finnhub"""

    print("\n" + "="*70)
    print("TEST FINNHUB API - NEWS & SENTIMENT")
    print("="*70 + "\n")

    finnhub = get_finnhub_collector()
    test_symbol = "AAPL"

    # Test 1: Company News
    print(f"📰 Test 1: News pour {test_symbol}")
    print("-" * 70)
    try:
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        news = finnhub.get_company_news(test_symbol, from_date, to_date)

        if news and len(news) > 0:
            print(f"✅ {len(news)} articles trouvés")
            # Afficher les 3 premiers
            for i, article in enumerate(news[:3]):
                print(f"\n   Article {i+1}:")
                print(f"   📌 {article.get('headline', 'N/A')[:80]}...")
                print(f"   🔗 {article.get('url', 'N/A')[:60]}...")
                print(f"   📅 {datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')}")
        else:
            print("⚠️  Aucun article trouvé")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 2: News Sentiment
    print(f"\n\n💭 Test 2: Sentiment des news pour {test_symbol}")
    print("-" * 70)
    try:
        sentiment = finnhub.get_news_sentiment(test_symbol)

        if sentiment:
            print(f"✅ Sentiment récupéré:")
            print(f"   Score buzz: {sentiment.get('buzz', {}).get('articlesInLastWeek', 0)} articles (7 jours)")
            print(f"   Score buzz: {sentiment.get('buzz', {}).get('buzz', 0):.2f}")
            print(f"   Sentiment positif: {sentiment.get('sentiment', {}).get('bullishPercent', 0):.1f}%")
            print(f"   Sentiment négatif: {sentiment.get('sentiment', {}).get('bearishPercent', 0):.1f}%")

            sentiment_score = sentiment.get('companyNewsScore', 0)
            if sentiment_score > 0:
                print(f"   📊 Score global: {sentiment_score:.2f} (POSITIF)")
            elif sentiment_score < 0:
                print(f"   📊 Score global: {sentiment_score:.2f} (NÉGATIF)")
            else:
                print(f"   📊 Score global: NEUTRE")
        else:
            print("⚠️  Sentiment non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 3: Social Sentiment
    print(f"\n\n🌐 Test 3: Sentiment réseaux sociaux pour {test_symbol}")
    print("-" * 70)
    try:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        social = finnhub.get_social_sentiment(test_symbol, from_date)

        if social and 'reddit' in social:
            reddit = social['reddit']
            print(f"✅ Sentiment Reddit:")
            print(f"   Mentions: {reddit.get('mention', 0)}")
            print(f"   Score positif: {reddit.get('positiveMention', 0)}")
            print(f"   Score négatif: {reddit.get('negativeMention', 0)}")
            print(f"   Score: {reddit.get('score', 0):.2f}")

        if social and 'twitter' in social:
            twitter = social['twitter']
            print(f"\n✅ Sentiment Twitter:")
            print(f"   Mentions: {twitter.get('mention', 0)}")
            print(f"   Score positif: {twitter.get('positiveMention', 0)}")
            print(f"   Score négatif: {twitter.get('negativeMention', 0)}")
            print(f"   Score: {twitter.get('score', 0):.2f}")

        if not social or (not social.get('reddit') and not social.get('twitter')):
            print("⚠️  Données sociales non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 4: Analyst Recommendations
    print(f"\n\n👔 Test 4: Recommandations analystes pour {test_symbol}")
    print("-" * 70)
    try:
        recommendations = finnhub.get_recommendation_trends(test_symbol)

        if recommendations and len(recommendations) > 0:
            latest = recommendations[0]
            print(f"✅ Recommandations (période: {latest.get('period', 'N/A')}):")
            print(f"   🟢 Strong Buy: {latest.get('strongBuy', 0)}")
            print(f"   🟢 Buy: {latest.get('buy', 0)}")
            print(f"   ⚪ Hold: {latest.get('hold', 0)}")
            print(f"   🔴 Sell: {latest.get('sell', 0)}")
            print(f"   🔴 Strong Sell: {latest.get('strongSell', 0)}")

            total = (latest.get('strongBuy', 0) + latest.get('buy', 0) +
                    latest.get('hold', 0) + latest.get('sell', 0) +
                    latest.get('strongSell', 0))
            print(f"   📊 Total analystes: {total}")
        else:
            print("⚠️  Recommandations non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 5: Price Target
    print(f"\n\n🎯 Test 5: Prix cible pour {test_symbol}")
    print("-" * 70)
    try:
        price_target = finnhub.get_price_target(test_symbol)

        if price_target:
            print(f"✅ Objectifs de prix:")
            print(f"   Prix actuel: ${price_target.get('lastUpdated', 'N/A')}")
            print(f"   Prix cible moyen: ${price_target.get('targetMean', 0):.2f}")
            print(f"   Prix cible médian: ${price_target.get('targetMedian', 0):.2f}")
            print(f"   Prix cible haut: ${price_target.get('targetHigh', 0):.2f}")
            print(f"   Prix cible bas: ${price_target.get('targetLow', 0):.2f}")
            print(f"   Nombre d'analystes: {price_target.get('numberOfAnalysts', 0)}")

            # Calculer le potentiel
            current = finnhub.get_quote(test_symbol).get('c', 0)
            target_mean = price_target.get('targetMean', 0)
            if current and target_mean:
                upside = ((target_mean - current) / current) * 100
                print(f"   📈 Potentiel: {upside:+.2f}%")
        else:
            print("⚠️  Prix cible non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 6: Earnings Calendar
    print(f"\n\n📅 Test 6: Calendrier earnings")
    print("-" * 70)
    try:
        from_date = datetime.now().strftime('%Y-%m-%d')
        to_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

        earnings = finnhub.get_earnings_calendar(from_date=from_date, to_date=to_date)

        if earnings and 'earningsCalendar' in earnings:
            calendar = earnings['earningsCalendar']
            print(f"✅ {len(calendar)} publications prévues dans les 30 prochains jours")

            # Afficher les 5 premières
            for i, event in enumerate(calendar[:5]):
                print(f"\n   {i+1}. {event.get('symbol', 'N/A')}")
                print(f"      Date: {event.get('date', 'N/A')}")
                print(f"      EPS estimé: ${event.get('epsEstimate', 0):.2f}")
        else:
            print("⚠️  Calendrier non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Résumé
    print("\n\n" + "="*70)
    print("RÉSUMÉ DES TESTS FINNHUB")
    print("="*70)
    print("✅ Tous les tests ont été exécutés")
    print("📊 Finnhub est maintenant intégré dans HelixOne!")
    print("\nFonctionnalités disponibles:")
    print("  - 📰 News en temps réel")
    print("  - 💭 Analyse de sentiment (news)")
    print("  - 🌐 Sentiment réseaux sociaux (Reddit, Twitter)")
    print("  - 👔 Recommandations analystes")
    print("  - 🎯 Objectifs de prix")
    print("  - 📅 Calendrier des earnings")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_finnhub()
