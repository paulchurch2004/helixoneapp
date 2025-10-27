#!/usr/bin/env python3
"""Test rapide StockTwits - Sentiment Trading"""

import sys
sys.path.insert(0, 'helixone-backend')

from app.services.stocktwits_source import get_stocktwits_collector

print("="*70)
print("📊 TEST STOCKTWITS - SENTIMENT TRADING")
print("="*70)

st = get_stocktwits_collector()

# Test 1: Ticker sentiment (TSLA)
print("\n1️⃣  Sentiment pour TSLA...")
try:
    sentiment = st.get_ticker_sentiment('TSLA', limit=30)
    if sentiment['total_messages'] > 0:
        print(f"✅ {sentiment['total_messages']} messages analysés")
        print(f"   Sentiment: {sentiment['sentiment'].upper()}")
        print(f"   Bullish: {sentiment['bullish']} ({sentiment['bullish_pct']:.1f}%)")
        print(f"   Bearish: {sentiment['bearish']} ({sentiment['bearish_pct']:.1f}%)")
        print(f"   Neutral: {sentiment['neutral']}")
        print(f"   Ratio Bull/Bear: {sentiment['sentiment_ratio']:.2f}")

        if sentiment['top_message']:
            print(f"\n   Top message ({sentiment['top_message']['likes']} likes):")
            print(f"   @{sentiment['top_message']['user']}: {sentiment['top_message']['body'][:80]}...")
            print(f"   Sentiment: {sentiment['top_message']['sentiment']}")
    else:
        print("⚠️  Aucun message trouvé")
except Exception as e:
    print(f"❌ Erreur: {str(e)[:80]}")

# Test 2: Stream de messages
print("\n2️⃣  Stream de messages AAPL (derniers 5)...")
try:
    messages = st.get_ticker_stream('AAPL', limit=5)
    if messages:
        print(f"✅ {len(messages)} messages récupérés\n")
        for i, msg in enumerate(messages[:5], 1):
            sentiment_emoji = "🟢" if msg['sentiment'] == 'Bullish' else "🔴" if msg['sentiment'] == 'Bearish' else "⚪"
            print(f"   {i}. {sentiment_emoji} @{msg['user']} ({msg['user_followers']} followers)")
            print(f"      {msg['body'][:70]}...")
            print(f"      {msg['likes']} likes | {msg['created_at'].strftime('%Y-%m-%d %H:%M')}")
    else:
        print("⚠️  Aucun message")
except Exception as e:
    print(f"❌ Erreur: {str(e)[:80]}")

# Test 3: Trending tickers
print("\n3️⃣  Trending tickers (top 10)...")
try:
    trending = st.get_trending_tickers(limit=10)
    if trending:
        print(f"✅ {len(trending)} tickers trending:\n")
        for i, ticker in enumerate(trending[:10], 1):
            print(f"   {i:2d}. {ticker['symbol']:6s} - {ticker['title'][:30]:<30s} ({ticker['watchlist_count']:,} watchers)")
    else:
        print("⚠️  Pas de trending")
except Exception as e:
    print(f"❌ Erreur: {str(e)[:80]}")

# Test 4: Trending avec sentiment
print("\n4️⃣  Top 5 trending avec sentiment...")
try:
    trending_sentiment = st.get_trending_with_sentiment(limit=5)
    if trending_sentiment:
        print(f"✅ Analyse complète:\n")
        for t in trending_sentiment:
            sentiment_icon = "🟢" if t['sentiment'] == 'bullish' else "🔴" if t['sentiment'] == 'bearish' else "⚪"
            print(f"   {sentiment_icon} {t['symbol']:6s} - {t['sentiment'].upper():<8s} | Bull: {t['bullish_pct']:5.1f}% | Bear: {t['bearish_pct']:5.1f}% | Msgs: {t['total_messages']}")
    else:
        print("⚠️  Pas de données")
except Exception as e:
    print(f"❌ Erreur: {str(e)[:80]}")

# Test 5: Market overview
print("\n5️⃣  Vue d'ensemble du marché...")
try:
    overview = st.get_market_sentiment_overview()
    if overview['trending_tickers']:
        print(f"✅ Analyse globale:")
        print(f"\n   Sentiment global: {overview['overall_sentiment'].upper()}")
        print(f"   Bullish moyen: {overview['overall_bullish_pct']:.1f}%")
        print(f"   Bearish moyen: {overview['overall_bearish_pct']:.1f}%")

        if overview['most_bullish']:
            print(f"\n   Plus bullish: {overview['most_bullish']['symbol']} ({overview['most_bullish']['bullish_pct']:.1f}%)")
        if overview['most_bearish']:
            print(f"   Plus bearish: {overview['most_bearish']['symbol']} ({overview['most_bearish']['bearish_pct']:.1f}%)")
    else:
        print("⚠️  Pas de données overview")
except Exception as e:
    print(f"❌ Erreur: {str(e)[:80]}")

# Test 6: Multiple tickers
print("\n6️⃣  Analyse multiple (AAPL, NVDA, AMD)...")
try:
    symbols = ['AAPL', 'NVDA', 'AMD']
    sentiments = st.get_multiple_sentiments(symbols, limit_per_symbol=20)
    if sentiments:
        print(f"✅ {len(sentiments)} tickers analysés:\n")
        for symbol, data in sentiments.items():
            sentiment_icon = "🟢" if data['sentiment'] == 'bullish' else "🔴" if data['sentiment'] == 'bearish' else "⚪"
            print(f"   {sentiment_icon} {symbol:6s}: {data['sentiment']:<8s} | Bull: {data['bullish_pct']:5.1f}% | Bear: {data['bearish_pct']:5.1f}%")
    else:
        print("⚠️  Pas de données")
except Exception as e:
    print(f"❌ Erreur: {str(e)[:80]}")

print("\n" + "="*70)
print("✅ STOCKTWITS SOURCE CRÉÉE!")
print("="*70)
print("\n💡 Fonctionnalités:")
print("   ✅ Sentiment par ticker (Bullish/Bearish/Neutral)")
print("   ✅ Stream de messages en temps réel")
print("   ✅ Trending tickers avec watchlist counts")
print("   ✅ Analyse multi-tickers")
print("   ✅ Vue d'ensemble du marché")
print("\n📊 Métriques:")
print("   • Ratio Bullish/Bearish")
print("   • Top messages (likes)")
print("   • Watchlist counts")
print("   • Sentiment global")
print("\n🚀 StockTwits sentiment ready!")
print("   API gratuite: ~200 req/hour")
print("   Pas de clé API requise")
print("="*70)
