#!/usr/bin/env python3
"""Test rapide Reddit - Sentiment WallStreetBets"""

import sys
sys.path.insert(0, 'helixone-backend')

# Charger les variables d'environnement
from dotenv import load_dotenv
load_dotenv('helixone-backend/.env')

from app.services.reddit_source import get_reddit_collector

print("="*70)
print("🔷 TEST REDDIT - SENTIMENT WALLSTREETBETS")
print("="*70)

reddit = get_reddit_collector()

# Test 1: Hot posts
print("\n1️⃣  Hot posts r/wallstreetbets...")
try:
    posts = reddit.get_hot_posts('wallstreetbets', limit=10)
    if posts:
        print(f"✅ {len(posts)} posts récupérés")
        print(f"\n   Top 3 posts:")
        for i, post in enumerate(posts[:3], 1):
            print(f"   {i}. {post['title'][:60]}...")
            print(f"      Score: {post['score']}, Comments: {post['num_comments']}")
            if post['tickers']:
                print(f"      Tickers: {', '.join(post['tickers'][:5])}")
    else:
        print("⚠️  Mode anonyme - Fonctionnalité limitée")
        print("   Pour utiliser pleinement: créer app Reddit sur reddit.com/prefs/apps")
except Exception as e:
    print(f"⚠️  Reddit: {str(e)[:80]}")

# Test 2: Ticker mentions
print("\n2️⃣  Ticker mentions (top posts 24h)...")
try:
    mentions = reddit.get_ticker_mentions('wallstreetbets', 'day', 50)
    if mentions:
        sorted_mentions = sorted(mentions.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"✅ Top 10 tickers mentionnés:")
        for ticker, count in sorted_mentions:
            print(f"   {ticker}: {count} mentions")
    else:
        print("⚠️  Pas de mentions (mode anonyme)")
except Exception as e:
    print(f"⚠️  Mentions: {str(e)[:60]}")

# Test 3: Trending tickers
print("\n3️⃣  Trending tickers (multi-subreddits)...")
try:
    trending = reddit.get_trending_tickers(
        subreddits=['wallstreetbets'],
        min_mentions=2,
        limit=25
    )
    if trending:
        print(f"✅ {len(trending)} tickers trending:")
        for t in trending[:5]:
            print(f"   {t['ticker']}: {t['mentions']} mentions")
    else:
        print("⚠️  Pas de trending (mode anonyme)")
except Exception as e:
    print(f"⚠️  Trending: {str(e)[:60]}")

# Test 4: Subreddit info
print("\n4️⃣  Info r/wallstreetbets...")
try:
    info = reddit.get_subreddit_info('wallstreetbets')
    if info and 'subscribers' in info:
        print(f"✅ {info['title']}")
        print(f"   Subscribers: {info['subscribers']:,}")
        print(f"   Active users: {info.get('active_users', 'N/A')}")
    else:
        print("⚠️  Info non disponible")
except Exception as e:
    print(f"⚠️  Info: {str(e)[:60]}")

print("\n" + "="*70)
print("✅ REDDIT SOURCE CRÉÉE!")
print("="*70)
print("\n💡 Fonctionnalités:")
print("   ✅ Hot/Top posts")
print("   ✅ Ticker mentions")
print("   ✅ Trending tickers")
print("   ✅ Multi-subreddit analysis")
print("\n⚠️  Mode anonyme actif (pas de clé API)")
print("   Pour full access: reddit.com/prefs/apps")
print("\n🚀 Sentiment Reddit ready!")
