#!/usr/bin/env python3
"""Test de diagnostic Reddit API"""

import os
import sys
sys.path.insert(0, 'helixone-backend')

# Charger les variables d'environnement
from dotenv import load_dotenv
load_dotenv('helixone-backend/.env')

print("="*70)
print("🔍 DIAGNOSTIC REDDIT API")
print("="*70)

# Afficher les clés (masquées)
client_id = os.getenv('REDDIT_CLIENT_ID', 'NOT_FOUND')
client_secret = os.getenv('REDDIT_CLIENT_SECRET', 'NOT_FOUND')
user_agent = os.getenv('REDDIT_USER_AGENT', 'NOT_FOUND')

print(f"\n✅ Variables d'environnement:")
print(f"   REDDIT_CLIENT_ID: {client_id[:10]}... (len={len(client_id)})")
print(f"   REDDIT_CLIENT_SECRET: {client_secret[:10]}... (len={len(client_secret)})")
print(f"   REDDIT_USER_AGENT: {user_agent}")

# Tester la connexion
print(f"\n🔌 Test de connexion à Reddit API...")
try:
    import praw

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False
    )

    # Test simple: récupérer un subreddit public
    print(f"   Tentative de connexion...")
    subreddit = reddit.subreddit('wallstreetbets')

    # Forcer une requête
    print(f"   Récupération du nom: {subreddit.display_name}")
    print(f"   Récupération des subscribers...")
    subscribers = subreddit.subscribers

    print(f"\n✅ SUCCÈS!")
    print(f"   r/wallstreetbets: {subscribers:,} subscribers")

    # Test de récupération de posts
    print(f"\n📝 Test récupération de posts...")
    posts = list(subreddit.hot(limit=3))
    print(f"   ✅ {len(posts)} posts récupérés")
    for i, post in enumerate(posts, 1):
        print(f"   {i}. {post.title[:50]}...")
        print(f"      Score: {post.score}, Comments: {post.num_comments}")

except Exception as e:
    print(f"\n❌ ERREUR: {str(e)}")
    print(f"\n💡 Solutions possibles:")
    print(f"   1. Vérifier que l'app Reddit est bien 'script' type")
    print(f"   2. Vérifier que les clés sont correctes")
    print(f"   3. Attendre quelques minutes (propagation)")
    print(f"   4. Regénérer le secret sur reddit.com/prefs/apps")

print("\n" + "="*70)
