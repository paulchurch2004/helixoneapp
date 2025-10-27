#!/usr/bin/env python3
"""Test rapide NewsAPI après configuration"""

import sys
import os
from pathlib import Path

# Charger .env explicitement
from dotenv import load_dotenv

# Trouver le .env
env_path = Path(__file__).parent / 'helixone-backend' / '.env'
load_dotenv(env_path)

# Ajouter helixone-backend au path
sys.path.insert(0, str(Path(__file__).parent / 'helixone-backend'))

print("\n" + "="*70)
print("📰 TEST RAPIDE NEWSAPI")
print("="*70 + "\n")

# Vérifier la clé
api_key = os.getenv('NEWSAPI_API_KEY')
print(f"🔑 Clé API détectée: {api_key[:8]}...{api_key[-4:] if api_key else 'AUCUNE'}\n")

if not api_key:
    print("❌ Clé API manquante!")
    sys.exit(1)

# Tester NewsAPI
try:
    from app.services.newsapi_source import get_newsapi_collector

    print("📡 Initialisation NewsAPI...")
    news = get_newsapi_collector()

    print("📚 Récupération des sources business...")
    sources_result = news.get_sources(category='business')
    sources = sources_result.get('sources', [])

    print(f"\n✅ SUCCESS! {len(sources)} sources business trouvées\n")

    # Afficher quelques sources
    print("Top 10 sources:")
    print("-" * 70)
    for source in sources[:10]:
        name = source.get('name', 'Unknown')
        country = source.get('country', 'XX').upper()
        print(f"  • {name:<40} ({country})")

    print("\n" + "="*70)
    print("🎉 NewsAPI fonctionne parfaitement!")
    print("="*70 + "\n")

except Exception as e:
    print(f"\n❌ Erreur: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
