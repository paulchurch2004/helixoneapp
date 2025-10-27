#!/usr/bin/env python3
"""
Script de test pour Alpha Vantage
"""
import os
import sys

# Charger les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv()

# Ajouter le chemin du backend au PYTHONPATH
sys.path.insert(0, '/Users/macintosh/Desktop/helixone/helixone-backend')

from app.services.alpha_vantage_collector import AlphaVantageCollector

def test_alpha_vantage():
    """Tester Alpha Vantage avec votre clé API"""

    print("=" * 60)
    print("🧪 TEST ALPHA VANTAGE")
    print("=" * 60)

    # Récupérer la clé depuis l'environnement
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key or api_key == "your_alpha_vantage_key_here":
        print("❌ ERREUR: Clé API non configurée dans .env")
        return False

    print(f"\n✅ Clé API trouvée: {api_key[:8]}...")

    # Initialiser le collecteur
    av = AlphaVantageCollector(api_key=api_key)

    # Test 1: Quote temps réel
    print("\n" + "-" * 60)
    print("📊 Test 1: Quote temps réel (AAPL)")
    print("-" * 60)

    try:
        quote = av.get_quote("AAPL")
        print(f"✅ Quote récupérée:")
        print(f"   Symbole: {quote['symbol']}")
        print(f"   Prix: ${quote['price']:.2f}")
        print(f"   Volume: {quote['volume']:,}")
        print(f"   Change: {quote['change']:+.2f} ({quote['change_percent']})")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    # Test 2: Company Overview
    print("\n" + "-" * 60)
    print("🏢 Test 2: Company Overview (AAPL)")
    print("-" * 60)

    try:
        overview = av.get_company_overview("AAPL")
        print(f"✅ Informations récupérées:")
        print(f"   Nom: {overview['name']}")
        print(f"   Secteur: {overview['sector']}")
        print(f"   Industrie: {overview['industry']}")
        print(f"   Market Cap: ${overview['market_cap']:,}")
        print(f"   P/E Ratio: {overview['pe_ratio']}")
        print(f"   Beta: {overview['beta']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    # Test 3: Statistiques d'utilisation
    print("\n" + "-" * 60)
    print("📈 Statistiques d'utilisation")
    print("-" * 60)

    stats = av.get_usage_stats()
    print(f"   Requêtes aujourd'hui: {stats['requests_today']}/{stats['max_requests_per_day']}")
    print(f"   Restantes: {stats['remaining']}")
    print(f"   Utilisé: {stats['percentage_used']:.1f}%")

    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("=" * 60)
    print("\n💡 Alpha Vantage fonctionne correctement!")
    print("   Vous pouvez maintenant collecter:")
    print("   - Prix historiques (20+ ans)")
    print("   - Données intraday")
    print("   - États financiers")
    print("   - Indicateurs techniques")

    return True


if __name__ == "__main__":
    success = test_alpha_vantage()
    sys.exit(0 if success else 1)
