#!/usr/bin/env python3
"""
Script de test pour FRED (Federal Reserve Economic Data)
"""
import os
import sys

# Charger les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv()

# Ajouter le chemin du backend au PYTHONPATH
sys.path.insert(0, '/Users/macintosh/Desktop/helixone/helixone-backend')

from app.services.fred_collector import FREDCollector
from datetime import datetime, timedelta

def test_fred():
    """Tester FRED avec votre clé API"""

    print("=" * 60)
    print("🧪 TEST FRED (Federal Reserve Economic Data)")
    print("=" * 60)

    # Récupérer la clé depuis l'environnement
    api_key = os.getenv("FRED_API_KEY")

    if not api_key or api_key == "your_fred_api_key_here":
        print("❌ ERREUR: Clé API FRED non configurée dans .env")
        return False

    print(f"\n✅ Clé API FRED trouvée: {api_key[:8]}...")

    # Initialiser le collecteur
    fred = FREDCollector(api_key=api_key)

    # Test 1: Fed Funds Rate (taux directeur)
    print("\n" + "-" * 60)
    print("📊 Test 1: Fed Funds Rate (taux directeur)")
    print("-" * 60)

    try:
        # Récupérer les 30 derniers jours
        start_date = datetime.now() - timedelta(days=30)
        fed_funds = fred.get_series('DFF', start_date=start_date)

        latest_rate = fed_funds.iloc[-1]
        latest_date = fed_funds.index[-1]

        print(f"✅ Données récupérées:")
        print(f"   Taux actuel: {latest_rate:.2f}%")
        print(f"   Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"   Observations: {len(fed_funds)}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    # Test 2: Inflation (CPI)
    print("\n" + "-" * 60)
    print("📈 Test 2: Inflation (CPI)")
    print("-" * 60)

    try:
        # Récupérer les 12 derniers mois
        start_date = datetime.now() - timedelta(days=365)
        cpi = fred.get_series('CPIAUCSL', start_date=start_date)

        latest_cpi = cpi.iloc[-1]
        latest_date = cpi.index[-1]

        print(f"✅ Données récupérées:")
        print(f"   CPI actuel: {latest_cpi:.2f}")
        print(f"   Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"   Observations: {len(cpi)}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    # Test 3: Unemployment Rate
    print("\n" + "-" * 60)
    print("👥 Test 3: Taux de chômage")
    print("-" * 60)

    try:
        # Récupérer les 12 derniers mois
        start_date = datetime.now() - timedelta(days=365)
        unemployment = fred.get_series('UNRATE', start_date=start_date)

        latest_rate = unemployment.iloc[-1]
        latest_date = unemployment.index[-1]

        print(f"✅ Données récupérées:")
        print(f"   Taux de chômage: {latest_rate:.1f}%")
        print(f"   Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"   Observations: {len(unemployment)}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    # Test 4: Yield Curve (courbe des taux)
    print("\n" + "-" * 60)
    print("📉 Test 4: Yield Curve (courbe des taux)")
    print("-" * 60)

    try:
        yield_curve = fred.get_yield_curve()

        print(f"✅ Courbe des taux récupérée:")
        for maturity, rate in sorted(yield_curve.items()):
            print(f"   {maturity:>4}: {rate:.2f}%")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    # Test 5: Yield Spread 10Y-2Y (indicateur de récession)
    print("\n" + "-" * 60)
    print("⚠️  Test 5: Yield Spread 10Y-2Y (indicateur récession)")
    print("-" * 60)

    try:
        start_date = datetime.now() - timedelta(days=365)
        spread = fred.calculate_yield_spread(start_date=start_date)

        current_spread = spread.iloc[-1]
        current_date = spread.index[-1]

        print(f"✅ Spread calculé:")
        print(f"   Spread actuel: {current_spread:.2f}%")
        print(f"   Date: {current_date.strftime('%Y-%m-%d')}")

        if current_spread < 0:
            print(f"   ⚠️  ALERTE: Courbe inversée (spread négatif)")
            print(f"      Risque de récession élevé!")
        else:
            print(f"   ✅ Courbe normale (pas d'inversion)")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    # Test 6: Métadonnées d'une série
    print("\n" + "-" * 60)
    print("ℹ️  Test 6: Métadonnées (Fed Funds Rate)")
    print("-" * 60)

    try:
        metadata = fred.get_series_info('DFF')

        print(f"✅ Métadonnées récupérées:")
        print(f"   ID: {metadata['id']}")
        print(f"   Titre: {metadata['title']}")
        print(f"   Unités: {metadata['units']}")
        print(f"   Fréquence: {metadata['frequency']}")
        print(f"   Dernière MAJ: {metadata['last_updated']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS FRED RÉUSSIS!")
    print("=" * 60)
    print("\n💡 FRED fonctionne parfaitement!")
    print("   Vous avez maintenant accès à:")
    print("   - 500,000+ séries économiques")
    print("   - Taux d'intérêt (Fed, Treasury)")
    print("   - Inflation (CPI, PCE, PPI)")
    print("   - PIB et croissance")
    print("   - Emploi et chômage")
    print("   - Yield curves et spreads")
    print("   - Et bien plus encore!")
    print("\n   🚀 ILLIMITÉ et GRATUIT!")

    return True


if __name__ == "__main__":
    success = test_fred()
    sys.exit(0 if success else 1)
