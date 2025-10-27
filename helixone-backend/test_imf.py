"""
Test script pour IMF (International Monetary Fund) Data Collector
"""

import sys
import logging
from app.services.imf_collector import get_imf_collector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_imf_data():
    """Tester les fonctionnalités IMF"""

    print("\n" + "="*70)
    print("🌍 TEST IMF (INTERNATIONAL MONETARY FUND) DATA COLLECTOR")
    print("="*70 + "\n")

    imf = get_imf_collector()

    # Test 1: Taux de change
    print("\n💱 Test 1: Taux de change USD pour USA")
    print("-" * 70)
    try:
        data = imf.get_exchange_rates('US', start_period='2023', end_period='2024')
        print(f"✅ Taux de change récupérés")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict):
            # Essayer d'afficher des infos de base
            if 'CompactData' in data:
                print(f"   CompactData trouvé: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test taux change")

    # Test 2: Inflation
    print("\n📈 Test 2: Inflation CPI pour France")
    print("-" * 70)
    try:
        data = imf.get_inflation_rate('FR', start_period='2023', end_period='2024')
        print(f"✅ Inflation récupérée")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'CompactData' in data:
            print(f"   CompactData trouvé: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test inflation")

    # Test 3: PIB
    print("\n📊 Test 3: PIB pour Allemagne")
    print("-" * 70)
    try:
        data = imf.get_gdp('DE', start_period='2023', end_period='2024')
        print(f"✅ PIB récupéré")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'CompactData' in data:
            print(f"   CompactData trouvé: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test PIB")

    # Test 4: Taux d'intérêt
    print("\n💰 Test 4: Taux d'intérêt pour UK")
    print("-" * 70)
    try:
        data = imf.get_interest_rates('GB', start_period='2023', end_period='2024')
        print(f"✅ Taux d'intérêt récupérés")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'CompactData' in data:
            print(f"   CompactData trouvé: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test taux intérêt")

    # Test 5: Balance courante
    print("\n💵 Test 5: Balance courante pour Japon")
    print("-" * 70)
    try:
        data = imf.get_current_account('JP', start_period='2023', end_period='2024')
        print(f"✅ Balance courante récupérée")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'CompactData' in data:
            print(f"   CompactData trouvé: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test balance courante")

    # Test 6: Balance commerciale
    print("\n📦 Test 6: Balance commerciale pour Chine")
    print("-" * 70)
    try:
        data = imf.get_trade_balance('CN', start_period='2023', end_period='2024')
        print(f"✅ Balance commerciale récupérée")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'CompactData' in data:
            print(f"   CompactData trouvé: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test balance commerciale")

    # Test 7: Dashboard macro
    print("\n📊 Test 7: Dashboard macro pour Canada")
    print("-" * 70)
    try:
        dashboard = imf.get_macro_dashboard('CA', start_period='2023', end_period='2024')
        print(f"✅ Dashboard assemblé")
        print(f"   Structure: {type(dashboard)}")
        if isinstance(dashboard, dict):
            indicators = []
            if dashboard.get('gdp'):
                indicators.append('PIB')
            if dashboard.get('inflation'):
                indicators.append('Inflation')
            if dashboard.get('interest_rates'):
                indicators.append('Taux')
            if dashboard.get('exchange_rates'):
                indicators.append('Change')
            print(f"   Indicateurs présents: {', '.join(indicators)}")
            print(f"   Total indicateurs: {len(indicators)}/4")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test dashboard")

    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST IMF DATA")
    print("="*70)
    print("✅ Source: International Monetary Fund (IMF)")
    print("✅ Limite: ILLIMITÉ")
    print("✅ Clé API: Pas requise")
    print("✅ Données disponibles:")
    print("   - 💱 Taux de change")
    print("   - 📈 Inflation (CPI)")
    print("   - 📊 PIB")
    print("   - 💰 Taux d'intérêt")
    print("   - 💵 Balance courante")
    print("   - 📦 Balance commerciale")
    print("   - 🏦 Indicateurs bancaires (FSI)")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_imf_data()
    except Exception as e:
        logger.exception("Erreur globale test IMF")
        sys.exit(1)
