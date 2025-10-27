"""
Test script pour BIS (Bank for International Settlements) Data Collector
"""

import sys
import logging
from app.services.bis_collector import get_bis_collector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_bis_data():
    """Tester les fonctionnalités BIS"""

    print("\n" + "="*70)
    print("🏦 TEST BIS (BANK FOR INTERNATIONAL SETTLEMENTS) DATA COLLECTOR")
    print("="*70 + "\n")

    bis = get_bis_collector()

    # Test 1: Crédit/PIB USA
    print("\n📊 Test 1: Ratio Crédit/PIB USA")
    print("-" * 70)
    try:
        data = bis.get_credit_to_gdp('US', start_period='2020-Q1', end_period='2024-Q4')
        print(f"✅ Crédit/PIB USA récupéré")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test crédit/PIB")

    # Test 2: Crédit total France
    print("\n💰 Test 2: Crédit total France")
    print("-" * 70)
    try:
        data = bis.get_total_credit('FR', start_period='2020-Q1', end_period='2024-Q4')
        print(f"✅ Crédit total France récupéré")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test crédit total")

    # Test 3: Titres de dette UK
    print("\n📜 Test 3: Titres de dette UK")
    print("-" * 70)
    try:
        data = bis.get_debt_securities('GB', start_period='2020-Q1', end_period='2024-Q4')
        print(f"✅ Titres de dette UK récupérés")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test titres dette")

    # Test 4: Taux de change effectif Allemagne
    print("\n💱 Test 4: Taux de change effectif réel Allemagne")
    print("-" * 70)
    try:
        data = bis.get_effective_exchange_rate('DE', rate_type='R', start_period='2020-01', end_period='2024-12')
        print(f"✅ Taux change effectif Allemagne récupéré")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test taux change")

    # Test 5: Prix immobilier Canada
    print("\n🏠 Test 5: Prix immobilier Canada")
    print("-" * 70)
    try:
        data = bis.get_property_prices('CA', start_period='2020-Q1', end_period='2024-Q4')
        print(f"✅ Prix immobilier Canada récupérés")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test prix immobilier")

    # Test 6: Dérivés OTC globaux
    print("\n📊 Test 6: Dérivés OTC globaux")
    print("-" * 70)
    try:
        data = bis.get_otc_derivatives(start_period='2020-12', end_period='2024-12')
        print(f"✅ Dérivés OTC récupérés")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test dérivés OTC")

    # Test 7: Taux directeurs Japon
    print("\n💰 Test 7: Taux directeurs banque centrale Japon")
    print("-" * 70)
    try:
        data = bis.get_policy_rates('JP', start_period='2020-01', end_period='2024-12')
        print(f"✅ Taux directeurs Japon récupérés")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test taux directeurs")

    # Test 8: Liquidité globale
    print("\n💧 Test 8: Liquidité globale")
    print("-" * 70)
    try:
        data = bis.get_global_liquidity(start_period='2020-Q1', end_period='2024-Q4')
        print(f"✅ Liquidité globale récupérée")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test liquidité")

    # Test 9: Dashboard financier Italie
    print("\n📊 Test 9: Dashboard financier Italie")
    print("-" * 70)
    try:
        dashboard = bis.get_financial_dashboard('IT', start_period='2020-Q1', end_period='2024-Q4')
        print(f"✅ Dashboard Italie assemblé")
        print(f"   Structure: {type(dashboard)}")
        if isinstance(dashboard, dict):
            indicators = []
            if dashboard.get('credit_to_gdp'):
                indicators.append('Crédit/PIB')
            if dashboard.get('total_credit'):
                indicators.append('Crédit total')
            if dashboard.get('debt_securities'):
                indicators.append('Dette')
            if dashboard.get('property_prices'):
                indicators.append('Immobilier')
            if dashboard.get('policy_rates'):
                indicators.append('Taux directeurs')
            if dashboard.get('exchange_rate'):
                indicators.append('Change')
            print(f"   Indicateurs présents: {', '.join(indicators)}")
            print(f"   Total indicateurs: {len(indicators)}/6")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test dashboard")

    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST BIS DATA")
    print("="*70)
    print("✅ Source: BIS (Bank for International Settlements)")
    print("✅ Limite: ILLIMITÉ")
    print("✅ Clé API: Pas requise")
    print("✅ Données disponibles:")
    print("   - 📊 Crédit/PIB")
    print("   - 💰 Crédit total")
    print("   - 📜 Titres de dette")
    print("   - 💱 Taux de change effectifs")
    print("   - 🏠 Prix immobilier")
    print("   - 📊 Dérivés OTC")
    print("   - 💰 Taux directeurs banques centrales")
    print("   - 💧 Liquidité globale")
    print("   - 🏦 Statistiques bancaires")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_bis_data()
    except Exception as e:
        logger.exception("Erreur globale test BIS")
        sys.exit(1)
