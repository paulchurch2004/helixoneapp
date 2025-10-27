"""
Test script pour OECD (Organisation for Economic Co-operation and Development) Data Collector
"""

import sys
import logging
from app.services.oecd_collector import get_oecd_collector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_oecd_data():
    """Tester les fonctionnalités OECD"""

    print("\n" + "="*70)
    print("🌍 TEST OECD (ORGANISATION FOR ECONOMIC CO-OPERATION) DATA COLLECTOR")
    print("="*70 + "\n")

    oecd = get_oecd_collector()

    # Test 1: PIB USA
    print("\n📊 Test 1: PIB USA")
    print("-" * 70)
    try:
        data = oecd.get_gdp('USA', start_time='2023', end_time='2024')
        print(f"✅ PIB USA récupéré")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict):
            if 'dataSets' in data:
                print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test PIB")

    # Test 2: Croissance PIB France
    print("\n📈 Test 2: Croissance PIB France")
    print("-" * 70)
    try:
        data = oecd.get_gdp_growth('FRA', start_time='2023', end_time='2024')
        print(f"✅ Croissance PIB France récupérée")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'dataSets' in data:
            print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test croissance PIB")

    # Test 3: Taux de chômage Allemagne
    print("\n💼 Test 3: Taux de chômage Allemagne")
    print("-" * 70)
    try:
        data = oecd.get_unemployment_rate('DEU', start_time='2023-01', end_time='2024-12')
        print(f"✅ Taux chômage Allemagne récupéré")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'dataSets' in data:
            print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test chômage")

    # Test 4: Inflation CPI UK
    print("\n📈 Test 4: Inflation CPI UK")
    print("-" * 70)
    try:
        data = oecd.get_inflation_cpi('GBR', start_time='2023-01', end_time='2024-12')
        print(f"✅ Inflation UK récupérée")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'dataSets' in data:
            print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test inflation")

    # Test 5: Taux d'intérêt Japon
    print("\n💰 Test 5: Taux d'intérêt Japon")
    print("-" * 70)
    try:
        data = oecd.get_interest_rates('JPN', start_time='2023-01', end_time='2024-12')
        print(f"✅ Taux intérêt Japon récupérés")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'dataSets' in data:
            print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test taux intérêt")

    # Test 6: Production industrielle Canada
    print("\n🏭 Test 6: Production industrielle Canada")
    print("-" * 70)
    try:
        data = oecd.get_industrial_production('CAN', start_time='2023-01', end_time='2024-12')
        print(f"✅ Production industrielle Canada récupérée")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'dataSets' in data:
            print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test production industrielle")

    # Test 7: CLI (Composite Leading Indicators) Australie
    print("\n📊 Test 7: CLI Australie")
    print("-" * 70)
    try:
        data = oecd.get_cli('AUS', start_time='2023-01', end_time='2024-12')
        print(f"✅ CLI Australie récupéré")
        print(f"   Structure: {type(data)}")
        if isinstance(data, dict) and 'dataSets' in data:
            print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test CLI")

    # Test 8: Dashboard économique pour Italie
    print("\n📊 Test 8: Dashboard économique Italie")
    print("-" * 70)
    try:
        dashboard = oecd.get_economic_dashboard('ITA', start_time='2023', end_time='2024')
        print(f"✅ Dashboard Italie assemblé")
        print(f"   Structure: {type(dashboard)}")
        if isinstance(dashboard, dict):
            indicators = []
            if dashboard.get('gdp'):
                indicators.append('PIB')
            if dashboard.get('gdp_growth'):
                indicators.append('Croissance')
            if dashboard.get('unemployment'):
                indicators.append('Chômage')
            if dashboard.get('inflation'):
                indicators.append('Inflation')
            if dashboard.get('interest_rates'):
                indicators.append('Taux')
            if dashboard.get('industrial_production'):
                indicators.append('Production')
            if dashboard.get('cli'):
                indicators.append('CLI')
            print(f"   Indicateurs présents: {', '.join(indicators)}")
            print(f"   Total indicateurs: {len(indicators)}/7")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test dashboard")

    # Test 9: Comparaison PIB USA vs Chine
    print("\n🌍 Test 9: Comparaison PIB USA vs CHN")
    print("-" * 70)
    try:
        comparison = oecd.get_country_comparison(['USA', 'CHN'], indicator='gdp', start_time='2023', end_time='2024')
        print(f"✅ Comparaison assemblée")
        print(f"   Structure: {type(comparison)}")
        if isinstance(comparison, dict):
            for country, data in comparison.items():
                status = "✅" if data else "❌"
                print(f"   {country}: {status}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test comparaison")

    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST OECD DATA")
    print("="*70)
    print("✅ Source: OECD (Organisation for Economic Co-operation)")
    print("✅ Limite: ILLIMITÉ")
    print("✅ Clé API: Pas requise")
    print("✅ Données disponibles:")
    print("   - 📊 PIB (GDP)")
    print("   - 📈 Croissance PIB")
    print("   - 💼 Taux de chômage")
    print("   - 📈 Inflation CPI")
    print("   - 💰 Taux d'intérêt")
    print("   - 🏭 Production industrielle")
    print("   - 📊 CLI (Composite Leading Indicators)")
    print("   - 📦 Balance commerciale")
    print("   - 💼 Taux d'emploi")
    print("   - 🌍 Comparaisons multi-pays")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_oecd_data()
    except Exception as e:
        logger.exception("Erreur globale test OECD")
        sys.exit(1)
