"""
Test script pour Eurostat Data Collector
"""

import sys
import logging
from app.services.eurostat_collector import get_eurostat_collector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_eurostat_data():
    """Tester les fonctionnalités Eurostat"""

    print("\n" + "="*70)
    print("🇪🇺 TEST EUROSTAT DATA COLLECTOR")
    print("="*70 + "\n")

    eurostat = get_eurostat_collector()

    # Test 1: PIB zone euro
    print("\n📊 Test 1: PIB zone euro (EU27)")
    print("-" * 70)
    try:
        data = eurostat.get_gdp('EU27_2020', start_period='2020', end_period='2024')
        print(f"✅ PIB EU27 récupéré")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test PIB")

    # Test 2: Croissance PIB France
    print("\n📈 Test 2: Croissance PIB France")
    print("-" * 70)
    try:
        data = eurostat.get_gdp_growth('FR', start_period='2020', end_period='2024')
        print(f"✅ Croissance PIB France récupérée")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test croissance PIB")

    # Test 3: Inflation HICP Allemagne
    print("\n📈 Test 3: Inflation HICP Allemagne")
    print("-" * 70)
    try:
        data = eurostat.get_inflation_hicp('DE', start_period='2020-01', end_period='2024-12')
        print(f"✅ Inflation HICP Allemagne récupérée")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test inflation")

    # Test 4: Taux inflation annuel Italie
    print("\n📈 Test 4: Taux inflation annuel Italie")
    print("-" * 70)
    try:
        data = eurostat.get_inflation_annual_rate('IT', start_period='2020-01', end_period='2024-12')
        print(f"✅ Taux inflation annuel Italie récupéré")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test taux inflation")

    # Test 5: Taux chômage Espagne
    print("\n💼 Test 5: Taux chômage Espagne")
    print("-" * 70)
    try:
        data = eurostat.get_unemployment_rate('ES', start_period='2020-01', end_period='2024-12')
        print(f"✅ Taux chômage Espagne récupéré")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test chômage")

    # Test 6: Production industrielle Pologne
    print("\n🏭 Test 6: Production industrielle Pologne")
    print("-" * 70)
    try:
        data = eurostat.get_industrial_production('PL', start_period='2020-01', end_period='2024-12')
        print(f"✅ Production industrielle Pologne récupérée")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test production industrielle")

    # Test 7: Confiance entreprises Pays-Bas
    print("\n📊 Test 7: Confiance entreprises Pays-Bas")
    print("-" * 70)
    try:
        data = eurostat.get_business_confidence('NL', start_period='2020-01', end_period='2024-12')
        print(f"✅ Confiance entreprises Pays-Bas récupérée")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test confiance entreprises")

    # Test 8: Confiance consommateurs Belgique
    print("\n📊 Test 8: Confiance consommateurs Belgique")
    print("-" * 70)
    try:
        data = eurostat.get_consumer_confidence('BE', start_period='2020-01', end_period='2024-12')
        print(f"✅ Confiance consommateurs Belgique récupérée")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test confiance consommateurs")

    # Test 9: Population Suède
    print("\n👥 Test 9: Population Suède")
    print("-" * 70)
    try:
        data = eurostat.get_population('SE', start_period='2020', end_period='2024')
        print(f"✅ Population Suède récupérée")
        print(f"   Structure: {type(data)}")
        print(f"   Taille réponse: {len(str(data))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test population")

    # Test 10: Dashboard économique Portugal
    print("\n📊 Test 10: Dashboard économique Portugal")
    print("-" * 70)
    try:
        dashboard = eurostat.get_economic_dashboard('PT', start_period='2020', end_period='2024')
        print(f"✅ Dashboard Portugal assemblé")
        print(f"   Structure: {type(dashboard)}")
        if isinstance(dashboard, dict):
            indicators = []
            if dashboard.get('gdp'):
                indicators.append('PIB')
            if dashboard.get('gdp_growth'):
                indicators.append('Croissance')
            if dashboard.get('inflation'):
                indicators.append('Inflation')
            if dashboard.get('unemployment'):
                indicators.append('Chômage')
            if dashboard.get('industrial_production'):
                indicators.append('Production')
            if dashboard.get('consumer_confidence'):
                indicators.append('Conf. Conso')
            if dashboard.get('business_confidence'):
                indicators.append('Conf. Entrep')
            if dashboard.get('population'):
                indicators.append('Population')
            print(f"   Indicateurs présents: {', '.join(indicators)}")
            print(f"   Total indicateurs: {len(indicators)}/8")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test dashboard")

    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST EUROSTAT DATA")
    print("="*70)
    print("✅ Source: Eurostat (European Statistics)")
    print("✅ Limite: ILLIMITÉ")
    print("✅ Clé API: Pas requise")
    print("✅ Données disponibles:")
    print("   - 📊 PIB (GDP)")
    print("   - 📈 Croissance PIB")
    print("   - 📈 Inflation HICP")
    print("   - 📈 Taux inflation annuel")
    print("   - 💼 Taux de chômage")
    print("   - 🏭 Production industrielle")
    print("   - 📊 Confiance des entreprises")
    print("   - 📊 Confiance des consommateurs")
    print("   - 📦 Balance commerciale")
    print("   - 👥 Population")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_eurostat_data()
    except Exception as e:
        logger.exception("Erreur globale test Eurostat")
        sys.exit(1)
