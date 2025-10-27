"""
Script de test pour World Bank API
Test des données macroéconomiques globales (GRATUIT - ILLIMITÉ)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.worldbank_collector import get_worldbank_collector

def test_worldbank():
    """Tester World Bank API"""

    print("\n" + "="*70)
    print("TEST WORLD BANK API - DONNÉES MACRO GLOBALES")
    print("GRATUIT - ILLIMITÉ - Pas de clé API requise")
    print("="*70 + "\n")

    wb = get_worldbank_collector()

    # Test 1: PIB USA
    print("🌍 Test 1: PIB USA (dernières années)")
    print("-" * 70)
    try:
        gdp = wb.get_gdp("USA", start_year=2018)

        if gdp and len(gdp) > 0:
            print(f"✅ {len(gdp)} années de données PIB USA:")
            for i, data in enumerate(gdp[:5]):
                year = data.get('date', 'N/A')
                value = data.get('value')
                if value:
                    print(f"   {year}: ${value:,.0f}")
        else:
            print("⚠️  Données PIB non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 2: PIB par habitant
    print(f"\n\n💰 Test 2: PIB par habitant (USA, FRA, CHN)")
    print("-" * 70)
    for country in ["USA", "FRA", "CHN"]:
        try:
            gdp_pc = wb.get_gdp_per_capita(country, start_year=2020)

            if gdp_pc and len(gdp_pc) > 0:
                latest = gdp_pc[0]
                year = latest.get('date')
                value = latest.get('value')
                if value:
                    print(f"✅ {country} ({year}): ${value:,.0f}/habitant")
        except Exception as e:
            print(f"❌ {country}: {e}")

    # Test 3: Inflation
    print(f"\n\n📈 Test 3: Inflation CPI (dernières années)")
    print("-" * 70)
    try:
        inflation = wb.get_inflation("USA", start_year=2019)

        if inflation and len(inflation) > 0:
            print(f"✅ {len(inflation)} années d'inflation USA:")
            for data in inflation[:5]:
                year = data.get('date')
                value = data.get('value')
                if value is not None:
                    print(f"   {year}: {value:.2f}%")
        else:
            print("⚠️  Données inflation non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 4: Chômage
    print(f"\n\n👔 Test 4: Taux de chômage")
    print("-" * 70)
    try:
        unemployment = wb.get_unemployment("USA", start_year=2019)

        if unemployment and len(unemployment) > 0:
            print(f"✅ {len(unemployment)} années de chômage USA:")
            for data in unemployment[:5]:
                year = data.get('date')
                value = data.get('value')
                if value is not None:
                    print(f"   {year}: {value:.2f}%")
        else:
            print("⚠️  Données chômage non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 5: Population
    print(f"\n\n👥 Test 5: Population totale")
    print("-" * 70)
    try:
        population = wb.get_population("USA", start_year=2019)

        if population and len(population) > 0:
            print(f"✅ Population USA:")
            for data in population[:5]:
                year = data.get('date')
                value = data.get('value')
                if value:
                    print(f"   {year}: {value:,.0f} habitants ({value/1e6:.1f}M)")
        else:
            print("⚠️  Données population non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 6: Dashboard économique complet
    print(f"\n\n📊 Test 6: Dashboard économique complet USA")
    print("-" * 70)
    try:
        dashboard = wb.get_economic_dashboard("USA", start_year=2022)

        print(f"✅ Dashboard récupéré avec {len(dashboard)} indicateurs:")

        for name, data in dashboard.items():
            if data and len(data) > 0:
                latest = data[0]
                year = latest.get('date')
                value = latest.get('value')
                if value is not None:
                    print(f"   {name}: Disponible ({len(data)} années)")

    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 7: Comparaison internationale
    print(f"\n\n🌍 Test 7: Comparaison PIB international (2022)")
    print("-" * 70)
    countries = {
        "USA": "États-Unis",
        "CHN": "Chine",
        "JPN": "Japon",
        "DEU": "Allemagne",
        "GBR": "Royaume-Uni",
        "FRA": "France"
    }

    gdp_comparison = []

    for code, name in countries.items():
        try:
            gdp_data = wb.get_gdp(code, start_year=2022)
            if gdp_data and len(gdp_data) > 0:
                latest = gdp_data[0]
                value = latest.get('value')
                year = latest.get('date')
                if value:
                    gdp_comparison.append((name, value, year))
        except:
            pass

    # Trier par PIB décroissant
    gdp_comparison.sort(key=lambda x: x[1], reverse=True)

    print(f"✅ Classement PIB:")
    for i, (country, gdp, year) in enumerate(gdp_comparison, 1):
        print(f"   {i}. {country}: ${gdp:,.0f} ({year})")

    # Test 8: Liste des pays
    print(f"\n\n🗺️  Test 8: Liste des pays disponibles")
    print("-" * 70)
    try:
        countries = wb.get_countries()

        if countries:
            print(f"✅ {len(countries)} pays disponibles dans World Bank")

            # Afficher quelques exemples
            print("\n   Exemples:")
            for country in countries[:10]:
                name = country.get('name')
                code = country.get('id')
                region = country.get('region', {}).get('value', 'N/A')
                if name and code:
                    print(f"   - {name} ({code}): {region}")
        else:
            print("⚠️  Liste des pays non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Résumé
    print("\n\n" + "="*70)
    print("RÉSUMÉ DES TESTS WORLD BANK")
    print("="*70)
    print("✅ Tous les tests ont été exécutés")
    print("🌍 World Bank API est maintenant intégré dans HelixOne!")
    print("\nCaractéristiques:")
    print("  - ✅ GRATUIT et ILLIMITÉ")
    print("  - ✅ Pas de clé API requise")
    print("  - ✅ 200+ pays")
    print("  - ✅ Historique jusqu'à 60+ ans")
    print("  - ✅ 1,400+ indicateurs économiques")
    print("\nIndicateurs disponibles:")
    print("  - 📊 PIB (nominal, par habitant, croissance)")
    print("  - 📈 Inflation (CPI)")
    print("  - 👔 Chômage")
    print("  - 👥 Population")
    print("  - 💰 Dette publique")
    print("  - 🌍 Commerce international")
    print("  - 🏦 Indicateurs de développement")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_worldbank()
