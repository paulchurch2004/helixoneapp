"""
Script de test pour Carbon Intensity API
Test des données ESG carbone - GRATUIT et ILLIMITÉ
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.carbon_intensity_source import get_carbon_intensity_collector


def test_carbon_intensity():
    """Tester Carbon Intensity API"""

    print("\n" + "="*70)
    print("🌱 TEST CARBON INTENSITY API - ESG DATA")
    print("GRATUIT - ILLIMITÉ - Pas de clé API requise - UK National Grid")
    print("="*70 + "\n")

    carbon = get_carbon_intensity_collector()

    # Test 1: Current Intensity
    print("⚡ Test 1: Intensité Carbone Actuelle (UK)")
    print("-" * 70)
    try:
        current = carbon.get_current_intensity()

        from_time = current['from']
        intensity = current['intensity']

        actual = intensity.get('actual', 'N/A')
        forecast = intensity.get('forecast', 'N/A')
        index = intensity['index']

        # Emoji based on index
        if index == 'very low':
            emoji = "🟢"
        elif index == 'low':
            emoji = "🟡"
        elif index == 'moderate':
            emoji = "🟠"
        elif index == 'high':
            emoji = "🔴"
        else:
            emoji = "⚫"

        print(f"\n{emoji} Intensité Carbone Actuelle:")
        print(f"   Période: {from_time[:16]}")
        print(f"   Actual:   {actual} gCO2/kWh")
        print(f"   Forecast: {forecast} gCO2/kWh")
        print(f"   Index:    {index.upper()}")

        # Interpretation
        if index in ['very low', 'low']:
            print(f"\n   💡 Bon moment pour les tâches énergivores!")
        else:
            print(f"\n   ⚠️  Période de forte intensité carbone")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 2: Generation Mix
    print("🔋 Test 2: Mix de Génération Électrique")
    print("-" * 70)
    try:
        mix = carbon.get_generation_mix()

        print(f"\n⚡ Mix de génération actuel:\n")

        generation = mix.get('generationmix', [])

        # Sort by percentage descending
        generation_sorted = sorted(generation, key=lambda x: x['perc'], reverse=True)

        print(f"{'Source':<15} {'Pourcentage':<12} {'Barre':<30}")
        print("-" * 70)

        for source in generation_sorted:
            fuel = source['fuel'].capitalize()
            perc = source['perc']

            # Visual bar
            bar_length = int(perc / 3.3)  # Scale to fit in 30 chars (100/3.3 ≈ 30)
            bar = "█" * bar_length

            # Color coding
            if fuel in ['Wind', 'Solar', 'Hydro']:
                icon = "🌱"
            elif fuel in ['Gas', 'Coal', 'Oil']:
                icon = "⚫"
            elif fuel == 'Nuclear':
                icon = "☢️ "
            else:
                icon = "⚪"

            print(f"{icon} {fuel:<12} {perc:>5.1f}%      {bar}")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 3: Renewable Percentage
    print("🌱 Test 3: Pourcentage d'Énergies Renouvelables")
    print("-" * 70)
    try:
        renewable_pct = carbon.get_renewable_percentage()
        fossil_pct = carbon.get_fossil_fuel_percentage()

        print(f"\n📊 Analyse du mix énergétique:\n")
        print(f"   🌱 Renouvelables: {renewable_pct:.1f}%")
        print(f"   ⚫ Fossiles:      {fossil_pct:.1f}%")
        print(f"   ☢️  Autre (nucléaire, imports): {100 - renewable_pct - fossil_pct:.1f}%")

        # Visual comparison
        print(f"\n   🌱 {'█' * int(renewable_pct/2)}  {renewable_pct:.1f}%")
        print(f"   ⚫ {'█' * int(fossil_pct/2)}  {fossil_pct:.1f}%")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 4: Clean Energy Period Check
    print("✅ Test 4: Vérification Période Énergie Propre")
    print("-" * 70)
    try:
        clean_check = carbon.is_clean_energy_period(threshold=40.0)

        print(f"\n🔍 Analyse période actuelle:\n")
        print(f"   Énergie propre: {'✅ OUI' if clean_check['is_clean'] else '❌ NON'}")
        print(f"   Renouvelables:  {clean_check['renewable_pct']:.1f}%")
        print(f"   Intensité CO2:  {clean_check['carbon_intensity']} gCO2/kWh")
        print(f"   Index:          {clean_check['index'].upper()}")
        print(f"\n   💡 {clean_check['message']}")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 5: ESG Score
    print("📊 Test 5: Score ESG (Simplifié)")
    print("-" * 70)
    try:
        esg = carbon.get_esg_score()

        print(f"\n🎯 Score ESG Environnemental:\n")
        print(f"   Score global:    {esg['esg_score']}/100")
        print(f"   Grade:           {esg['grade']}")
        print(f"\n   Composantes:")
        print(f"   - Renouvelables: {esg['renewable_pct']:.1f}% → {esg['components']['renewable_score']}/50 pts")
        print(f"   - Intensité CO2: {esg['carbon_intensity']} gCO2/kWh → {esg['components']['intensity_score']}/50 pts")

        # Visual grade
        grade_bar = {
            'A': '🟢🟢🟢🟢🟢',
            'B': '🟢🟢🟢🟢⚪',
            'C': '🟡🟡🟡⚪⚪',
            'D': '🟠🟠⚪⚪⚪',
            'F': '🔴⚪⚪⚪⚪'
        }

        print(f"\n   Grade visuel: {grade_bar.get(esg['grade'], '⚪⚪⚪⚪⚪')}")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 6: Regional Intensity (sample)
    print("🗺️  Test 6: Intensité Carbone Régionale")
    print("-" * 70)
    try:
        regions = carbon.get_regional_intensity()

        if regions:
            print(f"\n✅ {len(regions)} régions UK disponibles\n")

            print(f"{'Région':<25} {'Intensité':<12} {'Index':<15}")
            print("-" * 70)

            # Show first 10 regions
            for region in regions[:10]:
                name = region.get('shortname', 'Unknown')
                data = region.get('data', [])

                if data and len(data) > 0:
                    intensity_data = data[0].get('intensity', {})
                    intensity = intensity_data.get('forecast', 'N/A')
                    index = intensity_data.get('index', 'N/A')

                    if isinstance(intensity, (int, float)):
                        intensity_str = f"{intensity} gCO2/kWh"
                    else:
                        intensity_str = str(intensity)

                    print(f"{name:<25} {intensity_str:<12} {index:<15}")

            print()

        else:
            print("⚠️  Aucune donnée régionale disponible\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 7: Intensity Factors
    print("📋 Test 7: Facteurs d'Intensité par Combustible")
    print("-" * 70)
    try:
        factors = carbon.get_intensity_factors()

        if factors:
            print(f"\n✅ Facteurs d'intensité carbone:\n")

            print(f"{'Combustible':<20} {'Intensité (gCO2/kWh)':<25}")
            print("-" * 70)

            for factor in factors:
                fuel = factor.get('fuel', 'Unknown')
                intensity = factor.get('intensity', 'N/A')

                print(f"{fuel:<20} {intensity:<25}")

            print()

        else:
            print("⚠️  Aucun facteur disponible\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Summary
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST CARBON INTENSITY API")
    print("="*70)
    print("✅ Tous les tests exécutés")
    print("🌱 Carbon Intensity API intégré dans HelixOne!")
    print("\nCaractéristiques:")
    print("  - ✅ GRATUIT et ILLIMITÉ")
    print("  - ✅ Pas de clé API requise")
    print("  - ✅ Données UK National Grid (officiel)")
    print("  - ✅ Temps réel + prévisions")
    print("  - ✅ Mix énergétique détaillé")
    print("\nDonnées disponibles:")
    print("  - ⚡ Intensité carbone actuelle (gCO2/kWh)")
    print("  - 🔋 Mix de génération (sources)")
    print("  - 🌱 Pourcentage énergies renouvelables")
    print("  - 🗺️  Données régionales (UK)")
    print("  - 📊 Score ESG environnemental")
    print("  - 💡 Recommandations utilisation énergie")
    print("\nUtilisation:")
    print("  - Scoring ESG entreprises énergétiques")
    print("  - Optimisation tâches énergivores")
    print("  - Analyse empreinte carbone")
    print("  - Reporting développement durable")
    print("  - Trading certificats carbone")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_carbon_intensity()
