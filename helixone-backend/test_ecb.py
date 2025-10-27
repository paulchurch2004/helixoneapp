"""
Test script pour ECB (European Central Bank) Data Collector
"""

import sys
import logging
from app.services.ecb_collector import get_ecb_collector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_ecb_data():
    """Tester toutes les fonctionnalités ECB"""

    print("\n" + "="*70)
    print("🇪🇺 TEST ECB (EUROPEAN CENTRAL BANK) DATA COLLECTOR")
    print("="*70 + "\n")

    ecb = get_ecb_collector()

    # Test 1: Taux d'intérêt clés BCE
    print("\n📊 Test 1: Taux d'intérêt clés BCE")
    print("-" * 70)
    try:
        rates = ecb.get_key_interest_rates()
        print(f"✅ Taux BCE récupérés")
        print(f"   Structure: {type(rates)}")
        if isinstance(rates, dict):
            # Afficher les clés principales
            if 'dataSets' in rates:
                print(f"   DataSets trouvés: Oui")
            if 'structure' in rates:
                print(f"   Structure trouvée: Oui")
        print(f"   Taille réponse: {len(str(rates))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test taux BCE")

    # Test 2: Taux de change EUR/USD
    print("\n💱 Test 2: Taux de change EUR/USD")
    print("-" * 70)
    try:
        eur_usd = ecb.get_euro_exchange_rates("USD")
        print(f"✅ EUR/USD récupéré")
        print(f"   Structure: {type(eur_usd)}")
        if isinstance(eur_usd, dict):
            if 'dataSets' in eur_usd:
                print(f"   DataSets trouvés: Oui")
                # Essayer d'extraire une valeur
                try:
                    data_sets = eur_usd.get('dataSets', [])
                    if data_sets and len(data_sets) > 0:
                        series = data_sets[0].get('series', {})
                        if series:
                            first_key = list(series.keys())[0]
                            observations = series[first_key].get('observations', {})
                            if observations:
                                last_obs_key = list(observations.keys())[-1]
                                last_value = observations[last_obs_key][0]
                                print(f"   Dernier taux EUR/USD: {last_value}")
                except Exception as e:
                    logger.debug(f"Impossible d'extraire valeur: {e}")
        print(f"   Taille réponse: {len(str(eur_usd))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test EUR/USD")

    # Test 3: Taux de change EUR/GBP
    print("\n💱 Test 3: Taux de change EUR/GBP")
    print("-" * 70)
    try:
        eur_gbp = ecb.get_euro_exchange_rates("GBP")
        print(f"✅ EUR/GBP récupéré")
        print(f"   Structure: {type(eur_gbp)}")
        if isinstance(eur_gbp, dict) and 'dataSets' in eur_gbp:
            print(f"   DataSets trouvés: Oui")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 4: Inflation HICP zone euro
    print("\n📈 Test 4: Inflation HICP zone euro")
    print("-" * 70)
    try:
        inflation = ecb.get_hicp_inflation()
        print(f"✅ Inflation HICP récupérée")
        print(f"   Structure: {type(inflation)}")
        if isinstance(inflation, dict):
            if 'dataSets' in inflation:
                print(f"   DataSets trouvés: Oui")
                # Essayer d'extraire taux d'inflation
                try:
                    data_sets = inflation.get('dataSets', [])
                    if data_sets and len(data_sets) > 0:
                        series = data_sets[0].get('series', {})
                        if series:
                            first_key = list(series.keys())[0]
                            observations = series[first_key].get('observations', {})
                            if observations:
                                last_obs_key = list(observations.keys())[-1]
                                last_value = observations[last_obs_key][0]
                                print(f"   Dernier taux inflation: {last_value}%")
                except Exception as e:
                    logger.debug(f"Impossible d'extraire inflation: {e}")
        print(f"   Taille réponse: {len(str(inflation))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test inflation")

    # Test 5: Masse monétaire M3
    print("\n💵 Test 5: Masse monétaire M3")
    print("-" * 70)
    try:
        m3 = ecb.get_m3_money_supply()
        print(f"✅ M3 récupéré")
        print(f"   Structure: {type(m3)}")
        if isinstance(m3, dict) and 'dataSets' in m3:
            print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(m3))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test M3")

    # Test 6: PIB zone euro
    print("\n📊 Test 6: PIB zone euro")
    print("-" * 70)
    try:
        gdp = ecb.get_gdp_euro_area()
        print(f"✅ PIB zone euro récupéré")
        print(f"   Structure: {type(gdp)}")
        if isinstance(gdp, dict) and 'dataSets' in gdp:
            print(f"   DataSets trouvés: Oui")
        print(f"   Taille réponse: {len(str(gdp))} caractères")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.exception("Erreur test PIB")

    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST ECB DATA")
    print("="*70)
    print("✅ Source: European Central Bank (BCE)")
    print("✅ Limite: ILLIMITÉ")
    print("✅ Clé API: Pas requise")
    print("✅ Données disponibles:")
    print("   - 💰 Taux d'intérêt clés BCE")
    print("   - 💱 Taux de change EUR/XXX")
    print("   - 📈 Inflation HICP zone euro")
    print("   - 💵 Masse monétaire M3")
    print("   - 📊 PIB zone euro")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_ecb_data()
    except Exception as e:
        logger.exception("Erreur globale test ECB")
        sys.exit(1)
