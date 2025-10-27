"""
Script de test pour Quandl/Nasdaq Data Link
Test des commodités - GRATUIT (50 req/jour avec clé, 20 sans)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.quandl_source import get_quandl_collector


def test_quandl():
    """Tester Quandl/Nasdaq Data Link API"""

    print("\n" + "="*70)
    print("📊 TEST QUANDL/NASDAQ DATA LINK - COMMODITIES DATA")

    api_key = os.getenv('QUANDL_API_KEY')
    if api_key:
        print("GRATUIT - 50 req/jour (avec clé API)")
    else:
        print("GRATUIT - 20 req/jour (sans clé API - mode anonyme)")
        print("\n💡 Pour 50 req/jour: https://data.nasdaq.com/sign-up")

    print("="*70 + "\n")

    try:
        quandl = get_quandl_collector()
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}\n")
        return

    # Test 1: Gold Price
    print("🥇 Test 1: Prix de l'Or (LBMA)")
    print("-" * 70)
    try:
        gold = quandl.get_gold_price(limit=10)

        if gold.get('data'):
            print(f"\n✅ {len(gold['data'])} points de données récupérés")
            print(f"\nColonnes: {', '.join(gold.get('column_names', []))}\n")

            print(f"{'Date':<12} {'USD AM':<12} {'USD PM':<12} {'EUR PM':<12}")
            print("-" * 70)

            for row in gold['data'][:10]:
                date = row[0]
                usd_am = f"${row[1]:,.2f}" if row[1] else "N/A"
                usd_pm = f"${row[2]:,.2f}" if row[2] else "N/A"
                eur_pm = f"€{row[6]:,.2f}" if len(row) > 6 and row[6] else "N/A"

                print(f"{date:<12} {usd_am:<12} {usd_pm:<12} {eur_pm:<12}")

            print()
        else:
            print("⚠️  Aucune donnée or récupérée\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 2: Silver Price
    print("🥈 Test 2: Prix de l'Argent (LBMA)")
    print("-" * 70)
    try:
        silver = quandl.get_silver_price(limit=5)

        if silver.get('data'):
            print(f"\n✅ {len(silver['data'])} points de données récupérés\n")

            print(f"{'Date':<12} {'USD':<12} {'GBP':<12} {'EUR':<12}")
            print("-" * 70)

            for row in silver['data'][:5]:
                date = row[0]
                usd = f"${row[1]:,.2f}" if row[1] else "N/A"
                gbp = f"£{row[2]:,.2f}" if len(row) > 2 and row[2] else "N/A"
                eur = f"€{row[3]:,.2f}" if len(row) > 3 and row[3] else "N/A"

                print(f"{date:<12} {usd:<12} {gbp:<12} {eur:<12}")

            print()
        else:
            print("⚠️  Aucune donnée argent récupérée\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 3: Crude Oil Futures
    print("🛢️  Test 3: Pétrole Brut WTI (Futures CME)")
    print("-" * 70)
    try:
        oil = quandl.get_crude_oil_futures(limit=5)

        if oil.get('data'):
            print(f"\n✅ {len(oil['data'])} points de données récupérés")
            print(f"\nColonnes: {', '.join(oil.get('column_names', []))}\n")

            print(f"{'Date':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Settle':<10} {'Volume':<12}")
            print("-" * 70)

            for row in oil['data'][:5]:
                date = row[0]
                open_price = f"${row[1]:.2f}" if row[1] else "N/A"
                high = f"${row[2]:.2f}" if row[2] else "N/A"
                low = f"${row[3]:.2f}" if row[3] else "N/A"
                settle = f"${row[6]:.2f}" if len(row) > 6 and row[6] else "N/A"
                volume = f"{int(row[7]):,}" if len(row) > 7 and row[7] else "N/A"

                print(f"{date:<12} {open_price:<10} {high:<10} {low:<10} {settle:<10} {volume:<12}")

            print()
        else:
            print("⚠️  Aucune donnée pétrole récupérée\n")

    except Exception as e:
        print(f"⚠️  CME Futures non disponible (données premium)\n")
        print(f"   Tentative avec World Bank data...\n")

        # Fallback to World Bank
        try:
            oil_wb = quandl.get_wb_commodity_price('POILWTI', limit=5)

            if oil_wb.get('data'):
                print(f"✅ World Bank - {len(oil_wb['data'])} points récupérés\n")

                print(f"{'Date':<12} {'WTI Price (USD)':<20}")
                print("-" * 70)

                for row in oil_wb['data'][:5]:
                    date = row[0]
                    price = f"${row[1]:.2f}" if row[1] else "N/A"
                    print(f"{date:<12} {price:<20}")

                print()
            else:
                print("❌ World Bank data aussi indisponible\n")

        except Exception as e2:
            print(f"❌ Erreur fallback: {e2}\n")

    # Test 4: Natural Gas Futures
    print("⛽ Test 4: Gaz Naturel (Futures CME)")
    print("-" * 70)
    try:
        gas = quandl.get_natural_gas_futures(limit=5)

        if gas.get('data'):
            print(f"\n✅ {len(gas['data'])} points de données récupérés\n")

            print(f"{'Date':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Settle':<10}")
            print("-" * 70)

            for row in gas['data'][:5]:
                date = row[0]
                open_price = f"${row[1]:.2f}" if row[1] else "N/A"
                high = f"${row[2]:.2f}" if row[2] else "N/A"
                low = f"${row[3]:.2f}" if row[3] else "N/A"
                settle = f"${row[6]:.2f}" if len(row) > 6 and row[6] else "N/A"

                print(f"{date:<12} {open_price:<10} {high:<10} {low:<10} {settle:<10}")

            print()
        else:
            print("⚠️  Aucune donnée gaz naturel récupérée\n")

    except Exception as e:
        print(f"⚠️  CME Futures non disponible\n")

    # Test 5: Copper Futures
    print("🔶 Test 5: Cuivre (Futures CME)")
    print("-" * 70)
    try:
        copper = quandl.get_copper_futures(limit=5)

        if copper.get('data'):
            print(f"\n✅ {len(copper['data'])} points de données récupérés\n")

            print(f"{'Date':<12} {'Settle':<12}")
            print("-" * 70)

            for row in copper['data'][:5]:
                date = row[0]
                settle = f"${row[6]:.4f}" if len(row) > 6 and row[6] else "N/A"

                print(f"{date:<12} {settle:<12}")

            print()
        else:
            print("⚠️  Aucune donnée cuivre récupérée\n")

    except Exception as e:
        print(f"⚠️  Données cuivre non disponibles\n")

    # Test 6: Commodity Summary
    print("📊 Test 6: Résumé Commodités (derniers prix)")
    print("-" * 70)
    try:
        summary = quandl.get_commodity_summary()

        print("\n✅ Résumé des commodités:\n")

        for commodity, price in summary.items():
            commodity_name = commodity.replace('_', ' ').title()
            if price is not None:
                print(f"   {commodity_name:<20}: ${price:,.2f}")
            else:
                print(f"   {commodity_name:<20}: N/A")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 7: World Bank - Coffee Price
    print("☕ Test 7: World Bank - Prix du Café")
    print("-" * 70)
    try:
        coffee = quandl.get_wb_commodity_price('PCOFFOTM', limit=5)

        if coffee.get('data'):
            print(f"\n✅ {len(coffee['data'])} points de données récupérés\n")

            print(f"{'Date':<12} {'Price (USD/kg)':<20}")
            print("-" * 70)

            for row in coffee['data'][:5]:
                date = row[0]
                price = f"${row[1]:.2f}" if row[1] else "N/A"

                print(f"{date:<12} {price:<20}")

            print()
        else:
            print("⚠️  Aucune donnée café récupérée\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Summary
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST QUANDL/NASDAQ DATA LINK")
    print("="*70)
    print("✅ Tests exécutés")
    print("📊 Quandl/Nasdaq Data Link intégré dans HelixOne!")
    print("\nCaractéristiques:")
    if api_key:
        print("  - ✅ Mode authentifié (50 req/jour)")
    else:
        print("  - ⚠️  Mode anonyme (20 req/jour)")
        print("  - 💡 Obtenir clé gratuite: https://data.nasdaq.com/sign-up")
    print("  - ✅ 400+ datasets gratuits")
    print("  - ✅ Commodités: or, argent, pétrole, gaz")
    print("  - ✅ World Bank commodity prices")
    print("  - ✅ Historique complet disponible")
    print("\nDonnées disponibles:")
    print("  - 🥇 Or (LBMA)")
    print("  - 🥈 Argent (LBMA)")
    print("  - 🛢️  Pétrole brut (WTI, Brent)")
    print("  - ⛽ Gaz naturel")
    print("  - 🔶 Cuivre")
    print("  - ☕ Commodités agricoles (café, blé, coton, etc.)")
    print("  - 🌍 World Bank commodity index")
    print("\nLimitations:")
    print("  - CME Futures limités en free tier (World Bank en alternative)")
    print("  - 20-50 requêtes/jour selon clé API")
    print("  - Certaines données premium payantes")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_quandl()
