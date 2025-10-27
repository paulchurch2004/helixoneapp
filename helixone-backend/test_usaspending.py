"""
Script de test pour USAspending.gov API
Test des contrats fédéraux US - GRATUIT et ILLIMITÉ
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.usaspending_source import get_usaspending_collector


def test_usaspending():
    """Tester USAspending.gov API"""

    print("\n" + "="*70)
    print("🏛️  TEST USASPENDING.GOV API - FEDERAL SPENDING DATA")
    print("GRATUIT - ILLIMITÉ - Pas de clé API requise - US Treasury")
    print("="*70 + "\n")

    usa = get_usaspending_collector()

    # Test 1: Search by Recipient (Lockheed Martin)
    print("🏢 Test 1: Recherche Contrats - Lockheed Martin")
    print("-" * 70)
    try:
        contracts = usa.search_spending_by_recipient(
            "Lockheed Martin",
            fiscal_year=2024,
            limit=5
        )

        if contracts:
            print(f"\n✅ {len(contracts)} contrats trouvés (FY2024)\n")

            for i, contract in enumerate(contracts, 1):
                recipient = contract.get('Recipient Name', 'Unknown')
                amount = contract.get('Award Amount', 0)
                description = contract.get('Description', 'No description')[:50]

                print(f"{i}. {recipient}")
                print(f"   Montant: ${amount:,.0f}")
                print(f"   Description: {description}...")
                print()

        else:
            print("⚠️  Aucun contrat trouvé\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 2: Search by Recipient (Boeing)
    print("✈️  Test 2: Recherche Contrats - Boeing")
    print("-" * 70)
    try:
        contracts = usa.search_spending_by_recipient(
            "Boeing",
            fiscal_year=2024,
            limit=5
        )

        if contracts:
            print(f"\n✅ {len(contracts)} contrats Boeing trouvés\n")

            total = sum(c.get('Award Amount', 0) for c in contracts)
            print(f"   Total (top 5): ${total:,.0f}\n")

            for i, contract in enumerate(contracts, 1):
                amount = contract.get('Award Amount', 0)
                agency = contract.get('Awarding Agency', 'Unknown')

                print(f"   {i}. ${amount:,.0f} - {agency}")

            print()

        else:
            print("⚠️  Aucun contrat Boeing trouvé\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 3: Company Contract Summary (SpaceX)
    print("🚀 Test 3: Résumé Contrats - SpaceX (3 dernières années)")
    print("-" * 70)
    try:
        summary = usa.get_company_contract_summary("SpaceX", years=3)

        print(f"\n📊 Résumé contrats fédéraux SpaceX:\n")
        print(f"   Entreprise:         {summary['company_name']}")
        print(f"   Montant total:      ${summary['total_amount']:,.0f}")
        print(f"   Nombre de contrats: {summary['contract_count']}")
        print(f"   Moyenne annuelle:   ${summary['average_annual_amount']:,.0f}")

        print(f"\n   Détail par année:")

        for year_data in summary['yearly_breakdown']:
            fy = year_data['fiscal_year']
            amount = year_data['amount']
            count = year_data['count']

            print(f"   - FY{fy}: ${amount:,.0f} ({count} contrats)")

        print()

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 4: Top Contractors (Top 10)
    print("🏆 Test 4: Top 10 Contractants Fédéraux (FY2024)")
    print("-" * 70)
    try:
        top = usa.get_top_contractors(fiscal_year=2024, limit=10)

        if top:
            print(f"\n✅ Top {len(top)} contractants fédéraux:\n")

            print(f"{'Rang':<5} {'Entreprise':<40} {'Montant Total':<20}")
            print("-" * 70)

            for i, contractor in enumerate(top, 1):
                name = contractor['name'][:39]
                amount = contractor['amount']

                print(f"{i:<5} {name:<40} ${amount:>18,.0f}")

            print()

        else:
            print("⚠️  Aucune donnée top contractors\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 5: Contracts by NAICS (Aircraft Manufacturing - 336411)
    print("✈️  Test 5: Contrats par Industrie - Fabrication Avions (NAICS 336411)")
    print("-" * 70)
    try:
        contracts = usa.search_contracts_by_naics(
            "336411",
            fiscal_year=2024,
            limit=5
        )

        if contracts:
            print(f"\n✅ {len(contracts)} contrats trouvés\n")

            for i, contract in enumerate(contracts, 1):
                recipient = contract.get('Recipient Name', 'Unknown')
                amount = contract.get('Award Amount', 0)

                print(f"   {i}. {recipient}: ${amount:,.0f}")

            print()

        else:
            print("⚠️  Aucun contrat dans cette industrie\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Test 6: Agency Spending (DOD - 097)
    print("🪖 Test 6: Dépenses Agence - Department of Defense")
    print("-" * 70)
    try:
        dod = usa.get_agency_spending("097")

        if dod:
            print(f"\n✅ Données DOD récupérées\n")

            name = dod.get('agency_name', 'Department of Defense')
            code = dod.get('agency_code', '097')

            print(f"   Agence: {name}")
            print(f"   Code:   {code}")

            # Show some basic info if available
            if 'total_budgetary_resources' in dod:
                print(f"   Budget: ${dod['total_budgetary_resources']:,.0f}")

            print()

        else:
            print("⚠️  Aucune donnée DOD\n")

    except Exception as e:
        print(f"❌ Erreur: {e}\n")

    # Summary
    print("\n" + "="*70)
    print("📊 RÉSUMÉ TEST USASPENDING.GOV")
    print("="*70)
    print("✅ Tests exécutés")
    print("🏛️  USAspending.gov API intégré dans HelixOne!")
    print("\nCaractéristiques:")
    print("  - ✅ GRATUIT et ILLIMITÉ")
    print("  - ✅ Pas de clé API requise")
    print("  - ✅ Données US Department of Treasury (officiel)")
    print("  - ✅ Contrats, subventions, prêts fédéraux")
    print("  - ✅ Historique complet disponible")
    print("\nDonnées disponibles:")
    print("  - 🏢 Contrats par entreprise/bénéficiaire")
    print("  - 🏦 Dépenses par agence fédérale")
    print("  - 🏭 Contrats par industrie (NAICS)")
    print("  - 🏆 Top contractants fédéraux")
    print("  - 📊 Résumés et tendances multi-années")
    print("  - 💰 Montants et descriptions détaillées")
    print("\nUtilisation:")
    print("  - Analyse exposition secteur gouvernemental")
    print("  - Évaluation dépendance revenus fédéraux")
    print("  - Screening entreprises défense/aérospatial")
    print("  - Recherche opportunités contractuelles")
    print("  - Due diligence entreprises publiques")
    print("  - Analyse concurrentielle secteur public")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_usaspending()
