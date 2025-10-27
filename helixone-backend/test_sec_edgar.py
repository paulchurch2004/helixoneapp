"""
Script de test pour SEC Edgar API
Test des filings (10-K, 10-Q, 8-K) - GRATUIT
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.sec_edgar_collector import get_sec_edgar_collector

def test_sec_edgar():
    """Tester SEC Edgar API"""

    print("\n" + "="*70)
    print("TEST SEC EDGAR API - FILINGS SEC")
    print("GRATUIT - ILLIMITÉ - Pas de clé API requise")
    print("="*70 + "\n")

    sec = get_sec_edgar_collector()
    test_ticker = "AAPL"

    # Test 1: Trouver le CIK
    print(f"🔍 Test 1: Recherche CIK pour {test_ticker}")
    print("-" * 70)
    try:
        cik = sec.get_cik_by_ticker(test_ticker)

        if cik:
            print(f"✅ CIK trouvé: {cik}")
            print(f"   Format: 10 chiffres avec zéros devant")
        else:
            print(f"❌ CIK non trouvé pour {test_ticker}")
            return

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    # Test 2: Récupérer les filings 10-K (rapports annuels)
    print(f"\n\n📄 Test 2: Filings 10-K (rapports annuels)")
    print("-" * 70)
    try:
        filings_10k = sec.get_10k_filings(cik, limit=3)

        if filings_10k:
            print(f"✅ {len(filings_10k)} filings 10-K trouvés:\n")

            for i, filing in enumerate(filings_10k, 1):
                print(f"   {i}. Filing Date: {filing.get('filingDate')}")
                print(f"      Report Date: {filing.get('reportDate')}")
                print(f"      Accession #: {filing.get('accessionNumber')}")
                print(f"      Document: {filing.get('primaryDocument')}")

                # Construire URL
                url = sec.get_filing_url(cik, filing['accessionNumber'], filing['primaryDocument'])
                print(f"      URL: {url[:80]}...")
                print()
        else:
            print("⚠️  Aucun 10-K trouvé")

    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 3: Récupérer les filings 10-Q (rapports trimestriels)
    print(f"\n📄 Test 3: Filings 10-Q (rapports trimestriels)")
    print("-" * 70)
    try:
        filings_10q = sec.get_10q_filings(cik, limit=4)

        if filings_10q:
            print(f"✅ {len(filings_10q)} filings 10-Q trouvés")

            for i, filing in enumerate(filings_10q[:2], 1):
                print(f"\n   {i}. Filing Date: {filing.get('filingDate')}")
                print(f"      Report Date: {filing.get('reportDate')}")
        else:
            print("⚠️  Aucun 10-Q trouvé")

    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 4: Récupérer les 8-K (événements majeurs)
    print(f"\n\n📰 Test 4: Filings 8-K (événements majeurs)")
    print("-" * 70)
    try:
        filings_8k = sec.get_8k_filings(cik, limit=5)

        if filings_8k:
            print(f"✅ {len(filings_8k)} filings 8-K trouvés")

            for i, filing in enumerate(filings_8k[:3], 1):
                print(f"\n   {i}. Filing Date: {filing.get('filingDate')}")
                print(f"      Description: {filing.get('primaryDocDescription', 'N/A')}")
        else:
            print("⚠️  Aucun 8-K trouvé")

    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 5: Company Facts XBRL
    print(f"\n\n📊 Test 5: Company Facts (XBRL)")
    print("-" * 70)
    try:
        facts = sec.get_company_facts(cik)

        if facts and 'facts' in facts:
            print(f"✅ Facts XBRL récupérés:")
            print(f"   Entreprise: {facts.get('entityName', 'N/A')}")
            print(f"   CIK: {facts.get('cik', 'N/A')}")

            # Compter les concepts US-GAAP
            if 'us-gaap' in facts['facts']:
                concepts = facts['facts']['us-gaap']
                print(f"   Concepts US-GAAP: {len(concepts)}")

                # Afficher quelques concepts
                print(f"\n   Exemples de concepts:")
                for i, concept in enumerate(list(concepts.keys())[:5]):
                    print(f"   - {concept}")
        else:
            print("⚠️  Facts non disponibles")

    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 6: Revenue History via XBRL
    print(f"\n\n💰 Test 6: Historique des Revenus (XBRL)")
    print("-" * 70)
    try:
        revenues = sec.get_revenue_history(cik)

        if revenues:
            print(f"✅ {len(revenues)} périodes de revenus trouvées")

            # Trier par date et afficher les 5 plus récentes
            revenues_sorted = sorted(revenues, key=lambda x: x.get('end', ''), reverse=True)

            print(f"\n   5 périodes les plus récentes:")
            for i, rev in enumerate(revenues_sorted[:5], 1):
                end_date = rev.get('end', 'N/A')
                value = rev.get('val', 0)
                fy = rev.get('fy', 'N/A')
                fp = rev.get('fp', 'N/A')
                form = rev.get('form', 'N/A')

                if value:
                    print(f"   {i}. {end_date} (FY{fy} {fp}): ${value:,.0f} ({form})")
        else:
            print("⚠️  Revenus non trouvés")

    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Résumé
    print("\n\n" + "="*70)
    print("RÉSUMÉ DES TESTS SEC EDGAR")
    print("="*70)
    print("✅ Tous les tests ont été exécutés")
    print("📄 SEC Edgar API est maintenant intégré dans HelixOne!")
    print("\nCaractéristiques:")
    print("  - ✅ GRATUIT et ILLIMITÉ")
    print("  - ✅ Pas de clé API requise")
    print("  - ✅ Tous les filings SEC disponibles")
    print("  - ✅ Données XBRL structurées")
    print("\nFilings disponibles:")
    print("  - 📄 10-K (rapports annuels)")
    print("  - 📄 10-Q (rapports trimestriels)")
    print("  - 📰 8-K (événements majeurs)")
    print("  - 👤 Form 4 (insider transactions)")
    print("  - 🏦 13F-HR (institutional holdings)")
    print("  - 📊 Company Facts XBRL (données structurées)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_sec_edgar()
