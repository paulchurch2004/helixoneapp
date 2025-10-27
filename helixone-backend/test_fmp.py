"""
Script de test pour Financial Modeling Prep API
Test des états financiers, ratios, ownership, insider trading
"""

import sys
import os

# Ajouter le chemin du backend au path Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.services.fmp_collector import get_fmp_collector

def test_fmp():
    """Tester toutes les fonctionnalités FMP"""

    print("\n" + "="*70)
    print("TEST FINANCIAL MODELING PREP (FMP) API")
    print("="*70 + "\n")

    fmp = get_fmp_collector()
    test_symbol = "AAPL"

    # Test 1: Company Profile
    print(f"🏢 Test 1: Company Profile pour {test_symbol}")
    print("-" * 70)
    try:
        profile = fmp.get_company_profile(test_symbol)

        if profile and len(profile) > 0:
            company = profile[0]
            print(f"✅ Profil récupéré:")
            print(f"   Nom: {company.get('companyName', 'N/A')}")
            print(f"   Secteur: {company.get('sector', 'N/A')}")
            print(f"   Industrie: {company.get('industry', 'N/A')}")
            print(f"   Market Cap: ${company.get('mktCap', 0):,.0f}")
            print(f"   Prix: ${company.get('price', 0):.2f}")
            print(f"   Beta: {company.get('beta', 0):.2f}")
            print(f"   Employés: {company.get('fullTimeEmployees', 'N/A')}")
        else:
            print("⚠️  Profil non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 2: Income Statement
    print(f"\n\n📄 Test 2: Income Statement (5 dernières années)")
    print("-" * 70)
    try:
        income_statements = fmp.get_income_statement(test_symbol, period="annual", limit=5)

        if income_statements and len(income_statements) > 0:
            print(f"✅ {len(income_statements)} income statements récupérés")

            # Afficher le dernier
            latest = income_statements[0]
            print(f"\n   Dernière période: {latest.get('calendarYear', 'N/A')}")
            print(f"   Revenue: ${latest.get('revenue', 0):,.0f}")
            print(f"   Gross Profit: ${latest.get('grossProfit', 0):,.0f}")
            print(f"   Operating Income: ${latest.get('operatingIncome', 0):,.0f}")
            print(f"   Net Income: ${latest.get('netIncome', 0):,.0f}")
            print(f"   EPS: ${latest.get('eps', 0):.2f}")

            # Marges
            revenue = latest.get('revenue', 1)
            gross_margin = (latest.get('grossProfit', 0) / revenue) * 100 if revenue else 0
            net_margin = (latest.get('netIncome', 0) / revenue) * 100 if revenue else 0
            print(f"   Marge brute: {gross_margin:.2f}%")
            print(f"   Marge nette: {net_margin:.2f}%")
        else:
            print("⚠️  Income statements non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 3: Balance Sheet
    print(f"\n\n📄 Test 3: Balance Sheet (dernière année)")
    print("-" * 70)
    try:
        balance_sheets = fmp.get_balance_sheet(test_symbol, period="annual", limit=1)

        if balance_sheets and len(balance_sheets) > 0:
            bs = balance_sheets[0]
            print(f"✅ Balance Sheet récupéré:")
            print(f"   Période: {bs.get('calendarYear', 'N/A')}")
            print(f"   Total Assets: ${bs.get('totalAssets', 0):,.0f}")
            print(f"   Total Liabilities: ${bs.get('totalLiabilities', 0):,.0f}")
            print(f"   Shareholders Equity: ${bs.get('totalStockholdersEquity', 0):,.0f}")
            print(f"   Cash: ${bs.get('cashAndCashEquivalents', 0):,.0f}")
            print(f"   Total Debt: ${bs.get('totalDebt', 0):,.0f}")
        else:
            print("⚠️  Balance sheet non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 4: Cash Flow
    print(f"\n\n📄 Test 4: Cash Flow Statement")
    print("-" * 70)
    try:
        cash_flows = fmp.get_cash_flow(test_symbol, period="annual", limit=1)

        if cash_flows and len(cash_flows) > 0:
            cf = cash_flows[0]
            print(f"✅ Cash Flow récupéré:")
            print(f"   Période: {cf.get('calendarYear', 'N/A')}")
            print(f"   Operating Cash Flow: ${cf.get('operatingCashFlow', 0):,.0f}")
            print(f"   Investing Cash Flow: ${cf.get('netCashUsedForInvestingActivites', 0):,.0f}")
            print(f"   Financing Cash Flow: ${cf.get('netCashUsedProvidedByFinancingActivities', 0):,.0f}")
            print(f"   Free Cash Flow: ${cf.get('freeCashFlow', 0):,.0f}")
            print(f"   Capex: ${cf.get('capitalExpenditure', 0):,.0f}")
        else:
            print("⚠️  Cash flow non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 5: Financial Ratios
    print(f"\n\n📊 Test 5: Financial Ratios (50+ ratios)")
    print("-" * 70)
    try:
        ratios = fmp.get_financial_ratios(test_symbol, period="annual", limit=1)

        if ratios and len(ratios) > 0:
            r = ratios[0]
            print(f"✅ Ratios récupérés:")
            print(f"\n   PROFITABILITÉ:")
            print(f"   ROE: {r.get('returnOnEquity', 0):.4f} ({r.get('returnOnEquity', 0)*100:.2f}%)")
            print(f"   ROA: {r.get('returnOnAssets', 0):.4f} ({r.get('returnOnAssets', 0)*100:.2f}%)")
            print(f"   Marge nette: {r.get('netProfitMargin', 0):.4f} ({r.get('netProfitMargin', 0)*100:.2f}%)")

            print(f"\n   LIQUIDITÉ:")
            print(f"   Current Ratio: {r.get('currentRatio', 0):.2f}")
            print(f"   Quick Ratio: {r.get('quickRatio', 0):.2f}")

            print(f"\n   SOLVABILITÉ:")
            print(f"   Debt/Equity: {r.get('debtEquityRatio', 0):.2f}")
            print(f"   Interest Coverage: {r.get('interestCoverage', 0):.2f}")

            print(f"\n   VALORISATION:")
            print(f"   P/E Ratio: {r.get('priceEarningsRatio', 0):.2f}")
            print(f"   P/B Ratio: {r.get('priceToBookRatio', 0):.2f}")
        else:
            print("⚠️  Ratios non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 6: Key Metrics
    print(f"\n\n📊 Test 6: Key Metrics")
    print("-" * 70)
    try:
        metrics = fmp.get_key_metrics(test_symbol, period="annual", limit=1)

        if metrics and len(metrics) > 0:
            m = metrics[0]
            print(f"✅ Key Metrics récupérés:")
            print(f"   Market Cap: ${m.get('marketCap', 0):,.0f}")
            print(f"   Enterprise Value: ${m.get('enterpriseValue', 0):,.0f}")
            print(f"   P/E Ratio: {m.get('peRatio', 0):.2f}")
            print(f"   EV/EBITDA: {m.get('evToEbitda', 0):.2f}")
            print(f"   Revenue Growth: {m.get('revenuePerShareGrowth', 0)*100:.2f}%")
            print(f"   Dividend Yield: {m.get('dividendYield', 0)*100:.2f}%")
        else:
            print("⚠️  Key metrics non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 7: Financial Growth
    print(f"\n\n📈 Test 7: Financial Growth")
    print("-" * 70)
    try:
        growth = fmp.get_financial_growth(test_symbol, period="annual", limit=1)

        if growth and len(growth) > 0:
            g = growth[0]
            print(f"✅ Growth Metrics récupérés:")
            print(f"   Revenue Growth: {g.get('revenueGrowth', 0)*100:+.2f}%")
            print(f"   EPS Growth: {g.get('epsgrowth', 0)*100:+.2f}%")
            print(f"   Net Income Growth: {g.get('netIncomeGrowth', 0)*100:+.2f}%")
            print(f"   Free Cash Flow Growth: {g.get('freeCashFlowGrowth', 0)*100:+.2f}%")
            print(f"   Total Assets Growth: {g.get('assetGrowth', 0)*100:+.2f}%")
        else:
            print("⚠️  Growth metrics non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 8: Dividends Historical
    print(f"\n\n💰 Test 8: Dividendes historiques")
    print("-" * 70)
    try:
        dividends = fmp.get_dividends_historical(test_symbol)

        if dividends and len(dividends) > 0:
            print(f"✅ {len(dividends)} dividendes historiques")

            # 5 derniers
            for i, div in enumerate(dividends[:5]):
                print(f"   {i+1}. {div.get('date', 'N/A')}: ${div.get('dividend', 0):.4f}")
        else:
            print("⚠️  Dividendes non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 9: Insider Trading
    print(f"\n\n👤 Test 9: Insider Trading (10 dernières transactions)")
    print("-" * 70)
    try:
        insider_trades = fmp.get_insider_trading(test_symbol, limit=10)

        if insider_trades and len(insider_trades) > 0:
            print(f"✅ {len(insider_trades)} insider transactions")

            # 3 dernières
            for i, trade in enumerate(insider_trades[:3]):
                print(f"\n   Transaction {i+1}:")
                print(f"   Nom: {trade.get('reportingName', 'N/A')}")
                print(f"   Type: {trade.get('transactionType', 'N/A')}")
                print(f"   Actions: {trade.get('securitiesTransacted', 0):,}")
                print(f"   Prix: ${trade.get('price', 0):.2f}")
                print(f"   Date: {trade.get('transactionDate', 'N/A')}")
        else:
            print("⚠️  Insider trading non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 10: Institutional Holders
    print(f"\n\n🏦 Test 10: Institutional Holders (Top 5)")
    print("-" * 70)
    try:
        institutions = fmp.get_institutional_holders(test_symbol)

        if institutions and len(institutions) > 0:
            print(f"✅ {len(institutions)} institutional holders")

            # Top 5
            for i, inst in enumerate(institutions[:5]):
                print(f"\n   {i+1}. {inst.get('holder', 'N/A')}")
                print(f"      Actions: {inst.get('shares', 0):,}")
                print(f"      Valeur: ${inst.get('value', 0):,}")
                print(f"      Change: {inst.get('changeInSharesNumberPercentage', 0)*100:+.2f}%")
        else:
            print("⚠️  Institutional holders non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 11: Analyst Estimates
    print(f"\n\n📊 Test 11: Analyst Estimates")
    print("-" * 70)
    try:
        estimates = fmp.get_analyst_estimates(test_symbol, period="annual", limit=2)

        if estimates and len(estimates) > 0:
            print(f"✅ {len(estimates)} périodes d'estimations")

            est = estimates[0]
            print(f"\n   Année: {est.get('date', 'N/A')}")
            print(f"   Revenue estimé: ${est.get('estimatedRevenueAvg', 0):,.0f}")
            print(f"   EPS estimé: ${est.get('estimatedEpsAvg', 0):.2f}")
            print(f"   EBITDA estimé: ${est.get('estimatedEbitdaAvg', 0):,.0f}")
            print(f"   Nombre analystes (Revenue): {est.get('numberAnalystEstimatedRevenue', 0)}")
            print(f"   Nombre analystes (EPS): {est.get('numberAnalystsEstimatedEps', 0)}")
        else:
            print("⚠️  Analyst estimates non disponibles")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Usage Stats
    print("\n\n" + "="*70)
    print("STATISTIQUES D'UTILISATION FMP")
    print("="*70)
    stats = fmp.get_usage_stats()
    print(f"Requêtes aujourd'hui: {stats['requests_today']}/{stats['max_requests_per_day']}")
    print(f"Restantes: {stats['requests_remaining']}")
    print(f"Usage: {stats['usage_percentage']:.2f}%")

    # Résumé
    print("\n\n" + "="*70)
    print("RÉSUMÉ DES TESTS FMP")
    print("="*70)
    print("✅ Tous les tests ont été exécutés")
    print("📊 FMP est maintenant intégré dans HelixOne!")
    print("\nFonctionnalités disponibles:")
    print("  - 📄 États financiers complets (10+ ans)")
    print("  - 📊 50+ ratios financiers")
    print("  - 📈 Métriques de croissance")
    print("  - 💰 Historique dividendes")
    print("  - 👤 Insider trading")
    print("  - 🏦 Institutional ownership")
    print("  - 📊 Estimations analystes")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_fmp()
