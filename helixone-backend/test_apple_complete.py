"""
Test Complet - Apple (AAPL)
Interroge TOUTES les sources de données disponibles pour Apple
"""

import sys
import logging
from datetime import datetime, timedelta

# Import de tous les collectors
from app.services.alpha_vantage_collector import get_alpha_vantage_collector
from app.services.finnhub_collector import get_finnhub_collector
from app.services.fmp_collector import get_fmp_collector
from app.services.twelvedata_collector import get_twelvedata_collector
from app.services.sec_edgar_collector import get_sec_edgar_collector
from app.services.iex_cloud_collector import get_iex_cloud_collector
from app.services.google_trends_collector import get_google_trends_collector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

TICKER = "AAPL"
COMPANY_NAME = "Apple Inc."


def test_apple_complete():
    """Test complet de toutes les sources pour Apple"""

    print("\n" + "="*80)
    print(f"🍎 TEST COMPLET - {COMPANY_NAME} ({TICKER})")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    results = {
        'success': [],
        'partial': [],
        'failed': [],
        'total_tests': 0
    }

    # ========================================================================
    # 1. ALPHA VANTAGE - Données de marché
    # ========================================================================
    print("\n" + "="*80)
    print("📊 1. ALPHA VANTAGE - Données de marché")
    print("="*80)

    try:
        av = get_alpha_vantage_collector()

        # Quote intraday
        print("\n📈 Quote temps réel:")
        try:
            quote = av.get_quote(TICKER)
            if quote and '01. symbol' in quote:
                print(f"   ✅ Prix: ${quote.get('05. price', 'N/A')}")
                print(f"   ✅ Volume: {quote.get('06. volume', 'N/A')}")
                print(f"   ✅ Dernier trade: {quote.get('07. latest trading day', 'N/A')}")
                results['success'].append('Alpha Vantage - Quote')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse partielle")
                results['partial'].append('Alpha Vantage - Quote')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('Alpha Vantage - Quote')
            results['total_tests'] += 1

        # Données journalières
        print("\n📊 Données journalières:")
        try:
            daily = av.get_daily(TICKER)
            if daily and 'Time Series (Daily)' in daily:
                dates = list(daily['Time Series (Daily)'].keys())[:3]
                print(f"   ✅ {len(daily['Time Series (Daily)'])} jours de données")
                print(f"   ✅ Dernières dates: {', '.join(dates)}")
                results['success'].append('Alpha Vantage - Daily')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse partielle")
                results['partial'].append('Alpha Vantage - Daily')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('Alpha Vantage - Daily')
            results['total_tests'] += 1

    except Exception as e:
        print(f"❌ Erreur initialisation Alpha Vantage: {e}")
        results['failed'].append('Alpha Vantage - Init')
        results['total_tests'] += 1

    # ========================================================================
    # 2. FINNHUB - News et données fondamentales
    # ========================================================================
    print("\n" + "="*80)
    print("📰 2. FINNHUB - News et sentiments")
    print("="*80)

    try:
        finnhub = get_finnhub_collector()

        # News
        print("\n📰 Dernières news:")
        try:
            news = finnhub.get_company_news(TICKER)
            if news and len(news) > 0:
                print(f"   ✅ {len(news)} articles trouvés")
                for article in news[:3]:
                    print(f"   📄 {article.get('headline', 'No title')[:60]}...")
                results['success'].append('Finnhub - News')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Aucune news")
                results['partial'].append('Finnhub - News')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('Finnhub - News')
            results['total_tests'] += 1

        # Profile
        print("\n🏢 Profil entreprise:")
        try:
            profile = finnhub.get_company_profile(TICKER)
            if profile and 'name' in profile:
                print(f"   ✅ Nom: {profile.get('name')}")
                print(f"   ✅ Industrie: {profile.get('finnhubIndustry', 'N/A')}")
                print(f"   ✅ Market Cap: ${profile.get('marketCapitalization', 'N/A')}B")
                results['success'].append('Finnhub - Profile')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse partielle")
                results['partial'].append('Finnhub - Profile')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('Finnhub - Profile')
            results['total_tests'] += 1

    except Exception as e:
        print(f"❌ Erreur initialisation Finnhub: {e}")
        results['failed'].append('Finnhub - Init')
        results['total_tests'] += 1

    # ========================================================================
    # 3. FMP - Données fondamentales
    # ========================================================================
    print("\n" + "="*80)
    print("💰 3. FMP - Données fondamentales")
    print("="*80)

    try:
        fmp = get_fmp_collector()

        # Quote
        print("\n💵 Quote:")
        try:
            quote = fmp.get_quote(TICKER)
            if quote and len(quote) > 0:
                q = quote[0]
                print(f"   ✅ Prix: ${q.get('price', 'N/A')}")
                print(f"   ✅ Change: {q.get('change', 'N/A')} ({q.get('changesPercentage', 'N/A')}%)")
                print(f"   ✅ Market Cap: ${q.get('marketCap', 'N/A'):,}")
                results['success'].append('FMP - Quote')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse vide")
                results['partial'].append('FMP - Quote')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('FMP - Quote')
            results['total_tests'] += 1

        # Ratios financiers
        print("\n📊 Ratios financiers:")
        try:
            ratios = fmp.get_financial_ratios(TICKER)
            if ratios and len(ratios) > 0:
                r = ratios[0]
                print(f"   ✅ P/E Ratio: {r.get('peRatio', 'N/A')}")
                print(f"   ✅ ROE: {r.get('returnOnEquity', 'N/A')}")
                print(f"   ✅ Debt/Equity: {r.get('debtEquityRatio', 'N/A')}")
                results['success'].append('FMP - Ratios')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse vide")
                results['partial'].append('FMP - Ratios')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('FMP - Ratios')
            results['total_tests'] += 1

        # Revenus
        print("\n💵 Revenus:")
        try:
            income = fmp.get_income_statement(TICKER, limit=1)
            if income and len(income) > 0:
                i = income[0]
                print(f"   ✅ Revenus: ${i.get('revenue', 0):,}")
                print(f"   ✅ Net Income: ${i.get('netIncome', 0):,}")
                print(f"   ✅ Date: {i.get('date', 'N/A')}")
                results['success'].append('FMP - Income')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse vide")
                results['partial'].append('FMP - Income')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('FMP - Income')
            results['total_tests'] += 1

    except Exception as e:
        print(f"❌ Erreur initialisation FMP: {e}")
        results['failed'].append('FMP - Init')
        results['total_tests'] += 1

    # ========================================================================
    # 4. SEC EDGAR - Filings officiels
    # ========================================================================
    print("\n" + "="*80)
    print("📄 4. SEC EDGAR - Filings officiels")
    print("="*80)

    try:
        sec = get_sec_edgar_collector()

        # CIK lookup
        print("\n🔍 Recherche CIK:")
        try:
            cik = sec.get_cik_by_ticker(TICKER)
            if cik:
                print(f"   ✅ CIK trouvé: {cik}")

                # 10-K filings
                print("\n📋 Derniers filings 10-K:")
                try:
                    filings = sec.get_10k_filings(cik, limit=3)
                    if filings and len(filings) > 0:
                        print(f"   ✅ {len(filings)} filings 10-K trouvés")
                        for f in filings:
                            print(f"   📄 {f.get('filingDate')}: {f.get('form')} - {f.get('primaryDocument')}")
                        results['success'].append('SEC Edgar - 10-K')
                        results['total_tests'] += 1
                    else:
                        print(f"   ⚠️  Aucun filing")
                        results['partial'].append('SEC Edgar - 10-K')
                        results['total_tests'] += 1
                except Exception as e:
                    print(f"   ❌ Erreur 10-K: {e}")
                    results['failed'].append('SEC Edgar - 10-K')
                    results['total_tests'] += 1

                results['success'].append('SEC Edgar - CIK')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  CIK non trouvé")
                results['partial'].append('SEC Edgar - CIK')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('SEC Edgar - CIK')
            results['total_tests'] += 1

    except Exception as e:
        print(f"❌ Erreur initialisation SEC Edgar: {e}")
        results['failed'].append('SEC Edgar - Init')
        results['total_tests'] += 1

    # ========================================================================
    # 5. GOOGLE TRENDS - Intérêt de recherche
    # ========================================================================
    print("\n" + "="*80)
    print("🔍 5. GOOGLE TRENDS - Intérêt public")
    print("="*80)

    try:
        trends = get_google_trends_collector()

        print("\n📊 Sentiment score:")
        try:
            sentiment = trends.get_stock_sentiment_score(TICKER, timeframe='today 3-m')
            if sentiment and 'sentiment_score' in sentiment:
                print(f"   ✅ Score: {sentiment['sentiment_score']}")
                print(f"   ✅ Trend: {sentiment['trend']}")
                if 'current_interest' in sentiment:
                    print(f"   ✅ Intérêt actuel: {sentiment['current_interest']}/100")
                results['success'].append('Google Trends - Sentiment')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse partielle")
                results['partial'].append('Google Trends - Sentiment')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results['failed'].append('Google Trends - Sentiment')
            results['total_tests'] += 1

    except Exception as e:
        print(f"❌ Erreur initialisation Google Trends: {e}")
        results['failed'].append('Google Trends - Init')
        results['total_tests'] += 1

    # ========================================================================
    # 6. IEX CLOUD - Données temps réel (si clé disponible)
    # ========================================================================
    print("\n" + "="*80)
    print("⚡ 6. IEX CLOUD - Données temps réel")
    print("="*80)

    try:
        iex = get_iex_cloud_collector()

        print("\n📈 Quote temps réel:")
        try:
            quote = iex.get_quote(TICKER)
            if quote and 'symbol' in quote:
                print(f"   ✅ Symbol: {quote.get('symbol')}")
                print(f"   ✅ Prix: ${quote.get('latestPrice', 'N/A')}")
                print(f"   ✅ Market Cap: ${quote.get('marketCap', 'N/A'):,}")
                print(f"   ✅ P/E Ratio: {quote.get('peRatio', 'N/A')}")
                results['success'].append('IEX Cloud - Quote')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse partielle ou clé API manquante")
                results['partial'].append('IEX Cloud - Quote')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            print(f"   💡 Clé API probablement manquante")
            results['failed'].append('IEX Cloud - Quote')
            results['total_tests'] += 1

    except Exception as e:
        print(f"❌ Erreur initialisation IEX Cloud: {e}")
        results['failed'].append('IEX Cloud - Init')
        results['total_tests'] += 1

    # ========================================================================
    # 7. TWELVE DATA - Prix et indicateurs techniques (si clé disponible)
    # ========================================================================
    print("\n" + "="*80)
    print("📊 7. TWELVE DATA - Indicateurs techniques")
    print("="*80)

    try:
        twelve = get_twelvedata_collector()

        print("\n📈 Prix:")
        try:
            quote = twelve.get_quote(TICKER)
            if quote and 'symbol' in quote:
                print(f"   ✅ Symbol: {quote.get('symbol')}")
                print(f"   ✅ Prix: ${quote.get('close', 'N/A')}")
                print(f"   ✅ Change: {quote.get('change', 'N/A')}%")
                results['success'].append('Twelve Data - Quote')
                results['total_tests'] += 1
            else:
                print(f"   ⚠️  Réponse partielle ou clé API manquante")
                results['partial'].append('Twelve Data - Quote')
                results['total_tests'] += 1
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            print(f"   💡 Clé API probablement manquante")
            results['failed'].append('Twelve Data - Quote')
            results['total_tests'] += 1

    except Exception as e:
        print(f"❌ Erreur initialisation Twelve Data: {e}")
        results['failed'].append('Twelve Data - Init')
        results['total_tests'] += 1

    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DU TEST COMPLET - APPLE (AAPL)")
    print("="*80)

    total = results['total_tests']
    success = len(results['success'])
    partial = len(results['partial'])
    failed = len(results['failed'])

    success_rate = (success / total * 100) if total > 0 else 0

    print(f"\n📈 Statistiques:")
    print(f"   Tests totaux: {total}")
    print(f"   ✅ Succès: {success} ({success_rate:.1f}%)")
    print(f"   ⚠️  Partiels: {partial} ({partial/total*100:.1f}%)")
    print(f"   ❌ Échecs: {failed} ({failed/total*100:.1f}%)")

    print(f"\n✅ Tests réussis ({success}):")
    for test in results['success']:
        print(f"   • {test}")

    if results['partial']:
        print(f"\n⚠️  Tests partiels ({partial}):")
        for test in results['partial']:
            print(f"   • {test}")

    if results['failed']:
        print(f"\n❌ Tests échoués ({failed}):")
        for test in results['failed']:
            print(f"   • {test}")

    print("\n" + "="*80)
    print(f"🎯 TAUX DE RÉUSSITE: {success_rate:.1f}%")
    print("="*80 + "\n")

    # Recommandations
    print("💡 Recommandations:")
    if failed > 0:
        print("   • Obtenir les clés API manquantes (IEX Cloud, Twelve Data)")
        print("   • Vérifier la connectivité réseau pour SEC Edgar")
        print("   • Augmenter les limites de taux pour Google Trends")
    if success_rate >= 70:
        print("   ✅ Couverture excellente pour Apple!")
    elif success_rate >= 50:
        print("   ⚠️  Couverture correcte mais améliorable")
    else:
        print("   ❌ Couverture insuffisante - actions requises")

    print()


if __name__ == "__main__":
    try:
        test_apple_complete()
    except Exception as e:
        logger.exception("Erreur globale test Apple complet")
        sys.exit(1)
