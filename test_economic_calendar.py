"""
Test du calendrier économique et de l'intégration avec le portfolio analyzer
"""

import asyncio
import sys
import os

# Ajouter le chemin du module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helixone-backend'))

from app.services.economic_calendar_service import get_economic_calendar_service
from app.services.event_impact_predictor import get_event_impact_predictor


async def test_economic_calendar():
    """Test du calendrier économique"""
    print("=" * 80)
    print("TEST 1: CALENDRIER ÉCONOMIQUE")
    print("=" * 80)

    calendar = get_economic_calendar_service()

    # Récupérer les événements à venir (30 jours)
    print("\n📅 Récupération des événements économiques (30 jours)...")
    events = await calendar.get_upcoming_events(days=30, min_impact='medium')

    print(f"\n✅ {len(events)} événements trouvés:\n")
    for i, event in enumerate(events[:10], 1):  # Afficher les 10 premiers
        print(f"{i}. {event.title}")
        print(f"   Date: {event.date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Impact: {event.impact_level}")
        print(f"   Secteurs affectés: {', '.join(event.affected_sectors) if event.affected_sectors else 'N/A'}")
        print()

    # Récupérer les earnings à venir
    print("\n📊 Récupération des earnings à venir...")
    earnings = await calendar.get_upcoming_earnings(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        days=30
    )

    print(f"\n✅ {len(earnings)} earnings trouvés:\n")
    for earning in earnings[:5]:
        print(f"- {earning.ticker}: {earning.date.strftime('%Y-%m-%d')}")
        print(f"  EPS Estimate: ${earning.eps_estimate:.2f}" if earning.eps_estimate else "  EPS Estimate: N/A")
        print()


async def test_event_impact_predictor():
    """Test du prédicteur d'impact d'événements"""
    print("\n" + "=" * 80)
    print("TEST 2: PRÉDICTEUR D'IMPACT D'ÉVÉNEMENTS")
    print("=" * 80)

    predictor = get_event_impact_predictor()

    # Portfolio de test
    # Pour analyze_portfolio_event_risk: {ticker: {sector, weight}}
    portfolio_positions_risk = {
        'AAPL': {'sector': 'Technology', 'weight': 25.0},
        'JPM': {'sector': 'Financials', 'weight': 20.0},
        'VZ': {'sector': 'Communication Services', 'weight': 15.0},
        'JNJ': {'sector': 'Healthcare', 'weight': 15.0},
        'XOM': {'sector': 'Energy', 'weight': 15.0},
        'PFE': {'sector': 'Healthcare', 'weight': 10.0},
    }

    portfolio_sectors_list = ['Technology', 'Financials', 'Communication Services', 'Healthcare', 'Energy']

    # Pour predict_portfolio_impact: {ticker: quantity} et {ticker: sector}
    portfolio_positions = {ticker: 100.0 for ticker in portfolio_positions_risk.keys()}
    portfolio_sectors = {ticker: data['sector'] for ticker, data in portfolio_positions_risk.items()}

    # Analyser le risque événementiel du portfolio
    print("\n🔍 Analyse du risque événementiel (7 jours)...")
    event_risk = await predictor.analyze_portfolio_event_risk(
        portfolio_positions=portfolio_positions_risk,
        portfolio_sectors=portfolio_sectors_list,
        days_ahead=7
    )

    print(f"\n📊 RÉSUMÉ DU RISQUE ÉVÉNEMENTIEL:")
    print(f"   Total événements: {event_risk.total_events}")
    print(f"   Événements critiques: {event_risk.critical_events}")
    print(f"   Score de risque global: {event_risk.overall_risk_score:.1f}/100")
    print(f"   Niveau de risque: {event_risk.risk_level}")

    print(f"\n💼 RISQUES PAR SECTEUR:")
    for sector, risk in event_risk.sector_risks.items():
        print(f"   {sector}: {risk:.1f}/100")

    print(f"\n💡 RECOMMANDATIONS:")
    for i, rec in enumerate(event_risk.recommendations, 1):
        print(f"   {i}. {rec}")

    # Prédire impact sur les positions
    print("\n\n📈 Prédictions d'impact sur les positions (30 jours)...")
    predictions = await predictor.predict_portfolio_impact(
        portfolio_positions=portfolio_positions,
        portfolio_sectors=portfolio_sectors,
        days_ahead=30
    )

    print(f"\n✅ {len(predictions)} prédictions générées:\n")
    for pred in predictions[:15]:  # Afficher les 15 premières
        print(f"📅 {pred.event.title} ({pred.event.date.strftime('%Y-%m-%d')})")
        print(f"   Impact: {pred.event.impact_level}")

        if pred.ticker:
            print(f"   Ticker: {pred.ticker}")
        if pred.sector:
            print(f"   Secteur: {pred.sector}")

        print(f"   Impact prédit: {pred.predicted_impact_pct:+.2f}%")
        print(f"   Direction: {pred.direction}")
        print(f"   Confiance: {pred.confidence:.0f}%")

        if pred.factors:
            print(f"   Facteurs:")
            for factor in pred.factors[:2]:  # Max 2 facteurs
                print(f"      - {factor}")

        print()


async def main():
    """Fonction principale"""
    try:
        await test_economic_calendar()
        await test_event_impact_predictor()

        print("\n" + "=" * 80)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
