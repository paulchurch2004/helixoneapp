#!/usr/bin/env python3
"""
Test End-to-End du Système d'Analyse de Portefeuille
Teste le workflow complet : Collecte → Analyse → Prédiction → Recommandation → Alertes
"""

import sys
import asyncio
sys.path.insert(0, 'helixone-backend')

from dotenv import load_dotenv
load_dotenv('helixone-backend/.env')

from app.schemas.scenario import Portfolio
from app.services.portfolio.portfolio_scheduler import PortfolioScheduler

print("="*80)
print("🚀 TEST END-TO-END - SYSTÈME D'ANALYSE DE PORTEFEUILLE")
print("="*80)
print()

# Portfolio de démo
demo_portfolio = Portfolio(
    positions={
        'AAPL': 100,
        'TSLA': 50,
        'NVDA': 75,
        'MSFT': 80
    },
    cash=10000.0
)

print("📊 Portfolio de test:")
print(f"   Positions: {len(demo_portfolio.positions)}")
for ticker, qty in demo_portfolio.positions.items():
    print(f"   - {ticker}: {qty} actions")
print(f"   Cash: ${demo_portfolio.cash:,.2f}")
print()

print("🔄 Lancement de l'analyse complète...")
print("   Cela va prendre 30-60 secondes...")
print()

async def run_test():
    scheduler = PortfolioScheduler()

    # Lancer analyse manuelle
    await scheduler._run_complete_analysis(
        user_id="test_user",
        portfolio=demo_portfolio,
        analysis_time="manual"
    )

    # Récupérer résultats
    results = scheduler.last_analysis.get("test_user", {})

    print()
    print("="*80)
    print("✅ ANALYSE TERMINÉE")
    print("="*80)
    print()

    if results:
        print(f"📈 Résultats:")
        print(f"   Health Score: {results.get('health_score', 'N/A'):.0f}/100")
        print(f"   Alertes totales: {results.get('alerts_count', 0)}")
        print(f"   Alertes critiques: {results.get('critical_count', 0)}")
        print()

    print("💡 Prochaines étapes:")
    print("   1. Vérifier les logs ci-dessus pour le détail de l'analyse")
    print("   2. Intégrer avec votre base de données")
    print("   3. Créer les API endpoints")
    print("   4. Configurer les notifications push")
    print("   5. Démarrer le scheduler automatique")
    print()
    print("🎯 Le système est prêt à être intégré dans votre application !")
    print()

try:
    asyncio.run(run_test())
except KeyboardInterrupt:
    print("\n⏸️ Test interrompu")
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
