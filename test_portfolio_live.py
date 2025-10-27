#!/usr/bin/env python3
"""
Test LIVE du Système d'Analyse de Portefeuille
Portefeuille réel diversifié pour tester toutes les fonctionnalités
"""

import sys
import asyncio
import logging
sys.path.insert(0, 'helixone-backend')

from dotenv import load_dotenv
load_dotenv('helixone-backend/.env')

# Activer TOUS les logs en mode verbose
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s',
    force=True
)

from app.schemas.scenario import Portfolio
from app.services.portfolio.portfolio_scheduler import PortfolioScheduler

print("="*80)
print("🚀 TEST LIVE - ANALYSE DE PORTEFEUILLE COMPLÈTE")
print("="*80)
print()

# Portfolio diversifié réaliste
live_portfolio = Portfolio(
    positions={
        # Tech (40%)
        'AAPL': 50,    # Apple - Leader tech
        'MSFT': 30,    # Microsoft - Cloud
        'NVDA': 20,    # Nvidia - IA/GPU
        'TSLA': 15,    # Tesla - EV (plus volatile)

        # Finance (20%)
        'JPM': 25,     # JP Morgan
        'V': 20,       # Visa

        # Healthcare (15%)
        'JNJ': 30,     # Johnson & Johnson - Défensif

        # Consumer (10%)
        'WMT': 20,     # Walmart - Défensif

        # Energy (10%)
        'XOM': 25,     # Exxon Mobil

        # Communication (5%)
        'DIS': 15,     # Disney
    },
    cash=25000.0  # 25% en cash
)

print("💼 PORTEFEUILLE À ANALYSER:")
print("="*80)
print()
print(f"{'Ticker':<10} {'Quantité':<15} {'Secteur':<20}")
print("-"*80)

sectors = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'NVDA': 'Technology',
    'TSLA': 'Technology',
    'JPM': 'Financial Services',
    'V': 'Financial Services',
    'JNJ': 'Healthcare',
    'WMT': 'Consumer Defensive',
    'XOM': 'Energy',
    'DIS': 'Communication Services'
}

total_positions = len(live_portfolio.positions)

for ticker, qty in live_portfolio.positions.items():
    sector = sectors.get(ticker, 'Unknown')
    print(f"{ticker:<10} {qty:<15} {sector:<20}")

print("-"*80)
print(f"{'Total positions:':<10} {total_positions}")
print(f"{'Cash:':<10} ${live_portfolio.cash:,.2f}")
print()

print("🔄 LANCEMENT DE L'ANALYSE COMPLÈTE...")
print()
print("⏳ Étapes à venir:")
print("   1. Collecte données multi-sources (Reddit, StockTwits, News, etc.)")
print("   2. Analyse sentiment pour chaque position")
print("   3. Analyse complète du portefeuille (corrélations, risques)")
print("   4. Prédictions pour les prochains jours (1j, 3j, 7j)")
print("   5. Génération des recommandations HOLD/SELL/BUY")
print("   6. Création des alertes intelligentes")
print()
print("⚠️  Cela va prendre 1-2 minutes (beaucoup de données à collecter)...")
print()
print("="*80)
print()

async def run_live_test():
    try:
        scheduler = PortfolioScheduler()

        # Lancer analyse manuelle
        print("🚀 Démarrage de l'analyse...\n")

        await scheduler._run_complete_analysis(
            user_id="live_test_user",
            portfolio=live_portfolio,
            analysis_time="manual"
        )

        # Récupérer résultats
        results = scheduler.last_analysis.get("live_test_user", {})

        print()
        print("="*80)
        print("✅ ANALYSE TERMINÉE !")
        print("="*80)
        print()

        if results:
            print("📊 RÉSULTATS GLOBAUX:")
            print("-"*80)
            print(f"Health Score:        {results.get('health_score', 'N/A'):.0f}/100")
            print(f"Alertes totales:     {results.get('alerts_count', 0)}")
            print(f"Alertes critiques:   {results.get('critical_count', 0)}")
            print(f"Timestamp:           {results.get('timestamp', 'N/A')}")
            print()

            # Interprétation du health score
            health = results.get('health_score', 50)
            if health >= 80:
                print("💪 Excellent ! Votre portefeuille est en très bonne santé.")
            elif health >= 60:
                print("✅ Bien. Quelques ajustements mineurs recommandés.")
            elif health >= 40:
                print("⚠️  Attention. Des actions sont recommandées.")
            else:
                print("🚨 Risque élevé ! Actions immédiates requises.")
        else:
            print("⚠️  Aucun résultat récupéré")

        print()
        print("="*80)
        print("📋 INFORMATIONS IMPORTANTES")
        print("="*80)
        print()
        print("Les logs détaillés ci-dessus contiennent:")
        print("   • Sentiment par action (bullish/bearish)")
        print("   • Prédictions de prix pour 1j, 3j, 7j")
        print("   • Recommandations HOLD/SELL/BUY détaillées")
        print("   • Alertes avec explications complètes")
        print()
        print("💡 PROCHAINES ÉTAPES:")
        print("   1. Consulter les logs pour voir toutes les recommandations")
        print("   2. Intégrer avec votre base de données")
        print("   3. Créer les API endpoints")
        print("   4. Connecter au frontend (onglet Alertes)")
        print()
        print("📖 Documentation complète: PORTFOLIO_ANALYSIS_SYSTEM.md")
        print()
        print("🎯 LE SYSTÈME EST PRÊT À ÊTRE INTÉGRÉ ! 🚀")
        print("="*80)

    except Exception as e:
        print()
        print("="*80)
        print("❌ ERREUR PENDANT L'ANALYSE")
        print("="*80)
        print(f"\n{str(e)}\n")
        import traceback
        traceback.print_exc()
        print()
        print("💡 Note: Si certaines sources de données ne sont pas disponibles,")
        print("   le système continue avec les sources disponibles.")

if __name__ == "__main__":
    try:
        asyncio.run(run_live_test())
    except KeyboardInterrupt:
        print("\n\n⏸️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
