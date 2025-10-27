"""
Script pour tester la récupération du dashboard IBKR
"""

import sys
import os
from dotenv import load_dotenv
import asyncio

# IMPORTANT: Changer le répertoire de travail vers helixone-backend
backend_dir = os.path.join(os.path.dirname(__file__), 'helixone-backend')
os.chdir(backend_dir)

# Charger les variables d'environnement depuis .env
env_path = os.path.join(backend_dir, '.env')
load_dotenv(env_path)

# Ajouter le chemin du backend au PYTHONPATH
sys.path.insert(0, backend_dir)

from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.user import User
from app.models.ibkr import IBKRConnection
from app.services.ibkr_service import get_ibkr_service

async def test_ibkr_dashboard():
    """Tester la récupération du dashboard IBKR"""
    db: Session = SessionLocal()

    try:
        # Trouver l'utilisateur
        user = db.query(User).filter(User.email == "datacollector@helixone.com").first()

        if not user:
            print("❌ Utilisateur non trouvé")
            return

        print(f"✅ Utilisateur: {user.email}")

        # Vérifier la connexion IBKR
        connection = db.query(IBKRConnection).filter(
            IBKRConnection.user_id == user.id,
            IBKRConnection.is_active == True
        ).first()

        if not connection:
            print("❌ Pas de connexion IBKR configurée")
            return

        print(f"✅ Connexion IBKR: {connection.account_id}")
        print(f"   Auto-connect: {connection.auto_connect}")
        print(f"   Is connected: {connection.is_connected}")

        # Créer le service
        service = get_ibkr_service(db, user.id)

        print("\n📊 Tentative de connexion à IBKR...")

        # Connecter
        if not service.is_connected:
            success = await service.connect(auto_connect=True)
            if not success:
                print("❌ Échec de connexion à IBKR")
                return

        print("✅ Connecté à IBKR!")

        # Récupérer le portefeuille
        print("\n📊 Récupération du portefeuille...")
        portfolio = await service.get_portfolio()

        if not portfolio:
            print("❌ Impossible de récupérer le portefeuille")
            return

        print(f"\n✅ PORTEFEUILLE RÉCUPÉRÉ!")
        print(f"   Account: {portfolio['account_id']}")
        print(f"   Net Liquidation: {portfolio['net_liquidation']:.2f} {portfolio['currency']}")
        print(f"   Cash: {portfolio['total_cash']:.2f} {portfolio['currency']}")
        print(f"   Stock Value: {portfolio['stock_value']:.2f} {portfolio['currency']}")
        print(f"   Unrealized P&L: {portfolio['unrealized_pnl']:.2f} {portfolio['currency']}")
        print(f"   Positions: {len(portfolio['positions'])}")

        if portfolio['positions']:
            print(f"\n📦 POSITIONS:")
            for pos in portfolio['positions']:
                print(f"   • {pos['symbol']:8} {pos['position']:8.2f} @ {pos['avg_cost']:.2f}")

        # Sauvegarder un snapshot
        print(f"\n💾 Sauvegarde du snapshot...")
        snapshot = await service.save_portfolio_snapshot(portfolio)
        if snapshot:
            print(f"✅ Snapshot sauvegardé: {snapshot.id}")

        # Vérifier les alertes
        print(f"\n🔔 Vérification des alertes...")
        alerts = await service.check_alerts(portfolio)
        if alerts:
            print(f"✅ {len(alerts)} alerte(s) générée(s)")
            for alert in alerts:
                print(f"   • [{alert.severity}] {alert.title}")
        else:
            print(f"✅ Aucune alerte")

        # Déconnecter
        service.disconnect()
        print(f"\n🔌 Déconnecté de IBKR")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DASHBOARD IBKR")
    print("=" * 60)
    print()

    asyncio.run(test_ibkr_dashboard())
