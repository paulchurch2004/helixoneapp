"""
Script pour configurer la connexion IBKR pour un utilisateur
Lance ce script pour créer/mettre à jour la configuration IBKR dans la BDD
"""

import sys
import os
from dotenv import load_dotenv

# IMPORTANT: Changer le répertoire de travail vers helixone-backend
# pour que la base de données soit la même que celle utilisée par l'API
backend_dir = os.path.join(os.path.dirname(__file__), 'helixone-backend')
os.chdir(backend_dir)

# Charger les variables d'environnement depuis .env
env_path = os.path.join(backend_dir, '.env')
load_dotenv(env_path)

# Ajouter le chemin du backend au PYTHONPATH
sys.path.insert(0, backend_dir)

from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine
from app.models import Base
from app.models.user import User
from app.models.ibkr import IBKRConnection
from datetime import datetime

# Créer toutes les tables
Base.metadata.create_all(bind=engine)

def setup_ibkr_connection(
    email: str,
    account_id: str = "U17421384",
    host: str = "127.0.0.1",
    port: int = 7496,  # 7496=LIVE, 7497=PAPER
    client_id: int = 1
):
    """
    Configurer la connexion IBKR pour un utilisateur

    Args:
        email: Email de l'utilisateur
        account_id: ID du compte IBKR
        host: Host TWS/Gateway
        port: Port (7496=LIVE, 7497=PAPER)
        client_id: Client ID unique
    """
    db: Session = SessionLocal()

    try:
        # Chercher l'utilisateur
        user = db.query(User).filter(User.email == email).first()

        if not user:
            print(f"❌ Utilisateur {email} non trouvé")
            print(f"📋 Utilisateurs disponibles:")
            users = db.query(User).all()
            for u in users:
                print(f"   - {u.email} (créé le {u.created_at})")
            return False

        print(f"✅ Utilisateur trouvé: {user.email}")

        # Vérifier si une connexion existe déjà
        existing = db.query(IBKRConnection).filter(
            IBKRConnection.user_id == user.id,
            IBKRConnection.account_id == account_id
        ).first()

        if existing:
            print(f"🔄 Mise à jour de la connexion existante...")
            existing.connection_type = 'live'
            existing.host = host
            existing.port = port
            existing.client_id = client_id
            existing.auto_connect = True
            existing.is_active = True
            connection = existing
        else:
            print(f"➕ Création d'une nouvelle connexion...")
            connection = IBKRConnection(
                user_id=user.id,
                account_id=account_id,
                connection_type='live',
                host=host,
                port=port,
                client_id=client_id,
                auto_connect=True,
                is_active=True
            )
            db.add(connection)

        db.commit()
        db.refresh(connection)

        print(f"\n✅ CONNEXION IBKR CONFIGURÉE")
        print(f"   User: {user.email}")
        print(f"   Account ID: {connection.account_id}")
        print(f"   Type: {connection.connection_type.upper()}")
        print(f"   Host: {connection.host}:{connection.port}")
        print(f"   Client ID: {connection.client_id}")
        print(f"   Auto-connect: {'✅ OUI' if connection.auto_connect else '❌ NON'}")
        print(f"   Active: {'✅ OUI' if connection.is_active else '❌ NON'}")
        print(f"\n🔌 Au prochain démarrage, l'API se connectera automatiquement à IBKR!")

        return True

    except Exception as e:
        print(f"❌ Erreur: {e}")
        db.rollback()
        return False

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 CONFIGURATION CONNEXION IBKR")
    print("=" * 60)
    print()

    # Paramètres par défaut
    EMAIL = input("📧 Email de l'utilisateur (Enter pour datacollector@helixone.com): ").strip() or "datacollector@helixone.com"
    ACCOUNT_ID = input("🔢 Account ID IBKR (Enter pour U17421384): ").strip() or "U17421384"

    port_input = input("🔌 Port (7496=LIVE, 7497=PAPER, Enter=7496): ").strip()
    PORT = int(port_input) if port_input else 7496

    print()
    print(f"📋 Configuration:")
    print(f"   Email: {EMAIL}")
    print(f"   Account: {ACCOUNT_ID}")
    print(f"   Port: {PORT} ({'🔴 LIVE' if PORT == 7496 else '📝 PAPER'})")
    print()

    confirm = input("Confirmer ? (y/n): ").strip().lower()

    if confirm == 'y':
        success = setup_ibkr_connection(
            email=EMAIL,
            account_id=ACCOUNT_ID,
            port=PORT
        )

        if success:
            print("\n✅ Configuration réussie!")
            print("🔌 Redémarre ton serveur pour que la connexion automatique fonctionne.")
        else:
            print("\n❌ Configuration échouée.")
    else:
        print("❌ Annulé")
