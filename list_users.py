"""
Script pour lister tous les utilisateurs
"""

import sys
import os
from dotenv import load_dotenv

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

db: Session = SessionLocal()

try:
    users = db.query(User).all()

    if not users:
        print("❌ Aucun utilisateur trouvé dans la base de données")
        print("\n💡 Pour créer un utilisateur, utilise l'API:")
        print("   POST http://127.0.0.1:8000/auth/register")
        print("   Body: {\"email\": \"...\", \"password\": \"...\", \"first_name\": \"...\", \"last_name\": \"...\"}")
    else:
        print(f"✅ {len(users)} utilisateur(s) trouvé(s):\n")
        for user in users:
            print(f"  📧 {user.email}")
            print(f"     ID: {user.id}")
            print(f"     Nom: {user.first_name} {user.last_name}")
            print(f"     Créé le: {user.created_at}")
            print(f"     Actif: {'✅' if user.is_active else '❌'}")
            print(f"     Email vérifié: {'✅' if user.email_verified else '❌'}")
            print()

except Exception as e:
    print(f"❌ Erreur: {e}")
finally:
    db.close()
