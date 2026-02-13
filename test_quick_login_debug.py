#!/usr/bin/env python3
"""
Debug de la connexion rapide
"""

import sys
from pathlib import Path

# Ajouter src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from auth_manager import AuthManager

auth = AuthManager()

print("🔍 DEBUG CONNEXION RAPIDE")
print("=" * 60)

# 1. Email
email = auth.get_quick_login_email()
print(f"Email enregistré: {email}")

# 2. Mot de passe stocké
if email:
    password = auth.secure_storage.get_credentials(email)
    if password:
        print(f"Mot de passe trouvé: {'*' * len(password)}")
    else:
        print("❌ Pas de mot de passe stocké!")
        print("\nPour stocker le mot de passe:")
        print("1. Connectez-vous normalement")
        print("2. Cochez 'Se souvenir de cet appareil'")
        sys.exit(1)

# 3. Tenter la connexion
print("\n📡 Test connexion au backend...")
try:
    result = auth.login(email, password)
    print("✅ Connexion réussie!")
    print(f"Token: {result.get('access_token', '')[:20]}...")

    user = auth.get_current_user()
    print(f"Utilisateur: {user.get('email')}")

except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback

    traceback.print_exc()
