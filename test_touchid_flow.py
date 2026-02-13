#!/usr/bin/env python3
"""
Test du flow complet Touch ID
"""

import sys
from pathlib import Path

# Ajouter src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import time

from auth_manager import AuthManager

print("=" * 60)
print("TEST DU FLOW TOUCH ID")
print("=" * 60)

auth = AuthManager()

# 1. Vérifier disponibilité biométrie
print("\n1️⃣ Vérification de la biométrie...")
available = auth.is_biometric_available()
print(f"   Biométrie disponible: {available}")

biometry_type = auth.get_biometry_type()
print(f"   Type: {biometry_type}")

# 2. Vérifier si connexion rapide activée
print("\n2️⃣ Vérification connexion rapide...")
quick_login_enabled = auth.is_quick_login_enabled()
print(f"   Connexion rapide activée: {quick_login_enabled}")

if quick_login_enabled:
    email = auth.get_quick_login_email()
    print(f"   Email: {email}")
else:
    print("   ⚠️  Connexion rapide non activée!")
    print("   Pour l'activer:")
    print("   1. Connectez-vous normalement")
    print("   2. Cochez 'Se souvenir de cet appareil'")
    sys.exit(0)

# 3. Test de connexion biométrique
if available and quick_login_enabled:
    print("\n3️⃣ Test de connexion avec Touch ID...")
    print("   👆 Touchez le capteur Touch ID...")

    result = {"done": False, "success": False, "error": None}

    def on_result(success, error):
        result["success"] = success
        result["error"] = error
        result["done"] = True

        if success:
            print("   ✅ Authentification biométrique réussie!")

            # Vérifier si connecté
            if auth.is_logged_in():
                user = auth.get_current_user()
                print(f"   ✅ Connecté en tant que: {user.get('email')}")
            else:
                print("   ❌ Biométrie OK mais pas connecté au backend")
        else:
            print(f"   ❌ Échec: {error}")

    auth.biometric_login(callback=on_result)

    # Attendre le résultat
    timeout = 30
    elapsed = 0
    while not result["done"] and elapsed < timeout:
        time.sleep(0.5)
        elapsed += 0.5

    if not result["done"]:
        print("   ⏱️  Timeout!")

else:
    print("\n❌ Impossible de tester: biométrie ou connexion rapide non disponible")

print("\n" + "=" * 60)
print("FIN DU TEST")
print("=" * 60)
