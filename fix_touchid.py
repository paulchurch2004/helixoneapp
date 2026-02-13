#!/usr/bin/env python3
"""
Script pour activer Touch ID manuellement
"""

import getpass
import sys
from pathlib import Path

# Ajouter src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from auth_manager import AuthManager

print("=" * 60)
print("ACTIVATION TOUCH ID")
print("=" * 60)

auth = AuthManager()

# 1. Vérifier Touch ID disponible
if not auth.is_biometric_available():
    print("\n❌ Touch ID non disponible sur cet appareil")
    sys.exit(1)

print(f"\n✅ {auth.get_biometry_type().upper()} disponible")

# 2. Demander les credentials
print("\nEntrez vos identifiants HelixOne:")
email = input("Email: ").strip()
password = getpass.getpass("Mot de passe: ")

if not email or not password:
    print("❌ Email et mot de passe requis")
    sys.exit(1)

# 3. Vérifier qu'on peut se connecter
print("\n📡 Vérification des identifiants...")
try:
    auth.login(email, password)
    print("✅ Identifiants valides")
except Exception as e:
    print(f"❌ Identifiants invalides: {e}")
    sys.exit(1)

# 4. Activer quick login
print("\n🔐 Activation de la connexion rapide...")
if auth.enable_quick_login(email, password):
    print("✅ Connexion rapide activée!")

    # Vérifier
    stored_pwd = auth.secure_storage.get_credentials(email)
    if stored_pwd:
        print("✅ Mot de passe sécurisé dans le Keychain macOS")
    else:
        print("⚠️  Problème: mot de passe non stocké")

    quick_email = auth.get_quick_login_email()
    if quick_email == email:
        print(f"✅ Email enregistré: {quick_email}")
    else:
        print(f"⚠️  Problème avec l'email: {quick_email}")

    print("\n" + "=" * 60)
    print("🎉 Touch ID configuré avec succès!")
    print("=" * 60)
    print("\nVous pouvez maintenant:")
    print("1. Relancer l'application")
    print("2. Cliquer sur '👆 Connexion rapide avec Touch ID'")
    print("3. Toucher le capteur Touch ID")
    print()

else:
    print("❌ Échec activation quick login")
    sys.exit(1)
