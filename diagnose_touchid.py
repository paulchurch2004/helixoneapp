#!/usr/bin/env python3
"""
Diagnostic complet de Touch ID
"""

import sys
from pathlib import Path

# Ajouter src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import json

from auth_manager import AuthManager
from biometric_auth import BiometricAuth
from device_manager import DeviceManager
from secure_storage import SecureStorage

print("=" * 70)
print("DIAGNOSTIC TOUCH ID")
print("=" * 70)

# 1. BiometricAuth
print("\n📱 1. BIOMETRIC AUTH")
print("-" * 70)
bio = BiometricAuth()
print(f"Platform: {bio.platform}")
print(f"Biométrie disponible: {bio.is_available()}")
print(f"Type: {bio.get_biometry_type()}")

# 2. SecureStorage
print("\n🔐 2. SECURE STORAGE")
print("-" * 70)
storage = SecureStorage()
print(f"Use keyring: {storage.use_keyring}")
print(f"Service name: {storage.SERVICE_NAME}")

# 3. DeviceManager
print("\n💻 3. DEVICE MANAGER")
print("-" * 70)
device = DeviceManager()
print(f"Device ID: {device.get_device_id()[:16]}...")
print(f"Device name: {device.get_device_name()}")

# 4. AuthManager
print("\n🔑 4. AUTH MANAGER")
print("-" * 70)
auth = AuthManager()
print(f"Logged in: {auth.is_logged_in()}")
print(f"Biometric available: {auth.is_biometric_available()}")
print(f"Quick login enabled: {auth.is_quick_login_enabled()}")

if auth.is_quick_login_enabled():
    email = auth.get_quick_login_email()
    print(f"Quick login email: {email}")

    # Vérifier si mot de passe stocké
    if email:
        pwd = storage.get_credentials(email)
        if pwd:
            print(f"✅ Password stocké dans keychain: {'*' * len(pwd)}")
        else:
            print("❌ PAS DE PASSWORD STOCKÉ!")
            print("\n⚠️  PROBLÈME IDENTIFIÉ:")
            print("   Le fichier quick_login existe mais le mot de passe")
            print("   n'est pas dans le Keychain.")
            print("\n💡 SOLUTION:")
            print("   Lancez: python fix_touchid.py")
else:
    print("ℹ️  Connexion rapide non activée")
    print("\n💡 Pour activer:")
    print("   1. Connectez-vous à l'app")
    print("   2. Cochez 'Se souvenir de cet appareil'")
    print("   OU lancez: python fix_touchid.py")

# 5. Fichiers de config
print("\n📁 5. FICHIERS DE CONFIGURATION")
print("-" * 70)
home = Path.home()
quick_login_file = home / ".helixone_quick_login.json"
session_file = home / ".helixone_session.json"

if quick_login_file.exists():
    print(f"✅ {quick_login_file}")
    with open(quick_login_file) as f:
        data = json.load(f)
        print(f"   Email: {data.get('email')}")
        print(f"   Enabled: {data.get('enabled')}")
        print(f"   Device ID: {data.get('device_id', '')[:16]}...")
else:
    print(f"❌ {quick_login_file} (n'existe pas)")

if session_file.exists():
    print(f"✅ {session_file}")
    try:
        with open(session_file) as f:
            data = json.load(f)
            if "user" in data:
                print(f"   User: {data['user'].get('email')}")
            if "token" in data:
                print(f"   Token: {data['token'][:20]}...")
    except:
        print("   (erreur lecture)")
else:
    print(f"⚠️  {session_file} (n'existe pas)")

print("\n" + "=" * 70)
print("FIN DU DIAGNOSTIC")
print("=" * 70)
