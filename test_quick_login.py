#!/usr/bin/env python3
"""
Test du système de connexion rapide HelixOne
Teste device_id, secure storage, et biométrie
"""

import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from device_manager import DeviceManager
from secure_storage import SecureStorage
from biometric_auth import BiometricAuth
from auth_manager import AuthManager


def test_device_manager():
    """Test du gestionnaire d'appareil"""
    print("\n" + "=" * 60)
    print("🧪 TEST 1: Device Manager")
    print("=" * 60)

    dm = DeviceManager()

    device_id = dm.get_device_id()
    device_name = dm.get_device_name()

    print(f"✅ Device ID: {device_id}")
    print(f"✅ Device Name: {device_name}")

    # Vérifier que l'ID est persistant
    dm2 = DeviceManager()
    device_id2 = dm2.get_device_id()

    if device_id == device_id2:
        print("✅ Device ID est persistant")
    else:
        print("❌ Device ID n'est pas persistant")

    return True


def test_secure_storage():
    """Test du stockage sécurisé"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Secure Storage")
    print("=" * 60)

    storage = SecureStorage()

    test_email = "test@helixone.fr"
    test_password = "SecureP@ssw0rd123"

    # Test sauvegarde
    print(f"📝 Sauvegarde credentials pour {test_email}...")
    if storage.save_credentials(test_email, test_password):
        print("✅ Sauvegarde réussie")
    else:
        print("❌ Échec sauvegarde")
        return False

    # Test récupération
    print(f"🔍 Récupération credentials pour {test_email}...")
    retrieved = storage.get_credentials(test_email)
    if retrieved == test_password:
        print(f"✅ Récupération réussie")
    else:
        print(f"❌ Échec récupération: attendu '{test_password}', reçu '{retrieved}'")
        return False

    # Test suppression
    print(f"🗑️  Suppression credentials pour {test_email}...")
    if storage.delete_credentials(test_email):
        print("✅ Suppression réussie")
    else:
        print("❌ Échec suppression")
        return False

    # Vérifier suppression
    retrieved = storage.get_credentials(test_email)
    if retrieved is None:
        print("✅ Credentials bien supprimés")
    else:
        print(f"❌ Credentials toujours présents: {retrieved}")
        return False

    return True


def test_biometric_auth():
    """Test de l'authentification biométrique"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Biometric Authentication")
    print("=" * 60)

    bio = BiometricAuth()

    print(f"Platform: {bio.platform}")
    print(f"Biométrie disponible: {bio.is_available()}")
    print(f"Type de biométrie: {bio.get_biometry_type()}")

    if bio.is_available():
        print("\n✅ Biométrie disponible sur cet appareil")

        # Demander si l'utilisateur veut tester
        response = input("\n🔐 Voulez-vous tester l'authentification biométrique? (o/n): ")

        if response.lower() == 'o':
            print("⏳ En attente de l'authentification biométrique...")

            result = {'done': False, 'success': False}

            def on_result(success, error):
                result['done'] = True
                result['success'] = success
                if success:
                    print("✅ Authentification biométrique réussie!")
                else:
                    print(f"❌ Authentification échouée: {error}")

            bio.authenticate(
                reason="Test HelixOne - Connexion rapide",
                callback=on_result
            )

            # Attendre le résultat
            import time
            timeout = 30
            elapsed = 0
            while not result['done'] and elapsed < timeout:
                time.sleep(0.5)
                elapsed += 0.5

            if not result['done']:
                print("⏱️  Timeout - Pas de réponse")
                return False

            return result['success']
        else:
            print("⏭️  Test biométrique ignoré")
            return True
    else:
        print("⚠️  Biométrie non disponible sur cet appareil")
        return True


def test_auth_manager():
    """Test de l'AuthManager avec connexion rapide"""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: AuthManager Quick Login")
    print("=" * 60)

    auth = AuthManager()

    print(f"Connexion rapide activée: {auth.is_quick_login_enabled()}")
    print(f"Biométrie disponible: {auth.is_biometric_available()}")
    print(f"Type de biométrie: {auth.get_biometry_type()}")

    # Test activation connexion rapide
    test_email = "test.quick@helixone.fr"
    test_password = "QuickP@ss123"

    print(f"\n📝 Activation connexion rapide pour {test_email}...")
    if auth.enable_quick_login(test_email, test_password):
        print("✅ Connexion rapide activée")
    else:
        print("❌ Échec activation connexion rapide")
        return False

    # Vérifier
    if auth.is_quick_login_enabled():
        print("✅ Connexion rapide bien activée")
        saved_email = auth.get_quick_login_email()
        print(f"   Email sauvegardé: {saved_email}")
    else:
        print("❌ Connexion rapide non activée")
        return False

    # Test désactivation
    print(f"\n🗑️  Désactivation connexion rapide...")
    auth.disable_quick_login()

    if not auth.is_quick_login_enabled():
        print("✅ Connexion rapide désactivée")
    else:
        print("❌ Connexion rapide toujours activée")
        return False

    return True


def main():
    """Exécuter tous les tests"""
    print("\n" + "=" * 60)
    print("🚀 TEST DU SYSTÈME DE CONNEXION RAPIDE HELIXONE")
    print("=" * 60)

    results = {}

    # Test 1: Device Manager
    try:
        results['device_manager'] = test_device_manager()
    except Exception as e:
        print(f"❌ Erreur test device_manager: {e}")
        results['device_manager'] = False

    # Test 2: Secure Storage
    try:
        results['secure_storage'] = test_secure_storage()
    except Exception as e:
        print(f"❌ Erreur test secure_storage: {e}")
        results['secure_storage'] = False

    # Test 3: Biometric Auth
    try:
        results['biometric_auth'] = test_biometric_auth()
    except Exception as e:
        print(f"❌ Erreur test biometric_auth: {e}")
        results['biometric_auth'] = False

    # Test 4: Auth Manager
    try:
        results['auth_manager'] = test_auth_manager()
    except Exception as e:
        print(f"❌ Erreur test auth_manager: {e}")
        results['auth_manager'] = False

    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(1 for r in results.values() if r)

    print("\n" + "=" * 60)
    print(f"📈 RÉSULTAT GLOBAL: {passed}/{total} tests réussis")
    print("=" * 60)

    if passed == total:
        print("\n🎉 Tous les tests sont passés!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) ont échoué")
        return 1


if __name__ == "__main__":
    sys.exit(main())
