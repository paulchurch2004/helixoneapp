"""
Script de test de la configuration
Lance ce fichier pour vérifier que tout fonctionne
"""

import sys
import os

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.core.config import settings, validate_settings
    from app.core.security import (
        hash_password, 
        verify_password, 
        create_access_token, 
        decode_access_token
    )
    
    print("✅ Imports réussis")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("\nInstallez les dépendances avec :")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def test_configuration():
    """Teste la configuration"""
    print("\n" + "=" * 60)
    print("TEST DE LA CONFIGURATION")
    print("=" * 60)
    
    try:
        print(f"\n✅ App Name: {settings.APP_NAME}")
        print(f"✅ Version: {settings.APP_VERSION}")
        print(f"✅ Environment: {settings.ENVIRONMENT}")
        print(f"✅ Database: {settings.DATABASE_URL}")
        
        validate_settings()
        return True
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False


def test_security():
    """Teste les fonctions de sécurité"""
    print("\n" + "=" * 60)
    print("TEST DE LA SÉCURITÉ")
    print("=" * 60)
    
    try:
        # Test 1: Hash password
        password = "TestPassword123!"
        hashed = hash_password(password)
        print(f"\n✅ Password hashé: {hashed[:50]}...")
        
        # Test 2: Verify password
        if verify_password(password, hashed):
            print("✅ Vérification password: OK")
        else:
            print("❌ Vérification password: ÉCHEC")
            return False
        
        # Test 3: Vérifier rejet mauvais password
        if not verify_password("WrongPassword", hashed):
            print("✅ Rejet mauvais password: OK")
        else:
            print("❌ Rejet mauvais password: ÉCHEC")
            return False
        
        # Test 4: Créer JWT token
        token = create_access_token({"user_id": "test-123"})
        print(f"\n✅ JWT Token créé: {token[:50]}...")
        
        # Test 5: Décoder JWT token
        payload = decode_access_token(token)
        if payload and payload.get("user_id") == "test-123":
            print(f"✅ JWT décodé: user_id = {payload.get('user_id')}")
        else:
            print("❌ Décodage JWT: ÉCHEC")
            return False
        
        # Test 6: Rejeter token invalide
        invalid_payload = decode_access_token("invalid.token.here")
        if invalid_payload is None:
            print("✅ Rejet token invalide: OK")
        else:
            print("❌ Rejet token invalide: ÉCHEC")
            return False
        
        return True
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 DÉMARRAGE DES TESTS\n")
    
    success = True
    
    if not test_configuration():
        success = False
    
    if not test_security():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ✅ ✅ TOUS LES TESTS SONT PASSÉS ✅ ✅ ✅")
        print("=" * 60)
        print("\nVous pouvez passer à l'étape suivante !")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("=" * 60)
        print("\nVérifiez :")
        print("1. Votre fichier .env existe")
        print("2. SECRET_KEY est définie dans .env")
        print("3. Les dépendances sont installées")
    
    print()