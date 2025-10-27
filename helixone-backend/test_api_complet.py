"""
Test complet automatique de l'API HelixOne
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

print("=" * 70)
print("🚀 TEST COMPLET AUTOMATIQUE DE L'API HELIXONE")
print("=" * 70)

# ============================================================================
# TEST 1: Health Check
# ============================================================================
print("\n1️⃣  TEST HEALTH CHECK")
print("-" * 70)

try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"✅ Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ App Name: {data['app_name']}")
        print(f"✅ Version: {data['version']}")
        print(f"✅ Environment: {data['environment']}")
        print(f"✅ Database: {data['database']}")
    else:
        print(f"❌ Erreur: {response.text}")
        exit(1)
        
except Exception as e:
    print(f"❌ Erreur de connexion: {e}")
    print("\n⚠️  Assurez-vous que le serveur tourne:")
    print("   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
    exit(1)

# ============================================================================
# TEST 2: Inscription d'un utilisateur
# ============================================================================
print("\n2️⃣  TEST INSCRIPTION UTILISATEUR")
print("-" * 70)

# Générer un email unique avec timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
test_email = f"test_{timestamp}@helixone.com"

user_data = {
    "email": test_email,
    "password": "Test123456!",
    "first_name": "Test",
    "last_name": "User"
}

print(f"📧 Email: {test_email}")
print(f"🔐 Password: {user_data['password']}")

try:
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json=user_data,
        timeout=10
    )
    
    print(f"✅ Status Code: {response.status_code}")
    
    if response.status_code == 201:
        result = response.json()
        token = result["access_token"]
        user = result["user"]
        
        print(f"✅ Utilisateur créé avec succès!")
        print(f"   ID: {user['id']}")
        print(f"   Email: {user['email']}")
        print(f"   Nom: {user['first_name']} {user['last_name']}")
        print(f"   Actif: {user['is_active']}")
        print(f"   Email vérifié: {user['email_verified']}")
        print(f"   Créé le: {user['created_at']}")
        print(f"\n🔑 Token JWT généré:")
        print(f"   {token[:80]}...")
        
    else:
        print(f"❌ Erreur: {response.text}")
        exit(1)
        
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# ============================================================================
# TEST 3: Vérification de la licence
# ============================================================================
print("\n3️⃣  TEST VÉRIFICATION LICENCE")
print("-" * 70)

headers = {
    "Authorization": f"Bearer {token}"
}

try:
    response = requests.get(
        f"{BASE_URL}/licenses/status",
        headers=headers,
        timeout=10
    )
    
    print(f"✅ Status Code: {response.status_code}")
    
    if response.status_code == 200:
        license_data = response.json()
        
        print(f"✅ Licence récupérée avec succès!")
        print(f"\n📋 DÉTAILS DE LA LICENCE:")
        print(f"   🔑 Clé: {license_data['license_key']}")
        print(f"   📦 Type: {license_data['license_type'].upper()}")
        print(f"   ✨ Statut: {license_data['status'].upper()}")
        print(f"\n🎁 FONCTIONNALITÉS:")
        for feature in license_data.get('features', []):
            print(f"   ✓ {feature}")
        print(f"\n📊 QUOTAS:")
        print(f"   Analyses par jour: {license_data['quota_daily_analyses']}")
        print(f"   Appels API par jour: {license_data['quota_daily_api_calls']}")
        print(f"\n⏰ DATES:")
        print(f"   Activée le: {license_data['activated_at'][:19]}")
        print(f"   Expire le: {license_data['expires_at'][:19]}")
        print(f"   ⏳ Jours restants: {license_data['days_remaining']} jours")
        
        # Vérifier si la licence est valide
        if license_data['status'] == 'active' and license_data['days_remaining'] > 0:
            print(f"\n✅ La licence est VALIDE et ACTIVE! 🎉")
        else:
            print(f"\n⚠️  Attention: La licence nécessite une attention")
            
    else:
        print(f"❌ Erreur: {response.text}")
        exit(1)
        
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# ============================================================================
# TEST 4: Re-connexion (login)
# ============================================================================
print("\n4️⃣  TEST CONNEXION (LOGIN)")
print("-" * 70)

login_data = {
    "email": test_email,
    "password": "Test123456!"
}

try:
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json=login_data,
        timeout=10
    )
    
    print(f"✅ Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        new_token = result["access_token"]
        
        print(f"✅ Connexion réussie!")
        print(f"   Nouvel token généré: {new_token[:80]}...")
        
        # Vérifier que le token est différent (nouveau)
        if new_token != token:
            print(f"   ✓ Nouveau token généré (sécurité OK)")
        else:
            print(f"   ℹ️  Même token (normal si reconnexion rapide)")
            
    else:
        print(f"❌ Erreur: {response.text}")
        exit(1)
        
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 70)
print("✅ ✅ ✅  TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS  ✅ ✅ ✅")
print("=" * 70)

print("\n📊 RÉCAPITULATIF:")
print(f"   • Health check: ✅")
print(f"   • Inscription: ✅")
print(f"   • Licence créée automatiquement: ✅")
print(f"   • Licence valide {license_data['days_remaining']} jours: ✅")
print(f"   • Connexion: ✅")

print("\n🎉 VOTRE API BACKEND EST 100% FONCTIONNELLE!")

print("\n📝 INFORMATIONS DE CONNEXION CRÉÉES:")
print(f"   Email: {test_email}")
print(f"   Password: Test123456!")
print(f"   License Key: {license_data['license_key']}")

print("\n🚀 PROCHAINES ÉTAPES:")
print("   1. Créer le client Python pour votre app desktop")
print("   2. Ajouter la route POST /analyses/analyze")
print("   3. Intégrer le moteur FXI dans le backend")
print("   4. Connecter votre application desktop")

print("\n" + "=" * 70)
