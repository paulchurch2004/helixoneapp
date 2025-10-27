#!/usr/bin/env python3
"""
Script de test pour la collecte de données
"""

import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

# Login
print("🔐 Connexion...")
response = requests.post(
    f"{API_URL}/auth/login",
    json={"email": "datacollector@helixone.com", "password": "DataCollect@2025"}
)
token = response.json()["access_token"]
print(f"✅ Token obtenu: {token[:50]}...")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Test avec 1 seul symbol
print("\n📥 Test collecte AAPL (2024)...")
response = requests.post(
    f"{API_URL}/api/data/collect/daily",
    headers=headers,
    json={
        "symbols": ["AAPL"],
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-12-31T00:00:00",
        "adjusted": True
    }
)

if response.status_code == 200:
    job = response.json()
    job_id = job["id"]
    print(f"✅ Job créé: {job_id}")
    print(f"   Status: {job['status']}")
    print(f"   Nom: {job['job_name']}")

    # Suivre la progression
    print("\n⏳ Suivi de la progression...")
    for i in range(60):  # 60 secondes max
        time.sleep(2)

        status_response = requests.get(
            f"{API_URL}/api/data/jobs/{job_id}",
            headers=headers
        )

        if status_response.status_code == 200:
            status = status_response.json()
            progress = status.get("progress", 0)
            collected = status.get("records_collected", 0)
            failed = status.get("records_failed", 0)
            job_status = status.get("status", "")

            print(f"   [{i*2}s] Progress: {progress:.1f}% - Collected: {collected} - Failed: {failed} - Status: {job_status}")

            if job_status == "completed":
                print(f"\n✅ Collecte terminée!")
                print(f"   ✓ {collected} enregistrements collectés")
                print(f"   ✗ {failed} erreurs")
                break
            elif job_status == "failed":
                print(f"\n❌ Collecte échouée!")
                print(f"   Erreur: {status.get('error_message', 'Inconnue')}")
                break
else:
    print(f"❌ Erreur: {response.status_code}")
    print(response.text)
