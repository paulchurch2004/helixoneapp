#!/usr/bin/env python3
"""
Script de test pour vérifier que l'analyse complète fonctionne
"""

import sys
sys.path.insert(0, '/Users/macintosh/Desktop/helixone')

from helixone_client import HelixOneClient
from auth_session import get_auth_token

def test_deep_analysis():
    """Test l'endpoint deep_analyze"""

    print("=" * 70)
    print("🧪 TEST DE L'ANALYSE COMPLÈTE")
    print("=" * 70)

    # 1. Créer le client
    print("\n1️⃣  Initialisation du client...")
    client = HelixOneClient()

    # 2. Authentification
    print("2️⃣  Récupération du token...")
    token = get_auth_token()
    if not token:
        print("❌ Pas de token. Veuillez vous connecter dans l'interface d'abord.")
        return

    client.token = token
    print(f"✅ Token récupéré: {token[:20]}...")

    # 3. Vérifier que la méthode existe
    print("\n3️⃣  Vérification de la méthode deep_analyze()...")
    if not hasattr(client, 'deep_analyze'):
        print("❌ La méthode deep_analyze() n'existe pas!")
        print("   -> Vérifiez que helixone_client.py est à jour")
        return
    print("✅ Méthode deep_analyze() trouvée")

    # 4. Test avec AAPL
    ticker = "AAPL"
    print(f"\n4️⃣  Appel de l'analyse complète pour {ticker}...")
    print("   (Cela peut prendre 5-10 secondes...)")

    try:
        result = client.deep_analyze(ticker)
        print(f"✅ Analyse complète reçue!")

        # 5. Vérifier la structure de la réponse
        print("\n5️⃣  Vérification de la structure de la réponse...")

        expected_keys = [
            'ticker',
            'data_collection',
            'sentiment_analysis',
            'position_analysis',
            'ml_predictions',
            'recommendation',
            'alerts',
            'upcoming_events',
            'executive_summary'
        ]

        missing_keys = [key for key in expected_keys if key not in result]

        if missing_keys:
            print(f"⚠️  Clés manquantes: {missing_keys}")
        else:
            print("✅ Toutes les clés sont présentes!")

        # 6. Afficher un résumé
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DE L'ANALYSE")
        print("=" * 70)

        print(f"\n📈 Ticker: {result.get('ticker', 'N/A')}")

        # Position Analysis
        position = result.get('position_analysis', {})
        health_score = position.get('health_score', 'N/A')
        print(f"💚 Health Score: {health_score}/100")

        # Recommandation
        recommendation = result.get('recommendation', {})
        action = recommendation.get('action', 'N/A')
        confidence = recommendation.get('confidence', 'N/A')
        print(f"🎯 Recommandation: {action} (Confiance: {confidence}%)")

        # ML Predictions
        ml_preds = result.get('ml_predictions', {})
        signal = ml_preds.get('signal', 'N/A')
        print(f"🧠 Signal ML: {signal}")

        pred_1d = ml_preds.get('prediction_1d', {})
        if isinstance(pred_1d, dict):
            print(f"   1j: {pred_1d.get('direction', 'N/A')} ({pred_1d.get('confidence', 0):.0f}%)")

        pred_3d = ml_preds.get('prediction_3d', {})
        if isinstance(pred_3d, dict):
            print(f"   3j: {pred_3d.get('direction', 'N/A')} ({pred_3d.get('confidence', 0):.0f}%)")

        pred_7d = ml_preds.get('prediction_7d', {})
        if isinstance(pred_7d, dict):
            print(f"   7j: {pred_7d.get('direction', 'N/A')} ({pred_7d.get('confidence', 0):.0f}%)")

        # Sentiment
        sentiment = result.get('sentiment_analysis', {})
        sentiment_score = sentiment.get('sentiment_score', 'N/A')
        trend = sentiment.get('trend', 'N/A')
        print(f"💭 Sentiment: {sentiment_score}/100 (Trend: {trend})")

        # Alertes
        alerts = result.get('alerts', {})
        critical = len(alerts.get('critical', []))
        important = len(alerts.get('important', []))
        opportunity = len(alerts.get('opportunity', []))
        info = len(alerts.get('info', []))
        print(f"🚨 Alertes:")
        print(f"   🔴 Critiques: {critical}")
        print(f"   🟠 Importantes: {important}")
        print(f"   🟢 Opportunités: {opportunity}")
        print(f"   ℹ️  Info: {info}")

        # Events
        events = result.get('upcoming_events', [])
        print(f"📅 Événements à venir: {len(events)}")

        # Executive Summary
        summary = result.get('executive_summary', '')
        if summary:
            print(f"\n📋 Résumé Exécutif:")
            print(f"   {summary[:200]}...")

        # Data Collection
        data_collection = result.get('data_collection', {})
        sources_count = data_collection.get('sources_count', 0)
        print(f"\n📡 Sources utilisées: {sources_count}")

        print("\n" + "=" * 70)
        print("✅ TEST RÉUSSI!")
        print("=" * 70)
        print("\n💡 L'analyse complète fonctionne correctement!")
        print("   Vous pouvez maintenant l'utiliser dans l'interface.")

    except Exception as e:
        print(f"\n❌ ERREUR lors de l'appel de l'analyse:")
        print(f"   {str(e)}")
        print("\n🔍 Vérifiez que:")
        print("   - Le backend est lancé (http://localhost:8000)")
        print("   - L'endpoint /stock-deep-analysis existe (/docs)")
        print("   - Vous êtes authentifié")
        return

if __name__ == "__main__":
    test_deep_analysis()
