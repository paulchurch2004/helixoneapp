#!/usr/bin/env python3
"""
Test du chargement d'un vrai modèle ML
"""
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Charger .env
env_path = Path(__file__).parent / 'helixone-backend' / '.env'
load_dotenv(env_path)

# Ajouter le backend au path
sys.path.insert(0, str(Path(__file__).parent / 'helixone-backend'))

from app.services.portfolio.ml_signal_service import get_ml_signal_service

async def test_real_model():
    print("=" * 80)
    print("🧪 TEST DU VRAI MODÈLE ML - AAPL")
    print("=" * 80)

    # Obtenir le service
    ml_service = get_ml_signal_service()

    # Tester prédiction AAPL
    print("\n📊 Test prédiction AAPL...")
    prediction = await ml_service.get_prediction('AAPL')

    if prediction:
        print(f"\n✅ PRÉDICTION RÉUSSIE pour AAPL:")
        print(f"   Prix actuel: ${prediction.current_price:.2f}")
        print(f"   Signal: {prediction.signal} (force: {prediction.signal_strength:.0f}%)")
        print(f"\n   Prédictions:")
        print(f"   - 1 jour:  {prediction.prediction_1d} (confiance: {prediction.confidence_1d:.0f}%)")
        if prediction.predicted_price_1d:
            print(f"              → ${prediction.predicted_price_1d:.2f} ({prediction.predicted_change_1d:+.1f}%)")
        print(f"   - 3 jours: {prediction.prediction_3d} (confiance: {prediction.confidence_3d:.0f}%)")
        if prediction.predicted_price_3d:
            print(f"              → ${prediction.predicted_price_3d:.2f} ({prediction.predicted_change_3d:+.1f}%)")
        print(f"   - 7 jours: {prediction.prediction_7d} (confiance: {prediction.confidence_7d:.0f}%)")
        if prediction.predicted_price_7d:
            print(f"              → ${prediction.predicted_price_7d:.2f} ({prediction.predicted_change_7d:+.1f}%)")

        # Vérifier si c'est un vrai modèle ou default
        if prediction.confidence_1d == 50.0 and prediction.confidence_3d == 50.0:
            print("\n⚠️  ATTENTION: Ce sont des prédictions par défaut (pas de modèle chargé)")
        else:
            print("\n🎉 SUCCÈS: Vrai modèle chargé et utilisé!")
    else:
        print("\n❌ Échec: Aucune prédiction retournée")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    asyncio.run(test_real_model())
