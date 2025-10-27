#!/usr/bin/env python3
"""
Test rapide de prédiction ML avec auto-training
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

async def test_predictions():
    print("=" * 80)
    print("🧪 TEST PRÉDICTIONS ML")
    print("=" * 80)

    ml_service = get_ml_signal_service()

    # Tickers à tester
    tickers = ['AAPL', 'TSLA', 'NVDA']

    for ticker in tickers:
        print(f"\n📊 {ticker}:")
        print("-" * 80)

        prediction = await ml_service.get_prediction(ticker)

        print(f"   Signal: {prediction.signal} (force: {prediction.signal_strength:.0f}%)")
        print(f"   1j: {prediction.prediction_1d} (conf: {prediction.confidence_1d:.0f}%)")
        print(f"   3j: {prediction.prediction_3d} (conf: {prediction.confidence_3d:.0f}%)")
        print(f"   7j: {prediction.prediction_7d} (conf: {prediction.confidence_7d:.0f}%)")
        print(f"   Model: {prediction.model_version}")
        print(f"   Généré: {prediction.generated_at}")

    print("\n" + "=" * 80)
    print("✅ Tests terminés")
    print("=" * 80)

if __name__ == '__main__':
    asyncio.run(test_predictions())
