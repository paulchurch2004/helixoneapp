#!/usr/bin/env python3
"""
Test final : Tous les modèles ML réels (AAPL, MSFT, GOOGL)
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

async def test_all_models():
    print("=" * 80)
    print("🚀 TEST FINAL: TOUS LES MODÈLES ML RÉELS")
    print("=" * 80)

    # Obtenir le service
    ml_service = get_ml_signal_service()

    # Tester les 3 tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(f"📊 {ticker}")
        print(f"{'=' * 80}")

        prediction = await ml_service.get_prediction(ticker)

        if prediction:
            print(f"\n🎯 Signal: {prediction.signal} (force: {prediction.signal_strength:.0f}%)")
            print(f"   Model: {prediction.model_version}")
            print(f"\n   Horizons:")
            print(f"   1 jour:  {prediction.prediction_1d:5s} (confiance: {prediction.confidence_1d:5.1f}%) → {prediction.predicted_change_1d:+5.1f}%")
            print(f"   3 jours: {prediction.prediction_3d:5s} (confiance: {prediction.confidence_3d:5.1f}%) → {prediction.predicted_change_3d:+5.1f}%")
            print(f"   7 jours: {prediction.prediction_7d:5s} (confiance: {prediction.confidence_7d:5.1f}%) → {prediction.predicted_change_7d:+5.1f}%")

            # Moyenne des confiances
            avg_conf = (prediction.confidence_1d + prediction.confidence_3d + prediction.confidence_7d) / 3
            print(f"\n   Confiance moyenne: {avg_conf:.1f}%")
        else:
            print(f"   ❌ Pas de prédiction")

    # Test portfolio signals
    print(f"\n{'=' * 80}")
    print("📈 SIGNAUX PORTFOLIO")
    print(f"{'=' * 80}")

    signals = await ml_service.get_portfolio_signals(tickers)

    print(f"\n   Bullish (BUY):  {signals.bullish_count}")
    print(f"   Bearish (SELL): {signals.bearish_count}")
    print(f"   Neutral (HOLD): {signals.neutral_count}")
    print(f"   Confiance moy:  {signals.avg_confidence:.1f}%")

    if signals.top_buys:
        print(f"\n   🟢 Top BUY:  {', '.join(signals.top_buys)}")
    if signals.top_sells:
        print(f"   🔴 Top SELL: {', '.join(signals.top_sells)}")

    print(f"\n{'=' * 80}")
    print("✅ TEST TERMINÉ")
    print(f"{'=' * 80}\n")

if __name__ == '__main__':
    asyncio.run(test_all_models())
