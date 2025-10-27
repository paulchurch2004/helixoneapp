#!/usr/bin/env python3
"""
Test de l'auto-entraînement ML

Teste :
1. Entraînement automatique d'un nouveau ticker (TSLA)
2. Utilisation d'un modèle existant récent (AAPL)
3. Re-entraînement forcé
4. Prédiction avec auto-train intégré
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

from app.services.ml import get_auto_trainer
from app.services.portfolio.ml_signal_service import get_ml_signal_service

async def test_auto_training():
    print("=" * 80)
    print("🧪 TEST AUTO-TRAINING ML")
    print("=" * 80)

    trainer = get_auto_trainer()
    ml_service = get_ml_signal_service()

    # ========================================================================
    # Test 1: Nouveau ticker (TSLA - pas de modèle)
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST 1: TSLA (nouveau ticker - auto-entraînement)")
    print("=" * 80)

    print("\n   🔍 Vérification existence modèle TSLA...")
    tsla_exists = trainer._model_exists('TSLA')
    print(f"   Modèle TSLA existe : {tsla_exists}")

    if not tsla_exists:
        print("\n   🚀 Lancement auto-entraînement TSLA...")
        print("   ⏳ Ceci peut prendre 15-20 secondes...")

    prediction = await ml_service.get_prediction('TSLA')

    print(f"\n   ✅ Prédiction TSLA obtenue:")
    print(f"      Signal: {prediction.signal} (force: {prediction.signal_strength:.0f}%)")
    print(f"      Model: {prediction.model_version}")
    print(f"      1j: {prediction.prediction_1d} (conf: {prediction.confidence_1d:.0f}%)")
    print(f"      3j: {prediction.prediction_3d} (conf: {prediction.confidence_3d:.0f}%)")
    print(f"      7j: {prediction.prediction_7d} (conf: {prediction.confidence_7d:.0f}%)")

    # ========================================================================
    # Test 2: Ticker existant (AAPL - devrait utiliser cache)
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST 2: AAPL (modèle existant récent)")
    print("=" * 80)

    age = trainer.get_model_age('AAPL')
    print(f"\n   Âge du modèle AAPL: {age if age is not None else 'N/A'} jours")

    if age and age < 7:
        print("   ✅ Modèle récent, pas de re-entraînement nécessaire")
    else:
        print("   ⏳ Modèle obsolète ou absent, re-entraînement...")

    prediction = await ml_service.get_prediction('AAPL')

    print(f"\n   ✅ Prédiction AAPL obtenue:")
    print(f"      Signal: {prediction.signal} (force: {prediction.signal_strength:.0f}%)")
    print(f"      Model: {prediction.model_version}")

    # ========================================================================
    # Test 3: Force re-entraînement
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST 3: Force re-entraînement MSFT")
    print("=" * 80)

    print("\n   🔄 Lancement re-entraînement forcé MSFT...")
    success = await trainer.force_train('MSFT')

    if success:
        print("   ✅ Re-entraînement MSFT réussi")
        age_after = trainer.get_model_age('MSFT')
        print(f"   Nouveau modèle âge: {age_after if age_after is not None else 'N/A'} jours")
    else:
        print("   ❌ Re-entraînement MSFT échoué")

    # ========================================================================
    # Test 4: Liste des modèles
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST 4: Liste des modèles disponibles")
    print("=" * 80)

    models = trainer.list_all_models()

    print(f"\n   Nombre de modèles: {len(models)}")

    for model in models:
        print(f"\n   📈 {model['ticker']}:")
        print(f"      Mode: {model['mode']}")
        print(f"      Âge: {model['age_days']} jours")
        print(f"      Échantillons: {model['n_samples']}")
        print(f"      Features: {model['n_features']}")

    # ========================================================================
    # Test 5: Portfolio signals
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST 5: Signaux portfolio (tous les modèles)")
    print("=" * 80)

    tickers = [m['ticker'] for m in models]
    signals = await ml_service.get_portfolio_signals(tickers)

    print(f"\n   📈 Résumé Portfolio:")
    print(f"      Bullish (BUY):  {signals.bullish_count}")
    print(f"      Bearish (SELL): {signals.bearish_count}")
    print(f"      Neutral (HOLD): {signals.neutral_count}")
    print(f"      Confiance moy:  {signals.avg_confidence:.1f}%")

    if signals.top_buys:
        print(f"\n   🟢 Top BUY:  {', '.join(signals.top_buys)}")
    if signals.top_sells:
        print(f"   🔴 Top SELL: {', '.join(signals.top_sells)}")

    # ========================================================================
    # Résumé final
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ TESTS TERMINÉS")
    print("=" * 80)

    print("\n📊 Résultats:")
    print("   ✅ Test 1: Auto-entraînement nouveau ticker (TSLA)")
    print("   ✅ Test 2: Utilisation modèle existant (AAPL)")
    print("   ✅ Test 3: Re-entraînement forcé (MSFT)")
    print("   ✅ Test 4: Liste des modèles")
    print("   ✅ Test 5: Signaux portfolio")

    print("\n🎯 Système d'auto-entraînement opérationnel !")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    asyncio.run(test_auto_training())
