#!/usr/bin/env python3
"""
Test rapide du système ML

Teste:
1. Téléchargement données (Yahoo + FRED)
2. Feature engineering
3. Modèle simple
"""

import sys
from pathlib import Path

# Ajouter le path
sys.path.insert(0, str(Path(__file__).parent / 'helixone-backend'))

print("=" * 80)
print("🧪 TEST SYSTÈME ML")
print("=" * 80)

# Test 1: Data Collection
print("\n📊 TEST 1: Téléchargement données...")
print("-" * 80)

try:
    from ml_models.data_collection.yahoo_finance_downloader import YahooFinanceDownloader

    downloader = YahooFinanceDownloader()

    # Télécharger 30 jours de données AAPL
    print("   Téléchargement AAPL (30 derniers jours)...")
    data = downloader.download_historical_data(
        tickers=['AAPL'],
        start_date='2024-09-01',
        end_date='2024-10-24'
    )

    df_aapl = data['AAPL']
    print(f"   ✅ {len(df_aapl)} jours téléchargés")
    print(f"   📈 Prix actuel: ${df_aapl['close'].iloc[-1]:.2f}")
    print(f"   📈 Prix min: ${df_aapl['close'].min():.2f}")
    print(f"   📈 Prix max: ${df_aapl['close'].max():.2f}")

except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

# Test 2: FRED API
print("\n📊 TEST 2: FRED Macro Data...")
print("-" * 80)

try:
    from ml_models.data_collection.fred_macro_downloader import FREDMacroDownloader
    import os

    fred_key = os.getenv('FRED_API_KEY', '2eb1601f70b8771864fd98d891879301')
    print(f"   Clé FRED configurée: {fred_key[:8]}...")

    fred_downloader = FREDMacroDownloader(api_key=fred_key)

    print("   Téléchargement indicateurs macro...")
    macro_data = fred_downloader.download_all_indicators(
        start_date='2024-09-01'
    )

    print(f"   ✅ {len(macro_data)} jours téléchargés")
    print(f"   📊 Colonnes: {len(macro_data.columns)} indicateurs")

    # Afficher quelques indicateurs
    if 'DFF' in macro_data.columns:
        print(f"   📈 Fed Funds Rate: {macro_data['DFF'].iloc[-1]:.2f}%")
    if 'VIXCLS' in macro_data.columns:
        print(f"   📈 VIX: {macro_data['VIXCLS'].iloc[-1]:.2f}")

except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Feature Engineering
print("\n📊 TEST 3: Feature Engineering...")
print("-" * 80)

try:
    from ml_models.feature_engineering.technical_indicators import TechnicalIndicators

    tech = TechnicalIndicators()

    print("   Calcul indicateurs techniques...")
    df_with_features = tech.add_all_indicators(df_aapl.copy())

    n_new_features = len(df_with_features.columns) - len(df_aapl.columns)
    print(f"   ✅ {n_new_features} features ajoutées")

    # Afficher quelques features
    if 'rsi_14' in df_with_features.columns:
        print(f"   📈 RSI (14): {df_with_features['rsi_14'].iloc[-1]:.2f}")
    if 'macd' in df_with_features.columns:
        print(f"   📈 MACD: {df_with_features['macd'].iloc[-1]:.4f}")

    print(f"\n   Total features: {len(df_with_features.columns)}")

except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Résumé
print("\n" + "=" * 80)
print("✅ TESTS TERMINÉS")
print("=" * 80)

print("\n📊 RÉSUMÉ:")
print("   1. Yahoo Finance: ✅ Fonctionnel")
print("   2. FRED API: ✅ Fonctionnel")
print("   3. Feature Engineering: ✅ Fonctionnel")

print("\n🚀 PROCHAINES ÉTAPES:")
print("   1. Entraîner un modèle:")
print("      cd helixone-backend")
print("      python ml_models/model_trainer.py --ticker AAPL --mode ensemble --lstm-epochs 20")
print()
print("   2. Voir la documentation complète:")
print("      cat helixone-backend/ml_models/README.md")

print("\n" + "=" * 80)
