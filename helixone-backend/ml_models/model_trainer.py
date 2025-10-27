"""
Model Trainer - Script principal d'entraînement ML

Utilisation:
1. Google Colab (GPU gratuit):
   - Uploader ce fichier et ml_models/ sur Colab
   - Installer requirements: !pip install -r requirements_ml.txt
   - Exécuter: python model_trainer.py --ticker AAPL --mode ensemble

2. Local:
   - python model_trainer.py --ticker AAPL --mode ensemble --epochs 50

Modes:
- 'xgboost': XGBoost uniquement
- 'lstm': LSTM uniquement
- 'ensemble': Les deux (recommandé)
- 'all': Entraîner 10 tickers simultanément

Output:
- Modèles sauvegardés dans ml_models/saved_models/
- Métriques et graphiques dans ml_models/results/
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import numpy as np

# Ajouter le path parent pour imports
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.data_cache import DataCache
from feature_engineering.technical_indicators import TechnicalIndicators
from feature_engineering.macro_features import MacroFeatures
from feature_engineering.sentiment_features import SentimentFeatures
from feature_engineering.volume_features import VolumeFeatures
from feature_engineering.feature_selector import FeatureSelector
from models.xgboost_classifier import MultiHorizonClassifier
from models.lstm_predictor import MultiHorizonLSTM
from models.ensemble_model import MultiHorizonEnsemble

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrateur d'entraînement des modèles ML
    """

    def __init__(
        self,
        output_dir: str = 'ml_models/saved_models',
        results_dir: str = 'ml_models/results'
    ):
        """
        Args:
            output_dir: Répertoire pour sauvegarder les modèles
            results_dir: Répertoire pour les résultats/métriques
        """
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)

        # Créer répertoires
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Composants
        self.data_cache = DataCache()
        self.tech_indicators = TechnicalIndicators()
        self.macro_features = MacroFeatures()
        self.sentiment_features = SentimentFeatures()
        self.volume_features = VolumeFeatures()
        self.feature_selector = FeatureSelector(max_features=50)

        logger.info("ModelTrainer initialisé")
        logger.info(f"   Modèles: {self.output_dir}")
        logger.info(f"   Résultats: {self.results_dir}")

    def prepare_dataset(
        self,
        ticker: str,
        start_date: str = '2018-01-01',
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Prépare le dataset complet avec toutes les features

        Args:
            ticker: Ticker de l'action
            start_date: Date de début
            end_date: Date de fin (None = aujourd'hui)

        Returns:
            DataFrame avec toutes les features
        """
        logger.info(f"="*80)
        logger.info(f"PRÉPARATION DATASET: {ticker}")
        logger.info(f"="*80)

        # 1. Télécharger données brutes
        logger.info("\n1️⃣  Téléchargement données...")
        dataset = self.data_cache.get_ml_dataset(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            include_macro=True,
            include_sentiment=False  # Pour l'instant synthétique
        )

        df = dataset[ticker]
        logger.info(f"   ✅ {len(df)} jours téléchargés")

        # 2. Features techniques
        logger.info("\n2️⃣  Calcul features techniques...")
        df = self.tech_indicators.add_all_indicators(df)
        logger.info(f"   ✅ {len(df.columns)} colonnes totales")

        # 3. Features macro (si disponibles)
        logger.info("\n3️⃣  Ajout features macro...")
        if 'macro_data' in dataset:
            macro_df = dataset['macro_data']
            df = self.macro_features.add_macro_features(macro_df)
            # Merge avec prix
            df = df.merge(macro_df, left_index=True, right_index=True, how='left')
        logger.info(f"   ✅ {len(df.columns)} colonnes avec macro")

        # 4. Features volume
        logger.info("\n4️⃣  Features volume...")
        df = self.volume_features.add_volume_features(df)
        logger.info(f"   ✅ {len(df.columns)} colonnes avec volume")

        # 5. Features sentiment (synthétiques pour test)
        logger.info("\n5️⃣  Features sentiment...")
        df = self.sentiment_features.add_sentiment_features(df, ticker)
        logger.info(f"   ✅ {len(df.columns)} colonnes totales")

        # 6. Nettoyage
        logger.info("\n6️⃣  Nettoyage...")
        initial_rows = len(df)

        # Supprimer lignes où les colonnes OHLCV critiques sont NaN
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=critical_cols)

        # Pour les autres colonnes (indicateurs, macro), remplir les NaN
        # Forward fill d'abord (pour indicateurs récents)
        df = df.fillna(method='ffill')
        # Puis backward fill (pour début de série)
        df = df.fillna(method='bfill')
        # Enfin, mettre 0 pour les colonnes restantes avec NaN
        df = df.fillna(0)

        logger.info(f"   ✅ {len(df)} jours valides (supprimé {initial_rows - len(df)} lignes)")

        # Sauvegarder
        dataset_path = self.results_dir / f"{ticker}_dataset.csv"
        df.to_csv(dataset_path)
        logger.info(f"\n💾 Dataset sauvegardé: {dataset_path}")

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        method: str = 'xgboost'
    ) -> List[str]:
        """
        Sélectionne les meilleures features

        Args:
            df: DataFrame avec toutes les features
            method: Méthode de sélection ('xgboost', 'rf', 'rfe')

        Returns:
            Liste des features sélectionnées
        """
        logger.info(f"\n{'='*80}")
        logger.info("SÉLECTION DES FEATURES")
        logger.info(f"{'='*80}")

        # Features à exclure
        exclude = ['open', 'high', 'low', 'close', 'volume', 'adj_close']

        # Features candidates
        all_features = [col for col in df.columns if col not in exclude]

        logger.info(f"\n   Features candidates: {len(all_features)}")

        # Préparer target (UP/DOWN/FLAT à 1 jour)
        future_return = df['close'].pct_change(periods=1).shift(-1)
        y = pd.Series(1, index=df.index)  # FLAT par défaut
        y[future_return > 0.01] = 2  # UP
        y[future_return < -0.01] = 0  # DOWN

        # Supprimer NaN
        valid = ~y.isna()
        X = df.loc[valid, all_features]
        y = y[valid]

        # Sélection
        selected = self.feature_selector.select_features(X, y, method=method)

        logger.info(f"\n✅ {len(selected)} features sélectionnées:")
        for i, feat in enumerate(selected[:20], 1):
            importance = self.feature_selector.feature_importances.get(feat, 0)
            logger.info(f"   {i:2d}. {feat:30s} {importance:.4f}")

        if len(selected) > 20:
            logger.info(f"   ... ({len(selected) - 20} autres)")

        return selected

    def train_xgboost(
        self,
        df: pd.DataFrame,
        features: List[str],
        ticker: str,
        optimize: bool = True,
        n_trials: int = 30
    ):
        """
        Entraîne les modèles XGBoost multi-horizon

        Args:
            df: DataFrame avec features
            features: Liste des features
            ticker: Ticker
            optimize: Optimiser hyperparamètres
            n_trials: Trials Optuna
        """
        logger.info(f"\n{'='*80}")
        logger.info("ENTRAÎNEMENT XGBOOST")
        logger.info(f"{'='*80}")

        # Créer classifier
        clf = MultiHorizonClassifier()

        # Entraîner
        clf.train_all(
            df=df,
            features=features,
            train_split=0.8,
            optimize=optimize,
            n_trials=n_trials
        )

        # Sauvegarder
        model_path = self.output_dir / ticker / 'xgboost'
        clf.save_all(str(model_path))

        logger.info(f"\n✅ XGBoost sauvegardé: {model_path}")

        return clf

    def train_lstm(
        self,
        df: pd.DataFrame,
        features: List[str],
        ticker: str,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Entraîne les modèles LSTM multi-horizon

        Args:
            df: DataFrame avec features
            features: Liste des features
            ticker: Ticker
            epochs: Nombre d'époques
            batch_size: Taille de batch
        """
        logger.info(f"\n{'='*80}")
        logger.info("ENTRAÎNEMENT LSTM")
        logger.info(f"{'='*80}")

        # Créer LSTM
        lstm = MultiHorizonLSTM(lookback_window=30, lstm_units=[64, 32])

        # Entraîner
        lstm.train_all(
            df=df,
            features=features,
            train_split=0.8,
            epochs=epochs,
            batch_size=batch_size
        )

        # Sauvegarder
        model_path = self.output_dir / ticker / 'lstm'
        lstm.save_all(str(model_path))

        logger.info(f"\n✅ LSTM sauvegardé: {model_path}")

        return lstm

    def train_ensemble(
        self,
        df: pd.DataFrame,
        features: List[str],
        ticker: str,
        xgb_optimize: bool = True,
        xgb_trials: int = 30,
        lstm_epochs: int = 100
    ):
        """
        Entraîne les modèles ensemble multi-horizon

        Args:
            df: DataFrame avec features
            features: Liste des features
            ticker: Ticker
            xgb_optimize: Optimiser XGBoost
            xgb_trials: Trials Optuna
            lstm_epochs: Époques LSTM
        """
        logger.info(f"\n{'='*80}")
        logger.info("ENTRAÎNEMENT ENSEMBLE")
        logger.info(f"{'='*80}")

        # Créer ensemble
        ensemble = MultiHorizonEnsemble(xgb_weight=0.5, lstm_weight=0.5)

        # Entraîner
        ensemble.train_all(
            df=df,
            features=features,
            train_split=0.8,
            xgb_optimize=xgb_optimize,
            xgb_trials=xgb_trials,
            lstm_epochs=lstm_epochs
        )

        # Sauvegarder
        model_path = self.output_dir / ticker / 'ensemble'
        ensemble.save_all(str(model_path))

        logger.info(f"\n✅ Ensemble sauvegardé: {model_path}")

        return ensemble

    def train_single_ticker(
        self,
        ticker: str,
        mode: str = 'ensemble',
        start_date: str = '2018-01-01',
        xgb_optimize: bool = True,
        xgb_trials: int = 30,
        lstm_epochs: int = 100
    ):
        """
        Entraîne les modèles pour un seul ticker

        Args:
            ticker: Ticker à entraîner
            mode: 'xgboost', 'lstm', ou 'ensemble'
            start_date: Date de début
            xgb_optimize: Optimiser XGBoost
            xgb_trials: Trials Optuna
            lstm_epochs: Époques LSTM
        """
        logger.info(f"\n{'#'*80}")
        logger.info(f"# ENTRAÎNEMENT: {ticker}")
        logger.info(f"# Mode: {mode.upper()}")
        logger.info(f"{'#'*80}")

        start_time = datetime.now()

        # 1. Préparer dataset
        df = self.prepare_dataset(ticker, start_date=start_date)

        # 2. Sélectionner features
        features = self.select_features(df, method='xgboost')

        # 3. Entraîner selon mode
        if mode == 'xgboost':
            model = self.train_xgboost(df, features, ticker, xgb_optimize, xgb_trials)
        elif mode == 'lstm':
            model = self.train_lstm(df, features, ticker, lstm_epochs)
        elif mode == 'ensemble':
            model = self.train_ensemble(df, features, ticker, xgb_optimize, xgb_trials, lstm_epochs)
        else:
            raise ValueError(f"Mode invalide: {mode}")

        # 4. Sauvegarder métadonnées
        elapsed = (datetime.now() - start_time).total_seconds()

        metadata = {
            'ticker': ticker,
            'mode': mode,
            'start_date': start_date,
            'n_samples': len(df),
            'n_features': len(features),
            'features': features,
            'training_time_seconds': elapsed,
            'trained_at': datetime.now().isoformat()
        }

        metadata_path = self.output_dir / ticker / 'training_metadata.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ ENTRAÎNEMENT TERMINÉ: {ticker}")
        logger.info(f"{'='*80}")
        logger.info(f"   Temps: {elapsed/60:.1f} minutes")
        logger.info(f"   Samples: {len(df)}")
        logger.info(f"   Features: {len(features)}")
        logger.info(f"   Modèles: {self.output_dir / ticker}")

    def train_multiple_tickers(
        self,
        tickers: List[str],
        mode: str = 'ensemble',
        start_date: str = '2018-01-01',
        xgb_optimize: bool = True,
        xgb_trials: int = 20,
        lstm_epochs: int = 50
    ):
        """
        Entraîne les modèles pour plusieurs tickers

        Args:
            tickers: Liste de tickers
            mode: Mode d'entraînement
            start_date: Date de début
            xgb_optimize: Optimiser XGBoost
            xgb_trials: Trials Optuna (réduit pour multiple)
            lstm_epochs: Époques LSTM (réduit pour multiple)
        """
        logger.info(f"\n{'#'*80}")
        logger.info(f"# ENTRAÎNEMENT MULTIPLE")
        logger.info(f"# Tickers: {', '.join(tickers)}")
        logger.info(f"# Mode: {mode.upper()}")
        logger.info(f"{'#'*80}")

        start_time = datetime.now()
        results = {}

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n\n{'='*80}")
            logger.info(f"TICKER {i}/{len(tickers)}: {ticker}")
            logger.info(f"{'='*80}")

            try:
                self.train_single_ticker(
                    ticker=ticker,
                    mode=mode,
                    start_date=start_date,
                    xgb_optimize=xgb_optimize,
                    xgb_trials=xgb_trials,
                    lstm_epochs=lstm_epochs
                )
                results[ticker] = 'SUCCESS'

            except Exception as e:
                logger.error(f"❌ Erreur pour {ticker}: {e}")
                results[ticker] = f'ERROR: {str(e)}'

        # Résumé
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"\n\n{'='*80}")
        logger.info(f"✅ ENTRAÎNEMENT MULTIPLE TERMINÉ")
        logger.info(f"{'='*80}")
        logger.info(f"   Temps total: {elapsed/60:.1f} minutes")
        logger.info(f"   Succès: {sum(1 for r in results.values() if r == 'SUCCESS')}/{len(tickers)}")

        logger.info(f"\n📊 RÉSULTATS:")
        for ticker, status in results.items():
            emoji = "✅" if status == "SUCCESS" else "❌"
            logger.info(f"   {emoji} {ticker}: {status}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Entraîneur de modèles ML pour trading')

    parser.add_argument('--ticker', type=str, help='Ticker à entraîner (ex: AAPL)')
    parser.add_argument('--tickers', type=str, help='Liste de tickers séparés par des virgules (ex: AAPL,MSFT,GOOGL)')
    parser.add_argument('--mode', type=str, default='ensemble', choices=['xgboost', 'lstm', 'ensemble'],
                        help='Mode d\'entraînement')
    parser.add_argument('--start-date', type=str, default='2018-01-01', help='Date de début')
    parser.add_argument('--no-optimize', action='store_true', help='Désactiver l\'optimisation XGBoost')
    parser.add_argument('--xgb-trials', type=int, default=30, help='Nombre de trials Optuna pour XGBoost')
    parser.add_argument('--lstm-epochs', type=int, default=100, help='Nombre d\'époques LSTM')
    parser.add_argument('--output-dir', type=str, default='ml_models/saved_models', help='Répertoire de sortie')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING'],
                        help='Niveau de logging')

    args = parser.parse_args()

    # Configuration logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Créer trainer
    trainer = ModelTrainer(output_dir=args.output_dir)

    # Entraîner
    if args.tickers:
        # Multiple tickers
        tickers = [t.strip() for t in args.tickers.split(',')]
        trainer.train_multiple_tickers(
            tickers=tickers,
            mode=args.mode,
            start_date=args.start_date,
            xgb_optimize=not args.no_optimize,
            xgb_trials=args.xgb_trials,
            lstm_epochs=args.lstm_epochs
        )

    elif args.ticker:
        # Single ticker
        trainer.train_single_ticker(
            ticker=args.ticker,
            mode=args.mode,
            start_date=args.start_date,
            xgb_optimize=not args.no_optimize,
            xgb_trials=args.xgb_trials,
            lstm_epochs=args.lstm_epochs
        )

    else:
        # Par défaut: Top 10 tech stocks
        default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'INTC']
        logger.info(f"Aucun ticker spécifié, entraînement des 10 tech stocks par défaut")

        trainer.train_multiple_tickers(
            tickers=default_tickers,
            mode=args.mode,
            start_date=args.start_date,
            xgb_optimize=not args.no_optimize,
            xgb_trials=20,  # Réduit pour multiple
            lstm_epochs=50  # Réduit pour multiple
        )


if __name__ == '__main__':
    main()
