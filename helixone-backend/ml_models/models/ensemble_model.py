"""
Ensemble Model - Combine XGBoost + LSTM pour meilleures prédictions

Stratégies d'ensemble:
1. Voting: Majorité simple (classification)
2. Weighted Average: Moyenne pondérée (régression)
3. Stacking: Meta-learner qui apprend à combiner

Objectif: >75% accuracy, MAPE <5%
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from .xgboost_classifier import XGBoostClassifier, MultiHorizonClassifier
from .lstm_predictor import LSTMPredictor, MultiHorizonLSTM

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Modèle d'ensemble combinant XGBoost (classification) et LSTM (régression)
    """

    def __init__(
        self,
        prediction_horizon: int = 1,
        xgb_weight: float = 0.5,
        lstm_weight: float = 0.5
    ):
        """
        Args:
            prediction_horizon: Nombre de jours à prédire
            xgb_weight: Poids du XGBoost dans l'ensemble
            lstm_weight: Poids du LSTM dans l'ensemble
        """
        self.prediction_horizon = prediction_horizon
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight

        # Normaliser les poids
        total_weight = xgb_weight + lstm_weight
        self.xgb_weight /= total_weight
        self.lstm_weight /= total_weight

        # Modèles
        self.xgb = XGBoostClassifier(prediction_horizon=prediction_horizon)
        self.lstm = LSTMPredictor(prediction_horizon=prediction_horizon)

        self.is_trained = False

        logger.info(f"EnsembleModel initialisé (horizon={prediction_horizon}j)")
        logger.info(f"   XGBoost weight: {self.xgb_weight:.2f}")
        logger.info(f"   LSTM weight: {self.lstm_weight:.2f}")

    def train(
        self,
        df: pd.DataFrame,
        features: List[str],
        train_split: float = 0.8,
        xgb_optimize: bool = True,
        xgb_trials: int = 30,
        lstm_epochs: int = 100
    ):
        """
        Entraîne les deux modèles

        Args:
            df: DataFrame complet avec features et 'close'
            features: Liste des colonnes features
            train_split: Ratio train/test
            xgb_optimize: Optimiser XGBoost hyperparams
            xgb_trials: Trials Optuna pour XGBoost
            lstm_epochs: Époques LSTM
        """
        logger.info("=" * 80)
        logger.info(f"ENTRAÎNEMENT ENSEMBLE MODEL (horizon={self.prediction_horizon}j)")
        logger.info("=" * 80)

        # Split train/test
        split_idx = int(len(df) * train_split)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]

        logger.info(f"\nDataset:")
        logger.info(f"   Total: {len(df)} jours")
        logger.info(f"   Train: {len(df_train)} jours")
        logger.info(f"   Test:  {len(df_test)} jours")
        logger.info(f"   Features: {len(features)}")

        # ========================================
        # 1. XGBOOST CLASSIFIER
        # ========================================
        logger.info(f"\n{'='*80}")
        logger.info("1. ENTRAÎNEMENT XGBOOST")
        logger.info(f"{'='*80}")

        # Préparer labels XGBoost
        y_train_xgb = self.xgb.prepare_labels(df_train)
        y_test_xgb = self.xgb.prepare_labels(df_test)

        valid_train_xgb = ~y_train_xgb.isna()
        valid_test_xgb = ~y_test_xgb.isna()

        X_train_xgb = df_train.loc[valid_train_xgb, features]
        y_train_xgb = y_train_xgb[valid_train_xgb]
        X_test_xgb = df_test.loc[valid_test_xgb, features]
        y_test_xgb = y_test_xgb[valid_test_xgb]

        # Entraîner XGBoost
        self.xgb.train(
            X_train_xgb,
            y_train_xgb,
            X_val=X_test_xgb,
            y_val=y_test_xgb,
            optimize=xgb_optimize,
            n_trials=xgb_trials
        )

        # ========================================
        # 2. LSTM PREDICTOR
        # ========================================
        logger.info(f"\n{'='*80}")
        logger.info("2. ENTRAÎNEMENT LSTM")
        logger.info(f"{'='*80}")

        # Préparer séquences LSTM
        X_lstm, y_lstm = self.lstm.prepare_sequences(df, features)

        # Split (les séquences couvrent déjà train+test)
        # Calculer le split_idx pour les séquences
        lstm_split_idx = int(len(X_lstm) * train_split)

        X_train_lstm = X_lstm[:lstm_split_idx]
        y_train_lstm = y_lstm[:lstm_split_idx]
        X_test_lstm = X_lstm[lstm_split_idx:]
        y_test_lstm = y_lstm[lstm_split_idx:]

        # Entraîner LSTM
        self.lstm.train(
            X_train_lstm,
            y_train_lstm,
            X_val=X_test_lstm,
            y_val=y_test_lstm,
            epochs=lstm_epochs
        )

        # ========================================
        # 3. ÉVALUATION ENSEMBLE
        # ========================================
        logger.info(f"\n{'='*80}")
        logger.info("3. ÉVALUATION ENSEMBLE")
        logger.info(f"{'='*80}")

        self.is_trained = True

        # Métriques XGBoost
        xgb_metrics = self.xgb.training_metrics
        logger.info(f"\nXGBoost:")
        logger.info(f"   Train Accuracy: {xgb_metrics['train_accuracy']:.4f}")
        if 'val_accuracy' in xgb_metrics:
            logger.info(f"   Val Accuracy:   {xgb_metrics['val_accuracy']:.4f}")

        # Métriques LSTM
        lstm_metrics = self.lstm.training_metrics
        logger.info(f"\nLSTM:")
        logger.info(f"   Train MAPE: {lstm_metrics['train_mape']:.2f}%")
        if 'val_mape' in lstm_metrics:
            logger.info(f"   Val MAPE:   {lstm_metrics['val_mape']:.2f}%")
        logger.info(f"   Train R²:   {lstm_metrics['train_r2']:.4f}")
        if 'val_r2' in lstm_metrics:
            logger.info(f"   Val R²:     {lstm_metrics['val_r2']:.4f}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        logger.info("=" * 80)

    def predict(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Dict:
        """
        Prédit avec l'ensemble (combine XGBoost + LSTM)

        Args:
            df: DataFrame avec historique récent
            features: Features

        Returns:
            Dict avec prédiction combinée
        """
        if not self.is_trained:
            raise ValueError("Modèles non entraînés! Appeler .train() d'abord.")

        # ========================================
        # 1. XGBOOST PREDICTION
        # ========================================
        xgb_signal = self.xgb.get_signal(df[features].tail(1))

        # Convertir prédiction XGBoost en changement de prix estimé
        # UP = +1.5%, FLAT = 0%, DOWN = -1.5%
        xgb_direction = xgb_signal['prediction']
        if xgb_direction == 'UP':
            xgb_price_change_pct = 1.5
        elif xgb_direction == 'DOWN':
            xgb_price_change_pct = -1.5
        else:
            xgb_price_change_pct = 0.0

        # ========================================
        # 2. LSTM PREDICTION
        # ========================================
        lstm_signal = self.lstm.get_prediction_signal(df, features)

        # ========================================
        # 3. ENSEMBLE (WEIGHTED AVERAGE)
        # ========================================
        current_price = df['close'].iloc[-1]

        # Combiner les changements de prix (weighted average)
        ensemble_price_change_pct = (
            self.xgb_weight * xgb_price_change_pct +
            self.lstm_weight * lstm_signal['price_change_pct']
        )

        ensemble_predicted_price = current_price * (1 + ensemble_price_change_pct / 100)
        ensemble_price_change = ensemble_predicted_price - current_price

        # Direction de l'ensemble
        if ensemble_price_change_pct > 1.0:
            ensemble_direction = 'UP'
            ensemble_action = 'BUY'
        elif ensemble_price_change_pct < -1.0:
            ensemble_direction = 'DOWN'
            ensemble_action = 'SELL'
        else:
            ensemble_direction = 'FLAT'
            ensemble_action = 'HOLD'

        # Confiance de l'ensemble
        # Si XGBoost et LSTM sont d'accord, confiance élevée
        xgb_dir = xgb_signal['prediction']
        lstm_dir = lstm_signal['direction']

        if xgb_dir == lstm_dir:
            ensemble_confidence = 0.85  # Accord parfait
        elif (xgb_dir in ['UP', 'DOWN'] and lstm_dir == 'FLAT') or \
             (lstm_dir in ['UP', 'DOWN'] and xgb_dir == 'FLAT'):
            ensemble_confidence = 0.65  # Accord partiel
        else:
            ensemble_confidence = 0.45  # Désaccord

        # Ajuster confiance avec la confiance XGBoost
        ensemble_confidence = (ensemble_confidence + xgb_signal['confidence']) / 2

        # ========================================
        # 4. SIGNAL FINAL
        # ========================================
        signal = {
            'current_price': float(current_price),
            'predicted_price': float(ensemble_predicted_price),
            'price_change': float(ensemble_price_change),
            'price_change_pct': float(ensemble_price_change_pct),
            'direction': ensemble_direction,
            'action': ensemble_action,
            'confidence': float(ensemble_confidence),
            'horizon_days': self.prediction_horizon,
            'model': 'ENSEMBLE',
            'components': {
                'xgb': {
                    'prediction': xgb_signal['prediction'],
                    'confidence': xgb_signal['confidence'],
                    'weight': self.xgb_weight
                },
                'lstm': {
                    'predicted_price': lstm_signal['predicted_price'],
                    'price_change_pct': lstm_signal['price_change_pct'],
                    'weight': self.lstm_weight
                }
            }
        }

        return signal

    def save(self, path: str):
        """
        Sauvegarde l'ensemble

        Args:
            path: Chemin du répertoire
        """
        if not self.is_trained:
            raise ValueError("Aucun modèle à sauvegarder!")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Sauvegarder XGBoost
        xgb_path = path / "xgb"
        self.xgb.save(str(xgb_path))

        # Sauvegarder LSTM
        lstm_path = path / "lstm"
        self.lstm.save(str(lstm_path))

        # Sauvegarder métadonnées ensemble
        metadata = {
            'prediction_horizon': self.prediction_horizon,
            'xgb_weight': self.xgb_weight,
            'lstm_weight': self.lstm_weight,
            'saved_at': datetime.now().isoformat()
        }

        metadata_path = path / "ensemble.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Ensemble sauvegardé: {path}")

    def load(self, path: str):
        """
        Charge un ensemble sauvegardé

        Args:
            path: Chemin du répertoire
        """
        path = Path(path)

        # Charger XGBoost
        xgb_path = path / "xgb"
        self.xgb.load(str(xgb_path))

        # Charger LSTM
        lstm_path = path / "lstm"
        self.lstm.load(str(lstm_path))

        # Charger métadonnées
        metadata_path = path / "ensemble.meta.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.prediction_horizon = metadata['prediction_horizon']
        self.xgb_weight = metadata['xgb_weight']
        self.lstm_weight = metadata['lstm_weight']

        self.is_trained = True

        logger.info(f"✅ Ensemble chargé: {path}")
        logger.info(f"   Sauvegardé le: {metadata['saved_at']}")


# ============================================================================
# MULTI-HORIZON ENSEMBLE
# ============================================================================

class MultiHorizonEnsemble:
    """
    Combine ensembles pour 1j, 3j, 7j
    """

    def __init__(
        self,
        xgb_weight: float = 0.5,
        lstm_weight: float = 0.5
    ):
        self.ensembles = {
            '1d': EnsembleModel(
                prediction_horizon=1,
                xgb_weight=xgb_weight,
                lstm_weight=lstm_weight
            ),
            '3d': EnsembleModel(
                prediction_horizon=3,
                xgb_weight=xgb_weight,
                lstm_weight=lstm_weight
            ),
            '7d': EnsembleModel(
                prediction_horizon=7,
                xgb_weight=xgb_weight,
                lstm_weight=lstm_weight
            )
        }

        logger.info("MultiHorizonEnsemble initialisé (1d, 3d, 7d)")

    def train_all(
        self,
        df: pd.DataFrame,
        features: List[str],
        train_split: float = 0.8,
        xgb_optimize: bool = True,
        xgb_trials: int = 30,
        lstm_epochs: int = 100
    ):
        """
        Entraîne tous les ensembles

        Args:
            df: DataFrame complet
            features: Features
            train_split: Ratio train/test
            xgb_optimize: Optimiser XGBoost
            xgb_trials: Trials Optuna
            lstm_epochs: Époques LSTM
        """
        logger.info("=" * 80)
        logger.info("ENTRAÎNEMENT MULTI-HORIZON ENSEMBLE")
        logger.info("=" * 80)

        for horizon, ensemble in self.ensembles.items():
            logger.info(f"\n{'#'*80}")
            logger.info(f"# HORIZON: {horizon}")
            logger.info(f"{'#'*80}")

            ensemble.train(
                df,
                features,
                train_split=train_split,
                xgb_optimize=xgb_optimize,
                xgb_trials=xgb_trials,
                lstm_epochs=lstm_epochs
            )

        logger.info("\n" + "=" * 80)
        logger.info("✅ TOUS LES ENSEMBLES ENTRAÎNÉS")
        logger.info("=" * 80)

    def get_multi_horizon_signals(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Dict:
        """
        Signaux pour tous les horizons

        Args:
            df: DataFrame avec historique récent
            features: Features

        Returns:
            Dict avec signaux 1d, 3d, 7d
        """
        signals = {}

        for horizon, ensemble in self.ensembles.items():
            signals[horizon] = ensemble.predict(df, features)

        # Consensus global
        actions = [s['action'] for s in signals.values()]
        directions = [s['direction'] for s in signals.values()]

        # Majorité
        action_counts = {a: actions.count(a) for a in set(actions)}
        direction_counts = {d: directions.count(d) for d in set(directions)}

        consensus_action = max(action_counts, key=action_counts.get)
        consensus_direction = max(direction_counts, key=direction_counts.get)

        # Score consensus (toutes identiques = 100%)
        consensus_score = (action_counts[consensus_action] / len(actions)) * 100

        # Confiance moyenne
        avg_confidence = np.mean([s['confidence'] for s in signals.values()])

        return {
            'signals': signals,
            'consensus': {
                'action': consensus_action,
                'direction': consensus_direction,
                'score': float(consensus_score),
                'confidence': float(avg_confidence)
            },
            'current_price': signals['1d']['current_price']
        }

    def save_all(self, base_path: str):
        """Sauvegarde tous les ensembles"""
        for horizon, ensemble in self.ensembles.items():
            path = Path(base_path) / f"ensemble_{horizon}"
            ensemble.save(str(path))

    def load_all(self, base_path: str):
        """Charge tous les ensembles"""
        for horizon, ensemble in self.ensembles.items():
            path = Path(base_path) / f"ensemble_{horizon}"
            ensemble.load(str(path))


# ============================================================================
# SINGLETON
# ============================================================================

_multi_horizon_ensemble = None

def get_multi_horizon_ensemble() -> MultiHorizonEnsemble:
    """Retourne instance singleton"""
    global _multi_horizon_ensemble
    if _multi_horizon_ensemble is None:
        _multi_horizon_ensemble = MultiHorizonEnsemble()
    return _multi_horizon_ensemble


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ENSEMBLE MODEL - Test")
    print("=" * 80)

    # Créer données synthétiques
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'volume': np.random.uniform(1e6, 5e6, len(dates)),
        'rsi_14': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.randn(len(dates)),
        'bb_width': np.random.uniform(0, 5, len(dates)),
        'sma_20': 100 + np.cumsum(np.random.randn(len(dates)) * 1.5)
    }, index=dates)

    features = ['volume', 'rsi_14', 'macd', 'bb_width', 'sma_20']

    print(f"\n📊 Dataset: {len(df)} jours, {len(features)} features")

    # Test Ensemble 1 jour
    print("\n" + "=" * 80)
    print("TEST: Ensemble 1 jour")
    print("=" * 80)

    ensemble = EnsembleModel(prediction_horizon=1)

    # Entraîner (paramètres réduits pour test rapide)
    ensemble.train(
        df,
        features,
        train_split=0.8,
        xgb_optimize=False,
        lstm_epochs=10
    )

    # Test prédiction
    signal = ensemble.predict(df, features)

    print(f"\n📈 SIGNAL ENSEMBLE:")
    print(f"   Prix actuel: ${signal['current_price']:.2f}")
    print(f"   Prix prédit: ${signal['predicted_price']:.2f}")
    print(f"   Change: {signal['price_change_pct']:+.2f}%")
    print(f"   Direction: {signal['direction']}")
    print(f"   Action: {signal['action']}")
    print(f"   Confiance: {signal['confidence']:.2%}")

    print(f"\n📊 Composants:")
    print(f"   XGBoost: {signal['components']['xgb']['prediction']} "
          f"(confiance: {signal['components']['xgb']['confidence']:.2%}, "
          f"poids: {signal['components']['xgb']['weight']:.2f})")
    print(f"   LSTM: ${signal['components']['lstm']['predicted_price']:.2f} "
          f"({signal['components']['lstm']['price_change_pct']:+.2f}%, "
          f"poids: {signal['components']['lstm']['weight']:.2f})")

    print("\n✅ Test terminé!")
