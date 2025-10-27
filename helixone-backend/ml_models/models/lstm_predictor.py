"""
LSTM Predictor - Prédiction de prix avec réseaux de neurones récurrents

Objectifs:
- Prédire le prix futur (1j, 3j, 7j)
- Capture les patterns temporels
- Séquences de 30-60 jours pour prédiction
- Utilise TensorFlow/Keras

Architecture:
- Input: Séquence de features (lookback_window jours)
- LSTM layers avec dropout
- Dense output pour prédiction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """
    Prédicteur LSTM pour prix futurs
    """

    def __init__(
        self,
        prediction_horizon: int = 1,
        lookback_window: int = 30,
        lstm_units: List[int] = [64, 32],
        dropout: float = 0.2
    ):
        """
        Args:
            prediction_horizon: Nombre de jours à prédire
            lookback_window: Nombre de jours historiques à utiliser
            lstm_units: Unités pour chaque couche LSTM
            dropout: Taux de dropout
        """
        self.prediction_horizon = prediction_horizon
        self.lookback_window = lookback_window
        self.lstm_units = lstm_units
        self.dropout = dropout

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_metrics = {}

        logger.info(f"LSTM Predictor initialisé (horizon={prediction_horizon}j, lookback={lookback_window}j)")

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée les séquences temporelles pour LSTM

        Args:
            df: DataFrame avec features et 'close'
            features: Liste des colonnes features

        Returns:
            X: Array (samples, lookback_window, n_features)
            y: Array (samples,) - prix futur
        """
        self.feature_names = features

        # Normaliser les features
        X_scaled = self.scaler.fit_transform(df[features])

        # Créer séquences
        X_sequences = []
        y_targets = []

        for i in range(self.lookback_window, len(df) - self.prediction_horizon):
            # Séquence de lookback_window jours
            X_seq = X_scaled[i - self.lookback_window:i]

            # Target = prix dans prediction_horizon jours
            y_target = df['close'].iloc[i + self.prediction_horizon]

            X_sequences.append(X_seq)
            y_targets.append(y_target)

        X = np.array(X_sequences)
        y = np.array(y_targets)

        logger.info(f"Séquences créées:")
        logger.info(f"   X shape: {X.shape} (samples, lookback, features)")
        logger.info(f"   y shape: {y.shape}")

        return X, y

    def build_model(self, n_features: int):
        """
        Construit l'architecture LSTM

        Args:
            n_features: Nombre de features en input
        """
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(self.lookback_window, n_features)))

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)  # True sauf pour la dernière

            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout
            ))

        # Dense output
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1))  # Prédiction de prix (single value)

        # Compiler
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model

        logger.info("\n📊 Architecture LSTM:")
        model.summary(print_fn=logger.info)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15
    ):
        """
        Entraîne le modèle LSTM

        Args:
            X_train: Séquences d'entraînement
            y_train: Targets d'entraînement
            X_val: Séquences de validation
            y_val: Targets de validation
            epochs: Nombre d'époques max
            batch_size: Taille de batch
            patience: Early stopping patience
        """
        logger.info(f"Entraînement LSTM (samples={len(X_train)}, epochs={epochs})...")

        # Build model si pas déjà fait
        if self.model is None:
            n_features = X_train.shape[2]
            self.build_model(n_features)

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Entraînement
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )

        # Métriques finales
        y_train_pred = self.model.predict(X_train, verbose=0).flatten()

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # MAPE (Mean Absolute Percentage Error)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

        self.training_metrics = {
            'train_rmse': float(train_rmse),
            'train_mae': float(train_mae),
            'train_r2': float(train_r2),
            'train_mape': float(train_mape),
            'train_samples': len(X_train),
            'epochs_trained': len(history.history['loss']),
            'prediction_horizon': self.prediction_horizon
        }

        logger.info(f"\n✅ Entraînement terminé!")
        logger.info(f"   Époques: {self.training_metrics['epochs_trained']}")
        logger.info(f"   Train RMSE: ${train_rmse:.2f}")
        logger.info(f"   Train MAE: ${train_mae:.2f}")
        logger.info(f"   Train MAPE: {train_mape:.2f}%")
        logger.info(f"   Train R²: {train_r2:.4f}")

        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val, verbose=0).flatten()

            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

            self.training_metrics.update({
                'val_rmse': float(val_rmse),
                'val_mae': float(val_mae),
                'val_r2': float(val_r2),
                'val_mape': float(val_mape),
                'val_samples': len(X_val)
            })

            logger.info(f"\n   Val RMSE: ${val_rmse:.2f}")
            logger.info(f"   Val MAE: ${val_mae:.2f}")
            logger.info(f"   Val MAPE: {val_mape:.2f}%")
            logger.info(f"   Val R²: {val_r2:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les prix futurs

        Args:
            X: Séquences (samples, lookback_window, n_features)

        Returns:
            Array de prédictions de prix
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné! Appeler .train() d'abord.")

        predictions = self.model.predict(X, verbose=0).flatten()
        return predictions

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> float:
        """
        Prédit le prix futur à partir d'un DataFrame

        Args:
            df: DataFrame avec historique récent (>= lookback_window jours)
            features: Liste des features

        Returns:
            Prix prédit
        """
        if len(df) < self.lookback_window:
            raise ValueError(f"Besoin d'au moins {self.lookback_window} jours de données")

        # Prendre les derniers lookback_window jours
        df_recent = df.tail(self.lookback_window)

        # Normaliser
        X_scaled = self.scaler.transform(df_recent[features])

        # Créer séquence
        X_seq = X_scaled.reshape(1, self.lookback_window, len(features))

        # Prédire
        prediction = self.predict(X_seq)[0]

        return prediction

    def get_prediction_signal(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Dict:
        """
        Génère un signal de trading basé sur la prédiction

        Args:
            df: DataFrame avec historique
            features: Features

        Returns:
            Dict avec prédiction et signal
        """
        current_price = df['close'].iloc[-1]
        predicted_price = self.predict_from_dataframe(df, features)

        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100

        # Signal
        if price_change_pct > 1.0:
            action = 'BUY'
            direction = 'UP'
        elif price_change_pct < -1.0:
            action = 'SELL'
            direction = 'DOWN'
        else:
            action = 'HOLD'
            direction = 'FLAT'

        signal = {
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'direction': direction,
            'action': action,
            'horizon_days': self.prediction_horizon,
            'model': 'LSTM'
        }

        return signal

    def save(self, path: str):
        """
        Sauvegarde le modèle

        Args:
            path: Chemin du fichier (sans extension)
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder!")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modèle Keras
        model_path = path.with_suffix('.keras')
        self.model.save(str(model_path))

        # Sauvegarder le scaler
        scaler_path = path.with_suffix('.scaler.pkl')
        joblib.dump(self.scaler, str(scaler_path))

        # Sauvegarder les métadonnées
        metadata = {
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'prediction_horizon': self.prediction_horizon,
            'lookback_window': self.lookback_window,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'saved_at': datetime.now().isoformat()
        }

        metadata_path = path.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Modèle sauvegardé: {model_path}")

    def load(self, path: str):
        """
        Charge un modèle sauvegardé

        Args:
            path: Chemin du fichier (sans extension)
        """
        path = Path(path)

        # Charger le modèle Keras
        model_path = path.with_suffix('.keras')
        self.model = keras.models.load_model(str(model_path))

        # Charger le scaler
        scaler_path = path.with_suffix('.scaler.pkl')
        self.scaler = joblib.load(str(scaler_path))

        # Charger les métadonnées
        metadata_path = path.with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.training_metrics = metadata['training_metrics']
        self.prediction_horizon = metadata['prediction_horizon']
        self.lookback_window = metadata['lookback_window']
        self.lstm_units = metadata['lstm_units']
        self.dropout = metadata['dropout']

        logger.info(f"✅ Modèle chargé: {model_path}")
        logger.info(f"   Sauvegardé le: {metadata['saved_at']}")
        logger.info(f"   Train MAPE: {self.training_metrics.get('train_mape', 'N/A'):.2f}%")


# ============================================================================
# MULTI-HORIZON LSTM
# ============================================================================

class MultiHorizonLSTM:
    """
    Combine 3 LSTM pour prédictions multi-horizons (1j, 3j, 7j)
    """

    def __init__(
        self,
        lookback_window: int = 30,
        lstm_units: List[int] = [64, 32]
    ):
        self.predictors = {
            '1d': LSTMPredictor(
                prediction_horizon=1,
                lookback_window=lookback_window,
                lstm_units=lstm_units
            ),
            '3d': LSTMPredictor(
                prediction_horizon=3,
                lookback_window=lookback_window,
                lstm_units=lstm_units
            ),
            '7d': LSTMPredictor(
                prediction_horizon=7,
                lookback_window=lookback_window,
                lstm_units=lstm_units
            )
        }

        logger.info("MultiHorizonLSTM initialisé (1d, 3d, 7d)")

    def train_all(
        self,
        df: pd.DataFrame,
        features: List[str],
        train_split: float = 0.8,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Entraîne les 3 LSTM

        Args:
            df: DataFrame complet avec features et 'close'
            features: Liste des colonnes features
            train_split: Ratio train/test
            epochs: Nombre d'époques
            batch_size: Taille de batch
        """
        logger.info("=" * 80)
        logger.info("ENTRAÎNEMENT MULTI-HORIZON LSTM")
        logger.info("=" * 80)

        results = {}

        for horizon, predictor in self.predictors.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"HORIZON: {horizon}")
            logger.info(f"{'='*80}")

            # Préparer séquences
            X, y = predictor.prepare_sequences(df, features)

            # Split train/test
            split_idx = int(len(X) * train_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            logger.info(f"\nDataset:")
            logger.info(f"   Train: {len(X_train)} séquences")
            logger.info(f"   Test:  {len(X_test)} séquences")

            # Entraîner
            predictor.train(
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=epochs,
                batch_size=batch_size
            )

            results[horizon] = predictor.training_metrics

        logger.info("\n" + "=" * 80)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        logger.info("=" * 80)

        # Résumé
        logger.info("\n📊 RÉSUMÉ DES PERFORMANCES:")
        for horizon, metrics in results.items():
            logger.info(f"\n{horizon}:")
            logger.info(f"   Train MAPE: {metrics['train_mape']:.2f}%")
            if 'val_mape' in metrics:
                logger.info(f"   Val MAPE:   {metrics['val_mape']:.2f}%")
            logger.info(f"   Train R²:   {metrics['train_r2']:.4f}")
            if 'val_r2' in metrics:
                logger.info(f"   Val R²:     {metrics['val_r2']:.4f}")

    def get_multi_horizon_predictions(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Dict:
        """
        Prédictions pour tous les horizons

        Args:
            df: DataFrame avec historique récent
            features: Features

        Returns:
            Dict avec prédictions 1d, 3d, 7d
        """
        predictions = {}

        current_price = df['close'].iloc[-1]

        for horizon, predictor in self.predictors.items():
            signal = predictor.get_prediction_signal(df, features)
            predictions[horizon] = signal

        # Tendance globale
        trend_score = 0
        for pred in predictions.values():
            if pred['direction'] == 'UP':
                trend_score += 1
            elif pred['direction'] == 'DOWN':
                trend_score -= 1

        if trend_score > 0:
            overall_trend = 'BULLISH'
        elif trend_score < 0:
            overall_trend = 'BEARISH'
        else:
            overall_trend = 'NEUTRAL'

        return {
            'predictions': predictions,
            'current_price': float(current_price),
            'overall_trend': overall_trend,
            'trend_strength': abs(trend_score)
        }

    def save_all(self, base_path: str):
        """Sauvegarde tous les LSTM"""
        for horizon, predictor in self.predictors.items():
            path = Path(base_path) / f"lstm_{horizon}"
            predictor.save(str(path))

    def load_all(self, base_path: str):
        """Charge tous les LSTM"""
        for horizon, predictor in self.predictors.items():
            path = Path(base_path) / f"lstm_{horizon}"
            predictor.load(str(path))


# ============================================================================
# SINGLETON
# ============================================================================

_multi_horizon_lstm = None

def get_multi_horizon_lstm() -> MultiHorizonLSTM:
    """Retourne instance singleton"""
    global _multi_horizon_lstm
    if _multi_horizon_lstm is None:
        _multi_horizon_lstm = MultiHorizonLSTM()
    return _multi_horizon_lstm


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("LSTM PREDICTOR - Test")
    print("=" * 80)

    # Créer données synthétiques
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'volume': np.random.uniform(1e6, 5e6, len(dates)),
        'rsi_14': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.randn(len(dates)),
        'bb_width': np.random.uniform(0, 5, len(dates))
    }, index=dates)

    features = ['volume', 'rsi_14', 'macd', 'bb_width']

    print(f"\n📊 Dataset: {len(df)} jours, {len(features)} features")

    # Test LSTM 1 jour
    print("\n" + "=" * 80)
    print("TEST 1: LSTM 1 jour")
    print("=" * 80)

    predictor = LSTMPredictor(prediction_horizon=1, lookback_window=30)

    # Préparer séquences
    X, y = predictor.prepare_sequences(df, features)

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Entraîner (peu d'époques pour test rapide)
    predictor.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)

    # Test prédiction
    signal = predictor.get_prediction_signal(df.tail(50), features)
    print(f"\n📈 Signal de test:")
    print(f"   Prix actuel: ${signal['current_price']:.2f}")
    print(f"   Prix prédit (1j): ${signal['predicted_price']:.2f}")
    print(f"   Change: {signal['price_change_pct']:+.2f}%")
    print(f"   Action: {signal['action']}")

    print("\n✅ Test terminé!")
