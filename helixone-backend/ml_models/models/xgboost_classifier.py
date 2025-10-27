"""
XGBoost Classifier - Prédiction de direction (UP/DOWN/FLAT)

Objectifs:
- Précision >72%
- Prédire direction sur 1 jour, 3 jours, 7 jours
- Hyperparameter tuning avec Optuna
- SHAP explainability

Classes:
- UP: Price change > +1%
- DOWN: Price change < -1%
- FLAT: Price change entre -1% et +1%
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
import joblib
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """
    Classificateur XGBoost pour prédire la direction du prix
    """

    def __init__(
        self,
        prediction_horizon: int = 1,
        up_threshold: float = 0.01,
        down_threshold: float = -0.01
    ):
        """
        Args:
            prediction_horizon: Nombre de jours à prédire (1, 3, 7)
            up_threshold: Seuil pour classe UP (défaut +1%)
            down_threshold: Seuil pour classe DOWN (défaut -1%)
        """
        self.prediction_horizon = prediction_horizon
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

        self.model = None
        self.feature_names = []
        self.best_params = {}
        self.training_metrics = {}

        logger.info(f"XGBoost Classifier initialisé (horizon={prediction_horizon}j)")

    def prepare_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Crée les labels UP/DOWN/FLAT

        Args:
            df: DataFrame avec colonne 'close'

        Returns:
            Series avec labels (0=DOWN, 1=FLAT, 2=UP)
        """
        # Calculer le return futur
        future_return = df['close'].pct_change(periods=self.prediction_horizon).shift(-self.prediction_horizon)

        # Créer les classes
        labels = pd.Series(1, index=df.index)  # Par défaut FLAT
        labels[future_return > self.up_threshold] = 2  # UP
        labels[future_return < self.down_threshold] = 0  # DOWN

        # Statistiques
        value_counts = labels.value_counts()
        logger.info(f"Distribution des labels:")
        logger.info(f"   DOWN (0): {value_counts.get(0, 0)} ({value_counts.get(0, 0)/len(labels)*100:.1f}%)")
        logger.info(f"   FLAT (1): {value_counts.get(1, 0)} ({value_counts.get(1, 0)/len(labels)*100:.1f}%)")
        logger.info(f"   UP (2): {value_counts.get(2, 0)} ({value_counts.get(2, 0)/len(labels)*100:.1f}%)")

        return labels

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 50
    ) -> Dict:
        """
        Optimise les hyperparamètres avec Optuna

        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            n_trials: Nombre d'essais Optuna

        Returns:
            Meilleurs paramètres
        """
        logger.info(f"Optimisation hyperparamètres avec {n_trials} trials...")

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'tree_method': 'hist',
                'eval_metric': 'mlogloss',
                'num_class': 3
            }

            # Cross-validation time series
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)

            return np.mean(scores)

        # Optimisation
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"✅ Meilleure accuracy: {study.best_value:.4f}")
        logger.info(f"   Meilleurs params: {study.best_params}")

        self.best_params = study.best_params
        return study.best_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        optimize: bool = True,
        n_trials: int = 50
    ):
        """
        Entraîne le modèle XGBoost

        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Labels de validation (optionnel)
            optimize: Si True, optimise les hyperparamètres
            n_trials: Nombre de trials pour Optuna
        """
        logger.info(f"Entraînement XGBoost (samples={len(X_train)}, features={len(X_train.columns)})...")

        self.feature_names = list(X_train.columns)

        # Optimisation hyperparamètres
        if optimize:
            params = self.optimize_hyperparameters(X_train, y_train, n_trials=n_trials)
        else:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'min_child_weight': 3,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'tree_method': 'hist',
                'eval_metric': 'mlogloss',
                'num_class': 3
            }

        # Entraînement final
        self.model = xgb.XGBClassifier(**params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True
        )

        # Métriques d'entraînement
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        self.training_metrics = {
            'train_accuracy': train_accuracy,
            'train_samples': len(X_train),
            'n_features': len(self.feature_names),
            'prediction_horizon': self.prediction_horizon
        }

        logger.info(f"✅ Entraînement terminé!")
        logger.info(f"   Train accuracy: {train_accuracy:.4f}")

        # Validation
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            self.training_metrics['val_accuracy'] = val_accuracy
            self.training_metrics['val_samples'] = len(X_val)

            logger.info(f"   Val accuracy: {val_accuracy:.4f}")

            # Classification report
            logger.info("\n" + classification_report(
                y_val,
                y_val_pred,
                target_names=['DOWN', 'FLAT', 'UP']
            ))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit les classes

        Args:
            X: Features

        Returns:
            Array de prédictions (0=DOWN, 1=FLAT, 2=UP)
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné! Appeler .train() d'abord.")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités pour chaque classe

        Args:
            X: Features

        Returns:
            Array de shape (n_samples, 3) avec probas [DOWN, FLAT, UP]
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné! Appeler .train() d'abord.")

        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retourne l'importance des features

        Args:
            top_n: Nombre de top features à retourner

        Returns:
            DataFrame avec features et importances
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné!")

        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })

        importances = importances.sort_values('importance', ascending=False)

        return importances.head(top_n)

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

        # Sauvegarder le modèle XGBoost
        model_path = path.with_suffix('.json')
        self.model.save_model(str(model_path))

        # Sauvegarder les métadonnées
        metadata = {
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'training_metrics': self.training_metrics,
            'prediction_horizon': self.prediction_horizon,
            'up_threshold': self.up_threshold,
            'down_threshold': self.down_threshold,
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

        # Charger le modèle
        model_path = path.with_suffix('.json')
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))

        # Charger les métadonnées
        metadata_path = path.with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.best_params = metadata['best_params']
        self.training_metrics = metadata['training_metrics']
        self.prediction_horizon = metadata['prediction_horizon']
        self.up_threshold = metadata['up_threshold']
        self.down_threshold = metadata['down_threshold']

        logger.info(f"✅ Modèle chargé: {model_path}")
        logger.info(f"   Sauvegardé le: {metadata['saved_at']}")
        logger.info(f"   Train accuracy: {self.training_metrics.get('train_accuracy', 'N/A')}")

    def get_signal(self, X: pd.DataFrame) -> Dict:
        """
        Génère un signal de trading

        Args:
            X: Features (dernière ligne = position actuelle)

        Returns:
            Dict avec signal et probabilités
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné!")

        # Prédiction
        pred = self.predict(X.tail(1))[0]
        proba = self.predict_proba(X.tail(1))[0]

        # Classes
        classes = ['DOWN', 'FLAT', 'UP']
        pred_class = classes[pred]

        # Confiance = probabilité de la classe prédite
        confidence = proba[pred]

        signal = {
            'prediction': pred_class,
            'confidence': float(confidence),
            'probabilities': {
                'DOWN': float(proba[0]),
                'FLAT': float(proba[1]),
                'UP': float(proba[2])
            },
            'horizon_days': self.prediction_horizon,
            'action': 'BUY' if pred == 2 else ('SELL' if pred == 0 else 'HOLD')
        }

        return signal


# ============================================================================
# MULTI-HORIZON CLASSIFIER
# ============================================================================

class MultiHorizonClassifier:
    """
    Combine 3 classifiers pour prédictions multi-horizons (1j, 3j, 7j)
    """

    def __init__(self):
        self.classifiers = {
            '1d': XGBoostClassifier(prediction_horizon=1),
            '3d': XGBoostClassifier(prediction_horizon=3),
            '7d': XGBoostClassifier(prediction_horizon=7)
        }

        logger.info("MultiHorizonClassifier initialisé (1d, 3d, 7d)")

    def train_all(
        self,
        df: pd.DataFrame,
        features: List[str],
        train_split: float = 0.8,
        optimize: bool = True,
        n_trials: int = 30
    ):
        """
        Entraîne les 3 classifiers

        Args:
            df: DataFrame complet avec features et 'close'
            features: Liste des colonnes features
            train_split: Ratio train/test
            optimize: Optimiser hyperparamètres
            n_trials: Trials Optuna
        """
        logger.info("=" * 80)
        logger.info("ENTRAÎNEMENT MULTI-HORIZON")
        logger.info("=" * 80)

        # Split train/test (time series)
        split_idx = int(len(df) * train_split)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]

        logger.info(f"\nDataset:")
        logger.info(f"   Total: {len(df)} jours")
        logger.info(f"   Train: {len(df_train)} jours ({train_split*100:.0f}%)")
        logger.info(f"   Test:  {len(df_test)} jours ({(1-train_split)*100:.0f}%)")

        # Entraîner chaque classifier
        results = {}

        for horizon, clf in self.classifiers.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"HORIZON: {horizon}")
            logger.info(f"{'='*80}")

            # Préparer labels
            y_train = clf.prepare_labels(df_train)
            y_test = clf.prepare_labels(df_test)

            # Supprimer NaN (dernières lignes)
            valid_train = ~y_train.isna()
            valid_test = ~y_test.isna()

            X_train = df_train.loc[valid_train, features]
            y_train = y_train[valid_train]
            X_test = df_test.loc[valid_test, features]
            y_test = y_test[valid_test]

            # Entraîner
            clf.train(
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                optimize=optimize,
                n_trials=n_trials
            )

            # Feature importance
            logger.info(f"\n📊 Top 10 features ({horizon}):")
            top_features = clf.get_feature_importance(top_n=10)
            for idx, row in top_features.iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")

            results[horizon] = clf.training_metrics

        logger.info("\n" + "=" * 80)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        logger.info("=" * 80)

        # Résumé
        logger.info("\n📊 RÉSUMÉ DES PERFORMANCES:")
        for horizon, metrics in results.items():
            logger.info(f"\n{horizon}:")
            logger.info(f"   Train accuracy: {metrics['train_accuracy']:.4f}")
            if 'val_accuracy' in metrics:
                logger.info(f"   Val accuracy:   {metrics['val_accuracy']:.4f}")

    def get_multi_horizon_signal(self, X: pd.DataFrame) -> Dict:
        """
        Génère des signaux pour tous les horizons

        Args:
            X: Features (dernière ligne = position actuelle)

        Returns:
            Dict avec signaux pour 1d, 3d, 7d
        """
        signals = {}

        for horizon, clf in self.classifiers.items():
            signals[horizon] = clf.get_signal(X)

        # Consensus
        predictions = [s['prediction'] for s in signals.values()]

        # Score consensus (toutes les prédictions sont identiques)
        consensus_score = len(set(predictions)) == 1

        return {
            'signals': signals,
            'consensus': predictions[0] if consensus_score else 'MIXED',
            'consensus_score': 100 if consensus_score else 50
        }

    def save_all(self, base_path: str):
        """Sauvegarde tous les classifiers"""
        for horizon, clf in self.classifiers.items():
            path = Path(base_path) / f"xgb_{horizon}"
            clf.save(str(path))

    def load_all(self, base_path: str):
        """Charge tous les classifiers"""
        for horizon, clf in self.classifiers.items():
            path = Path(base_path) / f"xgb_{horizon}"
            clf.load(str(path))


# ============================================================================
# SINGLETON
# ============================================================================

_multi_horizon_classifier = None

def get_multi_horizon_classifier() -> MultiHorizonClassifier:
    """Retourne instance singleton"""
    global _multi_horizon_classifier
    if _multi_horizon_classifier is None:
        _multi_horizon_classifier = MultiHorizonClassifier()
    return _multi_horizon_classifier


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("XGBOOST CLASSIFIER - Test")
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

    # Test classifier 1 jour
    print("\n" + "=" * 80)
    print("TEST 1: Classifier 1 jour")
    print("=" * 80)

    clf = XGBoostClassifier(prediction_horizon=1)

    # Préparer données
    y = clf.prepare_labels(df)
    valid = ~y.isna()

    X = df.loc[valid, features]
    y = y[valid]

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Entraîner (sans optimization pour aller vite)
    clf.train(X_train, y_train, X_test, y_test, optimize=False)

    # Test prédiction
    signal = clf.get_signal(X_test)
    print(f"\n📈 Signal de test:")
    print(f"   Prédiction: {signal['prediction']}")
    print(f"   Confiance: {signal['confidence']:.2%}")
    print(f"   Action: {signal['action']}")

    print("\n✅ Test terminé!")
