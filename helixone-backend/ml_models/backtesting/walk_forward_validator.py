"""
Walk-Forward Validator - Validation réaliste des modèles ML

Principe:
1. Split données en windows (train/test)
2. Pour chaque window:
   - Entraîner sur période passée
   - Tester sur période future
   - Avancer la window
3. Agréger les résultats

But: Éviter l'overfitting, simuler trading réel
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Validateur walk-forward pour modèles ML
    """

    def __init__(
        self,
        train_window_days: int = 252,  # 1 an
        test_window_days: int = 63,  # ~3 mois
        step_days: int = 21  # Avancer de 1 mois
    ):
        """
        Args:
            train_window_days: Taille window d'entraînement (jours)
            test_window_days: Taille window de test (jours)
            step_days: Pas d'avancement (jours)
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days

        logger.info(f"WalkForwardValidator initialisé")
        logger.info(f"   Train window: {train_window_days} jours (~{train_window_days/252:.1f} ans)")
        logger.info(f"   Test window: {test_window_days} jours (~{test_window_days/21:.1f} mois)")
        logger.info(f"   Step: {step_days} jours (~{step_days/21:.1f} mois)")

    def generate_windows(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Génère les windows train/test

        Args:
            df: DataFrame complet

        Returns:
            Liste de tuples (df_train, df_test)
        """
        windows = []

        total_days = len(df)
        min_required = self.train_window_days + self.test_window_days

        if total_days < min_required:
            logger.warning(f"Pas assez de données: {total_days} < {min_required} jours requis")
            return windows

        # Calculer nombre de windows
        max_start = total_days - self.train_window_days - self.test_window_days
        n_windows = (max_start // self.step_days) + 1

        logger.info(f"   Génération de {n_windows} windows...")

        start_idx = 0

        while start_idx + self.train_window_days + self.test_window_days <= total_days:
            # Train window
            train_start = start_idx
            train_end = start_idx + self.train_window_days

            # Test window
            test_start = train_end
            test_end = test_start + self.test_window_days

            # Extraire
            df_train = df.iloc[train_start:train_end]
            df_test = df.iloc[test_start:test_end]

            windows.append((df_train, df_test))

            # Avancer
            start_idx += self.step_days

        logger.info(f"   ✅ {len(windows)} windows générées")

        return windows

    def validate(
        self,
        df: pd.DataFrame,
        train_fn,
        predict_fn,
        metric_fn
    ) -> Dict:
        """
        Exécute la validation walk-forward

        Args:
            df: DataFrame complet avec features
            train_fn: Fonction d'entraînement(df_train) -> model
            predict_fn: Fonction de prédiction(model, df_test) -> predictions
            metric_fn: Fonction de métrique(y_true, y_pred) -> score

        Returns:
            Dict avec résultats agrégés
        """
        logger.info(f"\n{'='*80}")
        logger.info("WALK-FORWARD VALIDATION")
        logger.info(f"{'='*80}")

        windows = self.generate_windows(df)

        if not windows:
            logger.error("Aucune window générée!")
            return {}

        results = []

        for i, (df_train, df_test) in enumerate(windows, 1):
            logger.info(f"\n📊 Window {i}/{len(windows)}")
            logger.info(f"   Train: {df_train.index[0]} → {df_train.index[-1]} ({len(df_train)} jours)")
            logger.info(f"   Test:  {df_test.index[0]} → {df_test.index[-1]} ({len(df_test)} jours)")

            try:
                # Entraîner
                logger.debug("   Entraînement...")
                model = train_fn(df_train)

                # Prédire
                logger.debug("   Prédiction...")
                predictions = predict_fn(model, df_test)

                # Évaluer
                logger.debug("   Évaluation...")
                score = metric_fn(df_test, predictions)

                results.append({
                    'window': i,
                    'train_start': df_train.index[0],
                    'train_end': df_train.index[-1],
                    'test_start': df_test.index[0],
                    'test_end': df_test.index[-1],
                    'score': score
                })

                logger.info(f"   ✅ Score: {score:.4f}")

            except Exception as e:
                logger.error(f"   ❌ Erreur: {e}")
                results.append({
                    'window': i,
                    'score': np.nan,
                    'error': str(e)
                })

        # Agréger résultats
        scores = [r['score'] for r in results if not np.isnan(r['score'])]

        if scores:
            summary = {
                'n_windows': len(windows),
                'n_successful': len(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'median_score': np.median(scores),
                'results': results
            }
        else:
            summary = {
                'n_windows': len(windows),
                'n_successful': 0,
                'results': results
            }

        # Afficher résumé
        logger.info(f"\n{'='*80}")
        logger.info("📊 RÉSUMÉ WALK-FORWARD")
        logger.info(f"{'='*80}")
        logger.info(f"   Windows: {summary['n_windows']}")
        logger.info(f"   Succès: {summary['n_successful']}")

        if summary['n_successful'] > 0:
            logger.info(f"   Score moyen: {summary['mean_score']:.4f} ± {summary['std_score']:.4f}")
            logger.info(f"   Score min: {summary['min_score']:.4f}")
            logger.info(f"   Score max: {summary['max_score']:.4f}")
            logger.info(f"   Score médian: {summary['median_score']:.4f}")

        return summary


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("WALK-FORWARD VALIDATOR - Test")
    print("=" * 80)

    # Créer données synthétiques
    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'feature1': np.random.randn(len(dates)),
        'feature2': np.random.randn(len(dates))
    }, index=dates)

    print(f"\n📊 Dataset: {len(df)} jours")
    print(f"   Période: {df.index[0]} → {df.index[-1]}")

    # Créer validator
    validator = WalkForwardValidator(
        train_window_days=252,  # 1 an
        test_window_days=63,  # 3 mois
        step_days=21  # 1 mois
    )

    # Fonctions dummy pour test
    def dummy_train(df_train):
        """Entraîne un modèle dummy (retourne juste la moyenne)"""
        return {'mean': df_train['close'].mean()}

    def dummy_predict(model, df_test):
        """Prédit avec le modèle dummy"""
        return pd.Series(model['mean'], index=df_test.index)

    def dummy_metric(df_test, predictions):
        """Calcule MAE"""
        return np.mean(np.abs(df_test['close'] - predictions))

    # Run validation
    print("\n" + "=" * 80)
    print("Exécution walk-forward validation...")
    print("=" * 80)

    results = validator.validate(
        df=df,
        train_fn=dummy_train,
        predict_fn=dummy_predict,
        metric_fn=dummy_metric
    )

    print("\n✅ Validation terminée!")
    print(f"\n📊 Résultats:")
    print(f"   {results['n_successful']}/{results['n_windows']} windows réussies")
    if results['n_successful'] > 0:
        print(f"   MAE moyen: ${results['mean_score']:.2f} ± ${results['std_score']:.2f}")
