"""
Feature Selector - Sélection des meilleures features pour ML

Méthodes:
1. Feature Importance (XGBoost)
2. Correlation analysis (éliminer corrélations >0.95)
3. Variance threshold (éliminer features constantes)
4. Recursive Feature Elimination (RFE)

Objectif: Sélectionner top 50 features les plus prédictives
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Sélectionneur de features intelligent"""

    def __init__(self, max_features: int = 50):
        """
        Args:
            max_features: Nombre maximum de features à garder
        """
        self.max_features = max_features
        self.selected_features = []
        self.feature_importances = {}

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'xgboost'
    ) -> List[str]:
        """
        Sélectionne les meilleures features

        Args:
            X: Features DataFrame
            y: Target (labels)
            method: 'xgboost', 'rf', 'rfe', ou 'correlation'

        Returns:
            Liste des features sélectionnées
        """
        logger.info(f"Sélection de {self.max_features} features parmi {len(X.columns)}...")

        # 1. Éliminer features avec variance nulle
        X_filtered = self._remove_low_variance(X)
        logger.info(f"   Après variance threshold: {len(X_filtered.columns)} features")

        # 2. Éliminer corrélations élevées
        X_filtered = self._remove_high_correlation(X_filtered)
        logger.info(f"   Après correlation removal: {len(X_filtered.columns)} features")

        # 3. Feature importance
        if method == 'xgboost':
            selected = self._select_by_xgboost(X_filtered, y)
        elif method == 'rf':
            selected = self._select_by_random_forest(X_filtered, y)
        elif method == 'rfe':
            selected = self._select_by_rfe(X_filtered, y)
        else:
            selected = list(X_filtered.columns[:self.max_features])

        self.selected_features = selected
        logger.info(f"✅ {len(selected)} features sélectionnées")

        return selected

    def _remove_low_variance(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Élimine features avec variance très faible"""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X.fillna(0))
        mask = selector.get_support()
        return X.loc[:, mask]

    def _remove_high_correlation(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Élimine features très corrélées"""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return X.drop(columns=to_drop)

    def _select_by_xgboost(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Sélection via XGBoost feature importance"""
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X.fillna(0), y)

        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        self.feature_importances = importances.to_dict()
        return list(importances.head(self.max_features).index)

    def _select_by_random_forest(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Sélection via Random Forest"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X.fillna(0), y)

        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        return list(importances.head(self.max_features).index)

    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Sélection via Recursive Feature Elimination"""
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=self.max_features, step=10)
        selector.fit(X.fillna(0), y)

        return list(X.columns[selector.support_])


def get_feature_selector(max_features: int = 50):
    return FeatureSelector(max_features=max_features)
