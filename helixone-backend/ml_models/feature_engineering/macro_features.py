"""
Macro Features - Features macro-économiques pour ML

Transforme les données FRED en features ML-ready

Features créées :
- Taux d'intérêt (levels + changes)
- Yield curve
- Inflation (levels + YoY)
- Unemployment
- VIX (volatilité)
- Dollar Index
- Market regime indicators
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MacroFeatures:
    """
    Extracteur de features macro-économiques
    """

    def __init__(self):
        logger.info("MacroFeatures initialisé")

    def add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les données macro en features ML-ready

        Args:
            df: DataFrame avec données macro brutes (colonnes FRED)

        Returns:
            DataFrame avec features transformées
        """
        df = df.copy()

        logger.debug("Transformation features macro...")

        # 1. Taux d'intérêt
        df = self._process_interest_rates(df)

        # 2. Yield Curve
        df = self._process_yield_curve(df)

        # 3. Inflation
        df = self._process_inflation(df)

        # 4. Emploi
        df = self._process_employment(df)

        # 5. Volatilité & Risk
        df = self._process_volatility(df)

        # 6. Indicateurs de régime
        df = self._detect_regime(df)

        logger.debug("✅ Features macro transformées")

        return df

    def _process_interest_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process interest rate features"""
        # Fed Funds Rate
        if 'DFF' in df.columns:
            df['fed_funds_level'] = df['DFF']
            df['fed_funds_change'] = df['DFF'].diff()
            df['fed_funds_change_3m'] = df['DFF'].diff(63)  # ~3 mois

            # Direction (hausse/baisse/stable)
            df['fed_funds_direction'] = np.sign(df['fed_funds_change'])

        # 10Y Treasury
        if 'DGS10' in df.columns:
            df['treasury_10y'] = df['DGS10']
            df['treasury_10y_change'] = df['DGS10'].diff()

        # 2Y Treasury
        if 'DGS2' in df.columns:
            df['treasury_2y'] = df['DGS2']

        return df

    def _process_yield_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process yield curve (indicateur de récession)"""
        if 'DGS10' in df.columns and 'DGS2' in df.columns:
            # Yield Curve = 10Y - 2Y
            df['yield_curve'] = df['DGS10'] - df['DGS2']

            # Inversion (signal récession)
            df['yield_curve_inverted'] = (df['yield_curve'] < 0).astype(int)

            # Steepness (pente de la courbe)
            df['yield_curve_steepness'] = df['yield_curve'].rolling(window=20).std()

        return df

    def _process_inflation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process inflation indicators"""
        if 'CPIAUCSL' in df.columns:
            # Inflation YoY
            df['inflation_yoy'] = df['CPIAUCSL'].pct_change(252) * 100  # ~1 an

            # Inflation change
            df['inflation_change'] = df['CPIAUCSL'].pct_change()

            # Acceleration
            df['inflation_acceleration'] = df['inflation_yoy'].diff()

            # High inflation regime (>3%)
            df['high_inflation'] = (df['inflation_yoy'] > 3).astype(int)

        return df

    def _process_employment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process employment indicators"""
        if 'UNRATE' in df.columns:
            # Unemployment level
            df['unemployment'] = df['UNRATE']

            # Change
            df['unemployment_change'] = df['UNRATE'].diff()

            # Rising unemployment (bearish)
            df['unemployment_rising'] = (df['unemployment_change'] > 0).astype(int)

        return df

    def _process_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process volatility indicators"""
        if 'VIXCLS' in df.columns:
            # VIX level
            df['vix'] = df['VIXCLS']

            # VIX categories
            df['vix_low'] = (df['vix'] < 15).astype(int)  # Low vol
            df['vix_medium'] = ((df['vix'] >= 15) & (df['vix'] < 25)).astype(int)
            df['vix_high'] = (df['vix'] >= 25).astype(int)  # High vol

            # VIX spike (fear)
            df['vix_spike'] = (df['vix'].pct_change() > 0.2).astype(int)  # +20%

        return df

    def _detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime (risk-on vs risk-off)"""
        # Composite regime indicator
        regime_score = 0

        # Risk-ON indicators
        if 'DFF' in df.columns:
            regime_score += (df['DFF'] < 2).astype(int)  # Low rates = risk on

        if 'VIXCLS' in df.columns:
            regime_score += (df['VIXCLS'] < 20).astype(int)  # Low VIX = risk on

        if 'yield_curve' in df.columns:
            regime_score += (df['yield_curve'] > 0).astype(int)  # Normal curve = risk on

        # Normalize (0-1)
        if isinstance(regime_score, pd.Series):
            df['risk_on_score'] = regime_score / 3
        else:
            df['risk_on_score'] = 0.5

        # Categorical
        df['market_regime'] = 'neutral'
        if 'risk_on_score' in df.columns:
            df.loc[df['risk_on_score'] > 0.66, 'market_regime'] = 'risk_on'
            df.loc[df['risk_on_score'] < 0.33, 'market_regime'] = 'risk_off'

        return df

    def get_current_regime(self, df: pd.DataFrame) -> dict:
        """
        Retourne le régime macro actuel

        Args:
            df: DataFrame avec features macro

        Returns:
            Dict avec informations régime
        """
        latest = df.iloc[-1]

        regime = {
            'fed_funds': latest.get('fed_funds_level', None),
            'yield_curve': latest.get('yield_curve', None),
            'yield_curve_inverted': bool(latest.get('yield_curve_inverted', False)),
            'inflation_yoy': latest.get('inflation_yoy', None),
            'vix': latest.get('vix', None),
            'market_regime': latest.get('market_regime', 'unknown'),
            'risk_on_score': latest.get('risk_on_score', 0.5)
        }

        return regime


# ============================================================================
# SINGLETON
# ============================================================================

_macro_features_instance = None

def get_macro_features() -> MacroFeatures:
    """Retourne une instance singleton"""
    global _macro_features_instance
    if _macro_features_instance is None:
        _macro_features_instance = MacroFeatures()
    return _macro_features_instance


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("MACRO FEATURES - Test")
    print("=" * 80)

    # Créer données synthétiques
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'DFF': np.random.uniform(0, 5, len(dates)),
        'DGS10': np.random.uniform(1, 4, len(dates)),
        'DGS2': np.random.uniform(0.5, 3, len(dates)),
        'CPIAUCSL': 250 + np.cumsum(np.random.randn(len(dates)) * 0.1),
        'UNRATE': np.random.uniform(3, 10, len(dates)),
        'VIXCLS': np.random.uniform(10, 40, len(dates))
    }, index=dates)

    print(f"\n📊 Dataset test: {len(df)} jours")
    print(df.head())

    # Ajouter features macro
    print("\n⚙️ Transformation features macro...")
    macro = MacroFeatures()
    df_transformed = macro.add_macro_features(df)

    print(f"\n✅ Features macro transformées!")
    print(f"   Features originales: {len(df.columns)}")
    print(f"   Features totales: {len(df_transformed.columns)}")

    print("\n📋 Nouvelles features:")
    new_features = [col for col in df_transformed.columns if col not in df.columns]
    for i, feat in enumerate(new_features, 1):
        print(f"   {i:2d}. {feat}")

    print("\n📈 Régime actuel:")
    current_regime = macro.get_current_regime(df_transformed)
    for key, value in current_regime.items():
        print(f"   {key}: {value}")

    print("\n📊 Exemple de valeurs:")
    cols = ['yield_curve', 'inflation_yoy', 'vix', 'market_regime', 'risk_on_score']
    available_cols = [c for c in cols if c in df_transformed.columns]
    print(df_transformed[available_cols].tail())

    print("\n✅ Test terminé!")
