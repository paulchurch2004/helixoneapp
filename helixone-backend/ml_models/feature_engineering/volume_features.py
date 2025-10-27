"""
Volume Features - Features basées sur le volume de trading

Le volume est un indicateur clé souvent ignoré mais très prédictif:
- Volume inhabituel = mouvement important à venir
- OBV (On-Balance Volume) = accumulation/distribution
- Volume Price Trend = force de la tendance
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VolumeFeatures:
    """Extracteur de features basées sur le volume"""

    def __init__(self):
        logger.info("VolumeFeatures initialisé")

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features de volume

        Args:
            df: DataFrame avec colonnes 'close', 'volume'

        Returns:
            DataFrame avec features volume
        """
        df = df.copy()

        close = df['close']
        volume = df['volume']

        # 1. Volume relatif
        df['volume_sma_20'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20']
        df['volume_zscore'] = (volume - df['volume_sma_20']) / volume.rolling(20).std()

        # 2. Volume trends
        df['volume_trend'] = volume.rolling(window=10).mean() / volume.rolling(window=30).mean()

        # 3. Price-Volume interaction
        df['price_volume_correlation'] = close.rolling(20).corr(volume)

        # 4. Volume spikes
        df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
        df['extreme_volume'] = (df['volume_ratio'] > 3).astype(int)

        # 5. Accumulation/Distribution
        df['ad_line'] = ((close - df['low']) - (df['high'] - close)) / (df['high'] - df['low']) * volume
        df['ad_line'] = df['ad_line'].fillna(0).cumsum()

        logger.debug(f"✅ Features volume ajoutées")
        return df


def get_volume_features():
    global _volume_features_instance
    if '_volume_features_instance' not in globals():
        _volume_features_instance = VolumeFeatures()
    return _volume_features_instance
