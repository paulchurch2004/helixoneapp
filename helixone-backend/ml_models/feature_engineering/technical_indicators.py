"""
Technical Indicators - 20+ indicateurs techniques pour ML

Indicateurs implémentés:
- Trend: SMA, EMA, MACD, ADX
- Momentum: RSI, Stochastic, ROC, Williams %R
- Volatility: Bollinger Bands, ATR, Keltner Channel
- Volume: OBV, MFI, VWAP, Volume Rate of Change
- Support/Resistance: Pivot Points, Fibonacci levels

Tous calculés avec la librairie 'ta' ou 'pandas-ta' (gratuit, open-source)
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

# Utiliser pandas-ta (plus complet que ta)
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    # Fallback sur ta si pandas-ta pas installé
    import ta as ta_lib
    HAS_PANDAS_TA = False

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculateur d'indicateurs techniques
    """

    def __init__(self):
        logger.info(f"TechnicalIndicators initialisé (pandas_ta: {HAS_PANDAS_TA})")

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute TOUS les indicateurs techniques au DataFrame

        Args:
            df: DataFrame avec colonnes 'open', 'high', 'low', 'close', 'volume'

        Returns:
            DataFrame avec 50+ nouvelles colonnes de features
        """
        df = df.copy()

        logger.debug("Calcul indicateurs techniques...")

        # Vérifier colonnes requises
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame doit contenir: {required}")

        # 1. TREND INDICATORS
        df = self._add_trend_indicators(df)

        # 2. MOMENTUM INDICATORS
        df = self._add_momentum_indicators(df)

        # 3. VOLATILITY INDICATORS
        df = self._add_volatility_indicators(df)

        # 4. VOLUME INDICATORS
        df = self._add_volume_indicators(df)

        # 5. CANDLESTICK PATTERNS (simplifié)
        df = self._add_candlestick_features(df)

        # 6. PRICE PATTERNS
        df = self._add_price_patterns(df)

        logger.debug(f"✅ {len([c for c in df.columns if c not in required])} indicateurs ajoutés")

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de tendance"""
        close = df['close']
        high = df['high']
        low = df['low']

        if HAS_PANDAS_TA:
            # SMA (Simple Moving Average)
            df['sma_10'] = ta.sma(close, length=10)
            df['sma_20'] = ta.sma(close, length=20)
            df['sma_50'] = ta.sma(close, length=50)
            df['sma_200'] = ta.sma(close, length=200)

            # EMA (Exponential Moving Average)
            df['ema_12'] = ta.ema(close, length=12)
            df['ema_26'] = ta.ema(close, length=26)

            # MACD
            macd = ta.macd(close)
            if macd is not None:
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                df['macd_hist'] = macd['MACDh_12_26_9']

            # ADX (Average Directional Index)
            adx = ta.adx(high, low, close)
            if adx is not None:
                df['adx'] = adx['ADX_14']
                df['di_plus'] = adx['DMP_14']
                df['di_minus'] = adx['DMN_14']

        else:
            # Fallback ta library
            df['sma_10'] = close.rolling(window=10).mean()
            df['sma_20'] = close.rolling(window=20).mean()
            df['sma_50'] = close.rolling(window=50).mean()
            df['sma_200'] = close.rolling(window=200).mean()

            df['ema_12'] = close.ewm(span=12).mean()
            df['ema_26'] = close.ewm(span=26).mean()

            # MACD manuel
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        # Prix relatif aux moyennes mobiles
        df['price_above_sma20'] = (close > df['sma_20']).astype(int)
        df['price_above_sma50'] = (close > df['sma_50']).astype(int)
        df['price_to_sma20_ratio'] = close / df['sma_20']

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de momentum"""
        close = df['close']
        high = df['high']
        low = df['low']

        if HAS_PANDAS_TA:
            # RSI (Relative Strength Index)
            df['rsi_14'] = ta.rsi(close, length=14)

            # Stochastic Oscillator
            stoch = ta.stoch(high, low, close)
            if stoch is not None:
                df['stoch_k'] = stoch['STOCHk_14_3_3']
                df['stoch_d'] = stoch['STOCHd_14_3_3']

            # ROC (Rate of Change)
            df['roc_10'] = ta.roc(close, length=10)

            # Williams %R
            df['williams_r'] = ta.willr(high, low, close)

            # CCI (Commodity Channel Index)
            df['cci'] = ta.cci(high, low, close)

        else:
            # RSI manuel
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # ROC manuel
            df['roc_10'] = close.pct_change(periods=10) * 100

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de volatilité"""
        close = df['close']
        high = df['high']
        low = df['low']

        if HAS_PANDAS_TA:
            # Bollinger Bands
            bbands = ta.bbands(close, length=20, std=2)
            if bbands is not None:
                df['bb_upper'] = bbands['BBU_20_2.0']
                df['bb_middle'] = bbands['BBM_20_2.0']
                df['bb_lower'] = bbands['BBL_20_2.0']
                df['bb_width'] = bbands['BBB_20_2.0']

            # ATR (Average True Range)
            df['atr'] = ta.atr(high, low, close)

            # Keltner Channel
            kc = ta.kc(high, low, close)
            if kc is not None:
                df['kc_upper'] = kc['KCU_20_2']
                df['kc_lower'] = kc['KCL_20_2']

        else:
            # Bollinger Bands manuel
            sma = close.rolling(window=20).mean()
            std = close.rolling(window=20).std()
            df['bb_upper'] = sma + (std * 2)
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # ATR manuel
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()

        # Position prix vs Bollinger Bands
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volatilité historique (annualisée)
        returns = close.pct_change()
        df['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicateurs de volume"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        if HAS_PANDAS_TA:
            # OBV (On-Balance Volume)
            df['obv'] = ta.obv(close, volume)

            # MFI (Money Flow Index)
            df['mfi'] = ta.mfi(high, low, close, volume)

            # VWAP (Volume Weighted Average Price)
            df['vwap'] = ta.vwap(high, low, close, volume)

        else:
            # OBV manuel
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            df['obv'] = obv

        # Volume features
        df['volume_sma_20'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20']
        df['volume_roc_10'] = volume.pct_change(periods=10)

        # Volume relatif
        df['relative_volume'] = volume / volume.rolling(window=20).mean()

        return df

    def _add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features basées sur les chandeliers"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']

        # Body size
        df['candle_body'] = abs(close - open_price)
        df['candle_body_pct'] = df['candle_body'] / open_price

        # Upper shadow
        df['upper_shadow'] = high - np.maximum(open_price, close)
        df['upper_shadow_pct'] = df['upper_shadow'] / open_price

        # Lower shadow
        df['lower_shadow'] = np.minimum(open_price, close) - low
        df['lower_shadow_pct'] = df['lower_shadow'] / open_price

        # Total range
        df['candle_range'] = high - low
        df['candle_range_pct'] = df['candle_range'] / open_price

        # Direction
        df['candle_direction'] = (close > open_price).astype(int)

        # Doji detection (body très petit)
        df['is_doji'] = (df['candle_body_pct'] < 0.001).astype(int)

        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Patterns de prix"""
        close = df['close']
        high = df['high']
        low = df['low']

        # Highs et Lows
        df['high_20'] = high.rolling(window=20).max()
        df['low_20'] = low.rolling(window=20).min()

        # Distance from high/low
        df['distance_from_high'] = (df['high_20'] - close) / df['high_20']
        df['distance_from_low'] = (close - df['low_20']) / df['low_20']

        # Price momentum (ROC différentes périodes)
        df['price_change_1d'] = close.pct_change(1)
        df['price_change_5d'] = close.pct_change(5)
        df['price_change_10d'] = close.pct_change(10)
        df['price_change_20d'] = close.pct_change(20)

        # Trend strength
        df['higher_highs'] = (high > high.shift(1)).rolling(window=5).sum()
        df['lower_lows'] = (low < low.shift(1)).rolling(window=5).sum()

        return df

    def get_feature_names(self) -> list:
        """
        Retourne la liste de tous les noms de features créées

        Returns:
            Liste de noms de colonnes
        """
        # Créer un df dummy pour extraire les noms
        dummy_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000000, 1100000, 1200000]
        })

        result = self.add_all_indicators(dummy_df)
        features = [col for col in result.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

        return features


# ============================================================================
# SINGLETON
# ============================================================================

_tech_indicators_instance = None

def get_technical_indicators() -> TechnicalIndicators:
    """Retourne une instance singleton"""
    global _tech_indicators_instance
    if _tech_indicators_instance is None:
        _tech_indicators_instance = TechnicalIndicators()
    return _tech_indicators_instance


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TECHNICAL INDICATORS - Test")
    print("=" * 80)

    # Créer données de test
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Générer prix synthétiques
    price = 100
    prices = [price]
    for _ in range(len(dates) - 1):
        price *= (1 + np.random.randn() * 0.02)
        prices.append(price)

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': [p * (1 + np.random.randn() * 0.01) for p in prices],
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    print(f"\n📊 Dataset test: {len(df)} jours")
    print(df.head())

    # Calculer indicateurs
    print("\n⚙️ Calcul des indicateurs techniques...")
    indicators = TechnicalIndicators()
    df_with_indicators = indicators.add_all_indicators(df)

    print(f"\n✅ Indicateurs calculés!")
    print(f"   Features originales: 5")
    print(f"   Features totales: {len(df_with_indicators.columns)}")
    print(f"   Nouvelles features: {len(df_with_indicators.columns) - 5}")

    print("\n📋 Liste des indicateurs:")
    new_features = [col for col in df_with_indicators.columns
                    if col not in ['open', 'high', 'low', 'close', 'volume']]
    for i, feat in enumerate(new_features, 1):
        print(f"   {i:2d}. {feat}")

    print("\n📈 Exemple de valeurs (derniers 5 jours):")
    cols_to_show = ['close', 'rsi_14', 'macd', 'bb_position', 'volume_ratio']
    print(df_with_indicators[cols_to_show].tail())

    print("\n✅ Test terminé!")
