"""
Indicateurs techniques personnalisés
"""

import pandas as pd
import numpy as np

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()
    
    # Éviter la division par zéro
    rs = gain / loss.replace(0, np.inf)
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.fillna(50)  # Valeur neutre si NaN

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    """Bollinger Bands"""
    sma_values = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper_band = sma_values + (std * num_std)
    lower_band = sma_values - (std * num_std)
    return sma_values, upper_band, lower_band

def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator %K and %D"""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    
    # Éviter la division par zéro
    denominator = highest_high - lowest_low
    k_percent = 100 * (close - lowest_low) / denominator.replace(0, 1)
    d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
    return k_percent.fillna(50), d_percent.fillna(50)

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
               lookback: int = 14) -> pd.Series:
    """Williams %R"""
    highest_high = high.rolling(window=lookback, min_periods=1).max()
    lowest_low = low.rolling(window=lookback, min_periods=1).min()
    
    # Éviter la division par zéro
    denominator = highest_high - lowest_low
    wr = -100 * (highest_high - close) / denominator.replace(0, 1)
    return wr.fillna(-50)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
        window: int = 14) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=window, min_periods=1).mean()