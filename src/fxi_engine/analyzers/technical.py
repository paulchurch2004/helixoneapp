"""
Analyseur technique avancé
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np

from ..core.config import EngineConfig
from ..utils.calculations import (
    sma, ema, rsi, macd, bollinger_bands, 
    stochastic_kd, williams_r, atr
)

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Analyse technique approfondie (0-100 points)"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
    
    def analyze(self, data: Dict, ticker: str) -> float:
        """
        Analyse technique complète avec 6 dimensions :
        1. Tendance (20 points)
        2. Momentum (20 points) 
        3. Volatilité (15 points)
        4. Niveaux clés (15 points)
        5. Volume (10 points)
        6. Signaux multiples (20 points)
        """
        try:
            # Récupérer les données historiques
            hist = data.get('yahoo', {}).get('history', pd.DataFrame())
            if hist is None or hist.empty or len(hist) < 60:
                logger.warning(f"Données historiques insuffisantes pour {ticker}")
                return 50.0
            
            # Prix et volumes
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']
            current_price = float(close.iloc[-1])
            
            # Calcul des indicateurs
            sma_20 = sma(close, 20)
            sma_50 = sma(close, 50)
            sma_200 = sma(close, 200)
            macd_line, macd_signal, macd_histogram = macd(close)
            rsi_14 = rsi(close, 14)
            bb_middle, bb_upper, bb_lower = bollinger_bands(close, 20, 2)
            stoch_k, stoch_d = stochastic_kd(high, low, close, 14, 3)
            williams_14 = williams_r(high, low, close, 14)
            atr_14 = atr(high, low, close, 14)
            
            score = 0.0
            
            # 1. TENDANCE (20 points max)
            trend_score = self._analyze_trend(
                current_price, sma_20, sma_50, sma_200
            )
            
            # 2. MOMENTUM (20 points max)
            momentum_score = self._analyze_momentum(
                rsi_14, macd_line, macd_signal, macd_histogram
            )
            
            # 3. VOLATILITÉ (15 points max)
            volatility_score = self._analyze_volatility(
                current_price, atr_14
            )
            
            # 4. NIVEAUX CLÉS (15 points max)
            levels_score = self._analyze_levels(
                current_price, bb_upper, bb_lower, stoch_k, williams_14
            )
            
            # 5. VOLUME (10 points max)
            volume_score = self._analyze_volume(volume)
            
            # 6. SIGNAUX MULTIPLES (20 points max)
            signals_score = self._analyze_signals(
                sma_20, sma_50, macd_histogram, rsi_14, stoch_k
            )
            
            total_score = (trend_score + momentum_score + volatility_score + 
                          levels_score + volume_score + signals_score)
            
            # Limiter entre 0 et 100
            final_score = min(100.0, max(0.0, total_score))
            
            logger.debug(f"Scores technique {ticker}: Trend={trend_score:.1f}, "
                        f"Momentum={momentum_score:.1f}, Vol={volatility_score:.1f}, "
                        f"Levels={levels_score:.1f}, Volume={volume_score:.1f}, "
                        f"Signals={signals_score:.1f}, Total={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Erreur analyse technique {ticker}: {e}")
            return 50.0
    
    def _analyze_trend(self, price: float, sma20: pd.Series, 
                      sma50: pd.Series, sma200: pd.Series) -> float:
        """Analyse de tendance basée sur les moyennes mobiles"""
        try:
            sma20_val = float(sma20.iloc[-1])
            sma50_val = float(sma50.iloc[-1])
            sma200_val = float(sma200.iloc[-1])
            
            if price > sma20_val > sma50_val > sma200_val:
                return 20.0  # Tendance haussière parfaite
            elif price > sma20_val > sma50_val:
                return 15.0  # Tendance haussière court terme
            elif price > sma50_val:
                return 10.0  # Tendance neutre positive
            elif price < sma20_val < sma50_val < sma200_val:
                return 2.0   # Tendance baissière
            else:
                return 7.0   # Tendance mixte
        except:
            return 10.0
    
    def _analyze_momentum(self, rsi: pd.Series, macd_line: pd.Series,
                         macd_signal: pd.Series, macd_hist: pd.Series) -> float:
        """Analyse du momentum"""
        score = 0.0
        
        try:
            # RSI (0-12 points)
            rsi_val = float(rsi.iloc[-1])
            if rsi_val < 30:
                score += 12.0  # RSI survente = opportunité
            elif 40 <= rsi_val <= 60:
                score += 8.0   # RSI neutre = stable
            elif 30 <= rsi_val <= 70:
                score += 6.0   # RSI acceptable
            else:
                score += 4.0   # RSI surachat = prudence
            
            # MACD (0-8 points)
            if len(macd_hist) > 1:
                macd_current = float(macd_hist.iloc[-1])
                macd_prev = float(macd_hist.iloc[-2])
                
                if macd_current > 0 and macd_current > macd_prev:
                    score += 8.0  # MACD positif et croissant
                elif macd_current > 0:
                    score += 6.0  # MACD positif
                elif macd_current > macd_prev:
                    score += 4.0  # MACD s'améliore
                else:
                    score += 2.0  # MACD faible
            
        except:
            score = 10.0  # Score neutre en cas d'erreur
        
        return min(20.0, score)
    
    def _analyze_volatility(self, price: float, atr: pd.Series) -> float:
        """Analyse de la volatilité (score élevé = faible volatilité)"""
        try:
            atr_val = float(atr.iloc[-1])
            atr_pct = (atr_val / price) * 100 if price > 0 else 0
            
            if atr_pct < 2:
                return 15.0  # Faible volatilité = bien pour la stabilité
            elif atr_pct < 4:
                return 10.0  # Volatilité modérée
            elif atr_pct < 6:
                return 6.0   # Volatilité élevée
            else:
                return 3.0   # Très haute volatilité = risqué
        except:
            return 7.0
    
    def _analyze_levels(self, price: float, bb_upper: pd.Series, 
                       bb_lower: pd.Series, stoch_k: pd.Series, 
                       williams: pd.Series) -> float:
        """Analyse des niveaux de support/résistance"""
        score = 0.0
        
        try:
            # Bollinger Bands position (0-8 points)
            bb_up_val = float(bb_upper.iloc[-1])
            bb_low_val = float(bb_lower.iloc[-1])
            bb_position = (price - bb_low_val) / (bb_up_val - bb_low_val) if bb_up_val > bb_low_val else 0.5
            
            if 0.2 <= bb_position <= 0.8:
                score += 8.0  # Position normale
            elif bb_position < 0.2:
                score += 6.0  # Proche du support = opportunité
            else:
                score += 4.0  # Proche de la résistance = prudence
            
            # Stochastique (0-4 points)
            stoch_val = float(stoch_k.iloc[-1])
            if stoch_val < 20:
                score += 4.0  # Stoch oversold
            elif 20 <= stoch_val <= 80:
                score += 3.0  # Stoch normal
            else:
                score += 2.0  # Stoch overbought
            
            # Williams %R (0-3 points)
            williams_val = float(williams.iloc[-1])
            if williams_val < -80:
                score += 3.0  # Williams oversold
            elif -80 <= williams_val <= -20:
                score += 2.0  # Williams normal
            else:
                score += 1.0  # Williams overbought
            
        except:
            score = 7.0
        
        return min(15.0, score)
    
    def _analyze_volume(self, volume: pd.Series) -> float:
        """Analyse du volume"""
        try:
            if len(volume) < 20:
                return 5.0
            
            current_vol = float(volume.iloc[-1])
            avg_vol = float(volume.rolling(20).mean().iloc[-1])
            
            volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            if volume_ratio > 1.5:
                return 10.0  # Volume élevé = conviction
            elif volume_ratio > 1.2:
                return 8.0   # Volume correct
            elif volume_ratio > 0.8:
                return 6.0   # Volume normal
            else:
                return 4.0   # Volume faible
        except:
            return 5.0
    
    def _analyze_signals(self, sma20: pd.Series, sma50: pd.Series, 
                        macd_hist: pd.Series, rsi: pd.Series, 
                        stoch_k: pd.Series) -> float:
        """Détection de signaux multiples"""
        signals = 0.0
        
        try:
            # Signal golden cross (0-8 points)
            if len(sma20) > 1 and len(sma50) > 1:
                sma20_current = float(sma20.iloc[-1])
                sma50_current = float(sma50.iloc[-1])
                sma20_prev = float(sma20.iloc[-2])
                sma50_prev = float(sma50.iloc[-2])
                
                if sma20_prev <= sma50_prev and sma20_current > sma50_current:
                    signals += 8.0  # Golden cross
                elif sma20_current > sma50_current:
                    signals += 4.0  # Maintien au-dessus
            
            # Signal RSI divergence (0-6 points)
            if len(rsi) > 1:
                rsi_current = float(rsi.iloc[-1])
                if rsi_current < 35:  # Zone de survente
                    signals += 6.0
                elif rsi_current > 65:  # Zone de surachat
                    signals += 3.0
            
            # Signal MACD momentum (0-6 points)
            if len(macd_hist) > 2:
                macd_current = float(macd_hist.iloc[-1])
                macd_prev = float(macd_hist.iloc[-2])
                if macd_current > macd_prev and macd_current > 0:
                    signals += 6.0
                elif macd_current > macd_prev:
                    signals += 3.0
            
        except Exception as e:
            logger.debug(f"Erreur détection signaux: {e}")
            signals = 8.0  # Score neutre
        
        return min(20.0, signals)