"""
Analyseur de risque
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np

from ..core.config import EngineConfig

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """Analyse des risques (0-100 points, 100 = faible risque)"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
    
    def analyze(self, data: Dict, ticker: str) -> float:
        """
        Analyse des risques - On part de 100 et on retire des points :
        1. Volatilité historique (-30 points max)
        2. Ratio d'endettement (-25 points max)
        3. Liquidité (-20 points max)
        4. Concentration sectorielle (-15 points max)
        5. Taille de l'entreprise (-10 points max)
        """
        try:
            info = data.get('yahoo', {}).get('info', {})
            history = data.get('yahoo', {}).get('history', pd.DataFrame())
            
            if not info:
                logger.warning(f"Pas d'informations de risque pour {ticker}")
                return 50.0
            
            risk_score = 100.0  # On commence à 100 (faible risque)
            
            # 1. VOLATILITÉ HISTORIQUE (-30 points max)
            volatility_penalty = self._analyze_volatility(history)
            
            # 2. RATIO D'ENDETTEMENT (-25 points max)
            debt_penalty = self._analyze_debt_ratio(info)
            
            # 3. LIQUIDITÉ (-20 points max)
            liquidity_penalty = self._analyze_liquidity(info)
            
            # 4. CONCENTRATION SECTORIELLE (-15 points max)
            sector_penalty = self._analyze_sector_risk(info)
            
            # 5. TAILLE DE L'ENTREPRISE (-10 points max)
            size_penalty = self._analyze_company_size(info)
            
            # Calculer le score final
            total_penalty = (volatility_penalty + debt_penalty + liquidity_penalty + 
                           sector_penalty + size_penalty)
            final_score = max(0.0, risk_score - total_penalty)
            
            logger.debug(f"Pénalités risque {ticker}: Vol={volatility_penalty:.1f}, "
                        f"Debt={debt_penalty:.1f}, Liq={liquidity_penalty:.1f}, "
                        f"Sector={sector_penalty:.1f}, Size={size_penalty:.1f}, "
                        f"Final={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Erreur analyse risque {ticker}: {e}")
            return 50.0
    
    def _analyze_volatility(self, history: pd.DataFrame) -> float:
        """Analyse de la volatilité historique"""
        if history is None or history.empty or len(history) < 30:
            return 15.0  # Pénalité moyenne si pas de données
        
        try:
            # Calculer la volatilité annualisée
            returns = history['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))  # Volatilité annualisée
            
            if volatility > 0.40:
                return 30.0  # Très volatile >40%
            elif volatility > 0.30:
                return 20.0  # Volatile >30%
            elif volatility > 0.20:
                return 10.0  # Modérément volatile >20%
            else:
                return 0.0   # Faible volatilité = pas de pénalité
        except:
            return 15.0
    
    def _analyze_debt_ratio(self, info: Dict) -> float:
        """Analyse du ratio d'endettement"""
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is None:
            return 10.0  # Pénalité moyenne si pas de données
        
        if debt_to_equity > 2.0:
            return 25.0  # Endettement très élevé >200%
        elif debt_to_equity > 1.0:
            return 15.0  # Endettement élevé >100%
        elif debt_to_equity > 0.5:
            return 5.0   # Endettement modéré >50%
        else:
            return 0.0   # Faible endettement = pas de pénalité
    
    def _analyze_liquidity(self, info: Dict) -> float:
        """Analyse de la liquidité"""
        current_ratio = info.get('currentRatio')
        if current_ratio is None:
            return 10.0  # Pénalité moyenne si pas de données
        
        if current_ratio < 1.0:
            return 20.0  # Problèmes de liquidité <1.0
        elif current_ratio < 1.5:
            return 10.0  # Liquidité faible <1.5
        else:
            return 0.0   # Liquidité correcte = pas de pénalité
    
    def _analyze_sector_risk(self, info: Dict) -> float:
        """Analyse du risque sectoriel"""
        sector = info.get('sector', '')
        
        # Secteurs considérés comme plus risqués
        high_risk_sectors = [
            'Biotechnology', 'Real Estate', 'Energy', 'Materials', 
            'Consumer Discretionary'
        ]
        
        medium_risk_sectors = [
            'Industrials', 'Financials'
        ]
        
        if any(risky_sector in sector for risky_sector in high_risk_sectors):
            return 15.0  # Secteur à haut risque
        elif any(medium_sector in sector for medium_sector in medium_risk_sectors):
            return 8.0   # Secteur à risque modéré
        else:
            return 0.0   # Secteur défensif = pas de pénalité
    
    def _analyze_company_size(self, info: Dict) -> float:
        """Analyse de la taille de l'entreprise"""
        market_cap = info.get('marketCap')
        if market_cap is None:
            return 5.0  # Pénalité moyenne si pas de données
        
        if market_cap < 1e9:  # < 1 milliard
            return 10.0  # Small cap = plus risqué
        elif market_cap < 10e9:  # < 10 milliards
            return 5.0   # Mid cap = légèrement plus risqué
        else:
            return 0.0   # Large cap = pas de pénalité