"""
Analyseur fondamental complet
"""

import logging
from typing import Dict, Optional

from ..core.config import EngineConfig

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """Analyse fondamentale approfondie (0-100 points)"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
    
    def analyze(self, data: Dict, ticker: str) -> float:
        """
        Analyse fondamentale avec 4 dimensions :
        1. Ratios de valorisation (25 points)
        2. Croissance (25 points)
        3. Rentabilité (25 points)
        4. Solidité financière (25 points)
        """
        try:
            # Récupérer les informations financières
            info = data.get('yahoo', {}).get('info', {})
            if not info:
                logger.warning(f"Pas d'informations financières pour {ticker}")
                return 50.0
            
            score = 0.0
            
            # 1. VALORISATION (25 points max)
            valuation_score = self._analyze_valuation(info)
            
            # 2. CROISSANCE (25 points max)
            growth_score = self._analyze_growth(info)
            
            # 3. RENTABILITÉ (25 points max)
            profitability_score = self._analyze_profitability(info)
            
            # 4. SOLIDITÉ FINANCIÈRE (25 points max)
            financial_score = self._analyze_financial_strength(info)
            
            total_score = valuation_score + growth_score + profitability_score + financial_score
            final_score = min(100.0, max(0.0, total_score))
            
            logger.debug(f"Scores fondamental {ticker}: Val={valuation_score:.1f}, "
                        f"Growth={growth_score:.1f}, Prof={profitability_score:.1f}, "
                        f"Fin={financial_score:.1f}, Total={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Erreur analyse fondamentale {ticker}: {e}")
            return 50.0
    
    def _analyze_valuation(self, info: Dict) -> float:
        """Analyse des ratios de valorisation"""
        score = 0.0
        
        # P/E Ratio (0-15 points)
        pe_ratio = info.get('forwardPE') or info.get('trailingPE')
        if pe_ratio and 0 < pe_ratio < 100:
            if pe_ratio < 15:
                score += 15.0  # PE très attractif
            elif pe_ratio < 25:
                score += 10.0  # PE correct
            elif pe_ratio < 40:
                score += 6.0   # PE élevé mais acceptable
            else:
                score += 3.0   # PE très élevé
        
        # PEG Ratio (0-10 points)
        peg_ratio = info.get('pegRatio')
        if peg_ratio and 0 < peg_ratio < 10:
            if peg_ratio < 1:
                score += 10.0  # PEG excellent (croissance > valorisation)
            elif peg_ratio < 2:
                score += 7.0   # PEG bon
            else:
                score += 3.0   # PEG élevé
        
        return min(25.0, score)
    
    def _analyze_growth(self, info: Dict) -> float:
        """Analyse de la croissance"""
        score = 0.0
        
        # Croissance des revenus (0-15 points)
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth is not None:
            if revenue_growth > 0.20:
                score += 15.0  # Croissance forte >20%
            elif revenue_growth > 0.10:
                score += 10.0  # Croissance bonne >10%
            elif revenue_growth > 0.05:
                score += 6.0   # Croissance modérée >5%
            elif revenue_growth > 0:
                score += 3.0   # Croissance positive
            # Pas de points si croissance négative
        
        # Croissance des bénéfices (0-10 points)
        earnings_growth = info.get('earningsGrowth')
        if earnings_growth is not None:
            if earnings_growth > 0.15:
                score += 10.0  # Croissance forte >15%
            elif earnings_growth > 0.05:
                score += 7.0   # Croissance correcte >5%
            elif earnings_growth > 0:
                score += 3.0   # Croissance positive
        
        return min(25.0, score)
    
    def _analyze_profitability(self, info: Dict) -> float:
        """Analyse de la rentabilité"""
        score = 0.0
        
        # ROE - Return on Equity (0-12 points)
        roe = info.get('returnOnEquity')
        if roe is not None:
            if roe > 0.20:
                score += 12.0  # ROE excellent >20%
            elif roe > 0.15:
                score += 9.0   # ROE très bon >15%
            elif roe > 0.10:
                score += 6.0   # ROE bon >10%
            elif roe > 0.05:
                score += 3.0   # ROE acceptable >5%
        
        # Marges bénéficiaires (0-8 points)
        profit_margins = info.get('profitMargins')
        if profit_margins is not None:
            if profit_margins > 0.20:
                score += 8.0   # Marges excellentes >20%
            elif profit_margins > 0.10:
                score += 6.0   # Marges bonnes >10%
            elif profit_margins > 0.05:
                score += 4.0   # Marges correctes >5%
            elif profit_margins > 0:
                score += 2.0   # Marges positives
        
        # Bonus marges brutes élevées (0-5 points)
        gross_margins = info.get('grossMargins')
        if gross_margins is not None and gross_margins > 0.30:
            score += 5.0  # Bonus pour marges brutes >30%
        
        return min(25.0, score)
    
    def _analyze_financial_strength(self, info: Dict) -> float:
        """Analyse de la solidité financière"""
        score = 0.0
        
        # Ratio d'endettement (0-12 points)
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                score += 12.0  # Très peu d'endettement
            elif debt_to_equity < 0.6:
                score += 9.0   # Endettement modéré
            elif debt_to_equity < 1.0:
                score += 5.0   # Endettement acceptable
            elif debt_to_equity < 2.0:
                score += 2.0   # Endettement élevé
            # Pas de points si endettement très élevé
        
        # Ratio de liquidité (0-6 points)
        current_ratio = info.get('currentRatio')
        if current_ratio is not None:
            if current_ratio > 2:
                score += 6.0   # Liquidité excellente
            elif current_ratio > 1.5:
                score += 4.0   # Liquidité bonne
            elif current_ratio > 1:
                score += 2.0   # Liquidité correcte
        
        # Cash flow libre positif (0-7 points)
        free_cashflow = info.get('freeCashflow')
        if free_cashflow is not None and free_cashflow > 0:
            score += 7.0  # Cash flow libre positif
        
        return min(25.0, score)