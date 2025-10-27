"""
Analyseur de sentiment marché
"""

import logging
from typing import Dict

from ..core.config import EngineConfig

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyse du sentiment marché (0-100 points)"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
    
    def analyze(self, data: Dict, ticker: str) -> float:
        """
        Analyse du sentiment avec 3 dimensions :
        1. Recommandations d'analystes (40 points)
        2. Objectifs de prix (30 points)
        3. Momentum institutionnel (30 points)
        """
        try:
            info = data.get('yahoo', {}).get('info', {})
            scraped_data = data.get('scraped', {})
            
            if not info:
                logger.warning(f"Pas d'informations de sentiment pour {ticker}")
                return 50.0
            
            score = 0.0
            
            # 1. RECOMMANDATIONS D'ANALYSTES (40 points max)
            analyst_score = self._analyze_analyst_recommendations(info)
            
            # 2. OBJECTIFS DE PRIX (30 points max)
            target_score = self._analyze_price_targets(info)
            
            # 3. MOMENTUM INSTITUTIONNEL (30 points max)
            institutional_score = self._analyze_institutional_momentum(info, scraped_data)
            
            total_score = analyst_score + target_score + institutional_score
            final_score = min(100.0, max(0.0, total_score))
            
            logger.debug(f"Scores sentiment {ticker}: Analysts={analyst_score:.1f}, "
                        f"Targets={target_score:.1f}, Inst={institutional_score:.1f}, "
                        f"Total={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Erreur analyse sentiment {ticker}: {e}")
            return 50.0
    
    def _analyze_analyst_recommendations(self, info: Dict) -> float:
        """Analyse des recommandations d'analystes"""
        recommendation_mean = info.get('recommendationMean')
        if recommendation_mean is None:
            return 20.0  # Score neutre
        
        # Yahoo donne: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
        if recommendation_mean <= 2.0:
            return 40.0  # Consensus très positif
        elif recommendation_mean <= 2.5:
            return 30.0  # Consensus positif
        elif recommendation_mean <= 3.5:
            return 20.0  # Consensus neutre
        else:
            return 10.0  # Consensus négatif
    
    def _analyze_price_targets(self, info: Dict) -> float:
        """Analyse des objectifs de prix"""
        target_mean_price = info.get('targetMeanPrice')
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if not target_mean_price or not current_price:
            return 15.0  # Score neutre
        
        # Calculer le potentiel de hausse
        upside = (target_mean_price - current_price) / current_price
        
        if upside > 0.20:
            return 30.0  # Potentiel >20%
        elif upside > 0.10:
            return 25.0  # Potentiel >10%
        elif upside > 0.05:
            return 20.0  # Potentiel >5%
        elif upside > 0:
            return 15.0  # Potentiel positif
        else:
            return 5.0   # Potentiel négatif
    
    def _analyze_institutional_momentum(self, info: Dict, scraped_data: Dict) -> float:
        """Analyse du momentum institutionnel"""
        score = 0.0
        
        # Détention institutionnelle (0-15 points)
        held_percent_institutions = info.get('heldPercentInstitutions')
        if held_percent_institutions is not None:
            if held_percent_institutions > 0.70:
                score += 15.0  # Forte détention institutionnelle >70%
            elif held_percent_institutions > 0.50:
                score += 12.0  # Bonne détention institutionnelle >50%
            elif held_percent_institutions > 0.30:
                score += 8.0   # Détention institutionnelle modérée >30%
            else:
                score += 5.0   # Faible détention institutionnelle
        
        # Ventes à découvert (0-10 points)
        short_percent = info.get('shortPercentOfFloat')
        if short_percent is not None:
            if short_percent < 0.05:
                score += 10.0  # Peu de ventes à découvert <5%
            elif short_percent < 0.10:
                score += 7.0   # Ventes à découvert modérées <10%
            elif short_percent < 0.20:
                score += 3.0   # Ventes à découvert élevées <20%
            # Pas de points si ventes à découvert très élevées
        
        # Détention par les insiders (0-5 points)
        held_percent_insiders = info.get('heldPercentInsiders')
        if held_percent_insiders is not None and held_percent_insiders > 0.10:
            score += 5.0  # Bonne détention par les insiders >10%
        
        return min(30.0, score)