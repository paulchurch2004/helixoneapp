"""
Validateurs pour les données d'entrée
"""

import re
from typing import Optional

def validate_ticker(ticker: str) -> bool:
    """
    Valide le format d'un ticker
    
    Args:
        ticker: Symbole à valider
        
    Returns:
        bool: True si le ticker est valide
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Nettoyer le ticker
    clean_ticker = ticker.upper().strip()
    
    # Format accepté : 1-10 caractères alphanumériques, points, tirets
    pattern = r'^[A-Z0-9\.-]{1,10}$'
    
    return bool(re.match(pattern, clean_ticker))

def validate_mode(mode: str) -> Optional[str]:
    """
    Valide et normalise le mode d'analyse
    
    Args:
        mode: Mode à valider
        
    Returns:
        str: Mode normalisé ou None si invalide
    """
    if not mode or not isinstance(mode, str):
        return None
    
    mode_lower = mode.lower().strip()
    
    valid_modes = {
        'standard': 'Standard',
        'conservative': 'Conservative',
        'aggressive': 'Aggressive',
        'court terme': 'Court Terme',
        'long terme': 'Long Terme'
    }
    
    return valid_modes.get(mode_lower)

def validate_score(score: float) -> float:
    """
    Valide et normalise un score
    
    Args:
        score: Score à valider
        
    Returns:
        float: Score normalisé entre 0 et 100
    """
    if score is None or not isinstance(score, (int, float)):
        return 50.0  # Score neutre par défaut
    
    return min(100.0, max(0.0, float(score)))