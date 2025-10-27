"""
INTERFACE DE COMPATIBILITÉ
Redirige vers le nouveau moteur FXI v2.0
"""

from src.fxi_engine import get_analysis as new_get_analysis
import logging

logger = logging.getLogger(__name__)

def get_analysis(ticker: str, mode: str = "Standard"):
    """
    Interface de compatibilité avec l'ancien système
    Utilise maintenant le moteur FXI v2.0
    """
    try:
        return new_get_analysis(ticker, mode)
    except Exception as e:
        logger.error(f"Erreur analyse {ticker}: {e}")
        return {
            'score_fxi': 50,
            'score': 50,
            'recommandation': 'ERREUR',
            'status': 'error',
            'error': str(e),
            'indicateurs': {},
            'fondamentaux': {},
            'esg': {},
            'timestamp': None
        }