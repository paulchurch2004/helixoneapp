"""
Portfolio Analysis System
Systeme d'analyse automatique de portefeuille avec recommandations
"""

from app.services.portfolio.data_aggregator import DataAggregator, get_data_aggregator

__all__ = [
    'DataAggregator',
    'get_data_aggregator',
]
