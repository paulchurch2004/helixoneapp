"""
Configuration centralisée pour le moteur FXI v2.0 avec analyse macro
"""

from dataclasses import dataclass

@dataclass
class EngineConfig:
    """Configuration du moteur d'analyse"""
    
    # Poids des analyses (doivent totaliser 1.0) - MAINTENANT 5 DIMENSIONS
    technical_weight: float = 0.25      # Réduit de 30% à 25%
    fundamental_weight: float = 0.30    # Réduit de 40% à 30%
    sentiment_weight: float = 0.20      # Inchangé à 20%
    risk_weight: float = 0.10           # Inchangé à 10%
    macro_weight: float = 0.15          # NOUVEAU : 15% pour macro-économie
    
    # Seuils de recommandation
    strong_buy_threshold: float = 80.0
    buy_threshold: float = 65.0
    hold_threshold: float = 45.0
    sell_threshold: float = 30.0
    
    # Configuration des timeouts et limites
    data_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    
    # Sources de données activées
    use_yahoo: bool = True
    use_scraping: bool = True
    
    # Mode de fonctionnement
    debug_mode: bool = False
    cache_enabled: bool = False  # À activer plus tard
    cache_duration: int = 300
    
    def validate(self) -> bool:
        """Valide la configuration - maintenant avec 5 poids"""
        weights_sum = (
            self.technical_weight + 
            self.fundamental_weight + 
            self.sentiment_weight + 
            self.risk_weight + 
            self.macro_weight
        )
        return abs(weights_sum - 1.0) < 0.001
    
    @classmethod
    def create_conservative(cls) -> 'EngineConfig':
        """Configuration conservative (plus de fondamentaux et macro)"""
        return cls(
            technical_weight=0.20,
            fundamental_weight=0.35,
            sentiment_weight=0.15,
            risk_weight=0.15,
            macro_weight=0.15
        )
    
    @classmethod
    def create_aggressive(cls) -> 'EngineConfig':
        """Configuration agressive (plus de technique, moins de macro)"""
        return cls(
            technical_weight=0.35,
            fundamental_weight=0.25,
            sentiment_weight=0.20,
            risk_weight=0.10,
            macro_weight=0.10
        )
    
    @classmethod
    def create_macro_focused(cls) -> 'EngineConfig':
        """Configuration axée sur la macro-économie"""
        return cls(
            technical_weight=0.20,
            fundamental_weight=0.25,
            sentiment_weight=0.15,
            risk_weight=0.15,
            macro_weight=0.25
        )

DEFAULT_CONFIG = EngineConfig()