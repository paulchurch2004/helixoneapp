"""
Moteur principal d'analyse FXI v2.0 avec analyse macro-économique
"""

import logging
import time
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

from .config import EngineConfig, DEFAULT_CONFIG
from ..analyzers.technical import TechnicalAnalyzer
from ..analyzers.fundamental import FundamentalAnalyzer
from ..analyzers.sentiment import SentimentAnalyzer
from ..analyzers.risk import RiskAnalyzer
from ..analyzers.macro import MacroEconomicAnalyzer
from ..data.collectors import DataCollector
from ..utils.validators import validate_ticker
from ..reports.generator import ReportGenerator

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Résultat d'analyse complet avec macro-économie"""
    ticker: str
    timestamp: datetime
    final_score: float
    recommendation: str
    confidence: float
    
    # Scores détaillés (maintenant 5 dimensions)
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    risk_score: float
    macro_score: float
    
    # Données et métadonnées
    data: Dict
    details: Dict
    execution_time: float
    data_quality: float

class FXIEngine:
    """Moteur d'analyse financière professionnel avec analyse macro-économique"""
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        if not self.config.validate():
            raise ValueError("Configuration invalide : les poids ne totalisent pas 1.0")
        
        # Initialiser tous les composants (maintenant 5 analyseurs)
        self.data_collector = DataCollector(self.config)
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.fundamental_analyzer = FundamentalAnalyzer(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.risk_analyzer = RiskAnalyzer(self.config)
        self.macro_analyzer = MacroEconomicAnalyzer(self.config)  # NOUVEAU
        self.report_generator = ReportGenerator(self.config)
        
        logger.info("FXI Engine v2.0 initialisé avec analyse macro-économique")
    
    def analyze(self, ticker: str, mode: str = "Standard") -> AnalysisResult:
        """
        Analyse complète d'un ticker avec 5 dimensions
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Validation du ticker
            if not validate_ticker(ticker):
                raise ValueError(f"Ticker invalide: {ticker}")
            
            logger.info(f"Démarrage analyse complète {ticker} en mode {mode}")
            
            # Ajuster la config selon le mode
            working_config = self._adjust_config_for_mode(mode)
            
            # 1. Collecte des données
            raw_data = self.data_collector.collect_all_data(ticker)
            data_quality = self._calculate_data_quality(raw_data)
            
            if data_quality < 0.3:
                logger.warning(f"Qualité des données faible pour {ticker}: {data_quality:.2f}")
            
            # 2. Analyses par dimension (maintenant 5)
            technical_score = self.technical_analyzer.analyze(raw_data, ticker)
            fundamental_score = self.fundamental_analyzer.analyze(raw_data, ticker)
            sentiment_score = self.sentiment_analyzer.analyze(raw_data, ticker)
            risk_score = self.risk_analyzer.analyze(raw_data, ticker)
            macro_score = self.macro_analyzer.analyze(raw_data, ticker)
            
            # 3. Score final pondéré (5 dimensions)
            final_score = (
                technical_score * working_config.technical_weight +
                fundamental_score * working_config.fundamental_weight +
                sentiment_score * working_config.sentiment_weight +
                risk_score * working_config.risk_weight +
                macro_score * working_config.macro_weight
            )
            
            # 4. Recommandation et confiance
            recommendation = self._determine_recommendation(final_score, working_config)
            confidence = self._calculate_confidence(
                technical_score, fundamental_score, sentiment_score, 
                risk_score, macro_score, data_quality
            )
            
            # 5. Détails pour le rapport
            details = self._compile_details(raw_data, ticker, mode)
            
            execution_time = time.time() - start_time
            
            result = AnalysisResult(
                ticker=ticker,
                timestamp=timestamp,
                final_score=round(final_score, 2),
                recommendation=recommendation,
                confidence=round(confidence, 2),
                technical_score=round(technical_score, 2),
                fundamental_score=round(fundamental_score, 2),
                sentiment_score=round(sentiment_score, 2),
                risk_score=round(risk_score, 2),
                macro_score=round(macro_score, 2),
                data=raw_data,
                details=details,
                execution_time=execution_time,
                data_quality=data_quality
            )
            
            logger.info(f"Analyse {ticker} terminée: Score {final_score:.1f}, "
                       f"{recommendation}, Macro {macro_score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de {ticker}: {e}")
            return self._create_error_result(ticker, str(e), timestamp, time.time() - start_time)
    
    def generate_report(self, analysis: AnalysisResult, format: str = "detailed") -> str:
        """Génère un rapport d'analyse"""
        return self.report_generator.generate(analysis, format)
    
    def _adjust_config_for_mode(self, mode: str) -> EngineConfig:
        """Ajuste la configuration selon le mode"""
        if mode.lower() == "conservative":
            return EngineConfig.create_conservative()
        elif mode.lower() == "aggressive":
            return EngineConfig.create_aggressive()
        else:
            return self.config
    
    def _calculate_data_quality(self, data: Dict) -> float:
        """Calcule la qualité des données collectées"""
        total_sources = 0
        successful_sources = 0
        
        for source, source_data in data.items():
            if source.startswith('_'):  # Métadonnées internes
                continue
            total_sources += 1
            if source_data and not source_data.get('error'):
                successful_sources += 1
        
        return successful_sources / total_sources if total_sources > 0 else 0.0
    
    def _determine_recommendation(self, score: float, config: EngineConfig) -> str:
        """Détermine la recommandation finale"""
        if score >= config.strong_buy_threshold:
            return "ACHAT FORT"
        elif score >= config.buy_threshold:
            return "ACHAT"
        elif score >= config.hold_threshold:
            return "CONSERVER"
        elif score >= config.sell_threshold:
            return "VENDRE"
        else:
            return "VENTE FORTE"
    
    def _calculate_confidence(self, tech: float, fund: float, sent: float, 
                            risk: float, macro: float, data_quality: float) -> float:
        """Calcule le niveau de confiance dans l'analyse (maintenant 5 scores)"""
        scores = [tech, fund, sent, risk, macro]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        convergence = max(0, 100 - variance)  # Plus les scores sont proches, plus la confiance est élevée
        
        data_factor = data_quality * 100
        confidence = (convergence * 0.7 + data_factor * 0.3)
        return min(100, max(0, confidence))
    
    def _compile_details(self, data: Dict, ticker: str, mode: str) -> Dict:
        """Compile les détails pour le rapport"""
        details = {
            'ticker': ticker,
            'mode': mode,
            'analysis_timestamp': datetime.now().isoformat(),
            'engine_version': '2.0.0-macro'
        }
        
        # Extraire les métriques clés
        if 'yahoo' in data and data['yahoo'] and 'info' in data['yahoo']:
            info = data['yahoo']['info']
            details.update({
                'current_price': info.get('currentPrice'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            })
        
        # Ajouter des métriques macro si disponibles
        details.update({
            'macro_environment': self._get_macro_summary(),
            'sector_macro_correlation': self._get_sector_macro_correlation(
                details.get('sector', 'Unknown')
            )
        })
        
        return details
    
    def _get_macro_summary(self) -> str:
        """Résumé de l'environnement macro-économique"""
        try:
            # Simulation - en production, utiliser les vraies données
            return "Taux Fed: 5.25%, Inflation: 3.2%, Croissance PIB: 2.1%"
        except:
            return "Données macro non disponibles"
    
    def _get_sector_macro_correlation(self, sector: str) -> str:
        """Corrélation du secteur avec l'environnement macro"""
        correlations = {
            'Technology': 'Sensible aux taux d\'intérêt (-)',
            'Financials': 'Bénéficie des taux élevés (+)',
            'Real Estate': 'Très sensible aux taux (-)',
            'Energy': 'Corrélé à l\'inflation (+)',
            'Consumer Staples': 'Défensif face à l\'inflation',
            'Healthcare': 'Peu sensible aux cycles macro',
            'Materials': 'Corrélé à la croissance (+)',
            'Industrials': 'Corrélé à la croissance (+)',
            'Consumer Discretionary': 'Sensible à la confiance des consommateurs',
            'Utilities': 'Sensible aux taux d\'intérêt (-)'
        }
        return correlations.get(sector, 'Corrélation neutre')
    
    def _create_error_result(self, ticker: str, error: str, 
                           timestamp: datetime, execution_time: float) -> AnalysisResult:
        """Crée un résultat d'erreur"""
        return AnalysisResult(
            ticker=ticker,
            timestamp=timestamp,
            final_score=50.0,
            recommendation="ERREUR - DONNÉES INSUFFISANTES",
            confidence=0.0,
            technical_score=50.0,
            fundamental_score=50.0,
            sentiment_score=50.0,
            risk_score=50.0,
            macro_score=50.0,
            data={'error': error},
            details={'error': error},
            execution_time=execution_time,
            data_quality=0.0
        )

def analyze_ticker(ticker: str, mode: str = "Standard", 
                  config: Optional[EngineConfig] = None) -> AnalysisResult:
    """Interface simple pour analyser un ticker avec analyse macro"""
    engine = FXIEngine(config)
    return engine.analyze(ticker, mode)