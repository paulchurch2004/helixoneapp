"""
Nouveau système de rapports utilisant le moteur FXI v2.0
"""

from src.fxi_engine.reports.generator import ReportGenerator
from src.fxi_engine.core.config import DEFAULT_CONFIG
from datetime import datetime
from dataclasses import dataclass

@dataclass
class CompatibilityResult:
    """Classe de compatibilité pour l'ancien format"""
    ticker: str
    timestamp: datetime
    final_score: float
    recommendation: str
    confidence: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    risk_score: float
    macro_score: float  # Ajout du score macro
    details: dict
    execution_time: float
    data_quality: float

def generer_rapport_v3(ticker: str, mode: str, result: dict) -> str:
    """
    Interface de compatibilité avec l'ancien système
    Utilise maintenant le nouveau générateur de rapports FXI v2.0
    """
    
    # Convertir l'ancien format vers le nouveau
    compat_result = CompatibilityResult(
        ticker=ticker,
        timestamp=datetime.now(),
        final_score=result.get('score_fxi', result.get('score', 50)),
        recommendation=result.get('recommandation', 'CONSERVER'),
        confidence=result.get('confidence', 50),
        technical_score=result.get('scores_detailles', {}).get('technique', 50),
        fundamental_score=result.get('scores_detailles', {}).get('fondamental', 50),
        sentiment_score=result.get('scores_detailles', {}).get('sentiment', 50),
        risk_score=result.get('scores_detailles', {}).get('risque', 50),
        macro_score=result.get('scores_detailles', {}).get('macro', 50),  # Ajout du score macro
        details={
            'current_price': result.get('fondamentaux', {}).get('prix'),
            'pe_ratio': result.get('fondamentaux', {}).get('PE'),
            'sector': result.get('fondamentaux', {}).get('secteur'),
            'industry': result.get('fondamentaux', {}).get('industrie'),
            'macro_environment': result.get('macro_environment', 'N/A'),  # Ajout environnement macro
        },
        execution_time=result.get('execution_time', 0),
        data_quality=result.get('data_quality', 1.0)
    )
    
    # Utiliser le nouveau générateur
    generator = ReportGenerator(DEFAULT_CONFIG)
    return generator.generate(compat_result, "detailed")

def generer_commentaire_strategique(ticker: str, mode: str, result: dict) -> str:
    """
    Commentaire stratégique simplifié
    """
    score = result.get('score_fxi', result.get('score', 50))
    reco = result.get('recommandation', 'CONSERVER')
    
    if score >= 70:
        return f"L'analyse de {ticker} en mode {mode} montre un potentiel d'investissement favorable avec un score de {score}/100. La recommandation {reco} s'appuie sur des indicateurs globalement positifs."
    elif score >= 45:
        return f"L'analyse de {ticker} révèle une situation mitigée avec un score de {score}/100. La recommandation {reco} suggère une approche prudente."
    else:
        return f"L'analyse de {ticker} indique des signaux défavorables avec un score de {score}/100. La recommandation {reco} conseille la prudence."