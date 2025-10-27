"""
Schémas Pydantic pour les analyses FXI
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AnalysisMode(str, Enum):
    """Modes d'analyse disponibles"""
    STANDARD = "Standard"
    CONSERVATIVE = "Conservative"
    AGGRESSIVE = "Aggressive"


class AnalysisRequest(BaseModel):
    """Requête d'analyse"""
    ticker: str = Field(..., description="Symbole de l'action")
    mode: AnalysisMode = Field(
        default=AnalysisMode.STANDARD,
        description="Mode d'analyse"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "mode": "Standard"
            }
        }


class AnalysisResult(BaseModel):
    """Résultat d'analyse FXI complet"""

    ticker: str = Field(..., description="Symbole de l'action")
    timestamp: datetime = Field(..., description="Date et heure de l'analyse")

    # Score final et recommandation
    final_score: float = Field(..., ge=0, le=100, description="Score FXI final (0-100)")
    recommendation: str = Field(..., description="Recommandation (ACHAT FORT, ACHAT, CONSERVER, VENDRE, VENTE FORTE)")
    confidence: float = Field(..., ge=0, le=100, description="Niveau de confiance (0-100)")

    # Scores détaillés par dimension
    technical_score: float = Field(..., ge=0, le=100, description="Score analyse technique")
    fundamental_score: float = Field(..., ge=0, le=100, description="Score analyse fondamentale")
    sentiment_score: float = Field(..., ge=0, le=100, description="Score analyse sentiment")
    risk_score: float = Field(..., ge=0, le=100, description="Score analyse risque (100=faible risque)")
    macro_score: float = Field(..., ge=0, le=100, description="Score analyse macro-économique")

    # Métadonnées
    execution_time: float = Field(..., description="Temps d'exécution (secondes)")
    data_quality: float = Field(..., ge=0, le=1, description="Qualité des données (0-1)")

    # Détails optionnels
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Détails supplémentaires de l'analyse"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "timestamp": "2024-01-15T10:30:00",
                "final_score": 78.5,
                "recommendation": "ACHAT",
                "confidence": 85.2,
                "technical_score": 72.0,
                "fundamental_score": 85.0,
                "sentiment_score": 80.0,
                "risk_score": 75.0,
                "macro_score": 81.0,
                "execution_time": 3.45,
                "data_quality": 0.95,
                "details": {
                    "current_price": 178.50,
                    "market_cap": 2800000000000,
                    "sector": "Technology",
                    "pe_ratio": 28.5
                }
            }
        }


class AnalysisScoresBreakdown(BaseModel):
    """Détail des scores par catégorie"""

    technical: Dict[str, float] = Field(..., description="Détail scores techniques")
    fundamental: Dict[str, float] = Field(..., description="Détail scores fondamentaux")
    sentiment: Dict[str, float] = Field(..., description="Détail scores sentiment")
    risk: Dict[str, float] = Field(..., description="Détail scores risque")
    macro: Dict[str, float] = Field(..., description="Détail scores macro")


class AnalysisResponse(BaseModel):
    """Réponse complète d'analyse avec breakdown"""

    analysis: AnalysisResult
    breakdown: Optional[AnalysisScoresBreakdown] = None
    report: Optional[str] = Field(
        None,
        description="Rapport textuel de l'analyse"
    )
