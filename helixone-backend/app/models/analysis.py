"""
Modèle Analysis - Table des analyses effectuées
"""

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


def generate_uuid():
    """Génère un UUID unique"""
    return str(uuid.uuid4())


class Analysis(Base):
    """
    Table des analyses
    
    Attributs:
        id: Identifiant unique (UUID)
        user_id: ID de l'utilisateur qui a fait l'analyse
        ticker: Symbole de l'action analysée (ex: AAPL)
        mode: Mode d'analyse (Standard, Conservative, Aggressive)
        score_final: Score FXI final (0-100)
        score_technique: Score technique (0-100)
        score_fondamental: Score fondamental (0-100)
        score_sentiment: Score sentiment (0-100)
        score_risque: Score risque (0-100)
        score_macro: Score macro-économique (0-100)
        recommendation: Recommandation (ACHAT, VENTE, etc.)
        confidence: Niveau de confiance (0-100)
        data_quality: Qualité des données (0-1)
        execution_time: Temps d'exécution en secondes
        result_json: Résultats complets en JSON
        created_at: Date de l'analyse
    
    Relations:
        user: Utilisateur qui a effectué l'analyse
    """
    
    __tablename__ = "analyses"
    
    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Informations de l'analyse
    ticker = Column(String, nullable=False, index=True)
    mode = Column(String, default="Standard")
    
    # Scores
    score_final = Column(Float, nullable=True)
    score_technique = Column(Float, nullable=True)
    score_fondamental = Column(Float, nullable=True)
    score_sentiment = Column(Float, nullable=True)
    score_risque = Column(Float, nullable=True)
    score_macro = Column(Float, nullable=True)
    
    # Résultats
    recommendation = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    data_quality = Column(Float, nullable=True)
    
    # Performance
    execution_time = Column(Float, nullable=True)
    
    # Données complètes
    result_json = Column(JSON, nullable=True)
    
    # Date
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relations
    # user = relationship("User", back_populates="analyses")
    
    def __repr__(self):
        return f"<Analysis {self.ticker} - Score: {self.score_final}>"