"""
Modèle UserProgress - Progression utilisateur dans les formations
"""

from sqlalchemy import Column, String, Integer, JSON, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base
from app.core.time_utils import utc_now_naive


def generate_uuid():
    """Génère un UUID unique"""
    return str(uuid.uuid4())


class UserProgress(Base):
    """
    Table de progression des utilisateurs dans les formations

    Attributs:
        id: Identifiant unique (UUID)
        user_id: ID de l'utilisateur (FK vers users.id)
        total_xp: Total de points XP accumulés
        level: Niveau actuel
        completed_modules: Liste des modules complétés (JSON)
        module_scores: Scores aux quiz (JSON: {module_id: score})
        current_streak: Nombre de jours consécutifs de formation
        last_activity: Date de dernière activité
        created_at: Date de création
        updated_at: Date de dernière mise à jour
    """

    __tablename__ = "user_progress"

    # Colonnes
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True, unique=True)

    # Progression
    total_xp = Column(Integer, default=0)
    level = Column(Integer, default=1)
    completed_modules = Column(JSON, default=list)  # Liste des IDs de modules complétés
    module_scores = Column(JSON, default=dict)  # {module_id: {"score": 85, "time_spent": 120, "completed_at": "2026-01-15"}}
    badges = Column(JSON, default=list)  # Liste des badges obtenus
    certifications = Column(JSON, default=list)  # Liste des certifications obtenues

    # Statistiques
    current_streak = Column(Integer, default=0)
    last_activity_date = Column(DateTime, nullable=True)

    # Métadonnées
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)

    def __repr__(self):
        return f"<UserProgress user_id={self.user_id} level={self.level} xp={self.total_xp}>"

    def to_dict(self):
        """Convertit l'objet en dictionnaire"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "total_xp": self.total_xp,
            "level": self.level,
            "completed_modules": self.completed_modules or [],
            "module_scores": self.module_scores or {},
            "badges": self.badges or [],
            "certifications": self.certifications or [],
            "current_streak": self.current_streak,
            "last_activity_date": self.last_activity_date.isoformat() if self.last_activity_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
