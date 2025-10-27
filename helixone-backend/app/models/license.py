"""
Modèle License - Table des licences/abonnements
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import uuid

from app.core.database import Base


def generate_uuid():
    """Génère un UUID unique"""
    return str(uuid.uuid4())


def generate_license_key():
    """Génère une clé de licence unique"""
    return f"HX-{uuid.uuid4().hex[:8].upper()}-{uuid.uuid4().hex[:8].upper()}"


class License(Base):
    """
    Table des licences
    
    Attributs:
        id: Identifiant unique (UUID)
        user_id: ID de l'utilisateur propriétaire
        license_key: Clé de licence unique
        machine_id: ID de la machine (pour empêcher le partage)
        license_type: Type (trial, basic, premium, professional)
        status: Statut (active, expired, suspended, cancelled)
        features: Liste des fonctionnalités activées (JSON)
        quota_daily_analyses: Nombre d'analyses par jour autorisées
        quota_daily_api_calls: Nombre d'appels API par jour
        created_at: Date de création
        activated_at: Date d'activation
        expires_at: Date d'expiration
        last_validated_at: Dernière vérification
        stripe_subscription_id: ID abonnement Stripe (si applicable)
        stripe_customer_id: ID client Stripe
    
    Relations:
        user: Utilisateur propriétaire
    """
    
    __tablename__ = "licenses"
    
    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    license_key = Column(String, unique=True, default=generate_license_key)
    machine_id = Column(String, nullable=True, index=True)
    
    # Type et statut
    license_type = Column(String, nullable=False, default="trial")  # trial, basic, premium, professional
    status = Column(String, nullable=False, default="active")  # active, expired, suspended, cancelled
    
    # Fonctionnalités et quotas
    features = Column(JSON, nullable=True)  # {"alerts": true, "exports": true, ...}
    quota_daily_analyses = Column(Integer, default=50)
    quota_daily_api_calls = Column(Integer, default=200)
    
    # Dates
    created_at = Column(DateTime, default=datetime.utcnow)
    activated_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    last_validated_at = Column(DateTime, nullable=True)
    
    # Stripe
    stripe_subscription_id = Column(String, nullable=True)
    stripe_customer_id = Column(String, nullable=True)
    
    # Relations
    # user = relationship("User", back_populates="licenses")
    
    def __repr__(self):
        return f"<License {self.license_key} - {self.license_type}>"
    
    def is_valid(self):
        """Vérifie si la licence est valide"""
        if self.status != "active":
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def days_remaining(self):
        """Retourne le nombre de jours restants"""
        if not self.expires_at:
            return None
        
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.days)