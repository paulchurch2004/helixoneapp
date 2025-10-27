"""
Modèle User - Table des utilisateurs
"""

from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


def generate_uuid():
    """Génère un UUID unique"""
    return str(uuid.uuid4())


class User(Base):
    """
    Table des utilisateurs
    
    Attributs:
        id: Identifiant unique (UUID)
        email: Email unique de l'utilisateur
        password_hash: Mot de passe hashé avec bcrypt
        first_name: Prénom (optionnel)
        last_name: Nom (optionnel)
        is_active: Compte actif ou suspendu
        email_verified: Email vérifié ou non
        created_at: Date de création du compte
        last_login: Dernière connexion
    
    Relations:
        licenses: Liste des licences de l'utilisateur
        analyses: Liste des analyses effectuées
    """
    
    __tablename__ = "users"
    
    # Colonnes
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relations (définies plus tard)
    # licenses = relationship("License", back_populates="user")
    # analyses = relationship("Analysis", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.email}>"