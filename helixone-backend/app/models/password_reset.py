"""
Modèle pour les tokens de réinitialisation de mot de passe
"""

from sqlalchemy import Column, String, DateTime, Boolean
from datetime import datetime, timedelta
from app.core.database import Base
import secrets


class PasswordResetToken(Base):
    """Token de réinitialisation de mot de passe"""

    __tablename__ = "password_reset_tokens"

    id = Column(String, primary_key=True)
    email = Column(String, nullable=False, index=True)
    reset_code = Column(String, nullable=False, unique=True)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

    def __init__(self, email: str, **kwargs):
        super().__init__(**kwargs)
        self.id = secrets.token_urlsafe(16)
        self.email = email
        self.reset_code = self.generate_reset_code()
        self.expires_at = datetime.utcnow() + timedelta(hours=1)  # Expire dans 1h

    @staticmethod
    def generate_reset_code() -> str:
        """Génère un code de réinitialisation à 6 chiffres"""
        return ''.join([str(secrets.randbelow(10)) for _ in range(6)])

    def is_valid(self) -> bool:
        """Vérifie si le token est toujours valide"""
        return not self.used and datetime.utcnow() < self.expires_at

    def __repr__(self):
        return f"<PasswordResetToken(email={self.email}, valid={self.is_valid()})>"
