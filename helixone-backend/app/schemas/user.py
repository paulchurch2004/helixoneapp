"""
Schemas Pydantic pour les utilisateurs
"""

from pydantic import BaseModel, EmailStr, field_validator, Field
from datetime import datetime
from typing import Optional
import re


class UserRegister(BaseModel):
    """Schema pour l'inscription"""
    email: EmailStr
    password: str = Field(..., min_length=12, max_length=128)
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Valide la force du mot de passe"""
        if len(v) < 12:
            raise ValueError('Le mot de passe doit contenir au moins 12 caractères')

        if not re.search(r'[A-Z]', v):
            raise ValueError('Le mot de passe doit contenir au moins une majuscule')

        if not re.search(r'[a-z]', v):
            raise ValueError('Le mot de passe doit contenir au moins une minuscule')

        if not re.search(r'\d', v):
            raise ValueError('Le mot de passe doit contenir au moins un chiffre')

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Le mot de passe doit contenir au moins un caractère spécial')

        return v

    @field_validator('first_name', 'last_name')
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Valide les noms (pas de caractères spéciaux dangereux)"""
        if v is not None and v.strip():
            # Supprime les espaces en trop
            v = v.strip()
            # Interdit les caractères potentiellement dangereux
            if re.search(r'[<>{}[\]\\]', v):
                raise ValueError('Le nom contient des caractères non autorisés')
        return v


class UserLogin(BaseModel):
    """Schema pour la connexion"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema pour la réponse utilisateur"""
    id: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_active: bool
    email_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Schema pour la réponse de connexion"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
