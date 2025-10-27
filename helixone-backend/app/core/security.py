"""
Sécurité : JWT tokens et hash des mots de passe
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db


# ==================================
# Hash des mots de passe avec bcrypt direct
# ==================================

def hash_password(password: str) -> str:
    """
    Hash un mot de passe avec bcrypt
    
    Exemple:
        hashed = hash_password("MonMotDePasse123!")
        # Résultat : "$2b$12$abc123..."
    """
    # Convertir en bytes et limiter à 72 bytes (limite bcrypt)
    password_bytes = password.encode('utf-8')[:72]
    
    # Générer le hash
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Retourner en string
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Vérifie qu'un mot de passe correspond au hash
    
    Exemple:
        is_valid = verify_password("MonMotDePasse123!", "$2b$12$abc123...")
        # Résultat : True ou False
    """
    try:
        # Convertir en bytes
        password_bytes = plain_password.encode('utf-8')[:72]
        hashed_bytes = hashed_password.encode('utf-8')
        
        # Vérifier
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False


# ==================================
# JWT Tokens
# ==================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Crée un JWT token
    
    Args:
        data: Données à encoder (ex: {"user_id": "123"})
        expires_delta: Durée de validité (optionnel)
    
    Returns:
        Token JWT signé
    
    Exemple:
        token = create_access_token({"user_id": "123"})
        # Résultat : "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Décode et vérifie un JWT token

    Args:
        token: Token JWT à vérifier

    Returns:
        Payload décodé si valide, None sinon

    Exemple:
        payload = decode_access_token("eyJhbGc...")
        # Résultat : {"user_id": "123", "exp": 1234567890} ou None
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None


# ==================================
# Authentification FastAPI
# ==================================

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Récupère l'utilisateur actuel à partir du token JWT

    Args:
        credentials: Token JWT Bearer
        db: Session de base de données

    Returns:
        User: Utilisateur authentifié

    Raises:
        HTTPException: Si le token est invalide ou l'utilisateur n'existe pas

    Usage:
        @router.get("/protected")
        async def protected_route(current_user: User = Depends(get_current_user)):
            return {"user": current_user.email}
    """
    # Import ici pour éviter les imports circulaires
    from app.models import User

    # Extraire le token
    token = credentials.credentials

    # Décoder le token
    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Récupérer l'user_id du payload (utilise "sub" ou "user_id" selon la convention)
    user_id: str = payload.get("sub") or payload.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Récupérer l'utilisateur en base (user_id est un UUID string)
    user = db.query(User).filter(User.id == user_id).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur non trouvé",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user