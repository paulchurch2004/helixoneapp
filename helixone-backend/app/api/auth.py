
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.core.email import email_service
from app.models import User, License, PasswordResetToken
from app.schemas.user import UserRegister, UserLogin, UserResponse, TokenResponse
from pydantic import BaseModel, EmailStr

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("3/hour")
def register(request: Request, user_data: UserRegister, db: Session = Depends(get_db)):
    """Créer un nouveau compte utilisateur"""
    
    # Vérifier si l'email existe déjà
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cet email est déjà utilisé"
        )
    
    # Créer l'utilisateur
    new_user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        is_active=True,
        email_verified=False
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Créer une licence d'essai
    trial_license = License(
        user_id=new_user.id,
        license_type="trial",
        status="active",
        features=["basic_analysis", "dashboard"],
        quota_daily_analyses=10,
        quota_daily_api_calls=50,
        activated_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=14)
    )
    
    db.add(trial_license)
    db.commit()

    # Envoyer l'email de bienvenue
    email_service.send_welcome_email(
        to_email=new_user.email,
        first_name=new_user.first_name
    )

    # Créer le token JWT
    access_token = create_access_token(
        data={"user_id": new_user.id, "email": new_user.email}
    )
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(new_user)
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, credentials: UserLogin, db: Session = Depends(get_db)):
    """Connexion utilisateur"""
    
    # Trouver l'utilisateur
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )
    
    # Vérifier le mot de passe
    if not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )
    
    # Vérifier que le compte est actif
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compte désactivé. Contactez le support."
        )
    
    # Mettre à jour last_login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Créer le token JWT
    access_token = create_access_token(
        data={"user_id": user.id, "email": user.email}
    )
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user)
    )


# Schémas pour reset password
class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    reset_code: str
    new_password: str


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
@limiter.limit("3/hour")
def forgot_password(request: Request, data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Génère un code de réinitialisation de mot de passe"""

    # Vérifier que l'utilisateur existe
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        # Pour des raisons de sécurité, on ne révèle pas si l'email existe
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucun compte associé à cet email"
        )

    # Supprimer les anciens tokens non utilisés pour cet email
    db.query(PasswordResetToken).filter(
        PasswordResetToken.email == data.email,
        PasswordResetToken.used == False
    ).delete()
    db.commit()

    # Créer un nouveau token
    reset_token = PasswordResetToken(email=data.email)
    db.add(reset_token)
    db.commit()

    # Envoyer le code par email
    email_service.send_password_reset_email(
        to_email=data.email,
        reset_code=reset_token.reset_code
    )

    return ForgotPasswordResponse(
        message="Code de réinitialisation envoyé par email"
    )


@router.post("/reset-password")
@limiter.limit("5/hour")
def reset_password(request: Request, data: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Réinitialise le mot de passe avec le code"""

    # Trouver le token valide
    reset_token = db.query(PasswordResetToken).filter(
        PasswordResetToken.email == data.email,
        PasswordResetToken.reset_code == data.reset_code,
        PasswordResetToken.used == False
    ).first()

    if not reset_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Code de réinitialisation invalide ou expiré"
        )

    # Vérifier l'expiration
    if not reset_token.is_valid():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Code de réinitialisation expiré"
        )

    # Trouver l'utilisateur
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )

    # Mettre à jour le mot de passe
    user.password_hash = hash_password(data.new_password)
    reset_token.used = True

    db.commit()

    return {"message": "Mot de passe réinitialisé avec succès"}
