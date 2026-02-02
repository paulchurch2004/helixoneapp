from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from datetime import timedelta
import pyotp

from app.core.time_utils import utc_now_naive
import qrcode
import io
import base64

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token, get_current_user
from app.core.email import email_service
from app.core.audit_logger import audit_log  # Audit logging
from app.models import User, License, PasswordResetToken
from app.schemas.user import UserRegister, UserLogin, UserResponse, TokenResponse
from pydantic import BaseModel, EmailStr
from typing import Optional

# Import du limiter global
from app.core.rate_limiter import limiter

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("3/hour")
def register(request: Request, user_data: UserRegister, db: Session = Depends(get_db)):
    """Créer un nouveau compte utilisateur"""
    
    # Vérifier si l'email existe déjà
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        # Audit log: tentative d'inscription avec email existant
        audit_log.log_suspicious_activity(
            user_id=None,
            activity_type="duplicate_registration",
            ip_address=request.client.host if hasattr(request, 'client') else None,
            details=f"Attempted registration with existing email: {user_data.email}"
        )
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
        activated_at=utc_now_naive(),
        expires_at=utc_now_naive() + timedelta(days=14)
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

    # Audit log: inscription réussie
    audit_log.log_auth_register(
        user_id=new_user.id,
        email=new_user.email,
        ip_address=request.client.host if hasattr(request, 'client') else None
    )

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(new_user)
    )


class UserLoginWith2FA(BaseModel):
    email: EmailStr
    password: str
    totp_code: Optional[str] = None


class Login2FARequiredResponse(BaseModel):
    requires_2fa: bool = True
    message: str = "Code 2FA requis"


@router.post("/login")
@limiter.limit("5/minute")
def login(request: Request, credentials: UserLoginWith2FA, db: Session = Depends(get_db)):
    """Connexion utilisateur avec support 2FA"""

    # Trouver l'utilisateur
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user:
        # Audit log: échec login - utilisateur inexistant
        audit_log.log_auth_failed(
            email=credentials.email,
            ip_address=request.client.host if hasattr(request, 'client') else None,
            reason="user_not_found"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )

    # Vérifier le mot de passe
    if not verify_password(credentials.password, user.password_hash):
        # Audit log: échec login - mauvais mot de passe
        audit_log.log_auth_failed(
            email=credentials.email,
            ip_address=request.client.host if hasattr(request, 'client') else None,
            reason="invalid_password"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect"
        )

    # Vérifier que le compte est actif
    if not user.is_active:
        # Audit log: tentative de connexion avec compte désactivé
        audit_log.log_access_denied(
            user_id=user.id,
            resource="login",
            ip_address=request.client.host if hasattr(request, 'client') else None,
            reason="account_disabled"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compte désactivé. Contactez le support."
        )

    # Vérifier 2FA si activé
    if user.totp_enabled and user.totp_secret:
        if not credentials.totp_code:
            # 2FA requis mais pas de code fourni
            return {"requires_2fa": True, "message": "Code 2FA requis"}

        # Vérifier le code TOTP
        totp = pyotp.TOTP(user.totp_secret)
        if not totp.verify(credentials.totp_code, valid_window=1):
            audit_log.log_auth_failed(
                email=credentials.email,
                ip_address=request.client.host if hasattr(request, 'client') else None,
                reason="invalid_2fa_code"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Code 2FA invalide"
            )

    # Mettre à jour last_login
    user.last_login = utc_now_naive()
    db.commit()

    # Créer le token JWT
    access_token = create_access_token(
        data={"user_id": user.id, "email": user.email}
    )

    # Audit log: connexion réussie
    audit_log.log_auth_success(
        user_id=user.id,
        ip_address=request.client.host if hasattr(request, 'client') else None,
        user_agent=request.headers.get("user-agent") if hasattr(request, 'headers') else None
    )

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user)
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Récupère les informations de l'utilisateur connecté

    Requiert: Token JWT valide dans le header Authorization

    Returns:
        UserResponse avec les informations de l'utilisateur
    """
    return UserResponse.model_validate(current_user)


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


# ============================================================================
# 2FA - Two-Factor Authentication
# ============================================================================

class Enable2FAResponse(BaseModel):
    secret: str
    qr_code: str  # Base64 encoded QR code
    provisioning_uri: str


class Verify2FARequest(BaseModel):
    totp_code: str


class TwoFAStatusResponse(BaseModel):
    enabled: bool
    message: str


@router.post("/2fa/setup", response_model=Enable2FAResponse)
def setup_2fa(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Génère un secret TOTP et un QR code pour configurer le 2FA.
    L'utilisateur doit scanner le QR code avec une app comme Google Authenticator.
    """
    if current_user.totp_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA déjà activé. Désactivez-le d'abord pour le reconfigurer."
        )

    # Générer un nouveau secret
    secret = pyotp.random_base32()

    # Créer l'URI de provisioning pour les apps authenticator
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=current_user.email,
        issuer_name="HelixOne"
    )

    # Générer le QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convertir en base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Sauvegarder temporairement le secret (pas encore activé)
    current_user.totp_secret = secret
    db.commit()

    return Enable2FAResponse(
        secret=secret,
        qr_code=f"data:image/png;base64,{qr_base64}",
        provisioning_uri=provisioning_uri
    )


@router.post("/2fa/verify", response_model=TwoFAStatusResponse)
def verify_and_enable_2fa(
    data: Verify2FARequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Vérifie le code TOTP et active le 2FA si correct.
    """
    if not current_user.totp_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Configurez d'abord le 2FA avec /2fa/setup"
        )

    if current_user.totp_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA déjà activé"
        )

    # Vérifier le code
    totp = pyotp.TOTP(current_user.totp_secret)
    if not totp.verify(data.totp_code, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Code invalide. Réessayez."
        )

    # Activer le 2FA
    current_user.totp_enabled = True
    db.commit()

    return TwoFAStatusResponse(
        enabled=True,
        message="2FA activé avec succès"
    )


@router.post("/2fa/disable", response_model=TwoFAStatusResponse)
def disable_2fa(
    data: Verify2FARequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Désactive le 2FA après vérification du code actuel.
    """
    if not current_user.totp_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA n'est pas activé"
        )

    # Vérifier le code actuel
    totp = pyotp.TOTP(current_user.totp_secret)
    if not totp.verify(data.totp_code, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Code invalide"
        )

    # Désactiver le 2FA
    current_user.totp_enabled = False
    current_user.totp_secret = None
    db.commit()

    return TwoFAStatusResponse(
        enabled=False,
        message="2FA désactivé"
    )


@router.get("/2fa/status", response_model=TwoFAStatusResponse)
def get_2fa_status(current_user: User = Depends(get_current_user)):
    """
    Retourne le statut du 2FA pour l'utilisateur connecté.
    """
    return TwoFAStatusResponse(
        enabled=current_user.totp_enabled,
        message="2FA activé" if current_user.totp_enabled else "2FA non activé"
    )


# ============================================================================
# GESTION DU COMPTE
# ============================================================================

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ChangePasswordResponse(BaseModel):
    success: bool
    message: str


@router.post("/change-password", response_model=ChangePasswordResponse)
def change_password(
    request: Request,
    data: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change le mot de passe de l'utilisateur connecté.
    Requiert le mot de passe actuel pour confirmation.
    """
    # Vérifier le mot de passe actuel
    if not verify_password(data.current_password, current_user.password_hash):
        audit_log.log_suspicious_activity(
            user_id=current_user.id,
            activity_type="failed_password_change",
            ip_address=request.client.host if hasattr(request, 'client') else None,
            details="Invalid current password provided"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mot de passe actuel incorrect"
        )

    # Vérifier que le nouveau mot de passe est différent
    if data.current_password == data.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le nouveau mot de passe doit être différent"
        )

    # Vérifier la longueur minimale
    if len(data.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le mot de passe doit contenir au moins 8 caractères"
        )

    # Mettre à jour le mot de passe
    current_user.password_hash = hash_password(data.new_password)
    db.commit()

    # Audit log
    audit_log.log_suspicious_activity(
        user_id=current_user.id,
        activity_type="password_changed",
        ip_address=request.client.host if hasattr(request, 'client') else None,
        details="Password changed successfully"
    )

    return ChangePasswordResponse(
        success=True,
        message="Mot de passe modifié avec succès"
    )


class DeleteAccountRequest(BaseModel):
    password: str
    confirmation: str  # Doit être "SUPPRIMER"


class DeleteAccountResponse(BaseModel):
    success: bool
    message: str


@router.post("/delete-account", response_model=DeleteAccountResponse)
def delete_account(
    request: Request,
    data: DeleteAccountRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Supprime définitivement le compte utilisateur.
    Requiert le mot de passe et la confirmation "SUPPRIMER".
    """
    # Vérifier la confirmation
    if data.confirmation != "SUPPRIMER":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tapez 'SUPPRIMER' pour confirmer"
        )

    # Vérifier le mot de passe
    if not verify_password(data.password, current_user.password_hash):
        audit_log.log_suspicious_activity(
            user_id=current_user.id,
            activity_type="failed_account_deletion",
            ip_address=request.client.host if hasattr(request, 'client') else None,
            details="Invalid password for account deletion"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mot de passe incorrect"
        )

    user_email = current_user.email
    user_id = current_user.id

    # Supprimer la licence associée
    db.query(License).filter(License.user_id == current_user.id).delete()

    # Supprimer les tokens de reset password
    db.query(PasswordResetToken).filter(PasswordResetToken.email == current_user.email).delete()

    # Supprimer l'utilisateur
    db.delete(current_user)
    db.commit()

    # Audit log
    audit_log.log_suspicious_activity(
        user_id=user_id,
        activity_type="account_deleted",
        ip_address=request.client.host if hasattr(request, 'client') else None,
        details=f"Account {user_email} deleted permanently"
    )

    return DeleteAccountResponse(
        success=True,
        message="Compte supprimé définitivement"
    )
