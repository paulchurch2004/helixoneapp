"""
Routes de gestion des licences
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import Optional

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models import User, License
from app.schemas.license import LicenseResponse

router = APIRouter()


def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> User:
    """Dépendance pour obtenir l'utilisateur courant"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token manquant"
        )
    
    token = authorization.replace("Bearer ", "")
    payload = decode_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré"
        )
    
    user_id = payload.get("user_id")
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur introuvable"
        )
    
    return user


@router.get("/status", response_model=LicenseResponse)
def get_license_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtenir le statut de la licence"""
    
    license = db.query(License).filter(
        License.user_id == current_user.id,
        License.status == "active"
    ).first()
    
    if not license:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucune licence active trouvée"
        )
    
    # Convertir le modèle en dict avec days_remaining calculé
    license_dict = {
        "id": license.id,
        "license_key": license.license_key,
        "license_type": license.license_type,
        "status": license.status,
        "features": license.features,
        "quota_daily_analyses": license.quota_daily_analyses,
        "quota_daily_api_calls": license.quota_daily_api_calls,
        "activated_at": license.activated_at,
        "expires_at": license.expires_at,
        "days_remaining": license.days_remaining()  # Appeler la méthode
    }
    
    return LicenseResponse(**license_dict)
