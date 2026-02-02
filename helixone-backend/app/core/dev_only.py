"""
Utilitaire pour restreindre les endpoints de développement
"""
from fastapi import HTTPException
from app.core.config import get_settings

settings = get_settings()


def check_dev_only():
    """
    Vérifie si l'endpoint dev peut être utilisé
    Bloque en production pour raisons de sécurité

    Raises:
        HTTPException: 403 si appelé en production
    """
    if settings.ENVIRONMENT == "production":
        raise HTTPException(
            status_code=403,
            detail="Cet endpoint de développement n'est pas disponible en production"
        )


def dev_only():
    """
    Dependency function pour FastAPI Depends()
    Utilisé pour restreindre les endpoints de développement

    Usage:
        @router.get("/dev/endpoint", dependencies=[Depends(dev_only)])
        async def dev_endpoint():
            ...

    Raises:
        HTTPException: 403 si appelé en production
    """
    check_dev_only()
