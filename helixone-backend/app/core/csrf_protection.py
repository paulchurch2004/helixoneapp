"""
Protection CSRF (Cross-Site Request Forgery) pour HelixOne

Ce module fournit une protection contre les attaques CSRF via:
- Tokens CSRF sécurisés (cryptographiquement forts)
- Validation automatique pour POST/PUT/DELETE/PATCH
- Exemptions pour endpoints publics
- Intégration avec FastAPI

Usage:
    from app.core.csrf_protection import csrf_protect, get_csrf_token

    # Dans un endpoint protégé
    @router.post("/api/trade")
    async def create_trade(
        request: Request,
        csrf_token: str = Depends(csrf_protect)
    ):
        # Le token CSRF est validé automatiquement
        ...

    # Générer un token pour le frontend
    @router.get("/csrf-token")
    async def get_token(request: Request):
        token = get_csrf_token(request)
        return {"csrf_token": token}
"""

import secrets
import hmac
import hashlib
from typing import Optional
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer
from datetime import timedelta

from app.core.time_utils import utc_now_naive


class CSRFProtection:
    """
    Gestionnaire de protection CSRF

    Features:
    - Tokens sécurisés (32 bytes random)
    - Signature HMAC pour validation
    - Expiration des tokens (2 heures par défaut)
    - Stockage en session (cookie httponly)
    """

    def __init__(self, secret_key: str, token_expiry_hours: int = 2):
        """
        Initialiser la protection CSRF

        Args:
            secret_key: Clé secrète pour signer les tokens
            token_expiry_hours: Durée de validité des tokens en heures
        """
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        self.token_expiry = timedelta(hours=token_expiry_hours)

    def generate_token(self, session_id: Optional[str] = None) -> str:
        """
        Générer un nouveau token CSRF

        Args:
            session_id: ID de session (optionnel, utilisé pour lier le token)

        Returns:
            Token CSRF sécurisé
        """
        # Générer un token aléatoire
        random_token = secrets.token_urlsafe(32)

        # Timestamp pour expiration
        timestamp = int(utc_now_naive().timestamp())

        # Combiner token + timestamp + session_id
        data = f"{random_token}:{timestamp}"
        if session_id:
            data += f":{session_id}"

        # Signer avec HMAC
        signature = hmac.new(
            self.secret_key,
            data.encode(),
            hashlib.sha256
        ).hexdigest()

        # Format: data.signature
        return f"{data}.{signature}"

    def validate_token(
        self,
        token: str,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Valider un token CSRF

        Args:
            token: Token à valider
            session_id: ID de session attendu

        Returns:
            True si le token est valide

        Raises:
            ValueError: Si le token est malformé
        """
        try:
            # Séparer data et signature
            parts = token.split(".")
            if len(parts) != 2:
                return False

            data, signature = parts

            # Vérifier la signature
            expected_signature = hmac.new(
                self.secret_key,
                data.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return False

            # Parser les données
            data_parts = data.split(":")
            if len(data_parts) < 2:
                return False

            random_token = data_parts[0]
            timestamp = int(data_parts[1])
            token_session_id = data_parts[2] if len(data_parts) > 2 else None

            # Vérifier l'expiration
            token_time = datetime.fromtimestamp(timestamp)
            if utc_now_naive() - token_time > self.token_expiry:
                return False

            # Vérifier le session_id si fourni
            if session_id and token_session_id != session_id:
                return False

            return True

        except (ValueError, IndexError, TypeError):
            return False


# Instance globale (sera initialisée avec la vraie SECRET_KEY au démarrage)
_csrf_protection: Optional[CSRFProtection] = None


def init_csrf_protection(secret_key: str):
    """
    Initialiser la protection CSRF avec la SECRET_KEY de l'app

    Args:
        secret_key: SECRET_KEY de l'application
    """
    global _csrf_protection
    _csrf_protection = CSRFProtection(secret_key)


def get_csrf_protection() -> CSRFProtection:
    """
    Obtenir l'instance de protection CSRF

    Returns:
        Instance CSRFProtection

    Raises:
        RuntimeError: Si pas initialisé
    """
    if _csrf_protection is None:
        raise RuntimeError(
            "CSRF protection not initialized. "
            "Call init_csrf_protection() first."
        )
    return _csrf_protection


# ============================================================================
# FONCTIONS HELPER POUR FASTAPI
# ============================================================================

def get_csrf_token(request: Request) -> str:
    """
    Générer ou récupérer un token CSRF pour la session

    Args:
        request: Request FastAPI

    Returns:
        Token CSRF
    """
    csrf = get_csrf_protection()

    # Essayer de récupérer un token existant depuis le cookie
    existing_token = request.cookies.get("csrf_token")

    # Récupérer session_id (si disponible)
    session_id = request.cookies.get("session_id")

    # Valider le token existant
    if existing_token and csrf.validate_token(existing_token, session_id):
        return existing_token

    # Sinon générer un nouveau token
    new_token = csrf.generate_token(session_id)
    return new_token


async def csrf_protect(request: Request) -> str:
    """
    Dependency FastAPI pour protéger les endpoints contre CSRF

    Usage:
        @router.post("/api/trade")
        async def create_trade(csrf_token: str = Depends(csrf_protect)):
            ...

    Args:
        request: Request FastAPI

    Returns:
        Token CSRF validé

    Raises:
        HTTPException: Si le token est manquant ou invalide
    """
    # Méthodes sûres (GET, HEAD, OPTIONS) ne nécessitent pas de protection
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return ""

    csrf = get_csrf_protection()

    # Récupérer le token depuis le header X-CSRF-Token
    token = request.headers.get("X-CSRF-Token")

    # Fallback: essayer dans le body (si form-data)
    if not token and request.method == "POST":
        try:
            form = await request.form()
            token = form.get("csrf_token")
        except Exception:
            pass

    # Token manquant
    if not token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing. Include X-CSRF-Token header."
        )

    # Récupérer session_id
    session_id = request.cookies.get("session_id")

    # Valider le token
    if not csrf.validate_token(token, session_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired CSRF token"
        )

    return token


# ============================================================================
# MIDDLEWARE CSRF (OPTIONNEL)
# ============================================================================

class CSRFMiddleware:
    """
    Middleware pour ajouter automatiquement le token CSRF aux réponses

    Usage:
        app.add_middleware(CSRFMiddleware)
    """

    def __init__(self, app, exempt_paths: Optional[list] = None):
        """
        Args:
            app: Application FastAPI
            exempt_paths: Liste de chemins exemptés (ex: ["/health", "/docs"])
        """
        self.app = app
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/openapi.json"]

    async def __call__(self, request: Request, call_next):
        """Traiter la requête"""
        # Vérifier si le chemin est exempté
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Générer/récupérer token CSRF pour cette session
        try:
            csrf_token = get_csrf_token(request)

            # Continuer la requête
            response = await call_next(request)

            # Ajouter le token dans un cookie httponly
            response.set_cookie(
                key="csrf_token",
                value=csrf_token,
                httponly=True,  # Pas accessible via JavaScript
                secure=True,    # HTTPS uniquement (désactiver en dev si HTTP)
                samesite="strict",  # Protection supplémentaire
                max_age=7200  # 2 heures
            )

            # Aussi ajouter dans un header pour faciliter l'usage côté client
            response.headers["X-CSRF-Token"] = csrf_token

            return response

        except Exception as e:
            # En cas d'erreur, continuer sans CSRF (logged)
            print(f"CSRF middleware error: {e}")
            return await call_next(request)


# ============================================================================
# EXEMPTIONS
# ============================================================================

# Liste des endpoints qui ne nécessitent pas de protection CSRF
CSRF_EXEMPT_ENDPOINTS = [
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/auth/login",  # Login utilise déjà username/password
    "/auth/register",  # Register aussi
]


def is_csrf_exempt(path: str) -> bool:
    """
    Vérifier si un chemin est exempté de CSRF

    Args:
        path: Chemin de l'endpoint

    Returns:
        True si exempté
    """
    return any(path.startswith(exempt) for exempt in CSRF_EXEMPT_ENDPOINTS)
