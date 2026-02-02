"""
Système d'audit logging pour HelixOne

Ce module fournit un système de logging d'audit sécurisé pour tracer:
- Authentifications (login/logout/échecs)
- Accès aux données sensibles (portfolio, trades)
- Modifications de configuration
- Actions critiques
- Tentatives d'accès non autorisés

IMPORTANT: Ne JAMAIS logger:
- Mots de passe (même hashés)
- Tokens JWT complets
- API Keys
- Secrets

Usage:
    from app.core.audit_logger import audit_log, AuditEvent

    # Logger un événement
    audit_log.log_auth_success(user_id="123", ip="1.2.3.4")
    audit_log.log_trade_executed(user_id="123", symbol="AAPL", quantity=10)
"""

import logging
import json
from typing import Optional, Dict, Any

from app.core.time_utils import utc_now_naive
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path


class AuditEventType(Enum):
    """Types d'événements d'audit"""

    # Authentification
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILED = "auth.login.failed"
    AUTH_LOGOUT = "auth.logout"
    AUTH_REGISTER = "auth.register"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTH_PASSWORD_RESET = "auth.password.reset"

    # Accès aux données
    DATA_PORTFOLIO_VIEW = "data.portfolio.view"
    DATA_PORTFOLIO_MODIFY = "data.portfolio.modify"
    DATA_ANALYSIS_VIEW = "data.analysis.view"
    DATA_ANALYSIS_RUN = "data.analysis.run"

    # Trading
    TRADE_EXECUTED = "trade.executed"
    TRADE_CANCELLED = "trade.cancelled"
    TRADE_MODIFIED = "trade.modified"

    # Configuration
    CONFIG_API_KEY_ADDED = "config.api_key.added"
    CONFIG_API_KEY_REMOVED = "config.api_key.removed"
    CONFIG_SETTINGS_CHANGED = "config.settings.changed"

    # Sécurité
    SECURITY_ACCESS_DENIED = "security.access_denied"
    SECURITY_RATE_LIMIT_EXCEEDED = "security.rate_limit_exceeded"
    SECURITY_INVALID_TOKEN = "security.invalid_token"
    SECURITY_SUSPICIOUS_ACTIVITY = "security.suspicious_activity"

    # Système
    SYSTEM_SECRET_ROTATED = "system.secret.rotated"
    SYSTEM_BACKUP_CREATED = "system.backup.created"
    SYSTEM_ERROR = "system.error"


class AuditSeverity(Enum):
    """Niveaux de sévérité"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Représentation d'un événement d'audit"""
    timestamp: str
    event_type: str
    severity: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour JSON"""
        return asdict(self)

    def to_json(self) -> str:
        """Convertir en JSON"""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Logger d'audit sécurisé

    Features:
    - Logs structurés en JSON
    - Rotation automatique des logs
    - Pas de données sensibles
    - Horodatage UTC
    - Contexte complet
    """

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialiser le logger d'audit

        Args:
            log_file: Chemin vers le fichier de logs (optionnel)
        """
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)

        # Nettoyer les handlers existants pour éviter les doublons et
        # permettre à chaque instance d'avoir ses propres handlers
        self.logger.handlers.clear()

        # Handler pour fichier
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file)
        else:
            # Par défaut, logs/audit.log
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            handler = logging.FileHandler(log_dir / "audit.log")

        # Format JSON pour parsing facile
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Aussi logger sur console si pas de log_file spécifique (= instance globale)
        if log_file is None or str(log_file) == "logs/audit.log":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.INFO,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        message: Optional[str] = None,
        **metadata
    ):
        """
        Logger un événement d'audit

        Args:
            event_type: Type d'événement
            severity: Niveau de sévérité
            user_id: ID de l'utilisateur (anonymisé en prod)
            ip_address: Adresse IP (anonymisée en prod)
            user_agent: User agent
            success: Si l'action a réussi
            message: Message descriptif
            **metadata: Données additionnelles (pas de secrets!)
        """
        event = AuditEvent(
            timestamp=utc_now_naive().isoformat() + "Z",
            event_type=event_type.value,
            severity=severity.value,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            message=message,
            metadata=metadata if metadata else None
        )

        # Logger en JSON
        log_line = event.to_json()

        # Choisir le niveau de log selon severity
        if severity == AuditSeverity.CRITICAL:
            self.logger.critical(log_line)
        elif severity == AuditSeverity.ERROR:
            self.logger.error(log_line)
        elif severity == AuditSeverity.WARNING:
            self.logger.warning(log_line)
        else:
            self.logger.info(log_line)

        # IMPORTANT: Flush immédiatement pour garantir que les logs d'audit
        # sont écrits sur disque (critique pour sécurité et compliance)
        for handler in self.logger.handlers:
            handler.flush()

    # ========================================================================
    # AUTHENTIFICATION
    # ========================================================================

    def log_auth_success(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Logger une connexion réussie"""
        self.log_event(
            AuditEventType.AUTH_LOGIN_SUCCESS,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            message=f"User {user_id} logged in successfully"
        )

    def log_auth_failed(
        self,
        email: str,
        ip_address: Optional[str] = None,
        reason: str = "invalid_credentials"
    ):
        """Logger une tentative de connexion échouée"""
        self.log_event(
            AuditEventType.AUTH_LOGIN_FAILED,
            severity=AuditSeverity.WARNING,
            ip_address=ip_address,
            success=False,
            message=f"Failed login attempt for {email}",
            email=email,
            reason=reason
        )

    def log_auth_logout(
        self,
        user_id: str,
        ip_address: Optional[str] = None
    ):
        """Logger une déconnexion"""
        self.log_event(
            AuditEventType.AUTH_LOGOUT,
            user_id=user_id,
            ip_address=ip_address,
            message=f"User {user_id} logged out"
        )

    def log_auth_register(
        self,
        user_id: str,
        email: str,
        ip_address: Optional[str] = None
    ):
        """Logger une inscription"""
        self.log_event(
            AuditEventType.AUTH_REGISTER,
            user_id=user_id,
            ip_address=ip_address,
            message=f"New user registered: {email}",
            email=email
        )

    # ========================================================================
    # PORTFOLIO & TRADING
    # ========================================================================

    def log_portfolio_view(
        self,
        user_id: str,
        ip_address: Optional[str] = None
    ):
        """Logger un accès au portfolio"""
        self.log_event(
            AuditEventType.DATA_PORTFOLIO_VIEW,
            user_id=user_id,
            ip_address=ip_address,
            message=f"User {user_id} viewed portfolio"
        )

    def log_portfolio_analysis_run(
        self,
        user_id: str,
        analysis_type: str,
        ip_address: Optional[str] = None
    ):
        """Logger l'exécution d'une analyse"""
        self.log_event(
            AuditEventType.DATA_ANALYSIS_RUN,
            user_id=user_id,
            ip_address=ip_address,
            message=f"User {user_id} ran {analysis_type} analysis",
            analysis_type=analysis_type
        )

    def log_trade_executed(
        self,
        user_id: str,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        ip_address: Optional[str] = None
    ):
        """Logger l'exécution d'un trade"""
        self.log_event(
            AuditEventType.TRADE_EXECUTED,
            severity=AuditSeverity.WARNING,  # Important!
            user_id=user_id,
            ip_address=ip_address,
            message=f"Trade executed: {side} {quantity} {symbol} @ {price}",
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side,
            total_value=quantity * price
        )

    # ========================================================================
    # SÉCURITÉ
    # ========================================================================

    def log_access_denied(
        self,
        user_id: Optional[str],
        resource: str,
        ip_address: Optional[str] = None,
        reason: str = "unauthorized"
    ):
        """Logger un accès refusé"""
        self.log_event(
            AuditEventType.SECURITY_ACCESS_DENIED,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            success=False,
            message=f"Access denied to {resource}",
            resource=resource,
            reason=reason
        )

    def log_rate_limit_exceeded(
        self,
        user_id: Optional[str],
        endpoint: str,
        ip_address: Optional[str] = None
    ):
        """Logger un dépassement de rate limit"""
        self.log_event(
            AuditEventType.SECURITY_RATE_LIMIT_EXCEEDED,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            ip_address=ip_address,
            success=False,
            message=f"Rate limit exceeded for {endpoint}",
            endpoint=endpoint
        )

    def log_invalid_token(
        self,
        ip_address: Optional[str] = None,
        reason: str = "invalid_signature"
    ):
        """Logger un token invalide"""
        self.log_event(
            AuditEventType.SECURITY_INVALID_TOKEN,
            severity=AuditSeverity.WARNING,
            ip_address=ip_address,
            success=False,
            message="Invalid token attempted",
            reason=reason
        )

    def log_suspicious_activity(
        self,
        user_id: Optional[str],
        activity_type: str,
        ip_address: Optional[str] = None,
        details: Optional[str] = None
    ):
        """Logger une activité suspecte"""
        self.log_event(
            AuditEventType.SECURITY_SUSPICIOUS_ACTIVITY,
            severity=AuditSeverity.ERROR,
            user_id=user_id,
            ip_address=ip_address,
            success=False,
            message=f"Suspicious activity detected: {activity_type}",
            activity_type=activity_type,
            details=details
        )

    # ========================================================================
    # CONFIGURATION & SYSTÈME
    # ========================================================================

    def log_config_change(
        self,
        user_id: str,
        setting_name: str,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None
    ):
        """Logger un changement de configuration"""
        self.log_event(
            AuditEventType.CONFIG_SETTINGS_CHANGED,
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            message=f"Configuration changed: {setting_name}",
            setting_name=setting_name,
            old_value=str(old_value) if old_value else None,
            new_value=str(new_value) if new_value else None
        )

    def log_secret_rotated(
        self,
        secret_name: str,
        rotated_by: Optional[str] = None
    ):
        """Logger une rotation de secret"""
        self.log_event(
            AuditEventType.SYSTEM_SECRET_ROTATED,
            severity=AuditSeverity.WARNING,
            user_id=rotated_by,
            message=f"Secret rotated: {secret_name}",
            secret_name=secret_name
        )

    def log_system_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None
    ):
        """Logger une erreur système"""
        self.log_event(
            AuditEventType.SYSTEM_ERROR,
            severity=AuditSeverity.ERROR,
            success=False,
            message=f"System error: {error_type}",
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace[:500] if stack_trace else None  # Limiter taille
        )


# Instance globale du logger d'audit
audit_log = AuditLogger()


# ============================================================================
# DECORATEURS POUR AUTOMATISER L'AUDIT LOGGING
# ============================================================================

def audit_endpoint(event_type: AuditEventType, severity: AuditSeverity = AuditSeverity.INFO):
    """
    Décorateur pour logger automatiquement les appels d'endpoints

    Usage:
        @audit_endpoint(AuditEventType.DATA_PORTFOLIO_VIEW)
        async def get_portfolio(user_id: str):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extraire user_id et request si disponibles
            user_id = kwargs.get("user_id") or kwargs.get("current_user", {}).get("user_id")
            request = kwargs.get("request")

            ip_address = None
            user_agent = None

            if request:
                ip_address = request.client.host if hasattr(request, 'client') else None
                user_agent = request.headers.get("user-agent") if hasattr(request, 'headers') else None

            try:
                # Exécuter la fonction
                result = await func(*args, **kwargs)

                # Logger succès
                audit_log.log_event(
                    event_type,
                    severity,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=True,
                    message=f"Endpoint {func.__name__} called successfully"
                )

                return result

            except Exception as e:
                # Logger échec
                audit_log.log_event(
                    event_type,
                    AuditSeverity.ERROR,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    message=f"Endpoint {func.__name__} failed: {str(e)}",
                    error=str(e)
                )
                raise

        return wrapper
    return decorator
