"""
Gestionnaire de secrets sécurisé pour HelixOne

Ce module fournit une gestion centralisée et sécurisée des secrets:
- Chargement depuis variables d'environnement ou Vault
- Validation stricte au démarrage
- Rotation des secrets
- Audit logging
- Pas de secrets dans les logs

Usage:
    from app.core.secrets_manager import secrets_manager

    # Obtenir un secret
    api_key = secrets_manager.get_secret("ALPHA_VANTAGE_API_KEY")

    # Vérifier si tous les secrets sont configurés
    if not secrets_manager.validate_all():
        raise Exception("Secrets manquants!")
"""

import os
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.core.time_utils import utc_now_naive

# Configuration du logger (sans afficher les secrets)
logger = logging.getLogger(__name__)


@dataclass
class SecretMetadata:
    """Métadonnées d'un secret"""
    name: str
    required: bool
    description: str
    last_accessed: Optional[datetime] = None
    rotation_days: Optional[int] = None  # Jours avant rotation recommandée


class SecretsManager:
    """
    Gestionnaire centralisé des secrets

    Features:
    - Chargement depuis .env ou variables d'environnement
    - Validation au démarrage
    - Audit logging (sans exposer les secrets)
    - Support pour Vault (optionnel)
    """

    def __init__(self):
        self._secrets_cache: Dict[str, str] = {}
        self._metadata: Dict[str, SecretMetadata] = {}
        self._audit_log: List[Dict] = []

        # Définir les secrets requis
        self._define_secrets()

    def _define_secrets(self):
        """Définir tous les secrets nécessaires avec leurs métadonnées"""

        # Secrets CRITIQUES (requis)
        self._register_secret(
            "SECRET_KEY",
            required=True,
            description="Clé secrète pour JWT (min 32 caractères)",
            rotation_days=90
        )

        self._register_secret(
            "DATABASE_URL",
            required=True,
            description="URL de connexion PostgreSQL",
            rotation_days=None  # Password DB change manuellement
        )

        # Secrets API (optionnels mais recommandés)
        api_keys = {
            "ALPHA_VANTAGE_API_KEY": "Clé API Alpha Vantage pour données de marché",
            "FRED_API_KEY": "Clé API FRED pour données macroéconomiques",
            "NEWS_API_KEY": "Clé API NewsAPI pour sentiment",
            "TWITTER_API_KEY": "Clé API Twitter pour sentiment social",
            "FINNHUB_API_KEY": "Clé API Finnhub pour données financières",
        }

        for key, description in api_keys.items():
            self._register_secret(
                key,
                required=False,
                description=description,
                rotation_days=180
            )

        # Secrets IBKR (optionnels)
        self._register_secret(
            "IBKR_HOST",
            required=False,
            description="Host IBKR Gateway/TWS",
            rotation_days=None
        )

        self._register_secret(
            "IBKR_PORT",
            required=False,
            description="Port IBKR Gateway/TWS",
            rotation_days=None
        )

        # Redis (optionnel)
        self._register_secret(
            "REDIS_URL",
            required=False,
            description="URL Redis avec authentification",
            rotation_days=90
        )

        # Sentry (optionnel)
        self._register_secret(
            "SENTRY_DSN",
            required=False,
            description="DSN Sentry pour error tracking",
            rotation_days=None
        )

    def _register_secret(
        self,
        name: str,
        required: bool,
        description: str,
        rotation_days: Optional[int] = None
    ):
        """Enregistrer un secret avec ses métadonnées"""
        self._metadata[name] = SecretMetadata(
            name=name,
            required=required,
            description=description,
            rotation_days=rotation_days
        )

    def load_secrets(self):
        """
        Charger tous les secrets depuis l'environnement

        Returns:
            bool: True si tous les secrets requis sont chargés
        """
        logger.info("🔐 Chargement des secrets...")

        missing_required = []

        for name, metadata in self._metadata.items():
            value = os.getenv(name)

            if value:
                self._secrets_cache[name] = value
                logger.info(f"✅ Secret '{name}' chargé")
            else:
                if metadata.required:
                    missing_required.append(name)
                    logger.error(f"❌ Secret REQUIS manquant: {name}")
                else:
                    logger.warning(f"⚠️  Secret optionnel manquant: {name}")

        if missing_required:
            logger.error(f"🚨 {len(missing_required)} secrets requis manquants!")
            for name in missing_required:
                meta = self._metadata[name]
                logger.error(f"   - {name}: {meta.description}")
            return False

        logger.info(f"✅ {len(self._secrets_cache)} secrets chargés avec succès")
        return True

    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Récupérer un secret

        Args:
            name: Nom du secret
            default: Valeur par défaut si non trouvé

        Returns:
            Valeur du secret ou default

        Raises:
            ValueError: Si secret requis manquant
        """
        # Audit logging
        self._audit_log.append({
            "action": "get_secret",
            "secret_name": name,
            "timestamp": utc_now_naive(),
            "success": name in self._secrets_cache
        })

        # Mettre à jour last_accessed
        if name in self._metadata:
            self._metadata[name].last_accessed = utc_now_naive()

        # Vérifier si le secret existe
        if name in self._secrets_cache:
            return self._secrets_cache[name]

        # Si secret requis mais manquant
        if name in self._metadata and self._metadata[name].required:
            raise ValueError(
                f"Secret requis '{name}' non configuré. "
                f"Description: {self._metadata[name].description}"
            )

        # Sinon retourner default
        return default

    def validate_all(self) -> bool:
        """
        Valider que tous les secrets requis sont présents

        Returns:
            bool: True si tous les secrets requis sont OK
        """
        missing = []

        for name, metadata in self._metadata.items():
            if metadata.required and name not in self._secrets_cache:
                missing.append(name)

        if missing:
            logger.error(f"❌ Validation échouée: {len(missing)} secrets manquants")
            return False

        logger.info("✅ Validation réussie: tous les secrets requis sont présents")
        return True

    def validate_secret_strength(self, name: str) -> bool:
        """
        Valider la force d'un secret (longueur, complexité)

        Args:
            name: Nom du secret à valider

        Returns:
            bool: True si le secret est suffisamment fort
        """
        value = self.get_secret(name)
        if not value:
            return False

        # Règles de validation
        if name == "SECRET_KEY":
            # Minimum 32 caractères pour SECRET_KEY
            if len(value) < 32:
                logger.warning(f"⚠️  {name} trop court (min 32 caractères)")
                return False

        # Vérifier qu'il ne s'agit pas d'une valeur de test évidente
        test_patterns = ["test", "demo", "example", "changeme", "password"]
        if any(pattern in value.lower() for pattern in test_patterns):
            logger.warning(f"⚠️  {name} semble être une valeur de test")
            return False

        return True

    def check_rotation_needed(self) -> List[str]:
        """
        Vérifier quels secrets nécessitent une rotation

        Returns:
            Liste des noms de secrets à faire tourner
        """
        needs_rotation = []

        for name, metadata in self._metadata.items():
            if metadata.rotation_days and metadata.last_accessed:
                days_since_access = (utc_now_naive() - metadata.last_accessed).days

                if days_since_access > metadata.rotation_days:
                    needs_rotation.append(name)
                    logger.warning(
                        f"⚠️  Secret '{name}' n'a pas été roté depuis "
                        f"{days_since_access} jours (recommandé: {metadata.rotation_days})"
                    )

        return needs_rotation

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """
        Récupérer les logs d'audit d'accès aux secrets

        Args:
            limit: Nombre maximum de logs à retourner

        Returns:
            Liste des derniers accès aux secrets
        """
        return self._audit_log[-limit:]

    def get_secrets_status(self) -> Dict:
        """
        Obtenir le statut de tous les secrets

        Returns:
            Dict avec le statut de chaque secret (sans les valeurs!)
        """
        status = {
            "loaded": len(self._secrets_cache),
            "required": sum(1 for m in self._metadata.values() if m.required),
            "optional": sum(1 for m in self._metadata.values() if not m.required),
            "needs_rotation": len(self.check_rotation_needed()),
            "secrets": {}
        }

        for name, metadata in self._metadata.items():
            status["secrets"][name] = {
                "configured": name in self._secrets_cache,
                "required": metadata.required,
                "description": metadata.description,
                "last_accessed": (
                    metadata.last_accessed.isoformat()
                    if metadata.last_accessed else None
                ),
                "rotation_days": metadata.rotation_days
            }

        return status

    def __repr__(self):
        """Représentation (sans exposer les secrets!)"""
        return (
            f"<SecretsManager("
            f"loaded={len(self._secrets_cache)}, "
            f"required={sum(1 for m in self._metadata.values() if m.required)})"
            f">"
        )


# Instance globale du gestionnaire de secrets
secrets_manager = SecretsManager()


def init_secrets() -> bool:
    """
    Initialiser et valider les secrets au démarrage de l'application

    Returns:
        bool: True si succès

    Raises:
        ValueError: Si secrets requis manquants
    """
    logger.info("🔐 Initialisation du gestionnaire de secrets...")

    # Charger les secrets
    if not secrets_manager.load_secrets():
        raise ValueError("Échec du chargement des secrets requis")

    # Valider tous les secrets
    if not secrets_manager.validate_all():
        raise ValueError("Validation des secrets échouée")

    # Valider la force de SECRET_KEY
    if not secrets_manager.validate_secret_strength("SECRET_KEY"):
        logger.warning("⚠️  SECRET_KEY n'est pas assez fort!")

    # Vérifier les rotations
    needs_rotation = secrets_manager.check_rotation_needed()
    if needs_rotation:
        logger.warning(
            f"⚠️  {len(needs_rotation)} secrets nécessitent une rotation: "
            f"{', '.join(needs_rotation)}"
        )

    logger.info("✅ Gestionnaire de secrets initialisé avec succès")
    return True


# Fonction helper pour utilisation simple
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Helper function pour obtenir un secret facilement

    Args:
        name: Nom du secret
        default: Valeur par défaut

    Returns:
        Valeur du secret
    """
    return secrets_manager.get_secret(name, default)
