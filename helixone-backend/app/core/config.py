"""
Configuration centralisée de l'application
Charge toutes les variables depuis .env
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List
from enum import Enum
import json


class Environment(str, Enum):
    """Environnements supportés"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Configuration de l'application"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Ignorer les variables inconnues pour flexibilité
    )

    # Application
    APP_NAME: str = "HelixOne API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    BACKEND_URL: str = "http://localhost:8000"

    # Base de données
    DATABASE_URL: str = "sqlite:///./helixone.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_QUOTES: int = 60  # Cache quotes: 1 minute
    CACHE_TTL_ANALYSIS: int = 900  # Cache analyses: 15 minutes

    # Sécurité JWT - TOUJOURS utiliser une clé depuis .env, jamais hardcodée
    SECRET_KEY: str = ""  # Défini dans .env
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost", "helixone://"]

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS depuis JSON string si nécessaire"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Fallback: split par virgule
                return [x.strip() for x in v.split(',') if x.strip()]
        return v

    @field_validator('ENVIRONMENT', mode='before')
    @classmethod
    def validate_environment(cls, v):
        """Valide que l'environnement est supporté"""
        valid_envs = ['development', 'staging', 'production']
        if v and v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT doit être: {', '.join(valid_envs)}")
        return v.lower() if v else 'development'

    def is_production(self) -> bool:
        """Retourne True si on est en production"""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """Retourne True si on est en développement"""
        return self.ENVIRONMENT == "development"
    
    # API Keys - Data Sources
    YAHOO_API_KEY: str = ""
    FINNHUB_API_KEY: str = ""
    FMP_API_KEY: str = ""  # Financial Modeling Prep
    MARKETSTACK_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    TWELVEDATA_API_KEY: str = ""
    ALPHA_VANTAGE_API_KEY: str = ""
    FRED_API_KEY: str = ""  # Federal Reserve Economic Data
    IEX_CLOUD_API_KEY: str = ""
    
    # Stripe
    STRIPE_SECRET_KEY: str = ""
    STRIPE_PUBLISHABLE_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    
    # Email - SMTP
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    FROM_EMAIL: str = "noreply@helixone.com"
    
    # Sentry
    SENTRY_DSN: str = ""

    # Réentraînement Automatique ML
    AUTO_RETRAIN_ENABLED: bool = False
    AUTO_RETRAIN_INTERVAL_DAYS: int = 7
    AUTO_RETRAIN_HOUR: int = 2  # 2h du matin
    AUTO_RETRAIN_TICKERS: str = ""  # Format: "AAPL,MSFT,GOOGL"

    def get_retrain_tickers(self) -> List[str]:
        """Retourne la liste des tickers à réentraîner"""
        if not self.AUTO_RETRAIN_TICKERS:
            return []
        return [t.strip().upper() for t in self.AUTO_RETRAIN_TICKERS.split(',') if t.strip()]


# Instance globale des settings
settings = Settings()


def get_settings() -> Settings:
    """
    Retourne l'instance des settings (pour dependency injection FastAPI)

    Returns:
        Settings: Configuration de l'application
    """
    return settings


def validate_settings(raise_on_error: bool = True) -> bool:
    """
    Vérifie que les variables critiques sont définies.

    Args:
        raise_on_error: Si True, lève une exception en cas d'erreur. Sinon retourne False.

    Returns:
        True si la configuration est valide
    """
    errors = []

    # Liste des clés par défaut insécurisées à détecter
    insecure_keys = [
        "xK9mP2nQ5vR8wY3zA6cF1hJ4kL7oT0uBv2X8pM4qN7rW",
        "CHANGEZ_MOI_AVEC_UNE_CLE_SECURISEE",
        "your_secret_key_here",
        "changeme",
        "secret",
    ]

    # Vérifier que SECRET_KEY existe et est sécurisée
    if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
        errors.append(
            "SECRET_KEY non définie ou trop courte dans .env! "
            "Générez une clé avec: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    elif settings.SECRET_KEY in insecure_keys:
        errors.append("SECRET_KEY par défaut/insécurisée détectée! Changez-la dans .env")

    # En production, vérifications supplémentaires
    if settings.is_production():
        if not settings.DATABASE_URL.startswith("postgresql"):
            errors.append("En production, utilisez PostgreSQL, pas SQLite")

    if errors:
        error_msg = "Erreurs de configuration:\n- " + "\n- ".join(errors)
        if raise_on_error:
            raise ValueError(error_msg)
        print(f"⚠️ {error_msg}")
        return False

    print("✅ Configuration chargée et validée")
    return True


# === VALIDATION DES TICKERS ===
import re

TICKER_REGEX = re.compile(r'^[A-Z0-9\.\-\^]{1,12}$')

def validate_ticker(ticker: str) -> bool:
    """
    Valide un ticker boursier

    Args:
        ticker: Symbole boursier (ex: AAPL, MSFT, BTC-USD)

    Returns:
        True si valide, False sinon
    """
    if not ticker or not isinstance(ticker, str):
        return False
    return bool(TICKER_REGEX.match(ticker.upper()))


def sanitize_ticker(ticker: str) -> str:
    """
    Nettoie et valide un ticker

    Args:
        ticker: Symbole boursier

    Returns:
        Ticker nettoyé en majuscules

    Raises:
        ValueError: Si ticker invalide
    """
    if not ticker:
        raise ValueError("Ticker vide")

    cleaned = ticker.strip().upper()

    if not validate_ticker(cleaned):
        raise ValueError(f"Ticker invalide: {ticker}")

    return cleaned


def sanitize_tickers(tickers: list) -> list:
    """
    Valide une liste de tickers

    Args:
        tickers: Liste de symboles

    Returns:
        Liste de tickers nettoyés
    """
    if not tickers:
        return []

    # Limite max 50 tickers par requête
    if len(tickers) > 50:
        raise ValueError("Maximum 50 tickers par requête")

    return [sanitize_ticker(t) for t in tickers]