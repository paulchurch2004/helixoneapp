"""
Configuration centralisée de l'application
Charge toutes les variables depuis .env
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Configuration de l'application"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
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

    # Sécurité JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost", "helixone://"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    
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
    
    # Email
    SENDGRID_API_KEY: str = ""
    FROM_EMAIL: str = "noreply@helixone.com"
    
    # Sentry
    SENTRY_DSN: str = ""


# Instance globale des settings
settings = Settings()


def validate_settings():
    """Vérifie que les variables critiques sont définies"""
    if not settings.SECRET_KEY or settings.SECRET_KEY == "CHANGEZ_MOI":
        raise ValueError(
            "SECRET_KEY non définie dans .env !\n"
            "Générez-en une sur : https://generate-secret.vercel.app/32"
        )
    
    print("✅ Configuration validée")
    return True