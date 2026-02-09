"""
HelixOne Backend API
Point d'entrée principal FastAPI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.database import engine
from app.core.rate_limiter import limiter
from app.models import Base

# Créer les tables au démarrage
Base.metadata.create_all(bind=engine)

# Créer l'application FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend API pour HelixOne - Formation Trading",
)

# Ajouter le rate limiter à l'app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Route de test
@app.get("/")
def root():
    """Route racine - Test de santé de l'API"""
    return {
        "message": "HelixOne API is running",
        "version": settings.APP_VERSION,
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Vérification de santé détaillée"""
    from datetime import datetime
    import psutil
    import os

    # Vérifier la base de données
    db_status = "connected"
    db_error = None
    try:
        from app.core.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = "disconnected"
        db_error = str(e)

    # Métriques système
    system_metrics = {}
    try:
        process = psutil.Process(os.getpid())
        system_metrics = {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "threads": process.num_threads(),
        }
    except Exception:
        pass

    overall_status = "healthy" if db_status == "connected" else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "components": {
            "database": {
                "status": db_status,
                "error": db_error
            },
        },
        "system": system_metrics
    }


@app.on_event("startup")
async def startup_event():
    """Événement exécuté au démarrage de l'application"""
    import logging
    logger = logging.getLogger(__name__)

    # Version identifier pour debug (print pour être visible dans les logs Render)
    BUILD_ID = "2025-02-08-resend-api-v3"
    print(f"=== HelixOne Backend Build: {BUILD_ID} ===")

    # Log email config (sans exposer la clé complète)
    api_key = settings.SMTP_PASSWORD or ""
    key_preview = f"{api_key[:6]}..." if len(api_key) > 6 else "(non configuré)"
    print(f"📧 Email Config: FROM={settings.FROM_EMAIL}, API_KEY={key_preview}")

    # Initialiser le cache Redis (optionnel)
    try:
        from app.core.cache import init_cache
        await init_cache()
    except Exception as e:
        logger.warning(f"Cache Redis non disponible: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Événement exécuté à l'arrêt de l'application"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Arrêt de HelixOne Backend...")

    try:
        from app.core.cache import close_cache
        await close_cache()
    except Exception as e:
        logger.error(f"Erreur fermeture cache: {e}")


# Import des routes
from app.api import auth, licenses, formation

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(licenses.router, prefix="/licenses", tags=["Licenses"])
app.include_router(formation.router, tags=["Formation & Paper Trading"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
