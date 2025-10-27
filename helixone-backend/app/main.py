"""
HelixOne Backend API
Point d'entrée principal FastAPI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.database import engine
from app.models import Base

# Créer les tables au démarrage
Base.metadata.create_all(bind=engine)

# Initialiser le rate limiter
limiter = Limiter(key_func=get_remote_address)

# Créer l'application FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend API pour HelixOne - Analyse d'actions avec IA",
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
def health_check():
    """Vérification de santé détaillée"""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "database": "connected"
    }


# ============================================
# ÉVÉNEMENTS DE DÉMARRAGE
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Événement exécuté au démarrage de l'application
    - Initialise les connexions IBKR auto-connect
    - Démarre le scheduler d'analyse de portefeuille
    """
    import logging
    import asyncio
    from app.services.ibkr_service import init_ibkr_connections
    from app.services.portfolio.portfolio_scheduler import get_portfolio_scheduler
    from app.core.database import SessionLocal

    logger = logging.getLogger(__name__)
    logger.info("🚀 Démarrage de HelixOne Backend...")

    # Initialiser les connexions IBKR dans une tâche en arrière-plan
    # pour éviter de bloquer le démarrage
    async def init_connections():
        db = SessionLocal()
        try:
            logger.info("📊 Initialisation des connexions IBKR...")
            await init_ibkr_connections(db)
            logger.info("✅ Connexions IBKR initialisées")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation IBKR: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            db.close()

    # Créer une tâche en arrière-plan
    asyncio.create_task(init_connections())

    # Démarrer le scheduler d'analyse de portefeuille
    try:
        logger.info("📅 Démarrage du Portfolio Scheduler...")
        scheduler = get_portfolio_scheduler()
        scheduler.start()
        logger.info("✅ Portfolio Scheduler démarré (7h00 + 17h00 EST)")
    except Exception as e:
        logger.error(f"❌ Erreur démarrage scheduler: {e}")

    # 🆕 Démarrer le ML Training Scheduler
    import os
    if os.getenv('ML_WEEKLY_RETRAIN_ENABLED', 'true').lower() == 'true':
        try:
            from app.services.ml import get_training_scheduler
            logger.info("🧠 Démarrage du ML Training Scheduler...")
            ml_scheduler = get_training_scheduler()
            ml_scheduler.start()

            next_run = ml_scheduler.get_next_run_time()
            if next_run:
                logger.info(f"✅ ML Scheduler démarré (prochain entraînement: {next_run})")
            else:
                logger.info("✅ ML Scheduler démarré")
        except Exception as e:
            logger.error(f"❌ Erreur démarrage ML scheduler: {e}")

    # 🆕 Pré-entraînement des top stocks (en arrière-plan)
    if os.getenv('ML_PRETRAIN_ON_STARTUP', 'true').lower() == 'true':
        async def pretrain():
            try:
                from app.services.ml import get_training_scheduler
                logger.info("🚀 Démarrage pré-entraînement ML...")
                ml_scheduler = get_training_scheduler()
                await ml_scheduler.pretrain_top_stocks()
                logger.info("✅ Pré-entraînement terminé")
            except Exception as e:
                logger.error(f"❌ Erreur pré-entraînement: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Créer tâche en arrière-plan pour ne pas bloquer le démarrage
        asyncio.create_task(pretrain())


@app.on_event("shutdown")
async def shutdown_event():
    """
    Événement exécuté à l'arrêt de l'application
    - Arrête les schedulers proprement
    """
    import logging
    from app.services.portfolio.portfolio_scheduler import get_portfolio_scheduler

    logger = logging.getLogger(__name__)
    logger.info("🛑 Arrêt de HelixOne Backend...")

    # Arrêter le Portfolio scheduler
    try:
        scheduler = get_portfolio_scheduler()
        scheduler.stop()
        logger.info("✅ Portfolio Scheduler arrêté")
    except Exception as e:
        logger.error(f"❌ Erreur arrêt scheduler: {e}")

    # 🆕 Arrêter le ML Training Scheduler
    try:
        from app.services.ml import get_training_scheduler
        ml_scheduler = get_training_scheduler()
        ml_scheduler.stop()
        logger.info("✅ ML Scheduler arrêté")
    except Exception as e:
        logger.error(f"❌ Erreur arrêt ML scheduler: {e}")


# Import des routes
from app.api import auth, licenses, market_data, analysis, formation, data_collection, ibkr, advanced_data_collection, portfolio  # , scenarios

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(licenses.router, prefix="/licenses", tags=["Licenses"])
app.include_router(market_data.router, prefix="/api/market", tags=["Market Data"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(formation.router, tags=["Formation & Paper Trading"])
app.include_router(data_collection.router, prefix="/api/data", tags=["Data Collection"])
app.include_router(advanced_data_collection.router, tags=["Advanced Data Collection"])
app.include_router(ibkr.router, prefix="/api/ibkr", tags=["Interactive Brokers"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio Analysis"])
# app.include_router(scenarios.router, tags=["Scenario Simulations"])  # Temporairement désactivé


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)