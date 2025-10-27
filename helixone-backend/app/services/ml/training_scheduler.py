"""
Training Scheduler - Scheduler pour re-entraînements périodiques

Fonctionnalités :
- Re-entraînement hebdomadaire automatique
- Pré-entraînement au démarrage
- Configuration via variables d'environnement
"""

import logging
import asyncio
from typing import Optional, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import os

from .auto_trainer import AutoTrainer, get_auto_trainer

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """
    Scheduler pour entraînements périodiques des modèles ML

    Usage:
        scheduler = TrainingScheduler(auto_trainer)
        scheduler.start()
        await scheduler.pretrain_top_stocks()
    """

    def __init__(self, auto_trainer: Optional[AutoTrainer] = None):
        """
        Args:
            auto_trainer: Instance de AutoTrainer (utilise singleton si None)
        """
        self.auto_trainer = auto_trainer or get_auto_trainer()
        self.scheduler = AsyncIOScheduler()
        self._is_running = False

        logger.info("TrainingScheduler initialisé")

    def start(self):
        """Démarre le scheduler"""
        if self._is_running:
            logger.warning("Scheduler déjà en cours d'exécution")
            return

        # Configuration depuis .env
        enabled = os.getenv('ML_WEEKLY_RETRAIN_ENABLED', 'true').lower() == 'true'

        if not enabled:
            logger.info("Re-entraînement hebdomadaire désactivé (ML_WEEKLY_RETRAIN_ENABLED=false)")
            return

        # Jour de la semaine (default: dimanche)
        day_of_week = os.getenv('ML_WEEKLY_RETRAIN_DAY', 'sunday').lower()

        # Heure (default: 2h du matin)
        try:
            hour = int(os.getenv('ML_WEEKLY_RETRAIN_HOUR', '2'))
        except ValueError:
            hour = 2
            logger.warning(f"ML_WEEKLY_RETRAIN_HOUR invalide, utilisation de 2h par défaut")

        # Mapper jour en format cron
        day_mapping = {
            'monday': 'mon',
            'tuesday': 'tue',
            'wednesday': 'wed',
            'thursday': 'thu',
            'friday': 'fri',
            'saturday': 'sat',
            'sunday': 'sun'
        }

        cron_day = day_mapping.get(day_of_week, 'sun')

        # Ajouter le job hebdomadaire
        self.scheduler.add_job(
            self.retrain_all_models,
            CronTrigger(day_of_week=cron_day, hour=hour, minute=0),
            id='weekly_retrain',
            name='Re-entraînement hebdomadaire des modèles ML',
            replace_existing=True
        )

        # Démarrer le scheduler
        self.scheduler.start()
        self._is_running = True

        logger.info(f"✅ Scheduler démarré - Re-entraînement chaque {day_of_week} à {hour}h")

    def stop(self):
        """Arrête le scheduler"""
        if not self._is_running:
            return

        self.scheduler.shutdown()
        self._is_running = False

        logger.info("🛑 Scheduler arrêté")

    async def retrain_all_models(self):
        """
        Re-entraîne tous les modèles existants

        Appelé automatiquement par le scheduler hebdomadaire
        """
        logger.info("=" * 80)
        logger.info("🔄 DÉBUT RE-ENTRAÎNEMENT HEBDOMADAIRE")
        logger.info("=" * 80)

        try:
            # Lister tous les modèles existants
            models = self.auto_trainer.list_all_models()

            if not models:
                logger.info("   Aucun modèle à re-entraîner")
                return

            tickers = [model['ticker'] for model in models]

            logger.info(f"   Modèles trouvés: {len(tickers)}")
            logger.info(f"   Tickers: {', '.join(tickers)}")

            # Mode d'entraînement depuis .env
            mode = os.getenv('ML_AUTO_TRAIN_MODE', 'xgboost')

            # Nombre max d'entraînements simultanés
            try:
                max_concurrent = int(os.getenv('ML_MAX_CONCURRENT_TRAINING', '2'))
            except ValueError:
                max_concurrent = 2

            logger.info(f"   Mode: {mode}")
            logger.info(f"   Concurrent: {max_concurrent}")

            # Entraîner en batch (force=True pour re-entraîner même si récents)
            results = {}
            for ticker in tickers:
                success = await self.auto_trainer.force_train(ticker, mode=mode)
                results[ticker] = success

            # Résumé
            success_count = sum(1 for v in results.values() if v)
            failure_count = len(results) - success_count

            logger.info("=" * 80)
            logger.info(f"✅ RE-ENTRAÎNEMENT TERMINÉ")
            logger.info(f"   Succès: {success_count}/{len(results)}")
            logger.info(f"   Échecs: {failure_count}/{len(results)}")

            if failure_count > 0:
                failed = [t for t, s in results.items() if not s]
                logger.warning(f"   Échecs: {', '.join(failed)}")

            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Erreur lors du re-entraînement hebdomadaire: {e}", exc_info=True)

    async def pretrain_top_stocks(self):
        """
        Pré-entraîne les top stocks au démarrage

        Appelé manuellement au démarrage de l'application
        """
        # Vérifier si activé
        enabled = os.getenv('ML_PRETRAIN_ON_STARTUP', 'true').lower() == 'true'

        if not enabled:
            logger.info("Pré-entraînement désactivé (ML_PRETRAIN_ON_STARTUP=false)")
            return

        # Liste des tickers à pré-entraîner
        tickers_env = os.getenv('ML_PRETRAIN_TICKERS', 'AAPL,MSFT,GOOGL,TSLA,AMZN,NVDA,META,NFLX')
        tickers = [t.strip() for t in tickers_env.split(',') if t.strip()]

        if not tickers:
            logger.info("Aucun ticker à pré-entraîner")
            return

        logger.info("=" * 80)
        logger.info("🚀 PRÉ-ENTRAÎNEMENT DES TOP STOCKS")
        logger.info("=" * 80)
        logger.info(f"   Tickers: {', '.join(tickers)}")

        # Configuration
        mode = os.getenv('ML_AUTO_TRAIN_MODE', 'xgboost')
        max_age_days = int(os.getenv('ML_MODEL_MAX_AGE_DAYS', '7'))
        max_concurrent = int(os.getenv('ML_MAX_CONCURRENT_TRAINING', '2'))

        logger.info(f"   Mode: {mode}")
        logger.info(f"   Max age: {max_age_days} jours")
        logger.info(f"   Concurrent: {max_concurrent}")

        try:
            # Entraîner en batch
            results = await self.auto_trainer.batch_train(
                tickers=tickers,
                max_age_days=max_age_days,
                mode=mode,
                max_concurrent=max_concurrent
            )

            # Résumé
            success_count = sum(1 for v in results.values() if v)
            skipped_count = 0  # Modèles déjà à jour
            failure_count = len(results) - success_count

            logger.info("=" * 80)
            logger.info(f"✅ PRÉ-ENTRAÎNEMENT TERMINÉ")
            logger.info(f"   Succès: {success_count}/{len(results)}")
            logger.info(f"   Échecs: {failure_count}/{len(results)}")

            if failure_count > 0:
                failed = [t for t, s in results.items() if not s]
                logger.warning(f"   Échecs: {', '.join(failed)}")

            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Erreur lors du pré-entraînement: {e}", exc_info=True)

    def get_next_run_time(self):
        """Retourne la date/heure du prochain re-entraînement"""
        if not self._is_running:
            return None

        job = self.scheduler.get_job('weekly_retrain')
        if job:
            return job.next_run_time

        return None


# ============================================================================
# SINGLETON
# ============================================================================

_training_scheduler_instance: Optional[TrainingScheduler] = None

def get_training_scheduler() -> TrainingScheduler:
    """Retourne l'instance singleton de TrainingScheduler"""
    global _training_scheduler_instance
    if _training_scheduler_instance is None:
        _training_scheduler_instance = TrainingScheduler()
    return _training_scheduler_instance
