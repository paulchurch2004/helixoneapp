"""
Auto Trainer - Service d'entraînement automatique des modèles ML

Fonctionnalités :
- Entraînement automatique à la demande
- Vérification de l'âge des modèles
- Re-entraînement si modèle trop vieux
- Logs détaillés et gestion d'erreurs
"""

import logging
import asyncio
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os

logger = logging.getLogger(__name__)


class AutoTrainer:
    """
    Service d'entraînement automatique des modèles ML

    Usage:
        trainer = AutoTrainer()
        success = await trainer.train_if_needed('AAPL', max_age_days=7)
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        trainer_script: Optional[str] = None
    ):
        """
        Args:
            models_dir: Répertoire des modèles (défaut: ml_models/saved_models)
            trainer_script: Script d'entraînement (défaut: ml_models/model_trainer.py)
        """
        # Chemins
        if models_dir is None:
            base_dir = Path(__file__).parent.parent.parent.parent
            models_dir = base_dir / 'ml_models' / 'saved_models'

        if trainer_script is None:
            base_dir = Path(__file__).parent.parent.parent.parent
            trainer_script = base_dir / 'ml_models' / 'model_trainer.py'

        self.models_dir = Path(models_dir)
        self.trainer_script = Path(trainer_script)

        # Vérifier que le script existe
        if not self.trainer_script.exists():
            logger.error(f"Script d'entraînement non trouvé : {self.trainer_script}")

        # Lock pour éviter entraînements simultanés du même ticker
        self._training_locks: Dict[str, asyncio.Lock] = {}

        logger.info(f"AutoTrainer initialisé")
        logger.info(f"   Modèles: {self.models_dir}")
        logger.info(f"   Trainer: {self.trainer_script}")

    async def train_if_needed(
        self,
        ticker: str,
        max_age_days: int = 7,
        mode: str = 'xgboost',
        force: bool = False
    ) -> bool:
        """
        Entraîne un modèle si nécessaire (n'existe pas ou trop vieux)

        Args:
            ticker: Ticker de l'action
            max_age_days: Âge maximum du modèle en jours avant re-entraînement
            mode: Mode d'entraînement (xgboost ou ensemble)
            force: Force le re-entraînement même si modèle récent

        Returns:
            bool: True si modèle prêt (existant valide ou nouvellement entraîné)
        """
        # Lock par ticker pour éviter entraînements concurrents
        if ticker not in self._training_locks:
            self._training_locks[ticker] = asyncio.Lock()

        async with self._training_locks[ticker]:
            # Vérifier si modèle existe
            model_exists = self._model_exists(ticker)

            if force:
                logger.info(f"🔄 Re-entraînement forcé demandé pour {ticker}")
                return await self._run_training(ticker, mode)

            if not model_exists:
                logger.info(f"🔧 Aucun modèle trouvé pour {ticker}, entraînement automatique...")
                return await self._run_training(ticker, mode)

            # Modèle existe, vérifier l'âge
            age_days = self.get_model_age(ticker)

            if age_days is None:
                logger.warning(f"⚠️ Impossible de déterminer l'âge du modèle {ticker}, re-entraînement...")
                return await self._run_training(ticker, mode)

            if age_days > max_age_days:
                logger.info(f"🔄 Modèle {ticker} obsolète ({age_days}j > {max_age_days}j), re-entraînement...")
                return await self._run_training(ticker, mode)

            # Modèle valide
            logger.info(f"✅ Modèle {ticker} existant valide (âge: {age_days}j)")
            return True

    async def force_train(self, ticker: str, mode: str = 'xgboost') -> bool:
        """
        Force l'entraînement d'un modèle (même s'il existe et est récent)

        Args:
            ticker: Ticker de l'action
            mode: Mode d'entraînement

        Returns:
            bool: True si succès
        """
        return await self.train_if_needed(ticker, force=True, mode=mode)

    def _model_exists(self, ticker: str) -> bool:
        """Vérifie si un modèle existe pour un ticker"""
        ticker_dir = self.models_dir / ticker

        if not ticker_dir.exists():
            return False

        # Vérifier si au moins un type de modèle existe
        xgboost_dir = ticker_dir / 'xgboost'
        ensemble_dir = ticker_dir / 'ensemble'

        xgboost_exists = (xgboost_dir / 'xgb_1d.json').exists()
        ensemble_exists = (ensemble_dir / 'ensemble.pkl').exists() if ensemble_dir.exists() else False

        return xgboost_exists or ensemble_exists

    def get_model_age(self, ticker: str) -> Optional[int]:
        """
        Retourne l'âge du modèle en jours

        Args:
            ticker: Ticker de l'action

        Returns:
            int: Âge en jours, ou None si impossible à déterminer
        """
        ticker_dir = self.models_dir / ticker

        if not ticker_dir.exists():
            return None

        # Chercher le fichier training_metadata.json
        metadata_file = ticker_dir / 'training_metadata.json'

        if not metadata_file.exists():
            # Fallback : utiliser la date de modification du dossier
            mod_time = datetime.fromtimestamp(ticker_dir.stat().st_mtime)
            age = datetime.now() - mod_time
            return age.days

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Parser la date d'entraînement
            trained_at_str = metadata.get('trained_at')
            if trained_at_str:
                trained_at = datetime.fromisoformat(trained_at_str)
                age = datetime.now() - trained_at
                return age.days

            return None

        except Exception as e:
            logger.error(f"Erreur lecture métadonnées {ticker}: {e}")
            return None

    async def _run_training(self, ticker: str, mode: str = 'xgboost') -> bool:
        """
        Exécute l'entraînement d'un modèle

        Args:
            ticker: Ticker de l'action
            mode: Mode d'entraînement (xgboost ou ensemble)

        Returns:
            bool: True si succès
        """
        try:
            # Configuration
            start_date = os.getenv('ML_TRAIN_START_DATE', '2022-01-01')

            # Trouver l'interpréteur Python du venv
            # __file__ = .../helixone-backend/app/services/ml/auto_trainer.py
            # On remonte jusqu'à helixone/ (parent de helixone-backend/)
            base_dir = Path(__file__).parent.parent.parent.parent.parent
            python_exe = base_dir / 'venv' / 'bin' / 'python3'

            if not python_exe.exists():
                # Fallback sur python système
                python_exe = 'python3'
                logger.warning(f"Venv non trouvé à {base_dir / 'venv'}, utilisation de python3 système")

            # Le répertoire de travail doit être helixone-backend pour que
            # model_trainer.py utilise les bons chemins relatifs
            backend_dir = Path(__file__).parent.parent.parent.parent

            # Commande d'entraînement
            cmd = [
                str(python_exe),
                str(self.trainer_script),
                '--ticker', ticker,
                '--mode', mode,
                '--no-optimize',  # Pas d'optimisation pour être rapide
                '--start-date', start_date
            ]

            logger.info(f"🚀 Lancement entraînement {ticker} ({mode})")
            logger.info(f"   Commande: {' '.join(cmd)}")
            logger.info(f"   Working dir: {backend_dir}")

            # Exécuter avec cwd=helixone-backend pour que les chemins relatifs
            # dans model_trainer.py soient corrects
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(backend_dir)
            )

            # Attendre la fin
            stdout, stderr = await process.communicate()

            # Logger stdout/stderr pour debug
            if stdout:
                stdout_msg = stdout.decode('utf-8')
                logger.debug(f"STDOUT: {stdout_msg[-1000:]}")  # Derniers 1000 caractères

            # Vérifier le code de retour
            if process.returncode == 0:
                logger.info(f"✅ Entraînement {ticker} terminé avec succès")

                # Vérifier que le modèle a bien été créé
                if not self._model_exists(ticker):
                    logger.error(f"❌ Entraînement {ticker} a réussi mais modèle non créé!")
                    if stderr:
                        error_msg = stderr.decode('utf-8')
                        logger.error(f"   STDERR: {error_msg}")
                    return False

                # Afficher les métriques si disponibles
                self._log_training_metrics(ticker)

                return True
            else:
                logger.error(f"❌ Entraînement {ticker} échoué (code: {process.returncode})")

                # Logger stderr pour debug
                if stderr:
                    error_msg = stderr.decode('utf-8')
                    logger.error(f"   Erreur: {error_msg[:500]}")  # Premiers 500 caractères

                return False

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement {ticker}: {e}", exc_info=True)
            return False

    def _log_training_metrics(self, ticker: str):
        """Affiche les métriques d'entraînement dans les logs"""
        try:
            ticker_dir = self.models_dir / ticker

            # Lire métadonnées globales
            metadata_file = ticker_dir / 'training_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                logger.info(f"   📊 Métadonnées {ticker}:")
                logger.info(f"      - Échantillons: {metadata.get('n_samples')}")
                logger.info(f"      - Features: {metadata.get('n_features')}")
                logger.info(f"      - Durée: {metadata.get('training_time_seconds', 0):.1f}s")

            # Lire métriques XGBoost si disponibles
            xgboost_dir = ticker_dir / 'xgboost'
            if xgboost_dir.exists():
                for horizon in ['1d', '3d', '7d']:
                    meta_file = xgboost_dir / f'xgb_{horizon}.meta.json'
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            meta = json.load(f)

                        metrics = meta.get('training_metrics', {})
                        val_acc = metrics.get('val_accuracy', 0)

                        logger.info(f"      - {horizon}: précision validation = {val_acc*100:.1f}%")

        except Exception as e:
            logger.debug(f"Impossible de lire métriques {ticker}: {e}")

    def list_all_models(self) -> List[Dict]:
        """
        Liste tous les modèles disponibles avec leurs métadonnées

        Returns:
            Liste de dictionnaires avec info sur chaque modèle
        """
        models = []

        if not self.models_dir.exists():
            return models

        for ticker_dir in self.models_dir.iterdir():
            if not ticker_dir.is_dir():
                continue

            ticker = ticker_dir.name

            # Vérifier si modèle valide
            if not self._model_exists(ticker):
                continue

            # Lire métadonnées
            metadata_file = ticker_dir / 'training_metadata.json'
            metadata = {}

            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass

            age_days = self.get_model_age(ticker)

            models.append({
                'ticker': ticker,
                'age_days': age_days,
                'mode': metadata.get('mode', 'unknown'),
                'n_samples': metadata.get('n_samples'),
                'n_features': metadata.get('n_features'),
                'trained_at': metadata.get('trained_at'),
                'training_time': metadata.get('training_time_seconds')
            })

        return models

    async def batch_train(
        self,
        tickers: List[str],
        max_age_days: int = 7,
        mode: str = 'xgboost',
        max_concurrent: int = 2
    ) -> Dict[str, bool]:
        """
        Entraîne plusieurs tickers en parallèle (limité)

        Args:
            tickers: Liste de tickers
            max_age_days: Âge maximum
            mode: Mode d'entraînement
            max_concurrent: Nombre max d'entraînements simultanés

        Returns:
            Dict {ticker: success}
        """
        results = {}

        # Créer semaphore pour limiter concurrence
        semaphore = asyncio.Semaphore(max_concurrent)

        async def train_with_semaphore(ticker: str):
            async with semaphore:
                success = await self.train_if_needed(ticker, max_age_days, mode)
                results[ticker] = success

        # Lancer tous les entraînements
        tasks = [train_with_semaphore(ticker) for ticker in tickers]
        await asyncio.gather(*tasks)

        return results


# ============================================================================
# SINGLETON
# ============================================================================

_auto_trainer_instance: Optional[AutoTrainer] = None

def get_auto_trainer() -> AutoTrainer:
    """Retourne l'instance singleton de AutoTrainer"""
    global _auto_trainer_instance
    if _auto_trainer_instance is None:
        _auto_trainer_instance = AutoTrainer()
    return _auto_trainer_instance
