import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import traceback
import json
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional
import threading
from functools import wraps

class HelixOneLogger:
    """Système de logging professionnel pour HelixOne"""
    
    def __init__(self):
        self.log_dir = Path.home() / ".helixone" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration des loggers
        self.app_logger = self._setup_logger("helixone", "app.log", logging.INFO)
        self.error_logger = self._setup_logger("helixone.errors", "errors.log", logging.ERROR)
        self.api_logger = self._setup_logger("helixone.api", "api.log", logging.DEBUG)
        self.user_logger = self._setup_logger("helixone.user", "user_actions.log", logging.INFO)
        
        # Statistiques d'erreurs
        self.error_stats = {}
        self.error_stats_lock = threading.Lock()
        
    def _setup_logger(self, name: str, filename: str, level: int) -> logging.Logger:
        """Configure un logger avec rotation des fichiers"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Éviter la duplication des handlers
        if logger.handlers:
            return logger
        
        # Format détaillé pour les fichiers
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler fichier avec rotation (10MB max, 5 fichiers)
        file_handler = RotatingFileHandler(
            self.log_dir / filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Handler console pour les erreurs critiques
        if level >= logging.ERROR:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.ERROR)
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_app_start(self, version: str = "unknown"):
        """Log du démarrage de l'application"""
        self.app_logger.info("="*60)
        self.app_logger.info(f"HelixOne v{version} - Démarrage")
        self.app_logger.info(f"Python {sys.version}")
        self.app_logger.info(f"Répertoire logs: {self.log_dir}")
        self.app_logger.info("="*60)
    
    def log_user_action(self, action: str, details: Dict[str, Any] = None):
        """Log des actions utilisateur pour analytics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details or {}
        }
        self.user_logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_api_call(self, service: str, endpoint: str, success: bool, 
                     duration_ms: int, error: str = None):
        """Log des appels API pour monitoring"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "endpoint": endpoint,
            "success": success,
            "duration_ms": duration_ms,
            "error": error
        }
        
        if success:
            self.api_logger.info(f"API_SUCCESS: {json.dumps(log_data)}")
        else:
            self.api_logger.error(f"API_ERROR: {json.dumps(log_data)}")
    
    def log_error(self, error: Exception, context: str = "", 
                  user_friendly_msg: str = None, additional_data: Dict = None):
        """Log d'erreur complet avec contexte"""
        error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        error_data = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "user_friendly_msg": user_friendly_msg,
            "additional_data": additional_data or {}
        }
        
        self.error_logger.error(json.dumps(error_data, ensure_ascii=False, indent=2))
        
        # Statistiques
        with self.error_stats_lock:
            error_type = type(error).__name__
            if error_type not in self.error_stats:
                self.error_stats[error_type] = 0
            self.error_stats[error_type] += 1
        
        return error_id
    
    def log_performance(self, operation: str, duration_ms: int, details: Dict = None):
        """Log de performance pour optimisation"""
        perf_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": duration_ms,
            "details": details or {}
        }
        
        level = logging.WARNING if duration_ms > 5000 else logging.INFO
        self.app_logger.log(level, f"PERFORMANCE: {json.dumps(perf_data)}")
    
    def get_error_stats(self) -> Dict[str, int]:
        """Retourne les statistiques d'erreurs"""
        with self.error_stats_lock:
            return self.error_stats.copy()
    
    def get_log_files(self) -> Dict[str, str]:
        """Retourne les chemins des fichiers de log"""
        return {
            "app": str(self.log_dir / "app.log"),
            "errors": str(self.log_dir / "errors.log"),
            "api": str(self.log_dir / "api.log"),
            "user": str(self.log_dir / "user_actions.log")
        }


# Instance globale
_logger_instance = None
_lock = threading.Lock()

def get_logger() -> HelixOneLogger:
    """Singleton thread-safe du logger"""
    global _logger_instance
    if _logger_instance is None:
        with _lock:
            if _logger_instance is None:
                _logger_instance = HelixOneLogger()
    return _logger_instance


# Décorateurs utiles
def log_errors(context: str = "", user_msg: str = None):
    """Décorateur pour logger automatiquement les erreurs"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger()
                error_id = logger.log_error(
                    e, 
                    context=context or f"{func.__module__}.{func.__name__}",
                    user_friendly_msg=user_msg,
                    additional_data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
                )
                # Re-lever l'exception avec l'ID d'erreur
                e.error_id = error_id
                raise
        return wrapper
    return decorator

def log_performance(operation_name: str = ""):
    """Décorateur pour mesurer les performances"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)
                get_logger().log_performance(
                    operation_name or f"{func.__module__}.{func.__name__}",
                    duration_ms
                )
                return result
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                get_logger().log_performance(
                    f"FAILED_{operation_name or func.__name__}",
                    duration_ms,
                    {"error": str(e)}
                )
                raise
        return wrapper
    return decorator

def log_user_action(action_name: str):
    """Décorateur pour logger les actions utilisateur"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                get_logger().log_user_action(
                    action_name,
                    {"success": True, "function": func.__name__}
                )
                return result
            except Exception as e:
                get_logger().log_user_action(
                    action_name,
                    {"success": False, "error": str(e), "function": func.__name__}
                )
                raise
        return wrapper
    return decorator


# Interface simple pour le reste du code
def log_info(message: str):
    """Log d'information simple"""
    get_logger().app_logger.info(message)

def log_warning(message: str):
    """Log d'avertissement simple"""
    get_logger().app_logger.warning(message)

def log_error_simple(message: str, exception: Exception = None):
    """Log d'erreur simple"""
    if exception:
        get_logger().log_error(exception, context=message)
    else:
        get_logger().error_logger.error(message)

def log_debug(message: str):
    """Log de debug simple"""
    get_logger().app_logger.debug(message)


# Gestionnaire d'exceptions global
def setup_global_exception_handler():
    """Configure un gestionnaire global d'exceptions non capturées"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Permettre Ctrl+C
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = get_logger()
        logger.error_logger.critical(
            "Exception non capturée",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception


if __name__ == "__main__":
    # Test du système de logging
    logger = get_logger()
    logger.log_app_start("1.0.0")
    
    # Test des différents types de logs
    log_info("Test du logging info")
    log_warning("Test du logging warning")
    
    # Test d'erreur
    try:
        raise ValueError("Test d'erreur")
    except Exception as e:
        logger.log_error(e, "Test du système", "Une erreur de test s'est produite")
    
    print("Tests terminés. Vérifiez les fichiers de log dans ~/.helixone/logs/")