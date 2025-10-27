"""
Services ML - Auto-training et scheduling

Expose les singletons pour faciliter les imports.
"""

from .auto_trainer import AutoTrainer, get_auto_trainer
from .training_scheduler import TrainingScheduler, get_training_scheduler

__all__ = [
    'AutoTrainer',
    'get_auto_trainer',
    'TrainingScheduler',
    'get_training_scheduler'
]
