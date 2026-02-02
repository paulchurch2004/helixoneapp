"""
Utilitaires de gestion du temps - Compatible Python 3.12+

datetime.utcnow() est déprécié depuis Python 3.12.
Ce module fournit des fonctions compatibles utilisant timezone-aware datetimes.
"""

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """
    Retourne l'heure actuelle en UTC (timezone-aware).

    Remplace datetime.utcnow() qui est déprécié en Python 3.12+.

    Returns:
        datetime: Heure actuelle UTC avec timezone info
    """
    return datetime.now(timezone.utc)


def utc_now_naive() -> datetime:
    """
    Retourne l'heure actuelle en UTC sans timezone (pour compatibilité SQLite).

    Certaines bases de données ne supportent pas les timezone-aware datetimes.
    Cette fonction retourne un datetime naïf en UTC.

    Returns:
        datetime: Heure actuelle UTC sans timezone info
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_utc(dt: datetime) -> datetime:
    """
    Convertit un datetime en UTC.

    Args:
        dt: datetime à convertir (peut être naïf ou aware)

    Returns:
        datetime: datetime en UTC
    """
    if dt.tzinfo is None:
        # Assume que c'est déjà en UTC si naïf
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_expired(expires_at: Optional[datetime]) -> bool:
    """
    Vérifie si une date d'expiration est passée.

    Args:
        expires_at: date d'expiration (peut être None)

    Returns:
        bool: True si expiré, False sinon
    """
    if expires_at is None:
        return False

    now = utc_now_naive() if expires_at.tzinfo is None else utc_now()
    return now > expires_at


def days_until(target_date: Optional[datetime]) -> Optional[int]:
    """
    Calcule le nombre de jours jusqu'à une date cible.

    Args:
        target_date: date cible (peut être None)

    Returns:
        int: nombre de jours (négatif si passé), None si target_date est None
    """
    if target_date is None:
        return None

    now = utc_now_naive() if target_date.tzinfo is None else utc_now()
    delta = target_date - now
    return delta.days
