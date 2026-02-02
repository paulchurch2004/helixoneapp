"""
Service de Cache Redis pour HelixOne

Fournit un cache centralisé pour:
- Quotes de marché (TTL: 60s)
- Analyses fondamentales (TTL: 1h)
- Prédictions ML (TTL: 4h)
- Données macro (TTL: 24h)

Usage:
    from app.core.cache import cache

    # Setter avec TTL automatique
    await cache.set_quote("AAPL", {"price": 150.0})

    # Getter
    data = await cache.get_quote("AAPL")

    # Decorator pour cacher automatiquement
    @cache.cached(prefix="analysis", ttl=3600)
    async def get_analysis(ticker: str):
        ...
"""

import json
import logging
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheService:
    """Service de cache Redis asynchrone"""

    # TTL par type de données (en secondes)
    TTL_QUOTE = 60           # 1 minute
    TTL_ANALYSIS = 3600      # 1 heure
    TTL_FUNDAMENTAL = 3600   # 1 heure
    TTL_ML_PREDICTION = 14400  # 4 heures
    TTL_MACRO = 86400        # 24 heures
    TTL_NEWS = 1800          # 30 minutes
    TTL_SENTIMENT = 3600     # 1 heure

    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._enabled = True

    async def connect(self) -> bool:
        """
        Établit la connexion à Redis

        Returns:
            True si connecté, False sinon
        """
        try:
            self._pool = ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=20,
                decode_responses=True
            )
            self._client = redis.Redis(connection_pool=self._pool)

            # Test de connexion
            await self._client.ping()
            logger.info("✅ Cache Redis connecté")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Redis non disponible, cache désactivé: {e}")
            self._enabled = False
            return False

    async def disconnect(self):
        """Ferme la connexion Redis"""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Cache Redis déconnecté")

    def _make_key(self, prefix: str, identifier: str) -> str:
        """Génère une clé de cache"""
        return f"helixone:{prefix}:{identifier}"

    async def get(self, prefix: str, identifier: str) -> Optional[Any]:
        """
        Récupère une valeur du cache

        Args:
            prefix: Type de données (quote, analysis, ml, etc.)
            identifier: Identifiant unique (ticker, user_id, etc.)

        Returns:
            Données ou None si non trouvé/expiré
        """
        if not self._enabled or not self._client:
            return None

        try:
            key = self._make_key(prefix, identifier)
            data = await self._client.get(key)

            if data:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(data)

            logger.debug(f"Cache MISS: {key}")
            return None

        except Exception as e:
            logger.error(f"Erreur cache get: {e}")
            return None

    async def set(
        self,
        prefix: str,
        identifier: str,
        data: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Stocke une valeur dans le cache

        Args:
            prefix: Type de données
            identifier: Identifiant unique
            data: Données à cacher (doit être JSON serializable)
            ttl: Durée de vie en secondes (optionnel)

        Returns:
            True si succès
        """
        if not self._enabled or not self._client:
            return False

        try:
            key = self._make_key(prefix, identifier)

            # TTL par défaut selon le type
            if ttl is None:
                ttl = self._get_default_ttl(prefix)

            await self._client.setex(
                key,
                timedelta(seconds=ttl),
                json.dumps(data, default=str)
            )

            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Erreur cache set: {e}")
            return False

    async def delete(self, prefix: str, identifier: str) -> bool:
        """Supprime une entrée du cache"""
        if not self._enabled or not self._client:
            return False

        try:
            key = self._make_key(prefix, identifier)
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Erreur cache delete: {e}")
            return False

    async def clear_prefix(self, prefix: str) -> int:
        """
        Supprime toutes les entrées avec un préfixe donné

        Returns:
            Nombre d'entrées supprimées
        """
        if not self._enabled or not self._client:
            return 0

        try:
            pattern = f"helixone:{prefix}:*"
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self._client.delete(*keys)

            logger.info(f"Cache cleared: {len(keys)} entrées pour {prefix}")
            return len(keys)

        except Exception as e:
            logger.error(f"Erreur cache clear: {e}")
            return 0

    def _get_default_ttl(self, prefix: str) -> int:
        """Retourne le TTL par défaut selon le type"""
        ttl_map = {
            "quote": self.TTL_QUOTE,
            "quotes": self.TTL_QUOTE,
            "analysis": self.TTL_ANALYSIS,
            "fundamental": self.TTL_FUNDAMENTAL,
            "ml": self.TTL_ML_PREDICTION,
            "prediction": self.TTL_ML_PREDICTION,
            "macro": self.TTL_MACRO,
            "news": self.TTL_NEWS,
            "sentiment": self.TTL_SENTIMENT,
        }
        return ttl_map.get(prefix, 3600)  # 1h par défaut

    # === Méthodes raccourcies par type ===

    async def get_quote(self, ticker: str) -> Optional[dict]:
        """Récupère une quote du cache"""
        return await self.get("quote", ticker.upper())

    async def set_quote(self, ticker: str, data: dict) -> bool:
        """Cache une quote"""
        return await self.set("quote", ticker.upper(), data, self.TTL_QUOTE)

    async def get_analysis(self, ticker: str) -> Optional[dict]:
        """Récupère une analyse du cache"""
        return await self.get("analysis", ticker.upper())

    async def set_analysis(self, ticker: str, data: dict) -> bool:
        """Cache une analyse"""
        return await self.set("analysis", ticker.upper(), data, self.TTL_ANALYSIS)

    async def get_ml_prediction(self, ticker: str) -> Optional[dict]:
        """Récupère une prédiction ML du cache"""
        return await self.get("ml", ticker.upper())

    async def set_ml_prediction(self, ticker: str, data: dict) -> bool:
        """Cache une prédiction ML"""
        return await self.set("ml", ticker.upper(), data, self.TTL_ML_PREDICTION)

    async def get_sentiment(self, ticker: str) -> Optional[dict]:
        """Récupère le sentiment du cache"""
        return await self.get("sentiment", ticker.upper())

    async def set_sentiment(self, ticker: str, data: dict) -> bool:
        """Cache le sentiment"""
        return await self.set("sentiment", ticker.upper(), data, self.TTL_SENTIMENT)

    # === Decorator pour caching automatique ===

    def cached(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable] = None
    ):
        """
        Decorator pour cacher automatiquement le résultat d'une fonction

        Usage:
            @cache.cached(prefix="analysis", ttl=3600)
            async def get_analysis(ticker: str):
                # Cette fonction sera cachée automatiquement
                return {"data": "..."}
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Construire la clé de cache
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # Clé par défaut basée sur les arguments
                    key_parts = [str(a) for a in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

                # Essayer de récupérer du cache
                cached_data = await self.get(prefix, cache_key)
                if cached_data is not None:
                    return cached_data

                # Exécuter la fonction
                result = await func(*args, **kwargs)

                # Cacher le résultat
                if result is not None:
                    await self.set(prefix, cache_key, result, ttl)

                return result

            return wrapper
        return decorator

    # === Stats et monitoring ===

    async def get_stats(self) -> dict:
        """Retourne les statistiques du cache"""
        if not self._enabled or not self._client:
            return {"enabled": False}

        try:
            info = await self._client.info("stats")
            memory = await self._client.info("memory")

            # Compter les clés par préfixe
            prefixes = ["quote", "analysis", "ml", "sentiment", "macro", "news"]
            counts = {}
            for prefix in prefixes:
                pattern = f"helixone:{prefix}:*"
                count = 0
                async for _ in self._client.scan_iter(match=pattern):
                    count += 1
                counts[prefix] = count

            return {
                "enabled": True,
                "connected": True,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "memory_used": memory.get("used_memory_human", "N/A"),
                "keys_by_type": counts,
                "total_keys": sum(counts.values())
            }

        except Exception as e:
            logger.error(f"Erreur stats cache: {e}")
            return {"enabled": True, "connected": False, "error": str(e)}


# === Singleton ===
cache = CacheService()


async def init_cache():
    """Initialise le cache au démarrage de l'application"""
    await cache.connect()


async def close_cache():
    """Ferme le cache à l'arrêt de l'application"""
    await cache.disconnect()
