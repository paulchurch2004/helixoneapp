"""
Interface abstraite pour les sources de données financières
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import date, datetime
import logging

from app.schemas.market import (
    Quote,
    HistoricalData,
    Fundamentals,
    NewsArticle,
    SearchResult
)

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """
    Interface abstraite pour toutes les sources de données financières

    Chaque source (Yahoo, Finnhub, FMP, etc.) doit implémenter cette interface
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"datasource.{self.name}")

    @abstractmethod
    async def get_quote(self, ticker: str) -> Optional[Quote]:
        """
        Récupère le quote en temps réel pour un ticker

        Args:
            ticker: Symbole de l'action (ex: AAPL)

        Returns:
            Quote ou None si erreur
        """
        pass

    @abstractmethod
    async def get_historical(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> Optional[HistoricalData]:
        """
        Récupère les données historiques

        Args:
            ticker: Symbole de l'action
            start_date: Date de début
            end_date: Date de fin
            interval: Intervalle (1d, 1wk, 1mo)

        Returns:
            HistoricalData ou None si erreur
        """
        pass

    @abstractmethod
    async def get_fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        """
        Récupère les données fondamentales

        Args:
            ticker: Symbole de l'action

        Returns:
            Fundamentals ou None si erreur
        """
        pass

    async def get_news(
        self,
        ticker: Optional[str] = None,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Récupère les actualités

        Args:
            ticker: Symbole optionnel (si None, news générales)
            limit: Nombre max d'articles

        Returns:
            Liste d'articles (vide si non implémenté ou erreur)
        """
        self.logger.warning(f"{self.name} ne supporte pas get_news()")
        return []

    async def search(self, query: str) -> List[SearchResult]:
        """
        Recherche de tickers

        Args:
            query: Terme de recherche

        Returns:
            Liste de résultats (vide si non implémenté)
        """
        self.logger.warning(f"{self.name} ne supporte pas search()")
        return []

    def is_available(self) -> bool:
        """
        Vérifie si la source est disponible

        Returns:
            True si la source est configurée et accessible
        """
        # Par défaut, vérifier si une API key est requise
        return True

    def _log_error(self, method: str, error: Exception, ticker: Optional[str] = None):
        """Helper pour logger les erreurs"""
        ticker_info = f" for {ticker}" if ticker else ""
        self.logger.error(
            f"{self.name}.{method}{ticker_info} failed: {type(error).__name__}: {str(error)}"
        )


class DataSourceError(Exception):
    """Exception pour les erreurs de source de données"""

    def __init__(self, source: str, message: str):
        self.source = source
        self.message = message
        super().__init__(f"[{source}] {message}")


class DataUnavailableError(DataSourceError):
    """Exception quand les données ne sont pas disponibles"""
    pass


class RateLimitError(DataSourceError):
    """Exception quand la limite de requêtes est atteinte"""
    pass


class InvalidTickerError(DataSourceError):
    """Exception quand le ticker est invalide"""
    pass
