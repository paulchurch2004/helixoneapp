"""
Endpoints API pour les données de marché
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional
from datetime import date, datetime, timedelta
import logging

from app.services.data_sources.aggregator import get_default_aggregator
from app.services.data_sources.base import (
    DataUnavailableError,
    InvalidTickerError
)
from app.schemas.market import (
    Quote,
    HistoricalData,
    Fundamentals,
    NewsResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/quote/{ticker}", response_model=Quote, tags=["Market Data"])
async def get_quote(ticker: str):
    """
    Récupère le quote en temps réel pour une action

    Args:
        ticker: Symbole de l'action (ex: AAPL, TSLA, MSFT)

    Returns:
        Quote avec prix actuel et informations

    Raises:
        404: Ticker non trouvé
        503: Données non disponibles
    """
    try:
        aggregator = get_default_aggregator()
        quote = await aggregator.get_quote(ticker)
        return quote

    except InvalidTickerError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{ticker}' non trouvé"
        )

    except DataUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Données non disponibles pour '{ticker}'"
        )

    except Exception as e:
        logger.error(f"Erreur inattendue get_quote({ticker}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur serveur"
        )


@router.get("/historical/{ticker}", response_model=HistoricalData, tags=["Market Data"])
async def get_historical(
    ticker: str,
    start_date: Optional[date] = Query(
        None,
        description="Date de début (YYYY-MM-DD). Par défaut: 2 ans en arrière"
    ),
    end_date: Optional[date] = Query(
        None,
        description="Date de fin (YYYY-MM-DD). Par défaut: aujourd'hui"
    ),
    interval: str = Query(
        "1d",
        description="Intervalle: 1d (jour), 1wk (semaine), 1mo (mois)"
    )
):
    """
    Récupère les données historiques pour une action

    Args:
        ticker: Symbole de l'action
        start_date: Date de début (défaut: 2 ans en arrière)
        end_date: Date de fin (défaut: aujourd'hui)
        interval: Intervalle (1d, 1wk, 1mo)

    Returns:
        HistoricalData avec liste de prix

    Raises:
        404: Ticker non trouvé
        503: Données non disponibles
    """
    # Dates par défaut
    if not end_date:
        end_date = date.today()

    if not start_date:
        start_date = end_date - timedelta(days=730)  # 2 ans

    # Valider l'intervalle
    valid_intervals = ["1d", "1wk", "1mo"]
    if interval not in valid_intervals:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Intervalle invalide. Valeurs acceptées: {valid_intervals}"
        )

    # Valider les dates
    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date doit être antérieur à end_date"
        )

    try:
        aggregator = get_default_aggregator()
        historical = await aggregator.get_historical(
            ticker,
            start_date,
            end_date,
            interval
        )
        return historical

    except InvalidTickerError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{ticker}' non trouvé"
        )

    except DataUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Données historiques non disponibles pour '{ticker}'"
        )

    except Exception as e:
        logger.error(f"Erreur inattendue get_historical({ticker}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur serveur"
        )


@router.get("/fundamentals/{ticker}", response_model=Fundamentals, tags=["Market Data"])
async def get_fundamentals(ticker: str):
    """
    Récupère les données fondamentales d'une entreprise

    Args:
        ticker: Symbole de l'action

    Returns:
        Fundamentals avec ratios financiers, profitabilité, etc.

    Raises:
        404: Ticker non trouvé
        503: Données non disponibles
    """
    try:
        aggregator = get_default_aggregator()
        fundamentals = await aggregator.get_fundamentals(ticker)
        return fundamentals

    except InvalidTickerError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{ticker}' non trouvé"
        )

    except DataUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Fondamentaux non disponibles pour '{ticker}'"
        )

    except Exception as e:
        logger.error(f"Erreur inattendue get_fundamentals({ticker}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur serveur"
        )


@router.get("/news/{ticker}", response_model=NewsResponse, tags=["Market Data"])
async def get_news(
    ticker: str,
    limit: int = Query(50, ge=1, le=100, description="Nombre d'articles (max 100)")
):
    """
    Récupère les actualités pour une action

    Args:
        ticker: Symbole de l'action
        limit: Nombre max d'articles (défaut: 50, max: 100)

    Returns:
        NewsResponse avec liste d'articles

    Raises:
        500: Erreur serveur
    """
    try:
        aggregator = get_default_aggregator()
        articles = await aggregator.get_news(ticker, limit=limit)

        return NewsResponse(
            ticker=ticker.upper(),
            articles=articles,
            total=len(articles),
            source="aggregated"
        )

    except Exception as e:
        logger.error(f"Erreur inattendue get_news({ticker}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur serveur"
        )


@router.get("/news", response_model=NewsResponse, tags=["Market Data"])
async def get_general_news(
    limit: int = Query(50, ge=1, le=100, description="Nombre d'articles (max 100)")
):
    """
    Récupère les actualités financières générales

    Args:
        limit: Nombre max d'articles (défaut: 50, max: 100)

    Returns:
        NewsResponse avec liste d'articles

    Raises:
        500: Erreur serveur
    """
    try:
        aggregator = get_default_aggregator()
        articles = await aggregator.get_news(ticker=None, limit=limit)

        return NewsResponse(
            ticker=None,
            articles=articles,
            total=len(articles),
            source="aggregated"
        )

    except Exception as e:
        logger.error(f"Erreur inattendue get_general_news: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur serveur"
        )


@router.get("/sources", tags=["Market Data"])
async def get_available_sources():
    """
    Retourne la liste des sources de données disponibles

    Returns:
        Dict avec liste des sources
    """
    aggregator = get_default_aggregator()
    sources = aggregator.get_available_sources()

    return {
        "sources": sources,
        "total": len(sources)
    }
