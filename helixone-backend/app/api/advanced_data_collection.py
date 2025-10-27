"""
API endpoints pour la collecte de données avancées
Sources:
- Alpha Vantage (marché + fondamentaux + indicateurs techniques)
- FRED (données macroéconomiques USA - illimité)
- Finnhub (news + sentiment + analystes)
- Financial Modeling Prep (états financiers + ratios + ownership + insider trading)
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.alpha_vantage_collector import get_alpha_vantage_collector
from app.services.fred_collector import get_fred_collector
from app.services.finnhub_collector import get_finnhub_collector
from app.services.fmp_collector import get_fmp_collector
from app.services.twelvedata_collector import get_twelvedata_collector
from app.services.worldbank_collector import get_worldbank_collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data/advanced", tags=["Advanced Data Collection"])


# ============================================================================
# SCHEMAS
# ============================================================================

class AlphaVantageQuoteRequest(BaseModel):
    symbol: str


class AlphaVantageHistoricalRequest(BaseModel):
    symbol: str
    outputsize: str = "full"  # "compact" (100 days) or "full" (20+ years)


class AlphaVantageIntradayRequest(BaseModel):
    symbol: str
    interval: str = "5min"  # "1min", "5min", "15min", "30min", "60min"
    outputsize: str = "full"


class AlphaVantageFundamentalRequest(BaseModel):
    symbols: List[str]


class FREDSeriesRequest(BaseModel):
    series_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class FREDMultipleSeriesRequest(BaseModel):
    series_ids: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class FinnhubNewsRequest(BaseModel):
    symbol: str
    from_date: Optional[str] = None  # Format: YYYY-MM-DD
    to_date: Optional[str] = None


class FinnhubSentimentRequest(BaseModel):
    symbol: str


class FinnhubRecommendationsRequest(BaseModel):
    symbol: str


class FinnhubPriceTargetRequest(BaseModel):
    symbol: str


class FinnhubEarningsRequest(BaseModel):
    symbol: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None


class FMPSymbolRequest(BaseModel):
    symbol: str
    period: str = "annual"  # "annual" ou "quarter"
    limit: int = 10


class FMPDividendsRequest(BaseModel):
    symbol: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None


class FMPInsiderTradingRequest(BaseModel):
    symbol: str
    limit: int = 100


class TwelveDataRequest(BaseModel):
    symbol: str
    interval: str = "1day"
    outputsize: int = 30


# ============================================================================
# ALPHA VANTAGE ENDPOINTS
# ============================================================================

@router.post("/alphavantage/quote")
async def get_alpha_vantage_quote(
    request: AlphaVantageQuoteRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer la quote temps réel d'un symbole via Alpha Vantage

    Args:
        symbol: Symbole du ticker (ex: "AAPL")

    Returns:
        Quote avec prix, volume, timestamp
    """
    try:
        logger.info(f"Alpha Vantage quote demandée pour {request.symbol} par {current_user.email}")

        av = get_alpha_vantage_collector()
        quote = av.get_quote(request.symbol)

        return {
            "success": True,
            "data": quote,
            "source": "alpha_vantage",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Erreur Alpha Vantage quote {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alphavantage/daily")
async def get_alpha_vantage_daily(
    request: AlphaVantageHistoricalRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Collecter les données journalières via Alpha Vantage

    Args:
        symbol: Symbole du ticker
        outputsize: "compact" (100 jours) ou "full" (20+ ans)

    Returns:
        Job ID pour suivre la collecte
    """
    try:
        logger.info(f"Alpha Vantage daily collecte demandée pour {request.symbol}")

        av = get_alpha_vantage_collector()

        # Collecter les données
        data, meta_data = av.get_daily_data(request.symbol, request.outputsize)

        # TODO: Stocker dans la base de données
        # Pour l'instant, retourner les données directement

        return {
            "success": True,
            "symbol": request.symbol,
            "records_collected": len(data),
            "date_range": {
                "start": data['timestamp'].min().isoformat(),
                "end": data['timestamp'].max().isoformat()
            },
            "metadata": meta_data,
            "source": "alpha_vantage"
        }

    except Exception as e:
        logger.error(f"Erreur Alpha Vantage daily {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alphavantage/intraday")
async def get_alpha_vantage_intraday(
    request: AlphaVantageIntradayRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Collecter les données intraday via Alpha Vantage

    Args:
        symbol: Symbole du ticker
        interval: "1min", "5min", "15min", "30min", "60min"
        outputsize: "compact" ou "full"

    Returns:
        Données intraday
    """
    try:
        logger.info(f"Alpha Vantage intraday {request.interval} pour {request.symbol}")

        av = get_alpha_vantage_collector()
        data, meta_data = av.get_intraday_data(
            request.symbol,
            request.interval,
            request.outputsize
        )

        return {
            "success": True,
            "symbol": request.symbol,
            "interval": request.interval,
            "records_collected": len(data),
            "date_range": {
                "start": data['timestamp'].min().isoformat(),
                "end": data['timestamp'].max().isoformat()
            },
            "metadata": meta_data,
            "source": "alpha_vantage"
        }

    except Exception as e:
        logger.error(f"Erreur Alpha Vantage intraday {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alphavantage/fundamentals")
async def get_alpha_vantage_fundamentals(
    request: AlphaVantageFundamentalRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Collecter les données fondamentales (company overview) pour plusieurs symboles

    Args:
        symbols: Liste de symboles

    Returns:
        Données fondamentales pour chaque symbole
    """
    try:
        logger.info(f"Alpha Vantage fundamentals pour {len(request.symbols)} symboles")

        av = get_alpha_vantage_collector()
        results = []

        for symbol in request.symbols:
            try:
                overview = av.get_company_overview(symbol)
                results.append({
                    "symbol": symbol,
                    "success": True,
                    "data": overview
                })
            except Exception as e:
                logger.warning(f"Erreur collecte fundamentals {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "success": False,
                    "error": str(e)
                })

        return {
            "success": True,
            "total_requested": len(request.symbols),
            "total_collected": sum(1 for r in results if r['success']),
            "results": results,
            "usage": av.get_usage_stats()
        }

    except Exception as e:
        logger.error(f"Erreur Alpha Vantage fundamentals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alphavantage/usage")
async def get_alpha_vantage_usage(
    current_user: User = Depends(get_current_user)
):
    """
    Obtenir les statistiques d'utilisation d'Alpha Vantage

    Returns:
        Nombre de requêtes utilisées aujourd'hui
    """
    try:
        av = get_alpha_vantage_collector()
        stats = av.get_usage_stats()

        return {
            "success": True,
            "usage": stats
        }

    except Exception as e:
        logger.error(f"Erreur Alpha Vantage usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FRED ENDPOINTS
# ============================================================================

@router.post("/fred/series")
async def get_fred_series(
    request: FREDSeriesRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer une série économique FRED

    Args:
        series_id: Code de la série (ex: "DFF", "CPIAUCSL")
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)

    Returns:
        Données de la série
    """
    try:
        logger.info(f"FRED série {request.series_id} demandée par {current_user.email}")

        fred = get_fred_collector()
        data = fred.get_series(
            request.series_id,
            request.start_date,
            request.end_date
        )

        # Convertir en format JSON-friendly
        data_dict = {
            "dates": data.index.strftime('%Y-%m-%d').tolist(),
            "values": data.values.tolist()
        }

        # Récupérer les métadonnées
        metadata = fred.get_series_info(request.series_id)

        return {
            "success": True,
            "series_id": request.series_id,
            "metadata": metadata,
            "data": data_dict,
            "count": len(data),
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED série {request.series_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fred/multiple-series")
async def get_fred_multiple_series(
    request: FREDMultipleSeriesRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer plusieurs séries économiques FRED

    Args:
        series_ids: Liste de codes de séries
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)

    Returns:
        Données pour toutes les séries
    """
    try:
        logger.info(f"FRED {len(request.series_ids)} séries demandées")

        fred = get_fred_collector()
        results = {}

        for series_id in request.series_ids:
            try:
                data = fred.get_series(series_id, request.start_date, request.end_date)
                results[series_id] = {
                    "success": True,
                    "dates": data.index.strftime('%Y-%m-%d').tolist(),
                    "values": data.values.tolist(),
                    "count": len(data)
                }
            except Exception as e:
                logger.warning(f"Erreur collecte série {series_id}: {e}")
                results[series_id] = {
                    "success": False,
                    "error": str(e)
                }

        return {
            "success": True,
            "total_requested": len(request.series_ids),
            "total_collected": sum(1 for r in results.values() if r['success']),
            "results": results,
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED multiple series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fred/interest-rates")
async def get_fred_interest_rates(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer tous les taux d'intérêt (Fed Funds, Treasury yields)

    Returns:
        DataFrame avec tous les taux
    """
    try:
        logger.info("FRED taux d'intérêt demandés")

        fred = get_fred_collector()

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        rates = fred.get_interest_rates(start, end)

        # Convertir en format JSON
        result = {
            "dates": rates.index.strftime('%Y-%m-%d').tolist(),
            "rates": {}
        }

        for col in rates.columns:
            result["rates"][col] = rates[col].tolist()

        return {
            "success": True,
            "data": result,
            "count": len(rates),
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED interest rates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fred/inflation")
async def get_fred_inflation(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer toutes les données d'inflation (CPI, PCE, PPI)

    Returns:
        DataFrame avec toutes les données d'inflation
    """
    try:
        logger.info("FRED données d'inflation demandées")

        fred = get_fred_collector()

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        inflation = fred.get_inflation_data(start, end)

        result = {
            "dates": inflation.index.strftime('%Y-%m-%d').tolist(),
            "indicators": {}
        }

        for col in inflation.columns:
            result["indicators"][col] = inflation[col].tolist()

        return {
            "success": True,
            "data": result,
            "count": len(inflation),
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED inflation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fred/employment")
async def get_fred_employment(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer toutes les données d'emploi (unemployment, payrolls, etc.)

    Returns:
        DataFrame avec toutes les données d'emploi
    """
    try:
        logger.info("FRED données d'emploi demandées")

        fred = get_fred_collector()

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        employment = fred.get_employment_data(start, end)

        result = {
            "dates": employment.index.strftime('%Y-%m-%d').tolist(),
            "indicators": {}
        }

        for col in employment.columns:
            result["indicators"][col] = employment[col].tolist()

        return {
            "success": True,
            "data": result,
            "count": len(employment),
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED employment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fred/gdp")
async def get_fred_gdp(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les données de PIB (GDP nominal, real, growth)

    Returns:
        DataFrame avec les données de PIB
    """
    try:
        logger.info("FRED données de PIB demandées")

        fred = get_fred_collector()

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        gdp = fred.get_gdp_data(start, end)

        result = {
            "dates": gdp.index.strftime('%Y-%m-%d').tolist(),
            "indicators": {}
        }

        for col in gdp.columns:
            result["indicators"][col] = gdp[col].tolist()

        return {
            "success": True,
            "data": result,
            "count": len(gdp),
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED GDP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fred/yield-curve")
async def get_fred_yield_curve(
    date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer la courbe des taux pour une date donnée

    Args:
        date: Date (optionnel, défaut: dernière date disponible)

    Returns:
        Courbe des taux avec toutes les maturités
    """
    try:
        logger.info("FRED yield curve demandée")

        fred = get_fred_collector()

        target_date = datetime.fromisoformat(date) if date else None
        yield_curve = fred.get_yield_curve(target_date)

        return {
            "success": True,
            "date": target_date.isoformat() if target_date else "latest",
            "yield_curve": yield_curve,
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED yield curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fred/yield-spread")
async def get_fred_yield_spread(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Calculer le spread 10Y-2Y (indicateur de récession)

    Returns:
        Série temporelle du spread
    """
    try:
        logger.info("FRED yield spread demandé")

        fred = get_fred_collector()

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        spread = fred.calculate_yield_spread(start, end)

        # Vérifier si courbe inversée
        current_spread = spread.iloc[-1]
        is_inverted = current_spread < 0

        return {
            "success": True,
            "data": {
                "dates": spread.index.strftime('%Y-%m-%d').tolist(),
                "spread": spread.values.tolist()
            },
            "current_spread": float(current_spread),
            "is_inverted": is_inverted,
            "warning": "⚠️ Courbe inversée - Risque de récession!" if is_inverted else None,
            "count": len(spread),
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED yield spread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fred/all-key-indicators")
async def get_fred_all_key_indicators(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer TOUS les indicateurs clés prédéfinis

    Returns:
        Dict avec tous les indicateurs économiques majeurs
    """
    try:
        logger.info("FRED tous les indicateurs clés demandés")

        fred = get_fred_collector()

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        indicators = fred.get_all_key_indicators(start, end)

        # Convertir en format JSON
        result = {}
        for name, series in indicators.items():
            result[name] = {
                "dates": series.index.strftime('%Y-%m-%d').tolist(),
                "values": series.values.tolist(),
                "count": len(series)
            }

        return {
            "success": True,
            "total_indicators": len(indicators),
            "indicators": result,
            "source": "FRED"
        }

    except Exception as e:
        logger.error(f"Erreur FRED all key indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FINNHUB ENDPOINTS
# ============================================================================

@router.post("/finnhub/company-news")
async def get_finnhub_company_news(
    request: FinnhubNewsRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les news d'une entreprise spécifique

    Args:
        symbol: Symbole du ticker
        from_date: Date de début (YYYY-MM-DD, défaut: 7 jours)
        to_date: Date de fin (YYYY-MM-DD, défaut: aujourd'hui)

    Returns:
        Liste des articles de news
    """
    try:
        logger.info(f"Finnhub news demandées pour {request.symbol}")

        finnhub = get_finnhub_collector()

        # Dates par défaut
        if not request.to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        else:
            to_date = request.to_date

        if not request.from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        else:
            from_date = request.from_date

        news = finnhub.get_company_news(request.symbol, from_date, to_date)

        return {
            "success": True,
            "symbol": request.symbol,
            "count": len(news),
            "news": news,
            "source": "finnhub"
        }

    except Exception as e:
        logger.error(f"Erreur Finnhub company news {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finnhub/news-sentiment")
async def get_finnhub_news_sentiment(
    request: FinnhubSentimentRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le sentiment agrégé des news pour une entreprise

    Args:
        symbol: Symbole du ticker

    Returns:
        Sentiment score et statistiques
    """
    try:
        logger.info(f"Finnhub sentiment demandé pour {request.symbol}")

        finnhub = get_finnhub_collector()
        sentiment = finnhub.get_news_sentiment(request.symbol)

        return {
            "success": True,
            "symbol": request.symbol,
            "sentiment": sentiment,
            "source": "finnhub"
        }

    except Exception as e:
        logger.error(f"Erreur Finnhub news sentiment {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finnhub/social-sentiment")
async def get_finnhub_social_sentiment(
    request: FinnhubSentimentRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le sentiment des réseaux sociaux (Reddit, Twitter)

    Args:
        symbol: Symbole du ticker

    Returns:
        Sentiment des réseaux sociaux
    """
    try:
        logger.info(f"Finnhub social sentiment demandé pour {request.symbol}")

        finnhub = get_finnhub_collector()

        # Derniers 30 jours
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        social = finnhub.get_social_sentiment(request.symbol, from_date)

        return {
            "success": True,
            "symbol": request.symbol,
            "social_sentiment": social,
            "source": "finnhub"
        }

    except Exception as e:
        logger.error(f"Erreur Finnhub social sentiment {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finnhub/recommendations")
async def get_finnhub_recommendations(
    request: FinnhubRecommendationsRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les recommandations d'analystes

    Args:
        symbol: Symbole du ticker

    Returns:
        Tendances des recommandations (strong buy, buy, hold, sell, strong sell)
    """
    try:
        logger.info(f"Finnhub recommendations demandées pour {request.symbol}")

        finnhub = get_finnhub_collector()
        recommendations = finnhub.get_recommendation_trends(request.symbol)

        return {
            "success": True,
            "symbol": request.symbol,
            "recommendations": recommendations,
            "source": "finnhub"
        }

    except Exception as e:
        logger.error(f"Erreur Finnhub recommendations {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finnhub/price-target")
async def get_finnhub_price_target(
    request: FinnhubPriceTargetRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les objectifs de prix des analystes

    Args:
        symbol: Symbole du ticker

    Returns:
        Prix cibles (high, low, mean, median)
    """
    try:
        logger.info(f"Finnhub price target demandé pour {request.symbol}")

        finnhub = get_finnhub_collector()
        price_target = finnhub.get_price_target(request.symbol)

        return {
            "success": True,
            "symbol": request.symbol,
            "price_target": price_target,
            "source": "finnhub"
        }

    except Exception as e:
        logger.error(f"Erreur Finnhub price target {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finnhub/earnings-calendar")
async def get_finnhub_earnings_calendar(
    request: FinnhubEarningsRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le calendrier des publications de résultats

    Args:
        symbol: Symbole (optionnel, si omis: tous les résultats)
        from_date: Date de début (YYYY-MM-DD)
        to_date: Date de fin (YYYY-MM-DD)

    Returns:
        Calendrier des earnings
    """
    try:
        logger.info(f"Finnhub earnings calendar demandé")

        finnhub = get_finnhub_collector()

        # Dates par défaut: prochains 30 jours
        if not request.from_date:
            from_date = datetime.now().strftime('%Y-%m-%d')
        else:
            from_date = request.from_date

        if not request.to_date:
            to_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        else:
            to_date = request.to_date

        earnings = finnhub.get_earnings_calendar(
            from_date=from_date,
            to_date=to_date,
            symbol=request.symbol
        )

        return {
            "success": True,
            "symbol": request.symbol if request.symbol else "all",
            "count": len(earnings) if earnings else 0,
            "earnings": earnings,
            "source": "finnhub"
        }

    except Exception as e:
        logger.error(f"Erreur Finnhub earnings calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/finnhub/market-sentiment")
async def get_finnhub_market_sentiment(
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le sentiment global du marché

    Returns:
        Sentiment du marché (VIX, fear & greed, etc.)
    """
    try:
        logger.info("Finnhub market sentiment demandé")

        finnhub = get_finnhub_collector()
        market_sentiment = finnhub.get_market_sentiment()

        return {
            "success": True,
            "market_sentiment": market_sentiment,
            "source": "finnhub"
        }

    except Exception as e:
        logger.error(f"Erreur Finnhub market sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FINANCIAL MODELING PREP (FMP) ENDPOINTS
# ============================================================================

@router.post("/fmp/income-statement")
async def get_fmp_income_statement(
    request: FMPSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le compte de résultat (Income Statement)

    Args:
        symbol: Symbole du ticker
        period: "annual" ou "quarter"
        limit: Nombre de périodes (max 10)

    Returns:
        Liste de comptes de résultat
    """
    try:
        logger.info(f"FMP Income Statement demandé pour {request.symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_income_statement(request.symbol, request.period, request.limit)

        return {
            "success": True,
            "symbol": request.symbol,
            "period": request.period,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP income statement {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/balance-sheet")
async def get_fmp_balance_sheet(
    request: FMPSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le bilan (Balance Sheet)

    Args:
        symbol: Symbole du ticker
        period: "annual" ou "quarter"
        limit: Nombre de périodes

    Returns:
        Liste de bilans
    """
    try:
        logger.info(f"FMP Balance Sheet demandé pour {request.symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_balance_sheet(request.symbol, request.period, request.limit)

        return {
            "success": True,
            "symbol": request.symbol,
            "period": request.period,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP balance sheet {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/cash-flow")
async def get_fmp_cash_flow(
    request: FMPSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le tableau de flux de trésorerie (Cash Flow Statement)

    Args:
        symbol: Symbole du ticker
        period: "annual" ou "quarter"
        limit: Nombre de périodes

    Returns:
        Liste de cash flow statements
    """
    try:
        logger.info(f"FMP Cash Flow demandé pour {request.symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_cash_flow(request.symbol, request.period, request.limit)

        return {
            "success": True,
            "symbol": request.symbol,
            "period": request.period,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP cash flow {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/financial-ratios")
async def get_fmp_financial_ratios(
    request: FMPSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer tous les ratios financiers (50+ ratios)

    Ratios inclus:
    - Profitabilité: ROE, ROA, marges
    - Liquidité: Current ratio, Quick ratio
    - Solvabilité: Debt/Equity, Interest coverage
    - Efficacité: Asset turnover, Inventory turnover
    - Valorisation: P/E, P/B, P/S, EV/EBITDA

    Args:
        symbol: Symbole du ticker
        period: "annual" ou "quarter"
        limit: Nombre de périodes

    Returns:
        Liste de ratios financiers
    """
    try:
        logger.info(f"FMP Financial Ratios demandé pour {request.symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_financial_ratios(request.symbol, request.period, request.limit)

        return {
            "success": True,
            "symbol": request.symbol,
            "period": request.period,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP financial ratios {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/key-metrics")
async def get_fmp_key_metrics(
    request: FMPSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les métriques clés (market cap, P/E, EPS, etc.)

    Args:
        symbol: Symbole du ticker
        period: "annual" ou "quarter"
        limit: Nombre de périodes

    Returns:
        Liste de key metrics
    """
    try:
        logger.info(f"FMP Key Metrics demandé pour {request.symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_key_metrics(request.symbol, request.period, request.limit)

        return {
            "success": True,
            "symbol": request.symbol,
            "period": request.period,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP key metrics {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/financial-growth")
async def get_fmp_financial_growth(
    request: FMPSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les taux de croissance (revenue growth, EPS growth, etc.)

    Args:
        symbol: Symbole du ticker
        period: "annual" ou "quarter"
        limit: Nombre de périodes

    Returns:
        Liste de financial growth metrics
    """
    try:
        logger.info(f"FMP Financial Growth demandé pour {request.symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_financial_growth(request.symbol, request.period, request.limit)

        return {
            "success": True,
            "symbol": request.symbol,
            "period": request.period,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP financial growth {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/company-profile")
async def get_fmp_company_profile(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le profil complet de l'entreprise

    Args:
        symbol: Symbole du ticker

    Returns:
        Profil de l'entreprise
    """
    try:
        logger.info(f"FMP Company Profile demandé pour {symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_company_profile(symbol)

        return {
            "success": True,
            "symbol": symbol,
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP company profile {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/dividends-historical")
async def get_fmp_dividends_historical(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer l'historique complet des dividendes

    Args:
        symbol: Symbole du ticker

    Returns:
        Liste des dividendes historiques
    """
    try:
        logger.info(f"FMP Dividends Historical demandé pour {symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_dividends_historical(symbol)

        return {
            "success": True,
            "symbol": symbol,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP dividends historical {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/insider-trading")
async def get_fmp_insider_trading(
    request: FMPInsiderTradingRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les transactions d'initiés (insider trading)

    Args:
        symbol: Symbole du ticker
        limit: Nombre de transactions

    Returns:
        Liste des insider trades
    """
    try:
        logger.info(f"FMP Insider Trading demandé pour {request.symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_insider_trading(request.symbol, request.limit)

        return {
            "success": True,
            "symbol": request.symbol,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP insider trading {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/institutional-holders")
async def get_fmp_institutional_holders(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les détenteurs institutionnels

    Args:
        symbol: Symbole du ticker

    Returns:
        Liste des institutional holders
    """
    try:
        logger.info(f"FMP Institutional Holders demandé pour {symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_institutional_holders(symbol)

        return {
            "success": True,
            "symbol": symbol,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP institutional holders {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fmp/analyst-estimates")
async def get_fmp_analyst_estimates(
    request: FMPSymbolRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer les estimations des analystes

    Args:
        symbol: Symbole du ticker
        period: "annual" ou "quarter"
        limit: Nombre de périodes

    Returns:
        Estimations analystes
    """
    try:
        logger.info(f"FMP Analyst Estimates demandé pour {request.symbol}")

        fmp = get_fmp_collector()
        data = fmp.get_analyst_estimates(request.symbol, request.period, request.limit)

        return {
            "success": True,
            "symbol": request.symbol,
            "period": request.period,
            "count": len(data),
            "data": data,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP analyst estimates {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fmp/usage")
async def get_fmp_usage(
    current_user: User = Depends(get_current_user)
):
    """
    Obtenir les statistiques d'utilisation FMP

    Returns:
        Nombre de requêtes utilisées aujourd'hui
    """
    try:
        fmp = get_fmp_collector()
        stats = fmp.get_usage_stats()

        return {
            "success": True,
            "usage": stats,
            "source": "FMP"
        }

    except Exception as e:
        logger.error(f"Erreur FMP usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TWELVE DATA ENDPOINTS
# ============================================================================

@router.post("/twelvedata/quote")
async def get_twelvedata_quote(
    request: TwelveDataRequest,
    current_user: User = Depends(get_current_user)
):
    """Récupérer quote temps réel via Twelve Data"""
    try:
        td = get_twelvedata_collector()
        data = td.get_quote(request.symbol, request.interval)
        return {"success": True, "data": data, "source": "TwelveData"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/twelvedata/time-series")
async def get_twelvedata_time_series(
    request: TwelveDataRequest,
    current_user: User = Depends(get_current_user)
):
    """Récupérer time series OHLCV"""
    try:
        td = get_twelvedata_collector()
        data = td.get_time_series(request.symbol, request.interval, request.outputsize)
        return {"success": True, "data": data, "source": "TwelveData"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twelvedata/usage")
async def get_twelvedata_usage(current_user: User = Depends(get_current_user)):
    """Stats d'utilisation Twelve Data"""
    try:
        td = get_twelvedata_collector()
        return {"success": True, "usage": td.get_usage_stats(), "source": "TwelveData"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WORLD BANK ENDPOINTS (GRATUIT - ILLIMITÉ)
# ============================================================================

@router.get("/worldbank/gdp/{country}")
async def get_worldbank_gdp(
    country: str,
    start_year: int = 2000,
    current_user: User = Depends(get_current_user)
):
    """Récupérer le PIB d'un pays"""
    try:
        wb = get_worldbank_collector()
        data = wb.get_gdp(country, start_year)
        return {"success": True, "country": country, "data": data, "source": "WorldBank"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/worldbank/dashboard/{country}")
async def get_worldbank_dashboard(
    country: str,
    start_year: int = 2010,
    current_user: User = Depends(get_current_user)
):
    """Dashboard économique complet pour un pays"""
    try:
        wb = get_worldbank_collector()
        data = wb.get_economic_dashboard(country, start_year)
        return {"success": True, "country": country, "data": data, "source": "WorldBank"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/worldbank/countries")
async def get_worldbank_countries(current_user: User = Depends(get_current_user)):
    """Liste de tous les pays"""
    try:
        wb = get_worldbank_collector()
        data = wb.get_countries()
        return {"success": True, "count": len(data), "data": data, "source": "WorldBank"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
