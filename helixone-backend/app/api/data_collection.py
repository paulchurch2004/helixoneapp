"""
API endpoints pour la collecte de données de marché
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, DataCollectionJob, SymbolMetadata, TimeframeType, CollectionStatus
from app.services.data_collector import (
    get_data_collector,
    collect_all_crises,
    CRISIS_PERIODS
)

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================
# SCHEMAS
# ============================================

class CollectDailyRequest(BaseModel):
    """Requête pour collecter des données journalières"""
    symbols: List[str] = Field(..., description="Liste de tickers", example=["AAPL", "MSFT"])
    start_date: datetime = Field(..., description="Date de début")
    end_date: datetime = Field(..., description="Date de fin")
    adjusted: bool = Field(True, description="Prix ajustés pour splits/dividendes")


class CollectIntradayRequest(BaseModel):
    """Requête pour collecter des données intraday"""
    symbols: List[str] = Field(..., description="Liste de tickers")
    interval: str = Field("1m", description="Intervalle: 1m, 5m, 15m, 30m, 1h")
    period: str = Field("7d", description="Période: 1d, 5d, 7d, 60d")


class CollectCrisisRequest(BaseModel):
    """Requête pour collecter les données d'une crise"""
    crisis_id: str = Field(..., description="ID de la crise", example="2008_crisis")
    additional_symbols: Optional[List[str]] = Field(None, description="Symbols supplémentaires")


class CollectionJobResponse(BaseModel):
    """Réponse avec info sur un job de collecte"""
    id: str
    job_name: str
    status: str
    progress: float
    records_collected: int
    records_failed: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class SymbolMetadataResponse(BaseModel):
    """Métadonnées d'un symbol"""
    symbol: str
    name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    country: Optional[str]
    last_data_update: Optional[datetime]

    class Config:
        from_attributes = True


class DataCoverageResponse(BaseModel):
    """Couverture des données pour un symbol"""
    symbol: str
    timeframes: Dict[str, Dict]
    total_records: int


class CrisisInfoResponse(BaseModel):
    """Information sur une crise historique"""
    id: str
    name: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    default_symbols: List[str]


# ============================================
# ENDPOINTS - COLLECTE
# ============================================

@router.post("/collect/daily", response_model=CollectionJobResponse)
async def collect_daily_data(
    request: CollectDailyRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Collecte des données journalières pour une liste de symboles

    La collecte se fait en arrière-plan et retourne immédiatement
    """
    try:
        # Créer un job
        job = DataCollectionJob(
            job_name=f"Daily Data: {', '.join(request.symbols[:3])}{'...' if len(request.symbols) > 3 else ''}",
            job_type="daily_historical",
            symbols=",".join(request.symbols),
            timeframe=TimeframeType.DAILY,
            start_date=request.start_date,
            end_date=request.end_date,
            source="yahoo",
            status=CollectionStatus.PENDING,
            user_id=current_user.id
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Lancer la collecte en arrière-plan
        def collect_task():
            collector = get_data_collector(db)
            import asyncio
            asyncio.run(collector.collect_multiple_symbols_daily(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                job_id=job.id
            ))

        background_tasks.add_task(collect_task)

        logger.info(f"📥 Job de collecte créé: {job.id}")

        return CollectionJobResponse.from_orm(job)

    except Exception as e:
        logger.error(f"Erreur création job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect/intraday", response_model=CollectionJobResponse)
async def collect_intraday_data(
    request: CollectIntradayRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Collecte des données intraday (minutes)

    Limité aux 60 derniers jours par Yahoo Finance
    """
    try:
        # Mapping interval
        interval_map = {
            "1m": TimeframeType.MINUTE_1,
            "5m": TimeframeType.MINUTE_5,
            "15m": TimeframeType.MINUTE_15,
            "30m": TimeframeType.MINUTE_30,
            "1h": TimeframeType.HOUR_1,
        }

        timeframe = interval_map.get(request.interval, TimeframeType.MINUTE_1)

        # Créer job
        job = DataCollectionJob(
            job_name=f"Intraday {request.interval}: {', '.join(request.symbols[:3])}",
            job_type="intraday",
            symbols=",".join(request.symbols),
            timeframe=timeframe,
            source="yahoo",
            status=CollectionStatus.PENDING,
            user_id=current_user.id
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Lancer en arrière-plan
        def collect_task():
            job.status = CollectionStatus.IN_PROGRESS
            job.started_at = datetime.utcnow()
            db.commit()

            collector = get_data_collector(db)
            total_collected = 0
            total_failed = 0

            for symbol in request.symbols:
                collected, failed = collector.collect_intraday_data(
                    symbol=symbol,
                    interval=request.interval,
                    period=request.period
                )
                total_collected += collected
                total_failed += failed

            job.status = CollectionStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.records_collected = total_collected
            job.records_failed = total_failed
            db.commit()

        background_tasks.add_task(collect_task)

        return CollectionJobResponse.from_orm(job)

    except Exception as e:
        logger.error(f"Erreur collecte intraday: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect/crisis/{crisis_id}", response_model=CollectionJobResponse)
async def collect_crisis_data(
    crisis_id: str,
    background_tasks: BackgroundTasks,
    additional_symbols: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Collecte les données pour une crise historique prédéfinie

    Crises disponibles: 2008_crisis, covid_2020, dotcom_2000, black_monday_1987
    """
    if crisis_id not in CRISIS_PERIODS:
        raise HTTPException(
            status_code=404,
            detail=f"Crise '{crisis_id}' non trouvée. Disponibles: {list(CRISIS_PERIODS.keys())}"
        )

    try:
        crisis_info = CRISIS_PERIODS[crisis_id]
        symbols = crisis_info['symbols'].copy()

        if additional_symbols:
            symbols.extend(additional_symbols)

        # Créer job
        job = DataCollectionJob(
            job_name=f"Crisis: {crisis_info['name']}",
            job_type="historical_crisis",
            symbols=",".join(symbols),
            timeframe=TimeframeType.DAILY,
            start_date=crisis_info['start'],
            end_date=crisis_info['end'],
            source="yahoo",
            status=CollectionStatus.PENDING,
            user_id=current_user.id
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Lancer collecte
        def collect_task():
            collector = get_data_collector(db)
            results = collector.collect_crisis_data(
                crisis_name=crisis_info['name'],
                symbols=symbols,
                start_date=crisis_info['start'],
                end_date=crisis_info['end']
            )

        background_tasks.add_task(collect_task)

        logger.info(f"🔥 Collecte crise {crisis_id}: {len(symbols)} symbols")

        return CollectionJobResponse.from_orm(job)

    except Exception as e:
        logger.error(f"Erreur collecte crise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect/all-crises", response_model=Dict[str, str])
async def collect_all_crises_endpoint(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Collecte les données pour TOUTES les crises historiques

    Attention: peut prendre plusieurs minutes
    """
    try:
        def collect_task():
            logger.info("🔥 Démarrage collecte TOUTES les crises")
            collect_all_crises(db)
            logger.info("✅ Collecte toutes crises terminée")

        background_tasks.add_task(collect_task)

        return {
            "status": "started",
            "message": f"Collecte démarrée pour {len(CRISIS_PERIODS)} crises"
        }

    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINTS - MÉTADONNÉES
# ============================================

@router.post("/metadata/{symbol}", response_model=SymbolMetadataResponse)
async def collect_symbol_metadata(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Collecte les métadonnées d'un symbol"""
    try:
        collector = get_data_collector(db)
        metadata = collector.collect_symbol_metadata(symbol.upper())

        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Métadonnées non disponibles pour {symbol}"
            )

        return SymbolMetadataResponse.from_orm(metadata)

    except Exception as e:
        logger.error(f"Erreur métadonnées: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata", response_model=List[SymbolMetadataResponse])
async def list_symbols_metadata(
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste les métadonnées de tous les symbols disponibles"""
    try:
        symbols = db.query(SymbolMetadata).filter(
            SymbolMetadata.is_active == True
        ).limit(limit).all()

        return [SymbolMetadataResponse.from_orm(s) for s in symbols]

    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINTS - JOBS & STATUT
# ============================================

@router.get("/jobs", response_model=List[CollectionJobResponse])
async def list_collection_jobs(
    limit: int = 50,
    status: Optional[CollectionStatus] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste les jobs de collecte"""
    try:
        query = db.query(DataCollectionJob).filter(
            DataCollectionJob.user_id == current_user.id
        )

        if status:
            query = query.filter(DataCollectionJob.status == status)

        jobs = query.order_by(
            DataCollectionJob.created_at.desc()
        ).limit(limit).all()

        return [CollectionJobResponse.from_orm(j) for j in jobs]

    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=CollectionJobResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Récupère le statut d'un job"""
    job = db.query(DataCollectionJob).filter(
        DataCollectionJob.id == job_id,
        DataCollectionJob.user_id == current_user.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job non trouvé")

    return CollectionJobResponse.from_orm(job)


@router.get("/coverage/{symbol}", response_model=DataCoverageResponse)
async def get_data_coverage(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Vérifie la couverture des données pour un symbol"""
    try:
        collector = get_data_collector(db)
        coverage = collector.get_data_coverage(symbol.upper())

        total = sum(tf['count'] for tf in coverage.values())

        return DataCoverageResponse(
            symbol=symbol.upper(),
            timeframes=coverage,
            total_records=total
        )

    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINTS - CRISES INFO
# ============================================

@router.get("/crises", response_model=List[CrisisInfoResponse])
async def list_crises():
    """Liste toutes les crises historiques disponibles"""
    crises = []

    for crisis_id, info in CRISIS_PERIODS.items():
        duration = (info['end'] - info['start']).days

        crises.append(CrisisInfoResponse(
            id=crisis_id,
            name=info['name'],
            start_date=info['start'],
            end_date=info['end'],
            duration_days=duration,
            default_symbols=info['symbols']
        ))

    return crises


@router.get("/crises/{crisis_id}", response_model=CrisisInfoResponse)
async def get_crisis_info(crisis_id: str):
    """Récupère les infos d'une crise spécifique"""
    if crisis_id not in CRISIS_PERIODS:
        raise HTTPException(status_code=404, detail="Crise non trouvée")

    info = CRISIS_PERIODS[crisis_id]
    duration = (info['end'] - info['start']).days

    return CrisisInfoResponse(
        id=crisis_id,
        name=info['name'],
        start_date=info['start'],
        end_date=info['end'],
        duration_days=duration,
        default_symbols=info['symbols']
    )
