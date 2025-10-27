"""
Endpoints API pour l'analyse de portefeuille automatique
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User
from app.models.portfolio import (
    PortfolioAnalysisHistory,
    PortfolioAlert,
    PortfolioRecommendation,
    RecommendationPerformance,
    AlertSeverity,
    AlertStatus,
    RecommendationType,
    AnalysisTimeType
)
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================
# SCHÉMAS PYDANTIC
# ============================================

class AlertResponse(BaseModel):
    """Réponse d'alerte"""
    id: str
    analysis_id: str
    severity: AlertSeverity
    status: AlertStatus
    ticker: Optional[str]
    title: str
    message: str
    action_required: Optional[str]
    confidence: Optional[float]
    created_at: datetime
    read_at: Optional[datetime]

    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    """Réponse de recommandation"""
    id: str
    analysis_id: str
    ticker: str
    action: RecommendationType
    confidence: float
    current_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    primary_reason: str
    detailed_reasons: Optional[List[str]]
    risk_factors: Optional[List[str]]
    suggested_action: Optional[str]
    prediction_1d: Optional[float]
    prediction_3d: Optional[float]
    prediction_7d: Optional[float]
    sentiment_score: Optional[float]
    created_at: datetime
    expires_at: Optional[datetime]

    class Config:
        from_attributes = True


class AnalysisHistoryResponse(BaseModel):
    """Réponse d'historique d'analyse"""
    id: str
    analysis_time: AnalysisTimeType
    num_positions: int
    health_score: float
    portfolio_sentiment: Optional[str]
    expected_return_7d: Optional[float]
    downside_risk_pct: Optional[float]
    num_alerts: int
    num_critical_alerts: int
    num_recommendations: int
    execution_time_seconds: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class AlertUpdateRequest(BaseModel):
    """Requête pour mettre à jour le statut d'une alerte"""
    status: AlertStatus


# ============================================
# ENDPOINTS ALERTES
# ============================================

@router.get("/alerts", response_model=List[AlertResponse], tags=["Portfolio"])
async def get_alerts(
    severity: Optional[AlertSeverity] = Query(None, description="Filtrer par sévérité"),
    status: Optional[AlertStatus] = Query(None, description="Filtrer par statut"),
    ticker: Optional[str] = Query(None, description="Filtrer par ticker"),
    days: int = Query(7, description="Nombre de jours à récupérer"),
    limit: int = Query(50, description="Nombre max d'alertes"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère les alertes de portefeuille pour l'utilisateur

    Permet de filtrer par :
    - Sévérité (critical, warning, opportunity, info)
    - Statut (new, read, acted, dismissed, archived)
    - Ticker spécifique
    - Période (nombre de jours)

    Returns:
        Liste des alertes triées par date (plus récentes en premier)
    """
    try:
        logger.info(f"Récupération alertes pour {current_user.email}")

        # Construire la requête de base
        query = db.query(PortfolioAlert).filter(
            PortfolioAlert.user_id == current_user.id
        )

        # Filtres optionnels
        if severity:
            query = query.filter(PortfolioAlert.severity == severity)

        if status:
            query = query.filter(PortfolioAlert.status == status)

        if ticker:
            query = query.filter(PortfolioAlert.ticker == ticker)

        # Filtre par date
        date_threshold = datetime.utcnow() - timedelta(days=days)
        query = query.filter(PortfolioAlert.created_at >= date_threshold)

        # Trier par date (plus récentes en premier) et limiter
        alerts = query.order_by(desc(PortfolioAlert.created_at)).limit(limit).all()

        logger.info(f"✅ {len(alerts)} alertes récupérées")
        return alerts

    except Exception as e:
        logger.error(f"❌ Erreur récupération alertes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des alertes: {str(e)}"
        )


@router.get("/alerts/{alert_id}", response_model=AlertResponse, tags=["Portfolio"])
async def get_alert(
    alert_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère une alerte spécifique par son ID

    Marque automatiquement l'alerte comme "read" si elle était "new"
    """
    try:
        # Récupérer l'alerte
        alert = db.query(PortfolioAlert).filter(
            PortfolioAlert.id == alert_id,
            PortfolioAlert.user_id == current_user.id
        ).first()

        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alerte non trouvée"
            )

        # Marquer comme lue si nouvelle
        if alert.status == AlertStatus.NEW and not alert.read_at:
            alert.status = AlertStatus.READ
            alert.read_at = datetime.utcnow()
            db.commit()
            db.refresh(alert)
            logger.info(f"Alerte {alert_id} marquée comme lue")

        return alert

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération alerte {alert_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération de l'alerte: {str(e)}"
        )


@router.patch("/alerts/{alert_id}", response_model=AlertResponse, tags=["Portfolio"])
async def update_alert_status(
    alert_id: str,
    update: AlertUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Met à jour le statut d'une alerte

    Permet de marquer une alerte comme :
    - read : Lue
    - acted : Action effectuée
    - dismissed : Ignorée
    - archived : Archivée
    """
    try:
        # Récupérer l'alerte
        alert = db.query(PortfolioAlert).filter(
            PortfolioAlert.id == alert_id,
            PortfolioAlert.user_id == current_user.id
        ).first()

        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alerte non trouvée"
            )

        # Mettre à jour le statut
        old_status = alert.status
        alert.status = update.status

        # Mettre à jour les timestamps appropriés
        if update.status == AlertStatus.READ and not alert.read_at:
            alert.read_at = datetime.utcnow()
        elif update.status == AlertStatus.ACTED and not alert.acted_at:
            alert.acted_at = datetime.utcnow()

        db.commit()
        db.refresh(alert)

        logger.info(f"Alerte {alert_id} : {old_status} → {update.status}")
        return alert

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour alerte {alert_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la mise à jour de l'alerte: {str(e)}"
        )


@router.get("/alerts/unread/count", tags=["Portfolio"])
async def get_unread_alerts_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère le nombre d'alertes non lues

    Utile pour afficher un badge de notification dans l'UI
    """
    try:
        count = db.query(PortfolioAlert).filter(
            PortfolioAlert.user_id == current_user.id,
            PortfolioAlert.status == AlertStatus.NEW
        ).count()

        # Compter aussi les critiques
        critical_count = db.query(PortfolioAlert).filter(
            PortfolioAlert.user_id == current_user.id,
            PortfolioAlert.status == AlertStatus.NEW,
            PortfolioAlert.severity == AlertSeverity.CRITICAL
        ).count()

        return {
            "unread_count": count,
            "critical_count": critical_count
        }

    except Exception as e:
        logger.error(f"❌ Erreur comptage alertes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du comptage des alertes: {str(e)}"
        )


# ============================================
# ENDPOINTS RECOMMANDATIONS
# ============================================

@router.get("/recommendations", response_model=List[RecommendationResponse], tags=["Portfolio"])
async def get_recommendations(
    action: Optional[RecommendationType] = Query(None, description="Filtrer par type d'action"),
    ticker: Optional[str] = Query(None, description="Filtrer par ticker"),
    min_confidence: float = Query(0, description="Confiance minimale (0-100)"),
    days: int = Query(7, description="Nombre de jours à récupérer"),
    limit: int = Query(50, description="Nombre max de recommandations"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère les recommandations de portefeuille pour l'utilisateur

    Permet de filtrer par :
    - Type d'action (strong_buy, buy, hold, sell, strong_sell)
    - Ticker spécifique
    - Confiance minimale
    - Période (nombre de jours)

    Returns:
        Liste des recommandations triées par confiance décroissante
    """
    try:
        logger.info(f"Récupération recommandations pour {current_user.email}")

        # Construire la requête de base
        query = db.query(PortfolioRecommendation).filter(
            PortfolioRecommendation.user_id == current_user.id
        )

        # Filtres optionnels
        if action:
            query = query.filter(PortfolioRecommendation.action == action)

        if ticker:
            query = query.filter(PortfolioRecommendation.ticker == ticker)

        if min_confidence > 0:
            query = query.filter(PortfolioRecommendation.confidence >= min_confidence)

        # Filtre par date
        date_threshold = datetime.utcnow() - timedelta(days=days)
        query = query.filter(PortfolioRecommendation.created_at >= date_threshold)

        # Trier par confiance décroissante et limiter
        recommendations = query.order_by(
            desc(PortfolioRecommendation.confidence)
        ).limit(limit).all()

        logger.info(f"✅ {len(recommendations)} recommandations récupérées")
        return recommendations

    except Exception as e:
        logger.error(f"❌ Erreur récupération recommandations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des recommandations: {str(e)}"
        )


@router.get("/recommendations/{recommendation_id}", response_model=RecommendationResponse, tags=["Portfolio"])
async def get_recommendation(
    recommendation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère une recommandation spécifique par son ID
    """
    try:
        recommendation = db.query(PortfolioRecommendation).filter(
            PortfolioRecommendation.id == recommendation_id,
            PortfolioRecommendation.user_id == current_user.id
        ).first()

        if not recommendation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Recommandation non trouvée"
            )

        return recommendation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération recommandation {recommendation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération de la recommandation: {str(e)}"
        )


# ============================================
# ENDPOINTS HISTORIQUE
# ============================================

@router.get("/history", response_model=List[AnalysisHistoryResponse], tags=["Portfolio"])
async def get_analysis_history(
    analysis_time: Optional[AnalysisTimeType] = Query(None, description="Filtrer par timing"),
    days: int = Query(30, description="Nombre de jours à récupérer"),
    limit: int = Query(50, description="Nombre max d'analyses"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère l'historique des analyses de portefeuille

    Permet de filtrer par :
    - Timing (morning, evening, manual, on_demand)
    - Période (nombre de jours)

    Returns:
        Liste des analyses triées par date (plus récentes en premier)
    """
    try:
        logger.info(f"Récupération historique pour {current_user.email}")

        # Construire la requête
        query = db.query(PortfolioAnalysisHistory).filter(
            PortfolioAnalysisHistory.user_id == current_user.id
        )

        # Filtres optionnels
        if analysis_time:
            query = query.filter(PortfolioAnalysisHistory.analysis_time == analysis_time)

        # Filtre par date
        date_threshold = datetime.utcnow() - timedelta(days=days)
        query = query.filter(PortfolioAnalysisHistory.created_at >= date_threshold)

        # Trier par date et limiter
        history = query.order_by(desc(PortfolioAnalysisHistory.created_at)).limit(limit).all()

        logger.info(f"✅ {len(history)} analyses récupérées")
        return history

    except Exception as e:
        logger.error(f"❌ Erreur récupération historique: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération de l'historique: {str(e)}"
        )


@router.get("/history/{analysis_id}", tags=["Portfolio"])
async def get_analysis_detail(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère les détails complets d'une analyse spécifique

    Inclut toutes les données JSON (positions, corrélations, risques, prédictions)
    """
    try:
        analysis = db.query(PortfolioAnalysisHistory).filter(
            PortfolioAnalysisHistory.id == analysis_id,
            PortfolioAnalysisHistory.user_id == current_user.id
        ).first()

        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analyse non trouvée"
            )

        # Retourner toutes les données incluant les JSON
        return {
            "id": analysis.id,
            "analysis_time": analysis.analysis_time,
            "num_positions": analysis.num_positions,
            "total_value": analysis.total_value,
            "cash_amount": analysis.cash_amount,
            "health_score": analysis.health_score,
            "portfolio_sentiment": analysis.portfolio_sentiment,
            "expected_return_7d": analysis.expected_return_7d,
            "downside_risk_pct": analysis.downside_risk_pct,
            "num_alerts": analysis.num_alerts,
            "num_critical_alerts": analysis.num_critical_alerts,
            "num_recommendations": analysis.num_recommendations,
            "execution_time_seconds": analysis.execution_time_seconds,
            "data_sources_used": analysis.data_sources_used,
            "positions_data": analysis.positions_data,
            "correlations_data": analysis.correlations_data,
            "risks_data": analysis.risks_data,
            "predictions_data": analysis.predictions_data,
            "created_at": analysis.created_at
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération détails analyse {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des détails: {str(e)}"
        )


@router.get("/dashboard/summary", tags=["Portfolio"])
async def get_dashboard_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère un résumé pour le dashboard de portefeuille

    Inclut:
    - Dernière analyse
    - Nombre d'alertes non lues
    - Nombre de recommandations actives
    - Health score actuel
    """
    try:
        # Dernière analyse
        last_analysis = db.query(PortfolioAnalysisHistory).filter(
            PortfolioAnalysisHistory.user_id == current_user.id
        ).order_by(desc(PortfolioAnalysisHistory.created_at)).first()

        # Alertes non lues
        unread_alerts = db.query(PortfolioAlert).filter(
            PortfolioAlert.user_id == current_user.id,
            PortfolioAlert.status == AlertStatus.NEW
        ).count()

        # Alertes critiques
        critical_alerts = db.query(PortfolioAlert).filter(
            PortfolioAlert.user_id == current_user.id,
            PortfolioAlert.status == AlertStatus.NEW,
            PortfolioAlert.severity == AlertSeverity.CRITICAL
        ).count()

        # Recommandations récentes (7 derniers jours)
        date_threshold = datetime.utcnow() - timedelta(days=7)
        active_recommendations = db.query(PortfolioRecommendation).filter(
            PortfolioRecommendation.user_id == current_user.id,
            PortfolioRecommendation.created_at >= date_threshold
        ).count()

        return {
            "last_analysis": {
                "id": last_analysis.id if last_analysis else None,
                "health_score": last_analysis.health_score if last_analysis else None,
                "sentiment": last_analysis.portfolio_sentiment if last_analysis else None,
                "expected_return_7d": last_analysis.expected_return_7d if last_analysis else None,
                "num_positions": last_analysis.num_positions if last_analysis else 0,
                "created_at": last_analysis.created_at if last_analysis else None
            },
            "alerts": {
                "unread_count": unread_alerts,
                "critical_count": critical_alerts
            },
            "recommendations": {
                "active_count": active_recommendations
            }
        }

    except Exception as e:
        logger.error(f"❌ Erreur récupération résumé dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération du résumé: {str(e)}"
        )
