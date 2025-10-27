"""
API endpoints pour l'intégration Interactive Brokers
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User
from app.models.ibkr import IBKRConnection, PortfolioSnapshot, IBKRAlert
from app.services.ibkr_service import get_ibkr_service

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================
# SCHEMAS
# ============================================

class IBKRConnectionConfig(BaseModel):
    """Configuration de connexion IBKR"""
    account_id: str = Field(..., description="ID du compte IBKR", example="U17421384")
    connection_type: str = Field("live", description="Type: live ou paper")
    host: str = Field("127.0.0.1", description="Host TWS/Gateway")
    port: int = Field(7496, description="Port: 7496=live, 7497=paper")
    client_id: int = Field(1, description="Client ID unique")
    auto_connect: bool = Field(True, description="Connexion automatique au lancement")


class IBKRConnectionResponse(BaseModel):
    """Réponse avec info de connexion"""
    id: str
    account_id: str
    connection_type: str
    is_connected: bool
    auto_connect: bool
    last_connected_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class PortfolioResponse(BaseModel):
    """Réponse avec le portefeuille"""
    account_id: str
    net_liquidation: float
    total_cash: float
    stock_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    buying_power: float
    available_funds: float
    currency: str
    positions: List[Dict]
    timestamp: str


class AlertResponse(BaseModel):
    """Réponse avec une alerte"""
    id: str
    alert_type: str
    severity: str
    symbol: Optional[str]
    title: str
    message: str
    recommendations: Optional[List[Dict]]
    is_active: bool
    is_acknowledged: bool
    triggered_at: datetime

    class Config:
        from_attributes = True


# ============================================
# ENDPOINTS - CONNEXION
# ============================================

@router.post("/connect", response_model=IBKRConnectionResponse)
async def connect_ibkr(
    config: IBKRConnectionConfig,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Configurer et connecter à Interactive Brokers

    Crée ou met à jour la configuration de connexion IBKR
    """
    try:
        # Vérifier si une config existe déjà
        existing = db.query(IBKRConnection).filter(
            IBKRConnection.user_id == current_user.id,
            IBKRConnection.account_id == config.account_id
        ).first()

        if existing:
            # Mettre à jour
            existing.connection_type = config.connection_type
            existing.host = config.host
            existing.port = config.port
            existing.client_id = config.client_id
            existing.auto_connect = config.auto_connect
            existing.is_active = True
            connection = existing
        else:
            # Créer nouveau
            connection = IBKRConnection(
                user_id=current_user.id,
                account_id=config.account_id,
                connection_type=config.connection_type,
                host=config.host,
                port=config.port,
                client_id=config.client_id,
                auto_connect=config.auto_connect
            )
            db.add(connection)

        db.commit()
        db.refresh(connection)

        # Tenter la connexion
        service = get_ibkr_service(db, current_user.id)
        service.connection_config = connection
        success = await service.connect(auto_connect=False)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Impossible de se connecter à IBKR. Vérifiez que TWS/Gateway est lancé et l'API activée."
            )

        logger.info(f"✅ IBKR connecté pour {current_user.email}: {config.account_id}")

        return IBKRConnectionResponse.from_orm(connection)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur connexion IBKR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connection", response_model=IBKRConnectionResponse)
async def get_connection_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Récupérer le statut de la connexion IBKR"""
    connection = db.query(IBKRConnection).filter(
        IBKRConnection.user_id == current_user.id,
        IBKRConnection.is_active == True
    ).first()

    if not connection:
        raise HTTPException(
            status_code=404,
            detail="Aucune connexion IBKR configurée"
        )

    return IBKRConnectionResponse.from_orm(connection)


@router.post("/disconnect")
async def disconnect_ibkr(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Déconnecter d'IBKR"""
    try:
        service = get_ibkr_service(db, current_user.id)
        await service.connect(auto_connect=True)  # Charger la config
        service.disconnect()

        return {"status": "disconnected"}

    except Exception as e:
        logger.error(f"Erreur déconnexion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINTS - PORTEFEUILLE
# ============================================

@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupérer le portefeuille IBKR en temps réel

    Retourne toutes les positions, cash, P&L, etc.
    """
    try:
        service = get_ibkr_service(db, current_user.id)

        # Connecter si nécessaire
        if not service.is_connected:
            success = await service.connect(auto_connect=True)
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Impossible de se connecter à IBKR"
                )

        # Récupérer le portefeuille
        portfolio = await service.get_portfolio()

        if not portfolio:
            raise HTTPException(
                status_code=500,
                detail="Impossible de récupérer le portefeuille"
            )

        # Sauvegarder le snapshot
        await service.save_portfolio_snapshot(portfolio)

        # Vérifier les alertes
        await service.check_alerts(portfolio)

        return PortfolioResponse(**portfolio)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération portefeuille: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/history")
async def get_portfolio_history(
    days: int = 7,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupérer l'historique du portefeuille

    Args:
        days: Nombre de jours d'historique (défaut: 7)
    """
    try:
        connection = db.query(IBKRConnection).filter(
            IBKRConnection.user_id == current_user.id,
            IBKRConnection.is_active == True
        ).first()

        if not connection:
            raise HTTPException(status_code=404, detail="Pas de connexion IBKR")

        # Récupérer les snapshots
        start_date = datetime.utcnow() - timedelta(days=days)
        snapshots = db.query(PortfolioSnapshot).filter(
            PortfolioSnapshot.connection_id == connection.id,
            PortfolioSnapshot.timestamp >= start_date
        ).order_by(PortfolioSnapshot.timestamp).all()

        history = []
        for snapshot in snapshots:
            history.append({
                'timestamp': snapshot.timestamp.isoformat(),
                'net_liquidation': snapshot.net_liquidation,
                'total_cash': snapshot.total_cash,
                'stock_value': snapshot.stock_value,
                'unrealized_pnl': snapshot.unrealized_pnl,
                'realized_pnl': snapshot.realized_pnl,
                'positions_count': len(snapshot.positions) if snapshot.positions else 0
            })

        return {
            'account_id': connection.account_id,
            'period_days': days,
            'snapshots_count': len(history),
            'history': history
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur historique: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINTS - ALERTES
# ============================================

@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    active_only: bool = True,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupérer les alertes du portefeuille

    Args:
        active_only: Uniquement les alertes actives (défaut: true)
        limit: Nombre max d'alertes (défaut: 50)
    """
    try:
        connection = db.query(IBKRConnection).filter(
            IBKRConnection.user_id == current_user.id,
            IBKRConnection.is_active == True
        ).first()

        if not connection:
            return []

        query = db.query(IBKRAlert).filter(
            IBKRAlert.connection_id == connection.id
        )

        if active_only:
            query = query.filter(IBKRAlert.is_active == True)

        alerts = query.order_by(
            IBKRAlert.triggered_at.desc()
        ).limit(limit).all()

        return [AlertResponse.from_orm(alert) for alert in alerts]

    except Exception as e:
        logger.error(f"Erreur récupération alertes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Acquitter une alerte"""
    try:
        alert = db.query(IBKRAlert).filter(
            IBKRAlert.id == alert_id
        ).first()

        if not alert:
            raise HTTPException(status_code=404, detail="Alerte non trouvée")

        alert.is_acknowledged = True
        alert.acknowledged_at = datetime.utcnow()
        db.commit()

        return {"status": "acknowledged"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur acquittement alerte: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Résoudre une alerte"""
    try:
        alert = db.query(IBKRAlert).filter(
            IBKRAlert.id == alert_id
        ).first()

        if not alert:
            raise HTTPException(status_code=404, detail="Alerte non trouvée")

        alert.is_active = False
        alert.resolved_at = datetime.utcnow()
        db.commit()

        return {"status": "resolved"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur résolution alerte: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINTS - ORDRES
# ============================================

@router.get("/orders")
async def get_orders(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupérer l'historique des ordres

    Args:
        days: Nombre de jours d'historique (défaut: 30)
    """
    try:
        service = get_ibkr_service(db, current_user.id)

        # Connecter si nécessaire
        if not service.is_connected:
            success = await service.connect(auto_connect=True)
            if not success:
                raise HTTPException(status_code=503, detail="Pas connecté à IBKR")

        # Récupérer les ordres
        orders = await service.get_orders_history(days)

        return {
            'period_days': days,
            'orders_count': len(orders),
            'orders': orders
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération ordres: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINTS - ANALYSE
# ============================================

@router.post("/analyze")
async def analyze_portfolio(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyser le portefeuille avec le moteur de scénarios

    Lance une analyse complète du portefeuille:
    - Stress tests
    - Scénarios de crises
    - Recommandations
    """
    try:
        service = get_ibkr_service(db, current_user.id)

        # Connecter si nécessaire
        if not service.is_connected:
            success = await service.connect(auto_connect=True)
            if not success:
                raise HTTPException(status_code=503, detail="Pas connecté à IBKR")

        # Récupérer le portefeuille
        portfolio = await service.get_portfolio()

        if not portfolio:
            raise HTTPException(status_code=500, detail="Impossible de récupérer le portefeuille")

        # TODO: Intégrer avec le moteur de scénarios
        # Pour l'instant, retourner un placeholder

        return {
            'status': 'analysis_started',
            'message': 'Analyse du portefeuille en cours...',
            'portfolio': {
                'net_liquidation': portfolio['net_liquidation'],
                'positions_count': len(portfolio['positions']),
                'currency': portfolio['currency']
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur analyse: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINTS - DASHBOARD
# ============================================

@router.get("/dashboard")
async def get_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupérer toutes les données pour le dashboard

    Retourne:
    - Portefeuille actuel
    - Alertes actives
    - Dernières transactions
    - Graphique de performance
    """
    try:
        service = get_ibkr_service(db, current_user.id)

        # Connecter si nécessaire
        if not service.is_connected:
            success = await service.connect(auto_connect=True)
            if not success:
                # Retourner des données vides si pas connecté
                return {
                    'connected': False,
                    'portfolio': None,
                    'alerts': [],
                    'performance': []
                }

        # Récupérer le portefeuille
        portfolio = await service.get_portfolio()

        if portfolio:
            # Sauvegarder snapshot
            await service.save_portfolio_snapshot(portfolio)

            # Vérifier alertes
            await service.check_alerts(portfolio)

        # Récupérer les alertes actives
        connection = db.query(IBKRConnection).filter(
            IBKRConnection.user_id == current_user.id,
            IBKRConnection.is_active == True
        ).first()

        alerts = []
        if connection:
            alerts_db = db.query(IBKRAlert).filter(
                IBKRAlert.connection_id == connection.id,
                IBKRAlert.is_active == True
            ).order_by(IBKRAlert.triggered_at.desc()).limit(5).all()

            alerts = [AlertResponse.from_orm(alert) for alert in alerts_db]

        # Récupérer performance (7 derniers jours)
        start_date = datetime.utcnow() - timedelta(days=7)
        snapshots = []
        if connection:
            snapshots_db = db.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.connection_id == connection.id,
                PortfolioSnapshot.timestamp >= start_date
            ).order_by(PortfolioSnapshot.timestamp).all()

            snapshots = [{
                'timestamp': s.timestamp.isoformat(),
                'net_liquidation': s.net_liquidation,
                'unrealized_pnl': s.unrealized_pnl
            } for s in snapshots_db]

        return {
            'connected': True,
            'portfolio': portfolio,
            'alerts': alerts,
            'performance': snapshots,
            'last_update': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Erreur dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dev-dashboard")
async def get_dev_dashboard(
    db: Session = Depends(get_db)
):
    """
    Dashboard IBKR public pour le mode développement

    ⚠️ ATTENTION: Cet endpoint ne nécessite pas d'authentification!
    À utiliser uniquement en développement.

    Utilise le premier compte IBKR avec auto_connect=True
    """
    try:
        # Trouver le premier compte avec auto_connect
        connection = db.query(IBKRConnection).filter(
            IBKRConnection.auto_connect == True,
            IBKRConnection.is_active == True
        ).first()

        if not connection:
            return {
                'connected': False,
                'message': 'Aucun compte IBKR configuré avec auto-connect',
                'portfolio': None,
                'alerts': [],
                'performance': []
            }

        service = get_ibkr_service(db, connection.user_id)

        # Connecter si nécessaire
        if not service.is_connected:
            success = await service.connect(auto_connect=True)
            if not success:
                return {
                    'connected': False,
                    'message': 'Impossible de se connecter à IBKR',
                    'portfolio': None,
                    'alerts': [],
                    'performance': []
                }

        # Récupérer le portefeuille
        portfolio = await service.get_portfolio()

        if portfolio:
            # Sauvegarder snapshot
            await service.save_portfolio_snapshot(portfolio)

            # Vérifier alertes
            await service.check_alerts(portfolio)

        # Récupérer les alertes actives
        alerts_db = db.query(IBKRAlert).filter(
            IBKRAlert.connection_id == connection.id,
            IBKRAlert.is_active == True
        ).order_by(IBKRAlert.triggered_at.desc()).limit(5).all()

        alerts = [AlertResponse.from_orm(alert) for alert in alerts_db]

        # Récupérer performance (7 derniers jours)
        start_date = datetime.utcnow() - timedelta(days=7)
        snapshots_db = db.query(PortfolioSnapshot).filter(
            PortfolioSnapshot.connection_id == connection.id,
            PortfolioSnapshot.timestamp >= start_date
        ).order_by(PortfolioSnapshot.timestamp).all()

        snapshots = [{
            'timestamp': s.timestamp.isoformat(),
            'net_liquidation': s.net_liquidation,
            'unrealized_pnl': s.unrealized_pnl
        } for s in snapshots_db]

        return {
            'connected': True,
            'portfolio': portfolio,
            'alerts': alerts,
            'performance': snapshots,
            'last_update': datetime.utcnow().isoformat(),
            'dev_mode': True
        }

    except Exception as e:
        logger.error(f"Erreur dev dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))
