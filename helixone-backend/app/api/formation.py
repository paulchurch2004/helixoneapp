"""
API Endpoints pour la formation commerciale et le paper trading
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
import json
import os
from pathlib import Path
from datetime import datetime, date

from app.services.paper_trading import PaperTradingService
from app.core.database import get_db
from app.models.user_progress import UserProgress
from app.models.user import User

router = APIRouter(prefix="/api/formation", tags=["Formation"])

# Mock user pour testing sans auth
class MockUser:
    def __init__(self):
        self.id = "bbbe7c84-2c7f-4773-8f18-15c2ea08d2a7"  # ID réel de test@test.com
        self.email = "test@test.com"

def get_mock_user():
    """Mock user pour testing"""
    return MockUser()


# ============================================================================
# MODELS
# ============================================================================

class OrderRequest(BaseModel):
    """Requête pour passer un ordre"""
    ticker: str
    quantity: int
    order_type: str = "market"


class ResetPortfolioRequest(BaseModel):
    """Requête pour réinitialiser le portfolio"""
    initial_capital: float = 100000.0


class StopLossRequest(BaseModel):
    """Requête pour définir un stop-loss"""
    ticker: str
    stop_price: float


class TakeProfitRequest(BaseModel):
    """Requête pour définir un take-profit"""
    ticker: str
    take_profit_price: float


class CancelOrderRequest(BaseModel):
    """Requête pour annuler des ordres"""
    ticker: str
    order_type: Optional[str] = "all"  # "stop_loss", "take_profit", or "all"


class ProgressSyncRequest(BaseModel):
    """Requête pour synchroniser la progression"""
    total_xp: int
    level: int
    completed_modules: List[str]
    module_scores: Dict[str, Dict]
    current_streak: int = 0
    badges: List[str] = []
    certifications: List[str] = []


# ============================================================================
# MODULES & CONTENU
# ============================================================================

@router.get("/modules/{parcours}")
async def get_modules(
    parcours: str,
    user: MockUser = Depends(get_mock_user)
):
    """
    Récupère tous les modules d'un parcours (débutant, intermédiaire, expert)

    Args:
        parcours: Type de parcours

    Returns:
        Liste des modules avec métadonnées
    """
    # Charger le fichier des modules
    modules_file = Path(__file__).parent.parent.parent.parent / "data" / "formation_commerciale" / "modules_complets.json"

    if not modules_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fichier des modules introuvable"
        )

    with open(modules_file, 'r', encoding='utf-8') as f:
        all_modules = json.load(f)

    # Filtrer par parcours
    modules = [m for m in all_modules if m.get('parcours') == parcours]

    # Retourner seulement les métadonnées (pas le contenu complet)
    return [
        {
            "id": m['id'],
            "titre": m['titre'],
            "description": m['description'],
            "durée": m['durée'],
            "difficulté": m['difficulté'],
            "points_xp": m['points_xp'],
            "prérequis": m.get('prérequis', [])
        }
        for m in modules
    ]


@router.get("/module/{module_id}")
async def get_module_detail(
    module_id: str,
    user: MockUser = Depends(get_mock_user)
):
    """
    Récupère le contenu complet d'un module

    Args:
        module_id: ID du module

    Returns:
        Module complet avec contenu, quiz, exercices
    """
    modules_file = Path(__file__).parent.parent.parent.parent / "data" / "formation_commerciale" / "modules_complets.json"

    if not modules_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fichier des modules introuvable"
        )

    with open(modules_file, 'r', encoding='utf-8') as f:
        all_modules = json.load(f)

    # Trouver le module
    module = next((m for m in all_modules if m['id'] == module_id), None)

    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Module {module_id} introuvable"
        )

    return module


@router.post("/module/{module_id}/complete")
async def complete_module(
    module_id: str,
    quiz_score: Optional[int] = None,
    time_spent: Optional[int] = None,
    user: MockUser = Depends(get_mock_user),
    db: Session = Depends(get_db)
):
    """
    Marque un module comme complété et met à jour la progression

    Args:
        module_id: ID du module
        quiz_score: Score au quiz (sur 100)
        time_spent: Temps passé en secondes

    Returns:
        Confirmation et XP gagnés
    """
    # Charger le module pour récupérer les points XP
    modules_file = Path(__file__).parent.parent.parent.parent / "data" / "formation_commerciale" / "modules_complets.json"

    with open(modules_file, 'r', encoding='utf-8') as f:
        all_modules = json.load(f)

    module = next((m for m in all_modules if m['id'] == module_id), None)

    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Module {module_id} introuvable"
        )

    xp_earned = module.get('points_xp', 100)

    # Si le score est excellent, bonus XP
    if quiz_score and quiz_score >= 90:
        xp_earned = int(xp_earned * 1.2)

    # Sauvegarder dans la base de données
    progress = db.query(UserProgress).filter(UserProgress.user_id == user.id).first()

    if not progress:
        # Créer une nouvelle progression
        progress = UserProgress(
            user_id=user.id,
            total_xp=0,
            level=1,
            completed_modules=[],
            module_scores={}
        )
        db.add(progress)

    # Mettre à jour la progression
    if module_id not in (progress.completed_modules or []):
        progress.completed_modules = (progress.completed_modules or []) + [module_id]

    progress.total_xp = (progress.total_xp or 0) + xp_earned

    # Calculer le niveau (100 XP par niveau)
    progress.level = 1 + (progress.total_xp // 100)

    # Sauvegarder le score du module
    module_scores = progress.module_scores or {}
    module_scores[module_id] = {
        "score": quiz_score,
        "time_spent": time_spent,
        "completed_at": datetime.now().isoformat()
    }
    progress.module_scores = module_scores

    # Mettre à jour la date d'activité
    progress.last_activity_date = datetime.now()
    progress.updated_at = datetime.now()

    db.commit()
    db.refresh(progress)

    return {
        "success": True,
        "module_id": module_id,
        "xp_earned": xp_earned,
        "quiz_score": quiz_score,
        "time_spent": time_spent,
        "total_xp": progress.total_xp,
        "level": progress.level,
        "message": f"Module complété ! +{xp_earned} XP"
    }


@router.get("/progress")
async def get_user_progress(
    user: MockUser = Depends(get_mock_user),
    db: Session = Depends(get_db)
):
    """
    Récupère la progression de l'utilisateur dans la formation

    Returns:
        Statistiques de progression
    """
    progress = db.query(UserProgress).filter(UserProgress.user_id == user.id).first()

    if not progress:
        # Retourner une progression vide pour un nouvel utilisateur
        return {
            "user_id": user.id,
            "total_xp": 0,
            "level": 1,
            "modules_completed": 0,
            "modules_total": 20,  # TODO: Compter dynamiquement
            "completion_percent": 0,
            "current_streak": 0,
            "completed_modules": [],
            "module_scores": {}
        }

    completed_count = len(progress.completed_modules or [])
    modules_total = 20  # TODO: Compter depuis modules_complets.json

    return {
        "user_id": user.id,
        "total_xp": progress.total_xp,
        "level": progress.level,
        "modules_completed": completed_count,
        "modules_total": modules_total,
        "completion_percent": int((completed_count / modules_total) * 100) if modules_total > 0 else 0,
        "current_streak": progress.current_streak or 0,
        "completed_modules": progress.completed_modules or [],
        "module_scores": progress.module_scores or {},
        "last_activity": progress.last_activity_date.isoformat() if progress.last_activity_date else None
    }


@router.post("/progress/sync")
async def sync_progress(
    data: ProgressSyncRequest,
    user: MockUser = Depends(get_mock_user),
    db: Session = Depends(get_db)
):
    """
    Synchronise la progression de l'utilisateur depuis le client

    Args:
        data: Données de progression à synchroniser

    Returns:
        Confirmation et progression mise à jour
    """
    progress = db.query(UserProgress).filter(UserProgress.user_id == user.id).first()

    if not progress:
        # Créer une nouvelle progression
        progress = UserProgress(
            user_id=user.id,
            total_xp=data.total_xp,
            level=data.level,
            completed_modules=data.completed_modules,
            module_scores=data.module_scores,
            current_streak=data.current_streak,
            badges=data.badges,
            certifications=data.certifications
        )
        db.add(progress)
    else:
        # Mettre à jour avec les données les plus récentes
        # On prend toujours le maximum (pour éviter de perdre de la progression)
        progress.total_xp = max(progress.total_xp or 0, data.total_xp)
        progress.level = max(progress.level or 1, data.level)

        # Fusionner les modules complétés (union)
        existing_modules = set(progress.completed_modules or [])
        new_modules = set(data.completed_modules)
        progress.completed_modules = list(existing_modules | new_modules)

        # Fusionner les scores (garder le meilleur score pour chaque module)
        existing_scores = progress.module_scores or {}
        for module_id, score_data in data.module_scores.items():
            if module_id not in existing_scores or score_data.get('score', 0) > existing_scores[module_id].get('score', 0):
                existing_scores[module_id] = score_data
        progress.module_scores = existing_scores

        # Fusionner les badges et certifications (union)
        existing_badges = set(progress.badges or [])
        new_badges = set(data.badges)
        progress.badges = list(existing_badges | new_badges)

        existing_certs = set(progress.certifications or [])
        new_certs = set(data.certifications)
        progress.certifications = list(existing_certs | new_certs)

        progress.current_streak = max(progress.current_streak or 0, data.current_streak)
        progress.last_activity_date = datetime.now()
        progress.updated_at = datetime.now()

    db.commit()
    db.refresh(progress)

    return {
        "success": True,
        "message": "Progression synchronisée avec succès",
        "progress": progress.to_dict()
    }


# ============================================================================
# PAPER TRADING (SIMULATEUR)
# ============================================================================

@router.post("/simulator/order")
async def place_order(
    order: OrderRequest,
    user: MockUser = Depends(get_mock_user)
):
    """
    Passe un ordre d'achat ou vente dans le simulateur

    Args:
        order: Détails de l'ordre (ticker, quantité, type)

    Returns:
        Résultat de l'ordre
    """
    service = PaperTradingService(user_id=str(user.id))

    result = await service.place_order(
        ticker=order.ticker,
        quantity=order.quantity,
        order_type=order.order_type
    )

    if not result.get('success'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get('error', 'Erreur lors du passage de l\'ordre')
        )

    return result


@router.get("/simulator/portfolio")
async def get_portfolio(
    user: MockUser = Depends(get_mock_user)
):
    """
    Récupère le portfolio simulé de l'utilisateur

    Returns:
        Valeur du portfolio et positions
    """
    service = PaperTradingService(user_id=str(user.id))

    portfolio_value = service.get_portfolio_value()
    positions = service.get_position_details()

    return {
        "portfolio": portfolio_value,
        "positions": positions
    }


@router.get("/simulator/history")
async def get_trade_history(
    limit: int = 50,
    user: MockUser = Depends(get_mock_user)
):
    """
    Récupère l'historique des trades

    Args:
        limit: Nombre maximum de trades

    Returns:
        Liste des trades
    """
    service = PaperTradingService(user_id=str(user.id))
    history = service.get_trade_history(limit=limit)

    return {
        "trades": history,
        "count": len(history)
    }


@router.get("/simulator/statistics")
async def get_statistics(
    user: MockUser = Depends(get_mock_user)
):
    """
    Récupère les statistiques de trading

    Returns:
        Statistiques complètes
    """
    service = PaperTradingService(user_id=str(user.id))
    stats = service.get_statistics()

    return stats


@router.post("/simulator/reset")
async def reset_portfolio(
    request: ResetPortfolioRequest,
    user: MockUser = Depends(get_mock_user)
):
    """
    Réinitialise le portfolio (recommencer à zéro)

    Args:
        request: Capital initial

    Returns:
        Confirmation
    """
    service = PaperTradingService(user_id=str(user.id))
    result = service.reset_portfolio(initial_capital=request.initial_capital)

    return result


# ============================================================================
# STOP-LOSS & TAKE-PROFIT
# ============================================================================

@router.post("/simulator/stop-loss")
async def set_stop_loss(
    request: StopLossRequest,
    user: MockUser = Depends(get_mock_user)
):
    """
    Définit un stop-loss pour une position

    Args:
        request: Ticker et prix de stop-loss

    Returns:
        Confirmation et détails du stop-loss
    """
    service = PaperTradingService(user_id=str(user.id))
    result = service.set_stop_loss(
        ticker=request.ticker,
        stop_price=request.stop_price
    )

    if not result.get('success'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get('error', 'Erreur lors de la définition du stop-loss')
        )

    return result


@router.post("/simulator/take-profit")
async def set_take_profit(
    request: TakeProfitRequest,
    user: MockUser = Depends(get_mock_user)
):
    """
    Définit un take-profit pour une position

    Args:
        request: Ticker et prix de take-profit

    Returns:
        Confirmation et détails du take-profit
    """
    service = PaperTradingService(user_id=str(user.id))
    result = service.set_take_profit(
        ticker=request.ticker,
        take_profit_price=request.take_profit_price
    )

    if not result.get('success'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get('error', 'Erreur lors de la définition du take-profit')
        )

    return result


@router.get("/simulator/orders")
async def get_active_orders(
    user: MockUser = Depends(get_mock_user)
):
    """
    Récupère tous les ordres actifs (stop-loss et take-profit)

    Returns:
        Liste des ordres actifs avec détails
    """
    service = PaperTradingService(user_id=str(user.id))
    orders = service.get_active_orders()

    return {
        "orders": orders,
        "count": len(orders)
    }


@router.post("/simulator/cancel-order")
async def cancel_order(
    request: CancelOrderRequest,
    user: MockUser = Depends(get_mock_user)
):
    """
    Annule un ou plusieurs ordres (stop-loss, take-profit, ou les deux)

    Args:
        request: Ticker et type d'ordre à annuler

    Returns:
        Confirmation de l'annulation
    """
    service = PaperTradingService(user_id=str(user.id))
    result = service.cancel_order(
        ticker=request.ticker,
        order_type=request.order_type
    )

    if not result.get('success'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get('error', 'Erreur lors de l\'annulation de l\'ordre')
        )

    return result


@router.post("/simulator/check-orders")
async def check_orders(
    user: MockUser = Depends(get_mock_user)
):
    """
    Vérifie et exécute les ordres stop-loss et take-profit en attente

    Returns:
        Liste des ordres exécutés
    """
    service = PaperTradingService(user_id=str(user.id))
    executed = service.check_and_execute_orders()

    return {
        "executed_orders": executed,
        "count": len(executed)
    }
