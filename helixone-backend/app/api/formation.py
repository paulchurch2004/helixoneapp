"""
API Endpoints pour la formation commerciale et le paper trading
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import os
from pathlib import Path

from app.services.paper_trading import PaperTradingService

router = APIRouter(prefix="/api/formation", tags=["Formation"])

# Mock user pour testing sans auth
class MockUser:
    def __init__(self):
        self.id = "test_user_001"
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
    user: MockUser = Depends(get_mock_user)
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
    # TODO: Implémenter la sauvegarde de progression en base de données
    # Pour le moment, retourner une réponse simulée

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

    return {
        "success": True,
        "module_id": module_id,
        "xp_earned": xp_earned,
        "quiz_score": quiz_score,
        "time_spent": time_spent,
        "message": f"Module complété ! +{xp_earned} XP"
    }


@router.get("/progress")
async def get_user_progress(
    user: MockUser = Depends(get_mock_user)
):
    """
    Récupère la progression de l'utilisateur dans la formation

    Returns:
        Statistiques de progression
    """
    # TODO: Implémenter lecture depuis base de données
    # Pour le moment, retourner des données simulées

    return {
        "user_id": user.id,
        "total_xp": 250,
        "level": 2,
        "modules_completed": 1,
        "modules_total": 20,
        "completion_percent": 5,
        "current_streak": 3,
        "completed_modules": ["basics_1"]
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
