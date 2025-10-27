"""
API Endpoints pour le Moteur de Simulation de Scénarios
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List
import logging

from app.core.database import get_db
from app.schemas.scenario import (
    StressTestRequest,
    HistoricalScenarioRequest,
    MonteCarloRequest,
    ScenarioSimulationResult,
    MonteCarloResult,
    PredefinedScenariosResponse,
    PredefinedScenario,
    HistoricalEventsResponse,
    HistoricalEventDetail
)
from app.services.scenario_engine import ScenarioEngine
from app.models.scenario import (
    PREDEFINED_STRESS_TESTS,
    PREDEFINED_HISTORICAL_EVENTS,
    HistoricalEvent,
    ScenarioSimulation
)
from app.models.user import User
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scenarios", tags=["Scenarios"])

# Instance globale du moteur
scenario_engine = ScenarioEngine()


@router.post("/stress-test", response_model=ScenarioSimulationResult)
async def run_stress_test(
    request: StressTestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Exécute un stress test sur un portfolio

    Exemples de stress tests:
    - Crash de marché (-20%, -30%, -50%)
    - Choc de taux (+2%, +5%)
    - Spike de volatilité (VIX x3, x5)
    - Choc d'inflation
    - Crise de liquidité
    """
    try:
        logger.info(f"🚀 Stress test demandé par user {current_user.id}")
        logger.info(f"   Type: {request.stress_test_type.value}")
        logger.info(f"   Portfolio: {len(request.portfolio)} positions")

        # Exécuter le stress test
        result = scenario_engine.run_stress_test(
            portfolio=request.portfolio,
            stress_test_type=request.stress_test_type,
            shock_percent=request.shock_percent,
            rate_change=request.rate_change,
            vix_multiplier=request.vix_multiplier
        )

        # Sauvegarder en DB
        try:
            simulation = ScenarioSimulation(
                scenario_id=None,  # Pas de scénario prédéfini
                user_id=current_user.id,
                portfolio_snapshot=request.portfolio,
                results={
                    "position_impacts": [p.dict() for p in result.position_impacts]
                },
                metrics=result.risk_metrics.dict(),
                recommendations=[r.dict() for r in result.recommendations],
                execution_time_ms=result.execution_time_ms
            )
            db.add(simulation)
            db.commit()
            logger.info(f"✅ Simulation sauvegardée: {simulation.id}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde DB: {e}")
            db.rollback()

        return result

    except Exception as e:
        logger.error(f"Erreur stress test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du stress test: {str(e)}"
        )


@router.get("/predefined", response_model=PredefinedScenariosResponse)
async def get_predefined_scenarios(
    current_user: User = Depends(get_current_user)
):
    """
    Liste tous les scénarios prédéfinis disponibles

    Retourne:
    - Stress tests standards
    - Événements historiques (2008, COVID, etc.)
    """
    try:
        # Stress tests
        stress_tests = [
            PredefinedScenario(
                name=st["name"],
                description=f"Test de résistance: {st['type']}",
                type=st["type"],
                parameters=st["parameters"]
            )
            for st in PREDEFINED_STRESS_TESTS
        ]

        # Événements historiques
        historical_events = [
            PredefinedScenario(
                name=event["name"],
                description=event.get("description", ""),
                type="historical",
                parameters={"event_name": event["name"]}
            )
            for event in PREDEFINED_HISTORICAL_EVENTS
        ]

        return PredefinedScenariosResponse(
            stress_tests=stress_tests,
            historical_events=historical_events,
            total=len(stress_tests) + len(historical_events)
        )

    except Exception as e:
        logger.error(f"Erreur get predefined scenarios: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/historical-events", response_model=HistoricalEventsResponse)
async def get_historical_events(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Liste tous les événements historiques disponibles avec détails complets

    Retourne des informations détaillées sur chaque crise:
    - 2008 Financial Crisis
    - COVID-19 2020
    - Dot-com Bubble 2000
    - etc.
    """
    try:
        # Pour l'instant, retourner les événements prédéfinis
        # TODO: Charger depuis la DB si disponible
        events = []

        for event_data in PREDEFINED_HISTORICAL_EVENTS:
            from datetime import datetime
            event = HistoricalEventDetail(
                id=event_data["name"].lower().replace(" ", "_"),
                name=event_data["name"],
                start_date=datetime.fromisoformat(event_data["start_date"]),
                end_date=datetime.fromisoformat(event_data["end_date"]),
                duration_days=(
                    datetime.fromisoformat(event_data["end_date"]) -
                    datetime.fromisoformat(event_data["start_date"])
                ).days,
                market_move_pct=event_data["market_move_pct"],
                volatility_avg=event_data["volatility_avg"],
                sector_impacts=event_data["sector_impacts"],
                macro_context=event_data["macro_context"],
                triggers=event_data["triggers"],
                recovery_pattern=event_data["recovery_pattern"],
                recovery_duration_days=event_data["recovery_duration_days"],
                description=event_data.get("description", "")
            )
            events.append(event)

        return HistoricalEventsResponse(
            events=events,
            total=len(events)
        )

    except Exception as e:
        logger.error(f"Erreur get historical events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/historical", response_model=ScenarioSimulationResult)
async def run_historical_scenario(
    request: HistoricalScenarioRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Rejoue un événement historique sur le portfolio actuel

    Applique les mêmes mouvements sectoriels qu'une crise historique
    (ex: 2008, COVID) sur votre portfolio d'aujourd'hui
    """
    try:
        logger.info(f"📜 Scénario historique: {request.event_name}")

        # Trouver l'événement
        event_data = next(
            (e for e in PREDEFINED_HISTORICAL_EVENTS if e["name"] == request.event_name),
            None
        )

        if not event_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Événement historique non trouvé: {request.event_name}"
            )

        # Collecter données de portfolio
        stock_data = scenario_engine._collect_stock_characteristics(list(request.portfolio.keys()))

        # Calculer valeur initiale
        portfolio_value_before = sum(
            stock_data.get(ticker, {}).get('price', 100) * qty
            for ticker, qty in request.portfolio.items()
        )

        # Appliquer les impacts sectoriels historiques
        from app.schemas.scenario import PositionImpact
        position_impacts = []

        for ticker, qty in request.portfolio.items():
            data = stock_data.get(ticker, {})
            price_before = data.get('price', 100)
            sector = data.get('sector', 'Unknown')

            # Impact sectoriel de l'événement
            sector_impact = event_data["sector_impacts"].get(sector, event_data["market_move_pct"])

            # Ajouter du bruit
            import numpy as np
            impact = sector_impact * np.random.normal(1.0, 0.1)

            price_after = price_before * (1 + impact / 100)
            price_after = max(price_after, 0.01)

            value_before = price_before * qty
            value_after = price_after * qty
            impact_amount = value_after - value_before

            position_impacts.append(PositionImpact(
                ticker=ticker,
                quantity=qty,
                price_before=price_before,
                price_after=price_after,
                value_before=value_before,
                value_after=value_after,
                impact_percent=impact,
                impact_amount=impact_amount
            ))

        # Calculer valeur finale
        portfolio_value_after = sum(pos.value_after for pos in position_impacts)
        impact_amount = portfolio_value_after - portfolio_value_before
        impact_percent = (impact_amount / portfolio_value_before) * 100

        # Calculer métriques
        risk_metrics = scenario_engine._calculate_risk_metrics(
            position_impacts,
            portfolio_value_before,
            portfolio_value_after
        )

        # Générer recommandations
        recommendations = scenario_engine._generate_recommendations(
            position_impacts,
            stock_data,
            risk_metrics,
            impact_percent
        )

        # Identifier pires/meilleures positions
        worst_position = max(position_impacts, key=lambda p: abs(p.impact_percent))
        best_position = min(position_impacts, key=lambda p: abs(p.impact_percent))

        result = ScenarioSimulationResult(
            scenario_name=f"Replay: {request.event_name}",
            scenario_type="historical",
            portfolio_value_before=portfolio_value_before,
            portfolio_value_after=portfolio_value_after,
            impact_percent=impact_percent,
            impact_amount=impact_amount,
            position_impacts=position_impacts,
            risk_metrics=risk_metrics,
            recommendations=recommendations,
            worst_position=f"{worst_position.ticker} ({worst_position.impact_percent:+.1f}%)",
            best_position=f"{best_position.ticker} ({best_position.impact_percent:+.1f}%)",
            execution_time_ms=0
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur scénario historique: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/monte-carlo", response_model=MonteCarloResult)
async def run_monte_carlo(
    request: MonteCarloRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Exécute une simulation Monte Carlo (10,000 trajectoires futures)

    Génère des milliers de scénarios possibles et calcule:
    - VaR (Value at Risk)
    - CVaR (Conditional VaR)
    - Probabilités de profit/perte
    - Percentiles de résultats
    """
    try:
        logger.info(f"🎲 Monte Carlo: {request.n_simulations} simulations")

        # TODO: Implémenter Monte Carlo avec le simulator existant
        # Pour l'instant, retourner un résultat mock

        import numpy as np

        initial_value = sum(
            scenario_engine._collect_stock_characteristics(list(request.portfolio.keys()))
            .get(ticker, {}).get('price', 100) * qty
            for ticker, qty in request.portfolio.items()
        )

        # Mock simulation
        mean_return = 0.08  # 8% par an
        volatility = 0.20  # 20% volatilité

        final_returns = np.random.normal(mean_return, volatility, request.n_simulations)
        final_values = initial_value * (1 + final_returns)

        result = MonteCarloResult(
            n_simulations=request.n_simulations,
            forecast_days=request.forecast_days,
            initial_value=initial_value,
            mean_final_value=float(np.mean(final_values)),
            median_final_value=float(np.median(final_values)),
            std_final_value=float(np.std(final_values)),
            min_final_value=float(np.min(final_values)),
            max_final_value=float(np.max(final_values)),
            mean_return=float(np.mean(final_returns) * 100),
            median_return=float(np.median(final_returns) * 100),
            std_return=float(np.std(final_returns) * 100),
            min_return=float(np.min(final_returns) * 100),
            max_return=float(np.max(final_returns) * 100),
            percentiles={
                f"p{p}": {
                    "value": float(np.percentile(final_values, p)),
                    "return": float(np.percentile(final_returns, p) * 100)
                }
                for p in [5, 25, 50, 75, 95]
            },
            var_95=float(np.percentile(final_returns, 5) * 100),
            cvar_95=float(np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]) * 100),
            var_99=float(np.percentile(final_returns, 1) * 100),
            cvar_99=float(np.mean(final_returns[final_returns <= np.percentile(final_returns, 1)]) * 100),
            prob_profit=float((final_returns > 0).mean() * 100),
            prob_loss_10pct=float((final_returns < -0.10).mean() * 100),
            prob_gain_20pct=float((final_returns > 0.20).mean() * 100),
            stress_score=70,
            recommendations=[],
            execution_time_ms=1000
        )

        return result

    except Exception as e:
        logger.error(f"Erreur Monte Carlo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/my-simulations")
async def get_my_simulations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """
    Récupère l'historique des simulations de l'utilisateur
    """
    try:
        simulations = db.query(ScenarioSimulation).filter(
            ScenarioSimulation.user_id == current_user.id
        ).order_by(
            ScenarioSimulation.created_at.desc()
        ).limit(limit).all()

        return {
            "simulations": [
                {
                    "id": str(sim.id),
                    "created_at": sim.created_at.isoformat(),
                    "portfolio_size": len(sim.portfolio_snapshot),
                    "metrics": sim.metrics,
                    "execution_time_ms": sim.execution_time_ms
                }
                for sim in simulations
            ],
            "total": len(simulations)
        }

    except Exception as e:
        logger.error(f"Erreur get simulations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
