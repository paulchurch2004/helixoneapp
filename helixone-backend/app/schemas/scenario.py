"""
Schemas Pydantic pour l'API de Simulation de Scénarios
Validation des requêtes et réponses
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class ScenarioType(str, Enum):
    STRESS_TEST = "stress_test"
    HISTORICAL = "historical"
    MACRO_ECONOMIC = "macro_economic"
    SECTORAL = "sectoral"
    COMPOSITE = "composite"
    ML_GENERATED = "ml_generated"
    CUSTOM = "custom"


class StressTestType(str, Enum):
    MARKET_CRASH = "market_crash"
    RATE_SHOCK = "rate_shock"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    INFLATION_SHOCK = "inflation_shock"


class RecoveryPattern(str, Enum):
    V_SHAPED = "V_shaped"
    U_SHAPED = "U_shaped"
    L_SHAPED = "L_shaped"
    W_SHAPED = "W_shaped"
    NIKE_SHAPED = "Nike_shaped"


# ============================================================================
# REQUÊTES
# ============================================================================

class StressTestRequest(BaseModel):
    """Requête pour stress test"""
    portfolio: Dict[str, float] = Field(..., description="Portfolio à tester: {ticker: quantity}")
    stress_test_type: StressTestType = Field(..., description="Type de stress test")
    shock_percent: Optional[float] = Field(None, description="Ampleur du choc en %")
    rate_change: Optional[float] = Field(None, description="Changement de taux en %")
    vix_multiplier: Optional[float] = Field(None, description="Multiplicateur de VIX")

    @validator('portfolio')
    def validate_portfolio(cls, v):
        if not v:
            raise ValueError("Portfolio ne peut pas être vide")
        if any(qty <= 0 for qty in v.values()):
            raise ValueError("Les quantités doivent être positives")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio": {"AAPL": 100, "MSFT": 50, "TSLA": 30},
                "stress_test_type": "market_crash",
                "shock_percent": -30
            }
        }


class HistoricalScenarioRequest(BaseModel):
    """Requête pour rejouer un événement historique"""
    portfolio: Dict[str, float] = Field(..., description="Portfolio à tester")
    event_name: str = Field(..., description="Nom de l'événement historique")

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio": {"AAPL": 100, "MSFT": 50},
                "event_name": "2008 Financial Crisis"
            }
        }


class CustomScenarioRequest(BaseModel):
    """Requête pour scénario personnalisé"""
    portfolio: Dict[str, float] = Field(..., description="Portfolio à tester")
    scenario_name: str = Field(..., description="Nom du scénario")
    parameters: Dict[str, Any] = Field(..., description="Paramètres du scénario")

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio": {"AAPL": 100},
                "scenario_name": "Fed Hawkish",
                "parameters": {
                    "interest_rate_change": 5.0,
                    "sector_impacts": {"Technology": -35}
                }
            }
        }


class MonteCarloRequest(BaseModel):
    """Requête pour simulation Monte Carlo"""
    portfolio: Dict[str, float] = Field(..., description="Portfolio à simuler")
    n_simulations: int = Field(10000, ge=1000, le=100000, description="Nombre de simulations")
    forecast_days: int = Field(252, ge=1, le=1260, description="Horizon en jours")
    use_ml: bool = Field(False, description="Utiliser ML pour générer les scénarios")

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio": {"AAPL": 100, "MSFT": 50},
                "n_simulations": 10000,
                "forecast_days": 252,
                "use_ml": True
            }
        }


class ScenarioComparisonRequest(BaseModel):
    """Requête pour comparer plusieurs scénarios"""
    portfolio: Dict[str, float] = Field(..., description="Portfolio à tester")
    scenario_ids: List[str] = Field(..., min_items=2, max_items=10, description="IDs des scénarios à comparer")


# ============================================================================
# RÉPONSES
# ============================================================================

class PositionImpact(BaseModel):
    """Impact sur une position individuelle"""
    ticker: str
    quantity: float
    price_before: float
    price_after: float
    value_before: float
    value_after: float
    impact_percent: float
    impact_amount: float


class RiskMetrics(BaseModel):
    """Métriques de risque calculées"""
    var_95: float = Field(..., description="Value at Risk 95%")
    cvar_95: float = Field(..., description="Conditional VaR 95%")
    var_99: Optional[float] = Field(None, description="Value at Risk 99%")
    cvar_99: Optional[float] = Field(None, description="Conditional VaR 99%")
    max_drawdown: float = Field(..., description="Drawdown maximum")
    stress_score: int = Field(..., ge=0, le=100, description="Score de résistance /100")
    recovery_time_days: Optional[int] = Field(None, description="Temps de récupération estimé")
    sharpe_ratio: Optional[float] = Field(None, description="Ratio de Sharpe")
    sortino_ratio: Optional[float] = Field(None, description="Ratio de Sortino")


class Recommendation(BaseModel):
    """Recommandation automatique"""
    type: str = Field(..., description="Type: hedging, diversification, position_sizing")
    priority: str = Field(..., description="Priority: high, medium, low")
    title: str = Field(..., description="Titre court")
    description: str = Field(..., description="Description détaillée")
    impact_reduction: Optional[float] = Field(None, description="Réduction d'impact estimée en %")
    suggested_action: Optional[Dict[str, Any]] = Field(None, description="Action suggérée")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "hedging",
                "priority": "high",
                "title": "Hedger l'exposition tech",
                "description": "Votre portfolio est surexposé au secteur technologique",
                "impact_reduction": 15.0,
                "suggested_action": {"buy": "SQQQ", "amount": 5000}
            }
        }


class ScenarioSimulationResult(BaseModel):
    """Résultat complet d'une simulation"""
    scenario_id: Optional[str] = None
    scenario_name: str
    scenario_type: str

    # Valeurs du portfolio
    portfolio_value_before: float
    portfolio_value_after: float
    impact_percent: float
    impact_amount: float

    # Impact par position
    position_impacts: List[PositionImpact]

    # Métriques de risque
    risk_metrics: RiskMetrics

    # Recommandations
    recommendations: List[Recommendation]

    # Détails additionnels
    worst_position: Optional[str] = None
    best_position: Optional[str] = None

    # Métadonnées
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "scenario_name": "Crash de Marché -30%",
                "scenario_type": "stress_test",
                "portfolio_value_before": 150000,
                "portfolio_value_after": 98000,
                "impact_percent": -34.7,
                "impact_amount": -52000,
                "position_impacts": [],
                "risk_metrics": {
                    "var_95": -38.2,
                    "cvar_95": -42.5,
                    "max_drawdown": -45.0,
                    "stress_score": 55
                },
                "recommendations": [],
                "execution_time_ms": 1234
            }
        }


class MonteCarloResult(BaseModel):
    """Résultat de simulation Monte Carlo"""
    n_simulations: int
    forecast_days: int
    initial_value: float

    # Statistiques finales
    mean_final_value: float
    median_final_value: float
    std_final_value: float
    min_final_value: float
    max_final_value: float

    # Returns
    mean_return: float
    median_return: float
    std_return: float
    min_return: float
    max_return: float

    # Percentiles
    percentiles: Dict[str, Dict[str, float]]  # {p5: {value: X, return: Y}, ...}

    # Risques
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float

    # Probabilités
    prob_profit: float = Field(..., description="Probabilité de profit en %")
    prob_loss_10pct: float = Field(..., description="Probabilité de perte >10%")
    prob_gain_20pct: float = Field(..., description="Probabilité de gain >20%")

    # Métriques additionnelles
    stress_score: int = Field(..., ge=0, le=100)
    recommendations: List[Recommendation] = []

    # Métadonnées
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "n_simulations": 10000,
                "forecast_days": 252,
                "initial_value": 100000,
                "mean_final_value": 108000,
                "median_final_value": 105000,
                "var_95": -15.5,
                "prob_profit": 65.0
            }
        }


class PredefinedScenario(BaseModel):
    """Scénario prédéfini disponible"""
    id: Optional[str] = None
    name: str
    description: str
    type: str
    parameters: Dict[str, Any]


class PredefinedScenariosResponse(BaseModel):
    """Liste des scénarios prédéfinis"""
    stress_tests: List[PredefinedScenario]
    historical_events: List[PredefinedScenario]
    total: int


class ScenarioComparisonResult(BaseModel):
    """Résultat de comparaison de scénarios"""
    portfolio_value_initial: float
    scenarios: List[ScenarioSimulationResult]
    worst_scenario: str
    best_scenario: str
    average_impact: float
    recommendations: List[Recommendation]


class HistoricalEventDetail(BaseModel):
    """Détails d'un événement historique"""
    id: str
    name: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    market_move_pct: float
    volatility_avg: float
    sector_impacts: Dict[str, float]
    macro_context: Dict[str, Any]
    triggers: List[str]
    recovery_pattern: str
    recovery_duration_days: int
    description: str


class HistoricalEventsResponse(BaseModel):
    """Liste des événements historiques disponibles"""
    events: List[HistoricalEventDetail]
    total: int
