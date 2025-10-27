"""
Modèles de données pour le Moteur de Simulation de Scénarios
Inspiré de BlackRock Aladdin - système de stress testing et Monte Carlo
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime
import enum


class ScenarioType(str, enum.Enum):
    """Types de scénarios disponibles"""
    STRESS_TEST = "stress_test"              # Test de résistance standard
    HISTORICAL = "historical"                # Rejeu de crise historique
    MACRO_ECONOMIC = "macro_economic"        # Choc macroéconomique
    SECTORAL = "sectoral"                    # Choc sectoriel
    COMPOSITE = "composite"                  # Combinaison de plusieurs chocs
    ML_GENERATED = "ml_generated"            # Généré par ML
    CUSTOM = "custom"                        # Personnalisé par utilisateur


class RecoveryPattern(str, enum.Enum):
    """Patterns de récupération après une crise"""
    V_SHAPED = "V_shaped"      # Chute rapide, rebond rapide
    U_SHAPED = "U_shaped"      # Chute, plateau, rebond
    L_SHAPED = "L_shaped"      # Chute, pas de rebond
    W_SHAPED = "W_shaped"      # Double dip
    NIKE_SHAPED = "Nike_shaped"  # Chute, rebond lent


class StressTestType(str, enum.Enum):
    """Types de stress tests standards"""
    MARKET_CRASH = "market_crash"           # Crash de marché général
    RATE_SHOCK = "rate_shock"               # Choc de taux d'intérêt
    VOLATILITY_SPIKE = "volatility_spike"   # Spike de volatilité (VIX)
    LIQUIDITY_CRISIS = "liquidity_crisis"   # Crise de liquidité
    INFLATION_SHOCK = "inflation_shock"     # Choc d'inflation


class Scenario(Base):
    """
    Définition d'un scénario de simulation
    """
    __tablename__ = "scenarios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    type = Column(SQLEnum(ScenarioType), nullable=False, index=True)

    # Paramètres du scénario (JSON flexible)
    parameters = Column(JSONB, nullable=False, default={})

    # Métadonnées
    is_predefined = Column(Boolean, default=False)  # Scénario prédéfini du système
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ML related
    ml_model_version = Column(String(50))  # Si généré par ML
    historical_event_id = Column(UUID(as_uuid=True), ForeignKey("historical_events.id"))

    # Relations
    historical_event = relationship("HistoricalEvent", back_populates="scenarios")
    simulations = relationship("ScenarioSimulation", back_populates="scenario", cascade="all, delete-orphan")


class HistoricalEvent(Base):
    """
    Crises historiques documentées (2008, COVID, etc.)
    Utilisées pour rejouer des scénarios historiques
    """
    __tablename__ = "historical_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, unique=True, index=True)

    # Dates
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)

    # Impact sur le marché
    market_move_pct = Column(Float, nullable=False)  # % de mouvement du S&P 500
    volatility_avg = Column(Float)  # VIX moyen pendant la crise

    # Impact sectoriel (JSON)
    sector_impacts = Column(JSONB, default={})  # {"Technology": -0.45, "Financials": -0.82, ...}

    # Contexte macro-économique
    macro_context = Column(JSONB, default={})  # {"interest_rate_start": 0.05, "gdp_growth": -0.03, ...}

    # Triggers et caractéristiques
    triggers = Column(JSONB, default=[])  # Liste de triggers: ["housing_bubble", "lehman_bankruptcy"]
    recovery_pattern = Column(SQLEnum(RecoveryPattern))
    recovery_duration_days = Column(Integer)  # Jours pour récupérer

    # Métadonnées
    description = Column(Text)
    extracted_at = Column(DateTime, default=datetime.utcnow)
    data_sources = Column(JSONB, default=[])  # Sources de données utilisées

    # Relations
    scenarios = relationship("Scenario", back_populates="historical_event")


class ScenarioSimulation(Base):
    """
    Résultat d'une simulation de scénario sur un portfolio
    """
    __tablename__ = "scenario_simulations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_id = Column(UUID(as_uuid=True), ForeignKey("scenarios.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Snapshot du portfolio au moment de la simulation
    portfolio_snapshot = Column(JSONB, nullable=False)  # {"AAPL": 100, "MSFT": 50, ...}

    # Résultats de la simulation
    results = Column(JSONB, nullable=False)  # Impact sur chaque position

    # Métriques de risque calculées
    metrics = Column(JSONB, nullable=False)  # VaR, CVaR, Stress Score, etc.

    # Recommandations générées
    recommendations = Column(JSONB, default=[])  # Liste de recommandations

    # Performance
    execution_time_ms = Column(Integer)  # Temps d'exécution

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    scenario = relationship("Scenario", back_populates="simulations")
    user = relationship("User")


class MLModel(Base):
    """
    Modèle de Machine Learning pour génération/prédiction de scénarios
    """
    __tablename__ = "ml_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, index=True)
    type = Column(String(50), nullable=False)  # classifier, regressor, generator, gan
    version = Column(String(20), nullable=False, index=True)

    # Training info
    trained_at = Column(DateTime, default=datetime.utcnow)
    training_data_size = Column(Integer)  # Nombre d'échantillons
    training_duration_sec = Column(Float)

    # Métriques de performance
    accuracy_metrics = Column(JSONB, default={})  # {"accuracy": 0.85, "r2": 0.75, ...}

    # Fichiers
    model_file_path = Column(String(500))  # Chemin vers le .pkl ou .h5
    config_file_path = Column(String(500))

    # Status
    is_active = Column(Boolean, default=True)  # Modèle actif (en production)
    description = Column(Text)

    # Métadonnées
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relations
    predictions = relationship("MLPrediction", back_populates="model")


class MLPrediction(Base):
    """
    Prédiction individuelle faite par un modèle ML
    """
    __tablename__ = "ml_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("ml_models.id"), nullable=False)

    # Input et output
    input_data = Column(JSONB, nullable=False)  # Données d'entrée
    prediction = Column(JSONB, nullable=False)  # Prédiction du modèle
    confidence = Column(Float)  # Niveau de confiance (0-1)

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    execution_time_ms = Column(Integer)

    # Relations
    model = relationship("MLModel", back_populates="predictions")


class ScenarioBacktest(Base):
    """
    Résultat de backtesting d'un scénario
    Pour valider la précision des prédictions
    """
    __tablename__ = "scenario_backtests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_id = Column(UUID(as_uuid=True), ForeignKey("scenarios.id"), nullable=False)

    # Période de test
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)

    # Résultats
    predicted_values = Column(JSONB, nullable=False)  # Valeurs prédites
    actual_values = Column(JSONB, nullable=False)  # Valeurs réelles

    # Métriques d'erreur
    metrics = Column(JSONB, default={})  # {"mape": 0.05, "rmse": 0.03, ...}

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))


# ============================================================================
# DONNÉES PRÉDÉFINIES - Crises Historiques
# ============================================================================

PREDEFINED_HISTORICAL_EVENTS = [
    {
        "name": "2008 Financial Crisis",
        "start_date": "2007-10-09",
        "end_date": "2009-03-09",
        "market_move_pct": -56.7,
        "volatility_avg": 32.5,
        "sector_impacts": {
            "Financials": -82.0,
            "Energy": -54.0,
            "Technology": -45.0,
            "Consumer Cyclical": -48.0,
            "Consumer Defensive": -18.0,
            "Healthcare": -23.0,
            "Utilities": -31.0
        },
        "macro_context": {
            "interest_rate_start": 4.75,
            "interest_rate_end": 0.25,
            "unemployment_start": 4.7,
            "unemployment_end": 8.6,
            "gdp_growth": -2.8
        },
        "triggers": ["housing_bubble", "lehman_bankruptcy", "credit_freeze"],
        "recovery_pattern": "V_shaped",
        "recovery_duration_days": 365,
        "description": "Crise financière mondiale déclenchée par l'effondrement du marché immobilier américain"
    },
    {
        "name": "COVID-19 Crash 2020",
        "start_date": "2020-02-19",
        "end_date": "2020-03-23",
        "market_move_pct": -33.9,
        "volatility_avg": 57.0,
        "sector_impacts": {
            "Energy": -65.0,
            "Financials": -45.0,
            "Consumer Cyclical": -42.0,
            "Industrials": -40.0,
            "Technology": -25.0,
            "Healthcare": -20.0,
            "Consumer Defensive": -15.0
        },
        "macro_context": {
            "interest_rate_start": 1.75,
            "interest_rate_end": 0.25,
            "unemployment_start": 3.5,
            "unemployment_end": 14.7
        },
        "triggers": ["pandemic", "lockdowns", "economic_shutdown"],
        "recovery_pattern": "Nike_shaped",
        "recovery_duration_days": 150,
        "description": "Crash éclair causé par la pandémie COVID-19 et les confinements mondiaux"
    },
    {
        "name": "Dot-com Bubble 2000",
        "start_date": "2000-03-10",
        "end_date": "2002-10-09",
        "market_move_pct": -49.1,
        "volatility_avg": 26.0,
        "sector_impacts": {
            "Technology": -78.0,
            "Communication Services": -60.0,
            "Consumer Cyclical": -35.0,
            "Financials": -30.0,
            "Consumer Defensive": -5.0,
            "Utilities": -10.0
        },
        "triggers": ["tech_bubble", "overvaluation", "earnings_miss"],
        "recovery_pattern": "U_shaped",
        "recovery_duration_days": 1800,
        "description": "Éclatement de la bulle internet avec effondrement des valeurs tech"
    }
]

PREDEFINED_STRESS_TESTS = [
    {
        "name": "Crash de Marché -20%",
        "type": "MARKET_CRASH",
        "parameters": {"shock_percent": -20}
    },
    {
        "name": "Crash de Marché -30%",
        "type": "MARKET_CRASH",
        "parameters": {"shock_percent": -30}
    },
    {
        "name": "Hausse Taux +2%",
        "type": "RATE_SHOCK",
        "parameters": {"rate_change": 2.0}
    },
    {
        "name": "Spike Volatilité VIX x3",
        "type": "VOLATILITY_SPIKE",
        "parameters": {"vix_multiplier": 3.0}
    }
]
