"""
Modèles pour le stockage des données d'impact des événements économiques

Tables:
- event_impact_history: Historique des impacts réels des événements
- event_predictions: Prédictions d'impact futures
- sector_event_correlations: Corrélations secteur ↔ événement
"""

from sqlalchemy import Column, String, Float, DateTime, Integer, Text, JSON, Index, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.core.database import Base


def generate_uuid():
    """Génère un UUID unique"""
    return str(uuid.uuid4())


# ============================================================================
# TABLES
# ============================================================================

class EventImpactHistory(Base):
    """
    Historique des impacts réels des événements économiques sur les actions

    Cette table permet d'apprendre des événements passés pour améliorer
    les prédictions futures (machine learning).

    Attributs:
        id: Identifiant unique
        event_id: ID de l'événement (FK vers economic_events)
        event_type: Type d'événement ('fed_meeting', 'cpi', 'earnings', etc.)
        event_date: Date de l'événement

        ticker: Symbole de l'action impactée
        sector: Secteur de l'action

        price_before: Prix avant l'événement (1h avant pour intraday)
        price_1d_after: Prix 1 jour après
        price_3d_after: Prix 3 jours après
        price_7d_after: Prix 7 jours après

        impact_1d_pct: Impact 1 jour (%)
        impact_3d_pct: Impact 3 jours (%)
        impact_7d_pct: Impact 7 jours (%)

        volume_change_pct: Variation du volume (%)
        volatility_increase: Augmentation de volatilité
        direction: Direction du mouvement ('up', 'down', 'neutral')

        event_actual_value: Valeur réelle de l'indicateur (si applicable)
        event_forecast_value: Valeur prévue
        surprise_pct: Surprise vs forecast (%)

        created_at: Date d'enregistrement

    Relations:
        economic_event: Événement économique associé
    """

    __tablename__ = "event_impact_history"

    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)

    # Événement
    event_id = Column(String, ForeignKey("economic_events.id", ondelete="CASCADE"), nullable=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # 'fed_meeting', 'cpi', etc.
    event_date = Column(DateTime, nullable=False, index=True)
    event_impact_level = Column(String(20))  # 'low', 'medium', 'high', 'critical'

    # Action impactée
    ticker = Column(String(20), nullable=False, index=True)
    sector = Column(String(100), nullable=False, index=True)
    market_cap = Column(Float)  # Capitalisation au moment de l'événement

    # Prix (stockés pour calculs futurs)
    price_before = Column(Float, nullable=False)
    price_1d_after = Column(Float)
    price_3d_after = Column(Float)
    price_7d_after = Column(Float)

    # Impacts calculés (%)
    impact_1d_pct = Column(Float)
    impact_3d_pct = Column(Float)
    impact_7d_pct = Column(Float)

    # Métriques supplémentaires
    volume_change_pct = Column(Float)  # Variation volume vs moyenne
    volatility_increase = Column(Float)  # Augmentation volatilité
    direction = Column(String(10))  # 'up', 'down', 'neutral'

    # Données de l'événement
    event_actual_value = Column(Float)  # Valeur réelle (ex: CPI = 3.2%)
    event_forecast_value = Column(Float)  # Valeur prévue
    surprise_pct = Column(Float)  # (actual - forecast) / forecast * 100

    # Contexte macro
    market_sentiment = Column(String(20))  # 'bullish', 'bearish', 'neutral' au moment de l'événement
    vix_level = Column(Float)  # Niveau VIX au moment de l'événement

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    data_quality = Column(Float, default=1.0)  # 0-1, qualité des données

    # Index composés pour requêtes rapides
    __table_args__ = (
        Index('idx_impact_history_event_type_sector', 'event_type', 'sector'),
        Index('idx_impact_history_ticker_event_date', 'ticker', 'event_date'),
        Index('idx_impact_history_event_type_date', 'event_type', 'event_date'),
    )

    def __repr__(self):
        return f"<EventImpactHistory {self.ticker} {self.event_type} {self.event_date}: {self.impact_1d_pct}%>"


class EventPrediction(Base):
    """
    Prédictions d'impact futur des événements à venir

    Générées par le moteur de prédiction, ces prédictions sont ensuite
    comparées à la réalité pour mesurer la précision du modèle.

    Attributs:
        id: Identifiant unique
        event_id: ID de l'événement futur (FK vers economic_events)
        prediction_date: Date de génération de la prédiction
        event_date: Date de l'événement

        ticker: Symbole de l'action
        sector: Secteur

        predicted_impact_1d_pct: Impact prédit 1j (%)
        predicted_impact_3d_pct: Impact prédit 3j (%)
        predicted_impact_7d_pct: Impact prédit 7j (%)

        confidence: Confiance dans la prédiction (0-100)
        direction: Direction prédite ('bullish', 'bearish', 'neutral')
        probability_up: Probabilité de hausse (%)
        probability_down: Probabilité de baisse (%)

        model_used: Modèle utilisé ('empirical', 'ml_v1', etc.)
        model_version: Version du modèle
        features_used: Features utilisées (JSON)

        prediction_factors: Facteurs de la prédiction (JSON array)
        sector_correlation: Corrélation secteur utilisée

        actual_impact_1d: Impact réel 1j (rempli après événement)
        actual_impact_3d: Impact réel 3j
        actual_impact_7d: Impact réel 7j

        accuracy_1d: Précision 1j (0-100, calculé après)
        accuracy_3d: Précision 3j
        accuracy_7d: Précision 7j

        user_id: ID utilisateur (si prédiction personnalisée)
        analysis_id: ID analyse portfolio associée

        created_at: Date de création
        updated_at: Dernière mise à jour (après événement)
    """

    __tablename__ = "event_predictions"

    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)

    # Événement
    event_id = Column(String, ForeignKey("economic_events.id", ondelete="CASCADE"), nullable=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    event_date = Column(DateTime, nullable=False, index=True)

    # Action concernée
    ticker = Column(String(20), nullable=False, index=True)
    sector = Column(String(100), nullable=False)

    # Prédictions
    predicted_impact_1d_pct = Column(Float, nullable=False)
    predicted_impact_3d_pct = Column(Float)
    predicted_impact_7d_pct = Column(Float)

    # Confiance
    confidence = Column(Float, nullable=False)  # 0-100
    direction = Column(String(10), nullable=False)  # 'bullish', 'bearish', 'neutral'
    probability_up = Column(Float)  # 0-100
    probability_down = Column(Float)  # 0-100

    # Modèle
    model_used = Column(String(50), default='empirical')  # Type de modèle
    model_version = Column(String(20), default='1.0')
    features_used = Column(JSON)  # Features du modèle ML

    # Explications
    prediction_factors = Column(JSON)  # Array de strings (raisons)
    sector_correlation = Column(Float)  # Corrélation secteur utilisée

    # Impacts réels (remplis après l'événement)
    actual_impact_1d = Column(Float)
    actual_impact_3d = Column(Float)
    actual_impact_7d = Column(Float)

    # Précision (calculée après événement)
    accuracy_1d = Column(Float)  # 0-100
    accuracy_3d = Column(Float)
    accuracy_7d = Column(Float)

    # Relations user/analysis
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    analysis_id = Column(String, nullable=True)  # ID de l'analyse portfolio

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Index composés
    __table_args__ = (
        Index('idx_ticker_event_date_pred', 'ticker', 'event_date'),
        Index('idx_user_event_date', 'user_id', 'event_date'),
        Index('idx_event_type_sector', 'event_type', 'sector'),
    )

    def __repr__(self):
        return f"<EventPrediction {self.ticker} {self.event_type} {self.predicted_impact_1d_pct}% conf:{self.confidence}>"


class SectorEventCorrelation(Base):
    """
    Corrélations entre types d'événements et secteurs

    Table pré-calculée pour accès rapide aux corrélations.
    Mise à jour périodiquement en analysant event_impact_history.

    Attributs:
        id: Identifiant unique
        event_type: Type d'événement
        sector: Secteur

        avg_impact_1d_pct: Impact moyen 1j (%)
        avg_impact_3d_pct: Impact moyen 3j (%)
        avg_impact_7d_pct: Impact moyen 7j (%)

        std_dev_1d: Écart-type 1j
        std_dev_3d: Écart-type 3j
        std_dev_7d: Écart-type 7j

        confidence: Confiance dans la corrélation (0-100)
        sample_size: Nombre d'événements analysés

        trend: Tendance générale ('positive', 'negative', 'neutral')
        strength: Force de la corrélation ('weak', 'moderate', 'strong')

        min_impact: Impact minimum observé
        max_impact: Impact maximum observé

        last_updated: Dernière mise à jour de ces stats
        data_quality: Qualité des données (0-1)
    """

    __tablename__ = "sector_event_correlations"

    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)

    # Corrélation
    event_type = Column(String(50), nullable=False, index=True)
    sector = Column(String(100), nullable=False, index=True)

    # Impacts moyens (%)
    avg_impact_1d_pct = Column(Float, nullable=False)
    avg_impact_3d_pct = Column(Float)
    avg_impact_7d_pct = Column(Float)

    # Écarts-types
    std_dev_1d = Column(Float, nullable=False)
    std_dev_3d = Column(Float)
    std_dev_7d = Column(Float)

    # Métriques
    confidence = Column(Float, nullable=False)  # 0-100, basé sur sample_size
    sample_size = Column(Integer, nullable=False)  # Nombre d'événements

    # Classification
    trend = Column(String(20))  # 'positive', 'negative', 'neutral'
    strength = Column(String(20))  # 'weak', 'moderate', 'strong'

    # Extrêmes
    min_impact = Column(Float)
    max_impact = Column(Float)

    # Contexte
    typical_direction = Column(String(10))  # Direction la plus fréquente
    volatility_increase_avg = Column(Float)  # Augmentation moyenne de volatilité

    # Métadonnées
    last_updated = Column(DateTime, default=datetime.utcnow, index=True)
    data_quality = Column(Float, default=1.0)  # 0-1

    # Index unique pour éviter doublons
    __table_args__ = (
        Index('idx_unique_event_sector', 'event_type', 'sector', unique=True),
    )

    def __repr__(self):
        return f"<SectorCorrelation {self.event_type} → {self.sector}: {self.avg_impact_1d_pct}% (n={self.sample_size})>"


class EventAlert(Base):
    """
    Alertes préventives avant événements critiques

    Générées automatiquement pour prévenir les utilisateurs des événements
    à venir qui peuvent impacter leur portefeuille.

    Attributs:
        id: Identifiant unique
        user_id: ID utilisateur
        event_id: ID événement (FK)

        alert_type: Type d'alerte ('pre_event', 'post_event')
        severity: Sévérité ('low', 'medium', 'high', 'critical')

        title: Titre de l'alerte
        message: Message détaillé (markdown)
        recommendations: Recommandations d'action (JSON array)

        affected_tickers: Tickers concernés du portfolio (JSON array)
        predicted_impacts: Impacts prédits (JSON)

        days_until_event: Jours avant événement
        is_read: Si l'alerte a été lue
        is_dismissed: Si l'alerte a été ignorée

        created_at: Date de création
        read_at: Date de lecture
    """

    __tablename__ = "event_alerts"

    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    event_id = Column(String, ForeignKey("economic_events.id", ondelete="CASCADE"), nullable=True, index=True)

    # Alerte
    alert_type = Column(String(20), nullable=False)  # 'pre_event', 'post_event'
    severity = Column(String(20), nullable=False, index=True)  # 'low', 'medium', 'high', 'critical'

    # Contenu
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)  # Markdown
    recommendations = Column(JSON)  # Array de strings

    # Impact portfolio
    affected_tickers = Column(JSON)  # Array de tickers
    predicted_impacts = Column(JSON)  # {ticker: predicted_impact_pct}

    # Timing
    days_until_event = Column(Integer)  # Jours avant événement

    # Statut
    is_read = Column(Boolean, default=False, index=True)
    is_dismissed = Column(Boolean, default=False)

    # Dates
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    read_at = Column(DateTime)

    # Index
    __table_args__ = (
        Index('idx_user_severity', 'user_id', 'severity'),
        Index('idx_user_created', 'user_id', 'created_at'),
    )

    def __repr__(self):
        return f"<EventAlert {self.severity} - {self.title} (User: {self.user_id})>"
