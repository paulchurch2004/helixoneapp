"""
Modèles Portfolio - Tables pour l'analyse automatique de portefeuille
"""

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON, Integer, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.core.database import Base


def generate_uuid():
    """Génère un UUID unique"""
    return str(uuid.uuid4())


# Enums
class AnalysisTimeType(str, enum.Enum):
    """Type de timing d'analyse"""
    MORNING = "morning"      # 7h00 EST
    EVENING = "evening"      # 17h00 EST
    MANUAL = "manual"        # Manuel
    ON_DEMAND = "on_demand"  # À la demande


class AlertSeverity(str, enum.Enum):
    """Niveau de sévérité des alertes"""
    CRITICAL = "critical"      # Action immédiate requise
    WARNING = "warning"        # Attention nécessaire
    OPPORTUNITY = "opportunity"  # Opportunité d'achat/vente
    INFO = "info"             # Information générale


class RecommendationType(str, enum.Enum):
    """Type de recommandation"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class AlertStatus(str, enum.Enum):
    """Statut d'une alerte"""
    NEW = "new"              # Nouvelle alerte non lue
    READ = "read"            # Lue mais non traitée
    ACTED = "acted"          # Action effectuée
    DISMISSED = "dismissed"  # Ignorée/écartée
    ARCHIVED = "archived"    # Archivée


class PortfolioAnalysisHistory(Base):
    """
    Historique des analyses de portefeuille

    Stocke chaque analyse complète effectuée (matin/soir/manuel)
    avec tous les résultats et métriques.

    Attributs:
        id: Identifiant unique (UUID)
        user_id: ID de l'utilisateur
        analysis_time: Type de timing (morning/evening/manual)
        num_positions: Nombre de positions dans le portefeuille
        total_value: Valeur totale du portefeuille
        cash_amount: Montant en cash

        health_score: Score de santé global (0-100)
        portfolio_sentiment: Sentiment global (bullish/bearish/neutral)
        expected_return_7d: Return attendu 7 jours (%)
        downside_risk_pct: Risque de baisse (%)

        num_alerts: Nombre total d'alertes générées
        num_critical_alerts: Nombre d'alertes critiques
        num_recommendations: Nombre de recommandations

        execution_time_seconds: Temps d'exécution total (secondes)
        data_sources_used: Sources de données utilisées (JSON array)

        positions_data: Données détaillées par position (JSON)
        correlations_data: Données de corrélations (JSON)
        risks_data: Données de risques (JSON)
        predictions_data: Prédictions complètes (JSON)

        created_at: Date/heure de l'analyse

    Relations:
        alerts: Liste des alertes générées
        recommendations: Liste des recommandations
    """

    __tablename__ = "portfolio_analysis_history"

    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Informations de l'analyse
    analysis_time = Column(SQLEnum(AnalysisTimeType), nullable=False, index=True)
    num_positions = Column(Integer, nullable=False)
    total_value = Column(Float, nullable=True)
    cash_amount = Column(Float, nullable=True)

    # Résultats globaux
    health_score = Column(Float, nullable=False)  # 0-100
    portfolio_sentiment = Column(String, nullable=True)  # bullish/bearish/neutral
    expected_return_7d = Column(Float, nullable=True)  # %
    downside_risk_pct = Column(Float, nullable=True)  # %

    # Statistiques
    num_alerts = Column(Integer, default=0)
    num_critical_alerts = Column(Integer, default=0)
    num_recommendations = Column(Integer, default=0)

    # Performance
    execution_time_seconds = Column(Float, nullable=True)
    data_sources_used = Column(JSON, nullable=True)  # ["reddit", "stocktwits", ...]

    # Données détaillées (JSON)
    positions_data = Column(JSON, nullable=True)  # Analyse par position
    correlations_data = Column(JSON, nullable=True)  # Corrélations
    risks_data = Column(JSON, nullable=True)  # Risques détaillés
    predictions_data = Column(JSON, nullable=True)  # Prédictions complètes

    # Date
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relations
    alerts = relationship("PortfolioAlert", back_populates="analysis", cascade="all, delete-orphan")
    recommendations = relationship("PortfolioRecommendation", back_populates="analysis", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<PortfolioAnalysis {self.analysis_time} - Health: {self.health_score}>"


class PortfolioAlert(Base):
    """
    Alertes générées par l'analyse de portefeuille

    Chaque alerte représente une information importante à communiquer
    à l'utilisateur (critique, warning, opportunité, info).

    Attributs:
        id: Identifiant unique (UUID)
        analysis_id: ID de l'analyse parente
        user_id: ID de l'utilisateur (pour requêtes rapides)

        severity: Niveau de sévérité (critical/warning/opportunity/info)
        status: Statut (new/read/acted/dismissed/archived)

        ticker: Symbole concerné (optionnel, peut être global)
        title: Titre court de l'alerte
        message: Message détaillé (markdown supporté)
        action_required: Action suggérée

        confidence: Niveau de confiance (0-100)
        push_notification: Si notification push envoyée

        metadata: Données supplémentaires (JSON)

        created_at: Date de création
        read_at: Date de lecture
        acted_at: Date d'action

    Relations:
        analysis: Analyse parente
        performance: Performance si c'était une recommandation
    """

    __tablename__ = "portfolio_alerts"

    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("portfolio_analysis_history.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Caractéristiques de l'alerte
    severity = Column(SQLEnum(AlertSeverity), nullable=False, index=True)
    status = Column(SQLEnum(AlertStatus), default=AlertStatus.NEW, nullable=False, index=True)

    # Contenu
    ticker = Column(String, nullable=True, index=True)  # Peut être NULL pour alertes globales
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)  # Markdown supporté
    action_required = Column(Text, nullable=True)

    # Métriques
    confidence = Column(Float, nullable=True)  # 0-100
    push_notification = Column(String, default=False)  # Si push envoyé

    # Données supplémentaires
    extra_data = Column(JSON, nullable=True)  # Métadonnées additionnelles

    # Dates
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    read_at = Column(DateTime, nullable=True)
    acted_at = Column(DateTime, nullable=True)

    # Relations
    analysis = relationship("PortfolioAnalysisHistory", back_populates="alerts")

    def __repr__(self):
        return f"<PortfolioAlert {self.severity} - {self.title}>"


class PortfolioRecommendation(Base):
    """
    Recommandations d'achat/vente générées par l'analyse

    Chaque recommandation représente une suggestion d'action
    sur une position spécifique avec explications détaillées.

    Attributs:
        id: Identifiant unique (UUID)
        analysis_id: ID de l'analyse parente
        user_id: ID de l'utilisateur

        ticker: Symbole de l'action
        action: Type de recommandation (strong_buy/buy/hold/sell/strong_sell)
        confidence: Niveau de confiance (0-100)

        current_price: Prix actuel au moment de la recommandation
        target_price: Prix cible suggéré
        stop_loss: Stop loss suggéré

        primary_reason: Raison principale (court)
        detailed_reasons: Raisons détaillées (JSON array)
        risk_factors: Facteurs de risque (JSON array)
        suggested_action: Action suggérée détaillée

        prediction_1d: Prédiction 1 jour (%)
        prediction_3d: Prédiction 3 jours (%)
        prediction_7d: Prédiction 7 jours (%)

        sentiment_score: Score de sentiment (-100 à +100)
        technical_score: Score technique (0-100)
        fundamental_score: Score fondamental (0-100)

        metadata: Données supplémentaires (JSON)

        created_at: Date de création
        expires_at: Date d'expiration (recommandations ont durée de vie limitée)

    Relations:
        analysis: Analyse parente
        performance: Performance de cette recommandation (si suivie)
    """

    __tablename__ = "portfolio_recommendations"

    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("portfolio_analysis_history.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Recommandation
    ticker = Column(String, nullable=False, index=True)
    action = Column(SQLEnum(RecommendationType), nullable=False, index=True)
    confidence = Column(Float, nullable=False)  # 0-100

    # Prix
    current_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)

    # Explications
    primary_reason = Column(Text, nullable=False)
    detailed_reasons = Column(JSON, nullable=True)  # Array de strings
    risk_factors = Column(JSON, nullable=True)  # Array de strings
    suggested_action = Column(Text, nullable=True)

    # Prédictions
    prediction_1d = Column(Float, nullable=True)  # %
    prediction_3d = Column(Float, nullable=True)  # %
    prediction_7d = Column(Float, nullable=True)  # %

    # Scores
    sentiment_score = Column(Float, nullable=True)  # -100 à +100
    technical_score = Column(Float, nullable=True)  # 0-100
    fundamental_score = Column(Float, nullable=True)  # 0-100

    # Données supplémentaires
    extra_data = Column(JSON, nullable=True)  # Métadonnées additionnelles

    # Dates
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=True)  # Recommandation valide jusqu'à

    # Relations
    analysis = relationship("PortfolioAnalysisHistory", back_populates="recommendations")
    performance = relationship("RecommendationPerformance", back_populates="recommendation", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<PortfolioRecommendation {self.ticker} - {self.action} ({self.confidence}%)>"


class RecommendationPerformance(Base):
    """
    Performance des recommandations (tracking)

    Permet de mesurer la précision des recommandations au fil du temps
    pour améliorer le système (machine learning).

    Attributs:
        id: Identifiant unique (UUID)
        recommendation_id: ID de la recommandation
        user_id: ID de l'utilisateur

        ticker: Symbole de l'action
        recommended_action: Action recommandée
        recommendation_confidence: Confiance de la recommandation

        price_at_recommendation: Prix au moment de la recommandation
        target_price: Prix cible suggéré

        price_after_1d: Prix après 1 jour
        price_after_3d: Prix après 3 jours
        price_after_7d: Prix après 7 jours

        actual_change_1d: Variation réelle 1j (%)
        actual_change_3d: Variation réelle 3j (%)
        actual_change_7d: Variation réelle 7j (%)

        predicted_change_1d: Variation prédite 1j (%)
        predicted_change_3d: Variation prédite 3j (%)
        predicted_change_7d: Variation prédite 7j (%)

        accuracy_1d: Précision 1j (0-100)
        accuracy_3d: Précision 3j (0-100)
        accuracy_7d: Précision 7j (0-100)

        user_followed: Si l'utilisateur a suivi la recommandation
        user_action: Action effectuée par l'utilisateur
        user_notes: Notes de l'utilisateur

        created_at: Date de création
        updated_at: Dernière mise à jour

    Relations:
        recommendation: Recommandation parente
    """

    __tablename__ = "recommendation_performance"

    # Colonnes principales
    id = Column(String, primary_key=True, default=generate_uuid)
    recommendation_id = Column(String, ForeignKey("portfolio_recommendations.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Informations de base
    ticker = Column(String, nullable=False, index=True)
    recommended_action = Column(SQLEnum(RecommendationType), nullable=False)
    recommendation_confidence = Column(Float, nullable=False)

    # Prix
    price_at_recommendation = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)

    # Prix réels mesurés
    price_after_1d = Column(Float, nullable=True)
    price_after_3d = Column(Float, nullable=True)
    price_after_7d = Column(Float, nullable=True)

    # Variations réelles
    actual_change_1d = Column(Float, nullable=True)  # %
    actual_change_3d = Column(Float, nullable=True)  # %
    actual_change_7d = Column(Float, nullable=True)  # %

    # Variations prédites (copie depuis recommendation)
    predicted_change_1d = Column(Float, nullable=True)  # %
    predicted_change_3d = Column(Float, nullable=True)  # %
    predicted_change_7d = Column(Float, nullable=True)  # %

    # Précision de la prédiction
    accuracy_1d = Column(Float, nullable=True)  # 0-100
    accuracy_3d = Column(Float, nullable=True)  # 0-100
    accuracy_7d = Column(Float, nullable=True)  # 0-100

    # Action utilisateur
    user_followed = Column(String, default=False)
    user_action = Column(String, nullable=True)  # "bought", "sold", "held", "ignored"
    user_notes = Column(Text, nullable=True)

    # Dates
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relations
    recommendation = relationship("PortfolioRecommendation", back_populates="performance")

    def __repr__(self):
        return f"<RecommendationPerformance {self.ticker} - Accuracy 7d: {self.accuracy_7d}%>"
