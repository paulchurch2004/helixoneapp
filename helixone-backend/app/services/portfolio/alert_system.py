"""
Alert System - Génère des alertes formatées et priorisées
Transforme les analyses et recommandations en alertes exploitables

Types d'alertes:
- CRITICAL: Action immédiate requise (STRONG_SELL)
- WARNING: Attention nécessaire (SELL, risques élevés)
- OPPORTUNITY: Occasion d'achat (BUY, STRONG_BUY)
- INFO: Informations générales (HOLD, mises à jour)
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from app.services.portfolio.portfolio_analyzer import PortfolioAnalysisResult
from app.services.portfolio.ml_signal_service import MLPortfolioSignals
from app.services.portfolio.recommendation_engine import (
    PortfolioRecommendations,
    Recommendation,
    RecommendationType,
    ActionPriority
)

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Niveau de sévérité d'une alerte"""
    CRITICAL = "critical"  # 🔴 Action immédiate
    WARNING = "warning"  # ⚠️ Attention requise
    OPPORTUNITY = "opportunity"  # 💡 Opportunité
    INFO = "info"  # ℹ️ Information


@dataclass
class Alert:
    """Alerte individuelle"""
    id: str
    ticker: Optional[str]  # None pour alertes portfolio
    severity: AlertSeverity
    priority: ActionPriority

    # Contenu
    title: str  # Titre court
    message: str  # Message formaté complet (markdown)
    summary: str  # Résumé en une ligne

    # Action recommandée
    action_required: str  # Description de l'action
    action_button_text: Optional[str] = None  # Texte du bouton d'action

    # Données structurées (pour UI)
    recommendation: Optional[RecommendationType] = None
    confidence: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

    # Métadonnées
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    read: bool = False

    # Pour notifications push
    push_notification: bool = True
    push_title: Optional[str] = None
    push_body: Optional[str] = None


@dataclass
class AlertBatch:
    """Lot d'alertes pour un cycle d'analyse"""
    batch_id: str
    analysis_time: str  # "morning" ou "evening"

    # Alertes par catégorie
    critical_alerts: List[Alert]
    warning_alerts: List[Alert]
    opportunity_alerts: List[Alert]
    info_alerts: List[Alert]

    # Résumé global
    summary: str
    total_alerts: int

    # Performance prévue
    expected_return_7d: float  # % attendu sur 7 jours

    generated_at: datetime = None


class AlertSystem:
    """
    Système de génération d'alertes
    Transforme analyses et recommandations en alertes exploitables
    """

    def __init__(self):
        logger.info("AlertSystem initialisé")

    def generate_alerts(
        self,
        analysis: PortfolioAnalysisResult,
        ml_predictions: MLPortfolioSignals,
        recommendations: PortfolioRecommendations,
        analysis_time: str = "morning"
    ) -> AlertBatch:
        """
        Génère toutes les alertes pour un cycle d'analyse

        Args:
            analysis: Résultat de l'analyse
            predictions: Prédictions futures
            recommendations: Recommandations
            analysis_time: "morning" ou "evening"

        Returns:
            AlertBatch avec toutes les alertes
        """
        logger.info(f"🔔 Génération alertes {analysis_time}")

        batch_id = f"alerts_{int(datetime.now().timestamp())}"

        critical = []
        warnings = []
        opportunities = []
        infos = []

        # 1. Alertes des recommandations critiques
        for rec in recommendations.critical_actions:
            alert = self._create_recommendation_alert(rec, analysis, predictions)
            critical.append(alert)

        # 2. Alertes des recommandations high priority
        for rec in recommendations.high_priority_actions:
            if rec.action in [RecommendationType.SELL, RecommendationType.STRONG_SELL]:
                warnings.append(self._create_recommendation_alert(rec, analysis, predictions))
            elif rec.action in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                opportunities.append(self._create_recommendation_alert(rec, analysis, predictions))

        # 3. Alertes des positions à surveiller (basées sur prédictions ML bearish)
        for ticker, pred in ml_predictions.predictions.items():
            if pred.signal == "SELL" and ticker not in [r.ticker for r in recommendations.critical_actions]:
                alert = self._create_watch_alert(ticker, analysis, ml_predictions)
                warnings.append(alert)

        # 4. Alertes des opportunités
        for opp in recommendations.new_opportunities[:3]:  # Top 3 opportunités
            alert = self._create_opportunity_alert(opp)
            opportunities.append(alert)

        # 5. Alertes de portefeuille
        for action in recommendations.portfolio_actions:
            alert = self._create_portfolio_alert(action)
            infos.append(alert)

        # 6. Alerte de performance attendue (basée sur signal ML moyen)
        # Calculer un expected_return approximatif depuis les prédictions ML
        bullish_pct = (ml_predictions.bullish_count / max(len(ml_predictions.predictions), 1)) * 100
        bearish_pct = (ml_predictions.bearish_count / max(len(ml_predictions.predictions), 1)) * 100
        expected_return_approx = (bullish_pct - bearish_pct) / 10  # Conversion approximative en %

        if abs(expected_return_approx) > 5:
            alert = self._create_performance_alert(ml_predictions)
            if expected_return_approx > 0:
                opportunities.append(alert)
            else:
                warnings.append(alert)

        # 7. Alerte de résumé global
        summary_alert = self._create_summary_alert(
            analysis, ml_predictions, recommendations, analysis_time
        )
        infos.insert(0, summary_alert)  # Toujours en premier

        # Générer résumé du batch
        total = len(critical) + len(warnings) + len(opportunities) + len(infos)

        if critical:
            summary = f"🚨 {len(critical)} ACTIONS CRITIQUES requises"
        elif warnings:
            summary = f"⚠️ {len(warnings)} alertes nécessitent attention"
        elif opportunities:
            summary = f"💡 {len(opportunities)} opportunités détectées"
        else:
            summary = "✅ Portefeuille stable, aucune action urgente"

        batch = AlertBatch(
            batch_id=batch_id,
            analysis_time=analysis_time,
            critical_alerts=critical,
            warning_alerts=warnings,
            opportunity_alerts=opportunities,
            info_alerts=infos,
            summary=summary,
            total_alerts=total,
            expected_return_7d=expected_return_approx,  # Calculé plus haut
            generated_at=datetime.now()
        )

        logger.info(
            f"✅ {total} alertes générées: "
            f"{len(critical)} critiques, {len(warnings)} warnings, "
            f"{len(opportunities)} opportunités, {len(infos)} infos"
        )

        return batch

    # ========================================================================
    # CRÉATION D'ALERTES SPÉCIFIQUES
    # ========================================================================

    def _create_recommendation_alert(
        self,
        rec: Recommendation,
        analysis: PortfolioAnalysisResult,
        ml_predictions: MLPortfolioSignals
    ) -> Alert:
        """Crée une alerte pour une recommandation"""

        # Déterminer sévérité
        if rec.action == RecommendationType.STRONG_SELL:
            severity = AlertSeverity.CRITICAL
            icon = "🔴"
            action_button = "VENDRE MAINTENANT"
        elif rec.action == RecommendationType.SELL:
            severity = AlertSeverity.WARNING
            icon = "⚠️"
            action_button = "RÉDUIRE POSITION"
        elif rec.action == RecommendationType.STRONG_BUY:
            severity = AlertSeverity.OPPORTUNITY
            icon = "🟢"
            action_button = "RENFORCER POSITION"
        elif rec.action == RecommendationType.BUY:
            severity = AlertSeverity.OPPORTUNITY
            icon = "💡"
            action_button = "ACHETER PLUS"
        else:  # HOLD
            severity = AlertSeverity.INFO
            icon = "ℹ️"
            action_button = None

        # Titre
        title = f"{icon} {rec.ticker} - {rec.action.value.replace('_', ' ')}"

        # Message formaté (markdown)
        message_parts = [
            f"## {icon} {rec.ticker} - Recommandation : {rec.action.value.replace('_', ' ')}",
            f"**Confiance :** {rec.confidence:.0f}%",
            "",
            f"### 📋 Raison principale",
            rec.primary_reason,
            "",
        ]

        if rec.detailed_reasons:
            message_parts.append("### 📊 Analyse détaillée")
            for reason in rec.detailed_reasons:
                message_parts.append(f"- {reason}")
            message_parts.append("")

        # Prédiction
        pred = ml_predictions.predictions.get(rec.ticker)
        if pred:
            message_parts.extend([
                "### 🔮 Prédiction (7 jours)",
                f"- **Direction :** {pred.overall_direction.capitalize()}",
                f"- **Mouvement attendu :** {pred.prediction_7d.expected_move_pct:+.1f}%",
                f"- **Probabilité hausse :** {pred.prediction_7d.probability_up:.0f}%",
                f"- **Probabilité baisse :** {pred.prediction_7d.probability_down:.0f}%",
                ""
            ])

        # Prix cibles
        if rec.target_price or rec.stop_loss:
            message_parts.append("### 🎯 Niveaux de prix")
            if rec.target_price:
                message_parts.append(f"- **Prix cible :** ${rec.target_price:.2f}")
            if rec.stop_loss:
                message_parts.append(f"- **Stop loss :** ${rec.stop_loss:.2f}")
            message_parts.append("")

        # Action suggérée
        message_parts.extend([
            "### 💡 Action suggérée",
            rec.suggested_action,
        ])

        if rec.quantity_suggestion:
            message_parts.append(f"**Quantité :** {rec.quantity_suggestion}")

        message_parts.append("")

        # Risques
        if rec.risk_factors:
            message_parts.append("### ⚠️ Facteurs de risque")
            for risk in rec.risk_factors:
                message_parts.append(f"- {risk}")

        message = "\n".join(message_parts)

        # Résumé court
        summary = f"{rec.ticker}: {rec.primary_reason}"

        # Notification push
        push_title = f"{icon} {rec.ticker} - {rec.action.value.replace('_', ' ')}"
        push_body = f"{rec.primary_reason} (confiance: {rec.confidence:.0f}%)"

        alert_id = f"rec_{rec.ticker}_{int(datetime.now().timestamp())}"

        return Alert(
            id=alert_id,
            ticker=rec.ticker,
            severity=severity,
            priority=rec.priority,
            title=title,
            message=message,
            summary=summary,
            action_required=rec.suggested_action,
            action_button_text=action_button,
            recommendation=rec.action,
            confidence=rec.confidence,
            target_price=rec.target_price,
            stop_loss=rec.stop_loss,
            created_at=datetime.now(),
            push_notification=severity in [AlertSeverity.CRITICAL, AlertSeverity.WARNING],
            push_title=push_title,
            push_body=push_body
        )

    def _create_watch_alert(
        self,
        ticker: str,
        analysis: PortfolioAnalysisResult,
        ml_predictions: MLPortfolioSignals
    ) -> Alert:
        """Crée une alerte de surveillance"""

        pred = ml_predictions.predictions.get(ticker)
        if not pred:
            return None

        title = f"👁️ {ticker} - À surveiller"

        message_parts = [
            f"## 👁️ {ticker} - Position à surveiller",
            "",
            f"Cette position nécessite votre attention dans les prochains jours.",
            "",
            f"### 🔮 Prédiction",
            f"- **Direction :** {pred.overall_direction.capitalize()}",
            f"- **Probabilité de baisse (7j) :** {pred.prediction_7d.probability_down:.0f}%",
            f"- **Mouvement attendu :** {pred.prediction_7d.expected_move_pct:+.1f}%",
            "",
            f"### 💡 Recommandation",
            "Surveiller de près. Considérer réduire si baisse confirmée.",
        ]

        message = "\n".join(message_parts)
        summary = f"{ticker}: Probabilité de baisse {pred.prediction_7d.probability_down:.0f}%"

        alert_id = f"watch_{ticker}_{int(datetime.now().timestamp())}"

        return Alert(
            id=alert_id,
            ticker=ticker,
            severity=AlertSeverity.WARNING,
            priority=ActionPriority.MEDIUM,
            title=title,
            message=message,
            summary=summary,
            action_required="Surveiller l'évolution",
            created_at=datetime.now(),
            push_notification=False
        )

    def _create_opportunity_alert(self, opportunity) -> Alert:
        """Crée une alerte d'opportunité d'achat"""

        title = f"💡 Opportunité - {opportunity.opportunity_type.replace('_', ' ').title()}"

        message_parts = [
            f"## 💡 Nouvelle opportunité détectée",
            "",
            f"**Type :** {opportunity.opportunity_type.replace('_', ' ').title()}",
            f"**Secteur :** {opportunity.sector}",
            f"**Score :** {opportunity.score:.0f}/100",
            "",
            f"### 📋 Raisons",
        ]

        for reason in opportunity.reasons:
            message_parts.append(f"- {reason}")

        message_parts.extend([
            "",
            f"### 💰 Suggestion d'allocation",
            f"**Allocation recommandée :** {opportunity.suggested_allocation:.1f}% du portefeuille",
            f"**Timing :** {opportunity.timing.replace('_', ' ').title()}",
        ])

        message = "\n".join(message_parts)
        summary = f"Opportunité {opportunity.opportunity_type} - Score {opportunity.score:.0f}/100"

        alert_id = f"opp_{opportunity.opportunity_type}_{int(datetime.now().timestamp())}"

        return Alert(
            id=alert_id,
            ticker=opportunity.ticker if opportunity.ticker != "???" else None,
            severity=AlertSeverity.OPPORTUNITY,
            priority=ActionPriority.LOW,
            title=title,
            message=message,
            summary=summary,
            action_required=f"Explorer options dans le secteur {opportunity.sector}",
            created_at=datetime.now(),
            push_notification=False
        )

    def _create_portfolio_alert(self, action: str) -> Alert:
        """Crée une alerte pour action portfolio"""

        title = "📊 Action portfolio recommandée"

        message = f"## 📊 Recommandation portefeuille\n\n{action}"

        alert_id = f"portfolio_{int(datetime.now().timestamp())}"

        return Alert(
            id=alert_id,
            ticker=None,
            severity=AlertSeverity.INFO,
            priority=ActionPriority.LOW,
            title=title,
            message=message,
            summary=action[:100],
            action_required=action,
            created_at=datetime.now(),
            push_notification=False
        )

    def _create_performance_alert(self, ml_predictions: MLPortfolioSignals) -> Alert:
        """Crée une alerte de performance attendue"""

        # Calculer expected return depuis ML signals
        bullish_pct = (ml_predictions.bullish_count / max(len(ml_predictions.predictions), 1)) * 100
        bearish_pct = (ml_predictions.bearish_count / max(len(ml_predictions.predictions), 1)) * 100
        expected = (bullish_pct - bearish_pct) / 10  # Conversion approximative

        if expected > 0:
            icon = "📈"
            severity = AlertSeverity.OPPORTUNITY
            title = f"{icon} Performance positive attendue"
        else:
            icon = "📉"
            severity = AlertSeverity.WARNING
            title = f"{icon} Prudence recommandée"

        message_parts = [
            f"## {icon} Prévision ML portefeuille",
            "",
            f"**Signaux bullish :** {ml_predictions.bullish_count}/{len(ml_predictions.predictions)}",
            f"**Signaux bearish :** {ml_predictions.bearish_count}/{len(ml_predictions.predictions)}",
            f"**Confiance moyenne :** {ml_predictions.avg_confidence:.1f}%",
            "",
        ]

        if expected > 0:
            message_parts.append("Le portefeuille devrait bien performer dans les prochains jours.")
        else:
            message_parts.append("⚠️ Attention: risque de baisse élevé. Considérer protection.")

        message = "\n".join(message_parts)
        summary = f"Return 7j attendu: {expected:+.1f}%"

        alert_id = f"perf_{int(datetime.now().timestamp())}"

        return Alert(
            id=alert_id,
            ticker=None,
            severity=severity,
            priority=ActionPriority.MEDIUM if severity == AlertSeverity.WARNING else ActionPriority.LOW,
            title=title,
            message=message,
            summary=summary,
            action_required="Surveiller l'évolution",
            created_at=datetime.now(),
            push_notification=severity == AlertSeverity.WARNING
        )

    def _create_summary_alert(
        self,
        analysis: PortfolioAnalysisResult,
        ml_predictions: MLPortfolioSignals,
        recommendations: PortfolioRecommendations,
        analysis_time: str
    ) -> Alert:
        """Crée l'alerte de résumé global"""

        time_emoji = "🌅" if analysis_time == "morning" else "🌆"
        time_label = "Analyse matinale" if analysis_time == "morning" else "Analyse du soir"

        title = f"{time_emoji} {time_label} - Résumé du portefeuille"

        message_parts = [
            f"## {time_emoji} {time_label}",
            f"*{datetime.now().strftime('%d/%m/%Y %H:%M')}*",
            "",
            f"### 💼 Vue d'ensemble",
            f"- **Valeur totale :** ${analysis.total_value:,.2f}",
            f"- **Positions :** {analysis.num_positions}",
            f"- **Health score :** {analysis.portfolio_health_score:.0f}/100",
            f"- **Sentiment :** {analysis.portfolio_sentiment.capitalize()}",
            "",
            f"### 🔮 Prévisions ML",
            f"- **Signaux bullish :** {ml_predictions.bullish_count}/{len(ml_predictions.predictions)}",
            f"- **Signaux bearish :** {ml_predictions.bearish_count}/{len(ml_predictions.predictions)}",
            f"- **Confiance moyenne :** {ml_predictions.avg_confidence:.1f}%",
            "",
            f"### 📊 Recommandations",
            recommendations.executive_summary,
            "",
        ]

        if recommendations.critical_actions:
            message_parts.append(f"⚠️ **{len(recommendations.critical_actions)} actions CRITIQUES requises**")

        message = "\n".join(message_parts)
        summary = f"{time_label}: {analysis.num_positions} positions, Health {analysis.portfolio_health_score:.0f}/100"

        alert_id = f"summary_{analysis_time}_{int(datetime.now().timestamp())}"

        return Alert(
            id=alert_id,
            ticker=None,
            severity=AlertSeverity.INFO,
            priority=ActionPriority.LOW,
            title=title,
            message=message,
            summary=summary,
            action_required="Consulter les alertes détaillées",
            created_at=datetime.now(),
            push_notification=bool(recommendations.critical_actions)
        )


# Singleton
_alert_system_instance = None

def get_alert_system() -> AlertSystem:
    """Retourne l'instance singleton"""
    global _alert_system_instance
    if _alert_system_instance is None:
        _alert_system_instance = AlertSystem()
    return _alert_system_instance
