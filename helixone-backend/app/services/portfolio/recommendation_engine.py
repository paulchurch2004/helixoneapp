"""
Recommendation Engine - Génère des recommandations d'investissement détaillées
Recommandations: STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY

Pour chaque recommandation, fournit:
- Explications détaillées (pourquoi)
- Score de confiance
- Prix cibles et stop-loss
- Horizon temporel
- Niveau de risque
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from app.services.portfolio.portfolio_analyzer import PortfolioAnalysisResult, PositionAnalysis
from app.services.portfolio.ml_signal_service import MLPortfolioSignals, MLPrediction

logger = logging.getLogger(__name__)


class RecommendationType(str, Enum):
    """Types de recommandations"""
    STRONG_SELL = "STRONG_SELL"
    SELL = "SELL"
    HOLD = "HOLD"
    BUY = "BUY"
    STRONG_BUY = "STRONG_BUY"


class ActionPriority(str, Enum):
    """Priorité d'action"""
    CRITICAL = "critical"  # Action immédiate requise
    HIGH = "high"  # Action dans les 24h
    MEDIUM = "medium"  # Action dans la semaine
    LOW = "low"  # Surveiller


@dataclass
class Recommendation:
    """Recommandation détaillée pour une position"""
    ticker: str
    action: RecommendationType
    confidence: float  # 0-100

    # Explications
    primary_reason: str  # Raison principale
    detailed_reasons: List[str]  # Liste de toutes les raisons
    risk_factors: List[str]  # Facteurs de risque

    # Action suggérée
    suggested_action: str  # Description de l'action à prendre
    quantity_suggestion: Optional[str] = None  # Ex: "Vendre 50%", "Acheter 10 actions"

    # Prix cibles
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    entry_price: Optional[float] = None  # Pour achats

    # Métadonnées
    priority: ActionPriority = ActionPriority.MEDIUM
    horizon: str = "medium_term"  # short_term, medium_term, long_term
    risk_level: str = "medium"  # low, medium, high

    created_at: datetime = None


@dataclass
class NewOpportunity:
    """Nouvelle opportunité d'achat (action pas encore en portefeuille)"""
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None

    # Raisons
    opportunity_type: str = "diversification"  # diversification, momentum, value, sentiment
    score: float = 0.0  # 0-100
    reasons: List[str] = None

    # Suggestion d'achat
    suggested_allocation: float = 0.0  # % du portefeuille à allouer
    entry_price_range: Optional[Tuple[float, float]] = None

    # Timing
    timing: str = "wait_for_dip"  # immediate, wait_for_dip, dollar_cost_average


@dataclass
class PortfolioRecommendations:
    """Ensemble des recommandations pour le portefeuille"""
    # Recommandations par position
    position_recommendations: Dict[str, Recommendation]

    # Nouvelles opportunités
    new_opportunities: List[NewOpportunity]

    # Recommandations de portefeuille
    portfolio_actions: List[str]  # Actions générales (rébalancement, etc.)

    # Priorités
    critical_actions: List[Recommendation]  # Actions critiques (STRONG_SELL)
    high_priority_actions: List[Recommendation]

    # Résumé exécutif
    executive_summary: str

    # Métadonnées
    generated_at: datetime = None


class RecommendationEngine:
    """
    Moteur de recommandations
    Génère des recommandations d'investissement basées sur toutes les analyses
    """

    def __init__(self):
        logger.info("RecommendationEngine initialisé")

    def generate_recommendations(
        self,
        portfolio: Dict,
        analysis: PortfolioAnalysisResult,
        ml_predictions: MLPortfolioSignals
    ) -> PortfolioRecommendations:
        """
        Génère toutes les recommandations pour le portefeuille

        Args:
            portfolio: Portfolio actuel
            analysis: Résultat de l'analyse
            ml_predictions: Prédictions ML (XGBoost + LSTM)

        Returns:
            PortfolioRecommendations complètes
        """
        # Gérer portfolio comme dict ou objet
        positions = portfolio.get('positions', portfolio) if isinstance(portfolio, dict) else portfolio.positions

        logger.info(f"🎯 Génération recommandations pour {len(positions)} positions")

        # Générer recommandations pour chaque position
        position_recs = {}
        for ticker in positions.keys():
            position_analysis = analysis.positions.get(ticker)
            ml_prediction = ml_predictions.predictions.get(ticker)

            if not position_analysis or not ml_prediction:
                logger.warning(f"Données manquantes pour {ticker}, skip recommandation")
                continue

            rec = self._generate_position_recommendation(
                ticker,
                positions[ticker],
                position_analysis,
                ml_prediction,
                analysis
            )

            position_recs[ticker] = rec

        # Identifier actions critiques
        critical_actions = [r for r in position_recs.values() if r.priority == ActionPriority.CRITICAL]
        high_priority = [r for r in position_recs.values() if r.priority == ActionPriority.HIGH]

        # Scanner nouvelles opportunités
        opportunities = self._scan_opportunities(portfolio, analysis, ml_predictions)

        # Générer recommandations portfolio
        portfolio_actions = self._generate_portfolio_actions(portfolio, analysis, ml_predictions)

        # Résumé exécutif
        exec_summary = self._generate_executive_summary(
            position_recs, critical_actions, opportunities, portfolio_actions
        )

        recommendations = PortfolioRecommendations(
            position_recommendations=position_recs,
            new_opportunities=opportunities,
            portfolio_actions=portfolio_actions,
            critical_actions=critical_actions,
            high_priority_actions=high_priority,
            executive_summary=exec_summary,
            generated_at=datetime.now()
        )

        logger.info(
            f"✅ Recommandations générées: "
            f"{len(critical_actions)} critiques, "
            f"{len(high_priority)} prioritaires, "
            f"{len(opportunities)} opportunités"
        )

        return recommendations

    # ========================================================================
    # GÉNÉRATION DE RECOMMANDATIONS
    # ========================================================================

    def _generate_position_recommendation(
        self,
        ticker: str,
        quantity: float,
        position: PositionAnalysis,
        prediction: MLPrediction,
        portfolio_analysis: PortfolioAnalysisResult
    ) -> Recommendation:
        """Génère une recommandation pour une position"""

        reasons = []
        risk_factors = []
        action = RecommendationType.HOLD
        confidence = 50.0
        priority = ActionPriority.MEDIUM

        # Score composite pour la décision
        decision_score = 0

        # ========== ANALYSE SENTIMENT ==========
        if position.sentiment:
            sentiment = position.sentiment

            if sentiment.current_sentiment == 'bearish':
                decision_score -= sentiment.current_bearish_score / 2
                reasons.append(
                    f"📉 Sentiment très négatif: {sentiment.current_bearish_score:.0f}% bearish "
                    f"(Reddit, StockTwits, News)"
                )

                if sentiment.current_bearish_score > 80:
                    risk_factors.append("Sentiment extrêmement négatif - Risque de panique selling")

            elif sentiment.current_sentiment == 'bullish':
                decision_score += sentiment.current_bullish_score / 2
                reasons.append(
                    f"📈 Sentiment positif: {sentiment.current_bullish_score:.0f}% bullish"
                )

                if sentiment.current_bullish_score > 85:
                    risk_factors.append("Euphorie possible - Risque de correction")

            # Consensus
            if sentiment.source_consensus < 40:
                risk_factors.append("Signaux contradictoires entre les sources d'information")
                decision_score -= 10

        # ========== ANALYSE ML (NEW!) ==========
        if position.ml_prediction:
            ml_pred = position.ml_prediction

            # Signal ML principal
            if ml_pred.signal == 'BUY':
                decision_score += (ml_pred.signal_strength / 100) * 25  # Impact fort
                reasons.append(
                    f"🧠 ML Signal: BUY (confiance: {ml_pred.signal_strength:.0f}%) - "
                    f"Prédictions: {ml_pred.prediction_1d} (1j), {ml_pred.prediction_3d} (3j), {ml_pred.prediction_7d} (7j)"
                )

                # Afficher prix prédit si disponible
                if ml_pred.predicted_change_3d:
                    reasons.append(
                        f"   Prix prédit 3j: ${ml_pred.predicted_price_3d:.2f} ({ml_pred.predicted_change_3d:+.1f}%)"
                    )

            elif ml_pred.signal == 'SELL':
                decision_score -= (ml_pred.signal_strength / 100) * 30  # Impact très fort pour SELL
                reasons.append(
                    f"🧠 ML Signal: SELL (confiance: {ml_pred.signal_strength:.0f}%) - "
                    f"Prédictions: {ml_pred.prediction_1d} (1j), {ml_pred.prediction_3d} (3j), {ml_pred.prediction_7d} (7j)"
                )
                risk_factors.append(f"Modèle ML prédit une baisse sur tous les horizons")

                if ml_pred.predicted_change_3d:
                    reasons.append(
                        f"   Prix prédit 3j: ${ml_pred.predicted_price_3d:.2f} ({ml_pred.predicted_change_3d:+.1f}%)"
                    )

            # Confiance consensus
            avg_ml_conf = (ml_pred.confidence_1d + ml_pred.confidence_3d + ml_pred.confidence_7d) / 3
            if avg_ml_conf > 80:
                reasons.append(f"   Haute confiance ML (avg: {avg_ml_conf:.0f}%)")
                confidence += 15  # Augmente la confiance globale de la recommandation
            elif avg_ml_conf < 50:
                risk_factors.append(f"Prédictions ML incertaines (confiance avg: {avg_ml_conf:.0f}%)")

        # ========== ANALYSE PRÉDICTION ==========
        if prediction.overall_direction == 'bearish':
            decision_score -= 20
            reasons.append(
                f"🔮 Prédiction baissière sur 7j: {prediction.prediction_7d.expected_move_pct:+.1f}% "
                f"(confiance: {prediction.overall_confidence:.0f}%)"
            )
        elif prediction.overall_direction == 'bullish':
            decision_score += 20
            reasons.append(
                f"🔮 Prédiction haussière sur 7j: {prediction.prediction_7d.expected_move_pct:+.1f}%"
            )

        # ========== ANALYSE RISQUES ==========
        if position.risks:
            decision_score -= len(position.risks) * 5
            risk_factors.extend(position.risks)

        # ========== CONCENTRATION ==========
        if position.portfolio_weight > 30:
            decision_score -= 15
            reasons.append(
                f"⚠️ Concentration excessive: {position.portfolio_weight:.1f}% du portefeuille"
            )
            risk_factors.append("Sur-exposition à une seule position")

        # ========== HEALTH SCORE ==========
        if position.health_score < 40:
            decision_score -= 15
            reasons.append(f"🏥 Health score faible: {position.health_score:.0f}/100")

        # ========== ÉVÉNEMENTS ÉCONOMIQUES ==========
        if portfolio_analysis.event_risks and position.sector:
            event_risks = portfolio_analysis.event_risks

            # Vérifier si le secteur est exposé à des événements critiques
            sector_risk = event_risks.sector_risks.get(position.sector, 0)

            if sector_risk > 60:  # Risque élevé
                decision_score -= 15
                reasons.append(
                    f"📅 Événements économiques à venir: risque élevé pour {position.sector} "
                    f"({sector_risk:.0f}/100)"
                )
                risk_factors.append(f"{event_risks.critical_events} événements critiques dans les 7 prochains jours")
            elif sector_risk > 40:  # Risque moyen
                decision_score -= 8
                reasons.append(f"📅 Événements économiques: risque modéré pour {position.sector}")

            # Ajouter les recommandations événementielles aux risk factors
            if event_risks.recommendations:
                # Filtrer les recommandations pertinentes pour ce ticker/secteur
                relevant_recs = [
                    rec for rec in event_risks.recommendations
                    if position.sector and position.sector.lower() in rec.lower()
                    or ticker.upper() in rec.upper()
                ]
                risk_factors.extend(relevant_recs[:2])  # Max 2 recommandations

        # ========== DÉCISION FINALE ==========
        if decision_score <= -40:
            action = RecommendationType.STRONG_SELL
            primary_reason = "Signaux très négatifs convergents"
            suggested_action = f"VENDRE immédiatement 75-100% de la position {ticker}"
            priority = ActionPriority.CRITICAL
            confidence = min(85, 60 + abs(decision_score) / 2)
            quantity_suggestion = "Vendre 75-100%"

        elif decision_score <= -20:
            action = RecommendationType.SELL
            primary_reason = "Risques significatifs identifiés"
            suggested_action = f"RÉDUIRE la position {ticker} de 30-50%"
            priority = ActionPriority.HIGH
            confidence = min(75, 50 + abs(decision_score))
            quantity_suggestion = "Vendre 30-50%"

        elif decision_score >= 40:
            action = RecommendationType.STRONG_BUY
            primary_reason = "Opportunité forte détectée"
            suggested_action = f"RENFORCER fortement la position {ticker}"
            priority = ActionPriority.HIGH
            confidence = min(85, 60 + decision_score / 2)
            quantity_suggestion = "Acheter +30-50%"

        elif decision_score >= 20:
            action = RecommendationType.BUY
            primary_reason = "Signaux positifs"
            suggested_action = f"RENFORCER modérément la position {ticker}"
            priority = ActionPriority.MEDIUM
            confidence = min(75, 50 + decision_score)
            quantity_suggestion = "Acheter +10-20%"

        else:
            action = RecommendationType.HOLD
            primary_reason = "Situation stable, surveiller"
            suggested_action = f"CONSERVER {ticker} et surveiller l'évolution"
            priority = ActionPriority.LOW
            confidence = 60
            quantity_suggestion = None

        # Prix cibles et stop-loss
        current_price = position.current_price

        if action in [RecommendationType.SELL, RecommendationType.STRONG_SELL]:
            # Stop loss agressif
            stop_loss = current_price * 0.95  # -5%
            target_price = None
        elif action in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
            # Target optimiste
            target_price = prediction.prediction_7d.target_price_bull
            stop_loss = current_price * 0.92  # -8%
        else:
            # HOLD
            target_price = prediction.prediction_7d.target_price_base
            stop_loss = current_price * 0.90  # -10%

        return Recommendation(
            ticker=ticker,
            action=action,
            confidence=confidence,
            primary_reason=primary_reason,
            detailed_reasons=reasons if reasons else ["Position stable, aucun signal fort"],
            risk_factors=risk_factors if risk_factors else ["Risques standards de marché"],
            suggested_action=suggested_action,
            quantity_suggestion=quantity_suggestion,
            target_price=target_price,
            stop_loss=stop_loss,
            priority=priority,
            horizon="medium_term",
            risk_level="high" if action in [RecommendationType.STRONG_SELL, RecommendationType.SELL] else "medium",
            created_at=datetime.now()
        )

    def _scan_opportunities(
        self,
        portfolio: Dict,
        analysis: PortfolioAnalysisResult,
        ml_predictions: MLPortfolioSignals
    ) -> List[NewOpportunity]:
        """Scanner de nouvelles opportunités d'achat"""
        opportunities = []

        # Opportunité 1: Diversification sectorielle
        if analysis.correlations and analysis.correlations.top_sector_weight > 40:
            # Suggérer secteur décorrélé
            underweight_sectors = self._find_underweight_sectors(analysis)

            for sector in underweight_sectors[:2]:
                opportunities.append(NewOpportunity(
                    ticker="???",  # TODO: Scanner le marché pour trouver tickers
                    sector=sector,
                    opportunity_type="diversification",
                    score=75,
                    reasons=[
                        f"Diversifier hors du secteur {analysis.correlations.top_sector}",
                        f"Secteur {sector} sous-représenté dans le portefeuille",
                        "Réduire la corrélation globale"
                    ],
                    suggested_allocation=10.0,  # 10% du portefeuille
                    timing="dollar_cost_average"
                ))

        # Opportunité 2: Actions défensives si risque élevé
        if analysis.risks and analysis.risks.overall_risk_score < 40:
            opportunities.append(NewOpportunity(
                ticker="???",
                sector="Healthcare/Consumer Staples",
                opportunity_type="risk_reduction",
                score=70,
                reasons=[
                    f"Risque portefeuille élevé (score: {analysis.risks.overall_risk_score:.0f}/100)",
                    "Ajouter des positions défensives",
                    "Exemples: JNJ, PG, KO, WMT"
                ],
                suggested_allocation=15.0,
                timing="immediate"
            ))

        return opportunities

    def _find_underweight_sectors(self, analysis: PortfolioAnalysisResult) -> List[str]:
        """Trouve les secteurs sous-représentés"""
        # Secteurs cibles pour diversification
        target_sectors = [
            "Healthcare",
            "Consumer Staples",
            "Utilities",
            "Energy",
            "Financials",
            "Industrials"
        ]

        current_sectors = set()
        if analysis.correlations and analysis.correlations.sector_concentration:
            current_sectors = set(analysis.correlations.sector_concentration.keys())

        # Secteurs absents ou sous-représentés
        underweight = [s for s in target_sectors if s not in current_sectors]

        return underweight

    def _generate_portfolio_actions(
        self,
        portfolio: Dict,
        analysis: PortfolioAnalysisResult,
        ml_predictions: MLPortfolioSignals
    ) -> List[str]:
        """Génère des recommandations générales de portefeuille"""
        actions = []

        # Rébalancement
        if analysis.correlations and analysis.correlations.top_sector_weight > 50:
            actions.append(
                f"🔄 RÉBALANCER: Réduire l'exposition au secteur {analysis.correlations.top_sector} "
                f"de {analysis.correlations.top_sector_weight:.0f}% à <40%"
            )

        # Gestion des risques
        if analysis.risks and analysis.risks.overall_risk_score < 40:
            actions.append(
                "⚠️ RÉDUIRE RISQUE: Ajouter des positions défensives (Healthcare, Consumer Staples)"
            )

        # Cash position
        cash_pct = (portfolio.cash / analysis.total_value * 100) if analysis.total_value > 0 else 0

        if cash_pct < 5:
            actions.append(
                "💵 LIQUIDITÉS: Augmenter la réserve de cash à 10-15% pour opportunités"
            )
        elif cash_pct > 30:
            actions.append(
                "💵 DÉPLOYER CASH: Trop de liquidités, investir progressivement"
            )

        # Downside risk (calculé depuis les prédictions ML)
        bearish_pct = (ml_predictions.bearish_count / max(len(ml_predictions.predictions), 1)) * 100
        if bearish_pct > 60:
            actions.append(
                f"🛡️ HEDGING: {bearish_pct:.0f}% des positions baissières, "
                "considérer protection (puts, inverse ETFs)"
            )

        return actions

    def _generate_executive_summary(
        self,
        recommendations: Dict[str, Recommendation],
        critical: List[Recommendation],
        opportunities: List[NewOpportunity],
        portfolio_actions: List[str]
    ) -> str:
        """Génère un résumé exécutif"""

        summary_parts = []

        # Actions critiques
        if critical:
            tickers = ", ".join([r.ticker for r in critical])
            summary_parts.append(f"🚨 ACTIONS URGENTES: {len(critical)} positions nécessitent une action immédiate ({tickers})")

        # Distribution des recommandations
        action_counts = {}
        for rec in recommendations.values():
            action_counts[rec.action.value] = action_counts.get(rec.action.value, 0) + 1

        summary_parts.append(
            f"📊 RÉPARTITION: "
            f"{action_counts.get('STRONG_SELL', 0)} Strong Sell, "
            f"{action_counts.get('SELL', 0)} Sell, "
            f"{action_counts.get('HOLD', 0)} Hold, "
            f"{action_counts.get('BUY', 0)} Buy, "
            f"{action_counts.get('STRONG_BUY', 0)} Strong Buy"
        )

        # Opportunités
        if opportunities:
            summary_parts.append(f"💡 {len(opportunities)} nouvelles opportunités identifiées")

        # Actions portfolio
        if portfolio_actions:
            summary_parts.append(f"🎯 {len(portfolio_actions)} actions recommandées au niveau portefeuille")

        return " | ".join(summary_parts)


# Singleton
_recommendation_engine_instance = None

def get_recommendation_engine() -> RecommendationEngine:
    """Retourne l'instance singleton"""
    global _recommendation_engine_instance
    if _recommendation_engine_instance is None:
        _recommendation_engine_instance = RecommendationEngine()
    return _recommendation_engine_instance
