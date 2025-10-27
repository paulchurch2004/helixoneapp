"""
Event Impact Predictor - Moteur de Prédiction d'Impact des Événements

Analyse l'historique des événements économiques pour prédire leur impact futur
sur les actions et secteurs du portefeuille.

Fonctionnalités:
- Analyse historique événement → impact prix
- Prédiction ML de l'impact futur
- Corrélations secteur ↔ type d'événement
- Génération d'alertes préventives
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics

from app.services.economic_calendar_service import (
    EconomicEvent,
    EarningsEvent,
    get_economic_calendar_service
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EventImpactPrediction:
    """Prédiction d'impact d'un événement sur une action/secteur"""
    event: EconomicEvent
    ticker: Optional[str]  # None si prédiction sectorielle
    sector: Optional[str]

    # Prédiction
    predicted_impact_pct: float  # % de variation attendue
    confidence: float  # 0-100, confiance dans la prédiction

    # Direction
    direction: str  # 'bullish', 'bearish', 'neutral'
    probability_up: float  # 0-100
    probability_down: float  # 0-100

    # Facteurs de la prédiction
    factors: List[str]  # Raisons de cette prédiction

    # Données historiques
    historical_avg_impact: Optional[float] = None  # Impact moyen historique
    historical_std: Optional[float] = None  # Écart-type
    sample_size: int = 0  # Nombre d'événements similaires analysés


@dataclass
class SectorImpactCorrelation:
    """Corrélation entre type d'événement et secteur"""
    event_type: str  # 'fed_meeting', 'cpi', 'earnings', etc.
    sector: str  # 'Technology', 'Financials', etc.

    avg_impact_pct: float  # Impact moyen (%)
    std_dev: float  # Écart-type
    confidence: float  # 0-100, basé sur sample size
    sample_size: int  # Nombre d'événements analysés

    # Tendance
    trend: str  # 'positive', 'negative', 'neutral'


@dataclass
class PortfolioEventRisk:
    """Risque global du portefeuille lié aux événements à venir"""
    total_events: int
    critical_events: int
    high_impact_events: int

    # Risque global
    overall_risk_score: float  # 0-100, 100 = très risqué
    risk_level: str  # 'low', 'medium', 'high', 'critical'

    # Par secteur
    sector_risks: Dict[str, float]  # {sector: risk_score}

    # Top événements à surveiller
    top_risk_events: List[EconomicEvent]

    # Suggestions
    recommendations: List[str]


# ============================================================================
# MOTEUR DE PRÉDICTION
# ============================================================================

class EventImpactPredictor:
    """
    Moteur de prédiction d'impact des événements économiques

    Utilise:
    - Analyse historique des corrélations
    - Règles empiriques (Fed → Financials, etc.)
    - Machine Learning (phase 2)
    """

    def __init__(self):
        self.calendar_service = get_economic_calendar_service()

        # Corrélations empiriques (basées sur études de marché)
        # TODO: Remplacer par ML avec données historiques réelles
        self.empirical_correlations = {
            # Fed Rate Decisions
            'fed_meeting': {
                'Technology': {'avg': -2.3, 'std': 1.5},  # Tech baisse quand taux montent
                'Financials': {'avg': +3.8, 'std': 2.1},  # Financials montent
                'Real Estate': {'avg': -3.2, 'std': 2.0},  # RE baisse
                'Utilities': {'avg': -1.5, 'std': 1.2},  # Utilities baissent
                'Consumer Discretionary': {'avg': -1.8, 'std': 1.4},
                'Consumer Staples': {'avg': +0.5, 'std': 0.8},  # Défensif
                'Healthcare': {'avg': +0.3, 'std': 1.0},  # Relativement stable
                'Industrials': {'avg': -1.2, 'std': 1.3},
                'Materials': {'avg': -1.0, 'std': 1.5},
                'Energy': {'avg': -0.5, 'std': 1.8},
                'Communication Services': {'avg': -1.5, 'std': 1.3}
            },
            # CPI (Inflation)
            'cpi': {
                'Technology': {'avg': -1.8, 'std': 2.0},  # Impact négatif si inflation élevée
                'Financials': {'avg': +1.2, 'std': 1.5},
                'Consumer Discretionary': {'avg': -2.5, 'std': 2.2},
                'Consumer Staples': {'avg': +0.8, 'std': 1.0},  # Pouvoir de pricing
                'Energy': {'avg': +2.0, 'std': 2.5},  # Corrélé à inflation
                'Real Estate': {'avg': -2.0, 'std': 1.8}
            },
            # GDP
            'gdp': {
                'All': {'avg': +1.5, 'std': 1.0}  # Généralement positif
            },
            # Non-Farm Payrolls
            'nfp': {
                'All': {'avg': +0.8, 'std': 1.5}  # Impact modéré
            },
            # Retail Sales
            'retail_sales': {
                'Consumer Discretionary': {'avg': +2.5, 'std': 2.0},
                'Consumer Staples': {'avg': +1.0, 'std': 1.2}
            }
        }

        logger.info("EventImpactPredictor initialisé")

    # ========================================================================
    # MÉTHODES PRINCIPALES
    # ========================================================================

    async def predict_portfolio_impact(
        self,
        portfolio_positions: Dict[str, float],  # {ticker: quantity}
        portfolio_sectors: Dict[str, str],  # {ticker: sector}
        days_ahead: int = 30
    ) -> List[EventImpactPrediction]:
        """
        Prédit l'impact des événements à venir sur le portefeuille

        Args:
            portfolio_positions: Positions du portefeuille
            portfolio_sectors: Mapping ticker → secteur
            days_ahead: Nombre de jours à analyser

        Returns:
            Liste de prédictions d'impact
        """
        logger.info(f"🔮 Prédiction impact événements ({days_ahead}j)")

        # Récupérer événements à venir
        upcoming_events = await self.calendar_service.get_upcoming_events(
            days=days_ahead,
            min_impact='medium'
        )

        if not upcoming_events:
            logger.info("Aucun événement à venir")
            return []

        predictions = []

        # Pour chaque événement
        for event in upcoming_events:
            # Prédire impact par ticker
            for ticker, quantity in portfolio_positions.items():
                sector = portfolio_sectors.get(ticker, 'Unknown')

                prediction = self._predict_event_impact_on_ticker(
                    event=event,
                    ticker=ticker,
                    sector=sector
                )

                if prediction:
                    predictions.append(prediction)

        logger.info(f"✅ {len(predictions)} prédictions générées")
        return predictions

    async def analyze_portfolio_event_risk(
        self,
        portfolio_positions: Dict[str, float],
        portfolio_sectors: Dict[str, str],
        days_ahead: int = 7
    ) -> PortfolioEventRisk:
        """
        Analyse le risque global du portefeuille lié aux événements à venir

        Args:
            portfolio_positions: Positions du portefeuille
            portfolio_sectors: Mapping ticker → secteur
            days_ahead: Période d'analyse

        Returns:
            Analyse de risque complète
        """
        logger.info(f"⚠️ Analyse risque événements ({days_ahead}j)")

        # Récupérer événements
        upcoming_events = await self.calendar_service.get_upcoming_events(
            days=days_ahead,
            min_impact='low'
        )

        # Compter par niveau
        critical_count = sum(1 for e in upcoming_events if e.impact_level == 'critical')
        high_count = sum(1 for e in upcoming_events if e.impact_level == 'high')

        # Calculer risque par secteur
        sector_risks = {}
        # portfolio_sectors peut être une liste ou un dict
        if isinstance(portfolio_sectors, list):
            unique_sectors = set(portfolio_sectors)
        else:
            unique_sectors = set(portfolio_sectors.values())

        for sector in unique_sectors:
            risk_score = self._calculate_sector_risk(
                sector=sector,
                events=upcoming_events
            )
            sector_risks[sector] = risk_score

        # Risque global (moyenne pondérée)
        overall_risk = statistics.mean(sector_risks.values()) if sector_risks else 0

        # Classifier niveau
        if overall_risk >= 75:
            risk_level = 'critical'
        elif overall_risk >= 50:
            risk_level = 'high'
        elif overall_risk >= 25:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Top événements à surveiller
        top_events = sorted(
            upcoming_events,
            key=lambda e: ['low', 'medium', 'high', 'critical'].index(e.impact_level),
            reverse=True
        )[:5]

        # Recommandations
        recommendations = self._generate_risk_recommendations(
            overall_risk=overall_risk,
            sector_risks=sector_risks,
            top_events=top_events
        )

        return PortfolioEventRisk(
            total_events=len(upcoming_events),
            critical_events=critical_count,
            high_impact_events=high_count,
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            sector_risks=sector_risks,
            top_risk_events=top_events,
            recommendations=recommendations
        )

    def get_sector_correlations(
        self,
        event_type: str
    ) -> List[SectorImpactCorrelation]:
        """
        Récupère les corrélations entre un type d'événement et tous les secteurs

        Args:
            event_type: Type d'événement ('fed_meeting', 'cpi', etc.)

        Returns:
            Liste de corrélations secteur
        """
        correlations = []

        event_data = self.empirical_correlations.get(event_type, {})

        for sector, stats in event_data.items():
            avg_impact = stats['avg']
            std_dev = stats['std']

            # Déterminer tendance
            if avg_impact > 1.0:
                trend = 'positive'
            elif avg_impact < -1.0:
                trend = 'negative'
            else:
                trend = 'neutral'

            # Confiance basée sur écart-type (plus c'est stable, plus confiant)
            confidence = max(0, 100 - (std_dev * 20))

            correlation = SectorImpactCorrelation(
                event_type=event_type,
                sector=sector,
                avg_impact_pct=avg_impact,
                std_dev=std_dev,
                confidence=confidence,
                sample_size=50,  # TODO: Remplacer par vraies données
                trend=trend
            )

            correlations.append(correlation)

        return correlations

    # ========================================================================
    # MÉTHODES PRIVÉES
    # ========================================================================

    def _predict_event_impact_on_ticker(
        self,
        event: EconomicEvent,
        ticker: str,
        sector: str
    ) -> Optional[EventImpactPrediction]:
        """Prédit l'impact d'un événement sur un ticker spécifique"""

        # Récupérer corrélation pour ce secteur et type d'événement
        event_correlations = self.empirical_correlations.get(event.event_type, {})

        # Chercher secteur spécifique ou 'All'
        correlation_data = event_correlations.get(sector) or event_correlations.get('All')

        if not correlation_data:
            return None  # Pas de donnée pour ce secteur

        avg_impact = correlation_data['avg']
        std_dev = correlation_data['std']

        # Déterminer direction
        if avg_impact > 0.5:
            direction = 'bullish'
            prob_up = 70
            prob_down = 30
        elif avg_impact < -0.5:
            direction = 'bearish'
            prob_up = 30
            prob_down = 70
        else:
            direction = 'neutral'
            prob_up = 50
            prob_down = 50

        # Confiance basée sur écart-type
        confidence = max(20, 100 - (std_dev * 25))

        # Facteurs explicatifs
        factors = self._generate_prediction_factors(
            event_type=event.event_type,
            sector=sector,
            avg_impact=avg_impact
        )

        return EventImpactPrediction(
            event=event,
            ticker=ticker,
            sector=sector,
            predicted_impact_pct=avg_impact,
            confidence=confidence,
            direction=direction,
            probability_up=prob_up,
            probability_down=prob_down,
            factors=factors,
            historical_avg_impact=avg_impact,
            historical_std=std_dev,
            sample_size=50  # TODO: Vraies données
        )

    def _calculate_sector_risk(
        self,
        sector: str,
        events: List[EconomicEvent]
    ) -> float:
        """Calcule le score de risque pour un secteur donné"""

        total_risk = 0
        event_count = 0

        for event in events:
            # Vérifier si événement affecte ce secteur
            if sector not in event.affected_sectors and 'All' not in event.affected_sectors:
                continue

            # Poids selon importance
            impact_weight = {
                'low': 10,
                'medium': 25,
                'high': 50,
                'critical': 100
            }.get(event.impact_level, 10)

            total_risk += impact_weight
            event_count += 1

        # Normaliser
        if event_count == 0:
            return 0

        risk_score = min(100, total_risk / event_count)
        return risk_score

    def _generate_prediction_factors(
        self,
        event_type: str,
        sector: str,
        avg_impact: float
    ) -> List[str]:
        """Génère les facteurs explicatifs d'une prédiction"""

        factors = []

        # Mapping raisons par événement/secteur
        reasons = {
            'fed_meeting': {
                'Technology': "Les entreprises tech sont sensibles aux taux d'intérêt (coût du capital élevé)",
                'Financials': "Les banques bénéficient de taux plus élevés (marges d'intérêt)",
                'Real Estate': "Le secteur immobilier est très sensible aux taux hypothécaires",
                'Utilities': "Secteur défensif avec beaucoup de dette, impacté négativement"
            },
            'cpi': {
                'Technology': "L'inflation élevée pousse la Fed à monter les taux, négatif pour la tech",
                'Energy': "L'énergie est corrélée positivement à l'inflation",
                'Consumer Discretionary': "Les consommateurs réduisent les dépenses non-essentielles"
            },
            'retail_sales': {
                'Consumer Discretionary': "Impact direct sur le chiffre d'affaires des retailers",
                'Consumer Staples': "Impact modéré, les produits essentiels sont moins affectés"
            }
        }

        event_reasons = reasons.get(event_type, {})
        sector_reason = event_reasons.get(sector, event_reasons.get('All', ''))

        if sector_reason:
            factors.append(sector_reason)

        # Ajouter magnitude
        if avg_impact > 2:
            factors.append(f"Impact historique fort: +{avg_impact:.1f}% en moyenne")
        elif avg_impact < -2:
            factors.append(f"Impact historique fort: {avg_impact:.1f}% en moyenne")

        # Default
        if not factors:
            factors.append(f"Basé sur analyse historique secteur {sector}")

        return factors

    def _generate_risk_recommendations(
        self,
        overall_risk: float,
        sector_risks: Dict[str, float],
        top_events: List[EconomicEvent]
    ) -> List[str]:
        """Génère des recommandations basées sur l'analyse de risque"""

        recommendations = []

        if overall_risk >= 75:
            recommendations.append("⚠️ Risque CRITIQUE détecté - Considérer réduction exposure ou hedging")

        if overall_risk >= 50:
            recommendations.append("Éviter leverage élevé durant cette période")

        # Recommandations par secteur
        high_risk_sectors = [s for s, r in sector_risks.items() if r > 60]
        if high_risk_sectors:
            recommendations.append(
                f"Surveiller de près les secteurs à risque: {', '.join(high_risk_sectors)}"
            )

        # Recommandations par événement
        if any(e.event_type == 'fed_meeting' for e in top_events):
            recommendations.append("Réunion Fed à venir - Volatilité attendue sur les marchés")

        if any(e.event_type == 'earnings' for e in top_events):
            recommendations.append("Season earnings - Ajuster positions avant publications")

        return recommendations


# ============================================================================
# SINGLETON
# ============================================================================

_predictor_instance = None

def get_event_impact_predictor() -> EventImpactPredictor:
    """Retourne l'instance singleton du prédicteur"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EventImpactPredictor()
    return _predictor_instance
