"""
Portfolio Analyzer - Analyse complète de portefeuille
Coordonne toutes les analyses : données, sentiment, corrélations, risques

C'est le cerveau du système d'analyse automatique
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

from app.services.portfolio.data_aggregator import get_data_aggregator, AggregatedStockData
from app.services.portfolio.sentiment_aggregator import get_sentiment_aggregator, SentimentTrend
from app.services.portfolio.ml_signal_service import get_ml_signal_service, MLPrediction
from app.services.economic_calendar_service import get_economic_calendar_service
from app.services.event_impact_predictor import get_event_impact_predictor, PortfolioEventRisk

logger = logging.getLogger(__name__)


@dataclass
class PositionAnalysis:
    """Analyse complète d'une position"""
    ticker: str
    quantity: float
    current_price: float
    position_value: float
    portfolio_weight: float  # % du portefeuille

    # Sentiment
    sentiment: Optional[SentimentTrend] = None

    # ML Predictions (NEW!)
    ml_prediction: Optional[MLPrediction] = None

    # Données fondamentales
    sector: Optional[str] = None
    industry: Optional[str] = None
    beta: Optional[float] = None
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None

    # Risques identifiés
    risks: List[str] = None

    # Score global (0-100, 100 = très bon)
    health_score: float = 50.0

    updated_at: datetime = None


@dataclass
class PortfolioCorrelation:
    """Analyse des corrélations entre positions"""
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    highly_correlated_pairs: List[Tuple[str, str, float]] = None  # (ticker1, ticker2, correlation)
    diversification_score: float = 50.0  # 0-100, 100 = très diversifié
    sector_concentration: Dict[str, float] = None  # {sector: % portefeuille}
    top_sector: Optional[str] = None
    top_sector_weight: float = 0.0


@dataclass
class PortfolioRisk:
    """Évaluation des risques du portefeuille"""
    # Risques par catégorie
    concentration_risk: str = 'medium'  # 'low', 'medium', 'high'
    sentiment_risk: str = 'medium'
    volatility_risk: str = 'medium'
    sector_risk: str = 'medium'

    # Score de risque global (0-100, 0 = très risqué, 100 = sûr)
    overall_risk_score: float = 50.0

    # Risques détectés
    risk_factors: List[str] = None

    # Recommandations de réduction de risque
    risk_mitigation_suggestions: List[str] = None


@dataclass
class PortfolioAnalysisResult:
    """Résultat complet de l'analyse de portefeuille"""
    # Métadonnées
    analysis_id: str
    user_id: Optional[str] = None
    analyzed_at: datetime = None

    # Portfolio
    total_value: float = 0.0
    cash: float = 0.0
    num_positions: int = 0

    # Analyses par position
    positions: Dict[str, PositionAnalysis] = None

    # Corrélations
    correlations: Optional[PortfolioCorrelation] = None

    # Risques
    risks: Optional[PortfolioRisk] = None

    # Risques liés aux événements économiques
    event_risks: Optional[PortfolioEventRisk] = None

    # Sentiment global du portefeuille
    portfolio_sentiment: str = 'neutral'  # 'bullish', 'bearish', 'neutral'
    portfolio_sentiment_score: float = 50.0  # 0-100

    # Health score global
    portfolio_health_score: float = 50.0  # 0-100, 100 = excellent

    # Alertes critiques
    critical_alerts: List[str] = None

    # Temps d'exécution
    execution_time_ms: int = 0

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        return asdict(self)


class PortfolioAnalyzer:
    """
    Analyseur de portefeuille
    Coordonne toutes les analyses pour générer un rapport complet
    """

    def __init__(self):
        self.data_aggregator = get_data_aggregator()
        self.sentiment_aggregator = get_sentiment_aggregator()
        self.calendar_service = get_economic_calendar_service()
        self.event_predictor = get_event_impact_predictor()

        logger.info("PortfolioAnalyzer initialisé (avec calendrier économique)")

    async def analyze_portfolio(
        self,
        portfolio: Dict,
        user_id: Optional[str] = None,
        deep_analysis: bool = True
    ) -> PortfolioAnalysisResult:
        """
        Analyse complète du portefeuille

        Args:
            portfolio: Portfolio à analyser
            user_id: ID de l'utilisateur
            deep_analysis: Inclure analyses approfondies (sentiment, corrélations)

        Returns:
            PortfolioAnalysisResult avec analyse complète
        """
        start_time = datetime.now()
        analysis_id = f"analysis_{int(start_time.timestamp())}"

        logger.info(f"🔍 Analyse portefeuille {analysis_id} ({len(portfolio.positions)} positions)")

        # Extraire les tickers
        tickers = list(portfolio.positions.keys())

        # Phase 1: Collecter toutes les données en parallèle
        logger.info("📊 Phase 1: Collecte données...")
        stock_data = await self.data_aggregator.aggregate_multiple_stocks(
            tickers,
            include_sentiment=deep_analysis,
            include_news=deep_analysis,
            include_fundamentals=True
        )

        # Phase 2: Analyser sentiment en profondeur
        sentiment_trends = {}
        if deep_analysis:
            logger.info("💬 Phase 2: Analyse sentiment...")
            for ticker in tickers:
                try:
                    trend = self.sentiment_aggregator.analyze_sentiment_trend(ticker, lookback_days=7)
                    sentiment_trends[ticker] = trend
                except Exception as e:
                    logger.error(f"Erreur analyse sentiment {ticker}: {e}")

        # Phase 2.5: Obtenir prédictions ML (NEW!)
        ml_predictions = {}
        if deep_analysis:
            logger.info("🧠 Phase 2.5: Prédictions ML...")
            try:
                ml_service = get_ml_signal_service()
                # Obtenir signaux ML pour tout le portfolio
                ml_signals = await ml_service.get_portfolio_signals(tickers)
                ml_predictions = ml_signals.predictions
                logger.info(f"   ✅ {len(ml_predictions)} prédictions ML générées")
                logger.info(f"   📊 Bullish: {ml_signals.bullish_count}, Bearish: {ml_signals.bearish_count}, Neutral: {ml_signals.neutral_count}")
            except Exception as e:
                logger.error(f"Erreur prédictions ML: {e}")

        # Phase 3: Calculer valeur totale et poids des positions
        logger.info("💰 Phase 3: Calcul valeurs...")
        total_value = portfolio.cash
        position_values = {}

        for ticker, quantity in portfolio.positions.items():
            data = stock_data.get(ticker)
            if data and data.price and data.price.current_price > 0:
                value = data.price.current_price * quantity
                position_values[ticker] = value
                total_value += value
            else:
                logger.warning(f"Prix non disponible pour {ticker}, valeur ignorée")
                position_values[ticker] = 0

        # Phase 4: Analyser chaque position
        logger.info("🔬 Phase 4: Analyse positions...")
        position_analyses = {}

        for ticker, quantity in portfolio.positions.items():
            data = stock_data.get(ticker)
            if not data or not data.price:
                continue

            weight = (position_values[ticker] / total_value * 100) if total_value > 0 else 0

            # Récupérer sentiment
            sentiment = sentiment_trends.get(ticker) if deep_analysis else None

            # Récupérer prédiction ML (NEW!)
            ml_pred = ml_predictions.get(ticker) if deep_analysis else None

            # Identifier risques
            risks = self._identify_position_risks(ticker, data, sentiment, weight)

            # Calculer health score (incluant ML maintenant)
            health_score = self._calculate_position_health(data, sentiment, risks, ml_pred)

            position_analyses[ticker] = PositionAnalysis(
                ticker=ticker,
                quantity=quantity,
                current_price=data.price.current_price,
                position_value=position_values[ticker],
                portfolio_weight=weight,
                sentiment=sentiment,
                ml_prediction=ml_pred,  # NEW!
                sector=data.fundamentals.sector if data.fundamentals else None,
                industry=data.fundamentals.industry if data.fundamentals else None,
                beta=data.fundamentals.beta if data.fundamentals else None,
                pe_ratio=data.fundamentals.pe_ratio if data.fundamentals else None,
                market_cap=data.fundamentals.market_cap if data.fundamentals else None,
                risks=risks,
                health_score=health_score,
                updated_at=datetime.now()
            )

        # Phase 5: Analyser corrélations
        logger.info("🔗 Phase 5: Analyse corrélations...")
        correlations = self._analyze_correlations(position_analyses) if deep_analysis else None

        # Phase 6: Évaluer risques globaux
        logger.info("⚠️  Phase 6: Évaluation risques...")
        portfolio_risks = self._assess_portfolio_risks(position_analyses, correlations)

        # Phase 7: Sentiment global du portefeuille
        logger.info("💭 Phase 7: Sentiment global...")
        portfolio_sentiment, portfolio_sentiment_score = self._calculate_portfolio_sentiment(position_analyses)

        # Phase 8: Health score global
        logger.info("🏥 Phase 8: Health score...")
        portfolio_health = self._calculate_portfolio_health(position_analyses, portfolio_risks, correlations)

        # Phase 9: Identifier alertes critiques
        logger.info("🚨 Phase 9: Alertes critiques...")
        critical_alerts = self._identify_critical_alerts(position_analyses, portfolio_risks, correlations)

        # Phase 10: Analyser événements économiques à venir
        logger.info("📅 Phase 10: Événements économiques...")
        event_risks = await self._analyze_economic_events(position_analyses, deep_analysis) if deep_analysis else None

        # Calculer temps d'exécution
        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

        result = PortfolioAnalysisResult(
            analysis_id=analysis_id,
            user_id=user_id,
            analyzed_at=start_time,
            total_value=total_value,
            cash=portfolio.cash,
            num_positions=len(portfolio.positions),
            positions=position_analyses,
            correlations=correlations,
            risks=portfolio_risks,
            event_risks=event_risks,
            portfolio_sentiment=portfolio_sentiment,
            portfolio_sentiment_score=portfolio_sentiment_score,
            portfolio_health_score=portfolio_health,
            critical_alerts=critical_alerts,
            execution_time_ms=execution_time
        )

        logger.info(
            f"✅ Analyse terminée en {execution_time}ms | "
            f"Health: {portfolio_health:.0f}/100 | "
            f"Sentiment: {portfolio_sentiment} | "
            f"Alertes: {len(critical_alerts) if critical_alerts else 0}"
        )

        return result

    # ========================================================================
    # MÉTHODES D'ANALYSE PRIVÉES
    # ========================================================================

    def _identify_position_risks(
        self,
        ticker: str,
        data: AggregatedStockData,
        sentiment: Optional[SentimentTrend],
        weight: float
    ) -> List[str]:
        """Identifie les risques d'une position"""
        risks = []

        # Risque de concentration
        if weight > 30:
            risks.append(f"⚠️ Concentration élevée ({weight:.1f}% du portefeuille)")
        elif weight > 20:
            risks.append(f"Concentration significative ({weight:.1f}%)")

        # Risque de sentiment
        if sentiment:
            if sentiment.current_bearish_score > 70:
                risks.append(f"Sentiment très négatif ({sentiment.current_bearish_score:.0f}% bearish)")
            elif sentiment.source_consensus < 40:
                risks.append("Signaux de sentiment contradictoires entre sources")

        # Risque de volatilité (beta)
        if data.fundamentals and data.fundamentals.beta:
            if data.fundamentals.beta > 1.5:
                risks.append(f"Volatilité élevée (beta={data.fundamentals.beta:.2f})")

        # Risque de valorisation
        if data.fundamentals and data.fundamentals.pe_ratio:
            if data.fundamentals.pe_ratio > 50:
                risks.append(f"Valorisation élevée (P/E={data.fundamentals.pe_ratio:.1f})")

        return risks

    def _calculate_position_health(
        self,
        data: AggregatedStockData,
        sentiment: Optional[SentimentTrend],
        risks: List[str],
        ml_prediction: Optional[MLPrediction] = None
    ) -> float:
        """Calcule le health score d'une position (0-100)"""
        score = 70.0  # Score de base

        # Impact du sentiment
        if sentiment:
            if sentiment.current_sentiment == 'bullish':
                score += 15
            elif sentiment.current_sentiment == 'bearish':
                score -= 20

            # Consensus
            if sentiment.source_consensus > 70:
                score += 10
            elif sentiment.source_consensus < 40:
                score -= 10

        # Impact ML (NEW!) - pondération forte car basé sur données
        if ml_prediction:
            # Signal ML
            if ml_prediction.signal == 'BUY':
                score += (ml_prediction.signal_strength / 100) * 20  # Max +20
            elif ml_prediction.signal == 'SELL':
                score -= (ml_prediction.signal_strength / 100) * 25  # Max -25

            # Confiance consensus (si les 3 horizons sont d'accord)
            avg_conf = (ml_prediction.confidence_1d + ml_prediction.confidence_3d + ml_prediction.confidence_7d) / 3
            if avg_conf > 75:
                score += 10  # Haute confiance = bonus
            elif avg_conf < 50:
                score -= 5  # Faible confiance = malus

        # Impact des risques
        score -= len(risks) * 5

        # Limiter entre 0 et 100
        return max(0, min(100, score))

    def _analyze_correlations(
        self,
        positions: Dict[str, PositionAnalysis]
    ) -> PortfolioCorrelation:
        """Analyse les corrélations entre positions"""

        # Calculer concentration sectorielle
        sector_weights = {}
        for pos in positions.values():
            if pos.sector:
                sector_weights[pos.sector] = sector_weights.get(pos.sector, 0) + pos.portfolio_weight

        top_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else (None, 0)

        # Score de diversification basé sur concentration sectorielle
        if not sector_weights:
            diversification_score = 50.0
        else:
            # Plus la concentration est faible, mieux c'est
            max_concentration = max(sector_weights.values())
            if max_concentration < 25:
                diversification_score = 90
            elif max_concentration < 40:
                diversification_score = 70
            elif max_concentration < 50:
                diversification_score = 50
            else:
                diversification_score = 30

        # TODO: Implémenter vraie matrice de corrélation avec données historiques
        correlation_matrix = None
        highly_correlated = []

        return PortfolioCorrelation(
            correlation_matrix=correlation_matrix,
            highly_correlated_pairs=highly_correlated,
            diversification_score=diversification_score,
            sector_concentration=sector_weights,
            top_sector=top_sector[0],
            top_sector_weight=top_sector[1]
        )

    def _assess_portfolio_risks(
        self,
        positions: Dict[str, PositionAnalysis],
        correlations: Optional[PortfolioCorrelation]
    ) -> PortfolioRisk:
        """Évalue les risques globaux du portefeuille"""

        risk_factors = []
        mitigation_suggestions = []

        # Risque de concentration
        if correlations:
            if correlations.top_sector_weight > 50:
                concentration_risk = 'high'
                risk_factors.append(f"Concentration sectorielle élevée ({correlations.top_sector}: {correlations.top_sector_weight:.0f}%)")
                mitigation_suggestions.append(f"Diversifier hors du secteur {correlations.top_sector}")
            elif correlations.top_sector_weight > 35:
                concentration_risk = 'medium'
            else:
                concentration_risk = 'low'
        else:
            concentration_risk = 'medium'

        # Risque de sentiment
        bearish_count = sum(1 for p in positions.values() if p.sentiment and p.sentiment.current_sentiment == 'bearish')
        if bearish_count > len(positions) / 2:
            sentiment_risk = 'high'
            risk_factors.append(f"{bearish_count}/{len(positions)} positions avec sentiment bearish")
        elif bearish_count > len(positions) / 3:
            sentiment_risk = 'medium'
        else:
            sentiment_risk = 'low'

        # Risque de volatilité
        high_beta_count = sum(1 for p in positions.values() if p.beta and p.beta > 1.3)
        if high_beta_count > len(positions) / 2:
            volatility_risk = 'high'
            risk_factors.append(f"{high_beta_count}/{len(positions)} positions à haute volatilité")
            mitigation_suggestions.append("Ajouter des positions défensives (beta < 1)")
        elif high_beta_count > len(positions) / 3:
            volatility_risk = 'medium'
        else:
            volatility_risk = 'low'

        # Score de risque global
        risk_scores = {
            'low': 80,
            'medium': 50,
            'high': 20
        }

        overall_risk_score = np.mean([
            risk_scores[concentration_risk],
            risk_scores[sentiment_risk],
            risk_scores[volatility_risk]
        ])

        return PortfolioRisk(
            concentration_risk=concentration_risk,
            sentiment_risk=sentiment_risk,
            volatility_risk=volatility_risk,
            sector_risk=concentration_risk,  # Même que concentration pour l'instant
            overall_risk_score=overall_risk_score,
            risk_factors=risk_factors,
            risk_mitigation_suggestions=mitigation_suggestions
        )

    def _calculate_portfolio_sentiment(
        self,
        positions: Dict[str, PositionAnalysis]
    ) -> Tuple[str, float]:
        """Calcule le sentiment global du portefeuille"""

        bullish_weight = 0
        bearish_weight = 0
        total_weight = 0

        for pos in positions.values():
            if pos.sentiment:
                weight = pos.portfolio_weight
                if pos.sentiment.current_sentiment == 'bullish':
                    bullish_weight += weight
                elif pos.sentiment.current_sentiment == 'bearish':
                    bearish_weight += weight
                total_weight += weight

        if total_weight == 0:
            return 'neutral', 50.0

        bullish_pct = (bullish_weight / total_weight) * 100
        bearish_pct = (bearish_weight / total_weight) * 100

        if bullish_pct > bearish_pct + 20:
            sentiment = 'bullish'
            score = 50 + (bullish_pct - bearish_pct) / 2
        elif bearish_pct > bullish_pct + 20:
            sentiment = 'bearish'
            score = 50 - (bearish_pct - bullish_pct) / 2
        else:
            sentiment = 'neutral'
            score = 50

        return sentiment, min(100, max(0, score))

    def _calculate_portfolio_health(
        self,
        positions: Dict[str, PositionAnalysis],
        risks: PortfolioRisk,
        correlations: Optional[PortfolioCorrelation]
    ) -> float:
        """Calcule le health score global du portefeuille"""

        # Moyenne des health scores des positions
        if positions:
            avg_position_health = np.mean([p.health_score for p in positions.values()])
        else:
            avg_position_health = 50

        # Health basé sur risques
        risk_health = risks.overall_risk_score

        # Health basé sur diversification
        if correlations:
            diversification_health = correlations.diversification_score
        else:
            diversification_health = 50

        # Moyenne pondérée
        overall_health = (
            avg_position_health * 0.5 +
            risk_health * 0.3 +
            diversification_health * 0.2
        )

        return overall_health

    def _identify_critical_alerts(
        self,
        positions: Dict[str, PositionAnalysis],
        risks: PortfolioRisk,
        correlations: Optional[PortfolioCorrelation]
    ) -> List[str]:
        """Identifie les alertes critiques nécessitant une action"""

        alerts = []

        # Alertes de risque
        if risks.overall_risk_score < 40:
            alerts.append("🚨 RISQUE ÉLEVÉ: Score de risque global faible (<40)")

        if risks.concentration_risk == 'high':
            alerts.append(f"⚠️ Concentration sectorielle critique: {correlations.top_sector} ({correlations.top_sector_weight:.0f}%)")

        if risks.sentiment_risk == 'high':
            alerts.append("⚠️ Sentiment majoritairement négatif sur le portefeuille")

        # Alertes par position
        for ticker, pos in positions.items():
            if pos.portfolio_weight > 30:
                alerts.append(f"⚠️ {ticker}: Position trop concentrée ({pos.portfolio_weight:.0f}%)")

            if pos.sentiment and pos.sentiment.current_bearish_score > 80:
                alerts.append(f"🚨 {ticker}: Sentiment extrêmement négatif ({pos.sentiment.current_bearish_score:.0f}% bearish)")

            if pos.health_score < 30:
                alerts.append(f"⚠️ {ticker}: Health score critique ({pos.health_score:.0f}/100)")

        return alerts

    async def _analyze_economic_events(
        self,
        positions: Dict[str, PositionAnalysis],
        deep_analysis: bool
    ) -> Optional[PortfolioEventRisk]:
        """
        Analyse les événements économiques à venir et leur impact sur le portefeuille

        Args:
            positions: Positions du portefeuille
            deep_analysis: Si True, analyse approfondie

        Returns:
            PortfolioEventRisk avec risques liés aux événements
        """
        try:
            # Extraire positions et secteurs
            portfolio_positions = {
                ticker: {'sector': pos.sector, 'weight': pos.portfolio_weight}
                for ticker, pos in positions.items()
                if pos.sector
            }

            portfolio_sectors = list(set(pos.sector for pos in positions.values() if pos.sector))

            # Analyser risques événementiels sur 7 jours
            event_risk = await self.event_predictor.analyze_portfolio_event_risk(
                portfolio_positions=portfolio_positions,
                portfolio_sectors=portfolio_sectors,
                days_ahead=7
            )

            logger.info(
                f"   → {event_risk.total_events} événements | "
                f"{event_risk.critical_events} critiques | "
                f"Risque: {event_risk.overall_risk_score:.0f}/100"
            )

            return event_risk

        except Exception as e:
            logger.error(f"Erreur analyse événements économiques: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


# Singleton
_portfolio_analyzer_instance = None

def get_portfolio_analyzer() -> PortfolioAnalyzer:
    """Retourne l'instance singleton"""
    global _portfolio_analyzer_instance
    if _portfolio_analyzer_instance is None:
        _portfolio_analyzer_instance = PortfolioAnalyzer()
    return _portfolio_analyzer_instance
