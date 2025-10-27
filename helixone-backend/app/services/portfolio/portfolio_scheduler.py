"""
Portfolio Scheduler - Orchestration automatique des analyses
Exécute l'analyse complète du portefeuille 2x par jour :
- 7h00 EST (avant ouverture US 9h30)
- 17h00 EST (après clôture US 16h00)

Workflow complet :
1. Récupérer portefeuille utilisateur
2. Data Aggregator : Collecter toutes les données
3. Portfolio Analyzer : Analyser le portefeuille
4. Scenario Predictor : Prédire les mouvements
5. Recommendation Engine : Générer recommandations
6. Alert System : Créer alertes
7. Notification Service : Envoyer notifications
8. Sauvegarder en DB
"""

import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pytz

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import desc

from app.services.portfolio.data_aggregator import get_data_aggregator
from app.services.portfolio.sentiment_aggregator import get_sentiment_aggregator
from app.services.portfolio.portfolio_analyzer import get_portfolio_analyzer
from app.services.portfolio.ml_signal_service import get_ml_signal_service
from app.services.portfolio.recommendation_engine import get_recommendation_engine
from app.services.portfolio.alert_system import get_alert_system

# Import pour sauvegarde DB
from app.core.database import SessionLocal
from app.models.portfolio import (
    PortfolioAnalysisHistory,
    PortfolioAlert,
    PortfolioRecommendation,
    AnalysisTimeType,
    AlertSeverity,
    RecommendationType,
    AlertStatus
)

logger = logging.getLogger(__name__)


class PortfolioScheduler:
    """
    Orchestrateur d'analyses de portefeuille
    Exécute automatiquement les analyses 2x/jour
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone=pytz.timezone('America/New_York'))

        # Services
        self.data_aggregator = get_data_aggregator()
        self.sentiment_aggregator = get_sentiment_aggregator()
        self.portfolio_analyzer = get_portfolio_analyzer()
        self.ml_signal_service = get_ml_signal_service()
        self.recommendation_engine = get_recommendation_engine()
        self.alert_system = get_alert_system()

        # État
        self.is_running = False
        self.last_analysis = {}

        logger.info("PortfolioScheduler initialisé")

    def start(self):
        """Démarre le scheduler"""
        if self.is_running:
            logger.warning("Scheduler déjà démarré")
            return

        # Morning analysis : 7h00 EST (avant ouverture US 9h30)
        self.scheduler.add_job(
            self.run_morning_analysis,
            CronTrigger(hour=7, minute=0, timezone='America/New_York'),
            id='morning_analysis',
            name='Morning Portfolio Analysis',
            replace_existing=True
        )

        # Evening analysis : 17h00 EST (après clôture US 16h00)
        self.scheduler.add_job(
            self.run_evening_analysis,
            CronTrigger(hour=17, minute=0, timezone='America/New_York'),
            id='evening_analysis',
            name='Evening Portfolio Analysis',
            replace_existing=True
        )

        self.scheduler.start()
        self.is_running = True

        logger.info("✅ Scheduler démarré (7h00 + 17h00 EST)")

    def stop(self):
        """Arrête le scheduler"""
        if not self.is_running:
            return

        self.scheduler.shutdown()
        self.is_running = False
        logger.info("Scheduler arrêté")

    async def run_morning_analysis(self):
        """Exécute l'analyse matinale pour tous les utilisateurs actifs"""
        logger.info("🌅 === ANALYSE MATINALE 7h00 EST ===")

        try:
            # Récupérer tous les utilisateurs avec portfolios actifs
            users_with_portfolios = await self._get_all_active_users()

            if not users_with_portfolios:
                logger.info("Aucun utilisateur avec portefeuille actif")
                return

            logger.info(f"📊 {len(users_with_portfolios)} utilisateur(s) à analyser")

            # Analyser chaque utilisateur en séquence
            for user_id in users_with_portfolios:
                try:
                    portfolio = await self._get_user_portfolio(user_id)

                    if portfolio and len(portfolio.positions) > 0:
                        logger.info(f"🔍 Analyse matinale pour {user_id}")
                        await self._run_complete_analysis(user_id, portfolio, "morning")
                    else:
                        logger.warning(f"Portefeuille vide pour {user_id}")

                except Exception as e:
                    logger.error(f"Erreur analyse {user_id}: {e}")
                    continue

            logger.info("✅ Analyses matinales terminées")

        except Exception as e:
            logger.error(f"Erreur analyse matinale: {e}", exc_info=True)

    async def run_evening_analysis(self):
        """Exécute l'analyse du soir pour tous les utilisateurs actifs"""
        logger.info("🌆 === ANALYSE DU SOIR 17h00 EST ===")

        try:
            # Récupérer tous les utilisateurs avec portfolios actifs
            users_with_portfolios = await self._get_all_active_users()

            if not users_with_portfolios:
                logger.info("Aucun utilisateur avec portefeuille actif")
                return

            logger.info(f"📊 {len(users_with_portfolios)} utilisateur(s) à analyser")

            # Analyser chaque utilisateur en séquence
            for user_id in users_with_portfolios:
                try:
                    portfolio = await self._get_user_portfolio(user_id)

                    if portfolio and len(portfolio.positions) > 0:
                        logger.info(f"🔍 Analyse du soir pour {user_id}")
                        await self._run_complete_analysis(user_id, portfolio, "evening")
                    else:
                        logger.warning(f"Portefeuille vide pour {user_id}")

                except Exception as e:
                    logger.error(f"Erreur analyse {user_id}: {e}")
                    continue

            logger.info("✅ Analyses du soir terminées")

        except Exception as e:
            logger.error(f"Erreur analyse du soir: {e}", exc_info=True)

    # ========================================================================
    # MÉTHODES AUXILIAIRES - Récupération utilisateurs/portfolios
    # ========================================================================

    async def _get_all_active_users(self) -> List[str]:
        """
        Récupère la liste de tous les utilisateurs avec portfolios actifs

        Returns:
            Liste des user_id avec portfolios IBKR actifs
        """
        db = SessionLocal()
        try:
            # Récupérer les utilisateurs avec connexions IBKR actives
            from app.models.ibkr import IBKRConnection

            active_connections = db.query(IBKRConnection).filter(
                IBKRConnection.is_connected == True,
                IBKRConnection.auto_connect == True
            ).all()

            user_ids = [conn.user_id for conn in active_connections]

            logger.info(f"✅ {len(user_ids)} utilisateur(s) avec connexion IBKR active")
            return user_ids

        except Exception as e:
            logger.error(f"Erreur récupération utilisateurs: {e}")
            return []
        finally:
            db.close()

    async def _get_user_portfolio(self, user_id: str) -> Optional[Dict]:
        """
        Récupère le portefeuille d'un utilisateur depuis IBKR

        Returns:
            Portfolio avec positions actuelles ou None
        """
        db = SessionLocal()
        try:
            # Récupérer les positions IBKR de l'utilisateur
            from app.models.ibkr import IBKRPosition, PortfolioSnapshot

            # Récupérer le dernier snapshot
            snapshot = db.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.user_id == user_id
            ).order_by(desc(PortfolioSnapshot.created_at)).first()

            if not snapshot:
                logger.warning(f"Pas de snapshot pour {user_id}")
                return None

            # Récupérer les positions du snapshot
            positions_records = db.query(IBKRPosition).filter(
                IBKRPosition.snapshot_id == snapshot.id,
                IBKRPosition.position != 0  # Uniquement positions non nulles
            ).all()

            if not positions_records:
                logger.warning(f"Pas de positions pour {user_id}")
                return None

            # Convertir en Portfolio
            positions = {}
            for pos in positions_records:
                if pos.symbol and pos.position > 0:
                    positions[pos.symbol] = float(pos.position)

            portfolio = Portfolio(
                positions=positions,
                cash=float(snapshot.total_cash) if snapshot.total_cash else 0.0
            )

            logger.info(f"✅ Portfolio {user_id}: {len(positions)} positions, ${portfolio.cash:.2f} cash")
            return portfolio

        except Exception as e:
            logger.error(f"Erreur récupération portfolio {user_id}: {e}")
            return None
        finally:
            db.close()

    async def _run_complete_analysis(
        self,
        user_id: str,
        portfolio: Dict,
        analysis_time: str
    ):
        """
        Exécute le workflow complet d'analyse

        Args:
            user_id: ID utilisateur
            portfolio: Portfolio à analyser
            analysis_time: "morning" ou "evening"
        """
        start_time = datetime.now()

        logger.info(
            f"🔍 Analyse {analysis_time} pour {user_id} "
            f"({len(portfolio.positions)} positions)"
        )

        try:
            # ================================================================
            # ÉTAPE 1 : Collecter toutes les données
            # ================================================================
            logger.info("📊 ÉTAPE 1/6 : Collecte données multi-sources...")

            tickers = list(portfolio.positions.keys())

            # Collecter données pour chaque ticker
            stock_data = await self.data_aggregator.aggregate_multiple_stocks(
                tickers,
                include_sentiment=True,
                include_news=True,
                include_fundamentals=True
            )

            logger.info(f"✅ Données collectées pour {len(stock_data)} tickers")

            # ================================================================
            # ÉTAPE 2 : Analyser sentiment en détail
            # ================================================================
            logger.info("💬 ÉTAPE 2/6 : Analyse sentiment approfondie...")

            sentiment_trends = {}
            for ticker in tickers:
                try:
                    trend = self.sentiment_aggregator.analyze_sentiment_trend(
                        ticker,
                        lookback_days=7
                    )
                    sentiment_trends[ticker] = trend
                except Exception as e:
                    logger.error(f"Erreur sentiment {ticker}: {e}")

            logger.info(f"✅ Sentiment analysé pour {len(sentiment_trends)} tickers")

            # ================================================================
            # ÉTAPE 3 : Analyse complète du portefeuille
            # ================================================================
            logger.info("🔬 ÉTAPE 3/6 : Analyse complète portefeuille...")

            analysis = await self.portfolio_analyzer.analyze_portfolio(
                portfolio,
                user_id=user_id,
                deep_analysis=True
            )

            logger.info(
                f"✅ Analyse terminée - Health score: {analysis.portfolio_health_score:.0f}/100"
            )

            # ================================================================
            # ÉTAPE 4 : Prédictions ML
            # ================================================================
            logger.info("🔮 ÉTAPE 4/6 : Prédictions ML (XGBoost + LSTM)...")

            # Récupérer les tickers et prix actuels du portfolio
            tickers = list(portfolio.positions.keys()) if hasattr(portfolio, 'positions') else []
            current_prices = {
                ticker: data.current_price
                for ticker, data in stock_data.items()
                if hasattr(data, 'current_price')
            }

            # Obtenir prédictions ML pour tout le portfolio
            ml_predictions = await self.ml_signal_service.get_portfolio_signals(
                tickers=tickers,
                current_prices=current_prices
            )

            logger.info(
                f"✅ Prédictions ML générées - Bullish: {ml_predictions.bullish_count}, "
                f"Bearish: {ml_predictions.bearish_count}, Confiance moyenne: {ml_predictions.avg_confidence:.1f}%"
            )

            # ================================================================
            # ÉTAPE 5 : Générer recommandations
            # ================================================================
            logger.info("🎯 ÉTAPE 5/6 : Génération recommandations...")

            recommendations = self.recommendation_engine.generate_recommendations(
                portfolio,
                analysis,
                predictions
            )

            logger.info(
                f"✅ {len(recommendations.position_recommendations)} recommandations générées"
            )

            # ================================================================
            # ÉTAPE 6 : Créer alertes
            # ================================================================
            logger.info("🔔 ÉTAPE 6/6 : Création alertes...")

            alert_batch = self.alert_system.generate_alerts(
                analysis,
                predictions,
                recommendations,
                analysis_time
            )

            logger.info(
                f"✅ {alert_batch.total_alerts} alertes créées "
                f"({len(alert_batch.critical_alerts)} critiques)"
            )

            # ================================================================
            # ÉTAPE 7 : Sauvegarder en base de données
            # ================================================================
            logger.info("💾 Sauvegarde résultats...")

            await self._save_analysis_results(
                user_id,
                analysis,
                predictions,
                recommendations,
                alert_batch,
                analysis_time
            )

            # ================================================================
            # ÉTAPE 8 : Envoyer notifications
            # ================================================================
            logger.info("📲 Envoi notifications...")

            await self._send_notifications(user_id, alert_batch)

            # ================================================================
            # FIN
            # ================================================================
            execution_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"✅ === ANALYSE {analysis_time.upper()} TERMINÉE ===\n"
                f"   User: {user_id}\n"
                f"   Positions: {len(portfolio.positions)}\n"
                f"   Health: {analysis.portfolio_health_score:.0f}/100\n"
                f"   Alertes: {alert_batch.total_alerts} ({len(alert_batch.critical_alerts)} critiques)\n"
                f"   Temps: {execution_time:.1f}s"
            )

            # Stocker résultat
            self.last_analysis[user_id] = {
                'time': analysis_time,
                'timestamp': datetime.now(),
                'alerts_count': alert_batch.total_alerts,
                'critical_count': len(alert_batch.critical_alerts),
                'health_score': analysis.portfolio_health_score
            }

        except Exception as e:
            logger.error(f"❌ Erreur workflow complet: {e}", exc_info=True)
            # TODO: Envoyer alerte d'erreur à l'admin

    # ========================================================================
    # SAUVEGARDE & NOTIFICATIONS
    # ========================================================================

    async def _save_analysis_results(
        self,
        user_id: str,
        analysis,
        predictions,
        recommendations,
        alert_batch,
        analysis_time: str = "manual"
    ):
        """
        Sauvegarde les résultats en base de données

        Args:
            user_id: ID de l'utilisateur
            analysis: Résultat de l'analyse de portefeuille
            predictions: Prédictions de scénario
            recommendations: Recommandations générées
            alert_batch: Batch d'alertes générées
            analysis_time: Type d'analyse (morning/evening/manual)
        """
        db = SessionLocal()
        try:
            logger.info(f"💾 Sauvegarde résultats pour {user_id} en DB...")

            # ================================================================
            # 1. Créer l'enregistrement de l'analyse principale
            # ================================================================

            # Mapper analysis_time string vers enum
            analysis_time_enum = {
                'morning': AnalysisTimeType.MORNING,
                'evening': AnalysisTimeType.EVENING,
                'manual': AnalysisTimeType.MANUAL,
                'on_demand': AnalysisTimeType.ON_DEMAND
            }.get(analysis_time, AnalysisTimeType.MANUAL)

            # Convertir les données en JSON pour stockage (uniquement types simples)
            positions_data = {
                ticker: {
                    'sentiment': pos.sentiment if hasattr(pos, 'sentiment') and isinstance(pos.sentiment, str) else None,
                    'risk_level': pos.risk_level if hasattr(pos, 'risk_level') and isinstance(pos.risk_level, str) else None,
                }
                for ticker, pos in analysis.positions.items()
            }

            correlations_data = None
            if analysis.correlations:
                # Convertir tuples en listes pour JSON
                highly_correlated = []
                if hasattr(analysis.correlations, 'highly_correlated_pairs') and analysis.correlations.highly_correlated_pairs:
                    highly_correlated = [
                        {'ticker1': pair[0], 'ticker2': pair[1], 'correlation': pair[2]}
                        for pair in analysis.correlations.highly_correlated_pairs
                    ]

                correlations_data = {
                    'diversification_score': analysis.correlations.diversification_score,
                    'sector_concentration': analysis.correlations.sector_concentration if analysis.correlations.sector_concentration else {},
                    'top_sector': analysis.correlations.top_sector,
                    'top_sector_weight': analysis.correlations.top_sector_weight,
                    'highly_correlated_pairs': highly_correlated,
                }

            risks_data = None
            if analysis.risks:
                risks_data = {
                    'concentration_risk': analysis.risks.concentration_risk,
                    'sentiment_risk': analysis.risks.sentiment_risk,
                    'volatility_risk': analysis.risks.volatility_risk,
                    'sector_risk': analysis.risks.sector_risk,
                    'overall_risk_score': analysis.risks.overall_risk_score,
                    'risk_factors': analysis.risks.risk_factors if analysis.risks.risk_factors else [],
                }

            predictions_data = {
                ticker: {
                    'direction': pred.overall_direction,
                    'confidence': pred.overall_confidence,
                    'return_1d': pred.prediction_1d.expected_move_pct,
                    'return_3d': pred.prediction_3d.expected_move_pct,
                    'return_7d': pred.prediction_7d.expected_move_pct,
                }
                for ticker, pred in predictions.stock_predictions.items()
            }

            # Créer l'historique d'analyse
            analysis_record = PortfolioAnalysisHistory(
                user_id=user_id,
                analysis_time=analysis_time_enum,
                num_positions=analysis.num_positions,
                total_value=analysis.total_value,
                cash_amount=0.0,  # TODO: Récupérer depuis portfolio
                health_score=analysis.portfolio_health_score,
                portfolio_sentiment=analysis.portfolio_sentiment,
                expected_return_7d=predictions.portfolio_expected_return_7d,
                downside_risk_pct=predictions.portfolio_downside_risk_7d,
                num_alerts=alert_batch.total_alerts,
                num_critical_alerts=len(alert_batch.critical_alerts),
                num_recommendations=len(recommendations.position_recommendations),
                execution_time_seconds=(datetime.now() - datetime.now()).total_seconds(),  # TODO: Track real time
                data_sources_used=['reddit', 'stocktwits', 'finnhub', 'alphavantage'],
                positions_data=positions_data,
                correlations_data=correlations_data,
                risks_data=risks_data,
                predictions_data=predictions_data
            )

            db.add(analysis_record)
            db.flush()  # Pour obtenir l'ID

            analysis_id = analysis_record.id
            logger.info(f"✅ Analyse sauvegardée: {analysis_id}")

            # ================================================================
            # 2. Sauvegarder toutes les alertes
            # ================================================================

            all_alerts = (
                alert_batch.critical_alerts +
                alert_batch.warning_alerts +
                alert_batch.opportunity_alerts +
                alert_batch.info_alerts
            )

            for alert in all_alerts:
                alert_record = PortfolioAlert(
                    analysis_id=analysis_id,
                    user_id=user_id,
                    severity=alert.severity,
                    status=AlertStatus.NEW,
                    ticker=alert.ticker,
                    title=alert.title,
                    message=alert.message,
                    action_required=alert.action_required,
                    confidence=alert.confidence,
                    push_notification=str(alert.push_notification),
                    extra_data={}
                )
                db.add(alert_record)

            logger.info(f"✅ {len(all_alerts)} alertes sauvegardées")

            # ================================================================
            # 3. Sauvegarder toutes les recommandations
            # ================================================================

            for rec in recommendations.position_recommendations:
                # Mapper action vers enum
                action_enum = {
                    'STRONG_BUY': RecommendationType.STRONG_BUY,
                    'BUY': RecommendationType.BUY,
                    'HOLD': RecommendationType.HOLD,
                    'SELL': RecommendationType.SELL,
                    'STRONG_SELL': RecommendationType.STRONG_SELL
                }.get(rec.action, RecommendationType.HOLD)

                # Récupérer les prédictions pour ce ticker
                stock_pred = predictions.stock_predictions.get(rec.ticker)

                rec_record = PortfolioRecommendation(
                    analysis_id=analysis_id,
                    user_id=user_id,
                    ticker=rec.ticker,
                    action=action_enum,
                    confidence=rec.confidence,
                    current_price=None,  # TODO: Récupérer prix actuel
                    target_price=rec.target_price,
                    stop_loss=rec.stop_loss,
                    primary_reason=rec.primary_reason,
                    detailed_reasons=rec.detailed_reasons,
                    risk_factors=rec.risk_factors,
                    suggested_action=rec.suggested_action,
                    prediction_1d=stock_pred.prediction_1d.expected_move_pct if stock_pred else None,
                    prediction_3d=stock_pred.prediction_3d.expected_move_pct if stock_pred else None,
                    prediction_7d=stock_pred.prediction_7d.expected_move_pct if stock_pred else None,
                    sentiment_score=None,  # TODO: Récupérer score sentiment
                    technical_score=None,
                    fundamental_score=None,
                    extra_data={},
                    expires_at=datetime.utcnow() + timedelta(days=7)  # Expire dans 7 jours
                )
                db.add(rec_record)

            logger.info(f"✅ {len(recommendations.position_recommendations)} recommandations sauvegardées")

            # Commit final
            db.commit()
            logger.info("✅ Toutes les données sauvegardées avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde DB: {e}", exc_info=True)
            db.rollback()
            raise
        finally:
            db.close()

    async def _send_notifications(self, user_id: str, alert_batch):
        """
        Envoie les notifications push

        TODO: Implémenter avec Firebase Cloud Messaging

        - Envoyer notifications pour alertes critiques
        - Envoyer résumé pour high priority
        - Grouper les notifications
        """
        logger.info(f"TODO: Envoyer notifications push à {user_id}")

        # Compter notifications à envoyer
        push_count = sum(
            1 for alert in (
                alert_batch.critical_alerts +
                alert_batch.warning_alerts +
                alert_batch.opportunity_alerts
            )
            if alert.push_notification
        )

        logger.info(f"📲 {push_count} notifications à envoyer")

    async def run_manual_analysis(
        self,
        user_id: str,
        portfolio: Dict
    ) -> dict:
        """
        Lance une analyse manuelle (pour API endpoint)

        Args:
            user_id: ID utilisateur
            portfolio: Portfolio à analyser

        Returns:
            Dict avec tous les résultats
        """
        logger.info(f"🔍 Analyse manuelle pour {user_id}")

        await self._run_complete_analysis(user_id, portfolio, "manual")

        # Retourner les résultats
        return self.last_analysis.get(user_id, {})


# Singleton
_scheduler_instance = None

def get_portfolio_scheduler() -> PortfolioScheduler:
    """Retourne l'instance singleton du scheduler"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = PortfolioScheduler()
    return _scheduler_instance


# Démarrage automatique au lancement de l'application
def start_scheduler():
    """Démarre le scheduler (appelé au startup de l'app)"""
    scheduler = get_portfolio_scheduler()
    scheduler.start()
    logger.info("🚀 Portfolio Scheduler démarré automatiquement")


# Arrêt propre
def stop_scheduler():
    """Arrête le scheduler (appelé au shutdown de l'app)"""
    scheduler = get_portfolio_scheduler()
    scheduler.stop()
    logger.info("⏸️ Portfolio Scheduler arrêté")
