"""
Endpoints API pour les analyses FXI
"""

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
import logging

from app.services.analysis.fxi_adapter import get_fxi_adapter
from app.services.data_sources.base import (
    DataUnavailableError,
    InvalidTickerError
)
from app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisMode
)
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import Analysis, User

router = APIRouter()
logger = logging.getLogger(__name__)

# Import des services d'analyse complète (utilisés dans l'analyse automatique)
from app.services.portfolio.data_aggregator import get_data_aggregator
from app.services.portfolio.sentiment_aggregator import get_sentiment_aggregator
from app.services.portfolio.portfolio_analyzer import get_portfolio_analyzer
from app.services.portfolio.ml_signal_service import get_ml_signal_service
from app.services.portfolio.recommendation_engine import get_recommendation_engine
from app.services.portfolio.alert_system import get_alert_system
from app.services.economic_calendar_service import get_economic_calendar_service


@router.post("/complete", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_complete(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Analyse complète FXI d'une action

    Utilise le moteur FXI v2.0 avec 5 dimensions d'analyse :
    - Technique : Indicateurs techniques, patterns, signaux
    - Fondamentale : Ratios financiers, croissance, rentabilité
    - Sentiment : Analyse du sentiment de marché
    - Risque : Volatilité, endettement, liquidité
    - Macro : Environnement macro-économique

    Args:
        request: Requête avec ticker et mode
        current_user: Utilisateur authentifié

    Returns:
        AnalysisResult avec scores et recommandation

    Raises:
        404: Ticker non trouvé
        503: Données non disponibles
    """
    try:
        logger.info(
            f"Analyse complète demandée par {current_user.email} "
            f"pour {request.ticker} en mode {request.mode}"
        )

        # Récupérer l'adaptateur FXI
        fxi_adapter = get_fxi_adapter()

        # Exécuter l'analyse
        result = await fxi_adapter.analyze_ticker(
            ticker=request.ticker,
            mode=request.mode.value
        )

        # Sauvegarder l'analyse en DB
        analysis_record = Analysis(
            user_id=current_user.id,
            ticker=request.ticker,
            mode=request.mode.value,
            score_final=result.final_score,
            score_technique=result.technical_score,
            score_fondamental=result.fundamental_score,
            score_sentiment=result.sentiment_score,
            score_risque=result.risk_score,
            score_macro=result.macro_score,
            recommendation=result.recommendation,
            confidence=result.confidence,
            execution_time=result.execution_time,
            result_json=result.details or {}
        )

        db.add(analysis_record)
        db.commit()

        logger.info(
            f"Analyse terminée pour {request.ticker}: "
            f"Score {result.final_score}, {result.recommendation}"
        )

        return result

    except InvalidTickerError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{request.ticker}' non trouvé"
        )

    except DataUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Données non disponibles pour '{request.ticker}'"
        )

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de {request.ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )


@router.get("/history", tags=["Analysis"])
async def get_analysis_history(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Récupère l'historique des analyses de l'utilisateur

    Args:
        limit: Nombre max d'analyses à retourner
        current_user: Utilisateur authentifié

    Returns:
        Liste des analyses récentes
    """
    try:
        analyses = db.query(Analysis).filter(
            Analysis.user_id == current_user.id
        ).order_by(
            Analysis.created_at.desc()
        ).limit(limit).all()

        return {
            "analyses": [
                {
                    "id": a.id,
                    "ticker": a.ticker,
                    "mode": a.mode,
                    "score_final": a.score_final,
                    "recommendation": a.recommendation,
                    "created_at": a.created_at
                }
                for a in analyses
            ],
            "total": len(analyses)
        }

    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur serveur"
        )


@router.get("/modes", tags=["Analysis"])
async def get_analysis_modes():
    """
    Retourne les modes d'analyse disponibles

    Returns:
        Liste des modes avec descriptions
    """
    return {
        "modes": [
            {
                "name": "Standard",
                "description": "Mode équilibré pour investisseurs diversifiés",
                "weights": {
                    "technical": 0.25,
                    "fundamental": 0.30,
                    "sentiment": 0.20,
                    "risk": 0.10,
                    "macro": 0.15
                }
            },
            {
                "name": "Conservative",
                "description": "Favorise les fondamentaux et la prudence",
                "weights": {
                    "technical": 0.20,
                    "fundamental": 0.35,
                    "sentiment": 0.15,
                    "risk": 0.15,
                    "macro": 0.15
                }
            },
            {
                "name": "Aggressive",
                "description": "Favorise l'analyse technique et le momentum",
                "weights": {
                    "technical": 0.35,
                    "fundamental": 0.25,
                    "sentiment": 0.20,
                    "risk": 0.10,
                    "macro": 0.10
                }
            }
        ]
    }


@router.post("/ml-enhanced", tags=["Analysis"])
async def analyze_ml_enhanced(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Analyse complète avec prédictions ML et recommandations

    Combine :
    - Analyse FXI (5 dimensions)
    - Prédictions ML (XGBoost + LSTM, 3 horizons: 1j, 3j, 7j)
    - Recommandations actionnables (BUY/HOLD/SELL)
    - Health Score (0-100)
    - Alerts automatiques

    Args:
        request: Requête avec ticker et mode
        current_user: Utilisateur authentifié

    Returns:
        Analyse complète avec prédictions ML intégrées
    """
    try:
        logger.info(
            f"Analyse ML enhanced demandée par {current_user.email} pour {request.ticker}"
        )

        # 1. Exécuter analyse FXI classique
        fxi_adapter = get_fxi_adapter()
        fxi_result = await fxi_adapter.analyze_ticker(
            ticker=request.ticker,
            mode=request.mode.value
        )

        # 2. Obtenir prédictions ML
        from app.services.portfolio.ml_signal_service import get_ml_signal_service

        ml_service = get_ml_signal_service()
        ml_prediction = await ml_service.get_prediction(request.ticker)

        # 3. Construire résultat combiné
        result = {
            # Scores FXI
            "score_fxi": fxi_result.final_score,
            "score_technique": fxi_result.technical_score,
            "score_fondamental": fxi_result.fundamental_score,
            "score_sentiment": fxi_result.sentiment_score,
            "score_risque": fxi_result.risk_score,
            "score_macro": fxi_result.macro_score,
            "recommandation": fxi_result.recommendation,
            "confidence": fxi_result.confidence,
            "details": fxi_result.details,

            # Prédictions ML (nouveau!)
            "ml_predictions": {
                "signal": ml_prediction.signal,  # BUY/HOLD/SELL
                "signal_strength": ml_prediction.signal_strength,
                "prediction_1d": ml_prediction.prediction_1d,
                "confidence_1d": ml_prediction.confidence_1d,
                "prediction_3d": ml_prediction.prediction_3d,
                "confidence_3d": ml_prediction.confidence_3d,
                "prediction_7d": ml_prediction.prediction_7d,
                "confidence_7d": ml_prediction.confidence_7d,
                "model_version": ml_prediction.model_version,
                "generated_at": ml_prediction.generated_at.isoformat()
            },

            # Health Score global (moyenne FXI + ML)
            "health_score": round((fxi_result.final_score + ml_prediction.signal_strength) / 2, 1),

            # Recommandation finale (consensus FXI + ML)
            "recommendation_final": _consensus_recommendation(
                fxi_result.recommendation,
                ml_prediction.signal
            ),

            # Métadonnées
            "ticker": request.ticker,
            "execution_time": fxi_result.execution_time,
            "timestamp": fxi_result.details.get("timestamp") if fxi_result.details else None
        }

        # 4. Sauvegarder en DB
        analysis_record = Analysis(
            user_id=current_user.id,
            ticker=request.ticker,
            mode=request.mode.value,
            score_final=result["health_score"],
            score_technique=fxi_result.technical_score,
            score_fondamental=fxi_result.fundamental_score,
            score_sentiment=fxi_result.sentiment_score,
            score_risque=fxi_result.risk_score,
            score_macro=fxi_result.macro_score,
            recommendation=result["recommendation_final"],
            confidence=fxi_result.confidence,
            execution_time=fxi_result.execution_time,
            result_json=result
        )

        db.add(analysis_record)
        db.commit()

        logger.info(
            f"Analyse ML terminée pour {request.ticker}: "
            f"Health Score {result['health_score']}, Recommandation {result['recommendation_final']}"
        )

        return result

    except InvalidTickerError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{request.ticker}' non trouvé"
        )

    except Exception as e:
        logger.error(f"Erreur analyse ML de {request.ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )


def _consensus_recommendation(fxi_rec: str, ml_signal: str) -> str:
    """
    Détermine recommandation finale basée sur consensus FXI + ML

    Règles:
    - Si les 2 sont d'accord → même recommandation
    - Si divergent mais pas opposés → HOLD
    - Si opposés → prioriser ML (plus récent)
    """
    # Normaliser
    fxi_rec = fxi_rec.upper()
    ml_signal = ml_signal.upper()

    # Mapping
    buy_signals = ['BUY', 'STRONG_BUY', 'ACHETER']
    hold_signals = ['HOLD', 'CONSERVER']
    sell_signals = ['SELL', 'STRONG_SELL', 'VENDRE']

    fxi_is_buy = fxi_rec in buy_signals
    fxi_is_hold = fxi_rec in hold_signals
    fxi_is_sell = fxi_rec in sell_signals

    ml_is_buy = ml_signal in buy_signals
    ml_is_hold = ml_signal in hold_signals
    ml_is_sell = ml_signal in sell_signals

    # Consensus
    if (fxi_is_buy and ml_is_buy):
        return "BUY"
    elif (fxi_is_sell and ml_is_sell):
        return "SELL"
    elif (fxi_is_hold and ml_is_hold):
        return "HOLD"
    elif (fxi_is_buy and ml_is_sell) or (fxi_is_sell and ml_is_buy):
        # Opposés → prioriser ML
        return ml_signal
    else:
        # Divergent mais pas opposés → HOLD
        return "HOLD"


# ============================================================================
# NOUVEL ENDPOINT: ANALYSE ULTRA-COMPLÈTE (comme analyse automatique)
# ============================================================================

@router.post("/stock-deep-analysis", tags=["Analysis"])
async def stock_deep_analysis(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Analyse ULTRA-COMPLÈTE d'une action unique
    
    Utilise TOUT le système d'analyse automatique (8 étapes):
    1. Collecte données multi-sources (35+ sources)
    2. Analyse sentiment approfondie (tendances, vélocité)
    3. Analyse complète (corrélations, risques)
    4. Prédictions ML (XGBoost + LSTM)
    5. Recommandations détaillées (actions concrètes)
    6. Alertes intelligentes (critique/important/info)
    7. Événements économiques à venir
    8. Métriques de performance
    
    C'est la MÊME analyse que celle faite automatiquement matin/soir,
    mais déclenchée à la demande pour une action spécifique.
    
    Returns:
        Analyse ultra-détaillée avec:
        - Toutes les données collectées (prix, sentiment, news, trends, etc.)
        - Prédictions ML multi-horizon (1j, 3j, 7j)
        - Recommandation détaillée avec explications
        - Alertes et risques identifiés
        - Événements économiques impactants
        - Actions concrètes à prendre
    """
    import time
    from datetime import datetime
    
    start_time = time.time()
    ticker = request.ticker.upper()
    
    logger.info(f"🚀 Analyse ultra-complète demandée pour {ticker} par {current_user.email}")
    
    try:
        # Initialiser tous les services
        data_aggregator = get_data_aggregator()
        sentiment_aggregator = get_sentiment_aggregator()
        portfolio_analyzer = get_portfolio_analyzer()
        ml_signal_service = get_ml_signal_service()
        recommendation_engine = get_recommendation_engine()
        alert_system = get_alert_system()
        calendar_service = get_economic_calendar_service()
        
        # ====================================================================
        # ÉTAPE 1/8: Collecte de Données Multi-Sources (35+ sources)
        # ====================================================================
        logger.info(f"📊 ÉTAPE 1/8: Collecte données multi-sources pour {ticker}...")
        
        stock_data = await data_aggregator.aggregate_stock_data(
            ticker,
            include_sentiment=True,
            include_news=True,
            include_fundamentals=True,
            include_trends=True
        )
        
        if not stock_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Impossible de collecter les données pour {ticker}"
            )
        
        logger.info(f"✅ Données collectées pour {ticker}")
        
        # ====================================================================
        # ÉTAPE 2/8: Analyse de Sentiment Approfondie
        # ====================================================================
        logger.info(f"💬 ÉTAPE 2/8: Analyse sentiment approfondie...")
        
        sentiment_trend = sentiment_aggregator.analyze_sentiment_trend(
            ticker,
            lookback_days=7
        )
        
        logger.info(f"✅ Sentiment analysé: {sentiment_trend.overall_sentiment if sentiment_trend else 'N/A'}")
        
        # ====================================================================
        # ÉTAPE 3/8: Analyse Complète de la Position
        # ====================================================================
        logger.info(f"🔬 ÉTAPE 3/8: Analyse complète position...")
        
        # Créer un mini-portfolio avec juste cette action pour utiliser PortfolioAnalyzer
        mini_portfolio = {
            'positions': {ticker: 1},  # 1 action pour l'analyse
            'cash': 0,
            'total_value': stock_data.price.current_price if stock_data.price else 100
        }
        
        analysis = await portfolio_analyzer.analyze_portfolio(
            mini_portfolio,
            user_id=str(current_user.id),
            deep_analysis=True
        )
        
        position_analysis = analysis.positions.get(ticker) if analysis.positions else None
        
        logger.info(f"✅ Analyse complète terminée")
        
        # ====================================================================
        # ÉTAPE 4/8: Prédictions ML (XGBoost + LSTM)
        # ====================================================================
        logger.info(f"🤖 ÉTAPE 4/8: Prédictions ML...")
        
        ml_prediction = await ml_signal_service.get_prediction(
            ticker,
            use_cache=False  # Force fresh prediction
        )
        
        if ml_prediction:
            logger.info(
                f"✅ ML Prédiction: {ml_prediction.signal} "
                f"(confiance {ml_prediction.signal_strength:.0f}%)"
            )
        else:
            logger.warning(f"⚠️ Pas de prédiction ML disponible pour {ticker}")
        
        # ====================================================================
        # ÉTAPE 5/8: Génération de Recommandations Détaillées
        # ====================================================================
        logger.info(f"🎯 ÉTAPE 5/8: Génération recommandations...")
        
        # Créer fake portfolio signals pour recommendation engine
        from app.services.portfolio.ml_signal_service import MLPortfolioSignals
        
        ml_signals = MLPortfolioSignals(
            predictions={ticker: ml_prediction} if ml_prediction else {},
            bullish_count=1 if ml_prediction and ml_prediction.signal in ['BUY', 'STRONG_BUY'] else 0,
            bearish_count=1 if ml_prediction and ml_prediction.signal in ['SELL', 'STRONG_SELL'] else 0,
            neutral_count=1 if ml_prediction and ml_prediction.signal == 'HOLD' else 0,
            avg_confidence=ml_prediction.signal_strength if ml_prediction else 50.0,
            top_buys=[ticker] if ml_prediction and ml_prediction.signal in ['BUY', 'STRONG_BUY'] else [],
            top_sells=[ticker] if ml_prediction and ml_prediction.signal in ['SELL', 'STRONG_SELL'] else [],
            generated_at=datetime.now()
        )
        
        recommendations = recommendation_engine.generate_recommendations(
            mini_portfolio,
            analysis,
            ml_signals
        )
        
        recommendation = recommendations.position_recommendations.get(ticker)
        
        logger.info(f"✅ Recommandation générée: {recommendation.action if recommendation else 'N/A'}")
        
        # ====================================================================
        # ÉTAPE 6/8: Création d'Alertes Intelligentes
        # ====================================================================
        logger.info(f"🔔 ÉTAPE 6/8: Création alertes...")
        
        alert_batch = alert_system.generate_alerts(
            analysis,
            ml_signals,
            recommendations,
            "on_demand"  # Type d'analyse
        )
        
        logger.info(f"✅ {alert_batch.total_alerts} alertes créées")
        
        # ====================================================================
        # ÉTAPE 7/8: Événements Économiques à Venir
        # ====================================================================
        logger.info(f"📅 ÉTAPE 7/8: Événements économiques...")
        
        upcoming_events = []
        try:
            events_7d = calendar_service.get_upcoming_events(days=7)
            
            # Filtrer événements pertinents pour ce ticker/secteur
            sector = stock_data.fundamentals.sector if stock_data.fundamentals else None
            for event in events_7d:
                # Ajouter tous les événements high/critical
                if event.impact in ['high', 'critical']:
                    upcoming_events.append({
                        'date': event.date.isoformat(),
                        'time': event.time,
                        'event': event.event,
                        'impact': event.impact,
                        'description': event.description
                    })
        except Exception as e:
            logger.error(f"Erreur récupération événements: {e}")
        
        logger.info(f"✅ {len(upcoming_events)} événements identifiés")
        
        # ====================================================================
        # ÉTAPE 8/8: Construction de la Réponse Finale
        # ====================================================================
        logger.info(f"📦 ÉTAPE 8/8: Construction réponse...")
        
        execution_time = int((time.time() - start_time) * 1000)
        
        # Construire la réponse ultra-complète
        response = {
            # Métadonnées
            "ticker": ticker,
            "analysis_type": "deep_analysis",
            "analyzed_at": datetime.now().isoformat(),
            "execution_time_ms": execution_time,
            
            # ÉTAPE 1: Données collectées (35+ sources)
            "data_collection": {
                "price_data": {
                    "current_price": stock_data.price.current_price if stock_data.price else None,
                    "change_pct": stock_data.price.change_pct if stock_data.price else None,
                    "volume": stock_data.price.volume if stock_data.price else None,
                    "day_high": stock_data.price.day_high if stock_data.price else None,
                    "day_low": stock_data.price.day_low if stock_data.price else None,
                    "year_high": stock_data.price.year_high if stock_data.price else None,
                    "year_low": stock_data.price.year_low if stock_data.price else None,
                    "market_cap": stock_data.price.market_cap if stock_data.price else None,
                    "beta": stock_data.price.beta if stock_data.price else None,
                } if stock_data.price else {},
                
                "fundamentals": {
                    "sector": stock_data.fundamentals.sector if stock_data.fundamentals else None,
                    "industry": stock_data.fundamentals.industry if stock_data.fundamentals else None,
                    "pe_ratio": stock_data.fundamentals.pe_ratio if stock_data.fundamentals else None,
                    "forward_pe": stock_data.fundamentals.forward_pe if stock_data.fundamentals else None,
                    "peg_ratio": stock_data.fundamentals.peg_ratio if stock_data.fundamentals else None,
                    "price_to_book": stock_data.fundamentals.price_to_book if stock_data.fundamentals else None,
                    "dividend_yield": stock_data.fundamentals.dividend_yield if stock_data.fundamentals else None,
                    "roe": stock_data.fundamentals.roe if stock_data.fundamentals else None,
                    "debt_to_equity": stock_data.fundamentals.debt_to_equity if stock_data.fundamentals else None,
                    "profit_margin": stock_data.fundamentals.profit_margin if stock_data.fundamentals else None,
                    "revenue_growth": stock_data.fundamentals.revenue_growth if stock_data.fundamentals else None,
                    "earnings_growth": stock_data.fundamentals.earnings_growth if stock_data.fundamentals else None,
                } if stock_data.fundamentals else {},
                
                "sentiment": {
                    "reddit_sentiment": stock_data.sentiment.reddit_sentiment if stock_data.sentiment else None,
                    "reddit_mentions": stock_data.sentiment.reddit_mentions if stock_data.sentiment else 0,
                    "reddit_bullish_pct": stock_data.sentiment.reddit_bullish_pct if stock_data.sentiment else 0,
                    "stocktwits_sentiment": stock_data.sentiment.stocktwits_sentiment if stock_data.sentiment else None,
                    "stocktwits_messages": stock_data.sentiment.stocktwits_messages if stock_data.sentiment else 0,
                    "news_sentiment": stock_data.sentiment.news_sentiment if stock_data.sentiment else None,
                    "news_count": stock_data.sentiment.news_count if stock_data.sentiment else 0,
                    "overall_sentiment": stock_data.sentiment.overall_sentiment if stock_data.sentiment else None,
                    "sentiment_confidence": stock_data.sentiment.sentiment_confidence if stock_data.sentiment else 0,
                } if stock_data.sentiment else {},
                
                "news": {
                    "articles": stock_data.news.articles[:5] if stock_data.news else [],  # Top 5
                    "total_count": stock_data.news.total_count if stock_data.news else 0,
                    "sentiment_score": stock_data.news.sentiment_score if stock_data.news else 0,
                } if stock_data.news else {},
                
                "macro_context": stock_data.macro.__dict__ if stock_data.macro else {}
            },
            
            # ÉTAPE 2: Analyse sentiment approfondie
            "sentiment_analysis": {
                "current_sentiment": sentiment_trend.current_sentiment if sentiment_trend else "neutral",
                "sentiment_trend": sentiment_trend.trend if sentiment_trend else "stable",
                "momentum": sentiment_trend.momentum if sentiment_trend else 0,
                "velocity": sentiment_trend.velocity if sentiment_trend else 0,
                "pattern_detected": sentiment_trend.pattern_detected if sentiment_trend else None,
                "confidence": sentiment_trend.confidence if sentiment_trend else 50,
                "explanation": sentiment_trend.explanation if sentiment_trend else "Sentiment analysis not available"
            } if sentiment_trend else {},
            
            # ÉTAPE 3: Analyse position complète
            "position_analysis": {
                "health_score": position_analysis.health_score if position_analysis else 50,
                "risks": position_analysis.risks if position_analysis else [],
                "sector": position_analysis.sector if position_analysis else None,
                "beta": position_analysis.beta if position_analysis else None,
                "pe_ratio": position_analysis.pe_ratio if position_analysis else None,
            } if position_analysis else {},
            
            # Portfolio context (même si 1 action)
            "portfolio_context": {
                "portfolio_health_score": analysis.portfolio_health_score if analysis else 50,
                "portfolio_sentiment": analysis.portfolio_sentiment if analysis else "neutral",
                "critical_alerts": analysis.critical_alerts if analysis else []
            } if analysis else {},
            
            # ÉTAPE 4: Prédictions ML
            "ml_predictions": {
                "signal": ml_prediction.signal if ml_prediction else "HOLD",
                "signal_strength": ml_prediction.signal_strength if ml_prediction else 50,
                "prediction_1d": ml_prediction.prediction_1d if ml_prediction else None,
                "confidence_1d": ml_prediction.confidence_1d if ml_prediction else 0,
                "predicted_price_1d": ml_prediction.predicted_price_1d if ml_prediction else None,
                "predicted_change_1d": ml_prediction.predicted_change_1d if ml_prediction else None,
                "prediction_3d": ml_prediction.prediction_3d if ml_prediction else None,
                "confidence_3d": ml_prediction.confidence_3d if ml_prediction else 0,
                "predicted_price_3d": ml_prediction.predicted_price_3d if ml_prediction else None,
                "predicted_change_3d": ml_prediction.predicted_change_3d if ml_prediction else None,
                "prediction_7d": ml_prediction.prediction_7d if ml_prediction else None,
                "confidence_7d": ml_prediction.confidence_7d if ml_prediction else 0,
                "predicted_price_7d": ml_prediction.predicted_price_7d if ml_prediction else None,
                "predicted_change_7d": ml_prediction.predicted_change_7d if ml_prediction else None,
                "model_version": ml_prediction.model_version if ml_prediction else "unknown"
            } if ml_prediction else {},
            
            # ÉTAPE 5: Recommandation détaillée
            "recommendation": {
                "action": recommendation.action.value if recommendation else "HOLD",
                "confidence": recommendation.confidence if recommendation else 50,
                "primary_reason": recommendation.primary_reason if recommendation else "No analysis available",
                "detailed_reasons": recommendation.detailed_reasons if recommendation else [],
                "risk_factors": recommendation.risk_factors if recommendation else [],
                "suggested_action": recommendation.suggested_action if recommendation else "No action required",
                "quantity_suggestion": recommendation.quantity_suggestion if recommendation else None,
                "target_price": recommendation.target_price if recommendation else None,
                "stop_loss": recommendation.stop_loss if recommendation else None,
                "entry_price": recommendation.entry_price if recommendation else None,
                "priority": recommendation.priority.value if recommendation else "low",
                "horizon": recommendation.horizon if recommendation else "medium_term",
                "risk_level": recommendation.risk_level if recommendation else "medium"
            } if recommendation else {},
            
            # ÉTAPE 6: Alertes
            "alerts": {
                "total": alert_batch.total_alerts if alert_batch else 0,
                "critical": [
                    {
                        "title": alert.title,
                        "message": alert.message,
                        "severity": alert.severity
                    }
                    for alert in (alert_batch.critical_alerts if alert_batch else [])
                ],
                "important": [
                    {
                        "title": alert.title,
                        "message": alert.message,
                        "severity": alert.severity
                    }
                    for alert in (alert_batch.important_alerts if alert_batch else [])
                ],
                "info": [
                    {
                        "title": alert.title,
                        "message": alert.message,
                        "severity": alert.severity
                    }
                    for alert in (alert_batch.info_alerts if alert_batch else [])
                ]
            } if alert_batch else {},
            
            # ÉTAPE 7: Événements économiques à venir
            "upcoming_events": upcoming_events,
            
            # Résumé exécutif
            "executive_summary": recommendations.executive_summary if recommendations else "Analysis completed",
            
            # Score FXI legacy (pour compatibilité)
            "final_score": analysis.portfolio_health_score if analysis else 50,
            "recommendation_final": recommendation.action.value if recommendation else "HOLD",
            "confidence_final": recommendation.confidence if recommendation else 50
        }
        
        logger.info(f"✅ Analyse ultra-complète terminée pour {ticker} en {execution_time}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur analyse ultra-complète {ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'analyse complète: {str(e)}"
        )
