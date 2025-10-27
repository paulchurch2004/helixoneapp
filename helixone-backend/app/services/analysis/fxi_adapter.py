"""
Adaptateur async pour le moteur FXI existant
Fait le pont entre le backend FastAPI et le moteur FXI v2.0
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Ajouter le chemin vers le moteur FXI existant
FXI_ENGINE_PATH = Path(__file__).parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(FXI_ENGINE_PATH))

from fxi_engine.core.engine import FXIEngine, AnalysisResult as FXIAnalysisResult
from fxi_engine.core.config import EngineConfig

from app.services.data_sources.aggregator import get_default_aggregator
from app.schemas.analysis import AnalysisResult, AnalysisRequest

logger = logging.getLogger(__name__)

# Thread pool pour exécuter le moteur FXI synchrone
_executor = ThreadPoolExecutor(max_workers=4)


class FXIAdapter:
    """
    Adaptateur qui fait le pont entre :
    - Le backend FastAPI (async)
    - Le moteur FXI existant (sync)
    - Le nouveau DataAggregator (async)
    """

    def __init__(self):
        self.data_aggregator = get_default_aggregator()
        logger.info("FXIAdapter initialisé")

    async def analyze_ticker(
        self,
        ticker: str,
        mode: str = "Standard"
    ) -> AnalysisResult:
        """
        Analyse complète d'un ticker avec le moteur FXI

        Args:
            ticker: Symbole de l'action
            mode: Standard, Conservative, Aggressive

        Returns:
            AnalysisResult adapté pour l'API
        """
        logger.info(f"Démarrage analyse FXI pour {ticker} en mode {mode}")

        try:
            # 1. Collecter les données avec notre DataAggregator
            raw_data = await self._collect_data_async(ticker)

            # 2. Exécuter l'analyse FXI dans un thread séparé (car sync)
            fxi_result = await self._run_fxi_analysis_async(ticker, mode, raw_data)

            # 3. Convertir le résultat FXI en format API
            api_result = self._convert_fxi_result_to_api(fxi_result)

            logger.info(
                f"Analyse terminée pour {ticker}: Score {api_result.final_score}, "
                f"{api_result.recommendation}"
            )
            return api_result

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de {ticker}: {e}")
            raise

    async def _collect_data_async(self, ticker: str) -> Dict:
        """
        Collecte les données avec notre DataAggregator

        Retourne un format compatible avec le moteur FXI existant
        """
        try:
            # Récupérer les données de notre aggregator
            quote = await self.data_aggregator.get_quote(ticker)
            historical = await self.data_aggregator.get_historical(
                ticker,
                start_date=quote.timestamp.date() - timedelta(days=365),
                end_date=quote.timestamp.date()
            )
            fundamentals = await self.data_aggregator.get_fundamentals(ticker)
            news = await self.data_aggregator.get_news(ticker, limit=50)

            # Convertir en format DataFrame pour compatibilité FXI
            import pandas as pd
            import yfinance as yf

            # Créer un DataFrame d'historique
            history_data = []
            for price in historical.prices:
                history_data.append({
                    'Date': price.date,
                    'Open': float(price.open),
                    'High': float(price.high),
                    'Low': float(price.low),
                    'Close': float(price.close),
                    'Volume': price.volume
                })

            history_df = pd.DataFrame(history_data)
            history_df.set_index('Date', inplace=True)

            # Construire l'objet info compatible Yahoo Finance
            info = {
                'currentPrice': float(quote.price),
                'regularMarketPrice': float(quote.price),
                'previousClose': float(quote.previous_close) if quote.previous_close else float(quote.price),
                'open': float(quote.open) if quote.open else float(quote.price),
                'dayHigh': float(quote.high) if quote.high else float(quote.price),
                'dayLow': float(quote.low) if quote.low else float(quote.price),
                'volume': quote.volume,
                'marketCap': int(fundamentals.market_cap) if fundamentals.market_cap else None,
                'longName': quote.name,
                'shortName': quote.name,

                # Fondamentaux
                'forwardPE': float(fundamentals.pe_ratio) if fundamentals.pe_ratio else None,
                'trailingPE': float(fundamentals.pe_ratio) if fundamentals.pe_ratio else None,
                'pegRatio': float(fundamentals.peg_ratio) if fundamentals.peg_ratio else None,
                'priceToBook': float(fundamentals.pb_ratio) if fundamentals.pb_ratio else None,
                'priceToSalesTrailing12Months': float(fundamentals.ps_ratio) if fundamentals.ps_ratio else None,
                'enterpriseValue': int(fundamentals.enterprise_value) if fundamentals.enterprise_value else None,
                'enterpriseToEbitda': float(fundamentals.ev_ebitda) if fundamentals.ev_ebitda else None,

                # Rentabilité
                'profitMargins': float(fundamentals.profit_margin) / 100 if fundamentals.profit_margin else None,
                'operatingMargins': float(fundamentals.operating_margin) / 100 if fundamentals.operating_margin else None,
                'returnOnEquity': float(fundamentals.roe) / 100 if fundamentals.roe else None,
                'returnOnAssets': float(fundamentals.roa) / 100 if fundamentals.roa else None,

                # Croissance
                'revenueGrowth': float(fundamentals.revenue_growth) / 100 if fundamentals.revenue_growth else None,
                'earningsGrowth': float(fundamentals.earnings_growth) / 100 if fundamentals.earnings_growth else None,
                'revenuePerShare': float(fundamentals.revenue_per_share) if fundamentals.revenue_per_share else None,
                'trailingEps': float(fundamentals.eps) if fundamentals.eps else None,

                # Santé financière
                'debtToEquity': float(fundamentals.debt_to_equity) if fundamentals.debt_to_equity else None,
                'currentRatio': float(fundamentals.current_ratio) if fundamentals.current_ratio else None,
                'quickRatio': float(fundamentals.quick_ratio) if fundamentals.quick_ratio else None,

                # Dividendes
                'dividendYield': float(fundamentals.dividend_yield) / 100 if fundamentals.dividend_yield else None,
                'payoutRatio': float(fundamentals.dividend_payout_ratio) / 100 if fundamentals.dividend_payout_ratio else None,

                # Secteur
                'sector': fundamentals.sector,
                'industry': fundamentals.industry,
            }

            # Format compatible avec FXI
            data = {
                'yahoo': {
                    'info': info,
                    'history': history_df,
                    'success': True
                },
                'scraped': {
                    'news_count': len(news),
                    'success': True
                }
            }

            logger.debug(f"Données collectées pour {ticker}: {len(history_df)} jours d'historique")
            return data

        except Exception as e:
            logger.error(f"Erreur lors de la collecte de données pour {ticker}: {e}")
            # Retourner un format minimal en cas d'erreur
            return {
                'yahoo': {'error': str(e), 'success': False},
                'scraped': {'error': str(e), 'success': False}
            }

    async def _run_fxi_analysis_async(
        self,
        ticker: str,
        mode: str,
        raw_data: Dict
    ) -> FXIAnalysisResult:
        """
        Exécute l'analyse FXI dans un thread séparé pour ne pas bloquer

        Le moteur FXI est synchrone, on l'exécute dans un ThreadPoolExecutor
        """
        loop = asyncio.get_event_loop()

        def run_sync_analysis():
            """Fonction synchrone à exécuter dans le thread"""
            # Créer une configuration selon le mode
            if mode.lower() == "conservative":
                config = EngineConfig.create_conservative()
            elif mode.lower() == "aggressive":
                config = EngineConfig.create_aggressive()
            else:
                config = EngineConfig()

            # Créer le moteur FXI
            engine = FXIEngine(config)

            # Remplacer le data collector par nos données
            # Créer un mock data collector qui retourne nos données
            class MockDataCollector:
                def collect_all_data(self, t):
                    return raw_data

            engine.data_collector = MockDataCollector()

            # Analyser avec nos données pré-collectées
            result = engine.analyze(ticker, mode)

            return result

        # Exécuter dans un thread séparé
        result = await loop.run_in_executor(_executor, run_sync_analysis)
        return result

    def _convert_fxi_result_to_api(self, fxi_result: FXIAnalysisResult) -> AnalysisResult:
        """
        Convertit le résultat FXI en format API

        Args:
            fxi_result: Résultat du moteur FXI

        Returns:
            AnalysisResult pour l'API
        """
        return AnalysisResult(
            ticker=fxi_result.ticker,
            timestamp=fxi_result.timestamp,
            final_score=fxi_result.final_score,
            recommendation=fxi_result.recommendation,
            confidence=fxi_result.confidence,

            # Scores détaillés
            technical_score=fxi_result.technical_score,
            fundamental_score=fxi_result.fundamental_score,
            sentiment_score=fxi_result.sentiment_score,
            risk_score=fxi_result.risk_score,
            macro_score=fxi_result.macro_score,

            # Métadonnées
            execution_time=fxi_result.execution_time,
            data_quality=fxi_result.data_quality,

            # Détails
            details=fxi_result.details
        )


# Instance globale
_adapter: Optional[FXIAdapter] = None


def get_fxi_adapter() -> FXIAdapter:
    """
    Retourne l'adaptateur FXI (singleton)

    Returns:
        FXIAdapter
    """
    global _adapter
    if _adapter is None:
        _adapter = FXIAdapter()
    return _adapter
