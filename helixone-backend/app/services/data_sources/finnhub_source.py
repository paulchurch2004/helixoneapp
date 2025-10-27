"""
Source de données Finnhub
API gratuite: 60 requêtes/minute
Documentation: https://finnhub.io/docs/api
"""

import logging
from typing import Optional
from datetime import date, datetime, timedelta
from decimal import Decimal
import finnhub
import asyncio
from functools import partial

from app.services.data_sources.base import BaseDataSource, DataUnavailableError
from app.schemas.market import Quote, HistoricalData, HistoricalPrice, Fundamentals, NewsArticle, ESGScore
from app.core.config import settings

logger = logging.getLogger(__name__)


class FinnhubSource(BaseDataSource):
    """
    Source de données Finnhub

    Capacités:
    - Prix en temps réel
    - Données historiques
    - Fondamentaux complets
    - Actualités avec sentiment
    - Insider transactions

    Limites gratuites:
    - 60 requêtes/minute
    - Données en temps réel avec léger délai
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client Finnhub

        Args:
            api_key: Clé API Finnhub (ou depuis settings)
        """
        self.api_key = api_key or settings.FINNHUB_API_KEY

        if not self.api_key:
            logger.warning("Finnhub API key not configured")
            self.client = None
        else:
            try:
                self.client = finnhub.Client(api_key=self.api_key)
                logger.info("Finnhub client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Finnhub client: {e}")
                self.client = None

    @property
    def name(self) -> str:
        return "Finnhub"

    def is_available(self) -> bool:
        """Vérifie si la source est disponible"""
        return self.client is not None

    async def get_quote(self, ticker: str) -> Optional[Quote]:
        """
        Récupère le prix en temps réel

        Args:
            ticker: Symbole de l'action

        Returns:
            Quote ou None si erreur
        """
        if not self.is_available():
            return None

        try:
            # Finnhub est synchrone, on l'exécute dans un executor
            loop = asyncio.get_event_loop()
            quote_data = await loop.run_in_executor(
                None,
                partial(self.client.quote, ticker)
            )

            if not quote_data or 'c' not in quote_data:
                return None

            # Récupérer aussi le profil pour avoir le nom
            profile = await loop.run_in_executor(
                None,
                partial(self.client.company_profile2, symbol=ticker)
            )

            name = profile.get('name', ticker) if profile else ticker

            current_price = quote_data['c']  # Current price
            previous_close = quote_data['pc']  # Previous close
            change = quote_data['d']  # Change
            change_percent = quote_data['dp']  # Change percent

            return Quote(
                ticker=ticker.upper(),
                name=name,
                price=Decimal(str(current_price)),
                previous_close=Decimal(str(previous_close)) if previous_close else None,
                open=None,  # Finnhub quote ne fournit pas l'open
                high=quote_data.get('h'),  # High
                low=quote_data.get('l'),  # Low
                volume=None,  # Pas dans quote simple
                change=Decimal(str(change)) if change else None,
                change_percent=Decimal(str(change_percent)) if change_percent else None,
                market_cap=None,  # Nécessite un appel séparé
                timestamp=datetime.now(),
                source="finnhub"
            )

        except Exception as e:
            logger.error(f"Finnhub.get_quote for {ticker} failed: {e}")
            return None

    async def get_historical(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> Optional[HistoricalData]:
        """
        Récupère les données historiques

        Args:
            ticker: Symbole de l'action
            start_date: Date de début
            end_date: Date de fin
            interval: Intervalle (1d uniquement supporté par Finnhub gratuit)

        Returns:
            HistoricalData ou None si erreur
        """
        if not self.is_available():
            return None

        try:
            # Convertir les dates en timestamps Unix
            start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

            loop = asyncio.get_event_loop()
            candles = await loop.run_in_executor(
                None,
                partial(
                    self.client.stock_candles,
                    ticker,
                    'D',  # Daily
                    start_timestamp,
                    end_timestamp
                )
            )

            if not candles or candles.get('s') != 'ok':
                return None

            # Construire la liste des prix
            prices = []
            timestamps = candles.get('t', [])
            opens = candles.get('o', [])
            highs = candles.get('h', [])
            lows = candles.get('l', [])
            closes = candles.get('c', [])
            volumes = candles.get('v', [])

            for i in range(len(timestamps)):
                price_date = datetime.fromtimestamp(timestamps[i]).date()
                prices.append(HistoricalPrice(
                    date=price_date,
                    open=Decimal(str(opens[i])),
                    high=Decimal(str(highs[i])),
                    low=Decimal(str(lows[i])),
                    close=Decimal(str(closes[i])),
                    volume=int(volumes[i]) if i < len(volumes) else 0,
                    adjusted_close=Decimal(str(closes[i]))  # Finnhub ne fournit pas adjusted
                ))

            return HistoricalData(
                ticker=ticker.upper(),
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                prices=prices,
                source="finnhub"
            )

        except Exception as e:
            logger.error(f"Finnhub.get_historical for {ticker} failed: {e}")
            return None

    async def get_fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        """
        Récupère les données fondamentales

        Args:
            ticker: Symbole de l'action

        Returns:
            Fundamentals ou None si erreur
        """
        if not self.is_available():
            return None

        try:
            loop = asyncio.get_event_loop()

            # Récupérer les métriques basiques
            metrics = await loop.run_in_executor(
                None,
                partial(self.client.company_basic_financials, ticker, 'all')
            )

            if not metrics or 'metric' not in metrics:
                return None

            metric = metrics['metric']

            # Récupérer aussi le profil de l'entreprise
            profile = await loop.run_in_executor(
                None,
                partial(self.client.company_profile2, symbol=ticker)
            )

            return Fundamentals(
                ticker=ticker.upper(),
                market_cap=Decimal(str(profile.get('marketCapitalization', 0) * 1_000_000)) if profile else None,
                pe_ratio=Decimal(str(metric.get('peBasicExclExtraTTM'))) if metric.get('peBasicExclExtraTTM') else None,
                forward_pe=Decimal(str(metric.get('peNormalizedAnnual'))) if metric.get('peNormalizedAnnual') else None,
                peg_ratio=None,  # Pas fourni par Finnhub
                price_to_book=Decimal(str(metric.get('pbAnnual'))) if metric.get('pbAnnual') else None,
                price_to_sales=Decimal(str(metric.get('psAnnual'))) if metric.get('psAnnual') else None,
                enterprise_value=None,
                ev_to_ebitda=None,
                profit_margin=Decimal(str(metric.get('netProfitMarginAnnual'))) if metric.get('netProfitMarginAnnual') else None,
                operating_margin=Decimal(str(metric.get('operatingMarginAnnual'))) if metric.get('operatingMarginAnnual') else None,
                return_on_assets=Decimal(str(metric.get('roaRfy'))) if metric.get('roaRfy') else None,
                return_on_equity=Decimal(str(metric.get('roeRfy'))) if metric.get('roeRfy') else None,
                revenue=None,
                revenue_per_share=Decimal(str(metric.get('revenuePerShareAnnual'))) if metric.get('revenuePerShareAnnual') else None,
                revenue_growth=Decimal(str(metric.get('revenueGrowthAnnual'))) if metric.get('revenueGrowthAnnual') else None,
                earnings_growth=None,
                eps=Decimal(str(metric.get('epsBasicExclExtraItemsAnnual'))) if metric.get('epsBasicExclExtraItemsAnnual') else None,
                debt_to_equity=Decimal(str(metric.get('totalDebtToTotalEquityAnnual'))) if metric.get('totalDebtToTotalEquityAnnual') else None,
                current_ratio=Decimal(str(metric.get('currentRatioAnnual'))) if metric.get('currentRatioAnnual') else None,
                quick_ratio=Decimal(str(metric.get('quickRatioAnnual'))) if metric.get('quickRatioAnnual') else None,
                dividend_yield=Decimal(str(metric.get('dividendYieldIndicatedAnnual'))) if metric.get('dividendYieldIndicatedAnnual') else None,
                dividend_payout_ratio=Decimal(str(metric.get('payoutRatioAnnual'))) if metric.get('payoutRatioAnnual') else None,
                beta=Decimal(str(metric.get('beta'))) if metric.get('beta') else None,
                week_52_high=Decimal(str(metric.get('52WeekHigh'))) if metric.get('52WeekHigh') else None,
                week_52_low=Decimal(str(metric.get('52WeekLow'))) if metric.get('52WeekLow') else None,
                shares_outstanding=int(profile.get('shareOutstanding', 0) * 1_000_000) if profile and profile.get('shareOutstanding') else None,
                sector=profile.get('finnhubIndustry') if profile else None,
                industry=profile.get('finnhubIndustry') if profile else None,
                employees=int(profile.get('employees', 0)) if profile else None,
                timestamp=datetime.now(),
                source="finnhub"
            )

        except Exception as e:
            logger.error(f"Finnhub.get_fundamentals for {ticker} failed: {e}")
            return None

    async def get_news(self, ticker: str, limit: int = 10) -> list[NewsArticle]:
        """
        Récupère les actualités avec sentiment

        Args:
            ticker: Symbole de l'action
            limit: Nombre max d'articles

        Returns:
            Liste d'articles
        """
        if not self.is_available():
            return []

        try:
            # Récupérer les news des 7 derniers jours
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            loop = asyncio.get_event_loop()
            news_data = await loop.run_in_executor(
                None,
                partial(
                    self.client.company_news,
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            )

            if not news_data:
                return []

            articles = []
            for item in news_data[:limit]:
                articles.append(NewsArticle(
                    title=item.get('headline', ''),
                    summary=item.get('summary', ''),
                    url=item.get('url', ''),
                    published_at=datetime.fromtimestamp(item.get('datetime', 0)),
                    source=item.get('source', 'finnhub'),
                    sentiment=item.get('sentiment')  # Finnhub fournit le sentiment!
                ))

            return articles

        except Exception as e:
            logger.error(f"Finnhub.get_news for {ticker} failed: {e}")
            return []

    async def get_esg_scores(self, ticker: str) -> Optional[ESGScore]:
        """
        Récupère les scores ESG (Environmental, Social, Governance)

        Args:
            ticker: Symbole de l'action

        Returns:
            ESGScore ou None si erreur
        """
        if not self.is_available():
            return None

        try:
            # Finnhub fournit des scores ESG via l'endpoint company/esg
            loop = asyncio.get_event_loop()

            # Utiliser l'API REST directement car le SDK Python n'a pas cette méthode
            import aiohttp
            url = f"https://finnhub.io/api/v1/stock/esg?symbol={ticker}&token={self.api_key}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Finnhub ESG returned status {response.status} for {ticker}")
                        return None

                    data = await response.json()

                    if not data or 'data' not in data:
                        return None

                    # Les données ESG de Finnhub sont dans 'data' array
                    esg_data = data['data']
                    if not esg_data or len(esg_data) == 0:
                        return None

                    # Prendre le plus récent
                    latest = esg_data[0]

                    return ESGScore(
                        ticker=ticker.upper(),
                        total_score=Decimal(str(latest.get('totalESG', 0))) if latest.get('totalESG') else None,
                        environment_score=Decimal(str(latest.get('environmentScore', 0))) if latest.get('environmentScore') else None,
                        social_score=Decimal(str(latest.get('socialScore', 0))) if latest.get('socialScore') else None,
                        governance_score=Decimal(str(latest.get('governanceScore', 0))) if latest.get('governanceScore') else None,
                        controversy_level=latest.get('controversyLevel'),
                        grade=self._calculate_esg_grade(latest.get('totalESG')),
                        source="finnhub",
                        timestamp=datetime.now()
                    )

        except Exception as e:
            logger.error(f"Finnhub.get_esg_scores for {ticker} failed: {e}")
            return None

    def _calculate_esg_grade(self, total_score: Optional[float]) -> Optional[str]:
        """
        Calcule une note ESG basée sur le score total

        Args:
            total_score: Score total ESG (0-100)

        Returns:
            Grade (A+, A, B+, B, C+, C, D, F)
        """
        if total_score is None:
            return None

        if total_score >= 90:
            return "A+"
        elif total_score >= 80:
            return "A"
        elif total_score >= 75:
            return "B+"
        elif total_score >= 70:
            return "B"
        elif total_score >= 65:
            return "C+"
        elif total_score >= 60:
            return "C"
        elif total_score >= 50:
            return "D"
        else:
            return "F"
