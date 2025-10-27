"""
Source de données Financial Modeling Prep (FMP)
API gratuite: 250 requêtes/jour
Documentation: https://site.financialmodelingprep.com/developer/docs
"""

import logging
from typing import Optional
from datetime import date, datetime
from decimal import Decimal
import aiohttp
import asyncio

from app.services.data_sources.base import BaseDataSource, DataUnavailableError
from app.schemas.market import Quote, HistoricalData, HistoricalPrice, Fundamentals, NewsArticle
from app.core.config import settings

logger = logging.getLogger(__name__)


class FMPSource(BaseDataSource):
    """
    Source de données Financial Modeling Prep

    Capacités:
    - Prix en temps réel
    - Données historiques très complètes
    - Fondamentaux détaillés (meilleurs ratios financiers)
    - Actualités
    - Données de bilans et revenus

    Limites gratuites:
    - 250 requêtes/jour
    - Données avec léger délai
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client FMP

        Args:
            api_key: Clé API FMP (ou depuis settings)
        """
        self.api_key = api_key or settings.FMP_API_KEY

        if not self.api_key or self.api_key == "your_key_here":
            logger.warning("FMP API key not configured")

    @property
    def name(self) -> str:
        return "FMP"

    def is_available(self) -> bool:
        """Vérifie si la source est disponible"""
        return bool(self.api_key and self.api_key != "your_key_here")

    async def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Effectue une requête HTTP vers l'API FMP

        Args:
            endpoint: Endpoint de l'API (ex: "quote/AAPL")
            params: Paramètres supplémentaires

        Returns:
            Réponse JSON
        """
        if params is None:
            params = {}

        params['apikey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise DataUnavailableError(
                        self.name,
                        f"HTTP {response.status}: {await response.text()}"
                    )

                return await response.json()

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
            # FMP retourne un array avec un seul élément
            data = await self._make_request(f"quote/{ticker}")

            if not data or len(data) == 0:
                return None

            quote_data = data[0]

            return Quote(
                ticker=ticker.upper(),
                name=quote_data.get('name', ticker),
                price=Decimal(str(quote_data.get('price', 0))),
                previous_close=Decimal(str(quote_data.get('previousClose', 0))) if quote_data.get('previousClose') else None,
                open=Decimal(str(quote_data.get('open', 0))) if quote_data.get('open') else None,
                high=Decimal(str(quote_data.get('dayHigh', 0))) if quote_data.get('dayHigh') else None,
                low=Decimal(str(quote_data.get('dayLow', 0))) if quote_data.get('dayLow') else None,
                volume=int(quote_data.get('volume', 0)) if quote_data.get('volume') else None,
                change=Decimal(str(quote_data.get('change', 0))) if quote_data.get('change') else None,
                change_percent=Decimal(str(quote_data.get('changesPercentage', 0))) if quote_data.get('changesPercentage') else None,
                market_cap=Decimal(str(quote_data.get('marketCap', 0))) if quote_data.get('marketCap') else None,
                timestamp=datetime.now(),
                source="fmp"
            )

        except Exception as e:
            logger.error(f"FMP.get_quote for {ticker} failed: {e}")
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
            interval: Intervalle (1d uniquement supporté)

        Returns:
            HistoricalData ou None si erreur
        """
        if not self.is_available():
            return None

        try:
            # FMP utilise le format YYYY-MM-DD
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }

            data = await self._make_request(f"historical-price-full/{ticker}", params)

            if not data or 'historical' not in data:
                return None

            prices = []
            for item in data['historical']:
                prices.append(HistoricalPrice(
                    date=datetime.strptime(item['date'], '%Y-%m-%d').date(),
                    open=Decimal(str(item['open'])),
                    high=Decimal(str(item['high'])),
                    low=Decimal(str(item['low'])),
                    close=Decimal(str(item['close'])),
                    volume=int(item.get('volume', 0)),
                    adjusted_close=Decimal(str(item.get('adjClose', item['close'])))
                ))

            # FMP retourne les données du plus récent au plus ancien, on inverse
            prices.reverse()

            return HistoricalData(
                ticker=ticker.upper(),
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                prices=prices,
                source="fmp"
            )

        except Exception as e:
            logger.error(f"FMP.get_historical for {ticker} failed: {e}")
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
            # Récupérer plusieurs endpoints en parallèle
            tasks = [
                self._make_request(f"profile/{ticker}"),
                self._make_request(f"key-metrics/{ticker}", {'limit': 1}),
                self._make_request(f"ratios/{ticker}", {'limit': 1})
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            profile_data = results[0] if not isinstance(results[0], Exception) and results[0] else None
            metrics_data = results[1] if not isinstance(results[1], Exception) and results[1] else None
            ratios_data = results[2] if not isinstance(results[2], Exception) and results[2] else None

            if not profile_data or len(profile_data) == 0:
                return None

            profile = profile_data[0]
            metrics = metrics_data[0] if metrics_data and len(metrics_data) > 0 else {}
            ratios = ratios_data[0] if ratios_data and len(ratios_data) > 0 else {}

            return Fundamentals(
                ticker=ticker.upper(),
                market_cap=Decimal(str(profile.get('mktCap', 0))) if profile.get('mktCap') else None,
                enterprise_value=Decimal(str(metrics.get('enterpriseValue', 0))) if metrics.get('enterpriseValue') else None,
                pe_ratio=Decimal(str(profile.get('pe', 0))) if profile.get('pe') else None,
                pb_ratio=Decimal(str(metrics.get('pbRatio', 0))) if metrics.get('pbRatio') else None,
                ps_ratio=Decimal(str(metrics.get('priceToSalesRatio', 0))) if metrics.get('priceToSalesRatio') else None,
                peg_ratio=Decimal(str(metrics.get('pegRatio', 0))) if metrics.get('pegRatio') else None,
                ev_ebitda=Decimal(str(metrics.get('evToEbitda', 0))) if metrics.get('evToEbitda') else None,
                profit_margin=Decimal(str(metrics.get('netProfitMargin', 0))) if metrics.get('netProfitMargin') else None,
                operating_margin=Decimal(str(metrics.get('operatingProfitMargin', 0))) if metrics.get('operatingProfitMargin') else None,
                roe=Decimal(str(metrics.get('roe', 0))) if metrics.get('roe') else None,
                roa=Decimal(str(metrics.get('roa', 0))) if metrics.get('roa') else None,
                roic=Decimal(str(metrics.get('roic', 0))) if metrics.get('roic') else None,
                revenue_growth=Decimal(str(metrics.get('revenueGrowth', 0))) if metrics.get('revenueGrowth') else None,
                earnings_growth=Decimal(str(metrics.get('earningsGrowth', 0))) if metrics.get('earningsGrowth') else None,
                revenue_per_share=Decimal(str(metrics.get('revenuePerShare', 0))) if metrics.get('revenuePerShare') else None,
                eps=Decimal(str(profile.get('eps', 0))) if profile.get('eps') else None,
                debt_to_equity=Decimal(str(ratios.get('debtEquityRatio', 0))) if ratios.get('debtEquityRatio') else None,
                current_ratio=Decimal(str(ratios.get('currentRatio', 0))) if ratios.get('currentRatio') else None,
                quick_ratio=Decimal(str(ratios.get('quickRatio', 0))) if ratios.get('quickRatio') else None,
                dividend_yield=Decimal(str(profile.get('dividendYield', 0))) if profile.get('dividendYield') else None,
                dividend_payout_ratio=Decimal(str(ratios.get('dividendPayoutRatio', 0))) if ratios.get('dividendPayoutRatio') else None,
                sector=profile.get('sector'),
                industry=profile.get('industry'),
                employees=int(profile.get('fullTimeEmployees', 0)) if profile.get('fullTimeEmployees') else None,
                description=profile.get('description'),
                source="fmp",
                updated_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"FMP.get_fundamentals for {ticker} failed: {e}")
            return None

    async def get_news(self, ticker: str, limit: int = 10) -> list[NewsArticle]:
        """
        Récupère les actualités

        Args:
            ticker: Symbole de l'action
            limit: Nombre max d'articles

        Returns:
            Liste d'articles
        """
        if not self.is_available():
            return []

        try:
            data = await self._make_request(f"stock_news", {'tickers': ticker, 'limit': limit})

            if not data:
                return []

            articles = []
            for item in data[:limit]:
                articles.append(NewsArticle(
                    title=item.get('title', ''),
                    description=item.get('text', ''),
                    url=item.get('url', ''),
                    source=item.get('site', 'fmp'),
                    published_at=datetime.strptime(item['publishedDate'], '%Y-%m-%d %H:%M:%S') if item.get('publishedDate') else datetime.now(),
                    sentiment=None,  # FMP ne fournit pas de sentiment
                    sentiment_score=None
                ))

            return articles

        except Exception as e:
            logger.error(f"FMP.get_news for {ticker} failed: {e}")
            return []
