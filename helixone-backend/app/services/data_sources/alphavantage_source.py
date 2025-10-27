"""
Source de données Alpha Vantage
API gratuite: 5 requêtes/minute, 500/jour
Documentation: https://www.alphavantage.co/documentation/
"""

import logging
from typing import Optional
from datetime import date, datetime, timedelta
from decimal import Decimal
import aiohttp

from app.services.data_sources.base import BaseDataSource, DataUnavailableError
from app.schemas.market import Quote, HistoricalData, HistoricalPrice, Fundamentals
from app.core.config import settings

logger = logging.getLogger(__name__)


class AlphaVantageSource(BaseDataSource):
    """
    Source de données Alpha Vantage

    Capacités:
    - Prix en temps réel
    - Données historiques complètes (daily, weekly, monthly)
    - Données techniques et indicateurs
    - Données de crypto et forex
    - Données fondamentales de base

    Limites gratuites:
    - 5 requêtes/minute
    - 500 requêtes/jour
    - Bonne qualité de données
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client Alpha Vantage

        Args:
            api_key: Clé API Alpha Vantage (ou depuis settings)
        """
        self.api_key = api_key or settings.ALPHA_VANTAGE_API_KEY

        if not self.api_key or self.api_key == "your_key_here":
            logger.warning("Alpha Vantage API key not configured")

    @property
    def name(self) -> str:
        return "AlphaVantage"

    def is_available(self) -> bool:
        """Vérifie si la source est disponible"""
        return bool(self.api_key and self.api_key != "your_key_here")

    async def _make_request(self, params: dict) -> dict:
        """
        Effectue une requête HTTP vers l'API Alpha Vantage

        Args:
            params: Paramètres de la requête

        Returns:
            Réponse JSON
        """
        params['apikey'] = self.api_key

        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    raise DataUnavailableError(
                        self.name,
                        f"HTTP {response.status}: {await response.text()}"
                    )

                data = await response.json()

                # Alpha Vantage retourne des messages d'erreur dans le JSON
                if 'Error Message' in data:
                    raise DataUnavailableError(
                        self.name,
                        data['Error Message']
                    )

                if 'Note' in data:
                    # Rate limit atteint
                    raise DataUnavailableError(
                        self.name,
                        "Rate limit atteint (5 requêtes/minute)"
                    )

                return data

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
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': ticker
            }

            data = await self._make_request(params)

            if not data or 'Global Quote' not in data:
                return None

            quote_data = data['Global Quote']

            if not quote_data:
                return None

            return Quote(
                ticker=ticker.upper(),
                name=ticker,  # Alpha Vantage ne fournit pas le nom dans GLOBAL_QUOTE
                price=Decimal(str(quote_data.get('05. price', 0))),
                previous_close=Decimal(str(quote_data.get('08. previous close', 0))) if quote_data.get('08. previous close') else None,
                open=Decimal(str(quote_data.get('02. open', 0))) if quote_data.get('02. open') else None,
                high=Decimal(str(quote_data.get('03. high', 0))) if quote_data.get('03. high') else None,
                low=Decimal(str(quote_data.get('04. low', 0))) if quote_data.get('04. low') else None,
                volume=int(quote_data.get('06. volume', 0)) if quote_data.get('06. volume') else None,
                change=Decimal(str(quote_data.get('09. change', 0))) if quote_data.get('09. change') else None,
                change_percent=Decimal(str(quote_data.get('10. change percent', '0%').rstrip('%'))) if quote_data.get('10. change percent') else None,
                timestamp=datetime.now(),
                source="alphavantage"
            )

        except Exception as e:
            logger.error(f"AlphaVantage.get_quote for {ticker} failed: {e}")
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
            interval: Intervalle (1d supporté)

        Returns:
            HistoricalData ou None si erreur
        """
        if not self.is_available():
            return None

        try:
            # Alpha Vantage a TIME_SERIES_DAILY pour les données historiques
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full'  # Récupérer toutes les données (20+ ans)
            }

            data = await self._make_request(params)

            if not data or 'Time Series (Daily)' not in data:
                return None

            time_series = data['Time Series (Daily)']

            prices = []
            for date_str, values in time_series.items():
                price_date = datetime.strptime(date_str, '%Y-%m-%d').date()

                # Filtrer uniquement les dates dans la plage demandée
                if start_date <= price_date <= end_date:
                    prices.append(HistoricalPrice(
                        date=price_date,
                        open=Decimal(str(values['1. open'])),
                        high=Decimal(str(values['2. high'])),
                        low=Decimal(str(values['3. low'])),
                        close=Decimal(str(values['4. close'])),
                        volume=int(values.get('6. volume', 0)),
                        adjusted_close=Decimal(str(values.get('5. adjusted close', values['4. close'])))
                    ))

            # Trier par date (du plus ancien au plus récent)
            prices.sort(key=lambda x: x.date)

            return HistoricalData(
                ticker=ticker.upper(),
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                prices=prices,
                source="alphavantage"
            )

        except Exception as e:
            logger.error(f"AlphaVantage.get_historical for {ticker} failed: {e}")
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
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker
            }

            data = await self._make_request(params)

            if not data or 'Symbol' not in data:
                return None

            return Fundamentals(
                ticker=ticker.upper(),
                market_cap=Decimal(str(data.get('MarketCapitalization', 0))) if data.get('MarketCapitalization') else None,
                pe_ratio=Decimal(str(data.get('PERatio', 0))) if data.get('PERatio') else None,
                forward_pe=Decimal(str(data.get('ForwardPE', 0))) if data.get('ForwardPE') else None,
                peg_ratio=Decimal(str(data.get('PEGRatio', 0))) if data.get('PEGRatio') else None,
                price_to_book=Decimal(str(data.get('PriceToBookRatio', 0))) if data.get('PriceToBookRatio') else None,
                price_to_sales=Decimal(str(data.get('PriceToSalesRatioTTM', 0))) if data.get('PriceToSalesRatioTTM') else None,
                enterprise_value=Decimal(str(data.get('EnterpriseValue', 0))) if data.get('EnterpriseValue') else None,
                ev_to_ebitda=Decimal(str(data.get('EVToEBITDA', 0))) if data.get('EVToEBITDA') else None,
                profit_margin=Decimal(str(data.get('ProfitMargin', 0))) if data.get('ProfitMargin') else None,
                operating_margin=Decimal(str(data.get('OperatingMarginTTM', 0))) if data.get('OperatingMarginTTM') else None,
                return_on_assets=Decimal(str(data.get('ReturnOnAssetsTTM', 0))) if data.get('ReturnOnAssetsTTM') else None,
                return_on_equity=Decimal(str(data.get('ReturnOnEquityTTM', 0))) if data.get('ReturnOnEquityTTM') else None,
                revenue=Decimal(str(data.get('RevenueTTM', 0))) if data.get('RevenueTTM') else None,
                revenue_per_share=Decimal(str(data.get('RevenuePerShareTTM', 0))) if data.get('RevenuePerShareTTM') else None,
                revenue_growth=Decimal(str(data.get('QuarterlyRevenueGrowthYOY', 0))) if data.get('QuarterlyRevenueGrowthYOY') else None,
                earnings_growth=Decimal(str(data.get('QuarterlyEarningsGrowthYOY', 0))) if data.get('QuarterlyEarningsGrowthYOY') else None,
                eps=Decimal(str(data.get('EPS', 0))) if data.get('EPS') else None,
                debt_to_equity=Decimal(str(data.get('DebtToEquity', 0))) if data.get('DebtToEquity') else None,
                current_ratio=Decimal(str(data.get('CurrentRatio', 0))) if data.get('CurrentRatio') else None,
                quick_ratio=Decimal(str(data.get('QuickRatio', 0))) if data.get('QuickRatio') else None,
                dividend_yield=Decimal(str(data.get('DividendYield', 0))) if data.get('DividendYield') else None,
                dividend_payout_ratio=Decimal(str(data.get('PayoutRatio', 0))) if data.get('PayoutRatio') else None,
                beta=Decimal(str(data.get('Beta', 0))) if data.get('Beta') else None,
                week_52_high=Decimal(str(data.get('52WeekHigh', 0))) if data.get('52WeekHigh') else None,
                week_52_low=Decimal(str(data.get('52WeekLow', 0))) if data.get('52WeekLow') else None,
                shares_outstanding=int(data.get('SharesOutstanding', 0)) if data.get('SharesOutstanding') else None,
                sector=data.get('Sector'),
                industry=data.get('Industry'),
                employees=int(data.get('FullTimeEmployees', 0)) if data.get('FullTimeEmployees') else None,
                description=data.get('Description'),
                timestamp=datetime.now(),
                source="alphavantage"
            )

        except Exception as e:
            logger.error(f"AlphaVantage.get_fundamentals for {ticker} failed: {e}")
            return None
