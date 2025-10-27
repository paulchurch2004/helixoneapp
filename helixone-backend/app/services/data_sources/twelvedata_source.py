"""
Source de données Twelve Data
API gratuite: 8 requêtes/minute, 800/jour
Documentation: https://twelvedata.com/docs
"""

import logging
from typing import Optional
from datetime import date, datetime
from decimal import Decimal
import aiohttp

from app.services.data_sources.base import BaseDataSource, DataUnavailableError
from app.schemas.market import Quote, HistoricalData, HistoricalPrice, Fundamentals
from app.core.config import settings

logger = logging.getLogger(__name__)


class TwelveDataSource(BaseDataSource):
    """
    Source de données Twelve Data

    Capacités:
    - Prix en temps réel
    - Données historiques intraday et daily
    - Données techniques
    - Bonne couverture internationale

    Limites gratuites:
    - 8 requêtes/minute
    - 800 requêtes/jour
    - Données avec léger délai
    """

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client Twelve Data

        Args:
            api_key: Clé API Twelve Data (ou depuis settings)
        """
        self.api_key = api_key or settings.TWELVEDATA_API_KEY

        if not self.api_key or self.api_key == "your_key_here":
            logger.warning("Twelve Data API key not configured")

    @property
    def name(self) -> str:
        return "TwelveData"

    def is_available(self) -> bool:
        """Vérifie si la source est disponible"""
        return bool(self.api_key and self.api_key != "your_key_here")

    async def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Effectue une requête HTTP vers l'API Twelve Data

        Args:
            endpoint: Endpoint de l'API
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

                data = await response.json()

                # Twelve Data retourne des erreurs dans le JSON
                if 'status' in data and data['status'] == 'error':
                    raise DataUnavailableError(
                        self.name,
                        data.get('message', 'Unknown error')
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
            data = await self._make_request("price", {'symbol': ticker})

            if not data or 'price' not in data:
                return None

            price = Decimal(str(data['price']))

            # Récupérer aussi le profil pour avoir plus d'infos
            try:
                quote_data = await self._make_request("quote", {'symbol': ticker})

                return Quote(
                    ticker=ticker.upper(),
                    name=quote_data.get('name', ticker),
                    price=price,
                    previous_close=Decimal(str(quote_data.get('previous_close', 0))) if quote_data.get('previous_close') else None,
                    open=Decimal(str(quote_data.get('open', 0))) if quote_data.get('open') else None,
                    high=Decimal(str(quote_data.get('high', 0))) if quote_data.get('high') else None,
                    low=Decimal(str(quote_data.get('low', 0))) if quote_data.get('low') else None,
                    volume=int(quote_data.get('volume', 0)) if quote_data.get('volume') else None,
                    change=Decimal(str(quote_data.get('change', 0))) if quote_data.get('change') else None,
                    change_percent=Decimal(str(quote_data.get('percent_change', 0))) if quote_data.get('percent_change') else None,
                    timestamp=datetime.now(),
                    source="twelvedata"
                )
            except:
                # Si quote échoue, retourner juste le prix
                return Quote(
                    ticker=ticker.upper(),
                    name=ticker,
                    price=price,
                    timestamp=datetime.now(),
                    source="twelvedata"
                )

        except Exception as e:
            logger.error(f"TwelveData.get_quote for {ticker} failed: {e}")
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
            interval: Intervalle (1d, 1h, etc.)

        Returns:
            HistoricalData ou None si erreur
        """
        if not self.is_available():
            return None

        try:
            # Calculer le nombre de jours
            days = (end_date - start_date).days

            # Twelve Data utilise "outputsize" pour limiter les résultats
            params = {
                'symbol': ticker,
                'interval': '1day' if interval == '1d' else interval,
                'outputsize': min(days + 10, 5000),  # Max 5000
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }

            data = await self._make_request("time_series", params)

            if not data or 'values' not in data:
                return None

            prices = []
            for item in data['values']:
                price_date = datetime.strptime(item['datetime'], '%Y-%m-%d').date()

                # Filtrer les dates hors de la plage demandée
                if start_date <= price_date <= end_date:
                    prices.append(HistoricalPrice(
                        date=price_date,
                        open=Decimal(str(item['open'])),
                        high=Decimal(str(item['high'])),
                        low=Decimal(str(item['low'])),
                        close=Decimal(str(item['close'])),
                        volume=int(item.get('volume', 0)) if item.get('volume') else 0,
                        adjusted_close=Decimal(str(item['close']))  # Twelve Data ne fournit pas adjusted
                    ))

            # Twelve Data retourne du plus récent au plus ancien, on inverse
            prices.reverse()

            return HistoricalData(
                ticker=ticker.upper(),
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                prices=prices,
                source="twelvedata"
            )

        except Exception as e:
            logger.error(f"TwelveData.get_historical for {ticker} failed: {e}")
            return None

    async def get_fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        """
        Récupère les données fondamentales (limité sur Twelve Data gratuit)

        Args:
            ticker: Symbole de l'action

        Returns:
            Fundamentals ou None si erreur
        """
        if not self.is_available():
            return None

        try:
            # Twelve Data gratuit a des fondamentaux limités
            # On essaie de récupérer ce qui est disponible
            data = await self._make_request("quote", {'symbol': ticker})

            if not data:
                return None

            # Twelve Data gratuit ne fournit pas beaucoup de fondamentaux
            # On retourne ce qu'on peut
            return Fundamentals(
                ticker=ticker.upper(),
                market_cap=None,  # Pas fourni dans le plan gratuit
                pe_ratio=None,
                sector=None,
                industry=None,
                source="twelvedata",
                updated_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"TwelveData.get_fundamentals for {ticker} failed: {e}")
            return None
