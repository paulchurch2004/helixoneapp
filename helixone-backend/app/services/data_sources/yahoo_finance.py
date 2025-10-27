"""
Source de données Yahoo Finance via yfinance
100% gratuit, pas besoin d'API key
"""

import yfinance as yf
from typing import Optional, List
from datetime import date, datetime
from decimal import Decimal
import pandas as pd

from app.services.data_sources.base import (
    BaseDataSource,
    DataUnavailableError,
    InvalidTickerError
)
from app.schemas.market import (
    Quote,
    HistoricalData,
    HistoricalPrice,
    Fundamentals,
    NewsArticle
)


class YahooFinanceSource(BaseDataSource):
    """
    Source de données Yahoo Finance
    Gratuit, fiable, mais non officiel
    """

    def __init__(self):
        super().__init__(api_key=None)  # Pas d'API key nécessaire

    async def get_quote(self, ticker: str) -> Optional[Quote]:
        """Récupère le quote en temps réel"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or 'currentPrice' not in info and 'regularMarketPrice' not in info:
                raise InvalidTickerError(self.name, f"Ticker {ticker} non trouvé")

            # Prix actuel
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                raise DataUnavailableError(self.name, f"Prix non disponible pour {ticker}")

            # Previous close pour calculer le changement
            previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose')

            change = None
            change_percent = None
            if previous_close and current_price:
                change = Decimal(str(current_price - previous_close))
                change_percent = Decimal(str((change / previous_close) * 100))

            quote = Quote(
                ticker=ticker.upper(),
                name=info.get('longName') or info.get('shortName'),
                price=Decimal(str(current_price)),
                change=change,
                change_percent=change_percent,
                volume=info.get('volume'),
                market_cap=info.get('marketCap'),
                open=Decimal(str(info['open'])) if info.get('open') else None,
                high=Decimal(str(info['dayHigh'])) if info.get('dayHigh') else None,
                low=Decimal(str(info['dayLow'])) if info.get('dayLow') else None,
                previous_close=Decimal(str(previous_close)) if previous_close else None,
                source="yahoo"
            )

            self.logger.info(f"Quote récupéré pour {ticker}: ${current_price}")
            return quote

        except (InvalidTickerError, DataUnavailableError):
            raise
        except Exception as e:
            self._log_error("get_quote", e, ticker)
            return None

    async def get_historical(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> Optional[HistoricalData]:
        """Récupère les données historiques"""
        try:
            stock = yf.Ticker(ticker)

            # Télécharger les données
            df = stock.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            if df.empty:
                raise DataUnavailableError(
                    self.name,
                    f"Pas de données historiques pour {ticker}"
                )

            # Convertir en liste de HistoricalPrice
            prices = []
            for idx, row in df.iterrows():
                price = HistoricalPrice(
                    date=idx.date(),
                    open=Decimal(str(row['Open'])),
                    high=Decimal(str(row['High'])),
                    low=Decimal(str(row['Low'])),
                    close=Decimal(str(row['Close'])),
                    volume=int(row['Volume']),
                    adjusted_close=Decimal(str(row['Close']))  # Yahoo ajuste déjà
                )
                prices.append(price)

            historical = HistoricalData(
                ticker=ticker.upper(),
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                prices=prices,
                source="yahoo"
            )

            self.logger.info(
                f"Données historiques récupérées pour {ticker}: {len(prices)} points"
            )
            return historical

        except DataUnavailableError:
            raise
        except Exception as e:
            self._log_error("get_historical", e, ticker)
            return None

    async def get_fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        """Récupère les données fondamentales"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                raise InvalidTickerError(self.name, f"Ticker {ticker} non trouvé")

            fundamentals = Fundamentals(
                ticker=ticker.upper(),
                # Valuation
                market_cap=info.get('marketCap'),
                enterprise_value=info.get('enterpriseValue'),
                pe_ratio=Decimal(str(info['trailingPE'])) if info.get('trailingPE') else None,
                pb_ratio=Decimal(str(info['priceToBook'])) if info.get('priceToBook') else None,
                ps_ratio=Decimal(str(info['priceToSalesTrailing12Months'])) if info.get('priceToSalesTrailing12Months') else None,
                peg_ratio=Decimal(str(info['pegRatio'])) if info.get('pegRatio') else None,
                ev_ebitda=Decimal(str(info['enterpriseToEbitda'])) if info.get('enterpriseToEbitda') else None,
                # Profitabilité
                profit_margin=Decimal(str(info['profitMargins'] * 100)) if info.get('profitMargins') else None,
                operating_margin=Decimal(str(info['operatingMargins'] * 100)) if info.get('operatingMargins') else None,
                roe=Decimal(str(info['returnOnEquity'] * 100)) if info.get('returnOnEquity') else None,
                roa=Decimal(str(info['returnOnAssets'] * 100)) if info.get('returnOnAssets') else None,
                # Croissance
                revenue_growth=Decimal(str(info['revenueGrowth'] * 100)) if info.get('revenueGrowth') else None,
                earnings_growth=Decimal(str(info['earningsGrowth'] * 100)) if info.get('earningsGrowth') else None,
                revenue_per_share=Decimal(str(info['revenuePerShare'])) if info.get('revenuePerShare') else None,
                eps=Decimal(str(info['trailingEps'])) if info.get('trailingEps') else None,
                # Santé financière
                debt_to_equity=Decimal(str(info['debtToEquity'])) if info.get('debtToEquity') else None,
                current_ratio=Decimal(str(info['currentRatio'])) if info.get('currentRatio') else None,
                quick_ratio=Decimal(str(info['quickRatio'])) if info.get('quickRatio') else None,
                # Dividendes
                dividend_yield=Decimal(str(info['dividendYield'] * 100)) if info.get('dividendYield') else None,
                dividend_payout_ratio=Decimal(str(info['payoutRatio'] * 100)) if info.get('payoutRatio') else None,
                # Informations générales
                sector=info.get('sector'),
                industry=info.get('industry'),
                employees=info.get('fullTimeEmployees'),
                description=info.get('longBusinessSummary'),
                source="yahoo"
            )

            self.logger.info(f"Fondamentaux récupérés pour {ticker}")
            return fundamentals

        except InvalidTickerError:
            raise
        except Exception as e:
            self._log_error("get_fundamentals", e, ticker)
            return None

    async def get_news(
        self,
        ticker: Optional[str] = None,
        limit: int = 50
    ) -> List[NewsArticle]:
        """Récupère les actualités"""
        if not ticker:
            self.logger.warning("Yahoo Finance nécessite un ticker pour les news")
            return []

        try:
            stock = yf.Ticker(ticker)
            news_data = stock.news

            if not news_data:
                return []

            articles = []
            for item in news_data[:limit]:
                article = NewsArticle(
                    title=item.get('title', ''),
                    description=item.get('summary'),
                    url=item.get('link', ''),
                    source=item.get('publisher', 'Yahoo Finance'),
                    published_at=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    related_tickers=[ticker.upper()]
                )
                articles.append(article)

            self.logger.info(f"News récupérées pour {ticker}: {len(articles)} articles")
            return articles

        except Exception as e:
            self._log_error("get_news", e, ticker)
            return []

    def is_available(self) -> bool:
        """Yahoo Finance est toujours disponible (pas d'API key)"""
        return True
