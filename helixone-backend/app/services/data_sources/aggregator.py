"""
Agrégateur de données avec fallback automatique
Inspiré d'Aladdin Data Cloud
"""

from typing import Optional, List
from datetime import date
import logging

from app.services.data_sources.base import (
    BaseDataSource,
    DataUnavailableError,
    InvalidTickerError
)
from app.services.data_sources.yahoo_finance import YahooFinanceSource
from app.services.data_sources.finnhub_source import FinnhubSource
from app.services.data_sources.fmp_source import FMPSource
from app.services.data_sources.alphavantage_source import AlphaVantageSource
from app.services.data_sources.twelvedata_source import TwelveDataSource
from app.services.data_sources.fred_source import FREDSource
from app.schemas.market import (
    Quote,
    HistoricalData,
    Fundamentals,
    NewsArticle,
    SearchResult,
    ESGScore
)

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Agrégateur intelligent de données financières

    Combine plusieurs sources avec fallback automatique:
    1. Essaie la source principale
    2. Si échec, essaie les sources de secours
    3. Retourne la première réponse valide
    """

    def __init__(self, sources: Optional[List[BaseDataSource]] = None):
        """
        Initialise l'agrégateur

        Args:
            sources: Liste optionnelle de sources personnalisées
        """
        if sources:
            self.sources = sources
        else:
            # Sources par défaut (ordre de priorité)
            self.sources = [
                YahooFinanceSource(),      # Priorité 1: Yahoo (gratuit, illimité, très fiable)
                FinnhubSource(),           # Priorité 2: Finnhub (60 req/min, ESG, news avec sentiment)
                AlphaVantageSource(),      # Priorité 3: Alpha Vantage (5 req/min, fondamentaux complets)
                FMPSource(),               # Priorité 4: FMP (250 req/jour, très bons ratios financiers)
                TwelveDataSource(),        # Priorité 5: Twelve Data (8 req/min, bonne couverture internationale)
                # FRED n'est pas ajouté ici car il ne fournit que des données macro (pas d'actions)
            ]

        # Filtrer uniquement les sources disponibles
        self.available_sources = [s for s in self.sources if s.is_available()]

        if not self.available_sources:
            logger.warning("Aucune source de données disponible!")
        else:
            logger.info(
                f"DataAggregator initialisé avec {len(self.available_sources)} sources: "
                f"{[s.name for s in self.available_sources]}"
            )

    async def get_quote(self, ticker: str) -> Quote:
        """
        Récupère le quote avec fallback automatique

        Args:
            ticker: Symbole de l'action

        Returns:
            Quote

        Raises:
            DataUnavailableError: Si aucune source ne fonctionne
        """
        ticker = ticker.upper()
        errors = []

        for source in self.available_sources:
            try:
                logger.debug(f"Tentative get_quote({ticker}) avec {source.name}")
                quote = await source.get_quote(ticker)

                if quote and self._is_valid_quote(quote):
                    logger.info(
                        f"✅ Quote récupéré pour {ticker} via {source.name}: ${quote.price}"
                    )
                    return quote
                else:
                    logger.warning(f"Quote invalide de {source.name} pour {ticker}")

            except InvalidTickerError as e:
                # Si le ticker est invalide, pas besoin d'essayer les autres sources
                logger.error(f"❌ Ticker invalide: {ticker}")
                raise

            except Exception as e:
                error_msg = f"{source.name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"⚠️  {error_msg}")
                continue

        # Aucune source n'a fonctionné
        error_detail = "; ".join(errors) if errors else "No sources available"
        logger.error(f"❌ Échec get_quote({ticker}) - Toutes les sources ont échoué: {error_detail}")
        raise DataUnavailableError("DataAggregator", f"Quote non disponible pour {ticker}: {error_detail}")

    async def get_historical(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> HistoricalData:
        """
        Récupère les données historiques avec fallback

        Args:
            ticker: Symbole de l'action
            start_date: Date de début
            end_date: Date de fin
            interval: Intervalle (1d, 1wk, 1mo)

        Returns:
            HistoricalData

        Raises:
            DataUnavailableError: Si aucune source ne fonctionne
        """
        ticker = ticker.upper()
        errors = []

        for source in self.available_sources:
            try:
                logger.debug(f"Tentative get_historical({ticker}) avec {source.name}")
                historical = await source.get_historical(ticker, start_date, end_date, interval)

                if historical and historical.prices:
                    logger.info(
                        f"✅ Données historiques récupérées pour {ticker} via {source.name}: "
                        f"{len(historical.prices)} points"
                    )
                    return historical
                else:
                    logger.warning(f"Données historiques vides de {source.name} pour {ticker}")

            except Exception as e:
                error_msg = f"{source.name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"⚠️  {error_msg}")
                continue

        # Aucune source n'a fonctionné
        error_detail = "; ".join(errors) if errors else "No sources available"
        logger.error(f"❌ Échec get_historical({ticker})")
        raise DataUnavailableError(
            "DataAggregator",
            f"Données historiques non disponibles pour {ticker}: {error_detail}"
        )

    async def get_fundamentals(self, ticker: str) -> Fundamentals:
        """
        Récupère les données fondamentales avec fallback

        Args:
            ticker: Symbole de l'action

        Returns:
            Fundamentals

        Raises:
            DataUnavailableError: Si aucune source ne fonctionne
        """
        ticker = ticker.upper()
        errors = []

        for source in self.available_sources:
            try:
                logger.debug(f"Tentative get_fundamentals({ticker}) avec {source.name}")
                fundamentals = await source.get_fundamentals(ticker)

                if fundamentals:
                    logger.info(f"✅ Fondamentaux récupérés pour {ticker} via {source.name}")
                    return fundamentals

            except Exception as e:
                error_msg = f"{source.name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"⚠️  {error_msg}")
                continue

        # Aucune source n'a fonctionné
        error_detail = "; ".join(errors) if errors else "No sources available"
        logger.error(f"❌ Échec get_fundamentals({ticker})")
        raise DataUnavailableError(
            "DataAggregator",
            f"Fondamentaux non disponibles pour {ticker}: {error_detail}"
        )

    async def get_news(
        self,
        ticker: Optional[str] = None,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Récupère les actualités de toutes les sources disponibles

        Args:
            ticker: Symbole optionnel
            limit: Nombre max d'articles total

        Returns:
            Liste d'articles (peut être vide)
        """
        all_articles = []

        for source in self.available_sources:
            try:
                articles = await source.get_news(ticker, limit=limit)
                if articles:
                    all_articles.extend(articles)
                    logger.debug(f"News récupérées de {source.name}: {len(articles)} articles")
            except Exception as e:
                logger.warning(f"Erreur get_news avec {source.name}: {e}")
                continue

        # Dédupliquer par URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        # Trier par date (plus récent en premier)
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)

        # Limiter au nombre demandé
        result = unique_articles[:limit]

        logger.info(f"News agrégées: {len(result)} articles uniques")
        return result

    def _is_valid_quote(self, quote: Quote) -> bool:
        """
        Vérifie qu'un quote est valide

        Args:
            quote: Quote à vérifier

        Returns:
            True si valide
        """
        if not quote:
            return False

        if not quote.price or quote.price <= 0:
            logger.warning(f"Quote invalide: prix = {quote.price}")
            return False

        return True

    async def get_esg_scores(self, ticker: str) -> Optional[ESGScore]:
        """
        Récupère les scores ESG avec fallback

        Args:
            ticker: Symbole de l'action

        Returns:
            ESGScore ou None si indisponible
        """
        ticker = ticker.upper()

        for source in self.available_sources:
            try:
                # Vérifier si la source supporte ESG
                if hasattr(source, 'get_esg_scores'):
                    logger.debug(f"Tentative get_esg_scores({ticker}) avec {source.name}")
                    esg = await source.get_esg_scores(ticker)

                    if esg:
                        logger.info(f"✅ Scores ESG récupérés pour {ticker} via {source.name}")
                        return esg

            except Exception as e:
                logger.warning(f"⚠️  Erreur ESG avec {source.name}: {e}")
                continue

        logger.info(f"Aucune source ESG disponible pour {ticker}")
        return None

    async def get_fundamentals_merged(self, ticker: str) -> Optional[Fundamentals]:
        """
        Récupère et fusionne les données fondamentales de plusieurs sources
        Inspiré de get_fondamentaux_unifies() dans source_manager.py

        Args:
            ticker: Symbole de l'action

        Returns:
            Fundamentals fusionné avec les meilleures données de chaque source
        """
        ticker = ticker.upper()
        all_fundamentals = []

        # Récupérer les fondamentaux de toutes les sources
        for source in self.available_sources:
            try:
                fundamentals = await source.get_fundamentals(ticker)
                if fundamentals:
                    all_fundamentals.append(fundamentals)
                    logger.debug(f"Fondamentaux récupérés de {source.name}")
            except Exception as e:
                logger.warning(f"Erreur fondamentaux avec {source.name}: {e}")
                continue

        if not all_fundamentals:
            logger.warning(f"Aucun fondamental disponible pour {ticker}")
            return None

        # Fusionner en choisissant la première valeur non-null de chaque champ
        # (stratégie: première source = priorité)
        base = all_fundamentals[0]

        # Pour chaque champ optionnel, prendre la première valeur non-None
        for other in all_fundamentals[1:]:
            if base.market_cap is None and other.market_cap:
                base.market_cap = other.market_cap
            if base.pe_ratio is None and other.pe_ratio:
                base.pe_ratio = other.pe_ratio
            if base.pb_ratio is None and other.pb_ratio:
                base.pb_ratio = other.pb_ratio
            if base.debt_to_equity is None and other.debt_to_equity:
                base.debt_to_equity = other.debt_to_equity
            if base.roe is None and other.roe:
                base.roe = other.roe
            if base.revenue_growth is None and other.revenue_growth:
                base.revenue_growth = other.revenue_growth
            if base.eps is None and other.eps:
                base.eps = other.eps
            if base.dividend_yield is None and other.dividend_yield:
                base.dividend_yield = other.dividend_yield
            if base.sector is None and other.sector:
                base.sector = other.sector
            if base.industry is None and other.industry:
                base.industry = other.industry
            # ... etc pour tous les champs

        # Mettre à jour la source pour indiquer la fusion
        base.source = f"merged:{','.join([f.source for f in all_fundamentals])}"

        logger.info(f"✅ Fondamentaux fusionnés pour {ticker} depuis {len(all_fundamentals)} sources")
        return base

    def get_available_sources(self) -> List[str]:
        """
        Retourne la liste des sources disponibles

        Returns:
            Liste des noms de sources
        """
        return [source.name for source in self.available_sources]


# Instance globale pour réutilisation
_default_aggregator: Optional[DataAggregator] = None


def get_default_aggregator() -> DataAggregator:
    """
    Retourne l'agrégateur par défaut (singleton)

    Returns:
        DataAggregator
    """
    global _default_aggregator
    if _default_aggregator is None:
        _default_aggregator = DataAggregator()
    return _default_aggregator
