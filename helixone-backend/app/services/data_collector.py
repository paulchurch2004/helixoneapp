"""
Service de collecte de données de marché
Supporte Yahoo Finance, Alpha Vantage, et autres sources
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from sqlalchemy.orm import Session
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import random

from app.models.market_data import (
    MarketDataOHLCV,
    MarketDataTick,
    MarketDataQuote,
    DataCollectionJob,
    SymbolMetadata,
    TimeframeType,
    DataSourceType,
    CollectionStatus
)

logger = logging.getLogger(__name__)


class DataCollectorService:
    """Service principal de collecte de données de marché"""

    def __init__(self, db: Session):
        self.db = db
        self.executor = ThreadPoolExecutor(max_workers=5)

    # ============================================
    # YAHOO FINANCE - Données journalières
    # ============================================

    def collect_daily_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        adjusted: bool = True
    ) -> Tuple[int, int]:
        """
        Collecte les données journalières pour un symbol

        Args:
            symbol: Ticker du symbol (ex: AAPL)
            start_date: Date de début
            end_date: Date de fin
            adjusted: Prix ajustés pour splits/dividendes

        Returns:
            (records_collected, records_failed)
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                logger.info(f"📥 Collecte données journalières: {symbol} du {start_date} au {end_date}")

                # Délai aléatoire pour éviter le rate limiting (1-3 secondes)
                time.sleep(random.uniform(1.0, 3.0))

                # Télécharger avec yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=adjusted
                )

                if df.empty:
                    logger.warning(f"Aucune donnée pour {symbol}")
                    return 0, 0

                records_collected = 0
                records_failed = 0

                # Sauvegarder chaque ligne
                for timestamp, row in df.iterrows():
                    try:
                        # Vérifier si déjà existe
                        existing = self.db.query(MarketDataOHLCV).filter(
                            MarketDataOHLCV.symbol == symbol,
                            MarketDataOHLCV.timeframe == TimeframeType.DAILY,
                            MarketDataOHLCV.timestamp == timestamp
                        ).first()

                        if existing:
                            # Update
                            existing.open = float(row['Open'])
                            existing.high = float(row['High'])
                            existing.low = float(row['Low'])
                            existing.close = float(row['Close'])
                            existing.volume = int(row['Volume'])
                            existing.is_adjusted = adjusted
                        else:
                            # Insert
                            market_data = MarketDataOHLCV(
                                symbol=symbol,
                                timeframe=TimeframeType.DAILY,
                                timestamp=timestamp,
                                open=float(row['Open']),
                                high=float(row['High']),
                                low=float(row['Low']),
                                close=float(row['Close']),
                                volume=int(row['Volume']),
                                source=DataSourceType.YAHOO_FINANCE,
                                is_adjusted=adjusted
                            )
                            self.db.add(market_data)

                        records_collected += 1

                    except Exception as e:
                        logger.error(f"Erreur sauvegarde ligne {timestamp}: {e}")
                        records_failed += 1

                self.db.commit()
                logger.info(f"✅ {symbol}: {records_collected} enregistrements sauvegardés")

                return records_collected, records_failed

            except Exception as e:
                retry_count += 1
                logger.error(f"Erreur collecte {symbol} (tentative {retry_count}/{max_retries}): {e}")

                if retry_count < max_retries:
                    # Attendre de plus en plus longtemps entre les retries
                    wait_time = retry_count * 5
                    logger.info(f"⏳ Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Échec après {max_retries} tentatives pour {symbol}")
                    self.db.rollback()
                    return 0, 0

        # Si on arrive ici, toutes les tentatives ont échoué
        self.db.rollback()
        return 0, 0

    # ============================================
    # YAHOO FINANCE - Données intraday
    # ============================================

    def collect_intraday_data(
        self,
        symbol: str,
        interval: str = "1m",
        period: str = "7d"
    ) -> Tuple[int, int]:
        """
        Collecte les données intraday (minutes)

        Args:
            symbol: Ticker
            interval: "1m", "2m", "5m", "15m", "30m", "60m", "90m"
            period: "1d", "5d", "7d", "60d", "1mo"

        Returns:
            (records_collected, records_failed)
        """
        try:
            logger.info(f"📥 Collecte intraday {interval}: {symbol} période {period}")

            # Mapping interval -> TimeframeType
            interval_map = {
                "1m": TimeframeType.MINUTE_1,
                "5m": TimeframeType.MINUTE_5,
                "15m": TimeframeType.MINUTE_15,
                "30m": TimeframeType.MINUTE_30,
                "60m": TimeframeType.HOUR_1,
                "1h": TimeframeType.HOUR_1,
            }

            timeframe = interval_map.get(interval, TimeframeType.MINUTE_1)

            # Télécharger
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"Aucune donnée intraday pour {symbol}")
                return 0, 0

            records_collected = 0
            records_failed = 0

            for timestamp, row in df.iterrows():
                try:
                    existing = self.db.query(MarketDataOHLCV).filter(
                        MarketDataOHLCV.symbol == symbol,
                        MarketDataOHLCV.timeframe == timeframe,
                        MarketDataOHLCV.timestamp == timestamp
                    ).first()

                    if existing:
                        existing.open = float(row['Open'])
                        existing.high = float(row['High'])
                        existing.low = float(row['Low'])
                        existing.close = float(row['Close'])
                        existing.volume = int(row['Volume'])
                    else:
                        market_data = MarketDataOHLCV(
                            symbol=symbol,
                            timeframe=timeframe,
                            timestamp=timestamp,
                            open=float(row['Open']),
                            high=float(row['High']),
                            low=float(row['Low']),
                            close=float(row['Close']),
                            volume=int(row['Volume']),
                            source=DataSourceType.YAHOO_FINANCE,
                            is_adjusted=False
                        )
                        self.db.add(market_data)

                    records_collected += 1

                except Exception as e:
                    logger.error(f"Erreur sauvegarde {timestamp}: {e}")
                    records_failed += 1

            self.db.commit()
            logger.info(f"✅ {symbol} {interval}: {records_collected} enregistrements")

            return records_collected, records_failed

        except Exception as e:
            logger.error(f"Erreur collecte intraday {symbol}: {e}")
            self.db.rollback()
            return 0, 0

    # ============================================
    # COLLECTE MULTI-SYMBOLES
    # ============================================

    async def collect_multiple_symbols_daily(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        job_id: Optional[str] = None
    ) -> Dict[str, Tuple[int, int]]:
        """
        Collecte données journalières pour plusieurs symbols en parallèle

        Args:
            symbols: Liste de tickers
            start_date: Date de début
            end_date: Date de fin
            job_id: ID du job de collecte (optionnel)

        Returns:
            Dict {symbol: (collected, failed)}
        """
        logger.info(f"📥 Collecte multiple: {len(symbols)} symbols")

        results = {}
        total_collected = 0
        total_failed = 0

        # Créer ou récupérer le job
        if job_id:
            job = self.db.query(DataCollectionJob).filter(
                DataCollectionJob.id == job_id
            ).first()
            if job:
                job.status = CollectionStatus.IN_PROGRESS
                job.started_at = datetime.utcnow()
                self.db.commit()

        # Collecter chaque symbol
        for idx, symbol in enumerate(symbols):
            try:
                collected, failed = self.collect_daily_data(
                    symbol, start_date, end_date
                )
                results[symbol] = (collected, failed)
                total_collected += collected
                total_failed += failed

                # Update job progress
                if job:
                    job.progress = ((idx + 1) / len(symbols)) * 100
                    job.records_collected = total_collected
                    job.records_failed = total_failed
                    self.db.commit()

            except Exception as e:
                logger.error(f"Erreur {symbol}: {e}")
                results[symbol] = (0, 0)

        # Finaliser le job
        if job:
            job.status = CollectionStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.records_collected = total_collected
            job.records_failed = total_failed
            self.db.commit()

        logger.info(f"✅ Collecte terminée: {total_collected} OK, {total_failed} erreurs")
        return results

    # ============================================
    # MÉTADONNÉES DES SYMBOLES
    # ============================================

    def collect_symbol_metadata(self, symbol: str) -> Optional[SymbolMetadata]:
        """
        Collecte les métadonnées d'un symbol (nom, secteur, etc.)

        Args:
            symbol: Ticker

        Returns:
            SymbolMetadata ou None
        """
        try:
            logger.info(f"📥 Collecte métadonnées: {symbol}")

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                logger.warning(f"Pas d'info pour {symbol}")
                return None

            # Vérifier si existe
            existing = self.db.query(SymbolMetadata).filter(
                SymbolMetadata.symbol == symbol
            ).first()

            if existing:
                # Update
                metadata = existing
            else:
                # Create
                metadata = SymbolMetadata(symbol=symbol)
                self.db.add(metadata)

            # Remplir les données
            metadata.name = info.get('longName') or info.get('shortName')
            metadata.exchange = info.get('exchange')
            metadata.sector = info.get('sector')
            metadata.industry = info.get('industry')
            metadata.market_cap = info.get('marketCap')
            metadata.country = info.get('country')
            metadata.currency = info.get('currency', 'USD')
            metadata.description = info.get('longBusinessSummary')
            metadata.website = info.get('website')
            metadata.average_volume = info.get('averageVolume')
            metadata.float_shares = info.get('floatShares')
            metadata.beta = info.get('beta')
            metadata.pe_ratio = info.get('trailingPE')
            metadata.dividend_yield = info.get('dividendYield')
            metadata.last_data_update = datetime.utcnow()

            self.db.commit()
            logger.info(f"✅ Métadonnées sauvegardées: {metadata.name}")

            return metadata

        except Exception as e:
            logger.error(f"Erreur métadonnées {symbol}: {e}")
            self.db.rollback()
            return None

    # ============================================
    # COLLECTE POUR CRISES HISTORIQUES
    # ============================================

    def collect_crisis_data(
        self,
        crisis_name: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Tuple[int, int]]:
        """
        Collecte données pour une crise historique spécifique

        Args:
            crisis_name: Nom de la crise (ex: "2008_crisis")
            symbols: Liste de tickers à analyser
            start_date: Début de la crise
            end_date: Fin de la crise

        Returns:
            Dict {symbol: (collected, failed)}
        """
        logger.info(f"📥 Collecte crise '{crisis_name}': {len(symbols)} symbols")

        # Créer un job de collecte
        job = DataCollectionJob(
            job_name=f"Crisis Data: {crisis_name}",
            job_type="historical_crisis",
            symbols=",".join(symbols),
            timeframe=TimeframeType.DAILY,
            start_date=start_date,
            end_date=end_date,
            source=DataSourceType.YAHOO_FINANCE,
            status=CollectionStatus.IN_PROGRESS
        )
        self.db.add(job)
        self.db.commit()

        # Collecter les données
        results = {}
        total_collected = 0
        total_failed = 0

        for idx, symbol in enumerate(symbols):
            try:
                collected, failed = self.collect_daily_data(
                    symbol, start_date, end_date
                )
                results[symbol] = (collected, failed)
                total_collected += collected
                total_failed += failed

                # Progress
                job.progress = ((idx + 1) / len(symbols)) * 100
                job.records_collected = total_collected
                job.records_failed = total_failed
                self.db.commit()

            except Exception as e:
                logger.error(f"Erreur {symbol}: {e}")
                results[symbol] = (0, 0)

        # Finaliser
        job.status = CollectionStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        self.db.commit()

        logger.info(f"✅ Crise '{crisis_name}': {total_collected} enregistrements")
        return results

    # ============================================
    # REQUÊTES DE DONNÉES
    # ============================================

    def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: TimeframeType,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Récupère les données OHLCV depuis la base de données

        Args:
            symbol: Ticker
            timeframe: Type de timeframe
            start_date: Date de début
            end_date: Date de fin

        Returns:
            DataFrame avec colonnes OHLCV
        """
        records = self.db.query(MarketDataOHLCV).filter(
            MarketDataOHLCV.symbol == symbol,
            MarketDataOHLCV.timeframe == timeframe,
            MarketDataOHLCV.timestamp >= start_date,
            MarketDataOHLCV.timestamp <= end_date
        ).order_by(MarketDataOHLCV.timestamp).all()

        if not records:
            return pd.DataFrame()

        data = []
        for r in records:
            data.append({
                'timestamp': r.timestamp,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def get_available_symbols(self) -> List[str]:
        """Retourne la liste des symbols pour lesquels on a des données"""
        symbols = self.db.query(SymbolMetadata.symbol).filter(
            SymbolMetadata.is_active == True
        ).all()
        return [s[0] for s in symbols]

    def get_data_coverage(self, symbol: str) -> Dict:
        """
        Retourne la couverture des données pour un symbol

        Returns:
            {
                'daily': {'start': date, 'end': date, 'count': int},
                '1m': {...},
                ...
            }
        """
        coverage = {}

        for timeframe in TimeframeType:
            records = self.db.query(MarketDataOHLCV).filter(
                MarketDataOHLCV.symbol == symbol,
                MarketDataOHLCV.timeframe == timeframe
            ).order_by(MarketDataOHLCV.timestamp).all()

            if records:
                coverage[timeframe.value] = {
                    'start': records[0].timestamp,
                    'end': records[-1].timestamp,
                    'count': len(records)
                }

        return coverage


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_data_collector(db: Session) -> DataCollectorService:
    """Factory pour créer une instance du collector"""
    return DataCollectorService(db)


# ============================================
# COLLECTES PRÉDÉFINIES POUR CRISES
# ============================================

CRISIS_PERIODS = {
    "2008_crisis": {
        "name": "2008 Financial Crisis",
        "start": datetime(2007, 10, 9),
        "end": datetime(2009, 3, 9),
        "symbols": ["SPY", "DIA", "QQQ", "XLF", "XLE", "XLK", "XLV", "BAC", "C", "GS", "JPM", "AIG"]
    },
    "covid_2020": {
        "name": "COVID-19 Crash",
        "start": datetime(2020, 2, 19),
        "end": datetime(2020, 3, 23),
        "symbols": ["SPY", "DIA", "QQQ", "XLE", "XLK", "XLV", "AAPL", "MSFT", "AMZN", "BA", "DIS", "AAL"]
    },
    "dotcom_2000": {
        "name": "Dot-com Bubble",
        "start": datetime(2000, 3, 10),
        "end": datetime(2002, 10, 9),
        "symbols": ["SPY", "QQQ", "XLK", "CSCO", "INTC", "MSFT", "ORCL", "AMZN", "EBAY"]
    },
    "black_monday_1987": {
        "name": "Black Monday 1987",
        "start": datetime(1987, 10, 15),
        "end": datetime(1987, 10, 22),
        "symbols": ["SPY", "DIA"]
    }
}


def collect_all_crises(db: Session) -> Dict:
    """Collecte les données pour toutes les crises historiques"""
    collector = get_data_collector(db)
    results = {}

    for crisis_id, crisis_info in CRISIS_PERIODS.items():
        logger.info(f"🔥 Collecte crise: {crisis_info['name']}")
        results[crisis_id] = collector.collect_crisis_data(
            crisis_name=crisis_info['name'],
            symbols=crisis_info['symbols'],
            start_date=crisis_info['start'],
            end_date=crisis_info['end']
        )

    return results
