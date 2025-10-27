"""
Data Cache - Interface unifiée pour Yahoo Finance + FRED + Sentiment

Cette classe combine toutes les sources de données et fournit un dataset
ML-ready avec toutes les features nécessaires.

Utilisation:
    cache = DataCache()
    dataset = cache.get_ml_dataset(
        tickers=['AAPL', 'MSFT'],
        start_date='2014-01-01',
        include_sentiment=True
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from pathlib import Path

from .yahoo_finance_downloader import YahooFinanceDownloader
from .fred_macro_downloader import FREDMacroDownloader

logger = logging.getLogger(__name__)


class DataCache:
    """
    Cache centralisé combinant toutes les sources de données
    """

    def __init__(
        self,
        yahoo_cache_path: str = 'data/yahoo_finance_cache.db',
        fred_cache_path: str = 'data/fred_macro_cache.db',
        fred_api_key: Optional[str] = None
    ):
        """
        Args:
            yahoo_cache_path: Chemin cache Yahoo Finance
            fred_cache_path: Chemin cache FRED
            fred_api_key: Clé API FRED (optionnelle)
        """
        # Créer le dossier data s'il n'existe pas
        Path('data').mkdir(exist_ok=True)

        self.yahoo_downloader = YahooFinanceDownloader(cache_path=yahoo_cache_path)
        self.fred_downloader = FREDMacroDownloader(
            api_key=fred_api_key,
            cache_path=fred_cache_path
        )

        logger.info("DataCache initialisé")

    def get_ml_dataset(
        self,
        tickers: List[str],
        start_date: str = None,
        end_date: str = None,
        include_macro: bool = True,
        include_sentiment: bool = False,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Crée un dataset ML-ready pour une liste de tickers

        Args:
            tickers: Liste de tickers
            start_date: Date début
            end_date: Date fin
            include_macro: Inclure données macro FRED
            include_sentiment: Inclure sentiment (nécessite services actifs)
            use_cache: Utiliser cache

        Returns:
            Dict {ticker: DataFrame avec toutes les features}
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"📦 Création dataset ML pour {len(tickers)} tickers")

        # 1. Télécharger données prix
        logger.info("📊 Téléchargement données prix...")
        price_data = self.yahoo_downloader.download_historical_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )

        # 2. Télécharger données macro
        macro_data = None
        if include_macro:
            logger.info("🌍 Téléchargement données macro...")
            macro_data = self.fred_downloader.download_all_indicators(
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache
            )

        # 3. Combiner tout
        logger.info("🔗 Fusion des données...")
        ml_dataset = {}

        for ticker, df_price in price_data.items():
            # Commencer avec prix
            df = df_price.copy()

            # Ajouter données macro (merge sur date)
            if macro_data is not None and not macro_data.empty:
                # Reindexer macro sur les mêmes dates que prix
                macro_aligned = macro_data.reindex(df.index, method='ffill')
                df = pd.concat([df, macro_aligned], axis=1)

            # TODO: Ajouter sentiment si demandé
            # if include_sentiment:
            #     sentiment_data = self._get_sentiment_features(ticker, df.index)
            #     df = pd.concat([df, sentiment_data], axis=1)

            ml_dataset[ticker] = df

        logger.info(f"✅ Dataset créé: {len(ml_dataset)} tickers")
        if ml_dataset:
            first_ticker = list(ml_dataset.keys())[0]
            logger.info(f"   Exemple {first_ticker}: {len(ml_dataset[first_ticker])} jours, {len(ml_dataset[first_ticker].columns)} features")

        return ml_dataset

    def update_all_caches(self, tickers: Optional[List[str]] = None):
        """
        Met à jour tous les caches (Yahoo + FRED)

        Args:
            tickers: Liste de tickers à mettre à jour (None = tous)
        """
        logger.info("🔄 Mise à jour des caches...")

        # Update Yahoo
        logger.info("📊 Mise à jour Yahoo Finance...")
        self.yahoo_downloader.update_cache(tickers)

        # Update FRED (pas de tickers, ce sont des séries macro)
        if self.fred_downloader.fred:
            logger.info("🌍 FRED déjà en cache (pas de mise à jour nécessaire)")

        logger.info("✅ Caches mis à jour")

    def get_latest_data(self, ticker: str) -> Optional[pd.Series]:
        """
        Récupère les dernières données pour un ticker

        Args:
            ticker: Ticker

        Returns:
            Series avec dernières données ou None
        """
        try:
            # Télécharger dernier jour
            data = self.yahoo_downloader.download_historical_data(
                tickers=[ticker],
                start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                use_cache=False
            )

            if ticker in data and not data[ticker].empty:
                return data[ticker].iloc[-1]

            return None

        except Exception as e:
            logger.error(f"Erreur récupération dernières données {ticker}: {e}")
            return None

    def get_cache_info(self) -> Dict:
        """
        Retourne des informations sur l'état des caches

        Returns:
            Dict avec statistiques
        """
        info = {
            'yahoo': {},
            'fred': {}
        }

        try:
            import sqlite3

            # Yahoo Finance cache
            conn = sqlite3.connect(self.yahoo_downloader.cache_path)
            cursor = conn.cursor()

            # Nombre de tickers
            cursor.execute("SELECT COUNT(DISTINCT ticker) FROM download_metadata")
            info['yahoo']['num_tickers'] = cursor.fetchone()[0]

            # Total records
            cursor.execute("SELECT SUM(total_records) FROM download_metadata")
            info['yahoo']['total_records'] = cursor.fetchone()[0] or 0

            # Période couverte
            cursor.execute("SELECT MIN(start_date), MAX(end_date) FROM download_metadata")
            result = cursor.fetchone()
            info['yahoo']['period'] = f"{result[0]} → {result[1]}"

            conn.close()

            # FRED cache
            conn = sqlite3.connect(self.fred_downloader.cache_path)
            cursor = conn.cursor()

            # Nombre de séries
            cursor.execute("SELECT COUNT(DISTINCT series_id) FROM series_metadata")
            info['fred']['num_series'] = cursor.fetchone()[0]

            # Total records
            cursor.execute("SELECT SUM(total_records) FROM series_metadata")
            info['fred']['total_records'] = cursor.fetchone()[0] or 0

            conn.close()

        except Exception as e:
            logger.error(f"Erreur récupération info cache: {e}")

        return info


# ============================================================================
# SINGLETON
# ============================================================================

_data_cache_instance = None

def get_data_cache(fred_api_key: Optional[str] = None) -> DataCache:
    """Retourne une instance singleton du cache"""
    global _data_cache_instance
    if _data_cache_instance is None:
        _data_cache_instance = DataCache(fred_api_key=fred_api_key)
    return _data_cache_instance


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("DATA CACHE - Test Complet")
    print("=" * 80)

    # Créer cache
    cache = DataCache()

    # Télécharger dataset pour 5 stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    print(f"\n📥 Téléchargement dataset pour {len(tickers)} stocks (5 ans)...")
    dataset = cache.get_ml_dataset(
        tickers=tickers,
        start_date='2019-01-01',
        include_macro=True,
        include_sentiment=False
    )

    print(f"\n✅ Dataset créé pour {len(dataset)} tickers")

    # Afficher exemple
    for ticker, df in dataset.items():
        print(f"\n📊 {ticker}:")
        print(f"   Période: {df.index.min()} → {df.index.max()}")
        print(f"   Jours: {len(df)}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Colonnes: {', '.join(df.columns[:10])}...")
        print(f"\n   Aperçu (5 derniers jours):")
        print(df[['close', 'volume', 'DFF', 'UNRATE']].tail())
        break  # Afficher seulement le premier

    # Info cache
    print("\n💾 Info caches:")
    info = cache.get_cache_info()
    print(f"   Yahoo: {info['yahoo']['num_tickers']} tickers, {info['yahoo']['total_records']} records")
    print(f"   Période: {info['yahoo']['period']}")
    print(f"   FRED: {info['fred']['num_series']} séries, {info['fred']['total_records']} records")

    print("\n✅ Test terminé!")
