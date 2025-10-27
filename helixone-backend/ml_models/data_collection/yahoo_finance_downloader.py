"""
Yahoo Finance Downloader - Téléchargement de données historiques OHLCV

Fonctionnalités :
- Télécharger 10+ ans de données pour n'importe quelle liste de tickers
- Cache local SQLite pour éviter re-téléchargement
- Mise à jour incrémentale (seulement nouvelles données)
- Support multi-threading pour téléchargement rapide
- Validation des données (détection gaps, valeurs manquantes)

Utilisation :
    downloader = YahooFinanceDownloader()
    data = downloader.download_historical_data(['AAPL', 'MSFT', 'GOOGL'],
                                                start_date='2014-01-01')
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sqlite3
import json

logger = logging.getLogger(__name__)


class YahooFinanceDownloader:
    """
    Téléchargeur de données Yahoo Finance avec cache intelligent
    """

    def __init__(self, cache_path: str = 'data/yahoo_finance_cache.db'):
        """
        Args:
            cache_path: Chemin vers la base de données cache SQLite
        """
        self.cache_path = cache_path
        self._init_cache()

        # Top 500 US stocks par market cap (vous pouvez personnaliser)
        self.default_tickers = self._load_sp500_tickers()

        logger.info(f"YahooFinanceDownloader initialisé avec cache: {cache_path}")

    def _init_cache(self):
        """Initialise la base de données cache"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                PRIMARY KEY (ticker, date)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ticker_date
            ON stock_data(ticker, date)
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_metadata (
                ticker TEXT PRIMARY KEY,
                last_updated TEXT,
                start_date TEXT,
                end_date TEXT,
                total_records INTEGER
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Cache database initialized")

    def _load_sp500_tickers(self) -> List[str]:
        """
        Charge la liste des tickers S&P 500 depuis Wikipedia

        Returns:
            Liste de tickers
        """
        try:
            # Télécharger depuis Wikipedia (gratuit)
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].tolist()

            # Nettoyer (remplacer . par -)
            tickers = [ticker.replace('.', '-') for ticker in tickers]

            logger.info(f"Chargé {len(tickers)} tickers S&P 500")
            return tickers

        except Exception as e:
            logger.warning(f"Erreur chargement S&P 500: {e}. Utilisation liste par défaut")
            # Liste par défaut des top stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'UNH', 'JNJ', 'XOM', 'V', 'PG', 'JPM', 'MA', 'HD', 'CVX', 'MRK',
                'ABBV', 'PEP', 'KO', 'AVGO', 'COST', 'LLY', 'WMT', 'MCD', 'TMO',
                'ACN', 'CSCO', 'ABT', 'DHR', 'NKE', 'VZ', 'ADBE', 'CRM', 'NFLX',
                'CMCSA', 'PFE', 'DIS', 'INTC', 'AMD', 'TXN', 'QCOM', 'UPS', 'NEE',
                'PM', 'UNP', 'RTX', 'HON', 'ORCL', 'IBM', 'BA', 'CAT', 'GE', 'AMGN'
            ]

    def download_historical_data(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = None,
        end_date: str = None,
        use_cache: bool = True,
        max_workers: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Télécharge des données historiques pour une liste de tickers

        Args:
            tickers: Liste de tickers (None = S&P 500)
            start_date: Date de début (YYYY-MM-DD), défaut = 10 ans
            end_date: Date de fin (YYYY-MM-DD), défaut = aujourd'hui
            use_cache: Utiliser le cache local
            max_workers: Nombre de threads pour téléchargement parallèle

        Returns:
            Dictionnaire {ticker: DataFrame} avec colonnes OHLCV
        """
        if tickers is None:
            tickers = self.default_tickers

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Téléchargement {len(tickers)} tickers de {start_date} à {end_date}")

        # Téléchargement parallèle avec barre de progression
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumettre tous les jobs
            futures = {
                executor.submit(
                    self._download_single_ticker,
                    ticker,
                    start_date,
                    end_date,
                    use_cache
                ): ticker
                for ticker in tickers
            }

            # Progress bar
            with tqdm(total=len(tickers), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        df = future.result()
                        if df is not None and len(df) > 0:
                            results[ticker] = df
                    except Exception as e:
                        logger.error(f"Erreur téléchargement {ticker}: {e}")

                    pbar.update(1)

        logger.info(f"✅ Téléchargement terminé: {len(results)}/{len(tickers)} réussis")
        return results

    def _download_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool
    ) -> Optional[pd.DataFrame]:
        """
        Télécharge les données pour un seul ticker

        Args:
            ticker: Ticker à télécharger
            start_date: Date début
            end_date: Date fin
            use_cache: Utiliser cache

        Returns:
            DataFrame avec données OHLCV ou None si erreur
        """
        try:
            # 🆕 VÉRIFIER D'ABORD SI CSV LOCAL EXISTE
            from pathlib import Path
            csv_path = Path(__file__).parent / 'data' / f'{ticker}_historical.csv'

            if csv_path.exists():
                logger.info(f"📂 Chargement depuis CSV local: {csv_path.name}")
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

                # Normaliser les noms de colonnes
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]

                # Filtrer par dates demandées
                df = df[(df.index >= start_date) & (df.index <= end_date)]

                if not df.empty:
                    logger.info(f"   ✅ {len(df)} jours chargés depuis CSV")
                    return df

            # Vérifier cache
            if use_cache:
                cached_data = self._get_from_cache(ticker, start_date, end_date)
                if cached_data is not None:
                    return cached_data

            # Télécharger depuis Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                logger.warning(f"{ticker}: Aucune donnée disponible")
                return None

            # Nettoyer les colonnes
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })

            # Garder seulement les colonnes importantes
            df = df[['open', 'high', 'low', 'close', 'volume', 'adj_close']]

            # Validation : supprimer les jours avec prix = 0 ou volume = 0
            df = df[(df['close'] > 0) & (df['volume'] > 0)]

            # Détecter et remplir les gaps (interpolation linéaire)
            df = df.interpolate(method='linear', limit_direction='both')

            # Sauvegarder dans cache
            if use_cache:
                self._save_to_cache(ticker, df)

            return df

        except Exception as e:
            logger.error(f"Erreur téléchargement {ticker}: {e}")
            return None

    def _get_from_cache(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les données depuis le cache

        Args:
            ticker: Ticker
            start_date: Date début
            end_date: Date fin

        Returns:
            DataFrame ou None si pas dans cache
        """
        try:
            conn = sqlite3.connect(self.cache_path)

            # Vérifier métadonnées
            metadata = pd.read_sql_query(
                "SELECT * FROM download_metadata WHERE ticker = ?",
                conn,
                params=(ticker,)
            )

            if metadata.empty:
                conn.close()
                return None

            # Vérifier si cache couvre la période demandée
            cached_start = metadata.iloc[0]['start_date']
            cached_end = metadata.iloc[0]['end_date']

            if cached_start <= start_date and cached_end >= end_date:
                # Cache complet, charger les données
                df = pd.read_sql_query(
                    """
                    SELECT * FROM stock_data
                    WHERE ticker = ?
                    AND date BETWEEN ? AND ?
                    ORDER BY date
                    """,
                    conn,
                    params=(ticker, start_date, end_date),
                    index_col='date',
                    parse_dates=['date']
                )

                conn.close()

                if not df.empty:
                    df = df.drop(columns=['ticker'])
                    logger.debug(f"{ticker}: Chargé depuis cache ({len(df)} jours)")
                    return df

            conn.close()
            return None

        except Exception as e:
            logger.debug(f"Cache miss pour {ticker}: {e}")
            return None

    def _save_to_cache(self, ticker: str, df: pd.DataFrame):
        """
        Sauvegarde les données dans le cache

        Args:
            ticker: Ticker
            df: DataFrame à sauvegarder
        """
        try:
            conn = sqlite3.connect(self.cache_path)

            # Préparer les données
            df_to_save = df.copy()
            df_to_save['ticker'] = ticker
            df_to_save['date'] = df_to_save.index.strftime('%Y-%m-%d')
            df_to_save = df_to_save.reset_index(drop=True)

            # Supprimer anciennes données
            conn.execute("DELETE FROM stock_data WHERE ticker = ?", (ticker,))

            # Insérer nouvelles données
            df_to_save.to_sql('stock_data', conn, if_exists='append', index=False)

            # Mettre à jour métadonnées
            metadata = {
                'ticker': ticker,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'start_date': df.index.min().strftime('%Y-%m-%d'),
                'end_date': df.index.max().strftime('%Y-%m-%d'),
                'total_records': len(df)
            }

            conn.execute("DELETE FROM download_metadata WHERE ticker = ?", (ticker,))
            pd.DataFrame([metadata]).to_sql('download_metadata', conn, if_exists='append', index=False)

            conn.commit()
            conn.close()

            logger.debug(f"{ticker}: Sauvegardé dans cache ({len(df)} jours)")

        except Exception as e:
            logger.error(f"Erreur sauvegarde cache {ticker}: {e}")

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Récupère le prix actuel (latest close) pour un ticker

        Args:
            ticker: Ticker

        Returns:
            Prix actuel ou None
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if not hist.empty:
                return hist['Close'].iloc[-1]
            return None
        except:
            return None

    def update_cache(self, tickers: Optional[List[str]] = None):
        """
        Met à jour le cache avec les dernières données
        (télécharge seulement les nouvelles données depuis dernière mise à jour)

        Args:
            tickers: Liste de tickers à mettre à jour (None = tous)
        """
        if tickers is None:
            # Récupérer tous les tickers en cache
            conn = sqlite3.connect(self.cache_path)
            metadata = pd.read_sql_query("SELECT ticker, end_date FROM download_metadata", conn)
            conn.close()

            if metadata.empty:
                logger.info("Cache vide, téléchargement complet nécessaire")
                return

            tickers = metadata['ticker'].tolist()

        logger.info(f"Mise à jour cache pour {len(tickers)} tickers")

        for ticker in tqdm(tickers, desc="Updating cache"):
            try:
                # Récupérer dernière date en cache
                conn = sqlite3.connect(self.cache_path)
                result = conn.execute(
                    "SELECT end_date FROM download_metadata WHERE ticker = ?",
                    (ticker,)
                ).fetchone()
                conn.close()

                if result is None:
                    continue

                last_date = result[0]
                today = datetime.now().strftime('%Y-%m-%d')

                # Télécharger seulement nouvelles données
                if last_date < today:
                    new_data = self._download_single_ticker(
                        ticker,
                        start_date=last_date,
                        end_date=today,
                        use_cache=False
                    )

                    if new_data is not None and len(new_data) > 0:
                        self._save_to_cache(ticker, new_data)
                        logger.debug(f"{ticker}: +{len(new_data)} nouveaux jours")

            except Exception as e:
                logger.error(f"Erreur mise à jour {ticker}: {e}")


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_downloader() -> YahooFinanceDownloader:
    """Retourne une instance singleton du downloader"""
    global _downloader_instance
    if '_downloader_instance' not in globals():
        _downloader_instance = YahooFinanceDownloader()
    return _downloader_instance


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Créer downloader
    downloader = YahooFinanceDownloader()

    # Télécharger top 50 stocks (10 ans)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
               'JPM', 'JNJ', 'V', 'WMT', 'PG', 'UNH', 'MA', 'HD']

    print("📥 Téléchargement données historiques (10 ans)...")
    data = downloader.download_historical_data(tickers, start_date='2014-01-01')

    print(f"\n✅ Téléchargé {len(data)} tickers")

    # Afficher exemple
    if 'AAPL' in data:
        print("\n📊 Exemple AAPL (derniers 5 jours):")
        print(data['AAPL'].tail())
        print(f"\nTotal jours: {len(data['AAPL'])}")

    # Tester mise à jour
    print("\n🔄 Mise à jour cache...")
    downloader.update_cache(tickers)

    print("\n✅ Terminé!")
