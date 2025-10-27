"""
FRED Macro Downloader - Données Macro-Économiques (100% Gratuit)

Source : Federal Reserve Economic Data (FRED) - St. Louis Fed
API : Gratuite avec clé (obtenir sur https://fred.stlouisfed.org/docs/api/api_key.html)

Indicateurs téléchargés :
- Fed Funds Rate (DFF) : Taux directeur de la Fed
- 10Y Treasury Yield (DGS10) : Rendement obligations 10 ans
- 2Y Treasury Yield (DGS2) : Rendement obligations 2 ans
- Yield Curve : 10Y - 2Y (indicateur de récession)
- Inflation (CPIAUCSL) : Consumer Price Index
- Unemployment Rate (UNRATE) : Taux de chômage
- GDP Growth (GDP) : Croissance du PIB
- Industrial Production (INDPRO) : Production industrielle
- Retail Sales (RSXFS) : Ventes au détail

Utilisation :
    downloader = FREDMacroDownloader(api_key='VOTRE_CLE_FRED')
    data = downloader.download_all_indicators(start_date='2014-01-01')
"""

from fredapi import Fred
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class FREDMacroDownloader:
    """
    Téléchargeur de données macro-économiques depuis FRED
    """

    # Mapping des séries FRED (code -> description)
    FRED_SERIES = {
        # Taux d'intérêt
        'DFF': 'Fed Funds Rate',
        'DGS10': '10-Year Treasury Yield',
        'DGS2': '2-Year Treasury Yield',
        'DGS5': '5-Year Treasury Yield',
        'T10Y2Y': 'Yield Curve (10Y-2Y)',

        # Inflation
        'CPIAUCSL': 'Consumer Price Index',
        'PCEPI': 'PCE Price Index',
        'CPILFESL': 'Core CPI (ex food/energy)',

        # Emploi
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'ICSA': 'Initial Claims',

        # Économie réelle
        'GDP': 'Gross Domestic Product',
        'GDPC1': 'Real GDP',
        'INDPRO': 'Industrial Production',
        'RSXFS': 'Retail Sales',
        'HOUST': 'Housing Starts',

        # Confiance
        'UMCSENT': 'Consumer Sentiment (Michigan)',
        'CSCICP03USM665S': 'Consumer Confidence (OECD)',

        # Marché
        'VIXCLS': 'VIX (Volatility Index)',
        'DEXUSEU': 'Dollar/Euro Exchange Rate',

        # Crédit
        'MORTGAGE30US': '30-Year Mortgage Rate',
        'BAMLH0A0HYM2': 'High Yield Corporate Bond Spread'
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_path: str = 'data/fred_macro_cache.db'
    ):
        """
        Args:
            api_key: Clé API FRED (gratuite)
            cache_path: Chemin vers cache SQLite
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.cache_path = cache_path
        self.fred = None

        if self.api_key:
            try:
                self.fred = Fred(api_key=self.api_key)
                logger.info("✅ FRED API initialisée")
            except Exception as e:
                logger.warning(f"⚠️ Erreur initialisation FRED API: {e}")
                logger.warning("Certaines features macro ne seront pas disponibles")
        else:
            logger.warning("⚠️ Pas de clé API FRED. Utilisez .env ou passez api_key=")
            logger.info("Obtenez une clé gratuite : https://fred.stlouisfed.org/docs/api/api_key.html")

        self._init_cache()

    def _get_api_key_from_env(self) -> Optional[str]:
        """Tente de récupérer la clé API depuis .env"""
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            return os.getenv('FRED_API_KEY')
        except:
            return None

    def _init_cache(self):
        """Initialise la base de données cache"""
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS macro_data (
                series_id TEXT NOT NULL,
                date TEXT NOT NULL,
                value REAL,
                PRIMARY KEY (series_id, date)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_series_date
            ON macro_data(series_id, date)
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS series_metadata (
                series_id TEXT PRIMARY KEY,
                description TEXT,
                last_updated TEXT,
                start_date TEXT,
                end_date TEXT,
                total_records INTEGER
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Cache FRED initialisé")

    def download_all_indicators(
        self,
        start_date: str = None,
        end_date: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Télécharge TOUS les indicateurs macro

        Args:
            start_date: Date début (YYYY-MM-DD)
            end_date: Date fin (YYYY-MM-DD)
            use_cache: Utiliser cache local

        Returns:
            DataFrame avec toutes les séries (colonnes = séries, index = dates)
        """
        if not self.fred:
            logger.error("❌ FRED API non initialisée. Impossible de télécharger les données")
            return pd.DataFrame()

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"📥 Téléchargement {len(self.FRED_SERIES)} indicateurs FRED...")

        all_data = {}

        for series_id, description in self.FRED_SERIES.items():
            try:
                data = self._download_single_series(
                    series_id,
                    start_date,
                    end_date,
                    use_cache
                )

                if data is not None and len(data) > 0:
                    all_data[series_id] = data
                    logger.debug(f"✅ {series_id}: {len(data)} points")
                else:
                    logger.warning(f"⚠️ {series_id}: Aucune donnée")

            except Exception as e:
                logger.error(f"❌ Erreur {series_id}: {e}")

        if not all_data:
            logger.error("❌ Aucune donnée macro téléchargée")
            return pd.DataFrame()

        # Combiner toutes les séries dans un seul DataFrame
        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Forward fill les valeurs manquantes (certaines séries sont mensuelles)
        df = df.ffill()

        # Calculer des indicateurs dérivés
        df = self._calculate_derived_indicators(df)

        logger.info(f"✅ {len(df.columns)} indicateurs téléchargés ({len(df)} jours)")
        return df

    def _download_single_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
        use_cache: bool
    ) -> Optional[pd.Series]:
        """
        Télécharge une seule série FRED

        Args:
            series_id: ID de la série FRED
            start_date: Date début
            end_date: Date fin
            use_cache: Utiliser cache

        Returns:
            Series avec les données ou None
        """
        try:
            # Vérifier cache
            if use_cache:
                cached_data = self._get_from_cache(series_id, start_date, end_date)
                if cached_data is not None:
                    return cached_data

            # Télécharger depuis FRED
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )

            if data.empty:
                return None

            # Sauvegarder dans cache
            if use_cache:
                self._save_to_cache(series_id, data)

            return data

        except Exception as e:
            logger.debug(f"Erreur téléchargement {series_id}: {e}")
            return None

    def _get_from_cache(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.Series]:
        """Récupère une série depuis le cache"""
        try:
            conn = sqlite3.connect(self.cache_path)

            # Vérifier métadonnées
            metadata = pd.read_sql_query(
                "SELECT * FROM series_metadata WHERE series_id = ?",
                conn,
                params=(series_id,)
            )

            if metadata.empty:
                conn.close()
                return None

            # Vérifier couverture
            cached_start = metadata.iloc[0]['start_date']
            cached_end = metadata.iloc[0]['end_date']

            if cached_start <= start_date and cached_end >= end_date:
                # Charger données
                df = pd.read_sql_query(
                    """
                    SELECT date, value FROM macro_data
                    WHERE series_id = ?
                    AND date BETWEEN ? AND ?
                    ORDER BY date
                    """,
                    conn,
                    params=(series_id, start_date, end_date),
                    index_col='date',
                    parse_dates=['date']
                )

                conn.close()

                if not df.empty:
                    return df['value']

            conn.close()
            return None

        except Exception as e:
            logger.debug(f"Cache miss {series_id}: {e}")
            return None

    def _save_to_cache(self, series_id: str, data: pd.Series):
        """Sauvegarde une série dans le cache"""
        try:
            conn = sqlite3.connect(self.cache_path)

            # Préparer données
            df = pd.DataFrame({
                'series_id': series_id,
                'date': data.index.strftime('%Y-%m-%d'),
                'value': data.values
            })

            # Supprimer anciennes données
            conn.execute("DELETE FROM macro_data WHERE series_id = ?", (series_id,))

            # Insérer nouvelles données
            df.to_sql('macro_data', conn, if_exists='append', index=False)

            # Métadonnées
            metadata = {
                'series_id': series_id,
                'description': self.FRED_SERIES.get(series_id, 'Unknown'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'start_date': data.index.min().strftime('%Y-%m-%d'),
                'end_date': data.index.max().strftime('%Y-%m-%d'),
                'total_records': len(data)
            }

            conn.execute("DELETE FROM series_metadata WHERE series_id = ?", (series_id,))
            pd.DataFrame([metadata]).to_sql('series_metadata', conn, if_exists='append', index=False)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Erreur sauvegarde cache {series_id}: {e}")

    def _calculate_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des indicateurs dérivés

        Args:
            df: DataFrame avec séries FRED

        Returns:
            DataFrame avec indicateurs additionnels
        """
        try:
            # Yield Curve (si pas déjà dans FRED)
            if 'DGS10' in df.columns and 'DGS2' in df.columns and 'T10Y2Y' not in df.columns:
                df['yield_curve'] = df['DGS10'] - df['DGS2']

            # Taux réel (taux nominal - inflation)
            if 'DGS10' in df.columns and 'CPIAUCSL' in df.columns:
                inflation_yoy = df['CPIAUCSL'].pct_change(periods=12) * 100  # Inflation annuelle
                df['real_10y_yield'] = df['DGS10'] - inflation_yoy

            # Changements (utile pour ML)
            if 'DFF' in df.columns:
                df['fed_funds_change'] = df['DFF'].diff()

            if 'UNRATE' in df.columns:
                df['unemployment_change'] = df['UNRATE'].diff()

            # Inflation YoY
            if 'CPIAUCSL' in df.columns:
                df['inflation_yoy'] = df['CPIAUCSL'].pct_change(periods=12) * 100

            # GDP Growth Rate (annualisé)
            if 'GDPC1' in df.columns:
                df['gdp_growth_yoy'] = df['GDPC1'].pct_change(periods=4) * 100  # Quarterly

        except Exception as e:
            logger.warning(f"Erreur calcul indicateurs dérivés: {e}")

        return df

    def get_latest_values(self) -> Dict[str, float]:
        """
        Récupère les valeurs les plus récentes de tous les indicateurs

        Returns:
            Dictionnaire {series_id: latest_value}
        """
        if not self.fred:
            return {}

        latest = {}

        for series_id in self.FRED_SERIES.keys():
            try:
                data = self.fred.get_series(series_id)
                if not data.empty:
                    latest[series_id] = data.iloc[-1]
            except:
                pass

        return latest


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_fred_downloader(api_key: Optional[str] = None) -> FREDMacroDownloader:
    """Retourne une instance singleton du downloader"""
    global _fred_downloader_instance
    if '_fred_downloader_instance' not in globals():
        _fred_downloader_instance = FREDMacroDownloader(api_key=api_key)
    return _fred_downloader_instance


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("FRED MACRO DOWNLOADER - Test")
    print("=" * 80)

    # IMPORTANT : Obtenir une clé API gratuite sur :
    # https://fred.stlouisfed.org/docs/api/api_key.html

    # Option 1 : Passer la clé directement
    # downloader = FREDMacroDownloader(api_key='VOTRE_CLE_ICI')

    # Option 2 : Utiliser .env (ajouter FRED_API_KEY=votre_cle)
    downloader = FREDMacroDownloader()

    if downloader.fred:
        print("\n📥 Téléchargement données macro (10 ans)...")
        data = downloader.download_all_indicators(start_date='2014-01-01')

        print(f"\n✅ Téléchargé {len(data.columns)} indicateurs")
        print(f"📊 Période : {data.index.min()} → {data.index.max()}")
        print(f"📈 Total : {len(data)} jours de données")

        print("\n📋 Indicateurs disponibles :")
        for col in data.columns:
            desc = downloader.FRED_SERIES.get(col, col)
            print(f"   - {col}: {desc}")

        print("\n💡 Dernières valeurs :")
        print(data.tail(1).T)

        print("\n📊 Exemples d'indicateurs dérivés :")
        derived = [col for col in data.columns if col not in downloader.FRED_SERIES]
        if derived:
            print(f"   {', '.join(derived)}")

    else:
        print("\n⚠️ Pas de clé API FRED")
        print("Obtenez une clé gratuite : https://fred.stlouisfed.org/docs/api/api_key.html")
        print("Puis ajoutez-la dans .env : FRED_API_KEY=votre_cle")

    print("\n✅ Test terminé!")
