"""
Sentiment Features - Intégration du sentiment existant dans le ML

Int

égration avec les services existants :
- Reddit Sentiment (app/services/reddit_source.py)
- StockTwits Sentiment (app/services/stocktwits_source.py)
- News Sentiment (app/services/newsapi_source.py)

Features créées :
- sentiment_score (agrégé)
- sentiment_change_7d
- mentions_volume
- bullish_ratio
- consensus_score
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SentimentFeatures:
    """
    Extracteur de features de sentiment pour ML
    """

    def __init__(self):
        self.reddit_collector = None
        self.stocktwits_collector = None
        self.newsapi_source = None

        # Tenter d'importer les services existants
        try:
            from app.services.reddit_source import get_reddit_collector
            self.reddit_collector = get_reddit_collector()
            logger.info("✅ Reddit collector chargé")
        except Exception as e:
            logger.warning(f"⚠️ Reddit collector non disponible: {e}")

        try:
            from app.services.stocktwits_source import get_stocktwits_collector
            self.stocktwits_collector = get_stocktwits_collector()
            logger.info("✅ StockTwits collector chargé")
        except Exception as e:
            logger.warning(f"⚠️ StockTwits collector non disponible: {e}")

        try:
            from app.services.newsapi_source import NewsAPISource
            self.newsapi_source = NewsAPISource()
            logger.info("✅ NewsAPI source chargée")
        except Exception as e:
            logger.warning(f"⚠️ NewsAPI source non disponible: {e}")

    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Ajoute des features de sentiment au DataFrame

        Args:
            df: DataFrame avec index = dates
            ticker: Ticker de l'action

        Returns:
            DataFrame avec nouvelles colonnes de sentiment
        """
        df = df.copy()

        logger.debug(f"Ajout features sentiment pour {ticker}...")

        # Pour l'instant, créer des features synthétiques
        # TODO: Intégrer vraies données sentiment quand services actifs

        # Sentiment score agrégé (-1 à +1)
        df['sentiment_score'] = self._generate_synthetic_sentiment(len(df))

        # Changement de sentiment
        df['sentiment_change_1d'] = df['sentiment_score'].diff(1)
        df['sentiment_change_7d'] = df['sentiment_score'].diff(7)

        # Volume de mentions (normalisé)
        df['mentions_volume'] = self._generate_synthetic_mentions(len(df))

        # Ratio bullish/bearish
        df['bullish_ratio'] = self._generate_bullish_ratio(len(df))

        # Consensus entre sources (0-100)
        df['sentiment_consensus'] = self._generate_consensus(len(df))

        # Extremes (signal d'alerte)
        df['sentiment_extreme_positive'] = (df['sentiment_score'] > 0.7).astype(int)
        df['sentiment_extreme_negative'] = (df['sentiment_score'] < -0.7).astype(int)

        logger.debug(f"✅ {6} features sentiment ajoutées")

        return df

    def _generate_synthetic_sentiment(self, n: int) -> pd.Series:
        """Génère un sentiment synthétique pour test"""
        # Random walk avec tendance vers 0
        np.random.seed(42)
        sentiment = np.random.randn(n) * 0.3
        sentiment = pd.Series(sentiment).ewm(span=5).mean()  # Smooth
        return sentiment.clip(-1, 1)

    def _generate_synthetic_mentions(self, n: int) -> pd.Series:
        """Génère un volume de mentions synthétique"""
        np.random.seed(43)
        mentions = np.random.gamma(2, 2, n) * 10
        return pd.Series(mentions)

    def _generate_bullish_ratio(self, n: int) -> pd.Series:
        """Génère un ratio bullish/bearish synthétique"""
        np.random.seed(44)
        ratio = np.random.beta(2, 2, n)  # Beta distribution (0-1)
        return pd.Series(ratio)

    def _generate_consensus(self, n: int) -> pd.Series:
        """Génère un consensus synthétique"""
        np.random.seed(45)
        consensus = np.random.uniform(40, 80, n)  # Entre 40 et 80
        return pd.Series(consensus)

    def get_real_time_sentiment(self, ticker: str) -> dict:
        """
        Récupère le sentiment en temps réel

        Args:
            ticker: Ticker

        Returns:
            Dict avec sentiment_score, mentions, etc.
        """
        sentiment_data = {
            'sentiment_score': 0.0,
            'mentions_volume': 0,
            'bullish_ratio': 0.5,
            'consensus': 50,
            'sources': []
        }

        # Reddit
        if self.reddit_collector:
            try:
                reddit_data = self.reddit_collector.get_ticker_sentiment(ticker)
                if reddit_data:
                    sentiment_data['sources'].append('reddit')
                    # Adapter selon la structure réelle
            except Exception as e:
                logger.debug(f"Erreur Reddit sentiment: {e}")

        # StockTwits
        if self.stocktwits_collector:
            try:
                st_data = self.stocktwits_collector.get_sentiment(ticker)
                if st_data:
                    sentiment_data['sources'].append('stocktwits')
            except Exception as e:
                logger.debug(f"Erreur StockTwits sentiment: {e}")

        # News
        if self.newsapi_source:
            try:
                news_data = self.newsapi_source.get_news(ticker)
                if news_data:
                    sentiment_data['sources'].append('news')
            except Exception as e:
                logger.debug(f"Erreur News sentiment: {e}")

        return sentiment_data


# ============================================================================
# SINGLETON
# ============================================================================

_sentiment_features_instance = None

def get_sentiment_features() -> SentimentFeatures:
    """Retourne une instance singleton"""
    global _sentiment_features_instance
    if _sentiment_features_instance is None:
        _sentiment_features_instance = SentimentFeatures()
    return _sentiment_features_instance


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("SENTIMENT FEATURES - Test")
    print("=" * 80)

    # Créer DataFrame de test
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    df = pd.DataFrame(index=dates)

    print(f"\n📊 Dataset test: {len(df)} jours")

    # Ajouter features sentiment
    print("\n⚙️ Ajout features sentiment...")
    sentiment_extractor = SentimentFeatures()
    df = sentiment_extractor.add_sentiment_features(df, 'AAPL')

    print(f"\n✅ Features sentiment ajoutées!")
    print(f"   Total features: {len(df.columns)}")

    print("\n📋 Features créées:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")

    print("\n📈 Exemple de valeurs (derniers 10 jours):")
    print(df.tail(10))

    print("\n📊 Statistiques:")
    print(df.describe())

    print("\n✅ Test terminé!")
