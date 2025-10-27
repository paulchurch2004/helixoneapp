"""
Modèles de données pour le stockage des news et sentiment
Sources: Finnhub, autres sources de news
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Text, Index, Boolean
from datetime import datetime
import uuid

from app.core.database import Base


class NewsArticle(Base):
    """
    Table pour stocker les articles de news financières
    """
    __tablename__ = "news_articles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), index=True)                    # Symbole associé (peut être NULL pour news générales)
    source = Column(String(100), nullable=False)                # Source (Finnhub, Bloomberg, Reuters, etc.)
    external_id = Column(String(200))                           # ID externe de l'article

    # Contenu
    headline = Column(String(500), nullable=False)              # Titre
    summary = Column(Text)                                      # Résumé/description
    url = Column(String(1000))                                  # URL de l'article
    image_url = Column(String(1000))                            # URL de l'image

    # Catégorisation
    category = Column(String(100))                              # Catégorie (general, forex, crypto, merger, etc.)
    related_symbols = Column(Text)                              # Symboles liés (JSON array)

    # Dates
    published_at = Column(DateTime, nullable=False, index=True) # Date de publication
    collected_at = Column(DateTime, default=datetime.utcnow)    # Date de collecte

    # Sentiment (si disponible)
    sentiment_score = Column(Float)                             # Score de sentiment (-1 à 1)
    sentiment_label = Column(String(20))                        # positive, negative, neutral

    # Index composé
    __table_args__ = (
        Index('idx_symbol_published', 'symbol', 'published_at'),
        Index('idx_published_category', 'published_at', 'category'),
    )

    def __repr__(self):
        return f"<NewsArticle {self.symbol} {self.published_at}: {self.headline[:50]}>"


class SentimentAnalysis(Base):
    """
    Table pour stocker l'analyse de sentiment agrégée
    """
    __tablename__ = "sentiment_analysis"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)         # Date de l'analyse

    # Sentiment News
    news_sentiment_score = Column(Float)                        # Score sentiment news (-1 à 1)
    news_buzz = Column(Float)                                   # Buzz/popularité dans les news
    news_articles_count = Column(Integer)                       # Nombre d'articles analysés

    # Sentiment Réseaux Sociaux
    social_sentiment_score = Column(Float)                      # Score sentiment social media
    reddit_mentions = Column(Integer)                           # Mentions sur Reddit
    reddit_score = Column(Float)                                # Score Reddit
    twitter_mentions = Column(Integer)                          # Mentions sur Twitter
    twitter_score = Column(Float)                               # Score Twitter

    # Sentiment Analystes
    analyst_recommendation_score = Column(Float)                # Score recommandations (1-5)
    strong_buy_count = Column(Integer)                          # Nombre de Strong Buy
    buy_count = Column(Integer)                                 # Nombre de Buy
    hold_count = Column(Integer)                                # Nombre de Hold
    sell_count = Column(Integer)                                # Nombre de Sell
    strong_sell_count = Column(Integer)                         # Nombre de Strong Sell

    # Sentiment Global
    composite_sentiment_score = Column(Float)                   # Score composite tous sources
    sentiment_label = Column(String(20))                        # bullish, bearish, neutral

    # Métadonnées
    source = Column(String(50), default="finnhub")
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index composé
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'),
    )

    def __repr__(self):
        return f"<SentimentAnalysis {self.symbol} {self.date} score={self.composite_sentiment_score}>"


class MarketSentiment(Base):
    """
    Table pour stocker le sentiment global du marché
    """
    __tablename__ = "market_sentiment"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Date
    date = Column(DateTime, nullable=False, index=True, unique=True)

    # Indices de volatilité
    vix_value = Column(Float)                                   # VIX (indice de peur)
    vix_change = Column(Float)                                  # Variation VIX

    # Sentiment général
    market_sentiment_score = Column(Float)                      # Score de sentiment marché (-1 à 1)
    market_sentiment_label = Column(String(20))                 # fear, greed, neutral

    # Fear & Greed Index (si disponible)
    fear_greed_index = Column(Float)                            # 0-100
    fear_greed_label = Column(String(20))                       # extreme_fear, fear, neutral, greed, extreme_greed

    # Indicateurs techniques
    put_call_ratio = Column(Float)                              # Ratio Put/Call
    market_momentum = Column(Float)                             # Momentum du marché
    market_breadth = Column(Float)                              # Largeur du marché (% actions en hausse)

    # Volume et activité
    total_volume = Column(Float)                                # Volume total du marché
    advancing_issues = Column(Integer)                          # Nombre d'actions en hausse
    declining_issues = Column(Integer)                          # Nombre d'actions en baisse

    # Métadonnées
    source = Column(String(50), default="finnhub")
    collected_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<MarketSentiment {self.date} VIX={self.vix_value} sentiment={self.market_sentiment_label}>"


class AnalystRecommendation(Base):
    """
    Table pour stocker les recommandations d'analystes
    """
    __tablename__ = "analyst_recommendations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    period = Column(DateTime, nullable=False, index=True)       # Période (mois)

    # Recommandations
    strong_buy = Column(Integer, default=0)
    buy = Column(Integer, default=0)
    hold = Column(Integer, default=0)
    sell = Column(Integer, default=0)
    strong_sell = Column(Integer, default=0)

    # Totaux
    total_recommendations = Column(Integer)                     # Total
    average_rating = Column(Float)                              # Note moyenne (1-5)

    # Métadonnées
    source = Column(String(50), default="finnhub")
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index composé
    __table_args__ = (
        Index('idx_symbol_period', 'symbol', 'period'),
    )

    def __repr__(self):
        return f"<AnalystRecommendation {self.symbol} {self.period} avg={self.average_rating}>"


class PriceTarget(Base):
    """
    Table pour stocker les objectifs de prix des analystes
    """
    __tablename__ = "price_targets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)

    # Objectifs de prix
    target_high = Column(Float)                                 # Prix cible haut
    target_low = Column(Float)                                  # Prix cible bas
    target_mean = Column(Float)                                 # Prix cible moyen
    target_median = Column(Float)                               # Prix cible médian

    # Nombre d'analystes
    number_of_analysts = Column(Integer)                        # Nombre d'analystes

    # Prix actuel (pour comparaison)
    current_price = Column(Float)                               # Prix actuel

    # Potentiel
    upside_potential = Column(Float)                            # Potentiel de hausse (%)

    # Métadonnées
    source = Column(String(50), default="finnhub")
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index composé
    __table_args__ = (
        Index('idx_symbol_date_pt', 'symbol', 'date'),
    )

    def __repr__(self):
        return f"<PriceTarget {self.symbol} {self.date} target=${self.target_mean}>"


class EarningsEvent(Base):
    """
    Table pour stocker les événements de publications de résultats
    """
    __tablename__ = "earnings_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)         # Date de publication
    fiscal_quarter = Column(String(10))                         # Q1, Q2, Q3, Q4
    fiscal_year = Column(Integer)                               # Année fiscale

    # EPS
    eps_actual = Column(Float)                                  # EPS réel
    eps_estimate = Column(Float)                                # EPS estimé
    eps_surprise = Column(Float)                                # Surprise (actual - estimate)
    eps_surprise_percent = Column(Float)                        # Surprise en %

    # Revenus
    revenue_actual = Column(Float)                              # Revenus réels
    revenue_estimate = Column(Float)                            # Revenus estimés
    revenue_surprise = Column(Float)                            # Surprise revenus
    revenue_surprise_percent = Column(Float)                    # Surprise revenus %

    # Timing
    timing = Column(String(20))                                 # before_market, after_market
    has_reported = Column(Boolean, default=False)               # Déjà publié?

    # Métadonnées
    source = Column(String(50), default="finnhub")
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index composé
    __table_args__ = (
        Index('idx_symbol_date_earnings', 'symbol', 'date'),
        Index('idx_date_reported', 'date', 'has_reported'),
    )

    def __repr__(self):
        return f"<EarningsEvent {self.symbol} {self.date} EPS={self.eps_actual}>"
