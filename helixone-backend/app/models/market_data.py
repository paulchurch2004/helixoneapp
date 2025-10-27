"""
Modèles de données pour le stockage des données de marché
Supporte: OHLCV journalier, intraday, minutes, et tick-by-tick
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Enum, ForeignKey, Index, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid

from app.core.database import Base


class TimeframeType(str, enum.Enum):
    """Types de timeframes pour les données"""
    TICK = "tick"                # Tick by tick
    SECOND_1 = "1s"             # 1 seconde
    SECOND_5 = "5s"             # 5 secondes
    MINUTE_1 = "1m"             # 1 minute
    MINUTE_5 = "5m"             # 5 minutes
    MINUTE_15 = "15m"           # 15 minutes
    MINUTE_30 = "30m"           # 30 minutes
    HOUR_1 = "1h"               # 1 heure
    HOUR_4 = "4h"               # 4 heures
    DAILY = "1d"                # Journalier
    WEEKLY = "1w"               # Hebdomadaire
    MONTHLY = "1M"              # Mensuel


class DataSourceType(str, enum.Enum):
    """Sources de données de marché"""
    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alphavantage"
    POLYGON = "polygon"
    IEX_CLOUD = "iex"
    FINNHUB = "finnhub"
    TWELVE_DATA = "twelvedata"
    MANUAL = "manual"


class CollectionStatus(str, enum.Enum):
    """Statut de collecte des données"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class MarketDataOHLCV(Base):
    """
    Table pour stocker les données OHLCV (Open, High, Low, Close, Volume)
    Supporte tous les timeframes de tick à mensuel
    """
    __tablename__ = "market_data_ohlcv"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)  # AAPL, MSFT, etc.
    exchange = Column(String(20))  # NASDAQ, NYSE, etc.
    timeframe = Column(Enum(TimeframeType), nullable=False, index=True)

    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)  # UTC timestamp

    # Prix OHLCV
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False, default=0)

    # Données supplémentaires
    vwap = Column(Float)  # Volume Weighted Average Price
    trades_count = Column(Integer)  # Nombre de trades dans la période

    # Métadonnées
    source = Column(Enum(DataSourceType), nullable=False)
    collected_at = Column(DateTime, default=datetime.utcnow)
    is_adjusted = Column(Boolean, default=False)  # Prix ajusté pour splits/dividendes

    # Index composé pour recherches rapides
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
        Index('idx_timestamp_symbol', 'timestamp', 'symbol'),
    )

    def __repr__(self):
        return f"<OHLCV {self.symbol} {self.timeframe} {self.timestamp} C:{self.close}>"


class MarketDataTick(Base):
    """
    Table pour stocker les données tick-by-tick
    Plus granulaire: chaque transaction individuelle
    """
    __tablename__ = "market_data_tick"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(20))

    # Timestamp précis (microsecondes)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Prix du tick
    price = Column(Float, nullable=False)
    size = Column(Integer, nullable=False)  # Quantité échangée

    # Bid/Ask spread
    bid = Column(Float)
    ask = Column(Float)
    bid_size = Column(Integer)
    ask_size = Column(Integer)

    # Prix mid (average bid/ask)
    mid_price = Column(Float)

    # Type de trade
    trade_type = Column(String(10))  # 'buy', 'sell', 'market', 'limit'

    # Conditions de trade
    conditions = Column(String(100))  # Conditions spéciales du trade

    # Métadonnées
    source = Column(Enum(DataSourceType), nullable=False)
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index pour recherches rapides
    __table_args__ = (
        Index('idx_tick_symbol_timestamp', 'symbol', 'timestamp'),
    )

    def __repr__(self):
        return f"<Tick {self.symbol} {self.timestamp} P:{self.price} S:{self.size}>"


class MarketDataQuote(Base):
    """
    Table pour stocker les quotes (bid/ask) en temps réel
    Utile pour le mid price et le spread analysis
    """
    __tablename__ = "market_data_quote"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(20))

    # Timestamp
    timestamp = Column(DateTime, nullable=False, index=True)

    # Bid/Ask
    bid = Column(Float, nullable=False)
    ask = Column(Float, nullable=False)
    bid_size = Column(Integer)
    ask_size = Column(Integer)

    # Calculs dérivés
    mid_price = Column(Float)  # (bid + ask) / 2
    spread = Column(Float)  # ask - bid
    spread_pct = Column(Float)  # spread / mid_price * 100

    # Métadonnées
    source = Column(Enum(DataSourceType), nullable=False)
    collected_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_quote_symbol_timestamp', 'symbol', 'timestamp'),
    )

    def __repr__(self):
        return f"<Quote {self.symbol} {self.timestamp} Bid:{self.bid} Ask:{self.ask}>"


class DataCollectionJob(Base):
    """
    Table pour tracker les jobs de collecte de données
    Permet de suivre l'historique et les erreurs
    """
    __tablename__ = "data_collection_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Job info
    job_name = Column(String(100), nullable=False)
    job_type = Column(String(50))  # 'historical', 'realtime', 'backfill'

    # Paramètres
    symbols = Column(String(1000))  # Liste de symbols (JSON array as string)
    timeframe = Column(Enum(TimeframeType))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    source = Column(Enum(DataSourceType), nullable=False)

    # Statut
    status = Column(Enum(CollectionStatus), default=CollectionStatus.PENDING)
    progress = Column(Float, default=0.0)  # Pourcentage 0-100

    # Résultats
    records_collected = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(String(1000))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Métadonnées
    user_id = Column(String, ForeignKey('users.id'))

    def __repr__(self):
        return f"<DataCollectionJob {self.job_name} {self.status}>"


class DataCollectionSchedule(Base):
    """
    Table pour les collectes de données planifiées/récurrentes
    Ex: collecter AAPL toutes les minutes, ou tous les jours
    """
    __tablename__ = "data_collection_schedules"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Schedule info
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    is_active = Column(Boolean, default=True)

    # Paramètres de collecte
    symbols = Column(String(1000))  # Liste de symbols
    timeframe = Column(Enum(TimeframeType), nullable=False)
    source = Column(Enum(DataSourceType), nullable=False)

    # Configuration de la récurrence
    cron_expression = Column(String(100))  # Ex: "*/5 * * * *" pour chaque 5 min
    interval_minutes = Column(Integer)  # Alternative: interval simple en minutes

    # Limites
    max_history_days = Column(Integer, default=365)  # Combien de jours garder

    # Statut
    last_run_at = Column(DateTime)
    last_run_status = Column(Enum(CollectionStatus))
    next_run_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Métadonnées
    created_by = Column(String, ForeignKey('users.id'))

    def __repr__(self):
        return f"<Schedule {self.name} {self.timeframe} Active:{self.is_active}>"


class SymbolMetadata(Base):
    """
    Table pour stocker les métadonnées des symboles
    Informations sur les entreprises, secteurs, etc.
    """
    __tablename__ = "symbol_metadata"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200))  # Nom complet de l'entreprise
    exchange = Column(String(20))

    # Classification
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)

    # Détails
    country = Column(String(50))
    currency = Column(String(10), default="USD")
    description = Column(String(2000))
    website = Column(String(200))

    # Données de trading
    average_volume = Column(Integer)  # Volume moyen sur 3 mois
    float_shares = Column(Integer)  # Nombre d'actions en circulation

    # Indicateurs
    beta = Column(Float)  # Volatilité relative au marché
    pe_ratio = Column(Float)
    dividend_yield = Column(Float)

    # Statut
    is_active = Column(Boolean, default=True)
    delisted_date = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_data_update = Column(DateTime)  # Dernière mise à jour des données

    def __repr__(self):
        return f"<Symbol {self.symbol} {self.name}>"
