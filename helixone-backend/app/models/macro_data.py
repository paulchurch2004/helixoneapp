"""
Modèles de données pour le stockage des données macroéconomiques
Source principale: FRED (Federal Reserve Economic Data)
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Enum, Index, Text
from datetime import datetime
import enum
import uuid

from app.core.database import Base


class IndicatorCategory(str, enum.Enum):
    """Catégories d'indicateurs macroéconomiques"""
    INTEREST_RATES = "interest_rates"        # Taux d'intérêt
    INFLATION = "inflation"                  # Inflation (CPI, PCE, PPI)
    GDP = "gdp"                             # PIB et croissance
    EMPLOYMENT = "employment"                # Emploi et chômage
    HOUSING = "housing"                      # Marché immobilier
    CONSUMER = "consumer"                    # Consommation et ventes
    PRODUCTION = "production"                # Production industrielle
    MONEY_CREDIT = "money_credit"            # Monnaie et crédit
    MARKET_INDICES = "market_indices"        # Indices boursiers
    DEBT = "debt"                           # Dette publique
    TRADE = "trade"                         # Commerce international
    OTHER = "other"                         # Autres


class IndicatorFrequency(str, enum.Enum):
    """Fréquence de publication des indicateurs"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class MacroEconomicData(Base):
    """
    Table pour stocker les données macroéconomiques

    Sources:
    - FRED (Federal Reserve Economic Data)
    - World Bank
    - IMF
    - ECB
    - OECD
    """
    __tablename__ = "macro_economic_data"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification de la série
    series_id = Column(String(50), nullable=False, index=True)  # Code FRED (ex: 'DFF', 'CPIAUCSL')
    series_name = Column(String(200), nullable=False)           # Nom de la série
    category = Column(Enum(IndicatorCategory), nullable=False, index=True)

    # Données
    timestamp = Column(DateTime, nullable=False, index=True)    # Date de l'observation
    value = Column(Float, nullable=False)                       # Valeur de l'indicateur
    units = Column(String(50))                                  # Unité (%, Index, Billions, etc.)

    # Métadonnées
    frequency = Column(Enum(IndicatorFrequency))               # Fréquence de publication
    seasonal_adjustment = Column(String(50))                    # Ajustement saisonnier
    source = Column(String(50), default="FRED")                # Source des données
    collected_at = Column(DateTime, default=datetime.utcnow)   # Date de collecte
    notes = Column(Text)                                        # Notes explicatives

    # Index composé pour recherches rapides
    __table_args__ = (
        Index('idx_series_timestamp', 'series_id', 'timestamp'),
        Index('idx_category_timestamp', 'category', 'timestamp'),
    )

    def __repr__(self):
        return f"<MacroData {self.series_id} {self.timestamp} {self.value}>"


class EconomicIndicatorMetadata(Base):
    """
    Table pour stocker les métadonnées des indicateurs économiques
    """
    __tablename__ = "economic_indicator_metadata"

    series_id = Column(String(50), primary_key=True)            # Code FRED
    title = Column(String(500), nullable=False)                 # Titre complet
    category = Column(Enum(IndicatorCategory), nullable=False)
    units = Column(String(100))                                 # Unités de mesure
    frequency = Column(Enum(IndicatorFrequency))               # Fréquence
    seasonal_adjustment = Column(String(100))                   # Type d'ajustement saisonnier
    source = Column(String(100))                                # Source (FRED, etc.)
    description = Column(Text)                                  # Description détaillée
    popularity = Column(Integer, default=0)                     # Score de popularité
    last_updated = Column(DateTime)                             # Dernière mise à jour
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<IndicatorMetadata {self.series_id}: {self.title}>"


class YieldCurve(Base):
    """
    Table pour stocker les courbes de taux (yield curves)
    """
    __tablename__ = "yield_curves"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Date et identifiants
    date = Column(DateTime, nullable=False, index=True)
    country = Column(String(10), default="US", index=True)      # Pays (US, DE, JP, etc.)

    # Taux par maturité (en %)
    rate_1m = Column(Float)      # 1 mois
    rate_3m = Column(Float)      # 3 mois
    rate_6m = Column(Float)      # 6 mois
    rate_1y = Column(Float)      # 1 an
    rate_2y = Column(Float)      # 2 ans
    rate_3y = Column(Float)      # 3 ans
    rate_5y = Column(Float)      # 5 ans
    rate_7y = Column(Float)      # 7 ans
    rate_10y = Column(Float)     # 10 ans
    rate_20y = Column(Float)     # 20 ans
    rate_30y = Column(Float)     # 30 ans

    # Spreads calculés
    spread_10y_2y = Column(Float)  # Spread 10Y-2Y (indicateur de récession)
    spread_10y_3m = Column(Float)  # Spread 10Y-3M

    # Métadonnées
    source = Column(String(50), default="FRED")
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index
    __table_args__ = (
        Index('idx_date_country', 'date', 'country'),
    )

    def __repr__(self):
        return f"<YieldCurve {self.date} {self.country} 10Y:{self.rate_10y}%>"


class EconomicEvent(Base):
    """
    Table pour stocker les événements économiques importants
    (Fed meetings, QE announcements, policy changes, etc.)
    """
    __tablename__ = "economic_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    event_type = Column(String(50), nullable=False, index=True)  # 'fed_meeting', 'qe', 'policy_change', etc.
    title = Column(String(500), nullable=False)
    description = Column(Text)

    # Timing
    date = Column(DateTime, nullable=False, index=True)
    announced_date = Column(DateTime)                            # Si différent de la date effective

    # Impact
    impact_level = Column(String(20))                            # 'low', 'medium', 'high', 'critical'
    affected_markets = Column(String(500))                       # Liste des marchés affectés

    # Données quantitatives
    value = Column(Float)                                        # Valeur numérique si applicable
    units = Column(String(50))                                   # Unité de la valeur

    # Métadonnées
    source = Column(String(100))
    url = Column(String(500))                                    # Lien vers l'annonce officielle
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<EconomicEvent {self.event_type} {self.date}: {self.title}>"
