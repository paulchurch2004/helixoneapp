"""
Schémas Pydantic pour les données de marché
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal


class Quote(BaseModel):
    """Quote en temps réel pour une action"""

    ticker: str = Field(..., description="Symbole de l'action")
    name: Optional[str] = Field(None, description="Nom de la compagnie")
    price: Decimal = Field(..., description="Prix actuel")
    change: Optional[Decimal] = Field(None, description="Changement ($)")
    change_percent: Optional[Decimal] = Field(None, description="Changement (%)")
    volume: Optional[int] = Field(None, description="Volume")
    market_cap: Optional[Decimal] = Field(None, description="Capitalisation boursière")

    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    previous_close: Optional[Decimal] = None

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(..., description="Source des données (yahoo, finnhub, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "price": 178.50,
                "change": 2.30,
                "change_percent": 1.31,
                "volume": 54000000,
                "market_cap": 2800000000000,
                "open": 176.20,
                "high": 179.10,
                "low": 175.80,
                "previous_close": 176.20,
                "source": "yahoo"
            }
        }


class HistoricalPrice(BaseModel):
    """Prix historique pour une date donnée"""

    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted_close: Optional[Decimal] = None


class HistoricalData(BaseModel):
    """Données historiques pour une action"""

    ticker: str
    start_date: date
    end_date: date
    interval: str = Field(..., description="1d, 1wk, 1mo")
    prices: List[HistoricalPrice]
    source: str

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "interval": "1d",
                "prices": [],
                "source": "yahoo"
            }
        }


class Fundamentals(BaseModel):
    """Données fondamentales d'une entreprise"""

    ticker: str

    # Valuation
    market_cap: Optional[Decimal] = None
    enterprise_value: Optional[Decimal] = None
    pe_ratio: Optional[Decimal] = None
    pb_ratio: Optional[Decimal] = None
    ps_ratio: Optional[Decimal] = None
    peg_ratio: Optional[Decimal] = None
    ev_ebitda: Optional[Decimal] = None

    # Profitabilité
    profit_margin: Optional[Decimal] = None
    operating_margin: Optional[Decimal] = None
    roe: Optional[Decimal] = None  # Return on Equity
    roa: Optional[Decimal] = None  # Return on Assets
    roic: Optional[Decimal] = None  # Return on Invested Capital

    # Croissance
    revenue_growth: Optional[Decimal] = None
    earnings_growth: Optional[Decimal] = None
    revenue_per_share: Optional[Decimal] = None
    eps: Optional[Decimal] = None  # Earnings Per Share

    # Santé financière
    debt_to_equity: Optional[Decimal] = None
    current_ratio: Optional[Decimal] = None
    quick_ratio: Optional[Decimal] = None

    # Dividendes
    dividend_yield: Optional[Decimal] = None
    dividend_payout_ratio: Optional[Decimal] = None

    # Informations générales
    sector: Optional[str] = None
    industry: Optional[str] = None
    employees: Optional[int] = None
    description: Optional[str] = None

    source: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class NewsArticle(BaseModel):
    """Article de news"""

    title: str
    description: Optional[str] = None
    url: str
    source: str
    published_at: datetime
    sentiment: Optional[str] = None  # positive, negative, neutral
    sentiment_score: Optional[Decimal] = None  # 0-100
    related_tickers: List[str] = []


class NewsResponse(BaseModel):
    """Réponse avec liste de news"""

    ticker: Optional[str] = None
    articles: List[NewsArticle]
    total: int
    source: str


class SearchResult(BaseModel):
    """Résultat de recherche de ticker"""

    ticker: str
    name: str
    exchange: Optional[str] = None
    type: Optional[str] = None  # Stock, ETF, Index
    currency: Optional[str] = None
    country: Optional[str] = None


class SearchResponse(BaseModel):
    """Réponse de recherche"""

    query: str
    results: List[SearchResult]
    total: int


class ESGScore(BaseModel):
    """Scores ESG (Environmental, Social, Governance)"""

    ticker: str
    total_score: Optional[Decimal] = Field(None, description="Score ESG total")
    environment_score: Optional[Decimal] = Field(None, description="Score environnemental")
    social_score: Optional[Decimal] = Field(None, description="Score social")
    governance_score: Optional[Decimal] = Field(None, description="Score de gouvernance")
    controversy_level: Optional[int] = Field(None, description="Niveau de controverse (1-5)")
    grade: Optional[str] = Field(None, description="Note ESG (A+, A, B, C, D, F)")
    source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "total_score": 75.5,
                "environment_score": 80.2,
                "social_score": 72.1,
                "governance_score": 74.3,
                "controversy_level": 2,
                "grade": "A",
                "source": "finnhub"
            }
        }


class MarketStatus(BaseModel):
    """Statut du marché"""

    market: str  # US, EU, ASIA
    is_open: bool
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
