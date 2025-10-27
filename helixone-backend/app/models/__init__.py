"""
Modèles SQLAlchemy
"""

from app.core.database import Base
from app.models.user import User
from app.models.license import License
from app.models.analysis import Analysis
from app.models.password_reset import PasswordResetToken
from app.models.market_data import (
    MarketDataOHLCV,
    MarketDataTick,
    MarketDataQuote,
    DataCollectionJob,
    DataCollectionSchedule,
    SymbolMetadata,
    TimeframeType,
    DataSourceType,
    CollectionStatus
)
from app.models.ibkr import (
    IBKRConnection,
    PortfolioSnapshot,
    IBKRPosition,
    IBKROrder,
    IBKRAlert,
    IBKRAccountSummary
)
from app.models.macro_data import (
    MacroEconomicData,
    EconomicIndicatorMetadata,
    YieldCurve,
    EconomicEvent,
    IndicatorCategory,
    IndicatorFrequency
)
from app.models.fundamental_data import (
    CompanyOverview,
    IncomeStatement,
    BalanceSheet,
    CashFlowStatement,
    EarningsCalendar
)
from app.models.news_sentiment import (
    NewsArticle,
    SentimentAnalysis,
    MarketSentiment,
    AnalystRecommendation,
    PriceTarget,
    EarningsEvent
)
from app.models.financial_ratios import (
    FinancialRatios,
    KeyMetrics,
    FinancialGrowth,
    DividendHistory,
    InsiderTrading,
    InstitutionalOwnership,
    AnalystEstimates
)
from app.models.portfolio import (
    PortfolioAnalysisHistory,
    PortfolioAlert,
    PortfolioRecommendation,
    RecommendationPerformance,
    AnalysisTimeType,
    AlertSeverity,
    RecommendationType,
    AlertStatus
)
from app.models.event_impact import (
    EventImpactHistory,
    EventPrediction,
    SectorEventCorrelation,
    EventAlert
)

__all__ = [
    "Base",
    "User",
    "License",
    "Analysis",
    "PasswordResetToken",
    "MarketDataOHLCV",
    "MarketDataTick",
    "MarketDataQuote",
    "DataCollectionJob",
    "DataCollectionSchedule",
    "SymbolMetadata",
    "TimeframeType",
    "DataSourceType",
    "CollectionStatus",
    "IBKRConnection",
    "PortfolioSnapshot",
    "IBKRPosition",
    "IBKROrder",
    "IBKRAlert",
    "IBKRAccountSummary",
    "MacroEconomicData",
    "EconomicIndicatorMetadata",
    "YieldCurve",
    "EconomicEvent",
    "IndicatorCategory",
    "IndicatorFrequency",
    "CompanyOverview",
    "IncomeStatement",
    "BalanceSheet",
    "CashFlowStatement",
    "EarningsCalendar",
    "NewsArticle",
    "SentimentAnalysis",
    "MarketSentiment",
    "AnalystRecommendation",
    "PriceTarget",
    "EarningsEvent",
    "FinancialRatios",
    "KeyMetrics",
    "FinancialGrowth",
    "DividendHistory",
    "InsiderTrading",
    "InstitutionalOwnership",
    "AnalystEstimates",
    "PortfolioAnalysisHistory",
    "PortfolioAlert",
    "PortfolioRecommendation",
    "RecommendationPerformance",
    "AnalysisTimeType",
    "AlertSeverity",
    "RecommendationType",
    "AlertStatus",
    "EventImpactHistory",
    "EventPrediction",
    "SectorEventCorrelation",
    "EventAlert"
]