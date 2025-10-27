"""
Modèles SQLAlchemy pour ratios financiers et métriques
Source: Financial Modeling Prep (FMP)

Author: HelixOne Team
"""

from sqlalchemy import Column, String, Float, DateTime, Integer, BigInteger, Index
from datetime import datetime
from app.core.database import Base


class FinancialRatios(Base):
    """
    Ratios financiers calculés (50+ ratios)
    Profitabilité, Liquidité, Solvabilité, Efficacité, Valorisation
    """
    __tablename__ = "financial_ratios"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    period = Column(String(20), nullable=False)  # "annual" ou "quarter"
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer, nullable=True)

    # === PROFITABILITÉ ===
    gross_profit_margin = Column(Float)  # Marge brute
    operating_profit_margin = Column(Float)  # Marge opérationnelle
    net_profit_margin = Column(Float)  # Marge nette
    return_on_assets = Column(Float)  # ROA
    return_on_equity = Column(Float)  # ROE
    return_on_capital_employed = Column(Float)  # ROCE

    # === LIQUIDITÉ ===
    current_ratio = Column(Float)  # Actif circulant / Passif circulant
    quick_ratio = Column(Float)  # (AC - Stocks) / PC
    cash_ratio = Column(Float)  # Cash / PC
    operating_cash_flow_ratio = Column(Float)  # OCF / PC

    # === SOLVABILITÉ ===
    debt_ratio = Column(Float)  # Dette totale / Actif total
    debt_to_equity = Column(Float)  # Dette / Capitaux propres
    long_term_debt_to_capitalization = Column(Float)
    interest_coverage = Column(Float)  # EBIT / Charges d'intérêts
    debt_service_coverage = Column(Float)

    # === EFFICACITÉ ===
    asset_turnover = Column(Float)  # CA / Actif moyen
    inventory_turnover = Column(Float)  # COGS / Stock moyen
    receivables_turnover = Column(Float)  # CA / Créances moyennes
    payables_turnover = Column(Float)
    days_sales_outstanding = Column(Float)  # DSO
    days_inventory_outstanding = Column(Float)  # DIO
    days_payables_outstanding = Column(Float)  # DPO
    cash_conversion_cycle = Column(Float)  # DSO + DIO - DPO

    # === VALORISATION ===
    price_to_earnings = Column(Float)  # P/E
    price_to_book = Column(Float)  # P/B
    price_to_sales = Column(Float)  # P/S
    price_to_free_cash_flow = Column(Float)  # P/FCF
    enterprise_value_to_ebitda = Column(Float)  # EV/EBITDA
    enterprise_value_to_sales = Column(Float)  # EV/Sales
    peg_ratio = Column(Float)  # (P/E) / Growth rate

    # === DIVIDENDES ===
    dividend_yield = Column(Float)
    dividend_payout_ratio = Column(Float)

    # === AUTRES ===
    earnings_per_share = Column(Float)  # EPS
    book_value_per_share = Column(Float)
    tangible_book_value_per_share = Column(Float)
    price_to_tangible_book = Column(Float)

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_fin_ratios_symbol_date', 'symbol', 'date'),
        Index('idx_fin_ratios_symbol_period', 'symbol', 'period'),
    )


class KeyMetrics(Base):
    """
    Métriques clés de l'entreprise
    Market cap, valuation, croissance, etc.
    """
    __tablename__ = "key_metrics"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    period = Column(String(20), nullable=False)
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer, nullable=True)

    # === CAPITALISATION ===
    market_capitalization = Column(BigInteger)  # Capitalisation boursière
    enterprise_value = Column(BigInteger)  # Valeur d'entreprise

    # === VALORISATION ===
    pe_ratio = Column(Float)
    peg_ratio = Column(Float)
    price_to_book = Column(Float)
    price_to_sales = Column(Float)
    price_to_free_cash_flow = Column(Float)
    ev_to_ebitda = Column(Float)
    ev_to_sales = Column(Float)
    ev_to_operating_cash_flow = Column(Float)

    # === CROISSANCE ===
    revenue_per_share = Column(Float)
    revenue_growth = Column(Float)  # Croissance CA
    earnings_growth = Column(Float)  # Croissance bénéfices
    operating_cash_flow_growth = Column(Float)

    # === PERFORMANCE ===
    roe = Column(Float)  # Return on Equity
    roa = Column(Float)  # Return on Assets
    roic = Column(Float)  # Return on Invested Capital
    net_income_per_share = Column(Float)
    operating_income_per_share = Column(Float)
    free_cash_flow_per_share = Column(Float)

    # === STRUCTURE ===
    shares_outstanding = Column(BigInteger)
    book_value_per_share = Column(Float)
    tangible_book_value_per_share = Column(Float)
    working_capital_per_share = Column(Float)

    # === DIVIDENDES ===
    dividend_yield = Column(Float)
    dividend_per_share = Column(Float)
    payout_ratio = Column(Float)

    # === DETTE ===
    debt_to_equity = Column(Float)
    debt_to_assets = Column(Float)
    net_debt_to_ebitda = Column(Float)
    interest_coverage = Column(Float)

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_key_metrics_symbol_date', 'symbol', 'date'),
        Index('idx_key_metrics_symbol_period', 'symbol', 'period'),
    )


class FinancialGrowth(Base):
    """
    Taux de croissance financière (YoY, QoQ)
    """
    __tablename__ = "financial_growth"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    period = Column(String(20), nullable=False)
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer, nullable=True)

    # === CROISSANCE DU CHIFFRE D'AFFAIRES ===
    revenue_growth = Column(Float)  # Croissance CA
    gross_profit_growth = Column(Float)
    ebitda_growth = Column(Float)
    operating_income_growth = Column(Float)
    net_income_growth = Column(Float)

    # === CROISSANCE PAR ACTION ===
    eps_growth = Column(Float)  # Croissance EPS
    revenue_per_share_growth = Column(Float)
    operating_cash_flow_per_share_growth = Column(Float)
    free_cash_flow_per_share_growth = Column(Float)
    book_value_per_share_growth = Column(Float)

    # === CROISSANCE DES ACTIFS ===
    total_assets_growth = Column(Float)
    total_liabilities_growth = Column(Float)
    shareholders_equity_growth = Column(Float)
    working_capital_growth = Column(Float)

    # === CROISSANCE DU CASH FLOW ===
    operating_cash_flow_growth = Column(Float)
    free_cash_flow_growth = Column(Float)
    capital_expenditure_growth = Column(Float)

    # === CROISSANCE DE LA DETTE ===
    total_debt_growth = Column(Float)
    net_debt_growth = Column(Float)

    # === AUTRES CROISSANCES ===
    receivables_growth = Column(Float)
    inventory_growth = Column(Float)
    rd_expense_growth = Column(Float)  # R&D
    sga_expense_growth = Column(Float)  # SG&A

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_fin_growth_symbol_date', 'symbol', 'date'),
        Index('idx_fin_growth_symbol_period', 'symbol', 'period'),
    )


class DividendHistory(Base):
    """
    Historique des dividendes
    """
    __tablename__ = "dividend_history"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)  # Ex-dividend date
    payment_date = Column(DateTime, nullable=True)
    record_date = Column(DateTime, nullable=True)
    declaration_date = Column(DateTime, nullable=True)

    # Montants
    dividend = Column(Float, nullable=False)  # Montant du dividende par action
    adjusted_dividend = Column(Float)  # Ajusté pour splits

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_dividend_symbol_date', 'symbol', 'date'),
    )


class InsiderTrading(Base):
    """
    Transactions d'initiés (Insider Trading)
    """
    __tablename__ = "insider_trading"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    filing_date = Column(DateTime, nullable=False, index=True)
    transaction_date = Column(DateTime, nullable=False, index=True)

    # Informations sur l'initié
    reporting_name = Column(String(200))  # Nom de l'initié
    relationship = Column(String(100))  # Position (CEO, CFO, Director, etc.)

    # Transaction
    transaction_type = Column(String(50))  # P-Purchase, S-Sale, A-Award, etc.
    securities_owned = Column(BigInteger)  # Actions détenues après transaction
    securities_transacted = Column(BigInteger)  # Nombre d'actions transigées
    price = Column(Float)  # Prix de transaction
    value = Column(Float)  # Valeur totale

    # Filing
    form_type = Column(String(20))  # Type de formulaire (Form 4, etc.)
    link = Column(String(500))  # Lien vers le filing SEC

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_insider_symbol_date', 'symbol', 'transaction_date'),
        Index('idx_insider_symbol_type', 'symbol', 'transaction_type'),
    )


class InstitutionalOwnership(Base):
    """
    Détention institutionnelle (13F filings)
    """
    __tablename__ = "institutional_ownership"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)  # Date du filing

    # Institution
    holder = Column(String(200), nullable=False)  # Nom de l'institution
    cik = Column(String(20))  # CIK number

    # Holdings
    shares = Column(BigInteger)  # Nombre d'actions détenues
    value = Column(BigInteger)  # Valeur en dollars
    weight_percent = Column(Float)  # % du portfolio de l'institution
    change = Column(BigInteger)  # Changement en actions
    change_percent = Column(Float)  # % de changement

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_inst_own_symbol_date', 'symbol', 'date'),
        Index('idx_inst_own_holder', 'holder'),
    )


class AnalystEstimates(Base):
    """
    Estimations des analystes (consensus)
    """
    __tablename__ = "analyst_estimates"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    period = Column(String(20), nullable=False)  # "annual" ou "quarter"
    fiscal_year = Column(Integer)
    fiscal_quarter = Column(Integer, nullable=True)

    # Estimations Revenue
    estimated_revenue_avg = Column(BigInteger)  # Moyenne consensus
    estimated_revenue_low = Column(BigInteger)
    estimated_revenue_high = Column(BigInteger)
    number_analyst_estimated_revenue = Column(Integer)

    # Estimations EPS
    estimated_eps_avg = Column(Float)
    estimated_eps_low = Column(Float)
    estimated_eps_high = Column(Float)
    number_analysts_estimated_eps = Column(Integer)

    # Estimations EBIT
    estimated_ebit_avg = Column(BigInteger, nullable=True)
    estimated_ebit_low = Column(BigInteger, nullable=True)
    estimated_ebit_high = Column(BigInteger, nullable=True)
    number_analysts_estimated_ebit = Column(Integer, nullable=True)

    # Estimations EBITDA
    estimated_ebitda_avg = Column(BigInteger, nullable=True)
    estimated_ebitda_low = Column(BigInteger, nullable=True)
    estimated_ebitda_high = Column(BigInteger, nullable=True)
    number_analysts_estimated_ebitda = Column(Integer, nullable=True)

    # Estimations Net Income
    estimated_net_income_avg = Column(BigInteger, nullable=True)
    estimated_net_income_low = Column(BigInteger, nullable=True)
    estimated_net_income_high = Column(BigInteger, nullable=True)
    number_analysts_estimated_net_income = Column(Integer, nullable=True)

    # Métadonnées
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_analyst_est_symbol_date', 'symbol', 'date'),
        Index('idx_analyst_est_symbol_period', 'symbol', 'period'),
    )
