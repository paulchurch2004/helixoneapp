"""
Modèles de données pour le stockage des données fondamentales
Source principale: Alpha Vantage Fundamental Data API
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, BigInteger, Text, Index
from datetime import datetime
import uuid

from app.core.database import Base


class CompanyOverview(Base):
    """
    Table pour stocker les informations générales sur les entreprises
    """
    __tablename__ = "company_overview"

    symbol = Column(String(20), primary_key=True)              # Symbole du ticker
    name = Column(String(500), nullable=False)                 # Nom de l'entreprise
    description = Column(Text)                                  # Description de l'activité

    # Classification
    sector = Column(String(100), index=True)                   # Secteur
    industry = Column(String(100), index=True)                 # Industrie
    exchange = Column(String(20))                              # Bourse de cotation
    country = Column(String(50))                               # Pays du siège
    currency = Column(String(10))                              # Devise de cotation

    # Valorisation
    market_cap = Column(BigInteger)                            # Capitalisation boursière
    pe_ratio = Column(Float)                                   # P/E ratio
    peg_ratio = Column(Float)                                  # PEG ratio
    book_value = Column(Float)                                 # Valeur comptable
    dividend_yield = Column(Float)                             # Rendement du dividende
    eps = Column(Float)                                        # Bénéfice par action
    revenue_per_share_ttm = Column(Float)                      # CA par action (TTM)

    # Marges
    profit_margin = Column(Float)                              # Marge bénéficiaire
    operating_margin_ttm = Column(Float)                       # Marge opérationnelle
    return_on_assets_ttm = Column(Float)                       # ROA
    return_on_equity_ttm = Column(Float)                       # ROE

    # Performance
    revenue_ttm = Column(BigInteger)                           # Chiffre d'affaires (TTM)
    gross_profit_ttm = Column(BigInteger)                      # Profit brut (TTM)
    diluted_eps_ttm = Column(Float)                            # EPS dilué (TTM)
    quarterly_earnings_growth_yoy = Column(Float)              # Croissance trimestrielle
    quarterly_revenue_growth_yoy = Column(Float)               # Croissance CA trimestrielle

    # Actions
    shares_outstanding = Column(BigInteger)                    # Actions en circulation
    shares_float = Column(BigInteger)                          # Float
    shares_short = Column(BigInteger)                          # Actions vendues à découvert
    short_ratio = Column(Float)                                # Short ratio

    # Prix
    week_52_high = Column(Float)                               # Plus haut 52 semaines
    week_52_low = Column(Float)                                # Plus bas 52 semaines
    day_50_ma = Column(Float)                                  # Moyenne mobile 50 jours
    day_200_ma = Column(Float)                                 # Moyenne mobile 200 jours

    # Risque
    beta = Column(Float)                                       # Beta (volatilité)

    # Dates
    last_updated = Column(DateTime)                            # Dernière mise à jour
    created_at = Column(DateTime, default=datetime.utcnow)
    collected_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CompanyOverview {self.symbol}: {self.name}>"


class IncomeStatement(Base):
    """
    Table pour stocker les comptes de résultat
    """
    __tablename__ = "income_statements"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    fiscal_date_ending = Column(DateTime, nullable=False, index=True)  # Date de clôture fiscale
    reported_currency = Column(String(10))

    # Revenus
    total_revenue = Column(BigInteger)                         # Chiffre d'affaires total
    cost_of_revenue = Column(BigInteger)                       # Coût des revenus
    gross_profit = Column(BigInteger)                          # Profit brut

    # Dépenses opérationnelles
    research_development = Column(BigInteger)                  # R&D
    selling_general_administrative = Column(BigInteger)        # Ventes, général, administratif
    operating_expenses = Column(BigInteger)                    # Total dépenses opérationnelles

    # Résultat opérationnel
    operating_income = Column(BigInteger)                      # Résultat opérationnel

    # Autres revenus/dépenses
    interest_income = Column(BigInteger)                       # Revenus d'intérêts
    interest_expense = Column(BigInteger)                      # Frais d'intérêts
    non_interest_income = Column(BigInteger)                   # Autres revenus
    other_non_operating_income = Column(BigInteger)            # Autres revenus non-opérationnels

    # Avant impôts
    income_before_tax = Column(BigInteger)                     # Résultat avant impôts
    income_tax_expense = Column(BigInteger)                    # Impôts sur les bénéfices

    # Résultat net
    net_income = Column(BigInteger)                            # Résultat net
    net_income_from_continuing_ops = Column(BigInteger)        # Résultat net des activités poursuivies

    # Par action
    eps = Column(Float)                                        # Bénéfice par action
    eps_diluted = Column(Float)                                # Bénéfice par action dilué

    # Métadonnées
    source = Column(String(50), default="alpha_vantage")
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index composé
    __table_args__ = (
        Index('idx_symbol_fiscal_date', 'symbol', 'fiscal_date_ending'),
    )

    def __repr__(self):
        return f"<IncomeStatement {self.symbol} {self.fiscal_date_ending}>"


class BalanceSheet(Base):
    """
    Table pour stocker les bilans
    """
    __tablename__ = "balance_sheets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    fiscal_date_ending = Column(DateTime, nullable=False, index=True)
    reported_currency = Column(String(10))

    # ACTIFS
    # Actifs courants
    cash_and_cash_equivalents = Column(BigInteger)             # Trésorerie
    short_term_investments = Column(BigInteger)                # Investissements court terme
    cash_and_short_term_investments = Column(BigInteger)       # Total trésorerie + investissements
    inventory = Column(BigInteger)                             # Inventaire
    current_accounts_receivable = Column(BigInteger)           # Créances clients
    total_current_assets = Column(BigInteger)                  # Total actifs courants

    # Actifs non-courants
    property_plant_equipment = Column(BigInteger)              # Immobilisations corporelles
    intangible_assets = Column(BigInteger)                     # Immobilisations incorporelles
    goodwill = Column(BigInteger)                              # Goodwill
    long_term_investments = Column(BigInteger)                 # Investissements long terme
    total_non_current_assets = Column(BigInteger)              # Total actifs non-courants

    # Total actifs
    total_assets = Column(BigInteger)                          # Total actifs

    # PASSIFS
    # Passifs courants
    current_accounts_payable = Column(BigInteger)              # Dettes fournisseurs
    current_debt = Column(BigInteger)                          # Dette court terme
    current_long_term_debt = Column(BigInteger)                # Partie courante dette LT
    total_current_liabilities = Column(BigInteger)             # Total passifs courants

    # Passifs non-courants
    long_term_debt = Column(BigInteger)                        # Dette long terme
    long_term_debt_noncurrent = Column(BigInteger)             # Dette LT non courante
    capital_lease_obligations = Column(BigInteger)             # Obligations de location
    total_non_current_liabilities = Column(BigInteger)         # Total passifs non-courants

    # Total passifs
    total_liabilities = Column(BigInteger)                     # Total passifs

    # CAPITAUX PROPRES
    common_stock = Column(BigInteger)                          # Actions ordinaires
    retained_earnings = Column(BigInteger)                     # Bénéfices non distribués
    treasury_stock = Column(BigInteger)                        # Actions propres
    total_shareholder_equity = Column(BigInteger)              # Total capitaux propres

    # Métadonnées
    source = Column(String(50), default="alpha_vantage")
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index composé
    __table_args__ = (
        Index('idx_symbol_fiscal_date_bs', 'symbol', 'fiscal_date_ending'),
    )

    def __repr__(self):
        return f"<BalanceSheet {self.symbol} {self.fiscal_date_ending}>"


class CashFlowStatement(Base):
    """
    Table pour stocker les tableaux de flux de trésorerie
    """
    __tablename__ = "cash_flow_statements"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    fiscal_date_ending = Column(DateTime, nullable=False, index=True)
    reported_currency = Column(String(10))

    # Flux de trésorerie opérationnels
    net_income = Column(BigInteger)                            # Résultat net
    depreciation = Column(BigInteger)                          # Dépréciation
    deferred_income_tax = Column(BigInteger)                   # Impôts différés
    stock_based_compensation = Column(BigInteger)              # Compensation en actions
    change_in_working_capital = Column(BigInteger)             # Variation du BFR
    change_in_receivables = Column(BigInteger)                 # Variation créances
    change_in_inventory = Column(BigInteger)                   # Variation stocks
    operating_cashflow = Column(BigInteger)                    # Flux opérationnels

    # Flux de trésorerie d'investissement
    capital_expenditures = Column(BigInteger)                  # Investissements (CAPEX)
    cashflow_from_investment = Column(BigInteger)              # Flux d'investissement

    # Flux de trésorerie de financement
    dividends_paid = Column(BigInteger)                        # Dividendes versés
    stock_sale_purchase = Column(BigInteger)                   # Vente/rachat d'actions
    debt_repayment = Column(BigInteger)                        # Remboursement de dette
    cashflow_from_financing = Column(BigInteger)               # Flux de financement

    # Variation nette
    net_change_in_cash = Column(BigInteger)                    # Variation nette de trésorerie

    # Free Cash Flow (calculé)
    free_cash_flow = Column(BigInteger)                        # FCF = Operating CF - CAPEX

    # Métadonnées
    source = Column(String(50), default="alpha_vantage")
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Index composé
    __table_args__ = (
        Index('idx_symbol_fiscal_date_cf', 'symbol', 'fiscal_date_ending'),
    )

    def __repr__(self):
        return f"<CashFlowStatement {self.symbol} {self.fiscal_date_ending}>"


class EarningsCalendar(Base):
    """
    Table pour stocker le calendrier des publications de résultats
    """
    __tablename__ = "earnings_calendar"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Identification
    symbol = Column(String(20), nullable=False, index=True)
    fiscal_date_ending = Column(DateTime, nullable=False, index=True)
    report_date = Column(DateTime, nullable=False, index=True)  # Date de publication

    # Résultats
    reported_eps = Column(Float)                               # EPS reporté
    estimated_eps = Column(Float)                              # EPS estimé (consensus)
    surprise = Column(Float)                                   # Surprise (reporté - estimé)
    surprise_percentage = Column(Float)                        # Surprise en %

    # Métadonnées
    source = Column(String(50), default="alpha_vantage")
    collected_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Earnings {self.symbol} {self.report_date} EPS:{self.reported_eps}>"
