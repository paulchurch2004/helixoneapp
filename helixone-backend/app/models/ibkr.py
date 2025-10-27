"""
Modèles de données pour l'intégration Interactive Brokers
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


class IBKRConnection(Base):
    """Configuration de connexion IBKR pour un utilisateur"""
    __tablename__ = "ibkr_connections"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)

    # Configuration
    account_id = Column(String, nullable=False)  # Ex: U17421384
    connection_type = Column(String, default='live')  # 'live' ou 'paper'
    host = Column(String, default='127.0.0.1')
    port = Column(Integer, default=7496)  # 7496=live, 7497=paper
    client_id = Column(Integer, default=1)

    # État
    is_active = Column(Boolean, default=True)
    is_connected = Column(Boolean, default=False)
    auto_connect = Column(Boolean, default=True)  # Connexion automatique au lancement

    # Timestamps
    last_connected_at = Column(DateTime)
    last_disconnected_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relations
    snapshots = relationship("PortfolioSnapshot", back_populates="connection", cascade="all, delete-orphan")
    alerts = relationship("IBKRAlert", back_populates="connection", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<IBKRConnection {self.account_id} {'🟢' if self.is_connected else '🔴'}>"


class PortfolioSnapshot(Base):
    """Snapshot du portefeuille à un instant T"""
    __tablename__ = "portfolio_snapshots"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    connection_id = Column(String, ForeignKey('ibkr_connections.id'), nullable=False)
    account_id = Column(String, nullable=False)

    # Valeurs globales du compte
    net_liquidation = Column(Float)  # Valeur totale du compte
    total_cash = Column(Float)  # Cash disponible
    stock_value = Column(Float)  # Valeur des positions
    unrealized_pnl = Column(Float)  # P&L non réalisé
    realized_pnl = Column(Float)  # P&L réalisé
    daily_pnl = Column(Float)  # P&L du jour

    # Métriques de risque
    buying_power = Column(Float)
    available_funds = Column(Float)
    excess_liquidity = Column(Float)

    # Devise
    currency = Column(String, default='EUR')

    # Positions (JSON array)
    positions = Column(JSON)  # [{symbol, quantity, avg_cost, market_value, ...}]

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relations
    connection = relationship("IBKRConnection", back_populates="snapshots")

    def __repr__(self):
        return f"<Snapshot {self.account_id} ${self.net_liquidation:.2f} @ {self.timestamp}>"


class IBKRPosition(Base):
    """Position individuelle dans le portefeuille"""
    __tablename__ = "ibkr_positions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    snapshot_id = Column(String, ForeignKey('portfolio_snapshots.id'), nullable=False)
    account_id = Column(String, nullable=False)

    # Détails de la position
    symbol = Column(String, nullable=False, index=True)
    sec_type = Column(String, default='STK')  # STK, OPT, FUT, etc.
    exchange = Column(String)
    currency = Column(String, default='EUR')

    # Quantités et prix
    position = Column(Float, nullable=False)  # Quantité (+ = long, - = short)
    avg_cost = Column(Float)  # Prix moyen d'achat
    market_price = Column(Float)  # Prix de marché actuel
    market_value = Column(Float)  # Valeur de marché totale

    # P&L
    unrealized_pnl = Column(Float)  # P&L non réalisé ($)
    unrealized_pnl_pct = Column(Float)  # P&L non réalisé (%)
    realized_pnl = Column(Float)  # P&L réalisé

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Position {self.symbol} {self.position} @ {self.avg_cost}>"


class IBKROrder(Base):
    """Historique des ordres IBKR"""
    __tablename__ = "ibkr_orders"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    connection_id = Column(String, ForeignKey('ibkr_connections.id'), nullable=False)

    # IDs
    order_id = Column(String, unique=True, nullable=False, index=True)
    perm_id = Column(String)  # Permanent ID

    # Symbole
    symbol = Column(String, nullable=False, index=True)
    sec_type = Column(String, default='STK')
    exchange = Column(String)
    currency = Column(String, default='EUR')

    # Détails de l'ordre
    action = Column(String)  # BUY, SELL
    order_type = Column(String)  # MARKET, LIMIT, STOP, etc.
    total_quantity = Column(Float)
    filled_quantity = Column(Float, default=0)
    remaining_quantity = Column(Float)

    # Prix
    limit_price = Column(Float)
    stop_price = Column(Float)
    avg_fill_price = Column(Float)

    # État
    status = Column(String)  # Submitted, PreSubmitted, Filled, Cancelled, etc.

    # Exécution
    commission = Column(Float)
    realized_pnl = Column(Float)

    # Timestamps
    submitted_at = Column(DateTime)
    filled_at = Column(DateTime)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Order {self.symbol} {self.action} {self.total_quantity} @ {self.status}>"


class IBKRAlert(Base):
    """Alertes générées sur le portefeuille IBKR"""
    __tablename__ = "ibkr_alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    connection_id = Column(String, ForeignKey('ibkr_connections.id'), nullable=False)

    # Type d'alerte
    alert_type = Column(String, nullable=False, index=True)
    # position_loss, portfolio_drawdown, concentration_risk, volatility_spike, etc.

    # Sévérité
    severity = Column(String, default='medium')  # low, medium, high, critical

    # Détails
    symbol = Column(String, index=True)  # Si lié à une position spécifique
    title = Column(String, nullable=False)
    message = Column(String, nullable=False)

    # Données associées (JSON)
    data = Column(JSON)  # Détails supplémentaires

    # Recommandations
    recommendations = Column(JSON)  # [{action, reason, priority}]

    # État
    is_active = Column(Boolean, default=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)

    # Timestamps
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime)

    # Relations
    connection = relationship("IBKRConnection", back_populates="alerts")

    def __repr__(self):
        return f"<Alert {self.alert_type} {self.severity} - {self.title}>"


class IBKRAccountSummary(Base):
    """Résumé détaillé du compte IBKR"""
    __tablename__ = "ibkr_account_summary"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    connection_id = Column(String, ForeignKey('ibkr_connections.id'), nullable=False)
    account_id = Column(String, nullable=False, index=True)

    # Toutes les clés du account summary IBKR
    account_type = Column(String)  # Individual, Margin, etc.
    cushion = Column(Float)  # Coussin de sécurité
    day_trades_remaining = Column(Integer)
    leverage = Column(Float)
    look_ahead_available_funds = Column(Float)
    look_ahead_excess_liquidity = Column(Float)
    look_ahead_initial_margin_req = Column(Float)
    look_ahead_maintenance_margin_req = Column(Float)
    maintenance_margin_req = Column(Float)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<AccountSummary {self.account_id} @ {self.timestamp}>"
