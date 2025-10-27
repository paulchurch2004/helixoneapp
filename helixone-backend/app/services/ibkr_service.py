"""
Service pour l'intégration Interactive Brokers
Gère la connexion, récupération du portefeuille, alertes
"""

from ib_insync import IB, util
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import asyncio

# Essayer d'appliquer nest_asyncio seulement si compatible avec la loop actuelle
# (nest_asyncio ne fonctionne pas avec uvloop utilisé par uvicorn)
try:
    import nest_asyncio
    try:
        loop = asyncio.get_event_loop()
        # Ne patcher que si ce n'est pas uvloop
        if 'uvloop' not in str(type(loop)):
            nest_asyncio.apply()
    except:
        # Pas de loop encore, ignorer
        pass
except ImportError:
    # nest_asyncio pas installé, ignorer
    pass

from app.models.ibkr import (
    IBKRConnection,
    PortfolioSnapshot,
    IBKRPosition,
    IBKROrder,
    IBKRAlert
)

logger = logging.getLogger(__name__)


class IBKRService:
    """Service principal pour IBKR"""

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self.ib = IB()
        self.connection_config: Optional[IBKRConnection] = None
        self.is_connected = False

    # ============================================
    # CONNEXION
    # ============================================

    async def connect(self, auto_connect=True) -> bool:
        """
        Connecter à Interactive Brokers

        Args:
            auto_connect: Si True, récupère la config depuis la BDD

        Returns:
            True si connexion réussie
        """
        try:
            if auto_connect:
                # Récupérer la configuration depuis la BDD
                self.connection_config = self.db.query(IBKRConnection).filter(
                    IBKRConnection.user_id == self.user_id,
                    IBKRConnection.is_active == True,
                    IBKRConnection.auto_connect == True
                ).first()

                if not self.connection_config:
                    logger.warning(f"Aucune configuration IBKR auto-connect pour user {self.user_id}")
                    return False

            if not self.connection_config:
                logger.error("Pas de configuration IBKR")
                return False

            logger.info(f"🔌 Connexion à IBKR: {self.connection_config.account_id}")

            # Connecter (utilisez connectAsync pour une connexion asynchrone)
            try:
                await self.ib.connectAsync(
                    self.connection_config.host,
                    self.connection_config.port,
                    clientId=self.connection_config.client_id,
                    timeout=10
                )
            except AttributeError:
                # Fallback si connectAsync n'existe pas
                self.ib.connect(
                    self.connection_config.host,
                    self.connection_config.port,
                    clientId=self.connection_config.client_id,
                    timeout=10
                )

            self.is_connected = True

            # Mettre à jour dans la BDD
            self.connection_config.is_connected = True
            self.connection_config.last_connected_at = datetime.utcnow()
            self.db.commit()

            logger.info(f"✅ Connecté à IBKR: {self.connection_config.account_id}")

            return True

        except Exception as e:
            logger.error(f"❌ Erreur connexion IBKR: {e}")
            if self.connection_config:
                self.connection_config.is_connected = False
                self.db.commit()
            return False

    def disconnect(self):
        """Déconnecter d'IBKR"""
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
                logger.info("🔌 Déconnecté d'IBKR")

            if self.connection_config:
                self.connection_config.is_connected = False
                self.connection_config.last_disconnected_at = datetime.utcnow()
                self.db.commit()

            self.is_connected = False

        except Exception as e:
            logger.error(f"Erreur déconnexion: {e}")

    # ============================================
    # RÉCUPÉRATION PORTEFEUILLE
    # ============================================

    async def get_portfolio(self) -> Optional[Dict]:
        """
        Récupérer le portefeuille complet

        Returns:
            Dict avec toutes les infos du portefeuille
        """
        if not self.is_connected:
            logger.warning("Pas connecté à IBKR")
            return None

        try:
            logger.info("📊 Récupération portefeuille IBKR...")

            # Attendre un peu pour recevoir les données
            await asyncio.sleep(1)

            # Account summary
            account_id = self.connection_config.account_id
            summary = self.ib.accountSummary(account_id)

            # Créer un dict des valeurs (ne convertir en float que les valeurs numériques)
            summary_dict = {}
            for item in summary:
                try:
                    # Essayer de convertir en float
                    summary_dict[item.tag] = float(item.value) if item.value else 0
                except (ValueError, TypeError):
                    # Si ce n'est pas un nombre, garder comme string
                    summary_dict[item.tag] = item.value if item.value else ""

            # Positions
            positions = self.ib.positions()
            positions_list = []

            for pos in positions:
                if pos.account == account_id:
                    position_dict = {
                        'symbol': pos.contract.symbol,
                        'sec_type': pos.contract.secType,
                        'exchange': pos.contract.exchange,
                        'currency': pos.contract.currency,
                        'position': pos.position,
                        'avg_cost': pos.avgCost,
                        'market_price': 0,  # À calculer
                        'market_value': 0,  # À calculer
                        'unrealized_pnl': 0,
                        'unrealized_pnl_pct': 0
                    }

                    # Calculer la market value si on a les données
                    if pos.position != 0 and pos.avgCost != 0:
                        # On va essayer d'obtenir le prix de marché
                        try:
                            ticker = self.ib.reqTickers(pos.contract)
                            if ticker and len(ticker) > 0:
                                market_price = ticker[0].marketPrice()
                                if market_price and market_price > 0:
                                    position_dict['market_price'] = market_price
                                    position_dict['market_value'] = market_price * pos.position

                                    # Calculer P&L
                                    cost_basis = pos.avgCost * pos.position
                                    position_dict['unrealized_pnl'] = position_dict['market_value'] - cost_basis
                                    if cost_basis != 0:
                                        position_dict['unrealized_pnl_pct'] = (position_dict['unrealized_pnl'] / abs(cost_basis)) * 100
                        except Exception as e:
                            logger.debug(f"Impossible d'obtenir prix marché pour {pos.contract.symbol}: {e}")

                    positions_list.append(position_dict)

            # Construire le résultat
            portfolio = {
                'account_id': account_id,
                'net_liquidation': summary_dict.get('NetLiquidation', 0),
                'total_cash': summary_dict.get('TotalCashValue', 0),
                'stock_value': summary_dict.get('GrossPositionValue', 0),
                'unrealized_pnl': summary_dict.get('UnrealizedPnL', 0),
                'realized_pnl': summary_dict.get('RealizedPnL', 0),
                'daily_pnl': 0,  # À calculer
                'buying_power': summary_dict.get('BuyingPower', 0),
                'available_funds': summary_dict.get('AvailableFunds', 0),
                'excess_liquidity': summary_dict.get('ExcessLiquidity', 0),
                'currency': 'EUR',
                'positions': positions_list,
                'timestamp': datetime.utcnow().isoformat()
            }

            logger.info(f"✅ Portefeuille récupéré: {len(positions_list)} positions")

            return portfolio

        except Exception as e:
            logger.error(f"❌ Erreur récupération portefeuille: {e}")
            return None

    async def save_portfolio_snapshot(self, portfolio: Dict) -> Optional[PortfolioSnapshot]:
        """
        Sauvegarder un snapshot du portefeuille dans la BDD

        Args:
            portfolio: Dict du portefeuille

        Returns:
            PortfolioSnapshot créé
        """
        try:
            snapshot = PortfolioSnapshot(
                connection_id=self.connection_config.id,
                account_id=portfolio['account_id'],
                net_liquidation=portfolio['net_liquidation'],
                total_cash=portfolio['total_cash'],
                stock_value=portfolio['stock_value'],
                unrealized_pnl=portfolio['unrealized_pnl'],
                realized_pnl=portfolio['realized_pnl'],
                daily_pnl=portfolio['daily_pnl'],
                buying_power=portfolio['buying_power'],
                available_funds=portfolio['available_funds'],
                excess_liquidity=portfolio['excess_liquidity'],
                currency=portfolio['currency'],
                positions=portfolio['positions']
            )

            self.db.add(snapshot)
            self.db.commit()
            self.db.refresh(snapshot)

            logger.info(f"💾 Snapshot sauvegardé: {snapshot.id}")

            return snapshot

        except Exception as e:
            logger.error(f"Erreur sauvegarde snapshot: {e}")
            self.db.rollback()
            return None

    # ============================================
    # ALERTES
    # ============================================

    async def check_alerts(self, portfolio: Dict) -> List[IBKRAlert]:
        """
        Vérifier si des alertes doivent être déclenchées

        Args:
            portfolio: Dict du portefeuille

        Returns:
            Liste des alertes créées
        """
        alerts = []

        try:
            # 1. Vérifier les pertes sur positions individuelles
            for position in portfolio['positions']:
                pnl_pct = position.get('unrealized_pnl_pct', 0)

                if pnl_pct < -10:
                    # Perte > 10% sur une position
                    alert = IBKRAlert(
                        connection_id=self.connection_config.id,
                        alert_type='position_loss',
                        severity='high' if pnl_pct < -15 else 'medium',
                        symbol=position['symbol'],
                        title=f"⚠️ Perte importante sur {position['symbol']}",
                        message=f"{position['symbol']} a perdu {abs(pnl_pct):.1f}% ({position['unrealized_pnl']:.2f} {portfolio['currency']})",
                        data={
                            'position': position,
                            'pnl_pct': pnl_pct,
                            'pnl': position['unrealized_pnl']
                        },
                        recommendations=[
                            {
                                'action': 'Placer un stop loss',
                                'reason': f'Limiter les pertes à {pnl_pct - 5:.0f}%',
                                'priority': 5
                            },
                            {
                                'action': 'Considérer une réduction de position',
                                'reason': 'Diminuer l\'exposition au risque',
                                'priority': 4
                            }
                        ]
                    )

                    self.db.add(alert)
                    alerts.append(alert)

            # 2. Vérifier le drawdown global du portefeuille
            total_pnl_pct = 0
            if portfolio['net_liquidation'] > 0 and portfolio['unrealized_pnl'] != 0:
                total_pnl_pct = (portfolio['unrealized_pnl'] / portfolio['net_liquidation']) * 100

            if total_pnl_pct < -5:
                alert = IBKRAlert(
                    connection_id=self.connection_config.id,
                    alert_type='portfolio_drawdown',
                    severity='high' if total_pnl_pct < -10 else 'medium',
                    title=f"⚠️ Drawdown du portefeuille",
                    message=f"Votre portefeuille a perdu {abs(total_pnl_pct):.1f}% ({portfolio['unrealized_pnl']:.2f} {portfolio['currency']})",
                    data={
                        'pnl_pct': total_pnl_pct,
                        'pnl': portfolio['unrealized_pnl'],
                        'net_liquidation': portfolio['net_liquidation']
                    },
                    recommendations=[
                        {
                            'action': 'Analyser avec le moteur de scénarios',
                            'reason': 'Identifier les risques systémiques',
                            'priority': 5
                        },
                        {
                            'action': 'Considérer un hedging',
                            'reason': 'Protéger contre une baisse continue',
                            'priority': 4
                        }
                    ]
                )

                self.db.add(alert)
                alerts.append(alert)

            # 3. Vérifier la concentration (si > 40% dans une seule position)
            if portfolio['net_liquidation'] > 0:
                for position in portfolio['positions']:
                    market_value = position.get('market_value', 0)
                    concentration_pct = (market_value / portfolio['net_liquidation']) * 100

                    if concentration_pct > 40:
                        alert = IBKRAlert(
                            connection_id=self.connection_config.id,
                            alert_type='concentration_risk',
                            severity='medium',
                            symbol=position['symbol'],
                            title=f"ℹ️ Concentration élevée sur {position['symbol']}",
                            message=f"{position['symbol']} représente {concentration_pct:.1f}% de votre portefeuille",
                            data={
                                'symbol': position['symbol'],
                                'concentration_pct': concentration_pct,
                                'market_value': market_value
                            },
                            recommendations=[
                                {
                                    'action': 'Diversifier votre portefeuille',
                                    'reason': 'Réduire le risque de concentration',
                                    'priority': 3
                                }
                            ]
                        )

                        self.db.add(alert)
                        alerts.append(alert)

            if alerts:
                self.db.commit()
                logger.info(f"🔔 {len(alerts)} nouvelles alertes créées")

            return alerts

        except Exception as e:
            logger.error(f"Erreur vérification alertes: {e}")
            self.db.rollback()
            return []

    # ============================================
    # HISTORIQUE ORDRES
    # ============================================

    async def get_orders_history(self, days: int = 30) -> List[Dict]:
        """
        Récupérer l'historique des ordres

        Args:
            days: Nombre de jours d'historique

        Returns:
            Liste des ordres
        """
        if not self.is_connected:
            return []

        try:
            trades = self.ib.trades()
            orders_list = []

            for trade in trades:
                order_dict = {
                    'order_id': str(trade.order.orderId),
                    'symbol': trade.contract.symbol,
                    'action': trade.order.action,
                    'order_type': trade.order.orderType,
                    'total_quantity': trade.order.totalQuantity,
                    'filled_quantity': trade.orderStatus.filled,
                    'remaining_quantity': trade.orderStatus.remaining,
                    'status': trade.orderStatus.status,
                    'avg_fill_price': trade.orderStatus.avgFillPrice,
                    'timestamp': datetime.utcnow().isoformat()
                }

                orders_list.append(order_dict)

            return orders_list

        except Exception as e:
            logger.error(f"Erreur récupération ordres: {e}")
            return []


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_ibkr_service(db: Session, user_id: str) -> IBKRService:
    """Factory pour créer une instance du service IBKR"""
    return IBKRService(db, user_id)


# Instance globale pour la connexion au démarrage
_global_ibkr_service: Optional[IBKRService] = None


async def init_ibkr_connections(db: Session):
    """
    Initialiser toutes les connexions IBKR au démarrage de l'application

    À appeler dans le startup event de FastAPI
    """
    global _global_ibkr_service

    logger.info("🚀 Initialisation des connexions IBKR...")

    # Récupérer toutes les connexions auto-connect
    connections = db.query(IBKRConnection).filter(
        IBKRConnection.is_active == True,
        IBKRConnection.auto_connect == True
    ).all()

    logger.info(f"Trouvé {len(connections)} connexion(s) IBKR à initialiser")

    for conn in connections:
        try:
            logger.info(f"Connexion IBKR pour user {conn.user_id} (compte {conn.account_id})...")

            service = IBKRService(db, conn.user_id)
            success = await service.connect(auto_connect=True)

            if success:
                logger.info(f"✅ Connecté: {conn.account_id}")

                # Récupérer le portefeuille initial
                portfolio = await service.get_portfolio()
                if portfolio:
                    await service.save_portfolio_snapshot(portfolio)
                    logger.info(f"💾 Snapshot initial sauvegardé")

            else:
                logger.warning(f"⚠️ Échec connexion: {conn.account_id}")

        except Exception as e:
            logger.error(f"Erreur initialisation IBKR pour {conn.account_id}: {e}")


def get_global_ibkr_service() -> Optional[IBKRService]:
    """Récupérer le service IBKR global"""
    return _global_ibkr_service
