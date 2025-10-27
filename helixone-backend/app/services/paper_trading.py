"""
Service de Paper Trading (simulation de trading)
Permet aux utilisateurs de trader avec de l'argent virtuel
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import yfinance as yf


class PaperTradingService:
    """Service de simulation de trading avec capital virtuel"""

    def __init__(self, user_id: str, data_dir: str = None):
        """
        Initialise le service pour un utilisateur

        Args:
            user_id: ID unique de l'utilisateur
            data_dir: Répertoire de stockage des portfolios (optionnel)
        """
        self.user_id = user_id

        # Définir le répertoire de données
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(__file__),
                "../../..",
                "data/paper_trading"
            )

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.portfolio_file = self.data_dir / f"portfolio_{user_id}.json"

        # Charger ou initialiser le portfolio
        self.portfolio = self._load_portfolio()

    def _load_portfolio(self) -> Dict:
        """Charge le portfolio depuis le fichier JSON"""
        if self.portfolio_file.exists():
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        else:
            # Créer un nouveau portfolio avec capital initial
            return {
                "user_id": self.user_id,
                "initial_capital": 100000.0,
                "cash": 100000.0,
                "positions": {},  # {ticker: quantity}
                "stop_loss_orders": {},  # {ticker: stop_price}
                "take_profit_orders": {},  # {ticker: take_profit_price}
                "history": [],  # Liste des trades
                "created_at": datetime.now().isoformat()
            }

    def _save_portfolio(self):
        """Sauvegarde le portfolio dans le fichier JSON"""
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)

    def _get_current_price(self, ticker: str) -> float:
        """
        Récupère le prix actuel d'une action via Yahoo Finance
        Fallback sur prix simulés si Yahoo Finance échoue (mode dev)

        Args:
            ticker: Symbole de l'action (ex: AAPL)

        Returns:
            Prix actuel de l'action

        Raises:
            ValueError: Si le ticker est invalide
        """
        # Prix simulés pour développement (si Yahoo Finance échoue)
        SIMULATED_PRICES = {
            "AAPL": 175.50,
            "MSFT": 380.25,
            "GOOGL": 140.75,
            "TSLA": 245.30,
            "NVDA": 495.80,
            "AMZN": 145.60,
            "META": 325.40,
            "NFLX": 425.90,
            "AMD": 135.20,
            "PYPL": 62.15
        }

        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")

            if data.empty:
                # Fallback sur prix simulé si disponible
                if ticker.upper() in SIMULATED_PRICES:
                    print(f"⚠️  Yahoo Finance indisponible pour {ticker}, utilisation prix simulé")
                    return SIMULATED_PRICES[ticker.upper()]
                raise ValueError(f"Ticker {ticker} invalide ou pas de données")

            # Prendre le dernier prix de clôture
            current_price = data['Close'].iloc[-1]
            return float(current_price)

        except Exception as e:
            # Fallback sur prix simulé
            if ticker.upper() in SIMULATED_PRICES:
                print(f"⚠️  Erreur Yahoo Finance pour {ticker}, utilisation prix simulé: {str(e)}")
                return SIMULATED_PRICES[ticker.upper()]
            raise ValueError(f"Erreur lors de la récupération du prix de {ticker}: {str(e)}")

    def _calculate_average_cost(self, ticker: str) -> float:
        """
        Calcule le coût moyen d'achat pour un ticker

        Args:
            ticker: Symbole de l'action

        Returns:
            Prix moyen d'achat
        """
        # Filtrer l'historique pour ce ticker (achats uniquement)
        buy_trades = [
            t for t in self.portfolio['history']
            if t['ticker'] == ticker and t['quantity'] > 0
        ]

        if not buy_trades:
            return 0.0

        total_cost = sum(t['price'] * t['quantity'] for t in buy_trades)
        total_quantity = sum(t['quantity'] for t in buy_trades)

        return total_cost / total_quantity if total_quantity > 0 else 0.0

    async def place_order(
        self,
        ticker: str,
        quantity: int,
        order_type: str = "market"
    ) -> Dict:
        """
        Passe un ordre d'achat ou de vente

        Args:
            ticker: Symbole de l'action (ex: AAPL)
            quantity: Nombre d'actions (positif = achat, négatif = vente)
            order_type: Type d'ordre ('market', 'limit', 'stop') - actuellement seul 'market' est supporté

        Returns:
            Dictionnaire avec le résultat de l'ordre

        Raises:
            ValueError: Si l'ordre est invalide
        """
        ticker = ticker.upper()

        # Récupérer le prix actuel
        try:
            current_price = self._get_current_price(ticker)
        except ValueError as e:
            return {
                "success": False,
                "error": str(e)
            }

        # ACHAT
        if quantity > 0:
            total_cost = current_price * quantity

            if total_cost > self.portfolio['cash']:
                return {
                    "success": False,
                    "error": f"Fonds insuffisants. Requis: ${total_cost:.2f}, Disponible: ${self.portfolio['cash']:.2f}"
                }

            # Déduire le cash
            self.portfolio['cash'] -= total_cost

            # Ajouter à la position
            if ticker in self.portfolio['positions']:
                self.portfolio['positions'][ticker] += quantity
            else:
                self.portfolio['positions'][ticker] = quantity

        # VENTE
        else:
            quantity_to_sell = abs(quantity)
            current_position = self.portfolio['positions'].get(ticker, 0)

            if current_position < quantity_to_sell:
                return {
                    "success": False,
                    "error": f"Position insuffisante. Possédé: {current_position}, Demandé: {quantity_to_sell}"
                }

            # Créditer le cash
            proceeds = current_price * quantity_to_sell
            self.portfolio['cash'] += proceeds

            # Retirer de la position
            self.portfolio['positions'][ticker] -= quantity_to_sell

            # Supprimer si position = 0
            if self.portfolio['positions'][ticker] == 0:
                del self.portfolio['positions'][ticker]

        # Enregistrer dans l'historique
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "quantity": quantity,
            "price": current_price,
            "total": current_price * abs(quantity),
            "type": "BUY" if quantity > 0 else "SELL",
            "order_type": order_type
        }

        self.portfolio['history'].append(trade_record)

        # Sauvegarder
        self._save_portfolio()

        return {
            "success": True,
            "trade": trade_record,
            "message": f"{'Achat' if quantity > 0 else 'Vente'} de {abs(quantity)} action(s) {ticker} à ${current_price:.2f}"
        }

    def get_portfolio_value(self) -> Dict:
        """
        Calcule la valeur totale du portfolio

        Returns:
            Dictionnaire avec les métriques du portfolio
        """
        total_positions_value = 0.0

        # Calculer la valeur de toutes les positions
        for ticker, qty in self.portfolio['positions'].items():
            try:
                current_price = self._get_current_price(ticker)
                total_positions_value += current_price * qty
            except ValueError:
                # Si on ne peut pas récupérer le prix, ignorer
                continue

        total_value = self.portfolio['cash'] + total_positions_value
        initial_capital = self.portfolio['initial_capital']

        pnl = total_value - initial_capital
        pnl_percent = (pnl / initial_capital) * 100 if initial_capital > 0 else 0

        return {
            "total_value": round(total_value, 2),
            "cash": round(self.portfolio['cash'], 2),
            "positions_value": round(total_positions_value, 2),
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_percent, 2),
            "initial_capital": initial_capital
        }

    def get_position_details(self) -> List[Dict]:
        """
        Récupère les détails de chaque position

        Returns:
            Liste des positions avec P&L
        """
        positions = []

        for ticker, qty in self.portfolio['positions'].items():
            if qty > 0:
                try:
                    current_price = self._get_current_price(ticker)
                    avg_cost = self._calculate_average_cost(ticker)

                    market_value = current_price * qty
                    cost_basis = avg_cost * qty
                    pnl = market_value - cost_basis
                    pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0

                    positions.append({
                        "ticker": ticker,
                        "quantity": qty,
                        "current_price": round(current_price, 2),
                        "average_cost": round(avg_cost, 2),
                        "market_value": round(market_value, 2),
                        "cost_basis": round(cost_basis, 2),
                        "pnl": round(pnl, 2),
                        "pnl_percent": round(pnl_percent, 2)
                    })
                except ValueError:
                    # Impossible de récupérer le prix, créer entrée avec données limitées
                    positions.append({
                        "ticker": ticker,
                        "quantity": qty,
                        "error": "Prix indisponible"
                    })

        return positions

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """
        Récupère l'historique des trades

        Args:
            limit: Nombre maximum de trades à retourner

        Returns:
            Liste des trades récents
        """
        # Retourner les derniers trades (les plus récents en premier)
        return list(reversed(self.portfolio['history'][-limit:]))

    def reset_portfolio(self, initial_capital: float = 100000.0) -> Dict:
        """
        Réinitialise le portfolio (utile pour recommencer)

        Args:
            initial_capital: Capital initial

        Returns:
            Nouveau portfolio
        """
        self.portfolio = {
            "user_id": self.user_id,
            "initial_capital": initial_capital,
            "cash": initial_capital,
            "positions": {},
            "history": [],
            "created_at": datetime.now().isoformat()
        }

        self._save_portfolio()

        return {
            "success": True,
            "message": f"Portfolio réinitialisé avec ${initial_capital:,.2f}",
            "portfolio": self.get_portfolio_value()
        }

    def get_statistics(self) -> Dict:
        """
        Calcule les statistiques de trading avancées

        Returns:
            Statistiques complètes incluant win rate, P&L moyen, meilleurs/pires trades, etc.
        """
        history = self.portfolio['history']

        if not history:
            return {
                "total_trades": 0,
                "buy_count": 0,
                "sell_count": 0,
                "message": "Aucun trade effectué"
            }

        buy_trades = [t for t in history if t['type'] == 'BUY']
        sell_trades = [t for t in history if t['type'] == 'SELL']

        total_bought = sum(t['total'] for t in buy_trades)
        total_sold = sum(t['total'] for t in sell_trades)

        portfolio_value = self.get_portfolio_value()

        # Stats de base
        stats = {
            "total_trades": len(history),
            "buy_count": len(buy_trades),
            "sell_count": len(sell_trades),
            "total_invested": round(total_bought, 2),
            "total_proceeds": round(total_sold, 2),
            "current_pnl": portfolio_value['pnl'],
            "current_pnl_percent": portfolio_value['pnl_percent'],
            "tickers_traded": list(set(t['ticker'] for t in history))
        }

        # Calcul des trades complets (paires achat/vente) pour stats avancées
        if sell_trades:
            closed_trades = []

            # Pour chaque vente, trouver les achats correspondants
            for sell in sell_trades:
                ticker = sell['ticker']
                sell_quantity = abs(sell['quantity'])
                sell_price = sell['price']

                # Trouver les achats antérieurs de ce ticker
                prior_buys = [
                    b for b in buy_trades
                    if b['ticker'] == ticker and b['timestamp'] < sell['timestamp']
                ]

                if prior_buys:
                    # Calculer le coût moyen d'achat
                    avg_buy_price = sum(b['price'] * b['quantity'] for b in prior_buys) / sum(b['quantity'] for b in prior_buys)

                    # P&L de ce trade
                    trade_pnl = (sell_price - avg_buy_price) * sell_quantity
                    trade_pnl_percent = ((sell_price - avg_buy_price) / avg_buy_price) * 100

                    closed_trades.append({
                        "ticker": ticker,
                        "buy_price": avg_buy_price,
                        "sell_price": sell_price,
                        "quantity": sell_quantity,
                        "pnl": trade_pnl,
                        "pnl_percent": trade_pnl_percent,
                        "timestamp": sell['timestamp']
                    })

            if closed_trades:
                # Stats de performance
                winning_trades = [t for t in closed_trades if t['pnl'] > 0]
                losing_trades = [t for t in closed_trades if t['pnl'] < 0]

                # Win rate
                win_count = len(winning_trades)
                loss_count = len(losing_trades)
                total_closed = len(closed_trades)
                win_rate = (win_count / total_closed) * 100 if total_closed > 0 else 0

                # P&L moyen
                avg_win = sum(t['pnl'] for t in winning_trades) / win_count if win_count > 0 else 0
                avg_loss = sum(t['pnl'] for t in losing_trades) / loss_count if loss_count > 0 else 0
                avg_pnl = sum(t['pnl'] for t in closed_trades) / total_closed if total_closed > 0 else 0

                # Ratio gain/perte moyen
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

                # Meilleur et pire trade
                best_trade = max(closed_trades, key=lambda t: t['pnl'])
                worst_trade = min(closed_trades, key=lambda t: t['pnl'])

                # Plus longue série de gains/pertes
                current_streak = 0
                max_win_streak = 0
                max_loss_streak = 0

                for trade in closed_trades:
                    if trade['pnl'] > 0:
                        if current_streak >= 0:
                            current_streak += 1
                        else:
                            current_streak = 1
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        if current_streak <= 0:
                            current_streak -= 1
                        else:
                            current_streak = -1
                        max_loss_streak = max(max_loss_streak, abs(current_streak))

                # Ratio Risk/Reward
                risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

                # Stats avancées
                stats['performance'] = {
                    "closed_trades": total_closed,
                    "winning_trades": win_count,
                    "losing_trades": loss_count,
                    "win_rate": round(win_rate, 2),
                    "avg_profit": round(avg_win, 2),
                    "avg_loss": round(avg_loss, 2),
                    "avg_pnl_per_trade": round(avg_pnl, 2),
                    "profit_factor": round(profit_factor, 2),
                    "risk_reward_ratio": round(risk_reward_ratio, 2),
                    "max_win_streak": max_win_streak,
                    "max_loss_streak": max_loss_streak,
                    "best_trade": {
                        "ticker": best_trade['ticker'],
                        "pnl": round(best_trade['pnl'], 2),
                        "pnl_percent": round(best_trade['pnl_percent'], 2),
                        "buy_price": round(best_trade['buy_price'], 2),
                        "sell_price": round(best_trade['sell_price'], 2)
                    },
                    "worst_trade": {
                        "ticker": worst_trade['ticker'],
                        "pnl": round(worst_trade['pnl'], 2),
                        "pnl_percent": round(worst_trade['pnl_percent'], 2),
                        "buy_price": round(worst_trade['buy_price'], 2),
                        "sell_price": round(worst_trade['sell_price'], 2)
                    }
                }

                # Ratio de Sharpe simplifié (approximation)
                if len(closed_trades) > 1:
                    returns = [t['pnl_percent'] for t in closed_trades]
                    avg_return = sum(returns) / len(returns)

                    # Écart-type
                    variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
                    std_dev = variance ** 0.5

                    # Sharpe ratio simplifié (sans taux sans risque)
                    sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0

                    stats['performance']['sharpe_ratio'] = round(sharpe_ratio, 2)
                    stats['performance']['volatility'] = round(std_dev, 2)

        return stats

    def set_stop_loss(self, ticker: str, stop_price: float) -> Dict:
        """
        Définit un stop-loss pour une position

        Args:
            ticker: Symbole de l'action
            stop_price: Prix du stop-loss

        Returns:
            Confirmation
        """
        ticker = ticker.upper()

        # Vérifier qu'on a une position
        if ticker not in self.portfolio['positions'] or self.portfolio['positions'][ticker] <= 0:
            return {
                "success": False,
                "error": f"Aucune position ouverte sur {ticker}"
            }

        # Vérifier que le stop-loss est cohérent
        try:
            current_price = self._get_current_price(ticker)
            
            # Pour une position longue, le stop doit être sous le prix actuel
            if stop_price >= current_price:
                return {
                    "success": False,
                    "error": f"Le stop-loss ({stop_price}) doit être INFÉRIEUR au prix actuel ({current_price})"
                }

        except ValueError as e:
            return {
                "success": False,
                "error": str(e)
            }

        # Initialiser le dict si nécessaire
        if 'stop_loss_orders' not in self.portfolio:
            self.portfolio['stop_loss_orders'] = {}

        # Enregistrer le stop-loss
        self.portfolio['stop_loss_orders'][ticker] = stop_price
        self._save_portfolio()

        return {
            "success": True,
            "ticker": ticker,
            "stop_price": stop_price,
            "current_price": current_price,
            "message": f"Stop-loss placé sur {ticker} à ${stop_price:.2f} (prix actuel: ${current_price:.2f})"
        }

    def set_take_profit(self, ticker: str, take_profit_price: float) -> Dict:
        """
        Définit un take-profit pour une position

        Args:
            ticker: Symbole de l'action
            take_profit_price: Prix du take-profit

        Returns:
            Confirmation
        """
        ticker = ticker.upper()

        # Vérifier qu'on a une position
        if ticker not in self.portfolio['positions'] or self.portfolio['positions'][ticker] <= 0:
            return {
                "success": False,
                "error": f"Aucune position ouverte sur {ticker}"
            }

        # Vérifier que le take-profit est cohérent
        try:
            current_price = self._get_current_price(ticker)
            
            # Pour une position longue, le TP doit être au-dessus du prix actuel
            if take_profit_price <= current_price:
                return {
                    "success": False,
                    "error": f"Le take-profit ({take_profit_price}) doit être SUPÉRIEUR au prix actuel ({current_price})"
                }

        except ValueError as e:
            return {
                "success": False,
                "error": str(e)
            }

        # Initialiser le dict si nécessaire
        if 'take_profit_orders' not in self.portfolio:
            self.portfolio['take_profit_orders'] = {}

        # Enregistrer le take-profit
        self.portfolio['take_profit_orders'][ticker] = take_profit_price
        self._save_portfolio()

        return {
            "success": True,
            "ticker": ticker,
            "take_profit_price": take_profit_price,
            "current_price": current_price,
            "message": f"Take-profit placé sur {ticker} à ${take_profit_price:.2f} (prix actuel: ${current_price:.2f})"
        }

    def check_and_execute_orders(self) -> List[Dict]:
        """
        Vérifie et exécute les ordres stop-loss et take-profit
        À appeler régulièrement pour simuler l'exécution automatique

        Returns:
            Liste des ordres exécutés
        """
        executed_orders = []

        # Initialiser les dicts si nécessaires
        if 'stop_loss_orders' not in self.portfolio:
            self.portfolio['stop_loss_orders'] = {}
        if 'take_profit_orders' not in self.portfolio:
            self.portfolio['take_profit_orders'] = {}

        # Vérifier les stop-loss
        for ticker, stop_price in list(self.portfolio['stop_loss_orders'].items()):
            if ticker not in self.portfolio['positions']:
                # Position fermée, supprimer le stop
                del self.portfolio['stop_loss_orders'][ticker]
                continue

            try:
                current_price = self._get_current_price(ticker)
                
                # Si prix actuel <= stop-loss → VENDRE
                if current_price <= stop_price:
                    quantity = self.portfolio['positions'][ticker]
                    
                    # Vendre toute la position
                    result = self.place_order(ticker, -quantity, "stop_loss")
                    
                    if result.get('success'):
                        executed_orders.append({
                            "type": "STOP-LOSS",
                            "ticker": ticker,
                            "trigger_price": stop_price,
                            "execution_price": current_price,
                            "quantity": quantity,
                            "message": f"Stop-loss exécuté sur {ticker}: vendu {quantity} actions à ${current_price:.2f}"
                        })
                        
                        # Supprimer le stop-loss
                        del self.portfolio['stop_loss_orders'][ticker]
                        
                        # Supprimer le take-profit associé s'il existe
                        if ticker in self.portfolio.get('take_profit_orders', {}):
                            del self.portfolio['take_profit_orders'][ticker]

            except ValueError:
                continue

        # Vérifier les take-profit
        for ticker, tp_price in list(self.portfolio['take_profit_orders'].items()):
            if ticker not in self.portfolio['positions']:
                # Position fermée, supprimer le TP
                del self.portfolio['take_profit_orders'][ticker]
                continue

            try:
                current_price = self._get_current_price(ticker)
                
                # Si prix actuel >= take-profit → VENDRE
                if current_price >= tp_price:
                    quantity = self.portfolio['positions'][ticker]
                    
                    # Vendre toute la position
                    result = self.place_order(ticker, -quantity, "take_profit")
                    
                    if result.get('success'):
                        executed_orders.append({
                            "type": "TAKE-PROFIT",
                            "ticker": ticker,
                            "trigger_price": tp_price,
                            "execution_price": current_price,
                            "quantity": quantity,
                            "message": f"Take-profit exécuté sur {ticker}: vendu {quantity} actions à ${current_price:.2f}"
                        })
                        
                        # Supprimer le take-profit
                        del self.portfolio['take_profit_orders'][ticker]
                        
                        # Supprimer le stop-loss associé s'il existe
                        if ticker in self.portfolio.get('stop_loss_orders', {}):
                            del self.portfolio['stop_loss_orders'][ticker]

            except ValueError:
                continue

        if executed_orders:
            self._save_portfolio()

        return executed_orders

    def get_active_orders(self) -> Dict:
        """
        Récupère tous les ordres actifs (stop-loss et take-profit)

        Returns:
            Dictionnaire des ordres actifs
        """
        orders = {
            "stop_loss": [],
            "take_profit": []
        }

        # Initialiser si nécessaire
        if 'stop_loss_orders' not in self.portfolio:
            self.portfolio['stop_loss_orders'] = {}
        if 'take_profit_orders' not in self.portfolio:
            self.portfolio['take_profit_orders'] = {}

        # Stop-loss
        for ticker, stop_price in self.portfolio['stop_loss_orders'].items():
            try:
                current_price = self._get_current_price(ticker)
                distance_percent = ((current_price - stop_price) / current_price) * 100
                
                orders["stop_loss"].append({
                    "ticker": ticker,
                    "stop_price": stop_price,
                    "current_price": current_price,
                    "distance_percent": round(distance_percent, 2)
                })
            except ValueError:
                continue

        # Take-profit
        for ticker, tp_price in self.portfolio['take_profit_orders'].items():
            try:
                current_price = self._get_current_price(ticker)
                distance_percent = ((tp_price - current_price) / current_price) * 100
                
                orders["take_profit"].append({
                    "ticker": ticker,
                    "take_profit_price": tp_price,
                    "current_price": current_price,
                    "distance_percent": round(distance_percent, 2)
                })
            except ValueError:
                continue

        return orders

    def cancel_order(self, ticker: str, order_type: str = "both") -> Dict:
        """
        Annule un ordre stop-loss ou take-profit

        Args:
            ticker: Symbole de l'action
            order_type: "stop_loss", "take_profit", ou "both"

        Returns:
            Confirmation
        """
        ticker = ticker.upper()
        canceled = []

        # Initialiser si nécessaire
        if 'stop_loss_orders' not in self.portfolio:
            self.portfolio['stop_loss_orders'] = {}
        if 'take_profit_orders' not in self.portfolio:
            self.portfolio['take_profit_orders'] = {}

        if order_type in ["stop_loss", "both"]:
            if ticker in self.portfolio['stop_loss_orders']:
                del self.portfolio['stop_loss_orders'][ticker]
                canceled.append("stop-loss")

        if order_type in ["take_profit", "both"]:
            if ticker in self.portfolio['take_profit_orders']:
                del self.portfolio['take_profit_orders'][ticker]
                canceled.append("take-profit")

        if canceled:
            self._save_portfolio()
            return {
                "success": True,
                "ticker": ticker,
                "canceled": canceled,
                "message": f"Ordres annulés pour {ticker}: {', '.join(canceled)}"
            }
        else:
            return {
                "success": False,
                "error": f"Aucun ordre trouvé pour {ticker}"
            }
