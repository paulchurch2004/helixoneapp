"""
Cost Models - Modèles de coûts de trading réalistes

Modélise:
1. Commissions (fixe + variable)
2. Slippage (écart bid-ask)
3. Market impact (gros ordres)
4. Coûts de financement (overnight)

But: Backtests réalistes, pas trop optimistes
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TradingCosts:
    """
    Calcule les coûts de trading réalistes
    """

    def __init__(
        self,
        commission_pct: float = 0.001,  # 0.1% par trade (broker low-cost)
        commission_min: float = 1.0,  # Commission minimum $1
        slippage_pct: float = 0.0005,  # 0.05% de slippage moyen
        market_impact_threshold: float = 10000,  # Au-dessus de $10k, impact
        market_impact_pct: float = 0.0002  # 0.02% d'impact additionnel
    ):
        """
        Args:
            commission_pct: Commission en % de la valeur
            commission_min: Commission minimum en $
            slippage_pct: Slippage moyen en %
            market_impact_threshold: Seuil pour market impact
            market_impact_pct: Impact additionnel pour gros ordres
        """
        self.commission_pct = commission_pct
        self.commission_min = commission_min
        self.slippage_pct = slippage_pct
        self.market_impact_threshold = market_impact_threshold
        self.market_impact_pct = market_impact_pct

    def calculate_commission(self, trade_value: float) -> float:
        """
        Calcule la commission

        Args:
            trade_value: Valeur du trade en $

        Returns:
            Commission en $
        """
        commission = trade_value * self.commission_pct
        return max(commission, self.commission_min)

    def calculate_slippage(self, price: float, quantity: int) -> float:
        """
        Calcule le slippage

        Args:
            price: Prix de l'action
            quantity: Nombre d'actions

        Returns:
            Slippage en $ par action
        """
        # Slippage de base
        base_slippage = price * self.slippage_pct

        # Market impact pour gros ordres
        trade_value = price * quantity
        if trade_value > self.market_impact_threshold:
            market_impact = price * self.market_impact_pct
            return base_slippage + market_impact

        return base_slippage

    def calculate_total_cost(
        self,
        price: float,
        quantity: int,
        side: str = 'buy'
    ) -> Dict[str, float]:
        """
        Calcule le coût total d'un trade

        Args:
            price: Prix de l'action
            quantity: Quantité
            side: 'buy' ou 'sell'

        Returns:
            Dict avec breakdown des coûts
        """
        trade_value = price * quantity

        # Commission
        commission = self.calculate_commission(trade_value)

        # Slippage
        slippage_per_share = self.calculate_slippage(price, quantity)
        total_slippage = slippage_per_share * quantity

        # Prix effectif
        if side == 'buy':
            effective_price = price + slippage_per_share
        else:
            effective_price = price - slippage_per_share

        # Coût total
        total_cost = commission + total_slippage

        # Coût en %
        cost_pct = (total_cost / trade_value) * 100

        return {
            'commission': commission,
            'slippage': total_slippage,
            'slippage_per_share': slippage_per_share,
            'total_cost': total_cost,
            'cost_pct': cost_pct,
            'effective_price': effective_price,
            'trade_value': trade_value
        }

    def get_summary(self) -> str:
        """Retourne un résumé des paramètres de coût"""
        summary = f"""
Trading Cost Model:
  Commission: {self.commission_pct*100:.2f}% (min ${self.commission_min})
  Slippage: {self.slippage_pct*100:.3f}%
  Market impact threshold: ${self.market_impact_threshold:,.0f}
  Market impact: {self.market_impact_pct*100:.3f}%
        """
        return summary


# ============================================================================
# PRESETS
# ============================================================================

class CostPresets:
    """Presets de coûts pour différents types de brokers"""

    @staticmethod
    def retail_broker() -> TradingCosts:
        """
        Broker retail typique (Interactive Brokers, TD Ameritrade)

        - Commissions: 0.1%
        - Slippage: 0.05%
        - Market impact: faible
        """
        return TradingCosts(
            commission_pct=0.001,
            commission_min=1.0,
            slippage_pct=0.0005,
            market_impact_threshold=10000,
            market_impact_pct=0.0002
        )

    @staticmethod
    def low_cost_broker() -> TradingCosts:
        """
        Broker low-cost (Robinhood, Webull)

        - Pas de commission fixe
        - Slippage: 0.1% (moins bon execution)
        - Market impact: moyen
        """
        return TradingCosts(
            commission_pct=0.0,
            commission_min=0.0,
            slippage_pct=0.001,
            market_impact_threshold=5000,
            market_impact_pct=0.0005
        )

    @staticmethod
    def institutional() -> TradingCosts:
        """
        Broker institutionnel

        - Commissions: 0.01%
        - Slippage: 0.02% (meilleur execution)
        - Market impact: élevé pour gros ordres
        """
        return TradingCosts(
            commission_pct=0.0001,
            commission_min=10.0,
            slippage_pct=0.0002,
            market_impact_threshold=50000,
            market_impact_pct=0.001
        )

    @staticmethod
    def aggressive() -> TradingCosts:
        """
        Modèle conservateur/pessimiste

        - Coûts élevés pour backtest réaliste
        """
        return TradingCosts(
            commission_pct=0.002,
            commission_min=2.0,
            slippage_pct=0.001,
            market_impact_threshold=5000,
            market_impact_pct=0.0008
        )


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TRADING COST MODELS - Test")
    print("=" * 80)

    # Test 1: Retail broker
    print("\n📊 TEST 1: Retail Broker")
    print("=" * 80)

    costs_retail = CostPresets.retail_broker()
    print(costs_retail.get_summary())

    # Exemple trade: acheter 100 actions AAPL à $150
    price = 150.0
    quantity = 100

    breakdown = costs_retail.calculate_total_cost(price, quantity, side='buy')

    print("\nExemple trade: BUY 100 AAPL @ $150")
    print(f"  Trade value: ${breakdown['trade_value']:,.2f}")
    print(f"  Commission: ${breakdown['commission']:.2f}")
    print(f"  Slippage: ${breakdown['slippage']:.2f} (${breakdown['slippage_per_share']:.4f}/share)")
    print(f"  Total cost: ${breakdown['total_cost']:.2f} ({breakdown['cost_pct']:.3f}%)")
    print(f"  Effective price: ${breakdown['effective_price']:.2f}")

    # Test 2: Low-cost broker
    print("\n\n📊 TEST 2: Low-Cost Broker")
    print("=" * 80)

    costs_lowcost = CostPresets.low_cost_broker()
    print(costs_lowcost.get_summary())

    breakdown = costs_lowcost.calculate_total_cost(price, quantity, side='buy')

    print("\nMême trade avec low-cost broker:")
    print(f"  Total cost: ${breakdown['total_cost']:.2f} ({breakdown['cost_pct']:.3f}%)")
    print(f"  Effective price: ${breakdown['effective_price']:.2f}")

    # Test 3: Gros ordre (market impact)
    print("\n\n📊 TEST 3: Gros Ordre (Market Impact)")
    print("=" * 80)

    large_quantity = 1000  # 1000 actions = $150k
    breakdown_large = costs_retail.calculate_total_cost(price, large_quantity, side='buy')

    print(f"\nGros ordre: BUY 1000 AAPL @ $150 (${price * large_quantity:,.0f})")
    print(f"  Commission: ${breakdown_large['commission']:.2f}")
    print(f"  Slippage (avec market impact): ${breakdown_large['slippage']:.2f}")
    print(f"  Total cost: ${breakdown_large['total_cost']:.2f} ({breakdown_large['cost_pct']:.3f}%)")
    print(f"  Effective price: ${breakdown_large['effective_price']:.2f}")

    # Comparaison
    print("\n\n📊 COMPARAISON DES BROKERS")
    print("=" * 80)
    print(f"{'Broker':<20} {'Commission':<15} {'Slippage':<15} {'Total':<15} {'Cost %':<10}")
    print("-" * 80)

    for name, preset in [
        ('Retail', CostPresets.retail_broker()),
        ('Low-Cost', CostPresets.low_cost_broker()),
        ('Institutional', CostPresets.institutional()),
        ('Aggressive', CostPresets.aggressive())
    ]:
        bd = preset.calculate_total_cost(price, quantity, side='buy')
        print(f"{name:<20} ${bd['commission']:<14.2f} ${bd['slippage']:<14.2f} "
              f"${bd['total_cost']:<14.2f} {bd['cost_pct']:<9.3f}%")

    print("\n✅ Tests terminés!")
