"""
Performance Metrics - Métriques de performance trading

Métriques calculées:
1. Return metrics: Total return, CAGR, max drawdown
2. Risk metrics: Volatility, VaR, CVaR
3. Risk-adjusted: Sharpe, Sortino, Calmar
4. Win rate, profit factor, etc.

Standards institutionnels
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calcule les métriques de performance trading
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Taux sans risque annuel (défaut 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(
        self,
        prices: pd.Series,
        log_returns: bool = False
    ) -> pd.Series:
        """
        Calcule les returns

        Args:
            prices: Série de prix
            log_returns: Si True, utilise log returns

        Returns:
            Série de returns
        """
        if log_returns:
            return np.log(prices / prices.shift(1))
        else:
            return prices.pct_change()

    def total_return(self, prices: pd.Series) -> float:
        """Return total en %"""
        return ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100

    def cagr(self, prices: pd.Series) -> float:
        """
        Compound Annual Growth Rate

        Formula: (Final/Initial)^(1/years) - 1
        """
        n_days = len(prices)
        years = n_days / 252  # Trading days

        if years == 0:
            return 0

        total_ret = prices.iloc[-1] / prices.iloc[0]
        return (total_ret ** (1 / years) - 1) * 100

    def volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Volatilité (écart-type des returns)

        Args:
            returns: Returns
            annualize: Si True, annualise (x sqrt(252))

        Returns:
            Volatilité en %
        """
        vol = returns.std()

        if annualize:
            vol *= np.sqrt(252)

        return vol * 100

    def max_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """
        Maximum drawdown

        Returns:
            Dict avec max_dd, max_dd_pct, duration, etc.
        """
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_pct = max_dd * 100

        # Date du max drawdown
        max_dd_date = drawdown.idxmin()

        # Duration (jours sous le max)
        dd_duration = 0
        if max_dd_date:
            # Trouver le peak avant le drawdown
            peak_date = running_max[:max_dd_date].idxmax()
            # Trouver la récupération après
            recovery_dates = prices[max_dd_date:][prices[max_dd_date:] >= prices[peak_date]]

            if len(recovery_dates) > 0:
                recovery_date = recovery_dates.index[0]
                dd_duration = (recovery_date - peak_date).days
            else:
                # Pas encore récupéré
                dd_duration = (prices.index[-1] - peak_date).days

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'max_dd_date': max_dd_date,
            'duration_days': dd_duration
        }

    def sharpe_ratio(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Sharpe Ratio = (Return - RiskFree) / Volatility

        Args:
            returns: Returns
            annualize: Si True, annualise

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe = excess_returns.mean() / returns.std()

        if annualize:
            sharpe *= np.sqrt(252)

        return sharpe

    def sortino_ratio(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Sortino Ratio = (Return - RiskFree) / Downside Deviation

        Comme Sharpe mais utilise seulement la volatilité à la baisse
        """
        excess_returns = returns - (self.risk_free_rate / 252)

        # Downside deviation (seulement returns négatifs)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return np.inf  # Pas de downside!

        downside_std = downside_returns.std()
        sortino = excess_returns.mean() / downside_std

        if annualize:
            sortino *= np.sqrt(252)

        return sortino

    def calmar_ratio(self, prices: pd.Series) -> float:
        """
        Calmar Ratio = CAGR / Max Drawdown

        Ratio return/risque (drawdown)
        """
        cagr_val = self.cagr(prices)
        max_dd = self.max_drawdown(prices)['max_drawdown_pct']

        if max_dd == 0:
            return np.inf

        return abs(cagr_val / max_dd)

    def var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Value at Risk (VaR)

        Args:
            returns: Returns
            confidence: Niveau de confiance (0.95 = 95%)

        Returns:
            VaR en % (ex: -5% = perte max attendue à 95% confiance)
        """
        return np.percentile(returns, (1 - confidence) * 100) * 100

    def cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Conditional Value at Risk (CVaR / Expected Shortfall)

        Perte moyenne dans les pires (1-confidence)% cas
        """
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var_threshold].mean()
        return cvar * 100

    def win_rate(self, returns: pd.Series) -> float:
        """
        Win rate = % de jours gagnants

        Returns:
            Win rate en %
        """
        winning_days = (returns > 0).sum()
        total_days = len(returns)

        return (winning_days / total_days) * 100

    def profit_factor(self, returns: pd.Series) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss

        Ratio gains/pertes
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return np.inf

        return gross_profit / gross_loss

    def calculate_all(
        self,
        prices: Optional[pd.Series] = None,
        returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calcule toutes les métriques

        Args:
            prices: Série de prix (equity curve)
            returns: Série de returns (si prices fourni, sera calculé)

        Returns:
            Dict avec toutes les métriques
        """
        if prices is None and returns is None:
            raise ValueError("Fournir prices ou returns")

        if returns is None:
            returns = self.calculate_returns(prices)

        # Calculer toutes les métriques
        metrics = {}

        # Return metrics
        if prices is not None:
            metrics['total_return'] = self.total_return(prices)
            metrics['cagr'] = self.cagr(prices)

            # Max drawdown
            dd = self.max_drawdown(prices)
            metrics.update({
                'max_drawdown': dd['max_drawdown_pct'],
                'max_dd_duration_days': dd['duration_days']
            })

            # Risk-adjusted
            metrics['calmar_ratio'] = self.calmar_ratio(prices)

        # Volatility
        metrics['volatility'] = self.volatility(returns)

        # Sharpe & Sortino
        metrics['sharpe_ratio'] = self.sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.sortino_ratio(returns)

        # Risk metrics
        metrics['var_95'] = self.var(returns, 0.95)
        metrics['cvar_95'] = self.cvar(returns, 0.95)

        # Win rate
        metrics['win_rate'] = self.win_rate(returns)
        metrics['profit_factor'] = self.profit_factor(returns)

        # Other stats
        metrics['best_day'] = returns.max() * 100
        metrics['worst_day'] = returns.min() * 100
        metrics['avg_return'] = returns.mean() * 100

        return metrics

    def print_summary(self, metrics: Dict):
        """Affiche un résumé formaté"""
        print("=" * 80)
        print("📊 PERFORMANCE METRICS")
        print("=" * 80)

        print("\n📈 RETURNS:")
        if 'total_return' in metrics:
            print(f"   Total Return: {metrics['total_return']:+.2f}%")
        if 'cagr' in metrics:
            print(f"   CAGR: {metrics['cagr']:+.2f}%")
        print(f"   Avg Daily Return: {metrics['avg_return']:+.4f}%")
        print(f"   Best Day: {metrics['best_day']:+.2f}%")
        print(f"   Worst Day: {metrics['worst_day']:+.2f}%")

        print("\n⚠️  RISK:")
        print(f"   Volatility (ann): {metrics['volatility']:.2f}%")
        if 'max_drawdown' in metrics:
            print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
            if 'max_dd_duration_days' in metrics:
                print(f"   Max DD Duration: {metrics['max_dd_duration_days']} days")
        print(f"   VaR (95%): {metrics['var_95']:.2f}%")
        print(f"   CVaR (95%): {metrics['cvar_95']:.2f}%")

        print("\n📊 RISK-ADJUSTED:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        if 'calmar_ratio' in metrics:
            print(f"   Calmar Ratio: {metrics['calmar_ratio']:.2f}")

        print("\n🎯 WIN STATS:")
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

        print("=" * 80)


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("PERFORMANCE METRICS - Test")
    print("=" * 80)

    # Créer données synthétiques (equity curve)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Simulate returns avec drift positif
    daily_returns = np.random.randn(len(dates)) * 0.015 + 0.0005  # ~12% CAGR, 15% vol
    cumulative_returns = (1 + daily_returns).cumprod()
    equity_curve = pd.Series(100000 * cumulative_returns, index=dates)

    print(f"\n📊 Equity Curve Synthétique:")
    print(f"   Période: {equity_curve.index[0]} → {equity_curve.index[-1]}")
    print(f"   Valeur initiale: ${equity_curve.iloc[0]:,.0f}")
    print(f"   Valeur finale: ${equity_curve.iloc[-1]:,.0f}")

    # Calculer métriques
    print("\n⚙️  Calcul des métriques...")
    calculator = PerformanceMetrics(risk_free_rate=0.02)

    metrics = calculator.calculate_all(prices=equity_curve)

    # Afficher résumé
    calculator.print_summary(metrics)

    # Test comparaison avec stratégie "buy & hold"
    print("\n\n" + "=" * 80)
    print("COMPARAISON: Stratégie ML vs Buy & Hold")
    print("=" * 80)

    # Buy & hold (juste une ligne droite avec même return total)
    bh_returns = np.ones(len(dates)) * 0.0003  # Return constant
    bh_equity = pd.Series(100000 * (1 + bh_returns).cumprod(), index=dates)

    print("\n📊 Buy & Hold:")
    metrics_bh = calculator.calculate_all(prices=bh_equity)
    print(f"   Total Return: {metrics_bh['total_return']:+.2f}%")
    print(f"   CAGR: {metrics_bh['cagr']:+.2f}%")
    print(f"   Sharpe: {metrics_bh['sharpe_ratio']:.2f}")
    print(f"   Max DD: {metrics_bh['max_drawdown']:.2f}%")

    print("\n📊 Stratégie ML:")
    print(f"   Total Return: {metrics['total_return']:+.2f}%")
    print(f"   CAGR: {metrics['cagr']:+.2f}%")
    print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {metrics['max_drawdown']:.2f}%")

    print("\n✅ Tests terminés!")
