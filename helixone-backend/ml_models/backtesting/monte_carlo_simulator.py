"""
Monte Carlo Simulator - Simulation de trajectoires futures

Utilise:
1. Historique de returns pour estimer distribution
2. Génère 10,000+ trajectoires possibles
3. Calcule statistiques: VaR, CVaR, probabilités

But: Quantifier l'incertitude, risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Simulateur Monte Carlo pour trajectoires de portfolio
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        forecast_days: int = 252,  # 1 an
        random_seed: Optional[int] = None
    ):
        """
        Args:
            n_simulations: Nombre de simulations
            forecast_days: Jours à projeter
            random_seed: Seed pour reproductibilité
        """
        self.n_simulations = n_simulations
        self.forecast_days = forecast_days
        self.random_seed = random_seed

        if random_seed:
            np.random.seed(random_seed)

        logger.info(f"MonteCarloSimulator initialisé")
        logger.info(f"   Simulations: {n_simulations:,}")
        logger.info(f"   Horizon: {forecast_days} jours")

    def estimate_parameters(
        self,
        returns: pd.Series,
        method: str = 'historical'
    ) -> Dict:
        """
        Estime les paramètres de la distribution

        Args:
            returns: Historique de returns
            method: 'historical', 'normal', ou 't-student'

        Returns:
            Dict avec paramètres (mean, std, etc.)
        """
        logger.info(f"Estimation paramètres (method={method})...")

        params = {
            'method': method,
            'mean': returns.mean(),
            'std': returns.std(),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis()
        }

        if method == 't-student':
            # Fit distribution t-student (fat tails)
            df, loc, scale = stats.t.fit(returns)
            params['df'] = df
            params['loc'] = loc
            params['scale'] = scale

        logger.info(f"   Mean return: {params['mean']*100:.4f}% par jour")
        logger.info(f"   Std: {params['std']*100:.4f}%")
        logger.info(f"   Skew: {params['skew']:.2f}")
        logger.info(f"   Kurtosis: {params['kurtosis']:.2f}")

        return params

    def generate_scenarios(
        self,
        initial_value: float,
        params: Dict
    ) -> np.ndarray:
        """
        Génère les scénarios Monte Carlo

        Args:
            initial_value: Valeur initiale du portfolio
            params: Paramètres de distribution

        Returns:
            Array (n_simulations, forecast_days) avec trajectoires
        """
        logger.info(f"Génération de {self.n_simulations:,} scénarios...")

        method = params['method']

        # Générer returns aléatoires
        if method == 'historical':
            # Bootstrap historique
            scenarios_returns = np.random.choice(
                params['returns_array'],
                size=(self.n_simulations, self.forecast_days),
                replace=True
            )

        elif method == 'normal':
            # Distribution normale
            scenarios_returns = np.random.normal(
                loc=params['mean'],
                scale=params['std'],
                size=(self.n_simulations, self.forecast_days)
            )

        elif method == 't-student':
            # Distribution t-student (fat tails)
            scenarios_returns = stats.t.rvs(
                df=params['df'],
                loc=params['loc'],
                scale=params['scale'],
                size=(self.n_simulations, self.forecast_days)
            )

        else:
            raise ValueError(f"Méthode inconnue: {method}")

        # Convertir returns en prix
        scenarios_prices = initial_value * (1 + scenarios_returns).cumprod(axis=1)

        # Ajouter colonne initiale
        scenarios_prices = np.column_stack([
            np.full(self.n_simulations, initial_value),
            scenarios_prices
        ])

        logger.info(f"   ✅ Scénarios générés: shape={scenarios_prices.shape}")

        return scenarios_prices

    def calculate_statistics(
        self,
        scenarios: np.ndarray,
        initial_value: float
    ) -> Dict:
        """
        Calcule les statistiques des scénarios

        Args:
            scenarios: Array de scénarios
            initial_value: Valeur initiale

        Returns:
            Dict avec statistiques
        """
        logger.info("Calcul statistiques...")

        # Valeurs finales
        final_values = scenarios[:, -1]

        # Returns finaux
        final_returns = (final_values / initial_value - 1) * 100

        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(final_values, percentiles)
        percentile_returns = np.percentile(final_returns, percentiles)

        # VaR & CVaR
        var_95 = np.percentile(final_returns, 5)
        cvar_95 = final_returns[final_returns <= var_95].mean()

        var_99 = np.percentile(final_returns, 1)
        cvar_99 = final_returns[final_returns <= var_99].mean()

        # Probabilités
        prob_profit = (final_returns > 0).mean() * 100
        prob_loss_10 = (final_returns < -10).mean() * 100
        prob_gain_20 = (final_returns > 20).mean() * 100

        stats_dict = {
            'initial_value': initial_value,
            'n_scenarios': len(scenarios),
            'forecast_days': scenarios.shape[1] - 1,

            # Statistiques finales
            'mean_final_value': final_values.mean(),
            'std_final_value': final_values.std(),
            'min_final_value': final_values.min(),
            'max_final_value': final_values.max(),
            'median_final_value': np.median(final_values),

            # Returns
            'mean_return': final_returns.mean(),
            'std_return': final_returns.std(),
            'min_return': final_returns.min(),
            'max_return': final_returns.max(),
            'median_return': np.median(final_returns),

            # Percentiles
            'percentiles': {
                f'p{p}': {'value': v, 'return': r}
                for p, v, r in zip(percentiles, percentile_values, percentile_returns)
            },

            # VaR & CVaR
            'var_95': var_95,
            'cvar_95': cvar_95,
            'var_99': var_99,
            'cvar_99': cvar_99,

            # Probabilités
            'prob_profit': prob_profit,
            'prob_loss_10pct': prob_loss_10,
            'prob_gain_20pct': prob_gain_20
        }

        return stats_dict

    def run_simulation(
        self,
        returns: pd.Series,
        initial_value: float = 100000,
        method: str = 'historical'
    ) -> Dict:
        """
        Exécute la simulation complète

        Args:
            returns: Historique de returns
            initial_value: Valeur initiale du portfolio
            method: Méthode de simulation

        Returns:
            Dict avec scénarios et statistiques
        """
        logger.info("=" * 80)
        logger.info("MONTE CARLO SIMULATION")
        logger.info("=" * 80)
        logger.info(f"   Valeur initiale: ${initial_value:,.0f}")
        logger.info(f"   Historique: {len(returns)} jours")
        logger.info(f"   Méthode: {method}")

        # 1. Estimer paramètres
        params = self.estimate_parameters(returns, method=method)

        # Ajouter returns array pour bootstrap
        if method == 'historical':
            params['returns_array'] = returns.values

        # 2. Générer scénarios
        scenarios = self.generate_scenarios(initial_value, params)

        # 3. Calculer statistiques
        stats = self.calculate_statistics(scenarios, initial_value)

        # 4. Résultats
        results = {
            'scenarios': scenarios,
            'statistics': stats,
            'parameters': params
        }

        # Afficher résumé
        self.print_summary(stats)

        return results

    def print_summary(self, stats: Dict):
        """Affiche un résumé des résultats"""
        print("\n" + "=" * 80)
        print("📊 RÉSULTATS MONTE CARLO")
        print("=" * 80)

        print(f"\n📈 VALEUR FINALE (après {stats['forecast_days']} jours):")
        print(f"   Moyenne: ${stats['mean_final_value']:,.0f} ({stats['mean_return']:+.2f}%)")
        print(f"   Médiane: ${stats['median_final_value']:,.0f} ({stats['median_return']:+.2f}%)")
        print(f"   Écart-type: ${stats['std_final_value']:,.0f}")
        print(f"   Min: ${stats['min_final_value']:,.0f} ({stats['min_return']:+.2f}%)")
        print(f"   Max: ${stats['max_final_value']:,.0f} ({stats['max_return']:+.2f}%)")

        print(f"\n📊 PERCENTILES:")
        for p in [5, 25, 50, 75, 95]:
            p_data = stats['percentiles'][f'p{p}']
            print(f"   {p}%: ${p_data['value']:,.0f} ({p_data['return']:+.2f}%)")

        print(f"\n⚠️  RISQUE:")
        print(f"   VaR (95%): {stats['var_95']:+.2f}%")
        print(f"   CVaR (95%): {stats['cvar_95']:+.2f}%")
        print(f"   VaR (99%): {stats['var_99']:+.2f}%")
        print(f"   CVaR (99%): {stats['cvar_99']:+.2f}%")

        print(f"\n🎲 PROBABILITÉS:")
        print(f"   Probabilité de profit: {stats['prob_profit']:.1f}%")
        print(f"   Probabilité perte >10%: {stats['prob_loss_10pct']:.1f}%")
        print(f"   Probabilité gain >20%: {stats['prob_gain_20pct']:.1f}%")

        print("=" * 80)


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("MONTE CARLO SIMULATOR - Test")
    print("=" * 80)

    # Créer données synthétiques
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

    # Returns avec drift positif + volatilité
    daily_returns = pd.Series(
        np.random.randn(len(dates)) * 0.015 + 0.0005,  # ~12% CAGR, 15% vol
        index=dates
    )

    print(f"\n📊 Historique de returns:")
    print(f"   {len(daily_returns)} jours")
    print(f"   Mean: {daily_returns.mean()*100:.4f}% par jour")
    print(f"   Std: {daily_returns.std()*100:.4f}%")
    print(f"   Sharpe (ann): {(daily_returns.mean() / daily_returns.std()) * np.sqrt(252):.2f}")

    # Test 1: Historical bootstrap
    print("\n\n" + "=" * 80)
    print("TEST 1: HISTORICAL BOOTSTRAP")
    print("=" * 80)

    sim = MonteCarloSimulator(n_simulations=10000, forecast_days=252, random_seed=42)
    results = sim.run_simulation(
        returns=daily_returns,
        initial_value=100000,
        method='historical'
    )

    # Test 2: Normal distribution
    print("\n\n" + "=" * 80)
    print("TEST 2: NORMAL DISTRIBUTION")
    print("=" * 80)

    sim_normal = MonteCarloSimulator(n_simulations=10000, forecast_days=252, random_seed=42)
    results_normal = sim_normal.run_simulation(
        returns=daily_returns,
        initial_value=100000,
        method='normal'
    )

    # Comparaison
    print("\n\n" + "=" * 80)
    print("COMPARAISON HISTORICAL vs NORMAL")
    print("=" * 80)

    print(f"\n{'Métrique':<30} {'Historical':<15} {'Normal':<15}")
    print("-" * 60)
    print(f"{'Mean final value':<30} ${results['statistics']['mean_final_value']:>13,.0f} "
          f"${results_normal['statistics']['mean_final_value']:>13,.0f}")
    print(f"{'Median final value':<30} ${results['statistics']['median_final_value']:>13,.0f} "
          f"${results_normal['statistics']['median_final_value']:>13,.0f}")
    print(f"{'VaR (95%)':<30} {results['statistics']['var_95']:>13.2f}% "
          f"{results_normal['statistics']['var_95']:>13.2f}%")
    print(f"{'Prob profit':<30} {results['statistics']['prob_profit']:>13.1f}% "
          f"{results_normal['statistics']['prob_profit']:>13.1f}%")

    print("\n✅ Tests terminés!")
