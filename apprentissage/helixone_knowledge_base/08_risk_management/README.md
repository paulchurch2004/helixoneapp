# 📕 MODULE 8: GESTION DES RISQUES
## VaR, CVaR et Mesures de Risque Cohérentes

---

## 📚 SOURCES PRINCIPALES
- **Artzner et al. (1999)**: Coherent Measures of Risk
- **McNeil, Frey & Embrechts**: Quantitative Risk Management

---

## 🎯 OBJECTIFS
1. Comprendre VaR et ses limitations
2. Maîtriser CVaR (Expected Shortfall)
3. Implémenter le stress testing
4. Calculer le risque de portefeuille

---

## 🔑 CONCEPTS FONDAMENTAUX

### 1. Value at Risk (VaR)

**Définition**: VaR_α est la perte maximale avec probabilité (1-α)
```
P(Loss ≤ VaR_α) = α
```

Pour α = 99%, VaR₉₉ = perte qu'on ne dépasse que 1% du temps.

**Méthodes de calcul**:

#### Historique
```python
def historical_var(returns, alpha=0.99):
    """
    Historical VaR from return distribution
    """
    return -np.percentile(returns, 100 * (1 - alpha))
```

#### Paramétrique (Normal)
```python
from scipy.stats import norm

def parametric_var(mu, sigma, alpha=0.99):
    """
    Parametric VaR assuming normal distribution
    """
    return -(mu + sigma * norm.ppf(1 - alpha))
```

#### Monte Carlo
```python
def monte_carlo_var(returns, n_simulations=100000, alpha=0.99):
    """
    Monte Carlo VaR
    """
    mu = returns.mean()
    sigma = returns.std()
    
    # Generate simulations
    simulated_returns = np.random.normal(mu, sigma, n_simulations)
    
    return -np.percentile(simulated_returns, 100 * (1 - alpha))
```

### 2. Expected Shortfall (CVaR)

**Définition**: Moyenne des pertes au-delà du VaR
```
ES_α = E[Loss | Loss > VaR_α]
```

CVaR est une **mesure de risque cohérente** (contrairement à VaR).

```python
def expected_shortfall(returns, alpha=0.99):
    """
    Expected Shortfall (CVaR)
    """
    var = historical_var(returns, alpha)
    losses = -returns
    return losses[losses > var].mean()


def parametric_es(mu, sigma, alpha=0.99):
    """
    Parametric ES for normal distribution
    """
    z = norm.ppf(alpha)
    es = sigma * norm.pdf(z) / (1 - alpha) - mu
    return es
```

### 3. Propriétés des Mesures de Risque Cohérentes

Une mesure ρ est cohérente si:
1. **Monotonie**: Si X ≤ Y alors ρ(X) ≥ ρ(Y)
2. **Sous-additivité**: ρ(X+Y) ≤ ρ(X) + ρ(Y) (diversification)
3. **Homogénéité positive**: ρ(λX) = λρ(X) pour λ > 0
4. **Invariance par translation**: ρ(X+c) = ρ(X) - c

⚠️ **VaR n'est PAS sous-additif** → CVaR préféré en pratique

---

## 💻 IMPLÉMENTATION COMPLÈTE

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

class RiskManager:
    """
    Comprehensive risk management toolkit
    """
    def __init__(self, returns, confidence_level=0.99):
        self.returns = np.array(returns)
        self.alpha = confidence_level
        self.mu = returns.mean()
        self.sigma = returns.std()
    
    # ========== VaR Methods ==========
    
    def var_historical(self):
        """Historical simulation VaR"""
        return -np.percentile(self.returns, 100 * (1 - self.alpha))
    
    def var_parametric(self, distribution='normal'):
        """Parametric VaR"""
        if distribution == 'normal':
            return -(self.mu + self.sigma * stats.norm.ppf(1 - self.alpha))
        elif distribution == 't':
            # Fit Student-t
            params = stats.t.fit(self.returns)
            return -stats.t.ppf(1 - self.alpha, *params)
    
    def var_cornish_fisher(self):
        """
        Cornish-Fisher expansion for non-normal VaR
        Accounts for skewness and kurtosis
        """
        z = stats.norm.ppf(1 - self.alpha)
        s = stats.skew(self.returns)
        k = stats.kurtosis(self.returns)
        
        # Cornish-Fisher adjustment
        z_cf = (z + (z**2 - 1) * s / 6 + 
                (z**3 - 3*z) * (k - 3) / 24 - 
                (2*z**3 - 5*z) * s**2 / 36)
        
        return -(self.mu + self.sigma * z_cf)
    
    def var_monte_carlo(self, n_sims=100000, model='normal'):
        """Monte Carlo VaR with different models"""
        if model == 'normal':
            sims = np.random.normal(self.mu, self.sigma, n_sims)
        elif model == 'garch':
            sims = self._simulate_garch(n_sims)
        elif model == 't':
            params = stats.t.fit(self.returns)
            sims = stats.t.rvs(*params, size=n_sims)
        
        return -np.percentile(sims, 100 * (1 - self.alpha))
    
    # ========== CVaR Methods ==========
    
    def cvar_historical(self):
        """Historical CVaR (Expected Shortfall)"""
        var = self.var_historical()
        losses = -self.returns
        return losses[losses > var].mean()
    
    def cvar_parametric(self):
        """Parametric CVaR for normal distribution"""
        z = stats.norm.ppf(self.alpha)
        return self.sigma * stats.norm.pdf(z) / (1 - self.alpha) - self.mu
    
    # ========== Portfolio Risk ==========
    
    def portfolio_var(self, weights, cov_matrix, method='parametric'):
        """
        Portfolio VaR
        """
        portfolio_return = np.dot(weights, self.mu)
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        if method == 'parametric':
            return -(portfolio_return + portfolio_vol * stats.norm.ppf(1 - self.alpha))
        else:
            # Historical simulation
            portfolio_returns = self.returns @ weights
            return -np.percentile(portfolio_returns, 100 * (1 - self.alpha))
    
    def component_var(self, weights, cov_matrix):
        """
        Component VaR - contribution of each asset to total VaR
        """
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_var = cov_matrix @ weights / portfolio_vol * stats.norm.ppf(self.alpha)
        component_var = weights * marginal_var
        return component_var
    
    def incremental_var(self, weights, cov_matrix, asset_idx, delta_weight=0.01):
        """
        Incremental VaR - change in VaR from small position change
        """
        var_base = self.portfolio_var(weights, cov_matrix)
        
        weights_new = weights.copy()
        weights_new[asset_idx] += delta_weight
        weights_new /= weights_new.sum()  # Renormalize
        
        var_new = self.portfolio_var(weights_new, cov_matrix)
        return var_new - var_base
    
    # ========== Stress Testing ==========
    
    def stress_test(self, scenarios):
        """
        Stress testing with predefined scenarios
        
        scenarios: dict of {name: {asset: shock}}
        """
        results = {}
        for name, shocks in scenarios.items():
            shocked_returns = self.returns.copy()
            for asset, shock in shocks.items():
                shocked_returns[:, asset] += shock
            
            results[name] = {
                'var': -np.percentile(shocked_returns.sum(axis=1), 100 * (1 - self.alpha)),
                'cvar': self._compute_cvar(shocked_returns.sum(axis=1)),
                'max_loss': -shocked_returns.sum(axis=1).min()
            }
        return pd.DataFrame(results).T
    
    def historical_stress_scenarios(self):
        """
        Common historical stress scenarios
        """
        return {
            'market_crash_2008': {'equity': -0.40, 'credit': -0.20, 'rates': -0.02},
            'covid_2020': {'equity': -0.35, 'credit': -0.15, 'vol': 0.50},
            'dot_com_2000': {'equity': -0.45, 'tech': -0.70, 'rates': 0.01},
            'black_monday_1987': {'equity': -0.23, 'vol': 1.50},
            'ltcm_1998': {'credit': -0.25, 'em': -0.40, 'liquidity': -0.30},
        }
    
    # ========== Backtesting ==========
    
    def backtest_var(self, window=252):
        """
        Backtest VaR model using rolling window
        """
        n = len(self.returns)
        violations = []
        var_estimates = []
        
        for i in range(window, n):
            historical_window = self.returns[i-window:i]
            var = -np.percentile(historical_window, 100 * (1 - self.alpha))
            var_estimates.append(var)
            
            actual_loss = -self.returns[i]
            violations.append(actual_loss > var)
        
        violation_rate = np.mean(violations)
        expected_rate = 1 - self.alpha
        
        # Kupiec test (proportion of failures)
        n_violations = sum(violations)
        n_obs = len(violations)
        
        lr_pof = -2 * (np.log((1-expected_rate)**(n_obs-n_violations) * expected_rate**n_violations) -
                       np.log((1-violation_rate)**(n_obs-n_violations) * violation_rate**n_violations))
        
        p_value = 1 - stats.chi2.cdf(lr_pof, 1)
        
        return {
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_statistic': lr_pof,
            'p_value': p_value,
            'pass': p_value > 0.05
        }
    
    # ========== Helper Methods ==========
    
    def _simulate_garch(self, n_sims):
        """Simple GARCH(1,1) simulation"""
        omega = self.sigma**2 * 0.05
        alpha = 0.10
        beta = 0.85
        
        sims = np.zeros(n_sims)
        sigma2 = self.sigma**2
        
        for i in range(n_sims):
            z = np.random.normal()
            sims[i] = np.sqrt(sigma2) * z + self.mu
            sigma2 = omega + alpha * sims[i]**2 + beta * sigma2
        
        return sims
    
    def _compute_cvar(self, returns):
        var = -np.percentile(returns, 100 * (1 - self.alpha))
        losses = -returns
        return losses[losses > var].mean()


# Example usage
if __name__ == "__main__":
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 1000)  # Daily returns
    
    rm = RiskManager(returns, confidence_level=0.99)
    
    print("=== VaR Estimates ===")
    print(f"Historical VaR:      {rm.var_historical():.4f}")
    print(f"Parametric VaR:      {rm.var_parametric():.4f}")
    print(f"Cornish-Fisher VaR:  {rm.var_cornish_fisher():.4f}")
    print(f"Monte Carlo VaR:     {rm.var_monte_carlo():.4f}")
    
    print("\n=== CVaR Estimates ===")
    print(f"Historical CVaR:     {rm.cvar_historical():.4f}")
    print(f"Parametric CVaR:     {rm.cvar_parametric():.4f}")
    
    print("\n=== Backtest Results ===")
    backtest = rm.backtest_var()
    print(f"Violation Rate: {backtest['violation_rate']:.2%}")
    print(f"Expected Rate:  {backtest['expected_rate']:.2%}")
    print(f"Kupiec Test:    {'PASS' if backtest['pass'] else 'FAIL'}")
```

---

## 📊 COMPARAISON VaR vs CVaR

| Propriété | VaR | CVaR |
|-----------|-----|------|
| Interprétation | Seuil de perte | Perte moyenne au-delà du seuil |
| Sous-additif | ❌ Non | ✅ Oui |
| Tail risk | Ignore | Capture |
| Réglementaire | Bâle II/III | Bâle III (à partir de 2023) |
| Calcul | Simple | Plus complexe |

---

## 🔗 RÉFÉRENCES
1. Artzner, P. et al. (1999). Coherent Measures of Risk
2. Rockafellar, R.T. & Uryasev, S. (2002). CVaR Optimization
3. Hull, J. - Risk Management and Financial Institutions
