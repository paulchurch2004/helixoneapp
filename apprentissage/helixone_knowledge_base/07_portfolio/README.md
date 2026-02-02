# 📙 MODULE 7: CONSTRUCTION DE PORTEFEUILLE
## Théorie Moderne du Portefeuille et Extensions

---

## 📚 SOURCES PRINCIPALES
- **Markowitz (1952)**: https://www.math.hkust.edu.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf
- **Columbia - Mean-Variance & CAPM**: https://www.columbia.edu/~mh2078/FoundationsFE/MeanVariance-CAPM.pdf
- **HKUST Portfolio Optimization**: https://palomar.home.ece.ust.hk/ELEC5470_lectures/slides_portfolio_optim.pdf
- **Boyd - Convex Optimization**: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

---

## 🎯 OBJECTIFS
1. Maîtriser l'optimisation Mean-Variance de Markowitz
2. Comprendre les extensions (Black-Litterman, Risk Parity)
3. Implémenter des contraintes pratiques
4. Gérer l'estimation des inputs

---

## 🔑 THÉORIE MEAN-VARIANCE (MARKOWITZ)

### 1. Formulation du Problème

**Notation**:
- w ∈ ℝⁿ: vecteur des poids du portefeuille
- μ ∈ ℝⁿ: rendements espérés
- Σ ∈ ℝⁿˣⁿ: matrice de covariance

**Rendement du portefeuille**: r_p = w'μ
**Variance du portefeuille**: σ²_p = w'Σw

### 2. Problème d'Optimisation

**Minimiser la variance pour un rendement cible**:
```
min   ½ w'Σw
s.t.  w'μ = r_target
      w'𝟙 = 1
```

**Maximiser le rendement pour une variance cible**:
```
max   w'μ
s.t.  w'Σw ≤ σ²_target
      w'𝟙 = 1
```

**Maximiser le ratio de Sharpe**:
```
max   (w'μ - r_f) / √(w'Σw)
s.t.  w'𝟙 = 1
```

### 3. Solution Analytique (sans contraintes de positivité)

**Portefeuille tangent** (max Sharpe):
```
w* = Σ⁻¹(μ - r_f𝟙) / (𝟙'Σ⁻¹(μ - r_f𝟙))
```

**Frontière efficiente** (two-fund separation):
```
w(γ) = w_min_var + γ(w_tangent - w_min_var)
```

---

## 💻 IMPLÉMENTATION PYTHON

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp

class MeanVarianceOptimizer:
    """
    Mean-Variance Portfolio Optimization
    """
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.02):
        self.mu = np.array(expected_returns)
        self.Sigma = np.array(cov_matrix)
        self.rf = risk_free_rate
        self.n_assets = len(self.mu)
    
    def minimum_variance(self, allow_short=False):
        """
        Minimum variance portfolio
        """
        w = cp.Variable(self.n_assets)
        
        objective = cp.Minimize(cp.quad_form(w, self.Sigma))
        constraints = [cp.sum(w) == 1]
        
        if not allow_short:
            constraints.append(w >= 0)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value
    
    def max_sharpe(self, allow_short=False):
        """
        Maximum Sharpe ratio portfolio
        """
        # Reformulate as convex problem
        y = cp.Variable(self.n_assets)
        kappa = cp.Variable()
        
        objective = cp.Minimize(cp.quad_form(y, self.Sigma))
        constraints = [
            (self.mu - self.rf) @ y == 1,
            cp.sum(y) == kappa,
            kappa >= 0
        ]
        
        if not allow_short:
            constraints.append(y >= 0)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        w = y.value / kappa.value
        return w
    
    def efficient_frontier(self, n_points=50, allow_short=False):
        """
        Compute efficient frontier
        """
        # Find min and max returns on frontier
        w_min_var = self.minimum_variance(allow_short)
        ret_min = self.mu @ w_min_var
        
        w_max_ret = self.max_return(allow_short)
        ret_max = self.mu @ w_max_ret
        
        target_returns = np.linspace(ret_min, ret_max, n_points)
        
        frontier = []
        for target in target_returns:
            w = self.efficient_return(target, allow_short)
            if w is not None:
                vol = np.sqrt(w @ self.Sigma @ w)
                ret = self.mu @ w
                sharpe = (ret - self.rf) / vol
                frontier.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': w
                })
        
        return pd.DataFrame(frontier)
    
    def efficient_return(self, target_return, allow_short=False):
        """
        Minimum variance for target return
        """
        w = cp.Variable(self.n_assets)
        
        objective = cp.Minimize(cp.quad_form(w, self.Sigma))
        constraints = [
            cp.sum(w) == 1,
            self.mu @ w >= target_return
        ]
        
        if not allow_short:
            constraints.append(w >= 0)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            return w.value
        return None
    
    def max_return(self, allow_short=False):
        """Maximum return portfolio"""
        if allow_short:
            # Unbounded if shorts allowed
            return None
        else:
            w = np.zeros(self.n_assets)
            w[np.argmax(self.mu)] = 1.0
            return w
    
    def with_constraints(self, target_return=None, target_risk=None,
                         min_weight=0, max_weight=1,
                         sector_constraints=None):
        """
        Optimization with practical constraints
        """
        w = cp.Variable(self.n_assets)
        
        # Base constraints
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight
        ]
        
        if target_return is not None:
            constraints.append(self.mu @ w >= target_return)
        
        if target_risk is not None:
            constraints.append(cp.quad_form(w, self.Sigma) <= target_risk**2)
        
        # Sector constraints (e.g., max 30% in tech)
        if sector_constraints is not None:
            for sector_indices, max_exposure in sector_constraints:
                constraints.append(cp.sum(w[sector_indices]) <= max_exposure)
        
        # Objective: maximize return - lambda * variance
        risk_aversion = 1.0
        objective = cp.Maximize(self.mu @ w - risk_aversion * cp.quad_form(w, self.Sigma))
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return w.value


class BlackLittermanModel:
    """
    Black-Litterman Model for combining market equilibrium with views
    """
    def __init__(self, cov_matrix, market_caps, risk_aversion=2.5, tau=0.05):
        self.Sigma = np.array(cov_matrix)
        self.market_caps = np.array(market_caps)
        self.delta = risk_aversion
        self.tau = tau
        self.n_assets = len(market_caps)
        
        # Market weights
        self.w_mkt = self.market_caps / self.market_caps.sum()
        
        # Implied equilibrium returns (reverse optimization)
        self.pi = self.delta * self.Sigma @ self.w_mkt
    
    def add_views(self, P, Q, omega=None):
        """
        Add investor views
        
        P: (k x n) pick matrix
        Q: (k,) view returns
        omega: (k x k) uncertainty matrix (if None, use He-Litterman)
        """
        P = np.array(P)
        Q = np.array(Q)
        
        if omega is None:
            # He-Litterman: omega proportional to P'ΣP
            omega = np.diag(np.diag(self.tau * P @ self.Sigma @ P.T))
        
        # Posterior mean (Black-Litterman formula)
        tau_Sigma = self.tau * self.Sigma
        M = np.linalg.inv(
            np.linalg.inv(tau_Sigma) + P.T @ np.linalg.inv(omega) @ P
        )
        
        self.mu_bl = M @ (
            np.linalg.inv(tau_Sigma) @ self.pi + 
            P.T @ np.linalg.inv(omega) @ Q
        )
        
        # Posterior covariance
        self.Sigma_bl = self.Sigma + M
        
        return self.mu_bl, self.Sigma_bl
    
    def optimize(self, allow_short=False):
        """
        Optimize using BL expected returns
        """
        optimizer = MeanVarianceOptimizer(
            self.mu_bl, self.Sigma_bl, risk_free_rate=0.02
        )
        return optimizer.max_sharpe(allow_short)


class RiskParity:
    """
    Risk Parity Portfolio Construction
    """
    def __init__(self, cov_matrix):
        self.Sigma = np.array(cov_matrix)
        self.n_assets = self.Sigma.shape[0]
    
    def optimize(self):
        """
        Equal risk contribution portfolio
        """
        def risk_contribution(w):
            # Portfolio volatility
            vol = np.sqrt(w @ self.Sigma @ w)
            # Marginal risk contribution
            mrc = self.Sigma @ w / vol
            # Risk contribution
            rc = w * mrc
            return rc
        
        def objective(w):
            rc = risk_contribution(w)
            target_rc = np.ones(self.n_assets) / self.n_assets
            # Minimize squared difference from equal RC
            return np.sum((rc - target_rc * np.sum(rc))**2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0.01, 1.0) for _ in range(self.n_assets)]
        
        # Initial guess (equal weight)
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            objective, w0, 
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        return result.x


# Example usage
if __name__ == "__main__":
    # Sample data (5 assets)
    np.random.seed(42)
    n_assets = 5
    
    # Expected returns (annualized)
    mu = np.array([0.10, 0.12, 0.08, 0.15, 0.07])
    
    # Correlation matrix
    corr = np.array([
        [1.0, 0.3, 0.2, 0.4, 0.1],
        [0.3, 1.0, 0.25, 0.5, 0.15],
        [0.2, 0.25, 1.0, 0.3, 0.4],
        [0.4, 0.5, 0.3, 1.0, 0.2],
        [0.1, 0.15, 0.4, 0.2, 1.0]
    ])
    
    # Volatilities (annualized)
    vols = np.array([0.15, 0.20, 0.12, 0.25, 0.10])
    
    # Covariance matrix
    Sigma = np.outer(vols, vols) * corr
    
    # Optimize
    optimizer = MeanVarianceOptimizer(mu, Sigma)
    
    # Minimum variance
    w_min_var = optimizer.minimum_variance(allow_short=False)
    print("Minimum Variance Portfolio:")
    print(f"Weights: {w_min_var.round(3)}")
    print(f"Return: {mu @ w_min_var:.2%}")
    print(f"Volatility: {np.sqrt(w_min_var @ Sigma @ w_min_var):.2%}")
    
    # Maximum Sharpe
    w_max_sharpe = optimizer.max_sharpe(allow_short=False)
    print("\nMaximum Sharpe Portfolio:")
    print(f"Weights: {w_max_sharpe.round(3)}")
    print(f"Return: {mu @ w_max_sharpe:.2%}")
    print(f"Volatility: {np.sqrt(w_max_sharpe @ Sigma @ w_max_sharpe):.2%}")
    
    # Risk Parity
    rp = RiskParity(Sigma)
    w_rp = rp.optimize()
    print("\nRisk Parity Portfolio:")
    print(f"Weights: {w_rp.round(3)}")
```

---

## 📊 ESTIMATION DES INPUTS

### Le Problème d'Estimation

L'optimisation MV est très sensible aux erreurs d'estimation de μ et Σ.

### Solutions

1. **Shrinkage de la covariance** (Ledoit-Wolf):
```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw.fit(returns)
Sigma_shrunk = lw.covariance_
```

2. **Bayes-Stein shrinkage** pour les rendements

3. **Contraintes de robustesse**:
```python
# Uncertainty sets for robust optimization
w = cp.Variable(n)
epsilon = 0.1  # Uncertainty radius

constraints.append(
    cp.norm(w @ Sigma_sqrt, 2) <= epsilon * target_return
)
```

---

## 🔗 RÉFÉRENCES
1. Markowitz, H. (1952). Portfolio Selection
2. Black, F. & Litterman, R. (1992). Global Portfolio Optimization
3. Maillard, S., Roncalli, T. & Teïletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios
