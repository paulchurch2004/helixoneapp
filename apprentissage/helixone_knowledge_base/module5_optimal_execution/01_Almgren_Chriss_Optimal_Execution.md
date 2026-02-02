# Almgren-Chriss: Optimal Execution of Portfolio Transactions

> **Source**: Almgren, R. and Chriss, N. (2000). "Optimal execution of portfolio transactions", Journal of Risk 3(2), 5-39
> **Aussi disponible sur**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=53501
> **Extrait pour la base de connaissances HelixOne**

---

## Abstract

We consider the execution of portfolio transactions with the aim of **minimizing a combination of volatility risk and transaction costs arising from permanent and temporary market impact**. For a simple linear cost model, we explicitly construct the **efficient frontier** in the space of time-dependent liquidation strategies, which have minimum expected cost for a given level of uncertainty. We may then select optimal strategies either by minimizing a quadratic utility function, or by minimizing Value at Risk. The latter choice leads to the concept of **Liquidity-adjusted VAR (L-VaR)**, that explicitly considers the best tradeoff between volatility risk and liquidation costs.

---

## 1. Introduction and Problem Statement

### The Optimal Execution Problem

**Goal**: Liquidate X shares of a stock before time T while minimizing:
- Transaction costs from market impact
- Volatility risk from price movements

**Key Trade-off**: 
- **Fast execution** → Higher transaction costs (more market impact)
- **Slow execution** → Higher volatility risk (more price uncertainty)

### Types of Trading Strategies

- **Static strategy**: Determined in advance of trading
- **Dynamic strategy**: Depends on the state of the market during execution

**Key Result**: In the Almgren-Chriss model, the statically optimal strategy is also dynamically optimal.

---

## 2. Model Setup

### 2.1 Notation and Definitions

**Time discretization:**
- Total time horizon: T
- Number of intervals: N  
- Interval length: τ = T/N
- Decision times: t_k = kτ for k = 0, 1, ..., N

**Trading trajectory and list:**
- **Holdings trajectory**: x_k = number of shares held at time t_k
- Initial condition: x_0 = X (total shares to liquidate)
- Terminal condition: x_N = 0 (fully liquidated)
- **Trade list**: n_k = x_{k-1} - x_k (shares sold in interval k)

### 2.2 Price Dynamics

**Unaffected price** (without our trades):
```
S̃_k = S_{k-1} + σ√τ · ξ_k
```

Where:
- σ = daily volatility
- ξ_k ~ N(0,1) i.i.d. random shocks

**Actual price** (with market impact):
```
S_k = S̃_k - g(n_k/τ)
```

### 2.3 Market Impact Functions

**Permanent Impact g(v):**
- Affects all future prices
- Linear model: g(v) = γ·v
- Where v = n_k/τ is the trading rate

**Temporary Impact h(v):**
- Affects only current trade's execution price
- Linear model: h(v) = ε·sign(v) + η·v
- Where ε = fixed cost, η = linear temporary impact

**Effective execution price:**
```
Ŝ_k = S_{k-1} - τ·g(n_k/τ) - h(n_k/τ)
     = S_{k-1} - γ·n_k - ε·sign(n_k) - η·(n_k/τ)
```

---

## 3. Cost of Trading

### 3.1 Implementation Shortfall

**Definition**: Difference between initial portfolio value and actual proceeds

```
Implementation Shortfall = X·S_0 - Σ_k n_k·Ŝ_k
```

### 3.2 Expected Cost and Variance

**Expected cost E[C]:**
```
E[C] = ½·γ·X² + ε·Σ_k |n_k| + η·Σ_k (n_k²/τ)
```

Components:
1. **Permanent impact cost**: ½·γ·X² (independent of strategy)
2. **Fixed transaction cost**: ε·Σ|n_k|
3. **Temporary impact cost**: η·Σ(n_k²/τ)

**Variance of cost V[C]:**
```
V[C] = σ²·Σ_k τ·x_k²
```

The variance depends on how long we hold shares (exposure to price risk).

---

## 4. Optimal Trading Strategies

### 4.1 Mean-Variance Optimization

**Objective**: Minimize
```
U(x) = E[C] + λ·V[C]
```

Where λ is the **risk aversion parameter**.

### 4.2 The Efficient Frontier

The efficient frontier represents all Pareto-optimal strategies:
- Minimum expected cost for a given variance
- Minimum variance for a given expected cost

**Shape**: Smooth, convex curve in (E[C], V[C]) space

**Minimum point**: The "naïve" strategy (equal-sized trades)
- Trading n_k = X/N at each interval
- Globally minimum expected cost
- Not minimum variance

### 4.3 Optimal Solution (Continuous-Time Limit)

**Optimal trading trajectory:**
```
x(t) = X · sinh(κ(T-t)) / sinh(κT)
```

**Optimal trading rate:**
```
v(t) = dx/dt = -X·κ · cosh(κ(T-t)) / sinh(κT)
```

Where **κ** is the **urgency parameter**:
```
κ = √(λσ² / η)
```

### 4.4 Half-Life of Optimal Trading

The parameter 1/κ has the interpretation of a **characteristic time scale** or "half-life" of liquidation.

**Key insight**: 
- Higher risk aversion (λ) → Faster execution (larger κ)
- Higher volatility (σ) → Faster execution
- Higher temporary impact (η) → Slower execution

---

## 5. Special Cases

### 5.1 Risk-Neutral Trader (λ = 0)

**Strategy**: Trade at constant rate (TWAP - Time Weighted Average Price)
```
n_k = X/N  for all k
```

**Minimizes**: Expected cost only (ignores variance)

### 5.2 Infinitely Risk-Averse (λ → ∞)

**Strategy**: Immediate liquidation (market order for all shares at t=0)
```
n_1 = X,  n_k = 0 for k > 1
```

**Minimizes**: Variance (eliminates price risk)

### 5.3 VWAP Strategy

Volume Weighted Average Price strategy:
- Trade proportionally to expected market volume
- Commonly used benchmark
- Not generally optimal in Almgren-Chriss framework

---

## 6. Value at Risk Approach

### 6.1 Liquidity-Adjusted VaR (L-VaR)

Instead of mean-variance, minimize:
```
L-VaR_α = E[C] + z_α·√V[C]
```

Where z_α is the α-quantile of the standard normal.

**Interpretation**: The maximum cost at confidence level α.

### 6.2 L-VaR Optimal Strategy

Same form as mean-variance optimal, with:
```
κ = z_α · σ / (2√η)
```

---

## 7. Extensions

### 7.1 Non-Linear Market Impact

**Power-law temporary impact:**
```
h(v) = η·|v|^β·sign(v)
```

Empirically, β ≈ 0.5 to 0.7 (concave for small v).

### 7.2 Stochastic Volatility and Liquidity

When σ or η vary stochastically:
- Optimal strategies become adaptive
- Requires dynamic programming approach

### 7.3 Parameter Uncertainty and Regime Shifts

**Regime shift model:**
- Parameters can change to one of p possible new states
- Pre-compute trajectories for each regime
- Reoptimize when shift occurs

---

## 8. Practical Implementation

### 8.1 Parameter Estimation

**Volatility σ:**
- Historical standard deviation of returns
- Or implied volatility from options

**Permanent impact γ:**
- Regression of price changes on signed volume
- Typical values: 0.01% to 0.1% per 1% of ADV

**Temporary impact η:**
- Estimated from trade execution data
- Square-root law often fits better than linear

### 8.2 Python Implementation Example

```python
import numpy as np

def almgren_chriss_trajectory(X, T, N, sigma, eta, lambda_):
    """
    Compute optimal liquidation trajectory.
    
    Parameters:
    - X: Initial shares to liquidate
    - T: Time horizon
    - N: Number of intervals
    - sigma: Volatility
    - eta: Temporary impact parameter
    - lambda_: Risk aversion
    
    Returns:
    - x: Holdings trajectory
    - n: Trade list
    """
    tau = T / N
    kappa = np.sqrt(lambda_ * sigma**2 / eta)
    
    # Time points
    t = np.linspace(0, T, N+1)
    
    # Optimal trajectory
    x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    
    # Trade list
    n = -np.diff(x)
    
    return x, n

def expected_cost(X, gamma, epsilon, eta, n, tau):
    """Compute expected cost of strategy."""
    permanent = 0.5 * gamma * X**2
    fixed = epsilon * np.sum(np.abs(n))
    temporary = eta * np.sum(n**2) / tau
    return permanent + fixed + temporary

def variance_cost(sigma, x, tau):
    """Compute variance of cost."""
    return sigma**2 * tau * np.sum(x[:-1]**2)
```

---

## 9. Key Results Summary

### Main Theorems

1. **Existence and Uniqueness**: For each value of risk aversion λ > 0, there exists a uniquely determined optimal execution strategy.

2. **Static Optimality**: The optimal strategy is deterministic (does not depend on price realizations during execution).

3. **Efficient Frontier**: The set of optimal strategies forms a smooth, convex efficient frontier in (E[C], V[C]) space.

4. **Closed-Form Solution**: For linear impact, the optimal trajectory has the explicit form involving sinh functions.

### Practical Insights

1. **Risk-averse traders front-load execution** - Trade more aggressively at the start
2. **Half-life concept** - Provides intuitive measure of execution urgency
3. **Trade-off visualization** - Efficient frontier shows cost-risk trade-off clearly
4. **Benchmark comparison** - Quantifies improvement over naïve strategies

---

## 10. References

1. Almgren, R. and Chriss, N. (2000). "Optimal execution of portfolio transactions", Journal of Risk 3(2), 5-39.

2. Bertsimas, D. and Lo, A. (1998). "Optimal control of execution costs", Journal of Financial Markets 1, 1-50.

3. Almgren, R. (2003). "Optimal execution with nonlinear impact functions and trading-enhanced risk", Applied Mathematical Finance 10, 1-18.

4. Gatheral, J. and Schied, A. (2011). "Optimal Trade Execution under Geometric Brownian Motion in the Almgren and Chriss Framework", IJTAF 14(3), 353-368.

5. Obizhaeva, A. and Wang, J. (2005). "Optimal trading strategy and supply/demand dynamics", MIT working paper.

---

## Appendix: Mathematical Details

### A.1 Derivation of Optimal Strategy

**Lagrangian:**
```
L = E[C] + λ·V[C] + μ·(Σn_k - X)
```

**First-order conditions:**
```
∂L/∂n_k = 2η·n_k/τ - 2λσ²τ·x_k + μ = 0
```

**Euler-Lagrange equation (continuous limit):**
```
η·v''(t) = λσ²·x(t)
```

**Boundary conditions:**
- x(0) = X
- x(T) = 0

**Solution:**
```
x(t) = X · sinh(κ(T-t)) / sinh(κT)
```

Where κ = √(λσ²/η).

### A.2 Efficient Frontier Equation

**Parametric form:**
```
E(κ) = ηX²/T · [κ·coth(κT) - 1/(κT)]
V(κ) = σ²X²T/3 · [1 - 3·coth(κT)/(κT) + 3/(κT)²]
```

As κ varies from 0 to ∞, traces out the efficient frontier.

---

*Document synthétisé à partir du paper fondateur "Optimal Execution of Portfolio Transactions" par Robert Almgren et Neil Chriss (2000). Ressource essentielle pour comprendre l'exécution optimale en finance quantitative.*
