# 📙 MODULE 5: EXÉCUTION OPTIMALE
## Modèle d'Almgren-Chriss et Extensions

---

## 📚 SOURCES PRINCIPALES
- **Almgren & Chriss (2001)**: https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf
- **Gatheral Slides**: http://mathfinance.sns.it/wp-content/uploads/2010/12/Gatheral_Optim_Exec.pdf
- **Almgren - Direct Estimation**: https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf

---

## 🎯 OBJECTIFS
1. Comprendre le trade-off risque/coût dans l'exécution
2. Dériver la trajectoire optimale d'Almgren-Chriss
3. Implémenter TWAP, VWAP et IS (Implementation Shortfall)
4. Modéliser l'impact de marché (temporaire et permanent)

---

## 🔑 LE MODÈLE ALMGREN-CHRISS

### 1. Setup du Problème

**Objectif**: Liquider X₀ actions sur l'horizon [0,T] en minimisant les coûts.

**Variables**:
- X₀: position initiale (nombre d'actions)
- T: horizon de trading
- N: nombre de périodes
- τ = T/N: durée d'une période
- xₖ: position à la fin de la période k
- nₖ = xₖ₋₁ - xₖ: actions vendues pendant la période k

**Contraintes**:
- x₀ = X₀ (position initiale)
- xₙ = 0 (liquidation complète)

### 2. Dynamique des Prix

**Prix fondamental** (sans notre trading):
```
S̃ₖ = S₀ + σ Σⱼ₌₁ᵏ εⱼ√τ
```
où εⱼ ~ N(0,1) i.i.d.

**Impact permanent** (modification permanente du prix):
```
g(v) = γv    (linéaire en vitesse de trading v = n/τ)
```

**Impact temporaire** (coût de la transaction):
```
h(v) = ε·sign(v) + η|v|
```
- ε: coût fixe (spread)
- η: impact temporaire linéaire

**Prix effectif de la kème transaction**:
```
S̃ₖ = Sₖ₋₁ - h(nₖ/τ)
```

### 3. Coût d'Exécution

**Coût total**:
```
C = Σₖ₌₁ᴺ nₖ(S₀ - S̃ₖ)
```

**Espérance du coût**:
```
E[C] = ½γX₀² + Σₖ₌₁ᴺ τh(nₖ/τ)
```

**Variance du coût**:
```
Var[C] = σ² Σₖ₌₁ᴺ τxₖ²
```

### 4. Problème d'Optimisation

**Fonction objectif** (Mean-Variance):
```
min E[C] + λ·Var[C]
```
où λ est l'aversion au risque.

**Équivalent**:
```
min ½γX₀² + ε Σₖnₖ + (η/τ) Σₖnₖ² + λσ² Σₖτxₖ²
```

### 5. Solution Optimale (Cas Continu)

Pour h(v) = η·v (impact linéaire), la trajectoire optimale est:

```
x(t) = X₀ · sinh(κ(T-t)) / sinh(κT)
```

où:
```
κ = √(λσ²/η)
```

**Vitesse de trading**:
```
v(t) = dx/dt = -X₀κ · cosh(κ(T-t)) / sinh(κT)
```

**Cas limites**:
- λ → 0 (neutre au risque): x(t) = X₀(1 - t/T) → **TWAP**
- λ → ∞ (très averse): exécution instantanée

---

## 💻 IMPLÉMENTATION PYTHON

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class AlmgrenChrissModel:
    """
    Almgren-Chriss Optimal Execution Model
    """
    def __init__(self, X0, T, sigma, eta, gamma_perm=0, epsilon=0):
        """
        Parameters:
        -----------
        X0 : float - Initial position (shares)
        T : float - Time horizon
        sigma : float - Volatility (daily)
        eta : float - Temporary impact parameter
        gamma_perm : float - Permanent impact parameter
        epsilon : float - Fixed cost (half spread)
        """
        self.X0 = X0
        self.T = T
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma_perm
        self.epsilon = epsilon
    
    def optimal_trajectory(self, lambda_risk, n_steps=100):
        """
        Compute optimal trajectory for given risk aversion
        """
        kappa = np.sqrt(lambda_risk * self.sigma**2 / self.eta)
        t = np.linspace(0, self.T, n_steps)
        
        if kappa * self.T < 1e-6:  # Nearly risk-neutral
            x = self.X0 * (1 - t / self.T)  # TWAP
        else:
            x = self.X0 * np.sinh(kappa * (self.T - t)) / np.sinh(kappa * self.T)
        
        return t, x
    
    def trading_rate(self, lambda_risk, n_steps=100):
        """
        Compute optimal trading rate v(t) = -dx/dt
        """
        kappa = np.sqrt(lambda_risk * self.sigma**2 / self.eta)
        t = np.linspace(0, self.T, n_steps)
        
        if kappa * self.T < 1e-6:
            v = np.ones_like(t) * self.X0 / self.T  # Constant rate (TWAP)
        else:
            v = self.X0 * kappa * np.cosh(kappa * (self.T - t)) / np.sinh(kappa * self.T)
        
        return t, v
    
    def expected_cost(self, trajectory, dt):
        """
        Calculate expected execution cost
        """
        n = np.diff(trajectory)  # Shares traded each period
        v = n / dt  # Trading rate
        
        # Permanent impact cost
        perm_cost = 0.5 * self.gamma * self.X0**2
        
        # Fixed costs
        fixed_cost = self.epsilon * np.sum(np.abs(n))
        
        # Temporary impact cost
        temp_cost = self.eta * np.sum(n**2) / dt
        
        return perm_cost + fixed_cost + temp_cost
    
    def variance_cost(self, trajectory, dt):
        """
        Calculate variance of execution cost
        """
        x = trajectory[:-1]  # Position at each time step
        return self.sigma**2 * dt * np.sum(x**2)
    
    def efficient_frontier(self, lambda_range=None, n_points=50):
        """
        Compute efficient frontier (E[C] vs Var[C])
        """
        if lambda_range is None:
            lambda_range = np.logspace(-2, 2, n_points)
        
        expected_costs = []
        variance_costs = []
        
        dt = self.T / 100
        
        for lam in lambda_range:
            t, x = self.optimal_trajectory(lam, n_steps=101)
            ec = self.expected_cost(x, dt)
            vc = self.variance_cost(x, dt)
            expected_costs.append(ec)
            variance_costs.append(vc)
        
        return np.array(expected_costs), np.array(variance_costs), lambda_range


def twap(X0, T, n_steps):
    """Time-Weighted Average Price strategy"""
    t = np.linspace(0, T, n_steps)
    x = X0 * (1 - t / T)
    return t, x


def vwap_target(volume_profile, X0):
    """
    VWAP strategy following volume profile
    volume_profile: array of expected volume fractions
    """
    cum_volume = np.cumsum(volume_profile) / np.sum(volume_profile)
    x = X0 * (1 - cum_volume)
    return x


# Example usage
if __name__ == "__main__":
    # Parameters (example: 1M shares, 1 day, typical values)
    model = AlmgrenChrissModel(
        X0=1_000_000,    # 1M shares
        T=1.0,            # 1 day
        sigma=0.02,       # 2% daily vol
        eta=2.5e-6,       # Temporary impact
        gamma_perm=0,     # Ignore permanent for simplicity
        epsilon=0         # Ignore fixed costs
    )
    
    # Compare trajectories for different risk aversions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    lambdas = [0.001, 0.1, 10]
    labels = ['Low Risk Aversion', 'Medium', 'High Risk Aversion']
    
    for lam, label in zip(lambdas, labels):
        t, x = model.optimal_trajectory(lam)
        axes[0].plot(t, x / model.X0, label=f'λ={lam}')
    
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Inventory (fraction)')
    axes[0].set_title('Optimal Trajectories')
    axes[0].legend()
    axes[0].grid(True)
    
    # Trading rate
    for lam, label in zip(lambdas, labels):
        t, v = model.trading_rate(lam)
        axes[1].plot(t, v, label=f'λ={lam}')
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Trading Rate')
    axes[1].set_title('Optimal Trading Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    # Efficient frontier
    ec, vc, lams = model.efficient_frontier()
    axes[2].plot(np.sqrt(vc), ec)
    axes[2].set_xlabel('Std Dev of Cost')
    axes[2].set_ylabel('Expected Cost')
    axes[2].set_title('Efficient Frontier')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('almgren_chriss_analysis.png', dpi=150)
    plt.show()
```

---

## 📊 COMPARAISON DES STRATÉGIES

| Stratégie | Description | Quand l'utiliser |
|-----------|-------------|------------------|
| **TWAP** | Trading uniforme | Faible urgence, marché stable |
| **VWAP** | Suit le profil de volume | Benchmark institutionnel |
| **IS** | Minimise Implementation Shortfall | Urgent, forte conviction |
| **Almgren-Chriss** | Optimal mean-variance | Trading quantitatif |
| **POV** | % du volume de marché | Grands ordres, discrétion |

---

## 🔗 EXTENSIONS DU MODÈLE

### 1. Impact Non-Linéaire
```
h(v) = η · v^α    (α < 1 typiquement)
```

### 2. Decay de l'Impact
```
Impact(t) = η · e^{-ρ(t-s)} · v(s)
```

### 3. Modèle Propagateur (Bouchaud et al.)
```
G(t-s) = (t-s)^{-γ}    (impact à mémoire longue)
```

### 4. Incertitude sur les Paramètres
- Apprentissage en ligne de η
- Estimation robuste

---

## 🔗 RÉFÉRENCES
1. Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions
2. Gatheral, J. (2010). No-Dynamic-Arbitrage and Market Impact
3. Obizhaeva, A. & Wang, J. (2013). Optimal Trading Strategy and Supply/Demand Dynamics
