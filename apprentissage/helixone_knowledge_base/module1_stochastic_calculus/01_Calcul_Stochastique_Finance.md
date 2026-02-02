# Calcul Stochastique pour la Finance

> **Sources**: Shreve, S.E. "Stochastic Calculus for Finance I & II", Springer
> **Extrait pour la base de connaissances HelixOne**

---

## 1. Introduction aux Processus Stochastiques

### 1.1 Définition

Un **processus stochastique** est une collection de variables aléatoires {X_t} indexées par le temps t.

**Types principaux:**
- **Temps discret**: t ∈ {0, 1, 2, ...}
- **Temps continu**: t ∈ [0, T]

### 1.2 Propriétés Importantes

**Trajectoire (Sample Path)**: Une réalisation particulière du processus ω → X_t(ω)

**Filtration**: Famille croissante de σ-algèbres F_t représentant l'information disponible au temps t.

**Adaptedness**: Un processus X_t est adapté si X_t est F_t-mesurable pour tout t.

---

## 2. Marche Aléatoire (Random Walk)

### 2.1 Marche Aléatoire Simple

```
S_n = S_0 + Σ_{i=1}^n ξ_i
```

Où ξ_i sont i.i.d. avec P(ξ_i = +1) = p et P(ξ_i = -1) = 1-p.

**Propriétés:**
- E[S_n] = S_0 + n(2p - 1)
- Var(S_n) = 4np(1-p)
- Pour p = 1/2: martingale

### 2.2 Marche Aléatoire Échelonnée

Pour modéliser des prix sur intervalle [0, T] avec n pas:

```
W^{(n)}(t) = (1/√n) · S_{⌊nt⌋}
```

**Théorème de Donsker**: W^{(n)} → W (mouvement brownien) en distribution.

---

## 3. Mouvement Brownien

### 3.1 Définition

Un **mouvement brownien standard** W = {W_t : t ≥ 0} est un processus tel que:

1. W_0 = 0
2. Trajectoires continues presque sûrement
3. **Incréments indépendants**: W_t - W_s est indépendant de F_s pour t > s
4. **Incréments gaussiens**: W_t - W_s ~ N(0, t-s)

### 3.2 Propriétés

**Moments:**
- E[W_t] = 0
- E[W_t²] = t
- E[W_t · W_s] = min(t, s)
- Var(W_t - W_s) = |t - s|

**Propriétés des trajectoires:**
- Continues mais nulle part différentiables
- Variation quadratique: [W, W]_t = t
- Variation totale: infinie sur tout intervalle

### 3.3 Mouvement Brownien Géométrique

```
S_t = S_0 · exp((μ - σ²/2)t + σW_t)
```

**Équation différentielle:**
```
dS_t = μ·S_t·dt + σ·S_t·dW_t
```

**Propriétés:**
- S_t > 0 pour tout t (modèle réaliste pour les prix)
- log(S_t/S_0) ~ N((μ - σ²/2)t, σ²t)
- E[S_t] = S_0·e^{μt}

---

## 4. Intégrale d'Itô

### 4.1 Construction

Pour une fonction simple f(t) = Σ_i c_i · 𝟙_{[t_i, t_{i+1})}(t):

```
∫_0^T f(t) dW_t = Σ_i c_i · (W_{t_{i+1}} - W_{t_i})
```

### 4.2 Propriétés

Pour f adaptée et E[∫_0^T f(t)² dt] < ∞:

**Isométrie d'Itô:**
```
E[(∫_0^T f(t) dW_t)²] = E[∫_0^T f(t)² dt]
```

**Martingale:**
```
E[∫_0^T f(t) dW_t | F_s] = ∫_0^s f(t) dW_t
```

**Moyenne nulle:**
```
E[∫_0^T f(t) dW_t] = 0
```

### 4.3 Différence avec l'Intégrale de Riemann

```
∫_0^T W_t dW_t = (1/2)(W_T² - T)  ≠  (1/2)W_T²
```

Le terme "-T" provient de la variation quadratique du mouvement brownien.

---

## 5. Lemme d'Itô

### 5.1 Formule d'Itô (1 dimension)

Pour f(t, X_t) où dX_t = μ(t,X)dt + σ(t,X)dW_t:

```
df = (∂f/∂t + μ·∂f/∂x + (1/2)σ²·∂²f/∂x²) dt + σ·∂f/∂x · dW_t
```

**En notation compacte:**
```
df = f_t dt + f_x dX + (1/2)f_{xx} d[X,X]
```

### 5.2 Formule d'Itô (Multidimensionnel)

Pour f(t, X^1_t, ..., X^n_t):

```
df = f_t dt + Σ_i f_{x_i} dX^i + (1/2) Σ_{i,j} f_{x_i x_j} d[X^i, X^j]
```

### 5.3 Exemples Importants

**Exemple 1: f(W_t) = W_t²**
```
d(W_t²) = 2W_t dW_t + dt
```

Donc: W_t² = 2∫W_s dW_s + t

**Exemple 2: f(S_t) = log(S_t) pour GBM**

Si dS = μS dt + σS dW:
```
d(log S) = (μ - σ²/2) dt + σ dW
```

**Exemple 3: f(t, W_t) = e^{αW_t - α²t/2}**
```
df = α·f·dW_t
```

C'est une martingale (martingale exponentielle).

---

## 6. Équations Différentielles Stochastiques (EDS)

### 6.1 Forme Générale

```
dX_t = μ(t, X_t) dt + σ(t, X_t) dW_t
```

Avec condition initiale X_0 = x_0.

**Forme intégrale:**
```
X_t = X_0 + ∫_0^t μ(s, X_s) ds + ∫_0^t σ(s, X_s) dW_s
```

### 6.2 Existence et Unicité

**Conditions de Lipschitz**: Si μ et σ sont Lipschitz en x et à croissance au plus linéaire, alors il existe une unique solution forte.

### 6.3 EDS Importantes en Finance

**Mouvement Brownien Géométrique:**
```
dS_t = μ·S_t dt + σ·S_t dW_t
```
Solution: S_t = S_0·exp((μ - σ²/2)t + σW_t)

**Processus d'Ornstein-Uhlenbeck (Mean-Reverting):**
```
dX_t = κ(θ - X_t) dt + σ dW_t
```
Solution: X_t = θ + (X_0 - θ)e^{-κt} + σ∫_0^t e^{-κ(t-s)} dW_s

**Modèle CIR (Cox-Ingersoll-Ross):**
```
dr_t = κ(θ - r_t) dt + σ√r_t dW_t
```
Utilisé pour les taux d'intérêt (r_t ≥ 0 si 2κθ ≥ σ²).

**Modèle de Heston (Volatilité Stochastique):**
```
dS_t = μ·S_t dt + √v_t·S_t dW_t^1
dv_t = κ(θ - v_t) dt + ξ√v_t dW_t^2
```
Avec Corr(dW^1, dW^2) = ρ.

---

## 7. Changement de Mesure et Théorème de Girsanov

### 7.1 Motivation

Transformer un processus avec drift en martingale (pricing risk-neutral).

### 7.2 Théorème de Girsanov

Si W_t est un brownien sous P et:
```
dQ/dP|_{F_t} = Z_t = exp(-∫_0^t θ_s dW_s - (1/2)∫_0^t θ_s² ds)
```

Alors:
```
W̃_t = W_t + ∫_0^t θ_s ds
```

est un mouvement brownien sous Q.

### 7.3 Application: Mesure Risk-Neutral

Pour dS = μ·S dt + σ·S dW sous P:

Avec θ = (μ - r)/σ, sous la mesure Q:
```
dS = r·S dt + σ·S dW̃
```

où W̃ est un brownien sous Q.

**Prix d'un dérivé:**
```
V_0 = e^{-rT} · E^Q[Payoff(S_T)]
```

---

## 8. Martingales

### 8.1 Définition

Un processus M_t est une **martingale** (par rapport à F_t sous P) si:
1. M_t est adapté et intégrable
2. E[M_t | F_s] = M_s pour tout s ≤ t

**Sous-martingale**: E[M_t | F_s] ≥ M_s
**Sur-martingale**: E[M_t | F_s] ≤ M_s

### 8.2 Exemples

- W_t (mouvement brownien)
- W_t² - t (carré compensé)
- exp(θW_t - θ²t/2) (martingale exponentielle)
- Prix actualisé d'un actif sous mesure risk-neutral

### 8.3 Propriétés

**Optional Stopping Theorem**: Sous certaines conditions, E[M_τ] = E[M_0] pour un temps d'arrêt τ.

**Représentation des Martingales**: Toute martingale (sous conditions) peut s'écrire:
```
M_t = M_0 + ∫_0^t H_s dW_s
```

---

## 9. Formule de Feynman-Kac

### 9.1 Énoncé

Si u(t, x) satisfait l'EDP:
```
∂u/∂t + μ(t,x)·∂u/∂x + (1/2)σ²(t,x)·∂²u/∂x² - r·u + f(t,x) = 0
```

Avec condition terminale u(T, x) = g(x).

Alors:
```
u(t, x) = E^{t,x}[e^{-r(T-t)}·g(X_T) + ∫_t^T e^{-r(s-t)}·f(s, X_s) ds]
```

### 9.2 Application: Black-Scholes

Pour un call européen avec f = 0:
```
C(t, S) = e^{-r(T-t)} · E^Q[max(S_T - K, 0) | S_t = S]
```

---

## 10. Équation de Black-Scholes

### 10.1 Hypothèses

1. Prix suit un GBM: dS = μS dt + σS dW
2. Taux sans risque r constant
3. Pas de dividendes
4. Pas de coûts de transaction
5. Trading continu possible

### 10.2 EDP de Black-Scholes

```
∂V/∂t + rS·∂V/∂S + (1/2)σ²S²·∂²V/∂S² - rV = 0
```

### 10.3 Formule pour Call Européen

```
C(S, t) = S·N(d₁) - K·e^{-r(T-t)}·N(d₂)
```

Où:
```
d₁ = [ln(S/K) + (r + σ²/2)(T-t)] / [σ√(T-t)]
d₂ = d₁ - σ√(T-t)
```

**Put Européen** (par parité put-call):
```
P(S, t) = K·e^{-r(T-t)}·N(-d₂) - S·N(-d₁)
```

### 10.4 Les Greeks

| Greek | Définition | Formule (Call) |
|-------|------------|----------------|
| Delta (Δ) | ∂V/∂S | N(d₁) |
| Gamma (Γ) | ∂²V/∂S² | n(d₁)/(Sσ√τ) |
| Theta (Θ) | ∂V/∂t | -Sn(d₁)σ/(2√τ) - rKe^{-rτ}N(d₂) |
| Vega (ν) | ∂V/∂σ | S√τ·n(d₁) |
| Rho (ρ) | ∂V/∂r | Kτe^{-rτ}N(d₂) |

---

## 11. Code Python

### 11.1 Simulation de Mouvement Brownien

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion(T, N, n_paths=1):
    """
    Simule des trajectoires de mouvement brownien.
    
    Parameters:
    - T: horizon temporel
    - N: nombre de pas
    - n_paths: nombre de trajectoires
    """
    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(n_paths, N)
    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    t = np.linspace(0, T, N + 1)
    return t, W

def simulate_gbm(S0, mu, sigma, T, N, n_paths=1):
    """
    Simule des trajectoires de GBM.
    """
    dt = T / N
    t, W = simulate_brownian_motion(T, N, n_paths)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return t, S
```

### 11.2 Black-Scholes Pricing

```python
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Prix d'un call européen Black-Scholes."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Prix d'un put européen Black-Scholes."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def delta_call(S, K, T, r, sigma):
    """Delta d'un call."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def gamma(S, K, T, r, sigma):
    """Gamma (même pour call et put)."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    """Vega (même pour call et put)."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)
```

### 11.3 Simulation EDS avec Euler-Maruyama

```python
def euler_maruyama(mu, sigma, X0, T, N, n_paths=1):
    """
    Schéma d'Euler-Maruyama pour EDS.
    
    dX = mu(t, X) dt + sigma(t, X) dW
    """
    dt = T / N
    X = np.zeros((n_paths, N + 1))
    X[:, 0] = X0
    
    for i in range(N):
        dW = np.sqrt(dt) * np.random.randn(n_paths)
        X[:, i+1] = X[:, i] + mu(i*dt, X[:, i])*dt + sigma(i*dt, X[:, i])*dW
    
    return X

# Exemple: Ornstein-Uhlenbeck
kappa, theta, sigma_ou = 2.0, 0.05, 0.1
mu_ou = lambda t, x: kappa * (theta - x)
sigma_ou_fn = lambda t, x: sigma_ou

X = euler_maruyama(mu_ou, sigma_ou_fn, X0=0.1, T=1, N=1000, n_paths=100)
```

---

## 12. Résumé des Formules Clés

### Mouvement Brownien
```
E[W_t] = 0,  Var(W_t) = t,  Cov(W_s, W_t) = min(s,t)
```

### Intégrale d'Itô
```
∫_0^T f dW: E[·] = 0,  E[(·)²] = E[∫f² dt]
```

### Lemme d'Itô
```
df = f_t dt + f_x dX + (1/2)f_{xx} σ² dt
```

### GBM
```
S_t = S_0·exp((μ - σ²/2)t + σW_t)
```

### Black-Scholes
```
C = SN(d₁) - Ke^{-rT}N(d₂)
```

---

## Références

1. Shreve, S.E. (2004). "Stochastic Calculus for Finance I: The Binomial Asset Pricing Model", Springer.
2. Shreve, S.E. (2004). "Stochastic Calculus for Finance II: Continuous-Time Models", Springer.
3. Øksendal, B. (2003). "Stochastic Differential Equations", Springer.
4. Hull, J.C. (2018). "Options, Futures, and Other Derivatives", Pearson.
5. Björk, T. (2009). "Arbitrage Theory in Continuous Time", Oxford University Press.

---

*Document synthétisé pour la base de connaissances HelixOne. Fondamentaux du calcul stochastique appliqué à la finance.*
