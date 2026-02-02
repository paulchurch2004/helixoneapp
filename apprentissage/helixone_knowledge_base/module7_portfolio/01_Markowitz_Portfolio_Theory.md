# Markowitz Mean-Variance Portfolio Theory

> **Source**: Markowitz, H. (1952). "Portfolio Selection", The Journal of Finance, 7(1), 77-91
> **Prix Nobel d'Économie**: 1990
> **Extrait pour la base de connaissances HelixOne**

---

## 1. Introduction et Contexte Historique

### 1.1 La Révolution Markowitz

En 1952, Harry Markowitz publie "Portfolio Selection" dans le Journal of Finance, posant les fondements de la **Modern Portfolio Theory (MPT)**. Ce papier introduit deux idées révolutionnaires:

1. **Perspective au niveau du portefeuille**: L'analyse ne doit pas porter sur les actifs individuels mais sur leurs interactions au sein du portefeuille
2. **Quantification du risque**: Le risque est mesuré par la variance (ou l'écart-type) des rendements

### 1.2 Insight Fondamental

> "Un actif ne doit pas être évalué isolément, mais par sa contribution au risque et au rendement global du portefeuille."

L'idée clé est que la **diversification** permet de réduire le risque sans sacrifier le rendement attendu, à condition que les actifs ne soient pas parfaitement corrélés.

---

## 2. Framework Mathématique

### 2.1 Notation

Considérons un portefeuille de **n actifs**:

| Symbole | Description |
|---------|-------------|
| w_i | Poids de l'actif i dans le portefeuille |
| μ_i | Rendement attendu de l'actif i |
| σ_i | Écart-type des rendements de l'actif i |
| σ_ij | Covariance entre les actifs i et j |
| ρ_ij | Corrélation entre les actifs i et j |
| **w** | Vecteur des poids (w_1, ..., w_n)^T |
| **μ** | Vecteur des rendements attendus |
| **Σ** | Matrice de covariance (n × n) |

### 2.2 Rendement du Portefeuille

Le rendement attendu du portefeuille est la moyenne pondérée des rendements:

```
E[R_p] = Σ_i w_i · μ_i = w^T · μ
```

### 2.3 Risque du Portefeuille

La variance du portefeuille:

```
Var(R_p) = Σ_i Σ_j w_i · w_j · σ_ij = w^T · Σ · w
```

L'écart-type (volatilité):

```
σ_p = √(w^T · Σ · w)
```

### 2.4 Effet de la Diversification

Pour deux actifs (cas simple):

```
σ_p² = w_1²·σ_1² + w_2²·σ_2² + 2·w_1·w_2·ρ_12·σ_1·σ_2
```

**Cas limites:**
- ρ = 1: Pas de bénéfice de diversification
- ρ = -1: Diversification parfaite possible (risque nul)
- ρ = 0: Réduction significative du risque

---

## 3. Le Problème d'Optimisation

### 3.1 Formulation Générale

**Objectif**: Maximiser le rendement pour un niveau de risque donné, ou minimiser le risque pour un rendement cible.

**Formulation mathématique** (minimisation du risque):

```
min_{w}  (1/2) · w^T · Σ · w

subject to:
    w^T · μ = μ_target    (rendement cible)
    w^T · 1 = 1           (poids somment à 1)
    w_i ≥ 0  ∀i          (optionnel: pas de vente à découvert)
```

### 3.2 Solution Analytique (Sans Contraintes de Positivité)

Le Lagrangien:

```
L = (1/2)·w^T·Σ·w - λ·(w^T·μ - μ_target) - γ·(w^T·1 - 1)
```

**Conditions du premier ordre:**

```
∂L/∂w = Σ·w - λ·μ - γ·1 = 0
```

**Solution:**

```
w* = λ·Σ^{-1}·μ + γ·Σ^{-1}·1
```

Où λ et γ sont déterminés par les contraintes.

---

## 4. La Frontière Efficiente

### 4.1 Définition

La **frontière efficiente** (efficient frontier) est l'ensemble des portefeuilles qui:
- Maximisent le rendement pour chaque niveau de risque
- Minimisent le risque pour chaque niveau de rendement

### 4.2 Forme Géométrique

Dans l'espace (σ_p, E[R_p]):
- La frontière forme une **hyperbole**
- La partie supérieure est la frontière efficiente
- La partie inférieure contient les portefeuilles inefficients

### 4.3 Portefeuilles Remarquables

**Global Minimum Variance Portfolio (GMVP):**
```
w_GMV = Σ^{-1}·1 / (1^T·Σ^{-1}·1)
```

**Portefeuille Tangent** (avec actif sans risque):
Le portefeuille qui maximise le ratio de Sharpe.

---

## 5. Avec un Actif Sans Risque

### 5.1 Capital Market Line (CML)

Avec un actif sans risque de rendement r_f:

```
E[R_p] = r_f + (E[R_m] - r_f)/σ_m · σ_p
```

### 5.2 Portefeuille Tangent (Market Portfolio)

```
w_tangent = Σ^{-1}·(μ - r_f·1) / (1^T·Σ^{-1}·(μ - r_f·1))
```

### 5.3 Ratio de Sharpe

```
Sharpe Ratio = (E[R_p] - r_f) / σ_p
```

Le portefeuille tangent maximise ce ratio.

### 5.4 Two-Fund Theorem

Tout portefeuille efficient peut être construit comme une combinaison de:
1. L'actif sans risque
2. Le portefeuille tangent

---

## 6. CAPM (Capital Asset Pricing Model)

### 6.1 Dérivation depuis Markowitz

En supposant que tous les investisseurs utilisent l'optimisation mean-variance:
- Tous détiennent le portefeuille de marché
- Le rendement attendu d'un actif dépend de sa covariance avec le marché

### 6.2 Équation du CAPM

```
E[R_i] - r_f = β_i · (E[R_m] - r_f)
```

Où le **bêta** est:

```
β_i = Cov(R_i, R_m) / Var(R_m) = σ_im / σ_m²
```

### 6.3 Security Market Line (SML)

Relation linéaire entre bêta et rendement attendu:
- β = 0: Rendement = r_f
- β = 1: Rendement = E[R_m]
- β > 1: Actif agressif (plus risqué que le marché)
- β < 1: Actif défensif

---

## 7. Implémentation Pratique

### 7.1 Estimation des Paramètres

**Rendements attendus:**
```python
μ = (1/T) · Σ_t r_t  # Moyenne historique
```

**Matrice de covariance:**
```python
Σ = (1/(T-1)) · Σ_t (r_t - μ)(r_t - μ)^T
```

### 7.2 Code Python avec PyPortfolioOpt

```python
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Données de prix
prices = pd.read_csv('stock_prices.csv', index_col='date')

# Estimation des paramètres
mu = expected_returns.mean_historical_return(prices)
Sigma = risk_models.sample_cov(prices)

# Optimisation
ef = EfficientFrontier(mu, Sigma)

# Options:
# 1. Maximum Sharpe Ratio
weights_sharpe = ef.max_sharpe(risk_free_rate=0.02)

# 2. Minimum Volatility
ef2 = EfficientFrontier(mu, Sigma)
weights_minvol = ef2.min_volatility()

# 3. Target Return
ef3 = EfficientFrontier(mu, Sigma)
weights_target = ef3.efficient_return(target_return=0.15)

# Afficher les poids
print(ef.clean_weights())
```

### 7.3 Implémentation from Scratch

```python
import numpy as np
from scipy.optimize import minimize

def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def negative_sharpe_ratio(weights, returns, cov_matrix, rf=0.02):
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return -(ret - rf) / vol

def optimize_portfolio(returns, cov_matrix, target='sharpe', rf=0.02):
    n = len(returns)
    init_weights = np.ones(n) / n
    
    # Contraintes
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Somme = 1
    ]
    
    # Bornes (pas de short)
    bounds = tuple((0, 1) for _ in range(n))
    
    if target == 'sharpe':
        result = minimize(
            negative_sharpe_ratio,
            init_weights,
            args=(returns, cov_matrix, rf),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    elif target == 'min_vol':
        result = minimize(
            portfolio_volatility,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
    
    return result.x

def efficient_frontier(returns, cov_matrix, n_points=100):
    """Calcule la frontière efficiente."""
    target_returns = np.linspace(returns.min(), returns.max(), n_points)
    frontier_vol = []
    
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, returns) - target}
        ]
        bounds = tuple((0, 1) for _ in range(len(returns)))
        
        result = minimize(
            portfolio_volatility,
            np.ones(len(returns)) / len(returns),
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        frontier_vol.append(result.fun)
    
    return target_returns, frontier_vol
```

---

## 8. Limitations et Extensions

### 8.1 Problèmes du Modèle Original

1. **Erreur d'estimation**: Les paramètres (μ, Σ) sont estimés avec erreur
2. **Instabilité**: Petits changements dans les inputs → grands changements dans les poids
3. **Solutions extrêmes**: Tendance à produire des poids extrêmes (longs/shorts)
4. **Hypothèses irréalistes**: 
   - Rendements normalement distribués
   - Investisseurs uniquement concernés par moyenne et variance

### 8.2 Techniques de Régularisation

**Shrinkage de la matrice de covariance:**
```
Σ_shrunk = (1-δ)·Σ_sample + δ·Σ_target
```

Où Σ_target est une matrice structurée (ex: identité, factorielle).

**Contraintes de position:**
- Limite max/min sur les poids
- Contraintes sectorielles
- Tracking error vs benchmark

### 8.3 Extensions Modernes

1. **Black-Litterman Model** (1992):
   - Combine équilibre de marché avec vues subjectives
   - Plus stable que Markowitz pur

2. **Robust Optimization**:
   - Prend en compte l'incertitude sur les paramètres
   - Optimise pour le pire cas

3. **Post-Modern Portfolio Theory**:
   - Utilise des mesures de risque asymétriques
   - Semi-variance, VaR, CVaR

4. **Machine Learning Integration**:
   - Prédiction des rendements avec ML
   - Estimation de covariance avec techniques de shrinkage

---

## 9. Hypothèses du Modèle

### 9.1 Hypothèses sur les Investisseurs

1. **Risk-averse**: Préfèrent moins de risque pour un rendement donné
2. **Utility maximizers**: Maximisent une fonction d'utilité concave
3. **Single-period**: Horizon d'investissement unique

### 9.2 Hypothèses sur les Marchés

1. **Marchés efficients**: Pas d'arbitrage
2. **Liquidité parfaite**: Pas de coûts de transaction
3. **Divisibilité**: On peut acheter n'importe quelle fraction d'actif

### 9.3 Hypothèses sur les Rendements

1. **Complètement caractérisés par μ et Σ**
2. **Distributions normales** (ou utilité quadratique)

---

## 10. Résumé des Formules Clés

### Portefeuille

| Quantité | Formule |
|----------|---------|
| Rendement attendu | E[R_p] = w^T · μ |
| Variance | σ_p² = w^T · Σ · w |
| Sharpe Ratio | SR = (E[R_p] - r_f) / σ_p |

### Portefeuilles Optimaux

| Portefeuille | Formule |
|--------------|---------|
| GMVP | w = Σ^{-1}·1 / (1^T·Σ^{-1}·1) |
| Tangent | w = Σ^{-1}·(μ - r_f·1) / (1^T·Σ^{-1}·(μ - r_f·1)) |

### CAPM

| Quantité | Formule |
|----------|---------|
| Bêta | β_i = Cov(R_i, R_m) / Var(R_m) |
| Rendement attendu | E[R_i] = r_f + β_i·(E[R_m] - r_f) |

---

## Références

1. Markowitz, H. (1952). "Portfolio Selection", The Journal of Finance, 7(1), 77-91.
2. Markowitz, H. (1959). "Portfolio Selection: Efficient Diversification of Investments", Wiley.
3. Sharpe, W.F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium", Journal of Finance.
4. Black, F. and Litterman, R. (1992). "Global Portfolio Optimization", Financial Analysts Journal.
5. Michaud, R.O. (1989). "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?", Financial Analysts Journal.

---

*Document synthétisé pour la base de connaissances HelixOne. Théorie fondamentale de l'optimisation de portefeuille.*
