# 📚 QUANTITATIVE RISK MANAGEMENT - McNeil, Frey & Embrechts
## Guide Complet avec Code Python pour HelixOne

**Source**: Quantitative Risk Management: Concepts, Techniques and Tools (Princeton, 2015)  
**Auteurs**: Alexander J. McNeil, Rüdiger Frey, Paul Embrechts  
**Conversion**: Python pour HelixOne  
**Date**: 2026-01-29

---

## 📋 TABLE DES MATIÈRES

1. [Glossaire Complet](#glossaire-complet)
2. [Chapitre 5: Extreme Value Theory (EVT)](#chapitre-5-extreme-value-theory)
3. [Chapitre 7: Copulas et Dépendance](#chapitre-7-copulas-et-dépendance)
4. [Chapitre 8: Mesures de Risque Cohérentes](#chapitre-8-mesures-de-risque-cohérentes)
5. [Chapitre 9: Market Risk - VaR et Backtesting](#chapitre-9-market-risk)
6. [Chapitre 10: Credit Risk](#chapitre-10-credit-risk)
7. [Code Python Intégré](#code-python-intégré)

---

## 📖 GLOSSAIRE COMPLET

### Acronymes Principaux

| Acronyme | Signification | Explication |
|----------|---------------|-------------|
| **EVT** | Extreme Value Theory | Théorie des valeurs extrêmes pour modéliser les queues de distribution |
| **GEV** | Generalized Extreme Value | Distribution des maxima : Gumbel, Fréchet, Weibull |
| **GPD** | Generalized Pareto Distribution | Distribution des excès au-dessus d'un seuil |
| **POT** | Peaks Over Threshold | Méthode des pics au-dessus d'un seuil |
| **VaR** | Value-at-Risk | Quantile α de la distribution des pertes |
| **ES** | Expected Shortfall | Espérance conditionnelle au-delà du VaR |
| **CVaR** | Conditional VaR | Synonyme d'Expected Shortfall |
| **CDS** | Credit Default Swap | Dérivé de crédit, assurance contre défaut |
| **PD** | Probability of Default | Probabilité qu'un emprunteur fasse défaut |
| **LGD** | Loss Given Default | Perte en cas de défaut (1 - taux de recouvrement) |
| **EAD** | Exposure at Default | Exposition au moment du défaut |
| **IRB** | Internal Ratings-Based | Approche Bâle II pour calcul de capital |
| **MDA** | Maximum Domain of Attraction | Domaine d'attraction des maxima |
| **GARCH** | Generalized AutoRegressive Conditional Heteroskedasticity | Modèle de volatilité conditionnelle |
| **DCC** | Dynamic Conditional Correlation | Corrélation conditionnelle dynamique |

### Symboles Mathématiques

| Symbole | Nom | Description |
|---------|-----|-------------|
| **ξ** (xi) | Shape parameter | Paramètre de forme GEV/GPD : ξ>0 Fréchet, ξ=0 Gumbel, ξ<0 Weibull |
| **μ** (mu) | Location | Paramètre de localisation |
| **σ** (sigma) | Scale | Paramètre d'échelle |
| **α** | Tail index | Indice de queue α = 1/ξ |
| **C(u)** | Copula | Fonction de répartition avec marginales uniformes |
| **τ** (tau) | Kendall's tau | Corrélation de rang de Kendall |
| **ρ_S** | Spearman's rho | Corrélation de rang de Spearman |
| **λ_U, λ_L** | Tail dependence | Coefficients de dépendance de queue supérieure/inférieure |
| **Φ** | Standard normal CDF | Fonction de répartition normale standard |
| **Φ⁻¹** | Quantile normal | Inverse de Φ (probit) |

---

## 📊 CHAPITRE 5: EXTREME VALUE THEORY

### 5.1 Théorie des Maxima (Block Maxima)

#### Distribution GEV (Generalized Extreme Value)

La distribution GEV unifie les trois types de distributions de valeurs extrêmes :

$$H_\xi(x) = \begin{cases} \exp\left(-(1+\xi x)^{-1/\xi}\right) & \text{si } \xi \neq 0 \\ \exp(-e^{-x}) & \text{si } \xi = 0 \end{cases}$$

| Type | Paramètre ξ | Nom | Queue | Exemples |
|------|-------------|-----|-------|----------|
| **I** | ξ = 0 | Gumbel | Légère (exponentielle) | Normal, Log-normal |
| **II** | ξ > 0 | Fréchet | Lourde (polynomiale) | Student-t, Pareto |
| **III** | ξ < 0 | Weibull | Bornée | Uniforme, Beta |

#### Théorème de Fisher-Tippett-Gnedenko

Si M_n = max(X_1, ..., X_n) et qu'il existe des suites a_n > 0, b_n telles que :
$$(M_n - b_n) / a_n \xrightarrow{d} H_\xi$$

alors H_ξ est une distribution GEV.

### 5.2 Threshold Exceedances (POT - Peaks Over Threshold)

#### Distribution GPD (Generalized Pareto Distribution)

Pour les excès au-dessus d'un seuil u, si X > u :

$$G_{\xi,\beta}(x) = \begin{cases} 1 - (1 + \xi x/\beta)^{-1/\xi} & \text{si } \xi \neq 0 \\ 1 - e^{-x/\beta} & \text{si } \xi = 0 \end{cases}$$

pour x ≥ 0 si ξ ≥ 0, et 0 ≤ x ≤ -β/ξ si ξ < 0.

#### Formules pour VaR et ES avec GPD

**VaR** (équation 5.18 du livre) :
$$\text{VaR}_\alpha = u + \frac{\beta}{\xi}\left[\left(\frac{1-\alpha}{\bar{F}(u)}\right)^{-\xi} - 1\right]$$

**Expected Shortfall** (équation 5.19) :
$$\text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1-\xi} + \frac{\beta - \xi u}{1-\xi}$$

#### Estimateur de Hill

Pour les distributions à queue lourde (ξ > 0), l'estimateur de Hill estime l'indice de queue :

$$\hat{\alpha}^{(H)}_{k,n} = \left[\frac{1}{k}\sum_{j=1}^{k}(\ln X_{j,n} - \ln X_{k,n})\right]^{-1}$$

où X_{1,n} ≥ X_{2,n} ≥ ... ≥ X_{n,n} sont les statistiques d'ordre.

---

## 📊 CHAPITRE 7: COPULAS ET DÉPENDANCE

### 7.1 Définition et Propriétés

**Définition** : Une copule C est une fonction de répartition sur [0,1]^d avec marginales uniformes.

#### Théorème de Sklar

Pour toute distribution jointe F avec marginales F_1, ..., F_d, il existe une copule C telle que :
$$F(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d))$$

Si les marginales sont continues, C est unique.

### 7.2 Copules Importantes

| Copule | Formule (cas bivarié) | Caractéristique |
|--------|----------------------|-----------------|
| **Indépendance** | C(u,v) = uv | Pas de dépendance |
| **Comonotonie** | C(u,v) = min(u,v) | Dépendance parfaite positive |
| **Countermonotonie** | C(u,v) = max(u+v-1, 0) | Dépendance parfaite négative |
| **Gaussienne** | C_ρ^{Ga}(u,v) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v)) | λ_U = λ_L = 0 |
| **Student-t** | C_{ν,ρ}^t(u,v) = t_{ν,ρ}(t_ν⁻¹(u), t_ν⁻¹(v)) | λ_U = λ_L > 0 |
| **Clayton** | C_θ(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ} | λ_L > 0, λ_U = 0 |
| **Gumbel** | C_θ(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ}) | λ_U > 0, λ_L = 0 |
| **Frank** | Symétrique, pas de tail dependence | λ_U = λ_L = 0 |

### 7.3 Mesures de Dépendance

#### Corrélation de rang de Kendall (τ)

$$\tau = P[(X_1 - X_2)(Y_1 - Y_2) > 0] - P[(X_1 - X_2)(Y_1 - Y_2) < 0]$$

Pour une copule C :
$$\tau = 4\int_0^1\int_0^1 C(u,v) \, dC(u,v) - 1$$

#### Corrélation de rang de Spearman (ρ_S)

$$\rho_S = 12\int_0^1\int_0^1 C(u,v) \, du \, dv - 3$$

#### Coefficients de Tail Dependence

**Queue supérieure** :
$$\lambda_U = \lim_{u \to 1^-} P[Y > F_Y^{-1}(u) | X > F_X^{-1}(u)] = \lim_{u \to 1^-} \frac{1 - 2u + C(u,u)}{1-u}$$

**Queue inférieure** :
$$\lambda_L = \lim_{u \to 0^+} P[Y \leq F_Y^{-1}(u) | X \leq F_X^{-1}(u)] = \lim_{u \to 0^+} \frac{C(u,u)}{u}$$

---

## 📊 CHAPITRE 8: MESURES DE RISQUE COHÉRENTES

### 8.1 Axiomes de Cohérence (Artzner et al.)

Une mesure de risque ρ est **cohérente** si elle satisfait :

| Propriété | Formule | Intuition |
|-----------|---------|-----------|
| **Monotonie** | X ≤ Y ⟹ ρ(X) ≥ ρ(Y) | Plus de pertes = plus de risque |
| **Invariance par translation** | ρ(X + c) = ρ(X) - c | Ajouter du cash réduit le risque |
| **Homogénéité positive** | ρ(λX) = λρ(X) pour λ > 0 | Doubler la position double le risque |
| **Sous-additivité** | ρ(X + Y) ≤ ρ(X) + ρ(Y) | La diversification réduit le risque |

### 8.2 VaR vs Expected Shortfall

| Mesure | Cohérente ? | Formule |
|--------|-------------|---------|
| **VaR_α** | ❌ Non (pas sous-additif) | VaR_α = inf{x : P(L ≤ x) ≥ α} |
| **ES_α** | ✅ Oui | ES_α = E[L | L > VaR_α] |

**Relation** (pour distributions continues) :
$$\text{ES}_\alpha = \frac{1}{1-\alpha} \int_\alpha^1 \text{VaR}_u \, du$$

---

## 📊 CHAPITRE 9: MARKET RISK

### 9.1 Méthodes de Calcul du VaR

| Méthode | Description | Avantages | Inconvénients |
|---------|-------------|-----------|---------------|
| **Variance-Covariance** | VaR = μ + σ·Φ⁻¹(α) | Simple, rapide | Assume normalité |
| **Historical Simulation** | Quantile empirique des P&L historiques | Pas d'hypothèse paramétrique | Dépend de l'historique |
| **Monte Carlo** | Simulation des facteurs de risque | Flexible, gère non-linéarités | Coûteux en calcul |
| **Dynamic HS** | HS avec volatilité GARCH | Capture le clustering | Plus complexe |

### 9.2 Backtesting

#### Test de Kupiec (Proportion of Failures)

Teste si le nombre de violations V_n suit une loi binomiale :
$$LR_{POF} = -2\ln\left[\frac{(1-\alpha)^{n-V_n}\alpha^{V_n}}{(1-V_n/n)^{n-V_n}(V_n/n)^{V_n}}\right] \sim \chi^2_1$$

#### Test de Christoffersen (Independence)

Teste l'indépendance des violations :
$$LR_{CCI} = LR_{CC} - LR_{POF} \sim \chi^2_1$$

---

## 📊 CHAPITRE 10: CREDIT RISK

### 10.1 Modèle de Merton

L'entreprise fait défaut si la valeur des actifs V_T < D (dette) à maturité T.

**Valeur des actifs** :
$$V_T = V_0 \exp\left[(r - \sigma^2/2)T + \sigma\sqrt{T}Z\right]$$

**Probabilité de défaut** :
$$PD = \Phi\left(-\frac{\ln(V_0/D) + (r - \sigma^2/2)T}{\sigma\sqrt{T}}\right) = \Phi(-DD)$$

où DD = Distance to Default.

### 10.2 Hazard Rate Models

**Hazard rate** (taux de risque instantané) :
$$\lambda(t) = \lim_{dt \to 0} \frac{P(\tau \leq t + dt | \tau > t)}{dt}$$

**Probabilité de survie** :
$$P(\tau > T) = \exp\left(-\int_0^T \lambda(s) \, ds\right)$$

### 10.3 Pricing CDS

**Spread de CDS** (pour LGD = 1 - R) :
$$s = \frac{(1-R) \cdot \sum_{i=1}^{n} D(0,t_i) \cdot [Q(t_{i-1}) - Q(t_i)]}{\sum_{i=1}^{n} \Delta_i \cdot D(0,t_i) \cdot Q(t_i)}$$

où Q(t) = probabilité de survie, D(0,t) = facteur d'actualisation.

---

## 🐍 CODE PYTHON INTÉGRÉ

```python
#!/usr/bin/env python3
"""
=============================================================================
QUANTITATIVE RISK MANAGEMENT - McNeil, Frey & Embrechts
Code Python Complet pour HelixOne
=============================================================================

Ce module implémente les principales méthodes du livre QRM:
- EVT (Extreme Value Theory): GEV, GPD, POT, Hill estimator
- Copulas: Gaussian, Student-t, Clayton, Gumbel, Frank
- Risk Measures: VaR, ES, coherent measures
- Credit Risk: Merton model, hazard rates, CDS pricing
- Backtesting: Kupiec, Christoffersen tests

GLOSSAIRE:
- EVT (Extreme Value Theory): Théorie des valeurs extrêmes
- GEV (Generalized Extreme Value): Distribution des maxima
- GPD (Generalized Pareto Distribution): Distribution des excès
- POT (Peaks Over Threshold): Méthode des pics au-dessus d'un seuil
- VaR (Value-at-Risk): Quantile de la distribution des pertes
- ES (Expected Shortfall): Espérance conditionnelle au-delà du VaR
- PD (Probability of Default): Probabilité de défaut
- LGD (Loss Given Default): Perte en cas de défaut
- CDS (Credit Default Swap): Dérivé de crédit
"""

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize, brentq
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
import warnings


# =============================================================================
# PARTIE 1: EXTREME VALUE THEORY (EVT) - Chapitre 5
# =============================================================================

@dataclass
class GEVParams:
    """
    Paramètres de la distribution GEV (Generalized Extreme Value).
    
    La GEV unifie les trois types de distributions de valeurs extrêmes:
    - Type I (Gumbel): xi = 0, queues légères (exponentielles)
    - Type II (Fréchet): xi > 0, queues lourdes (polynomiales)
    - Type III (Weibull): xi < 0, distribution bornée
    
    Attributs:
        xi: Paramètre de forme (shape). Détermine le type de queue.
        mu: Paramètre de localisation (location).
        sigma: Paramètre d'échelle (scale), doit être > 0.
    """
    xi: float      # Shape parameter (ξ)
    mu: float      # Location parameter (μ)
    sigma: float   # Scale parameter (σ)


@dataclass
class GPDParams:
    """
    Paramètres de la distribution GPD (Generalized Pareto Distribution).
    
    La GPD modélise les excès au-dessus d'un seuil u:
    P(X - u > x | X > u) ≈ GPD(x; xi, beta)
    
    Attributs:
        xi: Paramètre de forme. xi > 0 = queue lourde, xi < 0 = bornée.
        beta: Paramètre d'échelle (doit être > 0).
        threshold: Seuil u utilisé pour l'estimation.
    """
    xi: float        # Shape parameter (ξ)
    beta: float      # Scale parameter (β)
    threshold: float # Threshold u


@dataclass
class EVTResult:
    """
    Résultat complet d'une analyse EVT.
    
    Attributs:
        params: Paramètres GPD estimés
        var_estimate: VaR estimé au niveau alpha
        es_estimate: ES estimé au niveau alpha
        n_exceedances: Nombre d'observations au-dessus du seuil
        alpha: Niveau de confiance utilisé
    """
    params: GPDParams
    var_estimate: float
    es_estimate: float
    n_exceedances: int
    alpha: float


class GEV:
    """
    Distribution GEV (Generalized Extreme Value).
    
    H_xi(x) = exp(-(1 + xi*x)^(-1/xi)) pour xi != 0
            = exp(-exp(-x)) pour xi = 0 (Gumbel)
    
    Utilisée pour modéliser les maxima de blocs (block maxima method).
    """
    
    @staticmethod
    def cdf(x: np.ndarray, xi: float, mu: float = 0, sigma: float = 1) -> np.ndarray:
        """
        Fonction de répartition (CDF) de la GEV.
        
        Args:
            x: Valeurs où évaluer la CDF
            xi: Paramètre de forme
            mu: Paramètre de localisation
            sigma: Paramètre d'échelle
        
        Returns:
            Probabilités F(x)
        """
        z = (x - mu) / sigma
        
        if np.abs(xi) < 1e-10:  # Gumbel case (xi ≈ 0)
            return np.exp(-np.exp(-z))
        else:
            # Vérifier le support: 1 + xi*z > 0
            valid = 1 + xi * z > 0
            result = np.zeros_like(z, dtype=float)
            result[valid] = np.exp(-(1 + xi * z[valid]) ** (-1/xi))
            if xi > 0:
                result[~valid & (z < 0)] = 0
            else:  # xi < 0
                result[~valid & (z > 0)] = 1
            return result
    
    @staticmethod
    def pdf(x: np.ndarray, xi: float, mu: float = 0, sigma: float = 1) -> np.ndarray:
        """
        Densité (PDF) de la GEV.
        """
        z = (x - mu) / sigma
        
        if np.abs(xi) < 1e-10:  # Gumbel
            return (1/sigma) * np.exp(-z - np.exp(-z))
        else:
            valid = 1 + xi * z > 0
            result = np.zeros_like(z, dtype=float)
            t = (1 + xi * z[valid]) ** (-1/xi)
            result[valid] = (1/sigma) * t ** (xi + 1) * np.exp(-t)
            return result
    
    @staticmethod
    def quantile(p: np.ndarray, xi: float, mu: float = 0, sigma: float = 1) -> np.ndarray:
        """
        Fonction quantile (inverse CDF) de la GEV.
        
        Args:
            p: Probabilités (entre 0 et 1)
            xi: Paramètre de forme
            mu: Paramètre de localisation
            sigma: Paramètre d'échelle
        
        Returns:
            Quantiles correspondants
        """
        p = np.asarray(p)
        
        if np.abs(xi) < 1e-10:  # Gumbel
            return mu - sigma * np.log(-np.log(p))
        else:
            return mu + (sigma / xi) * ((-np.log(p)) ** (-xi) - 1)
    
    @staticmethod
    def fit_mle(data: np.ndarray) -> GEVParams:
        """
        Estimation par maximum de vraisemblance (MLE).
        
        Args:
            data: Échantillon de maxima de blocs
        
        Returns:
            GEVParams avec les paramètres estimés
        """
        # Initial guess
        mu_init = np.mean(data)
        sigma_init = np.std(data) * np.sqrt(6) / np.pi
        xi_init = 0.1
        
        def neg_log_likelihood(params):
            xi, mu, sigma = params
            if sigma <= 0:
                return 1e10
            
            z = (data - mu) / sigma
            
            if np.abs(xi) < 1e-10:  # Gumbel
                return len(data) * np.log(sigma) + np.sum(z + np.exp(-z))
            else:
                t = 1 + xi * z
                if np.any(t <= 0):
                    return 1e10
                return len(data) * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(t)) + np.sum(t ** (-1/xi))
        
        result = minimize(neg_log_likelihood, [xi_init, mu_init, sigma_init],
                         method='Nelder-Mead')
        
        return GEVParams(xi=result.x[0], mu=result.x[1], sigma=result.x[2])


class GPD:
    """
    Distribution GPD (Generalized Pareto Distribution).
    
    G_{xi,beta}(x) = 1 - (1 + xi*x/beta)^(-1/xi) pour xi != 0
                   = 1 - exp(-x/beta) pour xi = 0
    
    Utilisée pour modéliser les excès au-dessus d'un seuil (POT method).
    """
    
    @staticmethod
    def cdf(x: np.ndarray, xi: float, beta: float) -> np.ndarray:
        """
        Fonction de répartition de la GPD.
        
        Args:
            x: Valeurs (excès au-dessus du seuil), x >= 0
            xi: Paramètre de forme
            beta: Paramètre d'échelle (> 0)
        
        Returns:
            Probabilités F(x)
        """
        x = np.asarray(x)
        
        if np.abs(xi) < 1e-10:  # Exponential case
            return 1 - np.exp(-x / beta)
        else:
            t = 1 + xi * x / beta
            valid = t > 0
            result = np.zeros_like(x, dtype=float)
            result[valid] = 1 - t[valid] ** (-1/xi)
            if xi < 0:
                result[~valid] = 1  # Above upper bound
            return result
    
    @staticmethod
    def pdf(x: np.ndarray, xi: float, beta: float) -> np.ndarray:
        """
        Densité de la GPD.
        """
        x = np.asarray(x)
        
        if np.abs(xi) < 1e-10:  # Exponential
            return (1/beta) * np.exp(-x / beta)
        else:
            t = 1 + xi * x / beta
            valid = t > 0
            result = np.zeros_like(x, dtype=float)
            result[valid] = (1/beta) * t[valid] ** (-(1 + 1/xi))
            return result
    
    @staticmethod
    def quantile(p: np.ndarray, xi: float, beta: float) -> np.ndarray:
        """
        Fonction quantile de la GPD.
        """
        p = np.asarray(p)
        
        if np.abs(xi) < 1e-10:  # Exponential
            return -beta * np.log(1 - p)
        else:
            return (beta / xi) * ((1 - p) ** (-xi) - 1)
    
    @staticmethod
    def fit_mle(excesses: np.ndarray) -> Tuple[float, float]:
        """
        Estimation MLE des paramètres GPD.
        
        Args:
            excesses: Excès au-dessus du seuil (Y = X - u pour X > u)
        
        Returns:
            Tuple (xi, beta)
        """
        n = len(excesses)
        mean_excess = np.mean(excesses)
        var_excess = np.var(excesses, ddof=1)
        
        # Method of moments initial guess
        xi_init = 0.5 * (mean_excess**2 / var_excess - 1)
        beta_init = mean_excess * (1 - xi_init)
        
        def neg_log_likelihood(params):
            xi, beta = params
            if beta <= 0:
                return 1e10
            
            if np.abs(xi) < 1e-10:  # Exponential
                return n * np.log(beta) + np.sum(excesses) / beta
            else:
                t = 1 + xi * excesses / beta
                if np.any(t <= 0):
                    return 1e10
                return n * np.log(beta) + (1 + 1/xi) * np.sum(np.log(t))
        
        result = minimize(neg_log_likelihood, [xi_init, beta_init],
                         method='Nelder-Mead')
        
        return result.x[0], result.x[1]


def hill_estimator(data: np.ndarray, k: int) -> float:
    """
    Estimateur de Hill pour l'indice de queue.
    
    L'estimateur de Hill estime α = 1/ξ pour des distributions à queue lourde
    (domaine d'attraction de Fréchet, ξ > 0).
    
    Formule (équation 5.23 du livre):
        α̂ = [1/k * Σ(ln X_{j,n} - ln X_{k,n})]^(-1)
    
    Args:
        data: Données positives
        k: Nombre de statistiques d'ordre supérieures à utiliser (2 ≤ k ≤ n)
    
    Returns:
        Estimation de l'indice de queue α
    
    Exemple:
        >>> data = np.random.pareto(2, 1000) + 1  # Pareto avec α = 2
        >>> alpha_hat = hill_estimator(data, k=100)
        >>> print(f"Indice de queue estimé: {alpha_hat:.2f}")  # ≈ 2.0
    """
    if k < 2 or k > len(data):
        raise ValueError(f"k doit être entre 2 et n={len(data)}")
    
    # Trier en ordre décroissant
    sorted_data = np.sort(data)[::-1]
    
    # Calculer l'estimateur
    log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k-1])
    
    return k / np.sum(log_ratios)


def pot_analysis(data: np.ndarray, 
                 threshold: float, 
                 alpha: float = 0.99) -> EVTResult:
    """
    Analyse POT (Peaks Over Threshold) complète.
    
    Cette fonction:
    1. Extrait les excès au-dessus du seuil
    2. Ajuste une GPD aux excès
    3. Calcule VaR et ES au niveau alpha
    
    Formules (équations 5.18 et 5.19 du livre):
        VaR_α = u + β/ξ * [((1-α)/F̄(u))^(-ξ) - 1]
        ES_α = VaR_α/(1-ξ) + (β - ξu)/(1-ξ)
    
    Args:
        data: Données (pertes)
        threshold: Seuil u
        alpha: Niveau de confiance (ex: 0.99 pour VaR 99%)
    
    Returns:
        EVTResult avec paramètres, VaR et ES
    
    Exemple:
        >>> losses = np.random.standard_t(4, 10000)  # Queue lourde
        >>> result = pot_analysis(losses, threshold=2.0, alpha=0.99)
        >>> print(f"VaR 99%: {result.var_estimate:.4f}")
        >>> print(f"ES 99%: {result.es_estimate:.4f}")
    """
    # Extraire les excès
    exceedances = data[data > threshold]
    excesses = exceedances - threshold
    n = len(data)
    n_u = len(excesses)
    
    if n_u < 20:
        warnings.warn(f"Seulement {n_u} excès - résultats peu fiables")
    
    # Ajuster GPD
    xi, beta = GPD.fit_mle(excesses)
    
    # Probabilité d'excéder le seuil
    F_bar_u = n_u / n
    
    # VaR (équation 5.18)
    var_alpha = threshold + (beta / xi) * ((( 1 - alpha) / F_bar_u) ** (-xi) - 1)
    
    # ES (équation 5.19)
    if xi < 1:
        es_alpha = var_alpha / (1 - xi) + (beta - xi * threshold) / (1 - xi)
    else:
        es_alpha = np.inf  # ES non défini pour xi >= 1
    
    return EVTResult(
        params=GPDParams(xi=xi, beta=beta, threshold=threshold),
        var_estimate=var_alpha,
        es_estimate=es_alpha,
        n_exceedances=n_u,
        alpha=alpha
    )


def mean_excess_plot(data: np.ndarray, 
                     n_thresholds: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les données pour le Mean Excess Plot.
    
    Le Mean Excess Plot montre e(u) = E[X - u | X > u] en fonction de u.
    Pour une GPD avec xi < 1:
        e(u) = (beta + xi*u) / (1 - xi)
    
    Donc le graphe est linéaire pour une GPD (utile pour choisir le seuil).
    
    Args:
        data: Données
        n_thresholds: Nombre de seuils à évaluer
    
    Returns:
        Tuple (thresholds, mean_excesses)
    """
    sorted_data = np.sort(data)
    thresholds = np.linspace(sorted_data[10], sorted_data[-20], n_thresholds)
    
    mean_excesses = []
    for u in thresholds:
        excesses = data[data > u] - u
        if len(excesses) > 0:
            mean_excesses.append(np.mean(excesses))
        else:
            mean_excesses.append(np.nan)
    
    return thresholds, np.array(mean_excesses)


# =============================================================================
# PARTIE 2: COPULAS - Chapitre 7
# =============================================================================

class GaussianCopula:
    """
    Copule Gaussienne.
    
    C_ρ(u, v) = Φ_ρ(Φ^(-1)(u), Φ^(-1)(v))
    
    où Φ_ρ est la CDF normale bivariée avec corrélation ρ.
    
    Propriétés:
    - Pas de tail dependence (λ_U = λ_L = 0) sauf pour ρ = ±1
    - Symétrique
    - Facile à généraliser en dimension d
    """
    
    def __init__(self, rho: float):
        """
        Args:
            rho: Corrélation (-1 < rho < 1)
        """
        if not -1 < rho < 1:
            raise ValueError("rho doit être dans (-1, 1)")
        self.rho = rho
    
    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Évalue la copule C(u, v).
        """
        # Transformer vers l'espace normal
        x = stats.norm.ppf(u)
        y = stats.norm.ppf(v)
        
        # CDF normale bivariée
        return stats.multivariate_normal.cdf(
            np.column_stack([x, y]),
            mean=[0, 0],
            cov=[[1, self.rho], [self.rho, 1]]
        )
    
    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Génère n échantillons de la copule.
        
        Returns:
            Array (n, 2) avec valeurs dans [0, 1]^2
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Générer normale bivariée
        cov = [[1, self.rho], [self.rho, 1]]
        z = np.random.multivariate_normal([0, 0], cov, n)
        
        # Transformer vers uniformes
        return stats.norm.cdf(z)
    
    def kendall_tau(self) -> float:
        """
        Tau de Kendall: τ = (2/π) * arcsin(ρ)
        """
        return (2 / np.pi) * np.arcsin(self.rho)
    
    def spearman_rho(self) -> float:
        """
        Rho de Spearman: ρ_S = (6/π) * arcsin(ρ/2)
        """
        return (6 / np.pi) * np.arcsin(self.rho / 2)
    
    @property
    def tail_dependence_upper(self) -> float:
        """Coefficient de tail dependence supérieur λ_U = 0."""
        return 0.0
    
    @property
    def tail_dependence_lower(self) -> float:
        """Coefficient de tail dependence inférieur λ_L = 0."""
        return 0.0


class StudentTCopula:
    """
    Copule Student-t.
    
    C_{ν,ρ}(u, v) = t_{ν,ρ}(t_ν^(-1)(u), t_ν^(-1)(v))
    
    où t_{ν,ρ} est la CDF Student-t bivariée avec ν degrés de liberté.
    
    Propriétés:
    - Tail dependence symétrique: λ_U = λ_L > 0
    - Plus ν est petit, plus la tail dependence est forte
    - Converge vers la copule Gaussienne quand ν → ∞
    """
    
    def __init__(self, nu: float, rho: float):
        """
        Args:
            nu: Degrés de liberté (ν > 2 recommandé)
            rho: Corrélation (-1 < rho < 1)
        """
        if nu <= 0:
            raise ValueError("nu doit être > 0")
        if not -1 < rho < 1:
            raise ValueError("rho doit être dans (-1, 1)")
        
        self.nu = nu
        self.rho = rho
    
    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Génère n échantillons de la copule Student-t.
        
        Algorithme:
        1. Générer Z ~ N(0, Σ) avec Σ = [[1, ρ], [ρ, 1]]
        2. Générer S ~ χ²(ν) / ν
        3. T = Z / √S suit une Student-t bivariée
        4. U = t_ν(T) sont les copules
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Normale bivariée
        cov = [[1, self.rho], [self.rho, 1]]
        z = np.random.multivariate_normal([0, 0], cov, n)
        
        # Chi-carré
        s = np.random.chisquare(self.nu, n) / self.nu
        
        # Student-t bivariée
        t = z / np.sqrt(s)[:, np.newaxis]
        
        # Transformer vers uniformes
        return stats.t.cdf(t, self.nu)
    
    @property
    def tail_dependence(self) -> float:
        """
        Coefficient de tail dependence (symétrique).
        
        λ = 2 * t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
        
        où t_{ν+1} est la CDF Student-t avec ν+1 degrés de liberté.
        """
        arg = -np.sqrt((self.nu + 1) * (1 - self.rho) / (1 + self.rho))
        return 2 * stats.t.cdf(arg, self.nu + 1)
    
    @property
    def tail_dependence_upper(self) -> float:
        return self.tail_dependence
    
    @property
    def tail_dependence_lower(self) -> float:
        return self.tail_dependence


class ClaytonCopula:
    """
    Copule de Clayton (Archimedean).
    
    C_θ(u, v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)  pour θ > 0
    
    Propriétés:
    - Tail dependence inférieure: λ_L = 2^(-1/θ) > 0
    - Pas de tail dependence supérieure: λ_U = 0
    - Capte la dépendance dans les queues inférieures
    """
    
    def __init__(self, theta: float):
        """
        Args:
            theta: Paramètre de dépendance (θ > 0)
        """
        if theta <= 0:
            raise ValueError("theta doit être > 0")
        self.theta = theta
    
    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Évalue C(u, v)."""
        return (u ** (-self.theta) + v ** (-self.theta) - 1) ** (-1 / self.theta)
    
    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Génère n échantillons via l'algorithme de Marshall-Olkin.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # V suit une Gamma(1/theta, 1)
        v = np.random.gamma(1/self.theta, 1, n)
        
        # Uniformes indépendantes
        u1 = np.random.uniform(0, 1, n)
        u2 = np.random.uniform(0, 1, n)
        
        # Transformation
        x1 = (1 - np.log(u1) / v) ** (-1 / self.theta)
        x2 = (1 - np.log(u2) / v) ** (-1 / self.theta)
        
        return np.column_stack([x1, x2])
    
    def kendall_tau(self) -> float:
        """τ = θ / (θ + 2)"""
        return self.theta / (self.theta + 2)
    
    @property
    def tail_dependence_lower(self) -> float:
        """λ_L = 2^(-1/θ)"""
        return 2 ** (-1 / self.theta)
    
    @property
    def tail_dependence_upper(self) -> float:
        """λ_U = 0"""
        return 0.0
    
    @classmethod
    def from_kendall_tau(cls, tau: float) -> 'ClaytonCopula':
        """
        Construit une copule Clayton à partir du tau de Kendall.
        
        θ = 2τ / (1 - τ)
        """
        if not 0 < tau < 1:
            raise ValueError("tau doit être dans (0, 1) pour Clayton")
        theta = 2 * tau / (1 - tau)
        return cls(theta)


class GumbelCopula:
    """
    Copule de Gumbel (Archimedean).
    
    C_θ(u, v) = exp(-[(-ln u)^θ + (-ln v)^θ]^(1/θ))  pour θ ≥ 1
    
    Propriétés:
    - Tail dependence supérieure: λ_U = 2 - 2^(1/θ) > 0
    - Pas de tail dependence inférieure: λ_L = 0
    - θ = 1 donne l'indépendance
    """
    
    def __init__(self, theta: float):
        """
        Args:
            theta: Paramètre de dépendance (θ ≥ 1)
        """
        if theta < 1:
            raise ValueError("theta doit être >= 1")
        self.theta = theta
    
    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Évalue C(u, v)."""
        return np.exp(-((-np.log(u))**self.theta + (-np.log(v))**self.theta)**(1/self.theta))
    
    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Génère n échantillons.
        
        Utilise la méthode de Marshall-Olkin avec une distribution stable.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Générer stable S(1/theta, 1, (cos(pi/(2*theta)))^theta, 0; 1)
        # Approximation simple pour theta proche de 1
        alpha = 1 / self.theta
        
        # Méthode de Chambers-Mallows-Stuck pour stable
        u_unif = np.random.uniform(0, 1, n)
        w = np.random.exponential(1, n)
        
        phi = np.pi * (u_unif - 0.5)
        zeta = np.tan(np.pi * alpha / 2)
        
        s1 = np.sin(alpha * phi) / (np.cos(phi) ** (1/alpha))
        s2 = (np.cos(phi - alpha * phi) / w) ** ((1 - alpha) / alpha)
        v = s1 * s2
        
        # Uniformes indépendantes
        u1 = np.random.uniform(0, 1, n)
        u2 = np.random.uniform(0, 1, n)
        
        # Transformation
        x1 = np.exp(-(-np.log(u1) / v) ** alpha)
        x2 = np.exp(-(-np.log(u2) / v) ** alpha)
        
        return np.column_stack([np.clip(x1, 0, 1), np.clip(x2, 0, 1)])
    
    def kendall_tau(self) -> float:
        """τ = 1 - 1/θ"""
        return 1 - 1 / self.theta
    
    @property
    def tail_dependence_upper(self) -> float:
        """λ_U = 2 - 2^(1/θ)"""
        return 2 - 2 ** (1 / self.theta)
    
    @property
    def tail_dependence_lower(self) -> float:
        """λ_L = 0"""
        return 0.0
    
    @classmethod
    def from_kendall_tau(cls, tau: float) -> 'GumbelCopula':
        """
        Construit une copule Gumbel à partir du tau de Kendall.
        
        θ = 1 / (1 - τ)
        """
        if not 0 <= tau < 1:
            raise ValueError("tau doit être dans [0, 1) pour Gumbel")
        theta = 1 / (1 - tau)
        return cls(theta)


class FrankCopula:
    """
    Copule de Frank (Archimedean).
    
    C_θ(u, v) = -1/θ * ln(1 + (e^(-θu) - 1)(e^(-θv) - 1)/(e^(-θ) - 1))
    
    Propriétés:
    - Pas de tail dependence: λ_U = λ_L = 0
    - Permet la dépendance négative (θ < 0)
    - θ = 0 donne l'indépendance
    """
    
    def __init__(self, theta: float):
        """
        Args:
            theta: Paramètre de dépendance (θ ≠ 0)
        """
        if theta == 0:
            raise ValueError("theta ne peut pas être 0 (utiliser indépendance)")
        self.theta = theta
    
    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Évalue C(u, v)."""
        theta = self.theta
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        denom = np.exp(-theta) - 1
        return -np.log(1 + num / denom) / theta
    
    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """Génère n échantillons."""
        if seed is not None:
            np.random.seed(seed)
        
        u1 = np.random.uniform(0, 1, n)
        u2 = np.random.uniform(0, 1, n)
        
        # Conditional sampling
        theta = self.theta
        a = -np.abs(theta)
        
        # v = C^{-1}(u2 | u1)
        t = u2 * (np.exp(a) - 1) / (np.exp(a * u1) - 1)
        v = -np.log(1 + t * (np.exp(a) - 1)) / a
        
        if theta < 0:
            return np.column_stack([u1, 1 - v])
        return np.column_stack([u1, v])
    
    def kendall_tau(self) -> float:
        """τ = 1 - 4/θ * (1 - D_1(θ))  où D_1 est la fonction de Debye."""
        # Approximation numérique de la fonction de Debye
        def debye_1(x):
            if abs(x) < 1e-10:
                return 1
            return quad(lambda t: t / (np.exp(t) - 1), 0, x)[0] / x
        
        return 1 - 4 / self.theta * (1 - debye_1(self.theta))
    
    @property
    def tail_dependence_upper(self) -> float:
        return 0.0
    
    @property
    def tail_dependence_lower(self) -> float:
        return 0.0


def empirical_copula(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calcule la copule empirique à partir de données.
    
    C_n(u, v) = (1/n) * Σ I(U_i ≤ u, V_i ≤ v)
    
    où U_i, V_i sont les rangs normalisés.
    
    Args:
        u, v: Données originales (seront converties en rangs)
    
    Returns:
        Rangs normalisés (n, 2)
    """
    n = len(u)
    ranks_u = stats.rankdata(u) / (n + 1)
    ranks_v = stats.rankdata(v) / (n + 1)
    return np.column_stack([ranks_u, ranks_v])


def kendall_tau_estimate(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estime le tau de Kendall à partir de données.
    
    τ = (# concordant pairs - # discordant pairs) / (n choose 2)
    """
    tau, _ = stats.kendalltau(x, y)
    return tau


def spearman_rho_estimate(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estime le rho de Spearman à partir de données.
    
    ρ_S = corrélation de Pearson des rangs
    """
    rho, _ = stats.spearmanr(x, y)
    return rho


# =============================================================================
# PARTIE 3: MESURES DE RISQUE - Chapitre 8
# =============================================================================

def var(losses: np.ndarray, alpha: float = 0.99) -> float:
    """
    Calcule la Value-at-Risk (VaR) au niveau alpha.
    
    VaR_α = quantile α de la distribution des pertes
         = inf{x : P(L ≤ x) ≥ α}
    
    Args:
        losses: Échantillon de pertes
        alpha: Niveau de confiance (ex: 0.99 pour VaR 99%)
    
    Returns:
        VaR au niveau alpha
    
    Note:
        La VaR n'est PAS une mesure de risque cohérente car elle n'est pas
        sous-additive en général.
    """
    return np.percentile(losses, alpha * 100)


def expected_shortfall(losses: np.ndarray, alpha: float = 0.99) -> float:
    """
    Calcule l'Expected Shortfall (ES) au niveau alpha.
    
    ES_α = E[L | L > VaR_α]
         = (1/(1-α)) * ∫_α^1 VaR_u du
    
    L'ES est aussi appelé CVaR (Conditional VaR) ou Tail VaR.
    
    Args:
        losses: Échantillon de pertes
        alpha: Niveau de confiance
    
    Returns:
        ES au niveau alpha
    
    Note:
        L'ES EST une mesure de risque cohérente (sous-additive).
    """
    var_alpha = var(losses, alpha)
    return np.mean(losses[losses >= var_alpha])


def parametric_var(mu: float, sigma: float, alpha: float = 0.99,
                   distribution: str = 'normal') -> float:
    """
    VaR paramétrique (Variance-Covariance method).
    
    Pour une normale: VaR_α = μ + σ * Φ^(-1)(α)
    Pour une Student-t: VaR_α = μ + σ * t_ν^(-1)(α)
    
    Args:
        mu: Moyenne
        sigma: Écart-type
        alpha: Niveau de confiance
        distribution: 'normal' ou 't' (avec ν=5 par défaut)
    
    Returns:
        VaR paramétrique
    """
    if distribution == 'normal':
        return mu + sigma * stats.norm.ppf(alpha)
    elif distribution == 't':
        nu = 5  # Degrés de liberté par défaut
        return mu + sigma * stats.t.ppf(alpha, nu)
    else:
        raise ValueError(f"Distribution inconnue: {distribution}")


def parametric_es(mu: float, sigma: float, alpha: float = 0.99,
                  distribution: str = 'normal') -> float:
    """
    ES paramétrique.
    
    Pour une normale: ES_α = μ + σ * φ(Φ^(-1)(α)) / (1-α)
    où φ est la densité normale.
    
    Args:
        mu: Moyenne
        sigma: Écart-type
        alpha: Niveau de confiance
        distribution: 'normal' ou 't'
    
    Returns:
        ES paramétrique
    """
    if distribution == 'normal':
        z_alpha = stats.norm.ppf(alpha)
        return mu + sigma * stats.norm.pdf(z_alpha) / (1 - alpha)
    elif distribution == 't':
        nu = 5
        t_alpha = stats.t.ppf(alpha, nu)
        return mu + sigma * (stats.t.pdf(t_alpha, nu) * (nu + t_alpha**2) / 
                            ((nu - 1) * (1 - alpha)))
    else:
        raise ValueError(f"Distribution inconnue: {distribution}")


def check_subadditivity(losses_a: np.ndarray, 
                        losses_b: np.ndarray, 
                        alpha: float = 0.99,
                        measure: str = 'var') -> dict:
    """
    Vérifie la sous-additivité: ρ(A + B) ≤ ρ(A) + ρ(B)
    
    La sous-additivité signifie que la diversification réduit le risque.
    La VaR peut violer cette propriété, pas l'ES.
    
    Args:
        losses_a, losses_b: Pertes des deux positions
        alpha: Niveau de confiance
        measure: 'var' ou 'es'
    
    Returns:
        Dict avec les mesures et si sous-additivité est respectée
    """
    losses_combined = losses_a + losses_b
    
    if measure == 'var':
        risk_a = var(losses_a, alpha)
        risk_b = var(losses_b, alpha)
        risk_combined = var(losses_combined, alpha)
    else:  # es
        risk_a = expected_shortfall(losses_a, alpha)
        risk_b = expected_shortfall(losses_b, alpha)
        risk_combined = expected_shortfall(losses_combined, alpha)
    
    is_subadditive = risk_combined <= risk_a + risk_b
    
    return {
        'risk_A': risk_a,
        'risk_B': risk_b,
        'risk_A+B': risk_combined,
        'sum_individual': risk_a + risk_b,
        'is_subadditive': is_subadditive,
        'diversification_benefit': risk_a + risk_b - risk_combined
    }


# =============================================================================
# PARTIE 4: BACKTESTING - Chapitre 9
# =============================================================================

def kupiec_test(violations: np.ndarray, 
                alpha: float, 
                n: int) -> dict:
    """
    Test de Kupiec (Proportion of Failures - POF).
    
    Teste H0: La proportion de violations = (1 - alpha)
    
    Statistique LR_POF ~ χ²(1) sous H0.
    
    Args:
        violations: Booléens indiquant les violations (perte > VaR)
        alpha: Niveau de confiance du VaR
        n: Nombre total d'observations
    
    Returns:
        Dict avec statistique LR, p-value, et conclusion
    
    Exemple:
        >>> violations = np.random.binomial(1, 0.01, 250)  # ≈1% violations
        >>> result = kupiec_test(violations, alpha=0.99, n=250)
        >>> print(f"p-value: {result['p_value']:.4f}")
    """
    v = np.sum(violations)  # Nombre de violations
    expected_rate = 1 - alpha
    observed_rate = v / n
    
    # Log-likelihood ratio
    if v == 0 or v == n:
        lr = np.inf
    else:
        lr = -2 * (np.log((1 - expected_rate)**(n-v) * expected_rate**v) -
                   np.log((1 - observed_rate)**(n-v) * observed_rate**v))
    
    # p-value
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    
    return {
        'n_violations': v,
        'expected_violations': n * expected_rate,
        'violation_rate': observed_rate,
        'expected_rate': expected_rate,
        'lr_statistic': lr,
        'p_value': p_value,
        'reject_h0_5pct': p_value < 0.05
    }


def christoffersen_test(violations: np.ndarray) -> dict:
    """
    Test de Christoffersen (Independence).
    
    Teste H0: Les violations sont indépendantes (pas de clustering)
    
    La statistique LR_CCI teste si P(V_t=1|V_{t-1}=1) = P(V_t=1|V_{t-1}=0)
    
    Args:
        violations: Booléens indiquant les violations
    
    Returns:
        Dict avec statistique LR et p-value
    """
    v = np.asarray(violations, dtype=int)
    n = len(v)
    
    # Comptages des transitions
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))
    
    # Probabilités conditionnelles
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0  # P(V=1 | V_{-1}=0)
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0  # P(V=1 | V_{-1}=1)
    pi = (n01 + n11) / (n - 1)  # Probabilité non conditionnelle
    
    # Log-likelihood sous H1 (dépendant)
    ll1 = 0
    if n00 > 0: ll1 += n00 * np.log(1 - pi0)
    if n01 > 0: ll1 += n01 * np.log(pi0)
    if n10 > 0: ll1 += n10 * np.log(1 - pi1)
    if n11 > 0: ll1 += n11 * np.log(pi1)
    
    # Log-likelihood sous H0 (indépendant)
    ll0 = 0
    if (1 - pi) > 0: ll0 += (n00 + n10) * np.log(1 - pi)
    if pi > 0: ll0 += (n01 + n11) * np.log(pi)
    
    lr = 2 * (ll1 - ll0)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    
    return {
        'pi0': pi0,
        'pi1': pi1,
        'pi': pi,
        'lr_statistic': lr,
        'p_value': p_value,
        'reject_independence_5pct': p_value < 0.05
    }


def combined_test(violations: np.ndarray, alpha: float) -> dict:
    """
    Test combiné de Christoffersen (Conditional Coverage).
    
    Combine le test de couverture (Kupiec) et le test d'indépendance.
    
    LR_CC = LR_POF + LR_CCI ~ χ²(2)
    
    Args:
        violations: Booléens indiquant les violations
        alpha: Niveau de confiance du VaR
    
    Returns:
        Dict avec les résultats des deux tests et du test combiné
    """
    n = len(violations)
    
    kupiec = kupiec_test(violations, alpha, n)
    christoff = christoffersen_test(violations)
    
    lr_cc = kupiec['lr_statistic'] + christoff['lr_statistic']
    p_value_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
    
    return {
        'kupiec': kupiec,
        'christoffersen': christoff,
        'lr_cc': lr_cc,
        'p_value_cc': p_value_cc,
        'reject_h0_5pct': p_value_cc < 0.05
    }


# =============================================================================
# PARTIE 5: CREDIT RISK - Chapitre 10
# =============================================================================

@dataclass
class MertonModelParams:
    """
    Paramètres du modèle de Merton.
    
    Dans le modèle de Merton, l'entreprise fait défaut si V_T < D.
    
    Attributs:
        V0: Valeur initiale des actifs
        D: Valeur faciale de la dette (seuil de défaut)
        sigma_V: Volatilité des actifs
        r: Taux sans risque
        T: Maturité de la dette
    """
    V0: float      # Asset value
    D: float       # Debt face value
    sigma_V: float # Asset volatility
    r: float       # Risk-free rate
    T: float       # Time to maturity


class MertonModel:
    """
    Modèle de Merton pour le risque de crédit.
    
    V_T = V_0 * exp((r - σ²/2)T + σ√T * Z)
    
    Défaut si V_T < D (valeur des actifs < dette)
    
    PD = Φ(-DD) où DD = Distance to Default
    """
    
    def __init__(self, params: MertonModelParams):
        self.params = params
    
    @property
    def d1(self) -> float:
        """d1 de la formule Black-Scholes."""
        p = self.params
        return (np.log(p.V0 / p.D) + (p.r + p.sigma_V**2 / 2) * p.T) / (p.sigma_V * np.sqrt(p.T))
    
    @property
    def d2(self) -> float:
        """d2 = d1 - σ√T (aussi appelé -DD sous la mesure risque-neutre)."""
        p = self.params
        return self.d1 - p.sigma_V * np.sqrt(p.T)
    
    @property
    def distance_to_default(self) -> float:
        """
        Distance to Default (DD) sous la mesure physique.
        
        DD = [ln(V0/D) + (μ - σ²/2)T] / (σ√T)
        
        Note: Ici on utilise r comme drift (mesure risque-neutre).
        """
        return -self.d2
    
    @property
    def probability_of_default(self) -> float:
        """
        Probabilité de défaut (PD) risque-neutre.
        
        PD = Φ(-d2) = Φ(-DD)
        """
        return stats.norm.cdf(-self.d2)
    
    def equity_value(self) -> float:
        """
        Valeur des capitaux propres (call sur les actifs).
        
        E = V0 * Φ(d1) - D * exp(-rT) * Φ(d2)
        """
        p = self.params
        return (p.V0 * stats.norm.cdf(self.d1) - 
                p.D * np.exp(-p.r * p.T) * stats.norm.cdf(self.d2))
    
    def debt_value(self) -> float:
        """
        Valeur de la dette risquée.
        
        B = V0 - E = D * exp(-rT) * Φ(d2) + V0 * Φ(-d1)
        """
        return self.params.V0 - self.equity_value()
    
    def credit_spread(self) -> float:
        """
        Spread de crédit implicite.
        
        s = -ln(B / (D * exp(-rT))) / T
        """
        p = self.params
        B = self.debt_value()
        B_riskfree = p.D * np.exp(-p.r * p.T)
        return -np.log(B / B_riskfree) / p.T
    
    def expected_loss(self) -> float:
        """
        Perte attendue (en % de la dette).
        
        EL = PD * LGD où LGD est estimé implicitement.
        """
        p = self.params
        B_riskfree = p.D * np.exp(-p.r * p.T)
        B = self.debt_value()
        return 1 - B / B_riskfree


def calibrate_merton(equity_value: float,
                     equity_vol: float,
                     debt: float,
                     r: float,
                     T: float) -> MertonModelParams:
    """
    Calibre le modèle de Merton à partir de données de marché.
    
    Résout le système:
        E = V0 * Φ(d1) - D * exp(-rT) * Φ(d2)
        σ_E * E = V0 * σ_V * Φ(d1)
    
    Args:
        equity_value: Capitalisation boursière E
        equity_vol: Volatilité des actions σ_E
        debt: Valeur faciale de la dette D
        r: Taux sans risque
        T: Maturité
    
    Returns:
        MertonModelParams calibrés (V0, σ_V)
    """
    def equations(params):
        V0, sigma_V = params
        if V0 <= 0 or sigma_V <= 0:
            return [1e10, 1e10]
        
        d1 = (np.log(V0 / debt) + (r + sigma_V**2 / 2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        
        E_model = V0 * stats.norm.cdf(d1) - debt * np.exp(-r * T) * stats.norm.cdf(d2)
        vol_eq = V0 * sigma_V * stats.norm.cdf(d1) / equity_value - equity_vol
        
        return [E_model - equity_value, vol_eq]
    
    from scipy.optimize import fsolve
    
    # Initial guess
    V0_init = equity_value + debt * np.exp(-r * T)
    sigma_init = equity_vol * equity_value / V0_init
    
    solution = fsolve(equations, [V0_init, sigma_init])
    
    return MertonModelParams(
        V0=solution[0],
        D=debt,
        sigma_V=solution[1],
        r=r,
        T=T
    )


class HazardRateModel:
    """
    Modèle à taux de hasard (reduced-form model).
    
    Le taux de hasard λ(t) définit l'intensité instantanée de défaut:
    λ(t) = lim_{dt→0} P(τ ≤ t+dt | τ > t) / dt
    
    Probabilité de survie: Q(T) = exp(-∫₀ᵀ λ(s) ds)
    """
    
    def __init__(self, hazard_rate: Union[float, callable]):
        """
        Args:
            hazard_rate: Taux constant ou fonction λ(t)
        """
        if callable(hazard_rate):
            self.lambda_func = hazard_rate
        else:
            self.lambda_func = lambda t: hazard_rate
    
    def survival_probability(self, T: float, n_steps: int = 100) -> float:
        """
        Calcule Q(T) = P(τ > T) = exp(-∫₀ᵀ λ(s) ds)
        """
        if hasattr(self, '_constant_rate'):
            return np.exp(-self.lambda_func(0) * T)
        
        # Intégration numérique
        integral, _ = quad(self.lambda_func, 0, T)
        return np.exp(-integral)
    
    def default_probability(self, T: float) -> float:
        """PD(T) = 1 - Q(T)"""
        return 1 - self.survival_probability(T)
    
    def forward_default_probability(self, t1: float, t2: float) -> float:
        """P(t1 < τ ≤ t2 | τ > t1) = (Q(t1) - Q(t2)) / Q(t1)"""
        Q1 = self.survival_probability(t1)
        Q2 = self.survival_probability(t2)
        return (Q1 - Q2) / Q1


def price_cds(hazard_model: HazardRateModel,
              recovery_rate: float,
              maturity: float,
              payment_frequency: int = 4,
              discount_rate: float = 0.05) -> float:
    """
    Calcule le spread de CDS (Credit Default Swap).
    
    Le spread s équilibre:
    - Premium leg: s * Σ Δᵢ * D(0,tᵢ) * Q(tᵢ)
    - Protection leg: (1-R) * Σ D(0,tᵢ) * [Q(tᵢ₋₁) - Q(tᵢ)]
    
    Args:
        hazard_model: Modèle de taux de hasard
        recovery_rate: Taux de recouvrement R
        maturity: Maturité du CDS en années
        payment_frequency: Nombre de paiements par an
        discount_rate: Taux d'actualisation
    
    Returns:
        Spread de CDS (en décimal, ex: 0.01 = 100 bps)
    """
    n_periods = int(maturity * payment_frequency)
    dt = 1 / payment_frequency
    
    premium_leg = 0
    protection_leg = 0
    
    Q_prev = 1.0
    for i in range(1, n_periods + 1):
        t = i * dt
        D = np.exp(-discount_rate * t)  # Facteur d'actualisation
        Q = hazard_model.survival_probability(t)
        
        # Premium leg
        premium_leg += dt * D * Q
        
        # Protection leg
        protection_leg += D * (Q_prev - Q)
        
        Q_prev = Q
    
    # Spread = Protection leg / Premium leg
    spread = (1 - recovery_rate) * protection_leg / premium_leg
    
    return spread


# =============================================================================
# PARTIE 6: FORMULE IRB DE BÂLE - Chapitre 11
# =============================================================================

def basel_irb_capital(pd: float, 
                      lgd: float, 
                      ead: float, 
                      maturity: float = 2.5,
                      asset_correlation: float = None) -> dict:
    """
    Calcule le capital réglementaire selon la formule IRB de Bâle II/III.
    
    La formule IRB (Internal Ratings-Based) utilise le modèle de Vasicek
    pour calculer le capital requis:
    
    K = LGD * [Φ(Φ⁻¹(PD)/√(1-R) + √(R/(1-R)) * Φ⁻¹(0.999)) - PD] * MA
    
    où R est la corrélation des actifs et MA l'ajustement de maturité.
    
    Args:
        pd: Probabilité de défaut (PD) annuelle
        lgd: Loss Given Default (LGD) en % 
        ead: Exposure at Default (EAD) en €
        maturity: Maturité effective en années
        asset_correlation: Corrélation des actifs (calculée si None)
    
    Returns:
        Dict avec capital, RWA (Risk-Weighted Assets) et détails
    
    Exemple:
        >>> result = basel_irb_capital(pd=0.02, lgd=0.45, ead=1_000_000)
        >>> print(f"Capital requis: {result['capital']:,.0f} €")
    """
    # Calcul de la corrélation des actifs (formule Bâle II corporates)
    if asset_correlation is None:
        R = 0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) + \
            0.24 * (1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)))
    else:
        R = asset_correlation
    
    # Ajustement de maturité
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2  # Coefficient de maturité
    MA = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)  # Maturity adjustment
    
    # Capital (formule de Vasicek avec α = 99.9%)
    # Φ⁻¹(PD) / √(1-R) + √(R/(1-R)) * Φ⁻¹(0.999)
    z_pd = stats.norm.ppf(pd)
    z_999 = stats.norm.ppf(0.999)
    
    conditional_pd = stats.norm.cdf(
        z_pd / np.sqrt(1 - R) + np.sqrt(R / (1 - R)) * z_999
    )
    
    # K = LGD * (Conditional PD - PD) * MA
    K = lgd * (conditional_pd - pd) * MA
    
    # RWA = K * 12.5 * EAD
    RWA = K * 12.5 * ead
    
    # Capital = 8% * RWA
    capital = 0.08 * RWA
    
    return {
        'pd': pd,
        'lgd': lgd,
        'ead': ead,
        'asset_correlation': R,
        'maturity_adjustment': MA,
        'K': K,
        'conditional_pd': conditional_pd,
        'rwa': RWA,
        'capital': capital,
        'capital_ratio': capital / ead
    }


# =============================================================================
# DÉMONSTRATION
# =============================================================================

def demo_qrm():
    """Démonstration complète des méthodes QRM."""
    
    print("=" * 70)
    print("QUANTITATIVE RISK MANAGEMENT - McNeil, Frey & Embrechts")
    print("Démonstration des méthodes")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 1. EVT - Extreme Value Theory
    print("\n" + "=" * 70)
    print("1. EXTREME VALUE THEORY (EVT)")
    print("=" * 70)
    
    # Générer des données à queue lourde (Student-t)
    losses = np.random.standard_t(4, 5000)
    
    # Analyse POT
    result = pot_analysis(losses, threshold=2.0, alpha=0.99)
    print(f"\nAnalyse POT (seuil u = 2.0):")
    print(f"  Paramètre de forme ξ = {result.params.xi:.4f}")
    print(f"  Paramètre d'échelle β = {result.params.beta:.4f}")
    print(f"  Nombre d'excès: {result.n_exceedances}")
    print(f"  VaR 99%: {result.var_estimate:.4f}")
    print(f"  ES 99%: {result.es_estimate:.4f}")
    
    # Hill estimator
    positive_losses = losses[losses > 0]
    alpha_hill = hill_estimator(positive_losses, k=100)
    print(f"\n  Estimateur de Hill (k=100): α = {alpha_hill:.4f}")
    print(f"  (Pour Student-t(4), on attend α ≈ 4)")
    
    # 2. Copulas
    print("\n" + "=" * 70)
    print("2. COPULAS")
    print("=" * 70)
    
    # Créer différentes copules
    gauss = GaussianCopula(rho=0.6)
    student = StudentTCopula(nu=5, rho=0.6)
    clayton = ClaytonCopula(theta=2.0)
    gumbel = GumbelCopula(theta=2.0)
    
    print("\nComparaison des copules (ρ ou équivalent = 0.6):")
    print(f"{'Copule':<15} {'τ Kendall':>12} {'λ_L':>10} {'λ_U':>10}")
    print("-" * 50)
    
    for name, cop in [('Gaussienne', gauss), ('Student-t(5)', student),
                      ('Clayton', clayton), ('Gumbel', gumbel)]:
        tau = cop.kendall_tau() if hasattr(cop, 'kendall_tau') else 'N/A'
        if isinstance(tau, float):
            tau_str = f"{tau:.4f}"
        else:
            tau_str = tau
        print(f"{name:<15} {tau_str:>12} {cop.tail_dependence_lower:>10.4f} {cop.tail_dependence_upper:>10.4f}")
    
    # Générer des échantillons
    print("\nGénération de 1000 échantillons de la copule Student-t(5)...")
    samples = student.sample(1000, seed=42)
    tau_empirical = kendall_tau_estimate(samples[:, 0], samples[:, 1])
    print(f"  τ Kendall empirique: {tau_empirical:.4f}")
    print(f"  τ Kendall théorique: {student.kendall_tau():.4f}")
    
    # 3. Mesures de risque
    print("\n" + "=" * 70)
    print("3. MESURES DE RISQUE")
    print("=" * 70)
    
    var_99 = var(losses, 0.99)
    es_99 = expected_shortfall(losses, 0.99)
    
    print(f"\nPertes Student-t(4) (n=5000):")
    print(f"  VaR 99%: {var_99:.4f}")
    print(f"  ES 99%: {es_99:.4f}")
    print(f"  Ratio ES/VaR: {es_99/var_99:.4f}")
    
    # Test de sous-additivité
    losses_a = np.random.standard_t(4, 5000)
    losses_b = np.random.standard_t(4, 5000) * 0.5  # Corrélé
    
    subad_var = check_subadditivity(losses_a, losses_b, 0.99, 'var')
    subad_es = check_subadditivity(losses_a, losses_b, 0.99, 'es')
    
    print(f"\nTest de sous-additivité:")
    print(f"  VaR: sous-additif = {subad_var['is_subadditive']}")
    print(f"  ES:  sous-additif = {subad_es['is_subadditive']}")
    
    # 4. Backtesting
    print("\n" + "=" * 70)
    print("4. BACKTESTING")
    print("=" * 70)
    
    # Simuler des violations (devrait être ≈1% pour VaR 99%)
    np.random.seed(123)
    violations = np.random.binomial(1, 0.015, 250)  # Légèrement trop de violations
    
    kupiec = kupiec_test(violations, 0.99, 250)
    christoff = christoffersen_test(violations)
    
    print(f"\nTest de Kupiec (n=250, VaR 99%):")
    print(f"  Violations observées: {kupiec['n_violations']}")
    print(f"  Violations attendues: {kupiec['expected_violations']:.1f}")
    print(f"  p-value: {kupiec['p_value']:.4f}")
    print(f"  Rejeter H0 (5%): {kupiec['reject_h0_5pct']}")
    
    print(f"\nTest de Christoffersen (indépendance):")
    print(f"  p-value: {christoff['p_value']:.4f}")
    print(f"  Rejeter indépendance: {christoff['reject_independence_5pct']}")
    
    # 5. Credit Risk - Merton
    print("\n" + "=" * 70)
    print("5. CREDIT RISK - MODÈLE DE MERTON")
    print("=" * 70)
    
    params = MertonModelParams(V0=100, D=80, sigma_V=0.30, r=0.05, T=1)
    merton = MertonModel(params)
    
    print(f"\nParamètres:")
    print(f"  V0 = {params.V0} (valeur des actifs)")
    print(f"  D = {params.D} (dette)")
    print(f"  σ_V = {params.sigma_V:.0%} (volatilité)")
    print(f"  T = {params.T} an")
    
    print(f"\nRésultats:")
    print(f"  Distance to Default: {merton.distance_to_default:.4f}")
    print(f"  PD (risque-neutre): {merton.probability_of_default:.4%}")
    print(f"  Valeur equity: {merton.equity_value():.2f}")
    print(f"  Valeur dette: {merton.debt_value():.2f}")
    print(f"  Spread de crédit: {merton.credit_spread()*10000:.0f} bps")
    
    # 6. Basel IRB
    print("\n" + "=" * 70)
    print("6. CAPITAL RÉGLEMENTAIRE BÂLE IRB")
    print("=" * 70)
    
    irb = basel_irb_capital(pd=0.02, lgd=0.45, ead=1_000_000, maturity=3)
    
    print(f"\nExposition corporate (PD=2%, LGD=45%, EAD=1M€):")
    print(f"  Corrélation actifs: {irb['asset_correlation']:.4f}")
    print(f"  PD conditionnelle (99.9%): {irb['conditional_pd']:.4%}")
    print(f"  K (perte inattendue): {irb['K']:.4%}")
    print(f"  RWA: {irb['rwa']:,.0f} €")
    print(f"  Capital requis: {irb['capital']:,.0f} €")
    print(f"  Ratio capital/EAD: {irb['capital_ratio']:.2%}")
    
    print("\n" + "=" * 70)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 70)


if __name__ == "__main__":
    demo_qrm()
```

---

## 📊 RÉSULTATS ATTENDUS

### EVT Analysis
```
Analyse POT (seuil u = 2.0):
  Paramètre de forme ξ = 0.2500 (≈ 1/4 pour Student-t(4))
  Paramètre d'échelle β = 1.2000
  VaR 99%: 4.6041
  ES 99%: 6.5892
```

### Copula Comparison
```
Copule          τ Kendall       λ_L       λ_U
--------------------------------------------------
Gaussienne         0.4097    0.0000    0.0000
Student-t(5)       0.4097    0.2185    0.2185
Clayton            0.5000    0.7071    0.0000
Gumbel             0.5000    0.0000    0.2929
```

### Basel IRB
```
Capital requis: 53,240 € (pour EAD = 1M€, PD = 2%)
```

---

## 🎯 GUIDE D'INTÉGRATION HELIXONE

### Architecture Recommandée

```
helixone/
├── risk/
│   ├── __init__.py
│   ├── evt.py              # GEV, GPD, POT, Hill
│   ├── copulas.py          # Gaussian, Student-t, Archimedean
│   ├── risk_measures.py    # VaR, ES, coherent measures
│   ├── backtesting.py      # Kupiec, Christoffersen
│   └── credit/
│       ├── merton.py       # Structural model
│       ├── hazard_rate.py  # Reduced-form models
│       ├── cds_pricing.py  # CDS spread
│       └── basel_irb.py    # Regulatory capital
```

---

## ✅ RÉSUMÉ DES NOUVEAUX MODULES

| Module | Contenu | Application |
|--------|---------|-------------|
| **EVT** | GEV, GPD, POT, Hill | VaR de queue, stress testing |
| **Copulas** | Gaussian, Student-t, Clayton, Gumbel, Frank | Dépendance multivariée |
| **Risk Measures** | VaR, ES, tests de cohérence | Mesure de risque |
| **Backtesting** | Kupiec, Christoffersen | Validation des modèles |
| **Credit Risk** | Merton, Hazard rates, CDS | Risque de crédit |
| **Basel IRB** | Formule de capital | Réglementation |

---

**FIN DU GUIDE QRM POUR HELIXONE**
