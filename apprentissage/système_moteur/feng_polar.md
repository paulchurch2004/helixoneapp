# 📊 Ingénierie Financière : Une Perspective du Traitement du Signal

## Guide Complet avec Code Python

**Basé sur** : "A Signal Processing Perspective on Financial Engineering" - Feng & Palomar (2016)

---

## 📑 Table des Matières

1. [Introduction et Vue d'Ensemble](#1-introduction)
2. [Modélisation des Séries Temporelles Financières](#2-modélisation)
3. [Estimation des Paramètres (Moyenne et Covariance)](#3-estimation)
4. [Optimisation de Portefeuille](#4-portefeuille)
5. [Arbitrage Statistique](#5-arbitrage)
6. [Exécution d'Ordres](#6-execution)

---

## 1. Introduction et Vue d'Ensemble {#1-introduction}

### 1.1 Philosophie du Document

L'ingénierie financière et le traitement du signal partagent des fondations mathématiques communes :

| Ingénierie Financière | Traitement du Signal |
|----------------------|---------------------|
| Modèle ARMA (AutoRegressive Moving Average) | Modèle pôle-zéro rationnel |
| Estimateur de covariance par shrinkage | Diagonal loading en beamforming |
| Optimisation de portefeuille | Design de filtre/beamforming |
| Index tracking sparse | Récupération de signaux sparse |

### 1.2 Les Trois Piliers de l'Investissement Quantitatif

```
┌─────────────────────────────────┐
│     Modélisation Financière     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│   Stratégies d'Investissement   │
│  ┌───────────┐  ┌────────────┐  │
│  │ Portfolio │  │ Arbitrage  │  │
│  │   Optim.  │  │Statistique │  │
│  └───────────┘  └────────────┘  │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│      Exécution d'Ordres         │
└─────────────────────────────────┘
```

---

## 2. Modélisation des Séries Temporelles Financières {#2-modélisation}

### 2.1 Prix et Rendements

```python
"""
RENDEMENTS ET LOG-RENDEMENTS
============================

En finance, on travaille avec deux types de rendements :
- Rendement simple (linéaire) : R_t = (p_t - p_{t-1}) / p_{t-1}
- Log-rendement : r_t = log(p_t / p_{t-1}) = log(p_t) - log(p_{t-1})

Pourquoi les log-rendements ?
1. Additivité temporelle : r_t(k) = r_t + r_{t-1} + ... + r_{t-k+1}
2. Propriétés statistiques plus simples (distribution plus symétrique)
3. Plus faciles à modéliser
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def compute_returns(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les rendements simples et log-rendements à partir des prix.
    
    Le rendement simple mesure le profit relatif :
        R_t = (prix_final - prix_initial) / prix_initial
    
    Le log-rendement est l'approximation continue :
        r_t = ln(1 + R_t) ≈ R_t pour R_t petit
    
    Args:
        prices: Array des prix (shape: [T,] ou [T, N])
        
    Returns:
        (simple_returns, log_returns): Tuple de deux arrays
        
    Exemple:
        >>> prices = np.array([100, 105, 103, 108])
        >>> simple, log = compute_returns(prices)
        >>> print(f"Rendement simple jour 1: {simple[0]:.2%}")  # 5%
        >>> print(f"Log-rendement jour 1: {log[0]:.4f}")        # ~0.0488
    """
    prices = np.asarray(prices)
    
    # Rendement simple: R_t = p_t/p_{t-1} - 1
    simple_returns = prices[1:] / prices[:-1] - 1
    
    # Log-rendement: r_t = log(p_t) - log(p_{t-1})
    log_returns = np.log(prices[1:]) - np.log(prices[:-1])
    
    return simple_returns, log_returns


def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """
    Calcule le rendement d'un portefeuille.
    
    Le rendement d'un portefeuille est la somme pondérée des rendements :
        R_p = Σ w_i * R_i = w^T * R
    
    Args:
        weights: Vecteur des poids (doit sommer à 1 pour un portefeuille long-only)
        returns: Vecteur des rendements des actifs
        
    Returns:
        Rendement du portefeuille
        
    Exemple:
        >>> w = np.array([0.6, 0.4])  # 60% action A, 40% action B
        >>> r = np.array([0.05, -0.02])  # A: +5%, B: -2%
        >>> portfolio_return(w, r)  # 0.6*0.05 + 0.4*(-0.02) = 0.022 = 2.2%
    """
    return np.dot(weights, returns)
```

### 2.2 Structure Générale d'un Modèle

```python
"""
STRUCTURE GÉNÉRALE DES MODÈLES FINANCIERS
=========================================

La plupart des modèles décomposent le log-rendement r_t comme :

    r_t = μ_t + w_t

Où :
- μ_t = E[r_t | F_{t-1}] : moyenne conditionnelle (partie prévisible)
- w_t : bruit blanc avec covariance Σ_t (partie imprévisible)

Les modèles diffèrent par la façon dont ils spécifient μ_t et Σ_t.
"""

@dataclass
class FinancialModel:
    """
    Classe de base pour les modèles financiers.
    
    Un modèle financier doit pouvoir :
    1. Estimer ses paramètres à partir de données
    2. Prédire la moyenne conditionnelle μ_t
    3. Prédire la covariance conditionnelle Σ_t
    """
    
    def fit(self, returns: np.ndarray) -> 'FinancialModel':
        """Estime les paramètres du modèle."""
        raise NotImplementedError
    
    def predict_mean(self, history: np.ndarray) -> np.ndarray:
        """Prédit la moyenne conditionnelle μ_t."""
        raise NotImplementedError
    
    def predict_covariance(self, history: np.ndarray) -> np.ndarray:
        """Prédit la covariance conditionnelle Σ_t."""
        raise NotImplementedError
```

### 2.3 Modèle I.I.D. (Independent and Identically Distributed)

```python
"""
MODÈLE I.I.D.
=============

Le modèle le plus simple : les rendements sont i.i.d. (indépendants et 
identiquement distribués) avec moyenne μ et covariance Σ constantes.

    r_t = μ + w_t,    w_t ~ N(0, Σ)

C'est l'hypothèse fondamentale de la théorie de Markowitz (Nobel 1990).

Avantages :
- Simple à comprendre et à estimer
- Base de nombreuses théories fondamentales

Inconvénients :
- Ignore la dépendance temporelle
- Ignore la volatilité variable (clustering de volatilité)
"""

class IIDModel(FinancialModel):
    """
    Modèle I.I.D. : rendements indépendants avec moyenne et variance constantes.
    
    C'est le modèle utilisé dans l'optimisation de portefeuille classique
    de Markowitz (Mean-Variance Optimization).
    
    Attributs:
        mu: Moyenne des rendements (vecteur N×1)
        Sigma: Matrice de covariance (N×N)
    """
    
    def __init__(self):
        self.mu: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None
    
    def fit(self, returns: np.ndarray) -> 'IIDModel':
        """
        Estime μ et Σ par les estimateurs classiques.
        
        μ̂ = (1/T) Σ r_t           (moyenne empirique)
        Σ̂ = (1/T) Σ (r_t - μ̂)(r_t - μ̂)^T  (covariance empirique)
        
        Args:
            returns: Matrice T×N des rendements (T observations, N actifs)
            
        Returns:
            self (pour chaînage)
        """
        # Moyenne empirique (Sample Mean)
        self.mu = np.mean(returns, axis=0)
        
        # Covariance empirique (Sample Covariance Matrix - SCM)
        # Utilise ddof=0 pour être cohérent avec la formule MLE
        self.Sigma = np.cov(returns, rowvar=False, ddof=0)
        
        return self
    
    def predict_mean(self, history: np.ndarray = None) -> np.ndarray:
        """La moyenne conditionnelle est constante = μ."""
        return self.mu
    
    def predict_covariance(self, history: np.ndarray = None) -> np.ndarray:
        """La covariance conditionnelle est constante = Σ."""
        return self.Sigma


# Démonstration
def demo_iid_model():
    """Démontre le modèle I.I.D. avec des données simulées."""
    np.random.seed(42)
    
    # Paramètres vrais
    true_mu = np.array([0.001, 0.002, 0.0015])  # 0.1%, 0.2%, 0.15% par jour
    true_Sigma = np.array([
        [0.0004, 0.0002, 0.0001],
        [0.0002, 0.0006, 0.0002],
        [0.0001, 0.0002, 0.0005]
    ])
    
    # Générer T=500 observations
    T = 500
    returns = np.random.multivariate_normal(true_mu, true_Sigma, size=T)
    
    # Estimer le modèle
    model = IIDModel().fit(returns)
    
    print("=== Modèle I.I.D. ===")
    print(f"Moyenne estimée: {model.mu}")
    print(f"Vraie moyenne:   {true_mu}")
    print(f"\nCovariance estimée:\n{model.Sigma}")
    print(f"\nVraie covariance:\n{true_Sigma}")
    
    return model
```

### 2.4 Modèle Factoriel

```python
"""
MODÈLE FACTORIEL
================

Idée clé : Le marché est de grande dimension (N actifs), mais il est 
réellement "piloté" par un petit nombre K de facteurs (K << N).

    r_t = φ_0 + Π * f_t + w_t

Où :
- φ_0 : constante (N×1)
- f_t : vecteur des K facteurs (K×1)
- Π : matrice de chargement des facteurs (N×K)
- w_t : bruit idiosyncratique (spécifique à chaque actif)

Exemples de facteurs explicites :
- Rendement du marché (CAPM : Capital Asset Pricing Model)
- Taille de l'entreprise, ratio book-to-market (Fama-French)
- Momentum, volatilité (facteurs multi-facteurs)

Exemples de facteurs cachés :
- Composantes principales (PCA : Principal Component Analysis)
"""

class FactorModel(FinancialModel):
    """
    Modèle factoriel pour les rendements d'actifs.
    
    Décompose les rendements en :
    - Composante systématique : Π * f_t (expliquée par les facteurs)
    - Composante idiosyncratique : w_t (spécifique à chaque actif)
    
    La covariance des rendements se décompose comme :
        Σ = Π * Σ_f * Π^T + Σ_w
    
    où Σ_f est la covariance des facteurs et Σ_w celle des résidus.
    """
    
    def __init__(self, n_factors: int = 3):
        """
        Args:
            n_factors: Nombre K de facteurs à utiliser
        """
        self.n_factors = n_factors
        self.phi0: Optional[np.ndarray] = None  # Constante
        self.Pi: Optional[np.ndarray] = None    # Chargements (loadings)
        self.Sigma_f: Optional[np.ndarray] = None  # Covariance des facteurs
        self.Sigma_w: Optional[np.ndarray] = None  # Covariance résiduelle
    
    def fit_with_explicit_factors(
        self, 
        returns: np.ndarray, 
        factors: np.ndarray
    ) -> 'FactorModel':
        """
        Estime le modèle avec des facteurs explicites (observables).
        
        Utilise la régression linéaire :
            r_t = φ_0 + Π * f_t + w_t
        
        Args:
            returns: Matrice T×N des rendements
            factors: Matrice T×K des facteurs
            
        Returns:
            self
        """
        T, N = returns.shape
        K = factors.shape[1]
        
        # Ajouter une constante aux facteurs pour la régression
        X = np.column_stack([np.ones(T), factors])  # T × (K+1)
        
        # Régression OLS (Ordinary Least Squares) : β = (X^T X)^{-1} X^T Y
        beta = np.linalg.lstsq(X, returns, rcond=None)[0]
        
        self.phi0 = beta[0, :]  # Intercept (N,)
        self.Pi = beta[1:, :].T  # Chargements (N × K)
        
        # Résidus
        residuals = returns - X @ beta
        
        # Covariances
        self.Sigma_f = np.cov(factors, rowvar=False, ddof=0)
        self.Sigma_w = np.cov(residuals, rowvar=False, ddof=0)
        
        return self
    
    def fit_with_pca(self, returns: np.ndarray) -> 'FactorModel':
        """
        Estime le modèle avec des facteurs cachés via PCA.
        
        PCA (Principal Component Analysis) trouve les directions de 
        variance maximale dans les données.
        
        Les K premières composantes principales deviennent les facteurs.
        
        Args:
            returns: Matrice T×N des rendements
            
        Returns:
            self
        """
        T, N = returns.shape
        K = self.n_factors
        
        # Centrer les données
        mu = np.mean(returns, axis=0)
        returns_centered = returns - mu
        
        # Covariance empirique
        Sigma_emp = np.cov(returns_centered, rowvar=False, ddof=0)
        
        # Décomposition en valeurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_emp)
        
        # Trier par ordre décroissant
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Garder les K premiers
        E_K = eigenvectors[:, :K]  # N × K
        Lambda_K = np.diag(eigenvalues[:K])  # K × K
        
        # Dans le modèle PCA :
        # - Les chargements sont les vecteurs propres
        # - Les facteurs sont les projections : f_t = E_K^T * r_t
        self.Pi = E_K  # N × K
        
        # Reconstruction de la covariance
        # Σ = E_K * Λ_K * E_K^T + Σ_w
        self.Sigma_f = Lambda_K
        reconstructed = E_K @ Lambda_K @ E_K.T
        self.Sigma_w = Sigma_emp - reconstructed
        
        # Forcer la positivité de la covariance résiduelle
        # (peut être négative à cause des erreurs numériques)
        self.Sigma_w = np.maximum(self.Sigma_w, 0)
        
        self.phi0 = mu
        
        return self
    
    def get_covariance(self) -> np.ndarray:
        """
        Retourne la matrice de covariance implicite du modèle.
        
        Σ = Π * Σ_f * Π^T + Σ_w
        """
        return self.Pi @ self.Sigma_f @ self.Pi.T + self.Sigma_w


class CAPM:
    """
    CAPM : Capital Asset Pricing Model
    
    Le modèle à un facteur le plus célèbre (Sharpe, 1964 - Nobel 1990).
    
    Pour chaque actif i :
        E[r_i] - r_f = β_i * (E[r_M] - r_f)
    
    Où :
    - r_f : taux sans risque
    - r_M : rendement du portefeuille de marché
    - β_i : sensibilité de l'actif au marché
    - E[r_M] - r_f : prime de risque du marché
    
    Le β mesure le risque systématique (non-diversifiable).
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Args:
            risk_free_rate: Taux sans risque (quotidien)
        """
        self.rf = risk_free_rate
        self.betas: Optional[np.ndarray] = None
        self.alphas: Optional[np.ndarray] = None  # Intercepts (devraient être ~0)
    
    def fit(
        self, 
        asset_returns: np.ndarray, 
        market_returns: np.ndarray
    ) -> 'CAPM':
        """
        Estime les betas par régression.
        
        r_i,t - r_f = α_i + β_i * (r_M,t - r_f) + ε_i,t
        
        Args:
            asset_returns: Rendements des actifs (T × N)
            market_returns: Rendements du marché (T,)
        """
        T, N = asset_returns.shape
        
        # Excès de rendements
        excess_asset = asset_returns - self.rf
        excess_market = market_returns - self.rf
        
        self.betas = np.zeros(N)
        self.alphas = np.zeros(N)
        
        for i in range(N):
            # Régression simple : y = α + β*x
            # β = Cov(y, x) / Var(x)
            cov = np.cov(excess_asset[:, i], excess_market)[0, 1]
            var_market = np.var(excess_market)
            
            self.betas[i] = cov / var_market
            self.alphas[i] = np.mean(excess_asset[:, i]) - self.betas[i] * np.mean(excess_market)
        
        return self
    
    def expected_return(self, market_premium: float) -> np.ndarray:
        """
        Calcule le rendement espéré selon le CAPM.
        
        E[r_i] = r_f + β_i * (E[r_M] - r_f)
        
        Args:
            market_premium: E[r_M] - r_f (prime de risque du marché)
        """
        return self.rf + self.betas * market_premium
```

### 2.5 Modèles ARMA et VAR

```python
"""
MODÈLES VARMA (Vector AutoRegressive Moving Average)
====================================================

Ces modèles capturent la dépendance temporelle dans les rendements.

VAR(p) - Vector AutoRegressive d'ordre p :
    r_t = φ_0 + Φ_1 * r_{t-1} + ... + Φ_p * r_{t-p} + w_t

Les matrices Φ_i capturent comment les rendements passés 
affectent les rendements présents.

VMA(q) - Vector Moving Average d'ordre q :
    r_t = μ + w_t - Θ_1 * w_{t-1} - ... - Θ_q * w_{t-q}

Capture les "chocs" qui persistent quelques périodes.

VARMA(p,q) combine les deux.
"""

class VARModel(FinancialModel):
    """
    Modèle VAR(p) : Vector AutoRegressive d'ordre p.
    
    r_t = φ_0 + Σ_{i=1}^p Φ_i * r_{t-i} + w_t
    
    La moyenne conditionnelle dépend des p observations passées :
        μ_t = φ_0 + Σ Φ_i * r_{t-i}
    
    La covariance conditionnelle reste constante :
        Σ_t = Σ_w
    """
    
    def __init__(self, order: int = 1):
        """
        Args:
            order: Ordre p du modèle (nombre de lags)
        """
        self.p = order
        self.phi0: Optional[np.ndarray] = None
        self.Phi: Optional[List[np.ndarray]] = None  # Liste des Φ_i
        self.Sigma_w: Optional[np.ndarray] = None
    
    def fit(self, returns: np.ndarray) -> 'VARModel':
        """
        Estime le VAR(p) par OLS (Ordinary Least Squares).
        
        On réécrit le modèle en régression :
            r_t = [1, r_{t-1}^T, ..., r_{t-p}^T] * β + w_t
        
        Args:
            returns: Matrice T×N des rendements
        """
        T, N = returns.shape
        p = self.p
        
        if T <= p:
            raise ValueError(f"Pas assez d'observations. T={T} <= p={p}")
        
        # Construire les matrices pour la régression
        # Y = [r_p, r_{p+1}, ..., r_{T-1}]^T  de taille (T-p) × N
        Y = returns[p:]
        
        # X = [1, r_{t-1}, ..., r_{t-p}] de taille (T-p) × (1 + p*N)
        X = np.ones((T - p, 1 + p * N))
        for t in range(p, T):
            for lag in range(1, p + 1):
                start_col = 1 + (lag - 1) * N
                end_col = 1 + lag * N
                X[t - p, start_col:end_col] = returns[t - lag]
        
        # OLS : β = (X^T X)^{-1} X^T Y
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        # Extraire les paramètres
        self.phi0 = beta[0]
        self.Phi = []
        for lag in range(1, p + 1):
            start = 1 + (lag - 1) * N
            end = 1 + lag * N
            self.Phi.append(beta[start:end].T)  # N × N
        
        # Résidus et covariance
        residuals = Y - X @ beta
        self.Sigma_w = np.cov(residuals, rowvar=False, ddof=0)
        
        return self
    
    def predict_mean(self, history: np.ndarray) -> np.ndarray:
        """
        Prédit la moyenne conditionnelle.
        
        μ_t = φ_0 + Σ Φ_i * r_{t-i}
        
        Args:
            history: Les p dernières observations (p × N)
        """
        mu = self.phi0.copy()
        for i, Phi_i in enumerate(self.Phi):
            mu += Phi_i @ history[-(i + 1)]
        return mu
    
    def predict_covariance(self, history: np.ndarray = None) -> np.ndarray:
        """La covariance est constante dans un VAR."""
        return self.Sigma_w
```

### 2.6 Modèles de Volatilité (GARCH)

```python
"""
MODÈLES DE VOLATILITÉ CONDITIONNELLE
====================================

Les modèles précédents supposent Σ_t constant. En réalité, 
la volatilité varie dans le temps !

Faits stylisés de la volatilité financière :
1. Clustering : les périodes de haute volatilité se regroupent
2. Mean-reversion : la volatilité revient vers un niveau moyen
3. Asymétrie (leverage effect) : les baisses causent plus de volatilité

ARCH(m) - AutoRegressive Conditional Heteroskedasticity (Engle, 1982) :
    σ²_t = α_0 + Σ_{i=1}^m α_i * w²_{t-i}

La variance dépend des chocs passés au carré.

GARCH(m,s) - Generalized ARCH (Bollerslev, 1986) :
    σ²_t = α_0 + Σ_{i=1}^m α_i * w²_{t-i} + Σ_{j=1}^s β_j * σ²_{t-j}

Ajoute une composante autorégressive sur la variance elle-même.
"""

class GARCH11:
    """
    Modèle GARCH(1,1) univarié.
    
    σ²_t = ω + α * w²_{t-1} + β * σ²_{t-1}
    
    Où :
    - ω > 0 : constante
    - α ≥ 0 : coefficient ARCH (impact des chocs passés)
    - β ≥ 0 : coefficient GARCH (persistence de la volatilité)
    - α + β < 1 : condition de stationnarité
    
    La variance long-terme est : σ² = ω / (1 - α - β)
    
    C'est le modèle de volatilité le plus utilisé en pratique.
    """
    
    def __init__(self):
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.beta: float = 0.0
        self.mu: float = 0.0  # Moyenne des rendements
    
    def fit(self, returns: np.ndarray, method: str = 'mle') -> 'GARCH11':
        """
        Estime les paramètres par MLE (Maximum Likelihood Estimation).
        
        La log-vraisemblance gaussienne conditionnelle est :
            L = -0.5 * Σ [log(σ²_t) + w²_t/σ²_t]
        
        Args:
            returns: Vecteur des rendements (T,)
            method: 'mle' ou 'moment' (méthode des moments)
        """
        returns = np.asarray(returns).flatten()
        T = len(returns)
        
        # Moyenne
        self.mu = np.mean(returns)
        residuals = returns - self.mu
        
        # Variance inconditionnelle (pour initialisation)
        var_unconditional = np.var(residuals)
        
        def negative_log_likelihood(params):
            """Négatif de la log-vraisemblance (à minimiser)."""
            omega, alpha, beta = params
            
            # Contraintes
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            sigma2 = np.zeros(T)
            sigma2[0] = var_unconditional
            
            for t in range(1, T):
                sigma2[t] = omega + alpha * residuals[t-1]**2 + beta * sigma2[t-1]
            
            # Log-vraisemblance gaussienne
            ll = -0.5 * np.sum(np.log(sigma2) + residuals**2 / sigma2)
            
            return -ll  # On minimise le négatif
        
        # Initialisation
        x0 = [var_unconditional * 0.05, 0.05, 0.90]
        
        # Optimisation
        result = minimize(
            negative_log_likelihood,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        self.omega, self.alpha, self.beta = result.x
        
        return self
    
    def forecast_variance(
        self, 
        last_return: float, 
        last_variance: float,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Prévision de la variance sur un horizon donné.
        
        σ²_{t+h|t} = σ² + (α + β)^{h-1} * (σ²_{t+1|t} - σ²)
        
        où σ² = ω/(1-α-β) est la variance long-terme.
        
        Args:
            last_return: Dernier rendement observé
            last_variance: Dernière variance
            horizon: Horizon de prévision
            
        Returns:
            Array des variances prévues
        """
        forecasts = np.zeros(horizon)
        
        # Variance long-terme
        var_lt = self.omega / (1 - self.alpha - self.beta)
        
        # Première prévision
        residual = last_return - self.mu
        forecasts[0] = self.omega + self.alpha * residual**2 + self.beta * last_variance
        
        # Prévisions suivantes (convergent vers var_lt)
        persistence = self.alpha + self.beta
        for h in range(1, horizon):
            forecasts[h] = var_lt + persistence**h * (forecasts[0] - var_lt)
        
        return forecasts
    
    @property
    def long_term_variance(self) -> float:
        """Variance long-terme (inconditionnelle)."""
        return self.omega / (1 - self.alpha - self.beta)
    
    @property
    def half_life(self) -> float:
        """
        Demi-vie de la volatilité.
        
        Nombre de périodes pour que la moitié du choc soit absorbée.
        """
        persistence = self.alpha + self.beta
        if persistence >= 1:
            return float('inf')
        return np.log(0.5) / np.log(persistence)


def demo_garch():
    """Démontre le modèle GARCH(1,1)."""
    np.random.seed(42)
    
    # Simuler un processus GARCH(1,1)
    T = 1000
    omega, alpha, beta = 0.00001, 0.05, 0.93
    
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)  # Variance long-terme
    
    for t in range(1, T):
        returns[t] = np.sqrt(sigma2[t-1]) * np.random.randn()
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    
    # Estimer le modèle
    model = GARCH11().fit(returns)
    
    print("=== Modèle GARCH(1,1) ===")
    print(f"Paramètres vrais:    ω={omega:.6f}, α={alpha:.2f}, β={beta:.2f}")
    print(f"Paramètres estimés:  ω={model.omega:.6f}, α={model.alpha:.2f}, β={model.beta:.2f}")
    print(f"Persistence α+β: {model.alpha + model.beta:.4f}")
    print(f"Demi-vie: {model.half_life:.1f} périodes")
    
    return model
```

---

## 3. Estimation des Paramètres {#3-estimation}

### 3.1 Défis de l'Estimation en Finance

```python
"""
DÉFIS DE L'ESTIMATION EN FINANCE
================================

Deux problèmes majeurs rendent l'estimation difficile :

1. RÉGIME DE PETITS ÉCHANTILLONS (Small Sample Regime)
   - On a N actifs mais seulement T observations
   - Si T < N, la covariance empirique n'est pas inversible !
   - Même si T > N, l'estimation peut être très bruitée
   
   Exemple : 500 actifs du S&P 500, seulement 252 jours de trading/an
   → Pour 2 ans de données : T=504, N=500 → T ≈ N !

2. QUEUES ÉPAISSES (Heavy Tails)
   - Les rendements ne sont PAS gaussiens
   - Les événements extrêmes arrivent plus souvent que prévu par la loi normale
   - L'estimateur classique est très sensible aux outliers

SOLUTIONS :
- Estimateurs de shrinkage (régularisation)
- Estimateurs robustes (Huber, Tyler, etc.)
"""
```

### 3.2 Estimateurs de Shrinkage

```python
"""
ESTIMATEURS DE SHRINKAGE
========================

Idée : "Rétrécir" (shrink) l'estimateur vers une cible structurée.

Forme générale :
    θ̃ = ρ * T + (1 - ρ) * θ̂

Où :
- θ̂ : estimateur empirique (bruyant mais non biaisé)
- T : cible (biaisée mais stable)
- ρ : paramètre de shrinkage (compromis biais-variance)

En augmentant ρ :
- On réduit la variance
- On augmente le biais
- On améliore souvent le MSE (Mean Squared Error) total !

C'est l'équivalent du "diagonal loading" en beamforming.
"""

def shrinkage_mean(
    returns: np.ndarray,
    target: Optional[np.ndarray] = None,
    Sigma: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Estimateur de shrinkage de James-Stein pour la moyenne.
    
    L'estimateur de James-Stein (1961) montre que la moyenne empirique
    est DOMINÉE (au sens du MSE) par un estimateur de shrinkage
    quand la dimension N ≥ 3 !
    
    μ̃ = ρ * b + (1 - ρ) * μ̂
    
    Args:
        returns: Matrice T×N des rendements
        target: Cible b (par défaut : grand mean)
        Sigma: Covariance vraie (si connue)
        
    Returns:
        Moyenne shrinkée
    """
    T, N = returns.shape
    
    # Moyenne empirique
    mu_hat = np.mean(returns, axis=0)
    
    # Cible par défaut : grand mean (moyenne des moyennes)
    if target is None:
        target = np.ones(N) * np.mean(mu_hat)
    
    # Covariance (estimée si pas fournie)
    if Sigma is None:
        Sigma = np.cov(returns, rowvar=False, ddof=1)
    
    # Paramètre de shrinkage optimal (formule de James-Stein)
    eigenvalues = np.linalg.eigvalsh(Sigma)
    lambda_avg = np.mean(eigenvalues)
    lambda_max = np.max(eigenvalues)
    
    diff = mu_hat - target
    diff_norm_sq = np.dot(diff, diff)
    
    if diff_norm_sq > 1e-10:
        rho = (1 / T) * (N * lambda_avg - 2 * lambda_max) / diff_norm_sq
        rho = max(0, min(1, rho))  # Borner entre 0 et 1
    else:
        rho = 0
    
    return rho * target + (1 - rho) * mu_hat


class LedoitWolfShrinkage:
    """
    Estimateur de Ledoit-Wolf pour la matrice de covariance.
    
    Shrink la covariance empirique vers une cible simple (souvent λ*I).
    
    Σ̃ = ρ * λ̃ * I + (1 - ρ) * Σ̂
    
    Où :
    - Σ̂ : covariance empirique
    - λ̃ = Tr(Σ)/N : moyenne des variances
    - ρ : paramètre optimal minimisant E[||Σ̃ - Σ||²_F]
    
    Formule de Ledoit-Wolf (2004) pour ρ optimal :
    
    ρ = min(1, (1/T) * Σ ||r_t r_t^T - Σ̂||²_F / ||Σ̂ - λ̃I||²_F)
    """
    
    def __init__(self):
        self.rho: float = 0.0
        self.lambda_: float = 0.0
        self.Sigma_shrunk: Optional[np.ndarray] = None
    
    def fit(self, returns: np.ndarray) -> 'LedoitWolfShrinkage':
        """
        Calcule l'estimateur de Ledoit-Wolf.
        
        Args:
            returns: Matrice T×N des rendements (déjà centrés ou non)
            
        Returns:
            self
        """
        T, N = returns.shape
        
        # Centrer les données
        mean = np.mean(returns, axis=0)
        X = returns - mean  # T × N
        
        # Covariance empirique
        Sigma_hat = (X.T @ X) / T  # N × N
        
        # Cible : λ * I
        self.lambda_ = np.trace(Sigma_hat) / N
        
        # Calcul de δ² = ||Σ̂ - λI||²_F / N²
        delta_sq = np.sum((Sigma_hat - self.lambda_ * np.eye(N))**2) / N**2
        
        # Calcul de β² (terme de correction)
        # β² = (1/T²) * Σ_t ||x_t x_t^T - Σ̂||²_F
        beta_sq = 0.0
        for t in range(T):
            x_t = X[t:t+1].T  # N × 1
            sample_cov = x_t @ x_t.T  # N × N
            beta_sq += np.sum((sample_cov - Sigma_hat)**2)
        beta_sq = beta_sq / (T**2 * N**2)
        
        # Paramètre de shrinkage
        self.rho = min(1.0, beta_sq / delta_sq) if delta_sq > 0 else 1.0
        
        # Covariance shrinkée
        self.Sigma_shrunk = (
            self.rho * self.lambda_ * np.eye(N) + 
            (1 - self.rho) * Sigma_hat
        )
        
        return self
    
    def get_covariance(self) -> np.ndarray:
        """Retourne la covariance shrinkée."""
        return self.Sigma_shrunk


def demo_shrinkage():
    """Compare la covariance empirique et Ledoit-Wolf."""
    np.random.seed(42)
    
    # Vraie covariance
    N = 50
    true_Sigma = np.eye(N)
    for i in range(N):
        for j in range(N):
            true_Sigma[i, j] = 0.5 ** abs(i - j)  # Structure AR(1)
    
    # Générer des données (peu d'échantillons)
    T = 60  # T proche de N !
    returns = np.random.multivariate_normal(np.zeros(N), true_Sigma, size=T)
    
    # Covariance empirique
    Sigma_hat = np.cov(returns, rowvar=False, ddof=0)
    
    # Ledoit-Wolf
    lw = LedoitWolfShrinkage().fit(returns)
    
    # Erreurs
    error_scm = np.linalg.norm(Sigma_hat - true_Sigma, 'fro')
    error_lw = np.linalg.norm(lw.Sigma_shrunk - true_Sigma, 'fro')
    
    print("=== Comparaison Shrinkage ===")
    print(f"N = {N}, T = {T}")
    print(f"Erreur covariance empirique : {error_scm:.4f}")
    print(f"Erreur Ledoit-Wolf :          {error_lw:.4f}")
    print(f"Amélioration : {(error_scm - error_lw) / error_scm * 100:.1f}%")
    print(f"Paramètre de shrinkage ρ : {lw.rho:.4f}")
    
    return lw
```

### 3.3 Estimateurs Robustes

```python
"""
ESTIMATEURS ROBUSTES
====================

Les estimateurs classiques (moyenne, covariance) sont sensibles aux outliers.
Les estimateurs robustes downweight les observations extrêmes.

M-ESTIMATEURS :
Généralisent le MLE (Maximum Likelihood Estimator) avec des poids adaptatifs.

    μ = Σ w_1(d_t) * r_t / Σ w_1(d_t)
    Σ = (1/T) Σ w_2(d_t) * (r_t - μ)(r_t - μ)^T

Où d_t = (r_t - μ)^T Σ^{-1} (r_t - μ) est la distance de Mahalanobis.

Différents choix de w(d) donnent différents estimateurs :
- w(d) = 1 : estimateurs classiques (pas robustes)
- w(d) = (N+1)/(1+d) : MLE de Cauchy (robuste, queues très épaisses)
- w(d) = N/d : estimateur de Tyler (très robuste)
"""

class TylerEstimator:
    """
    Estimateur de Tyler pour la matrice de scatter (forme).
    
    C'est l'estimateur MLE de la distribution "Angular Gaussian",
    qui est invariant à l'échelle des observations.
    
    L'estimateur de Tyler est très robuste car il ne dépend que
    des DIRECTIONS des observations, pas de leurs normes.
    
    Équation du point fixe :
        Σ = (N/T) Σ_t [r_t r_t^T / (r_t^T Σ^{-1} r_t)]
    
    Conditions d'existence : T ≥ N + 1
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        """
        Args:
            max_iter: Nombre maximum d'itérations
            tol: Tolérance pour la convergence
        """
        self.max_iter = max_iter
        self.tol = tol
        self.Sigma: Optional[np.ndarray] = None
    
    def fit(self, data: np.ndarray, mu: Optional[np.ndarray] = None) -> 'TylerEstimator':
        """
        Estime la matrice de scatter par l'algorithme itératif de Tyler.
        
        Args:
            data: Matrice T×N des observations
            mu: Moyenne (si connue, sinon supposée 0)
            
        Returns:
            self
        """
        T, N = data.shape
        
        if T < N + 1:
            raise ValueError(f"Pas assez d'échantillons. T={T} < N+1={N+1}")
        
        # Centrer si moyenne fournie
        if mu is not None:
            data = data - mu
        
        # Initialisation : identité
        Sigma = np.eye(N)
        
        for iteration in range(self.max_iter):
            Sigma_old = Sigma.copy()
            
            # Calcul de la nouvelle estimation
            Sigma_inv = np.linalg.inv(Sigma)
            Sigma_new = np.zeros((N, N))
            
            for t in range(T):
                r_t = data[t]
                # Distance de Mahalanobis au carré
                d_t = r_t @ Sigma_inv @ r_t
                # Poids de Tyler : w(d) = N/d
                weight = N / d_t if d_t > 1e-10 else N / 1e-10
                Sigma_new += weight * np.outer(r_t, r_t)
            
            Sigma_new /= T
            
            # Normaliser par la trace (Tyler est défini à un scalaire près)
            Sigma = Sigma_new / np.trace(Sigma_new) * N
            
            # Vérifier la convergence
            diff = np.linalg.norm(Sigma - Sigma_old, 'fro')
            if diff < self.tol:
                break
        
        self.Sigma = Sigma
        return self
    
    def get_scatter(self) -> np.ndarray:
        """Retourne la matrice de scatter estimée."""
        return self.Sigma


class RegularizedTyler:
    """
    Estimateur de Tyler régularisé.
    
    Quand T < N, Tyler classique n'existe pas.
    On ajoute un terme de régularisation :
    
    Σ = (1/(1+α)) * Tyler_update + (α/(1+α)) * T
    
    Où T est une cible (souvent l'identité).
    
    C'est l'équivalent du shrinkage pour les estimateurs robustes.
    """
    
    def __init__(
        self, 
        alpha: float = 0.1,
        target: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        """
        Args:
            alpha: Paramètre de régularisation (α ≥ 0)
            target: Matrice cible (par défaut : identité)
            max_iter: Itérations max
            tol: Tolérance
        """
        self.alpha = alpha
        self.target = target
        self.max_iter = max_iter
        self.tol = tol
        self.Sigma: Optional[np.ndarray] = None
    
    def fit(self, data: np.ndarray) -> 'RegularizedTyler':
        """
        Estime la matrice de scatter régularisée.
        
        Args:
            data: Matrice T×N des observations (centrées)
        """
        T, N = data.shape
        
        # Cible par défaut : identité
        if self.target is None:
            target = np.eye(N)
        else:
            target = self.target
        
        # Initialisation
        Sigma = np.eye(N)
        alpha = self.alpha
        
        for iteration in range(self.max_iter):
            Sigma_old = Sigma.copy()
            
            # Tyler update
            Sigma_inv = np.linalg.inv(Sigma)
            tyler_part = np.zeros((N, N))
            
            for t in range(T):
                r_t = data[t]
                d_t = r_t @ Sigma_inv @ r_t
                weight = N / d_t if d_t > 1e-10 else N / 1e-10
                tyler_part += weight * np.outer(r_t, r_t)
            
            tyler_part /= T
            
            # Combinaison avec la cible
            Sigma = (1 / (1 + alpha)) * tyler_part + (alpha / (1 + alpha)) * target
            
            # Convergence
            diff = np.linalg.norm(Sigma - Sigma_old, 'fro')
            if diff < self.tol:
                break
        
        self.Sigma = Sigma
        return self


def demo_robust_estimation():
    """Compare les estimateurs classiques et robustes avec outliers."""
    np.random.seed(42)
    
    N = 10
    T = 100
    
    # Vraie covariance
    true_Sigma = np.eye(N)
    
    # Générer des données normales
    data_clean = np.random.multivariate_normal(np.zeros(N), true_Sigma, size=T)
    
    # Ajouter des outliers (5%)
    n_outliers = int(0.05 * T)
    outlier_indices = np.random.choice(T, n_outliers, replace=False)
    data_with_outliers = data_clean.copy()
    data_with_outliers[outlier_indices] = np.random.multivariate_normal(
        np.ones(N) * 5, true_Sigma * 0.1, size=n_outliers
    )
    
    # Estimations
    scm_clean = np.cov(data_clean, rowvar=False, ddof=0)
    scm_outliers = np.cov(data_with_outliers, rowvar=False, ddof=0)
    
    tyler = TylerEstimator().fit(data_with_outliers)
    
    # Erreurs
    error_scm_clean = np.linalg.norm(scm_clean - true_Sigma, 'fro')
    error_scm_outliers = np.linalg.norm(scm_outliers - true_Sigma, 'fro')
    error_tyler = np.linalg.norm(tyler.Sigma - true_Sigma, 'fro')
    
    print("=== Estimation Robuste ===")
    print(f"Erreur SCM (données propres) :  {error_scm_clean:.4f}")
    print(f"Erreur SCM (avec outliers) :    {error_scm_outliers:.4f}")
    print(f"Erreur Tyler (avec outliers) :  {error_tyler:.4f}")
    
    return tyler
```

---

## 4. Optimisation de Portefeuille {#4-portefeuille}

### 4.1 Framework de Markowitz

```python
"""
OPTIMISATION DE PORTEFEUILLE DE MARKOWITZ
=========================================

Harry Markowitz (1952, Nobel 1990) a formalisé le compromis rendement-risque.

Problème de base :
    max  w^T μ - (λ/2) w^T Σ w
    s.t. w^T 1 = 1

Où :
- w : vecteur des poids du portefeuille
- μ : vecteur des rendements espérés
- Σ : matrice de covariance des rendements
- λ : paramètre d'aversion au risque

Interprétation :
- w^T μ : rendement espéré du portefeuille
- w^T Σ w : variance (risque) du portefeuille
- λ : combien on "sacrifie" de rendement pour réduire le risque

FRONTIÈRE EFFICIENTE :
L'ensemble des portefeuilles optimaux forme une hyperbole.
Aucun portefeuille ne peut avoir plus de rendement pour le même risque,
ou moins de risque pour le même rendement.
"""

class MarkowitzOptimizer:
    """
    Optimisation de portefeuille Mean-Variance de Markowitz.
    
    Résout plusieurs variantes du problème :
    1. Minimum Variance Portfolio (MVP)
    2. Maximum Sharpe Ratio (tangency portfolio)
    3. Mean-Variance optimal avec λ fixé
    4. Target return portfolio
    """
    
    def __init__(
        self, 
        mu: np.ndarray, 
        Sigma: np.ndarray,
        risk_free_rate: float = 0.0
    ):
        """
        Args:
            mu: Vecteur des rendements espérés (N,)
            Sigma: Matrice de covariance (N×N)
            risk_free_rate: Taux sans risque
        """
        self.mu = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        self.rf = risk_free_rate
        self.N = len(mu)
    
    def minimum_variance_portfolio(self) -> np.ndarray:
        """
        Portefeuille de variance minimale (MVP).
        
        min  w^T Σ w
        s.t. w^T 1 = 1
        
        Solution analytique :
            w_MVP = Σ^{-1} 1 / (1^T Σ^{-1} 1)
        
        Ce portefeuille ignore complètement les rendements espérés !
        Utile quand on ne fait pas confiance aux estimations de μ.
        
        Returns:
            Poids du MVP
        """
        ones = np.ones(self.N)
        Sigma_inv = np.linalg.inv(self.Sigma)
        
        w = Sigma_inv @ ones
        w = w / np.sum(w)  # Normaliser
        
        return w
    
    def maximum_sharpe_ratio(self) -> np.ndarray:
        """
        Portefeuille de Sharpe ratio maximal (tangency portfolio).
        
        max  (w^T μ - r_f) / sqrt(w^T Σ w)
        s.t. w^T 1 = 1
        
        Solution analytique :
            w_SR = Σ^{-1} (μ - r_f * 1) / [1^T Σ^{-1} (μ - r_f * 1)]
        
        C'est le portefeuille tangent à la frontière efficiente
        depuis le point (0, r_f).
        
        Returns:
            Poids du portefeuille de Sharpe max
        """
        Sigma_inv = np.linalg.inv(self.Sigma)
        excess_return = self.mu - self.rf
        
        w = Sigma_inv @ excess_return
        w = w / np.sum(w)  # Normaliser
        
        return w
    
    def mean_variance_optimal(self, risk_aversion: float) -> np.ndarray:
        """
        Portefeuille MV-optimal pour un niveau d'aversion au risque.
        
        max  w^T μ - (λ/2) w^T Σ w
        s.t. w^T 1 = 1
        
        Solution via multiplicateurs de Lagrange.
        
        Args:
            risk_aversion: Paramètre λ > 0
            
        Returns:
            Poids optimaux
        """
        lam = risk_aversion
        Sigma_inv = np.linalg.inv(self.Sigma)
        ones = np.ones(self.N)
        
        # Termes intermédiaires
        A = ones @ Sigma_inv @ self.mu
        B = self.mu @ Sigma_inv @ self.mu
        C = ones @ Sigma_inv @ ones
        
        # Multiplicateur de Lagrange
        gamma = (A - lam) / C
        
        # Solution
        w = (1/lam) * Sigma_inv @ (self.mu - gamma * ones)
        
        return w
    
    def target_return_portfolio(self, target_return: float) -> np.ndarray:
        """
        Portefeuille de variance minimale pour un rendement cible.
        
        min  w^T Σ w
        s.t. w^T μ = r_target
             w^T 1 = 1
        
        Args:
            target_return: Rendement cible
            
        Returns:
            Poids optimaux
        """
        Sigma_inv = np.linalg.inv(self.Sigma)
        ones = np.ones(self.N)
        
        # Termes pour la solution analytique
        A = ones @ Sigma_inv @ self.mu
        B = self.mu @ Sigma_inv @ self.mu
        C = ones @ Sigma_inv @ ones
        D = B * C - A**2
        
        # Multiplicateurs de Lagrange
        lambda1 = (C * target_return - A) / D
        lambda2 = (B - A * target_return) / D
        
        # Solution
        w = lambda1 * (Sigma_inv @ self.mu) + lambda2 * (Sigma_inv @ ones)
        
        return w
    
    def efficient_frontier(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la frontière efficiente.
        
        Args:
            n_points: Nombre de points sur la frontière
            
        Returns:
            (risks, returns): Arrays des risques et rendements
        """
        # Bornes des rendements
        w_mvp = self.minimum_variance_portfolio()
        min_return = w_mvp @ self.mu
        max_return = np.max(self.mu)  # Approximation
        
        target_returns = np.linspace(min_return, max_return, n_points)
        risks = []
        returns = []
        
        for r in target_returns:
            try:
                w = self.target_return_portfolio(r)
                portfolio_risk = np.sqrt(w @ self.Sigma @ w)
                risks.append(portfolio_risk)
                returns.append(r)
            except:
                pass
        
        return np.array(risks), np.array(returns)
    
    def portfolio_stats(self, weights: np.ndarray) -> dict:
        """
        Calcule les statistiques d'un portefeuille.
        
        Args:
            weights: Poids du portefeuille
            
        Returns:
            Dict avec return, risk, sharpe_ratio
        """
        port_return = weights @ self.mu
        port_risk = np.sqrt(weights @ self.Sigma @ weights)
        sharpe = (port_return - self.rf) / port_risk if port_risk > 0 else 0
        
        return {
            'return': port_return,
            'risk': port_risk,
            'sharpe_ratio': sharpe
        }


def demo_markowitz():
    """Démontre l'optimisation de Markowitz."""
    np.random.seed(42)
    
    # 5 actifs avec caractéristiques différentes
    mu = np.array([0.10, 0.12, 0.08, 0.15, 0.09])  # Rendements annuels
    
    # Matrice de covariance (corrélations réalistes)
    volatilities = np.array([0.15, 0.20, 0.10, 0.25, 0.12])
    correlations = np.array([
        [1.0, 0.5, 0.3, 0.4, 0.2],
        [0.5, 1.0, 0.4, 0.6, 0.3],
        [0.3, 0.4, 1.0, 0.3, 0.5],
        [0.4, 0.6, 0.3, 1.0, 0.4],
        [0.2, 0.3, 0.5, 0.4, 1.0]
    ])
    Sigma = np.outer(volatilities, volatilities) * correlations
    
    # Optimiser
    optimizer = MarkowitzOptimizer(mu, Sigma, risk_free_rate=0.02)
    
    w_mvp = optimizer.minimum_variance_portfolio()
    w_sharpe = optimizer.maximum_sharpe_ratio()
    w_mv = optimizer.mean_variance_optimal(risk_aversion=2.0)
    
    print("=== Optimisation de Portefeuille Markowitz ===")
    print("\nPoids du Minimum Variance Portfolio:")
    print(f"  {w_mvp}")
    print(f"  Stats: {optimizer.portfolio_stats(w_mvp)}")
    
    print("\nPoids du Maximum Sharpe Ratio Portfolio:")
    print(f"  {w_sharpe}")
    print(f"  Stats: {optimizer.portfolio_stats(w_sharpe)}")
    
    print("\nPoids du MV-Optimal (λ=2):")
    print(f"  {w_mv}")
    print(f"  Stats: {optimizer.portfolio_stats(w_mv)}")
    
    return optimizer
```

### 4.2 Optimisation Robuste

```python
"""
OPTIMISATION DE PORTEFEUILLE ROBUSTE
====================================

Le problème de Markowitz est TRÈS sensible aux erreurs d'estimation !

Problème : On ne connaît pas μ et Σ exacts, on les ESTIME.
Les erreurs d'estimation se propagent et amplifient les erreurs de décision.

SOLUTION : Optimisation robuste (worst-case optimization)

Idée : Au lieu d'optimiser pour μ̂ et Σ̂ estimés,
optimiser pour le PIRE CAS dans un ensemble d'incertitude.

    max  min        w^T μ - (λ/2) w^T Σ w
     w   (μ,Σ)∈U

Où U est l'ensemble d'incertitude autour des estimations.
"""

class RobustMarkowitz:
    """
    Optimisation de portefeuille robuste.
    
    Modélise l'incertitude sur μ et Σ et optimise pour le pire cas.
    
    Ensemble d'incertitude sur μ (ellipsoïdal) :
        U_μ = {μ : (μ - μ̂)^T Σ_μ^{-1} (μ - μ̂) ≤ κ_μ²}
    
    Cela donne une formulation robuste :
        max  w^T μ̂ - κ_μ ||Σ_μ^{1/2} w|| - (λ/2) w^T Σ̂ w
    """
    
    def __init__(
        self,
        mu_hat: np.ndarray,
        Sigma_hat: np.ndarray,
        kappa_mu: float = 1.0,
        Sigma_mu: Optional[np.ndarray] = None
    ):
        """
        Args:
            mu_hat: Estimation de μ
            Sigma_hat: Estimation de Σ
            kappa_mu: Rayon de l'ensemble d'incertitude sur μ
            Sigma_mu: Covariance de l'erreur d'estimation de μ
        """
        self.mu_hat = mu_hat
        self.Sigma_hat = Sigma_hat
        self.kappa_mu = kappa_mu
        self.N = len(mu_hat)
        
        # Par défaut, l'incertitude sur μ est proportionnelle à Σ
        if Sigma_mu is None:
            self.Sigma_mu = Sigma_hat / 100  # Heuristique
        else:
            self.Sigma_mu = Sigma_mu
    
    def robust_optimal(self, risk_aversion: float) -> np.ndarray:
        """
        Résout le problème robuste par optimisation numérique.
        
        Le worst-case (pire cas) sur μ dans l'ellipsoïde donne :
            μ_worst = μ̂ - κ_μ * Σ_μ^{1/2} * w / ||Σ_μ^{1/2} w||
        
        Args:
            risk_aversion: Paramètre λ
            
        Returns:
            Poids robustes
        """
        lam = risk_aversion
        
        # Racine carrée de Sigma_mu
        eigvals, eigvecs = np.linalg.eigh(self.Sigma_mu)
        Sigma_mu_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T
        
        def objective(w):
            """Objectif robuste (à maximiser, donc on retourne -objectif)."""
            # Terme de rendement espéré
            expected_return = w @ self.mu_hat
            
            # Pénalité robuste (pire cas sur μ)
            robust_penalty = self.kappa_mu * np.linalg.norm(Sigma_mu_sqrt @ w)
            
            # Terme de variance
            variance = w @ self.Sigma_hat @ w
            
            return -(expected_return - robust_penalty - 0.5 * lam * variance)
        
        # Contrainte : somme des poids = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Point de départ : équipondéré
        w0 = np.ones(self.N) / self.N
        
        # Optimisation
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            constraints=constraints
        )
        
        return result.x


def demo_robust_portfolio():
    """Compare portefeuille classique et robuste."""
    np.random.seed(42)
    
    N = 5
    
    # "Vrais" paramètres
    true_mu = np.array([0.10, 0.12, 0.08, 0.15, 0.09])
    volatilities = np.array([0.15, 0.20, 0.10, 0.25, 0.12])
    corr = np.array([
        [1.0, 0.5, 0.3, 0.4, 0.2],
        [0.5, 1.0, 0.4, 0.6, 0.3],
        [0.3, 0.4, 1.0, 0.3, 0.5],
        [0.4, 0.6, 0.3, 1.0, 0.4],
        [0.2, 0.3, 0.5, 0.4, 1.0]
    ])
    true_Sigma = np.outer(volatilities, volatilities) * corr
    
    # Estimations bruitées (simulant l'erreur d'estimation)
    mu_hat = true_mu + np.random.randn(N) * 0.03
    Sigma_hat = true_Sigma * (1 + np.random.randn(N, N) * 0.1)
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2  # Symétriser
    
    # Portefeuille classique
    classic = MarkowitzOptimizer(mu_hat, Sigma_hat)
    w_classic = classic.mean_variance_optimal(risk_aversion=2.0)
    
    # Portefeuille robuste
    robust = RobustMarkowitz(mu_hat, Sigma_hat, kappa_mu=1.5)
    w_robust = robust.robust_optimal(risk_aversion=2.0)
    
    # Évaluation avec les VRAIS paramètres
    true_optimizer = MarkowitzOptimizer(true_mu, true_Sigma)
    
    print("=== Comparaison Classique vs Robuste ===")
    print(f"\nPortefeuille classique:")
    print(f"  Poids: {w_classic}")
    print(f"  Performance vraie: {true_optimizer.portfolio_stats(w_classic)}")
    
    print(f"\nPortefeuille robuste:")
    print(f"  Poids: {w_robust}")
    print(f"  Performance vraie: {true_optimizer.portfolio_stats(w_robust)}")
    
    return w_classic, w_robust
```

### 4.3 Risk Parity

```python
"""
PORTEFEUILLE RISK PARITY
========================

Idée : Au lieu d'égaliser les CAPITAUX (équipondéré),
égaliser les CONTRIBUTIONS AU RISQUE.

Définitions :
- Risque du portefeuille : σ_p = sqrt(w^T Σ w)
- Contribution marginale au risque de l'actif i : ∂σ_p/∂w_i = (Σw)_i / σ_p
- Contribution au risque de l'actif i : RC_i = w_i * (Σw)_i / σ_p

Risk Parity demande : RC_1 = RC_2 = ... = RC_N

C'est équivalent à : w_i * (Σw)_i = w_j * (Σw)_j pour tous i, j

Avantages :
- Pas besoin d'estimer μ (seulement Σ)
- Diversification du risque
- Très utilisé en pratique (Bridgewater "All Weather")
"""

class RiskParityPortfolio:
    """
    Calcule le portefeuille Risk Parity.
    
    Chaque actif contribue également au risque total du portefeuille.
    """
    
    def __init__(self, Sigma: np.ndarray):
        """
        Args:
            Sigma: Matrice de covariance
        """
        self.Sigma = Sigma
        self.N = Sigma.shape[0]
    
    def risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calcule les contributions au risque de chaque actif.
        
        RC_i = w_i * (Σw)_i / σ_p
        
        Args:
            weights: Poids du portefeuille
            
        Returns:
            Vecteur des contributions au risque
        """
        sigma_p = np.sqrt(weights @ self.Sigma @ weights)
        marginal_contrib = self.Sigma @ weights / sigma_p
        risk_contrib = weights * marginal_contrib
        return risk_contrib
    
    def optimize(self, budget: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Trouve le portefeuille Risk Parity.
        
        Minimise : Σ_i Σ_j [w_i(Σw)_i - w_j(Σw)_j]²
        
        Args:
            budget: Contributions au risque cibles (par défaut : égales)
            
        Returns:
            Poids Risk Parity
        """
        if budget is None:
            budget = np.ones(self.N) / self.N  # Égales
        
        def objective(w):
            """Mesure l'écart aux contributions cibles."""
            sigma_p_sq = w @ self.Sigma @ w
            if sigma_p_sq < 1e-10:
                return 1e10
            
            rc = w * (self.Sigma @ w) / np.sqrt(sigma_p_sq)
            
            # Écart aux contributions cibles
            return np.sum((rc - budget * np.sum(rc))**2)
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Budget
        ]
        bounds = [(0.001, None) for _ in range(self.N)]  # Long-only
        
        # Point de départ : équipondéré
        w0 = np.ones(self.N) / self.N
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        return result.x / np.sum(result.x)  # Normaliser
    
    def analyze(self, weights: np.ndarray) -> dict:
        """
        Analyse un portefeuille en termes de contributions au risque.
        """
        rc = self.risk_contributions(weights)
        sigma = np.sqrt(weights @ self.Sigma @ weights)
        
        return {
            'weights': weights,
            'risk_contributions': rc,
            'risk_contribution_pct': rc / sigma * 100,
            'portfolio_volatility': sigma,
            'rc_herfindahl': np.sum((rc / sigma)**2)  # Concentration
        }


def demo_risk_parity():
    """Démontre le portefeuille Risk Parity."""
    np.random.seed(42)
    
    # Covariance avec volatilités différentes
    volatilities = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    N = len(volatilities)
    corr = 0.3 * np.ones((N, N)) + 0.7 * np.eye(N)
    Sigma = np.outer(volatilities, volatilities) * corr
    
    rp = RiskParityPortfolio(Sigma)
    
    # Comparer équipondéré et risk parity
    w_equal = np.ones(N) / N
    w_rp = rp.optimize()
    
    print("=== Comparaison Équipondéré vs Risk Parity ===")
    print("\nPortefeuille équipondéré:")
    analysis_eq = rp.analyze(w_equal)
    print(f"  Poids: {analysis_eq['weights']}")
    print(f"  Contributions au risque (%): {analysis_eq['risk_contribution_pct']}")
    
    print("\nPortefeuille Risk Parity:")
    analysis_rp = rp.analyze(w_rp)
    print(f"  Poids: {analysis_rp['weights']}")
    print(f"  Contributions au risque (%): {analysis_rp['risk_contribution_pct']}")
    
    print("\nObservation : Les actifs plus volatils ont des poids plus FAIBLES")
    print("dans le portefeuille Risk Parity pour égaliser les contributions.")
    
    return w_rp
```

---

## 5. Arbitrage Statistique {#5-arbitrage}

### 5.1 Coïntégration

```python
"""
COÏNTÉGRATION ET ARBITRAGE STATISTIQUE
======================================

COÏNTÉGRATION vs CORRÉLATION :

Corrélation : mesure si deux séries BOUGENT ENSEMBLE à court terme.
Coïntégration : mesure si deux séries RESTENT PROCHES à long terme.

Deux séries I(1) (integrated of order 1) sont coïntégrées si une
combinaison linéaire est I(0) (stationnaire).

Exemple intuitif : Un ivrogne et son chien.
- Le chien court partout (non stationnaire)
- L'ivrogne marche au hasard (non stationnaire)
- Mais la DISTANCE entre eux reste bornée (stationnaire) !

PAIRS TRADING :
1. Trouver deux actifs coïntégrés
2. Quand le spread diverge → parier sur la convergence
   - Long l'actif sous-évalué
   - Short l'actif surévalué
3. Fermer quand le spread revient à la moyenne
"""

class CointegrationTest:
    """
    Test de coïntégration d'Engle-Granger.
    
    Procédure :
    1. Régresser y sur x : y_t = α + β*x_t + ε_t
    2. Tester si les résidus ε_t sont stationnaires (test ADF)
    
    Si les résidus sont stationnaires, y et x sont coïntégrés.
    """
    
    def __init__(self):
        self.beta: Optional[float] = None
        self.alpha: Optional[float] = None
        self.residuals: Optional[np.ndarray] = None
        self.adf_stat: Optional[float] = None
        self.is_cointegrated: Optional[bool] = None
    
    def fit(
        self, 
        y: np.ndarray, 
        x: np.ndarray, 
        significance: float = 0.05
    ) -> 'CointegrationTest':
        """
        Effectue le test de coïntégration.
        
        Args:
            y: Première série (dépendante)
            x: Deuxième série (indépendante)
            significance: Niveau de significativité
            
        Returns:
            self
        """
        T = len(y)
        
        # Régression : y = α + β*x + ε
        X = np.column_stack([np.ones(T), x])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        self.alpha = coeffs[0]
        self.beta = coeffs[1]
        
        # Résidus (spread)
        self.residuals = y - self.alpha - self.beta * x
        
        # Test ADF (Augmented Dickey-Fuller) sur les résidus
        self.adf_stat = self._adf_test(self.residuals)
        
        # Valeurs critiques approximatives (Engle-Granger)
        # Ces valeurs sont différentes du test ADF standard !
        critical_values = {0.01: -3.96, 0.05: -3.37, 0.10: -3.07}
        
        self.is_cointegrated = self.adf_stat < critical_values.get(significance, -3.37)
        
        return self
    
    def _adf_test(self, series: np.ndarray, max_lags: int = None) -> float:
        """
        Test ADF (Augmented Dickey-Fuller).
        
        Teste H0: la série a une racine unitaire (non stationnaire)
        contre H1: la série est stationnaire.
        
        Returns:
            Statistique ADF (plus négative = plus de preuves de stationnarité)
        """
        T = len(series)
        if max_lags is None:
            max_lags = int((T - 1)**(1/3))
        
        # Différence de la série
        diff = np.diff(series)
        lagged = series[:-1]
        
        # Construire les lags pour augmentation
        X = np.column_stack([np.ones(T-1), lagged])
        
        # Ajouter les lags des différences
        for lag in range(1, max_lags + 1):
            if lag < len(diff):
                lagged_diff = np.zeros(T - 1)
                lagged_diff[lag:] = diff[:-lag]
                X = np.column_stack([X, lagged_diff])
        
        # Régression
        y = diff
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Résidus
        residuals = y - X @ beta
        
        # Statistique t pour le coefficient de series[t-1]
        sigma = np.std(residuals)
        se_beta = sigma / np.std(lagged)
        t_stat = beta[1] / se_beta
        
        return t_stat
    
    def get_spread(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Calcule le spread (z-score) pour de nouvelles données.
        """
        spread = y - self.alpha - self.beta * x
        return (spread - np.mean(self.residuals)) / np.std(self.residuals)


class PairsTrading:
    """
    Stratégie de Pairs Trading basée sur la coïntégration.
    
    Trading rules :
    - Open long spread (long y, short β*x) quand z-score < -entry_threshold
    - Open short spread (short y, long β*x) quand z-score > +entry_threshold
    - Close position quand z-score revient à ±exit_threshold
    
    Le profit vient de la MEAN-REVERSION du spread.
    """
    
    def __init__(
        self,
        entry_threshold: float = 2.0,  # Nombre de σ pour entrer
        exit_threshold: float = 0.5,   # Nombre de σ pour sortir
        stop_loss: float = 4.0         # Stop loss en σ
    ):
        """
        Args:
            entry_threshold: Seuil d'entrée (en z-scores)
            exit_threshold: Seuil de sortie
            stop_loss: Stop loss
        """
        self.entry = entry_threshold
        self.exit = exit_threshold
        self.stop = stop_loss
        
        self.coint_test: Optional[CointegrationTest] = None
    
    def fit(self, y: np.ndarray, x: np.ndarray) -> 'PairsTrading':
        """
        Calibre la stratégie sur des données historiques.
        
        Args:
            y: Prix de l'actif y (à acheter quand spread bas)
            x: Prix de l'actif x (à shorter quand spread bas)
        """
        self.coint_test = CointegrationTest().fit(y, x)
        
        if not self.coint_test.is_cointegrated:
            print("⚠️ Attention: les actifs ne semblent pas coïntégrés!")
        
        return self
    
    def generate_signals(
        self, 
        y: np.ndarray, 
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère les signaux de trading.
        
        Args:
            y, x: Séries de prix
            
        Returns:
            (positions, z_scores): Position (-1, 0, +1) et z-scores
        """
        T = len(y)
        z = self.coint_test.get_spread(y, x)
        
        positions = np.zeros(T)
        current_position = 0
        
        for t in range(1, T):
            if current_position == 0:
                # Pas de position → chercher entrée
                if z[t] < -self.entry:
                    current_position = 1  # Long spread
                elif z[t] > self.entry:
                    current_position = -1  # Short spread
            
            elif current_position == 1:  # Long spread
                # Sortie si revient à la moyenne ou stop loss
                if z[t] > -self.exit or z[t] < -self.stop:
                    current_position = 0
            
            elif current_position == -1:  # Short spread
                if z[t] < self.exit or z[t] > self.stop:
                    current_position = 0
            
            positions[t] = current_position
        
        return positions, z
    
    def backtest(
        self, 
        y: np.ndarray, 
        x: np.ndarray
    ) -> dict:
        """
        Backteste la stratégie.
        
        Args:
            y, x: Séries de prix
            
        Returns:
            Statistiques du backtest
        """
        positions, z = self.generate_signals(y, x)
        
        # Rendements du spread
        # Long spread = long y - β*short x
        beta = self.coint_test.beta
        
        returns_y = np.diff(y) / y[:-1]
        returns_x = np.diff(x) / x[:-1]
        
        spread_returns = returns_y - beta * returns_x
        
        # PnL de la stratégie
        strategy_returns = positions[:-1] * spread_returns
        
        # Statistiques
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        n_trades = np.sum(np.abs(np.diff(positions)) > 0) // 2
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'n_trades': n_trades,
            'positions': positions,
            'z_scores': z,
            'strategy_returns': strategy_returns
        }


def demo_pairs_trading():
    """Démontre le pairs trading avec données simulées."""
    np.random.seed(42)
    T = 500
    
    # Simuler deux actifs coïntégrés
    # x suit un random walk
    x = np.cumsum(np.random.randn(T) * 0.02) + 100
    
    # y = 0.5*x + spread stationnaire
    beta_true = 0.5
    spread = np.zeros(T)
    for t in range(1, T):
        spread[t] = 0.8 * spread[t-1] + np.random.randn() * 0.5  # AR(1)
    
    y = beta_true * x + 50 + spread
    
    # Stratégie
    strategy = PairsTrading(entry_threshold=1.5, exit_threshold=0.2)
    strategy.fit(y[:250], x[:250])  # Train sur première moitié
    
    print("=== Pairs Trading ===")
    print(f"Beta estimé: {strategy.coint_test.beta:.4f} (vrai: {beta_true})")
    print(f"Coïntégration: {'Oui' if strategy.coint_test.is_cointegrated else 'Non'}")
    
    # Backtest sur deuxième moitié
    results = strategy.backtest(y[250:], x[250:])
    
    print(f"\nBacktest (250 jours):")
    print(f"  Rendement total: {results['total_return']:.2%}")
    print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Nombre de trades: {results['n_trades']}")
    
    return strategy, results
```

---

## 6. Exécution d'Ordres {#6-execution}

```python
"""
EXÉCUTION OPTIMALE D'ORDRES
===========================

Problème : Exécuter un gros ordre (ex: vendre 1M d'actions) impacte le marché !

MARKET IMPACT :
- Impact temporaire : affecte le prix pendant l'exécution
- Impact permanent : shift durable du prix

Modèle d'Almgren-Chriss (2001) :

Prix d'exécution : S_k = S_0 - γ*Σ_{j<k} n_j - η*n_k/τ + σ*ε_k

Où :
- S_0 : prix initial
- n_k : quantité tradée au pas k
- γ : impact permanent (par unité tradée)
- η : impact temporaire (par unité/temps)
- τ : intervalle de temps
- σ : volatilité

OBJECTIF : Minimiser le coût d'exécution
- Trader vite → gros impact temporaire
- Trader lentement → exposition au risque de prix

C'est un compromis rendement-risque similaire au beamforming !
"""

class AlmgrenChrissModel:
    """
    Modèle d'Almgren-Chriss pour l'exécution optimale.
    
    Minimise : E[Coût] + λ * Var[Coût]
    
    Où λ est l'aversion au risque de l'exécuteur.
    """
    
    def __init__(
        self,
        total_shares: float,
        T: int,
        sigma: float,
        gamma: float,
        eta: float,
        tau: float = 1.0
    ):
        """
        Args:
            total_shares: X, quantité totale à exécuter
            T: Nombre de périodes d'exécution
            sigma: Volatilité du prix
            gamma: Paramètre d'impact permanent
            eta: Paramètre d'impact temporaire
            tau: Durée d'une période
        """
        self.X = total_shares
        self.T = T
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        self.tau = tau
    
    def optimal_trajectory(self, risk_aversion: float) -> np.ndarray:
        """
        Calcule la trajectoire d'exécution optimale.
        
        La solution est exponentielle pour λ > 0 :
        
        n_k = X * sinh(κ*(T-k)) / sinh(κ*T)
        
        où κ = √(λσ²/η)
        
        Args:
            risk_aversion: Paramètre λ
            
        Returns:
            Quantités n_k à trader à chaque période
        """
        lam = risk_aversion
        
        if lam == 0:
            # Solution TWAP : trade uniforme
            return np.ones(self.T) * self.X / self.T
        
        # Paramètre κ
        kappa = np.sqrt(lam * self.sigma**2 * self.tau / self.eta)
        
        # Trajectoire optimale
        n = np.zeros(self.T)
        denom = np.sinh(kappa * self.T)
        
        for k in range(self.T):
            n[k] = self.X * np.sinh(kappa * (self.T - k)) / denom
            # Correction : la formule donne le remaining, on veut le trade
        
        # Convertir position restante en trades
        remaining = np.zeros(self.T + 1)
        remaining[0] = self.X
        for k in range(self.T):
            # remaining[k+1] = remaining[k] * sinh(κ*(T-k-1)) / sinh(κ*(T-k))
            if k < self.T - 1:
                remaining[k+1] = self.X * np.sinh(kappa * (self.T - k - 1)) / denom
            else:
                remaining[k+1] = 0
        
        trades = -np.diff(remaining)
        
        return trades
    
    def expected_cost(
        self, 
        trades: np.ndarray, 
        S0: float = 100.0
    ) -> float:
        """
        Calcule le coût espéré d'exécution.
        
        E[Coût] = (1/2)*γ*X² + η*Σ n_k²/τ
        
        Args:
            trades: Quantités à chaque période
            S0: Prix initial
            
        Returns:
            Coût espéré
        """
        permanent_cost = 0.5 * self.gamma * self.X**2
        temporary_cost = self.eta * np.sum(trades**2) / self.tau
        
        return permanent_cost + temporary_cost
    
    def variance_cost(self, trades: np.ndarray) -> float:
        """
        Calcule la variance du coût d'exécution.
        
        Var[Coût] = σ² * τ * Σ_k (Σ_{j≥k} n_j)²
        
        Args:
            trades: Quantités à chaque période
            
        Returns:
            Variance du coût
        """
        remaining = np.cumsum(trades[::-1])[::-1]  # Position restante
        return self.sigma**2 * self.tau * np.sum(remaining**2)
    
    def simulate_execution(
        self, 
        trades: np.ndarray, 
        S0: float = 100.0,
        n_simulations: int = 1000
    ) -> dict:
        """
        Simule l'exécution pour estimer le coût réel.
        
        Args:
            trades: Stratégie d'exécution
            S0: Prix initial
            n_simulations: Nombre de simulations
            
        Returns:
            Statistiques du coût
        """
        costs = []
        
        for _ in range(n_simulations):
            S = S0
            total_cost = 0
            remaining = self.X
            
            for k, n_k in enumerate(trades):
                # Bruit de prix
                noise = self.sigma * np.sqrt(self.tau) * np.random.randn()
                
                # Prix d'exécution avec impact
                S_exec = S - self.eta * n_k / self.tau + noise
                
                # Coût
                total_cost += n_k * (S0 - S_exec)
                
                # Mettre à jour le prix (impact permanent)
                S = S - self.gamma * n_k + noise
                remaining -= n_k
            
            costs.append(total_cost)
        
        costs = np.array(costs)
        
        return {
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'var_cost': np.var(costs),
            '5th_percentile': np.percentile(costs, 5),
            '95th_percentile': np.percentile(costs, 95)
        }


def demo_execution():
    """Démontre l'exécution optimale d'ordres."""
    
    # Paramètres
    model = AlmgrenChrissModel(
        total_shares=10000,  # 10,000 actions à vendre
        T=20,                # 20 périodes
        sigma=0.02,          # 2% volatilité par période
        gamma=0.001,         # Impact permanent
        eta=0.01,            # Impact temporaire
        tau=1.0
    )
    
    print("=== Exécution Optimale d'Ordres (Almgren-Chriss) ===")
    
    # Différentes stratégies
    strategies = {
        'TWAP (λ=0)': model.optimal_trajectory(0),
        'Averse (λ=0.1)': model.optimal_trajectory(0.1),
        'Très averse (λ=1)': model.optimal_trajectory(1.0)
    }
    
    for name, trades in strategies.items():
        expected = model.expected_cost(trades)
        variance = model.variance_cost(trades)
        
        print(f"\n{name}:")
        print(f"  Trades: {trades[:5].round(0)}...")
        print(f"  Coût espéré: {expected:.2f}")
        print(f"  Variance: {variance:.2f}")
        print(f"  Std: {np.sqrt(variance):.2f}")
    
    print("\nObservation:")
    print("- TWAP (Time-Weighted Average Price) trade uniformément")
    print("- Stratégie averse trade plus au début pour réduire le risque")
    
    return model


# =============================================================================
# EXÉCUTION DES DÉMOS
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("INGÉNIERIE FINANCIÈRE : UNE PERSPECTIVE DU TRAITEMENT DU SIGNAL")
    print("="*70)
    
    print("\n" + "="*70)
    demo_iid_model()
    
    print("\n" + "="*70)
    demo_garch()
    
    print("\n" + "="*70)
    demo_shrinkage()
    
    print("\n" + "="*70)
    demo_robust_estimation()
    
    print("\n" + "="*70)
    demo_markowitz()
    
    print("\n" + "="*70)
    demo_robust_portfolio()
    
    print("\n" + "="*70)
    demo_risk_parity()
    
    print("\n" + "="*70)
    demo_pairs_trading()
    
    print("\n" + "="*70)
    demo_execution()
    
    print("\n" + "="*70)
    print("FIN DES DÉMONSTRATIONS")
    print("="*70)
```

---

## 📚 Glossaire des Acronymes

| Acronyme | Signification | Traduction/Explication |
|----------|--------------|------------------------|
| **ARMA** | AutoRegressive Moving Average | Moyenne mobile autorégressive |
| **VAR** | Vector AutoRegressive | Vecteur autorégressif |
| **GARCH** | Generalized AutoRegressive Conditional Heteroskedasticity | Modèle de volatilité conditionnelle |
| **MLE** | Maximum Likelihood Estimation | Estimation par maximum de vraisemblance |
| **OLS** | Ordinary Least Squares | Moindres carrés ordinaires |
| **SCM** | Sample Covariance Matrix | Matrice de covariance empirique |
| **MVP** | Minimum Variance Portfolio | Portefeuille de variance minimale |
| **CAPM** | Capital Asset Pricing Model | Modèle d'évaluation des actifs financiers |
| **PCA** | Principal Component Analysis | Analyse en composantes principales |
| **ADF** | Augmented Dickey-Fuller | Test de stationnarité |
| **I(d)** | Integrated of order d | Intégré d'ordre d (d différenciations pour stationnarité) |
| **VaR** | Value at Risk | Valeur à risque |
| **CVaR** | Conditional Value at Risk | Valeur à risque conditionnelle |
| **TWAP** | Time-Weighted Average Price | Prix moyen pondéré dans le temps |

---

## 🔗 Connexions avec le Traitement du Signal

| Finance | Traitement du Signal |
|---------|---------------------|
| Optimisation de portefeuille | Design de beamforming |
| Shrinkage covariance | Diagonal loading |
| Modèle ARMA | Modèle pôle-zéro |
| Index tracking sparse | Compressed sensing |
| Exécution d'ordres | Scheduling réseau |

---

*Document généré à partir de "A Signal Processing Perspective on Financial Engineering" - Feng & Palomar (2016)*
