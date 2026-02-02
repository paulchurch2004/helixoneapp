# 📊 PyPortfolioOpt - Guide Complet pour HelixOne
## Optimisation de Portefeuille en Python

---

# TABLE DES MATIÈRES

1. [Introduction et Installation](#1-introduction-et-installation)
2. [Rendements Espérés (Expected Returns)](#2-rendements-espérés-expected-returns)
3. [Modèles de Risque (Risk Models)](#3-modèles-de-risque-risk-models)
4. [Frontière Efficiente (Efficient Frontier)](#4-frontière-efficiente-efficient-frontier)
5. [Fonctions Objectif (Objective Functions)](#5-fonctions-objectif-objective-functions)
6. [Modèle Black-Litterman](#6-modèle-black-litterman)
7. [HRP - Hierarchical Risk Parity](#7-hrp-hierarchical-risk-parity)
8. [CVaR et Semivariance](#8-cvar-et-semivariance)
9. [Allocation Discrète](#9-allocation-discrète)
10. [Exemples Complets](#10-exemples-complets)

---

# 1. INTRODUCTION ET INSTALLATION

## 1.1 Qu'est-ce que PyPortfolioOpt ?

```python
"""
PyPortfolioOpt
==============
Bibliothèque Python pour l'optimisation de portefeuille.

Fonctionnalités principales:
- MVO (Mean-Variance Optimization) - Optimisation Moyenne-Variance (Markowitz)
- Black-Litterman allocation
- HRP (Hierarchical Risk Parity) - Parité de Risque Hiérarchique
- CVaR (Conditional Value at Risk) - Valeur à Risque Conditionnelle
- CLA (Critical Line Algorithm) - Algorithme de la Ligne Critique

Workflow typique:
1. Charger les prix historiques
2. Calculer les rendements espérés (mu)
3. Calculer la matrice de covariance (S ou Sigma)
4. Optimiser le portefeuille
5. Obtenir l'allocation discrète (nombre d'actions à acheter)
"""
```

## 1.2 Installation

```bash
# Installation standard
pip install pyportfolioopt

# Installation avec toutes les dépendances
pip install pyportfolioopt[all]

# Dépendances principales
pip install pandas numpy scipy cvxpy scikit-learn
```

## 1.3 Imports de Base

```python
import pandas as pd
import numpy as np

# Imports PyPortfolioOpt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import objective_functions
from pypfopt import BlackLittermanModel, black_litterman
from pypfopt import HRPOpt
from pypfopt import CLA
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientCVaR, EfficientSemivariance
```

## 1.4 Chargement des Données

```python
"""
Format des données requis:
- DataFrame pandas
- Index = dates (DatetimeIndex)
- Colonnes = tickers (symboles boursiers)
- Valeurs = prix ajustés (adjusted close)
"""

# Méthode 1: Depuis un fichier CSV
df = pd.read_csv(
    "stock_prices.csv", 
    parse_dates=True, 
    index_col="date"
)

# Méthode 2: Avec yfinance
import yfinance as yf

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'XOM']
df = yf.download(tickers, start='2018-01-01', end='2023-12-31')['Adj Close']

print(f"Forme des données: {df.shape}")
print(f"Période: {df.index[0]} à {df.index[-1]}")
print(f"Tickers: {list(df.columns)}")
```

---

# 2. RENDEMENTS ESPÉRÉS (EXPECTED RETURNS)

## 2.1 Vue d'ensemble

```python
"""
Module expected_returns
=======================
Calcule les estimations de rendements futurs à partir des prix historiques.

Méthodes disponibles:
1. mean_historical_return() - Moyenne historique simple
2. ema_historical_return() - Moyenne pondérée exponentiellement (EMA)
3. capm_return() - Modèle CAPM (Capital Asset Pricing Model)

Par convention, les rendements sont ANNUALISÉS (frequency=252 jours de trading).
"""
from pypfopt import expected_returns
```

## 2.2 Rendement Historique Moyen

```python
def mean_historical_return(
    prices,                    # DataFrame de prix
    returns_data=False,        # True si on passe des rendements au lieu de prix
    compounding=True,          # Moyenne géométrique (CAGR) si True, arithmétique sinon
    frequency=252,             # Jours de trading par an
    log_returns=False          # Utiliser les log-returns
):
    """
    Calcule le rendement historique moyen annualisé.
    
    Formule (compounding=True):
        mu = (1 + r)^frequency - 1
        où r = moyenne des rendements quotidiens
    
    Formule (compounding=False):
        mu = r * frequency
    """
    pass

# Exemple d'utilisation
mu = expected_returns.mean_historical_return(df)
print("Rendements espérés annualisés:")
print(mu.sort_values(ascending=False))

"""
Exemple de sortie:
AMZN    0.312
META    0.287
AAPL    0.245
MSFT    0.221
...
"""
```

## 2.3 Rendement EMA (Exponentially-Weighted Mean)

```python
def ema_historical_return(
    prices,
    returns_data=False,
    compounding=True,
    span=500,           # Fenêtre EMA (demi-vie environ span/3)
    frequency=252,
    log_returns=False
):
    """
    Calcule le rendement moyen pondéré exponentiellement.
    
    Avantage: Donne plus de poids aux données récentes.
    
    span=500 signifie que les données d'il y a 500 jours ont
    environ 37% du poids des données les plus récentes.
    """
    pass

# Exemple
mu_ema = expected_returns.ema_historical_return(df, span=180)
print("Rendements EMA (span=180):")
print(mu_ema.sort_values(ascending=False))
```

## 2.4 Rendement CAPM

```python
def capm_return(
    prices,
    market_prices=None,        # Prix du benchmark (ex: SPY)
    returns_data=False,
    risk_free_rate=0.0,        # Taux sans risque
    compounding=True,
    frequency=252,
    log_returns=False
):
    """
    Calcule les rendements espérés selon le CAPM.
    
    Formule CAPM:
        R_i = R_f + β_i * (E(R_m) - R_f)
    
    Où:
        R_i = rendement espéré de l'actif i
        R_f = taux sans risque (risk-free rate)
        β_i = beta de l'actif (sensibilité au marché)
        E(R_m) = rendement espéré du marché
    
    Si market_prices=None, utilise la moyenne équipondérée comme proxy du marché.
    """
    pass

# Exemple avec SPY comme benchmark
spy = yf.download('SPY', start='2018-01-01', end='2023-12-31')['Adj Close']
mu_capm = expected_returns.capm_return(
    df, 
    market_prices=spy,
    risk_free_rate=0.02  # Taux sans risque de 2%
)
print("Rendements CAPM:")
print(mu_capm)
```

## 2.5 Fonctions Utilitaires

```python
# Convertir prix en rendements
returns = expected_returns.returns_from_prices(df, log_returns=False)
print(f"Rendements quotidiens: {returns.shape}")

# Convertir rendements en pseudo-prix (utile pour certaines fonctions)
pseudo_prices = expected_returns.prices_from_returns(returns)

# Fonction générique pour choisir la méthode
mu = expected_returns.return_model(
    df, 
    method="ema_historical_return",  # ou "mean_historical_return", "capm_return"
    span=200
)
```

---

# 3. MODÈLES DE RISQUE (RISK MODELS)

## 3.1 Vue d'ensemble

```python
"""
Module risk_models
==================
Calcule la matrice de covariance des rendements.

La matrice de covariance (Σ ou S) est CRUCIALE car:
- Elle capture la volatilité de chaque actif
- Elle capture les corrélations entre actifs
- Elle est utilisée dans TOUTES les optimisations MVO

Problème: La covariance échantillon a une HAUTE erreur d'estimation.
Solution: Techniques de shrinkage (rétrécissement) pour réduire l'erreur.

Méthodes disponibles:
1. sample_cov() - Covariance échantillon simple
2. semicovariance() - Ne considère que les rendements négatifs
3. exp_cov() - Covariance pondérée exponentiellement
4. CovarianceShrinkage - Méthodes de shrinkage (Ledoit-Wolf, OAS)
"""
from pypfopt import risk_models
```

## 3.2 Covariance Échantillon

```python
def sample_cov(
    prices,
    returns_data=False,
    frequency=252,           # Annualisation
    log_returns=False
):
    """
    Calcule la matrice de covariance échantillon annualisée.
    
    Formule:
        S = Cov(returns) * frequency
    
    Avantages:
        - Simple et intuitif
        - Estimateur non biaisé
    
    Inconvénients:
        - Haute erreur d'estimation pour beaucoup d'actifs
        - L'optimiseur peut sur-pondérer les erreurs
    """
    pass

# Exemple
S = risk_models.sample_cov(df)
print(f"Forme de la matrice de covariance: {S.shape}")
print("\nCovariance échantillon:")
print(S.round(4))
```

## 3.3 Semicovariance

```python
def semicovariance(
    prices,
    returns_data=False,
    benchmark=0.000079,      # Benchmark quotidien ≈ 2% annuel
    frequency=252,
    log_returns=False
):
    """
    Calcule la semicovariance (downside covariance).
    
    Ne considère que les rendements INFÉRIEURS au benchmark.
    
    Formule:
        semicov = E[min(r_i - B, 0) * min(r_j - B, 0)]
    
    Avantage: Capture uniquement le risque de perte (downside risk).
    """
    pass

# Exemple
S_semi = risk_models.semicovariance(df, benchmark=0)  # benchmark = 0
print("Semicovariance (downside only):")
print(S_semi.round(4))
```

## 3.4 Covariance Exponentielle

```python
def exp_cov(
    prices,
    returns_data=False,
    span=180,                # Fenêtre EMA
    frequency=252,
    log_returns=False
):
    """
    Calcule la covariance pondérée exponentiellement.
    
    Donne plus de poids aux données récentes.
    Utile si vous pensez que les corrélations récentes sont plus pertinentes.
    """
    pass

# Exemple
S_exp = risk_models.exp_cov(df, span=60)  # EMA sur 60 jours
print("Covariance exponentielle (span=60):")
print(S_exp.round(4))
```

## 3.5 Shrinkage de Covariance (Ledoit-Wolf)

```python
"""
Shrinkage = "Rétrécissement"
============================
Combine la covariance échantillon avec un estimateur structuré
pour réduire l'erreur d'estimation.

Formule:
    S_shrunk = δ * F + (1 - δ) * S
    
Où:
    S = covariance échantillon
    F = target structuré (identité, facteur unique, corrélation constante)
    δ = paramètre de shrinkage (0 à 1)
"""

class CovarianceShrinkage:
    """
    Implémente plusieurs méthodes de shrinkage.
    """
    
    def __init__(self, prices, returns_data=False, frequency=252, log_returns=False):
        pass
    
    def shrunk_covariance(self, delta=0.2):
        """Shrinkage manuel avec delta fixe."""
        pass
    
    def ledoit_wolf(self, shrinkage_target="constant_variance"):
        """
        Ledoit-Wolf shrinkage avec estimation optimale de delta.
        
        Targets disponibles:
        - "constant_variance": Identité × variance moyenne
        - "single_factor": Modèle à un facteur (Sharpe)
        - "constant_correlation": Corrélation constante
        """
        pass
    
    def oracle_approximating(self):
        """
        OAS (Oracle Approximating Shrinkage).
        Alternative à Ledoit-Wolf, parfois plus performante.
        """
        pass

# Exemples
cs = risk_models.CovarianceShrinkage(df)

# Ledoit-Wolf avec target "constant_variance"
S_lw = cs.ledoit_wolf(shrinkage_target="constant_variance")
print(f"Delta Ledoit-Wolf: {cs.delta:.4f}")

# Ledoit-Wolf avec target "single_factor"
S_lw_sf = risk_models.CovarianceShrinkage(df).ledoit_wolf(
    shrinkage_target="single_factor"
)

# Ledoit-Wolf avec target "constant_correlation"
S_lw_cc = risk_models.CovarianceShrinkage(df).ledoit_wolf(
    shrinkage_target="constant_correlation"
)

# Oracle Approximating Shrinkage
S_oas = risk_models.CovarianceShrinkage(df).oracle_approximating()
```

## 3.6 Fonction Générique

```python
# Utiliser la fonction générique risk_matrix()
S = risk_models.risk_matrix(
    df,
    method="ledoit_wolf"  # ou "sample_cov", "semicovariance", "exp_cov", etc.
)

# Méthodes disponibles:
methods = [
    "sample_cov",
    "semicovariance",
    "exp_cov",
    "ledoit_wolf",
    "ledoit_wolf_constant_variance",
    "ledoit_wolf_single_factor",
    "ledoit_wolf_constant_correlation",
    "oracle_approximating"
]
```

## 3.7 Utilitaires

```python
# Convertir covariance en corrélation
corr = risk_models.cov_to_corr(S)
print("Matrice de corrélation:")
print(corr.round(2))

# Convertir corrélation en covariance (besoin des volatilités)
stdevs = np.sqrt(np.diag(S))
S_rebuilt = risk_models.corr_to_cov(corr, stdevs)

# Vérifier/fixer les matrices non positive-semidefinite
S_fixed = risk_models.fix_nonpositive_semidefinite(S, fix_method="spectral")
# fix_method: "spectral" (met les eigenvalues négatives à 0) ou "diag"
```

---

# 4. FRONTIÈRE EFFICIENTE (EFFICIENT FRONTIER)

## 4.1 Introduction à la MVO

```python
"""
Mean-Variance Optimization (MVO)
================================
Développée par Harry Markowitz (1952), Prix Nobel 1990.

Principe: Trouver le portefeuille qui:
- Maximise le rendement pour un niveau de risque donné, OU
- Minimise le risque pour un rendement cible

La "Frontière Efficiente" est l'ensemble de tous les portefeuilles optimaux.

Formulation mathématique (minimisation de variance):
    min   w'Σw
    s.t.  w'μ ≥ target_return
          Σw = 1
          w ≥ 0 (long only)

Où:
    w = vecteur de poids
    Σ = matrice de covariance
    μ = vecteur de rendements espérés
"""
```

## 4.2 Classe EfficientFrontier

```python
from pypfopt import EfficientFrontier

class EfficientFrontier:
    """
    Classe principale pour l'optimisation MVO.
    
    Méthodes d'optimisation:
    - min_volatility(): Minimise la volatilité
    - max_sharpe(): Maximise le Sharpe Ratio (portefeuille tangent)
    - max_quadratic_utility(): Maximise l'utilité quadratique
    - efficient_risk(): Maximise le rendement pour une volatilité cible
    - efficient_return(): Minimise la volatilité pour un rendement cible
    """
    
    def __init__(
        self,
        expected_returns,        # pd.Series ou array de mu
        cov_matrix,              # pd.DataFrame ou array de Σ
        weight_bounds=(0, 1),    # (min, max) pour chaque poids
        solver=None,             # Solveur CVXPY (auto par défaut)
        verbose=False,
        solver_options=None
    ):
        """
        Initialise l'optimiseur.
        
        weight_bounds:
            (0, 1) = long only, max 100% par actif
            (-1, 1) = permet les ventes à découvert
            (0, 0.1) = max 10% par actif
            [(0, 0.5), (0, 0.3), ...] = limites par actif
        """
        pass

# Initialisation standard
ef = EfficientFrontier(mu, S)

# Avec ventes à découvert autorisées
ef_short = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

# Avec limite max de 10% par position
ef_constrained = EfficientFrontier(mu, S, weight_bounds=(0, 0.1))
```

## 4.3 Minimiser la Volatilité

```python
def min_volatility(self):
    """
    Trouve le portefeuille de variance minimale (MVP).
    
    C'est le point le plus à gauche sur la frontière efficiente.
    
    Formulation:
        min  w'Σw
        s.t. Σw = 1
             w ≥ 0
    
    Returns:
        OrderedDict: Poids optimaux
    """
    pass

# Exemple
ef = EfficientFrontier(mu, S)
weights = ef.min_volatility()
print("Portefeuille de variance minimale:")
for ticker, weight in ef.clean_weights().items():
    if weight > 0.001:
        print(f"  {ticker}: {weight:.2%}")

# Performance
ret, vol, sharpe = ef.portfolio_performance(verbose=True)
"""
Expected annual return: 15.2%
Annual volatility: 12.1%
Sharpe Ratio: 1.01
"""
```

## 4.4 Maximiser le Sharpe Ratio

```python
def max_sharpe(self, risk_free_rate=0.0):
    """
    Trouve le portefeuille qui maximise le Sharpe Ratio.
    
    Aussi appelé "portefeuille tangent" car il est tangent à la
    ligne du marché des capitaux (CML - Capital Market Line).
    
    Sharpe Ratio = (μ_p - R_f) / σ_p
    
    Args:
        risk_free_rate: Taux sans risque (annualisé, même fréquence que mu)
    
    Note: Utilise une transformation convexe pour résoudre ce
    problème non-convexe.
    """
    pass

# Exemple
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe(risk_free_rate=0.02)  # Rf = 2%

print("Portefeuille Max Sharpe:")
cleaned = ef.clean_weights()
for ticker, weight in cleaned.items():
    if weight > 0.001:
        print(f"  {ticker}: {weight:.2%}")

# Performance
ret, vol, sharpe = ef.portfolio_performance(verbose=True, risk_free_rate=0.02)
"""
Expected annual return: 28.5%
Annual volatility: 18.3%
Sharpe Ratio: 1.45
"""
```

## 4.5 Rendement Cible (efficient_return)

```python
def efficient_return(self, target_return, market_neutral=False):
    """
    Portefeuille de Markowitz: Minimise la volatilité pour un
    rendement cible.
    
    Formulation:
        min  w'Σw
        s.t. w'μ ≥ target_return
             Σw = 1
    
    Args:
        target_return: Rendement annualisé cible (ex: 0.15 pour 15%)
        market_neutral: Si True, les poids somment à 0 (long/short)
    """
    pass

# Exemple: portefeuille visant 20% de rendement
ef = EfficientFrontier(mu, S)
weights = ef.efficient_return(target_return=0.20)
ef.portfolio_performance(verbose=True)

# Portefeuille market-neutral (long/short, poids = 0)
ef_neutral = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
weights_neutral = ef_neutral.efficient_return(
    target_return=0.15, 
    market_neutral=True
)
print(f"Somme des poids: {sum(weights_neutral.values()):.4f}")  # ≈ 0
```

## 4.6 Risque Cible (efficient_risk)

```python
def efficient_risk(self, target_volatility, market_neutral=False):
    """
    Maximise le rendement pour une volatilité cible.
    
    Formulation:
        max  w'μ
        s.t. w'Σw ≤ target_volatility²
             Σw = 1
    
    Args:
        target_volatility: Volatilité annualisée cible (ex: 0.15 pour 15%)
    """
    pass

# Exemple: max rendement avec volatilité ≤ 15%
ef = EfficientFrontier(mu, S)
weights = ef.efficient_risk(target_volatility=0.15)
ret, vol, sharpe = ef.portfolio_performance(verbose=True)
```

## 4.7 Utilité Quadratique

```python
def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
    """
    Maximise l'utilité quadratique.
    
    U(w) = w'μ - (δ/2) * w'Σw
    
    Args:
        risk_aversion (δ): Coefficient d'aversion au risque
            - δ = 0: Maximise uniquement le rendement (très risqué)
            - δ = 1: Équilibre rendement/risque standard
            - δ → ∞: Minimise uniquement le risque
    """
    pass

# Exemples avec différentes aversions au risque
for delta in [0.5, 1, 2, 5]:
    ef = EfficientFrontier(mu, S)
    ef.max_quadratic_utility(risk_aversion=delta)
    ret, vol, sharpe = ef.portfolio_performance()
    print(f"δ={delta}: Return={ret:.1%}, Vol={vol:.1%}, Sharpe={sharpe:.2f}")
```

## 4.8 Nettoyage et Sauvegarde des Poids

```python
# Nettoyer les poids (arrondir, supprimer les presque-zéros)
cleaned = ef.clean_weights(cutoff=0.001, rounding=4)
# cutoff: Poids < cutoff sont mis à 0
# rounding: Nombre de décimales

# Sauvegarder les poids
ef.save_weights_to_file("portfolio_weights.csv")  # ou .json, .txt
```

---

# 5. FONCTIONS OBJECTIF (OBJECTIVE FUNCTIONS)

## 5.1 Vue d'ensemble

```python
"""
Module objective_functions
==========================
Fonctions objectif utilisées dans l'optimisation.

Peuvent être utilisées:
1. En interne par EfficientFrontier
2. Comme objectifs personnalisés avec add_objective()
3. Pour calculer des métriques sur un portefeuille existant
"""
from pypfopt import objective_functions
```

## 5.2 Fonctions Principales

```python
def portfolio_variance(w, cov_matrix):
    """
    Variance du portefeuille: w'Σw
    
    Args:
        w: Poids (np.array ou cp.Variable)
        cov_matrix: Matrice de covariance
    
    Returns:
        Variance (σ²), pas volatilité (σ)
    """
    pass

def portfolio_return(w, expected_returns, negative=True):
    """
    Rendement du portefeuille: w'μ
    
    Args:
        negative: Si True, retourne -w'μ (pour minimisation)
    """
    pass

def sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate=0.0, negative=True):
    """
    Sharpe Ratio: (w'μ - Rf) / σ
    """
    pass

def quadratic_utility(w, expected_returns, cov_matrix, risk_aversion, negative=True):
    """
    Utilité quadratique: w'μ - (δ/2) * w'Σw
    """
    pass
```

## 5.3 Régularisation L2

```python
def L2_reg(w, gamma=1):
    """
    Régularisation L2: γ * ||w||²
    
    Ajoute une pénalité sur les poids extrêmes.
    Encourage des portefeuilles plus diversifiés.
    
    Args:
        gamma: Force de la régularisation
            - gamma = 0: Pas de régularisation
            - gamma = 1: Régularisation modérée
            - gamma → ∞: Force les poids vers l'équipondération
    """
    pass

# Exemple: Max Sharpe avec régularisation L2
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.L2_reg, gamma=0.5)
weights = ef.max_sharpe()

# Comparer avec/sans régularisation
print("Nombre de positions non-nulles:")
print(f"  Sans L2: {sum(1 for w in ef_no_reg.clean_weights().values() if w > 0.01)}")
print(f"  Avec L2: {sum(1 for w in ef.clean_weights().values() if w > 0.01)}")
```

## 5.4 Coûts de Transaction

```python
def transaction_cost(w, w_prev, k=0.001):
    """
    Modèle simple de coûts de transaction.
    
    cost = k * ||w - w_prev||₁
    
    Args:
        w: Nouveaux poids
        w_prev: Anciens poids
        k: Coût par unité de poids échangée (défaut: 10 bps)
    """
    pass

# Exemple: Rebalancement avec coûts
w_current = np.array([0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0])  # Poids actuels

ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.transaction_cost, w_prev=w_current, k=0.001)
weights = ef.max_sharpe()
```

## 5.5 Tracking Error

```python
def ex_ante_tracking_error(w, cov_matrix, benchmark_weights):
    """
    Tracking Error ex-ante (prévu): (w - w_b)'Σ(w - w_b)
    
    Mesure l'écart attendu par rapport à un benchmark.
    
    Args:
        benchmark_weights: Poids du benchmark (ex: S&P 500)
    """
    pass

def ex_post_tracking_error(w, historic_returns, benchmark_returns):
    """
    Tracking Error ex-post (réalisé): Var(r_p - r_b)
    
    Mesure l'écart historique par rapport à un benchmark.
    """
    pass

# Exemple: Minimiser la volatilité tout en restant proche du benchmark
benchmark = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3])  # Équipondéré

ef = EfficientFrontier(mu, S)
ef.add_objective(
    objective_functions.ex_ante_tracking_error, 
    cov_matrix=S.values, 
    benchmark_weights=benchmark
)
weights = ef.min_volatility()
```

## 5.6 Ajouter des Contraintes et Objectifs Personnalisés

```python
# Ajouter une contrainte
ef = EfficientFrontier(mu, S)

# Contrainte: poids de AAPL ≥ 5%
ef.add_constraint(lambda w: w[0] >= 0.05)

# Contrainte: somme de secteur Tech ≤ 40%
tech_indices = [0, 1, 2, 3]  # AAPL, MSFT, GOOGL, AMZN
ef.add_constraint(lambda w: sum(w[i] for i in tech_indices) <= 0.40)

# Ajouter un objectif personnalisé
def custom_objective(w, extra_param=1):
    return extra_param * cp.norm(w, 1)  # Pénalité L1

ef.add_objective(custom_objective, extra_param=0.1)

# Optimiser
weights = ef.max_sharpe()
```

---

# 6. MODÈLE BLACK-LITTERMAN

## 6.1 Introduction

```python
"""
Modèle Black-Litterman (1992)
=============================
Combine un PRIOR (estimation préalable des rendements) avec les VIEWS
(opinions de l'investisseur) pour obtenir un POSTERIOR (estimation combinée).

Avantages:
1. Incorpore les opinions de l'investisseur de façon rationnelle
2. Produit des portefeuilles plus stables que MVO classique
3. Réduit les positions extrêmes

Formule du Posterior:
    E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]

Où:
    π = Prior (souvent market-implied returns)
    P = Matrice de picking (quels actifs concernés par les views)
    Q = Vecteur des views
    Ω = Incertitude sur les views
    τ = Scalaire (poids du prior vs views, typiquement 0.05)
    Σ = Matrice de covariance
"""
from pypfopt import BlackLittermanModel, black_litterman
```

## 6.2 Rendements Implicites du Marché

```python
def market_implied_prior_returns(market_caps, risk_aversion, cov_matrix, risk_free_rate=0.0):
    """
    Calcule les rendements implicites du marché (prior).
    
    Idée: Si le marché est à l'équilibre, les poids de marché sont optimaux.
    On peut "reverse-engineer" les rendements espérés implicites.
    
    Formule:
        π = δΣw_mkt + Rf
    
    Args:
        market_caps: Capitalisation boursière de chaque actif
        risk_aversion (δ): Aversion au risque du marché
        cov_matrix: Matrice de covariance
        risk_free_rate: Taux sans risque
    """
    pass

def market_implied_risk_aversion(market_prices, frequency=252, risk_free_rate=0.0):
    """
    Estime l'aversion au risque implicite du marché.
    
    δ = (R_m - Rf) / σ²_m
    
    Args:
        market_prices: Prix du marché (ex: SPY, S&P 500)
    """
    pass

# Exemple
import yfinance as yf

# Charger les prix du marché (S&P 500)
spy = yf.download('SPY', start='2018-01-01')['Adj Close']

# Estimer l'aversion au risque
delta = black_litterman.market_implied_risk_aversion(spy, risk_free_rate=0.02)
print(f"Aversion au risque implicite: {delta:.2f}")

# Capitalisations boursières (en milliards)
mcaps = {
    'AAPL': 2800,
    'MSFT': 2500,
    'GOOGL': 1800,
    'AMZN': 1500,
    'META': 800,
    'JPM': 500,
    'BAC': 300,
    'XOM': 450
}

# Calculer le prior
prior = black_litterman.market_implied_prior_returns(
    mcaps, 
    delta, 
    S,
    risk_free_rate=0.02
)
print("\nRendements implicites du marché:")
print(prior.sort_values(ascending=False))
```

## 6.3 Définir les Views

```python
"""
Types de Views
==============
1. Absolute Views: "AAPL va retourner 25%"
2. Relative Views: "GOOGL va surperformer META de 5%"

Format pour absolute_views: {ticker: expected_return}
Format pour views détaillées: P (picking matrix) et Q (vector)
"""

# Méthode 1: Views Absolues (simple)
views = {
    'AAPL': 0.25,   # AAPL va faire 25%
    'META': -0.10,  # META va faire -10%
    'JPM': 0.15     # JPM va faire 15%
}

bl = BlackLittermanModel(
    S,
    pi=prior,           # ou "market", "equal"
    absolute_views=views
)

# Méthode 2: Views avec P et Q (pour views relatives)
# View 1: AAPL va faire 25%
# View 2: GOOGL va surperformer META de 5%
# View 3: JPM + BAC vont surperformer XOM de 10%

Q = np.array([0.25, 0.05, 0.10]).reshape(-1, 1)

# P: Matrice de picking (lignes = views, colonnes = actifs)
# Ordre des actifs: AAPL, MSFT, GOOGL, AMZN, META, JPM, BAC, XOM
P = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],       # AAPL
    [0, 0, 1, 0, -1, 0, 0, 0],      # GOOGL - META
    [0, 0, 0, 0, 0, 0.5, 0.5, -1],  # 0.5*JPM + 0.5*BAC - XOM
])

bl = BlackLittermanModel(
    S,
    pi=prior,
    Q=Q,
    P=P
)
```

## 6.4 Classe BlackLittermanModel

```python
class BlackLittermanModel:
    def __init__(
        self,
        cov_matrix,
        pi=None,                    # Prior: array, "market", ou "equal"
        absolute_views=None,        # Views absolues (dict)
        Q=None,                     # Vecteur des views
        P=None,                     # Matrice de picking
        omega=None,                 # Incertitude des views: array, "default", "idzorek"
        view_confidences=None,      # Confiances pour méthode Idzorek
        tau=0.05,                   # Poids prior vs views
        risk_aversion=1,            # Aversion au risque
        market_caps=None,           # Pour pi="market"
        risk_free_rate=0.0
    ):
        """
        Initialise le modèle Black-Litterman.
        
        omega (incertitude des views):
        - "default": Proportionnel à la variance des assets dans la view
        - "idzorek": Utilise view_confidences (0 à 1) pour chaque view
        - array: Matrice diagonale personnalisée
        """
        pass
    
    def bl_returns(self):
        """Calcule les rendements postérieurs."""
        pass
    
    def bl_cov(self):
        """Calcule la covariance postérieure."""
        pass
    
    def bl_weights(self, risk_aversion=None):
        """Calcule les poids implicites (sans optimisation)."""
        pass

# Exemple complet
bl = BlackLittermanModel(
    S,
    pi="market",
    market_caps=mcaps,
    absolute_views={'AAPL': 0.25, 'META': -0.05},
    omega="default",
    tau=0.05
)

# Obtenir les rendements postérieurs
posterior_returns = bl.bl_returns()
print("\nRendements postérieurs Black-Litterman:")
print(posterior_returns.sort_values(ascending=False))

# Utiliser avec EfficientFrontier
ef = EfficientFrontier(posterior_returns, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
```

## 6.5 Méthode Idzorek

```python
"""
Méthode Idzorek (2005)
======================
Permet de spécifier l'incertitude des views en pourcentage de confiance.

Confiance 0% = On ignore complètement la view
Confiance 100% = La view est certaine
"""

# Views avec niveaux de confiance
views = {
    'AAPL': 0.20,   # AAPL +20%
    'GOOGL': 0.15,  # GOOGL +15%  
    'META': -0.10   # META -10%
}

# Confiances (entre 0 et 1)
confidences = [0.8, 0.6, 0.5]  # 80%, 60%, 50%

bl = BlackLittermanModel(
    S,
    pi="market",
    market_caps=mcaps,
    absolute_views=views,
    omega="idzorek",
    view_confidences=confidences,
    tau=0.05
)

posterior = bl.bl_returns()
```

---

# 7. HRP - HIERARCHICAL RISK PARITY

## 7.1 Introduction

```python
"""
Hierarchical Risk Parity (HRP)
==============================
Développé par Marcos López de Prado (2016).

Alternative à MVO qui:
1. N'utilise PAS de matrice inverse (plus stable)
2. N'a PAS besoin des rendements espérés
3. Utilise le clustering hiérarchique pour diversifier

Algorithme en 3 étapes:
1. Tree Clustering: Grouper les actifs par corrélation
2. Quasi-Diagonalization: Réorganiser la matrice de covariance
3. Recursive Bisection: Allouer le capital récursivement

Avantages:
- Plus robuste hors-échantillon que MVO
- Pas besoin d'estimer les rendements espérés
- Produit des portefeuilles naturellement diversifiés
"""
from pypfopt import HRPOpt
```

## 7.2 Classe HRPOpt

```python
class HRPOpt:
    """
    Optimisation par Hierarchical Risk Parity.
    """
    
    def __init__(self, returns=None, cov_matrix=None):
        """
        Initialise HRP.
        
        Args:
            returns: DataFrame de rendements historiques
            cov_matrix: Matrice de covariance (alternative aux returns)
        
        Note: Fournir returns OU cov_matrix (pas les deux obligatoirement).
        """
        pass
    
    def optimize(self, linkage_method="single"):
        """
        Calcule les poids HRP.
        
        Args:
            linkage_method: Méthode de clustering scipy
                - "single": Plus proches voisins
                - "complete": Plus lointains voisins
                - "average": Moyenne des distances
                - "ward": Minimise la variance intra-cluster
        
        Returns:
            OrderedDict: Poids optimaux
        """
        pass

# Exemple
# Calculer les rendements
returns = df.pct_change().dropna()

# HRP
hrp = HRPOpt(returns)
weights = hrp.optimize(linkage_method="single")

print("Poids HRP:")
for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {ticker}: {weight:.2%}")

# Performance
ret, vol, sharpe = hrp.portfolio_performance(verbose=True, frequency=252)
```

## 7.3 Visualisation du Dendrogramme

```python
from pypfopt import plotting
import matplotlib.pyplot as plt

# Créer le portefeuille HRP
hrp = HRPOpt(returns)
weights = hrp.optimize()

# Afficher le dendrogramme
fig, ax = plt.subplots(figsize=(12, 6))
plotting.plot_dendrogram(hrp, ax=ax, show_tickers=True)
plt.title("Clustering Hiérarchique des Actifs")
plt.tight_layout()
plt.show()
```

## 7.4 HRP vs MVO - Comparaison

```python
"""
Comparaison HRP vs MVO
======================
"""
# MVO
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

ef = EfficientFrontier(mu, S)
mvo_weights = ef.max_sharpe()
mvo_ret, mvo_vol, mvo_sharpe = ef.portfolio_performance()

# HRP
hrp = HRPOpt(returns)
hrp_weights = hrp.optimize()
hrp_ret, hrp_vol, hrp_sharpe = hrp.portfolio_performance()

print("Comparaison MVO vs HRP:")
print(f"{'Métrique':<20} {'MVO':>10} {'HRP':>10}")
print("-" * 40)
print(f"{'Rendement'::<20} {mvo_ret:>10.1%} {hrp_ret:>10.1%}")
print(f"{'Volatilité':<20} {mvo_vol:>10.1%} {hrp_vol:>10.1%}")
print(f"{'Sharpe Ratio':<20} {mvo_sharpe:>10.2f} {hrp_sharpe:>10.2f}")
print(f"{'Nb positions':<20} {sum(1 for w in mvo_weights.values() if w>0.01):>10} {sum(1 for w in hrp_weights.values() if w>0.01):>10}")
```

---

# 8. CVaR ET SEMIVARIANCE

## 8.1 CVaR (Conditional Value at Risk)

```python
"""
CVaR (Conditional Value at Risk)
================================
Aussi appelé Expected Shortfall (ES).

VaR (Value at Risk): "La perte maximale avec probabilité (1-β)"
CVaR: "La perte moyenne dans les (1-β)% pires cas"

Exemple (β=95%):
- VaR 95%: "On ne perd pas plus de X dans 95% des cas"
- CVaR 95%: "Dans les 5% pires cas, on perd en moyenne Y"

Avantages du CVaR:
- Cohérent (satisfait les axiomes des mesures de risque)
- Capture le risque de queue (tail risk)
- Convexe (facile à optimiser)
"""
from pypfopt.efficient_frontier import EfficientCVaR

class EfficientCVaR(EfficientFrontier):
    """
    Optimisation sur la frontière moyenne-CVaR.
    """
    
    def __init__(
        self,
        expected_returns,
        returns,              # Rendements historiques (pas juste covariance!)
        beta=0.95,           # Niveau de confiance
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None
    ):
        """
        Args:
            returns: DataFrame de rendements historiques (REQUIS)
            beta: Niveau de confiance (0.95 = CVaR sur les 5% pires cas)
        """
        pass
    
    def min_cvar(self, market_neutral=False):
        """Minimise le CVaR."""
        pass
    
    def efficient_return(self, target_return, market_neutral=False):
        """Minimise le CVaR pour un rendement cible."""
        pass
    
    def efficient_risk(self, target_cvar, market_neutral=False):
        """Maximise le rendement pour un CVaR cible."""
        pass

# Exemple
returns = df.pct_change().dropna()
mu = expected_returns.mean_historical_return(df)

# CVaR au niveau 95%
ef_cvar = EfficientCVaR(mu, returns, beta=0.95)
weights = ef_cvar.min_cvar()

print("Portefeuille Min-CVaR:")
for ticker, weight in ef_cvar.clean_weights().items():
    if weight > 0.01:
        print(f"  {ticker}: {weight:.2%}")

# Performance
ret, cvar = ef_cvar.portfolio_performance(verbose=True)
print(f"\nExpected Return: {ret:.1%}")
print(f"CVaR (95%): {cvar:.2%}")
```

## 8.2 Semivariance

```python
"""
Semivariance / Semidéviation
============================
Mesure de risque qui ne considère que les rendements NÉGATIFS.

Idée: Les investisseurs ne se soucient pas de la "volatilité positive".

Semivariance = E[min(r - B, 0)²]
Où B = benchmark (souvent 0 ou le taux sans risque)
"""
from pypfopt.efficient_frontier import EfficientSemivariance

class EfficientSemivariance(EfficientFrontier):
    """
    Optimisation sur la frontière moyenne-semivariance.
    """
    
    def __init__(
        self,
        expected_returns,
        returns,
        benchmark=0,           # Benchmark pour le downside
        frequency=252,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None
    ):
        pass
    
    def min_semivariance(self, market_neutral=False):
        """Minimise la semivariance."""
        pass
    
    def efficient_return(self, target_return, market_neutral=False):
        """Minimise la semivariance pour un rendement cible."""
        pass
    
    def efficient_risk(self, target_semideviation, market_neutral=False):
        """Maximise le rendement pour une semidéviation cible."""
        pass

# Exemple
ef_semi = EfficientSemivariance(mu, returns, benchmark=0)
weights = ef_semi.min_semivariance()

print("Portefeuille Min-Semivariance:")
for ticker, weight in ef_semi.clean_weights().items():
    if weight > 0.01:
        print(f"  {ticker}: {weight:.2%}")

# Performance
ret, semi = ef_semi.portfolio_performance(verbose=True)
```

---

# 9. ALLOCATION DISCRÈTE

## 9.1 Du Poids Continu aux Actions

```python
"""
Allocation Discrète
===================
Les poids optimaux sont CONTINUS (ex: 15.37%).
En pratique, on doit acheter un NOMBRE ENTIER d'actions.

Problème:
- Portefeuille de $10,000
- AAPL poids optimal = 15.37% = $1,537
- Prix AAPL = $175
- Actions théoriques = $1,537 / $175 = 8.78

On ne peut pas acheter 8.78 actions!

Solution: Algorithmes d'allocation discrète.
"""
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
```

## 9.2 Classe DiscreteAllocation

```python
class DiscreteAllocation:
    """
    Convertit les poids continus en allocation discrète.
    """
    
    def __init__(
        self,
        weights,                    # Dict {ticker: weight}
        latest_prices,              # pd.Series des prix actuels
        total_portfolio_value=10000,
        short_ratio=None            # Pour portefeuilles long/short
    ):
        pass
    
    def greedy_portfolio(self, reinvest=False, verbose=False):
        """
        Allocation gloutonne (greedy).
        
        Algorithme:
        1. Pour chaque actif, acheter floor(weight * value / price) actions
        2. Avec le cash restant, acheter l'actif le plus sous-pondéré
        3. Répéter jusqu'à épuisement du cash
        
        Args:
            reinvest: Réinvestir le cash des ventes à découvert?
            verbose: Afficher les détails
        
        Returns:
            (allocation, leftover): (dict d'actions, cash restant)
        """
        pass
    
    def lp_portfolio(self, reinvest=False, verbose=False, solver=None):
        """
        Allocation par programmation linéaire.
        
        Résout un problème d'optimisation entière pour minimiser
        l'écart par rapport aux poids cibles.
        
        Plus précis mais plus lent que greedy.
        """
        pass

# Exemple complet
# 1. Optimiser
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

# 2. Obtenir les prix actuels
latest_prices = get_latest_prices(df)
# Ou manuellement:
# latest_prices = df.iloc[-1]

# 3. Allocation discrète
da = DiscreteAllocation(
    cleaned_weights,
    latest_prices,
    total_portfolio_value=50000  # $50,000 à investir
)

# Méthode Greedy
allocation, leftover = da.greedy_portfolio(verbose=True)

print("\nAllocation (Greedy):")
print("-" * 40)
for ticker, shares in allocation.items():
    price = latest_prices[ticker]
    value = shares * price
    print(f"  {ticker}: {shares} actions (${value:,.2f})")
print(f"\nCash restant: ${leftover:,.2f}")

# Méthode LP (plus précise)
da2 = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=50000)
allocation_lp, leftover_lp = da2.lp_portfolio(verbose=True)
print(f"\nCash restant (LP): ${leftover_lp:,.2f}")
```

## 9.3 Portefeuille Long/Short

```python
# Pour les portefeuilles avec ventes à découvert
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
weights = ef.efficient_return(target_return=0.15, market_neutral=True)

da = DiscreteAllocation(
    weights,
    latest_prices,
    total_portfolio_value=100000,
    short_ratio=0.3  # 130/30 portfolio
)

allocation, leftover = da.greedy_portfolio(reinvest=True)

print("Allocation Long/Short:")
longs = {k: v for k, v in allocation.items() if v > 0}
shorts = {k: v for k, v in allocation.items() if v < 0}

print("\nPositions LONG:")
for ticker, shares in longs.items():
    print(f"  {ticker}: +{shares} actions")

print("\nPositions SHORT:")
for ticker, shares in shorts.items():
    print(f"  {ticker}: {shares} actions")
```

---

# 10. EXEMPLES COMPLETS

## 10.1 Workflow Standard

```python
"""
Workflow Complet PyPortfolioOpt
===============================
De A à Z: données brutes → ordres d'achat
"""
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# ============================================
# 1. CHARGER LES DONNÉES
# ============================================
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
           'JPM', 'BAC', 'XOM', 'JNJ', 'PG']

df = yf.download(tickers, start='2019-01-01', end='2024-01-01')['Adj Close']
print(f"Données: {df.shape[0]} jours, {df.shape[1]} actifs")

# ============================================
# 2. CALCULER LES ESTIMATIONS
# ============================================
# Rendements espérés (EMA pour plus de réactivité)
mu = expected_returns.ema_historical_return(df, span=252)

# Matrice de covariance (Ledoit-Wolf shrinkage pour stabilité)
S = risk_models.CovarianceShrinkage(df).ledoit_wolf()

print("\nRendements espérés:")
print(mu.sort_values(ascending=False).round(3))

# ============================================
# 3. OPTIMISER LE PORTEFEUILLE
# ============================================
# Max Sharpe avec contraintes
ef = EfficientFrontier(mu, S)

# Contrainte: max 20% par position
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.20))

# Ajouter régularisation L2 pour diversifier
from pypfopt import objective_functions
ef.add_objective(objective_functions.L2_reg, gamma=0.1)

# Optimiser
weights = ef.max_sharpe(risk_free_rate=0.04)  # Rf = 4%
cleaned_weights = ef.clean_weights()

print("\nPoids optimaux:")
for ticker, weight in sorted(cleaned_weights.items(), key=lambda x: -x[1]):
    if weight > 0.01:
        print(f"  {ticker}: {weight:.1%}")

# Performance attendue
ret, vol, sharpe = ef.portfolio_performance(verbose=True, risk_free_rate=0.04)

# ============================================
# 4. ALLOCATION DISCRÈTE
# ============================================
portfolio_value = 100000  # $100,000

latest_prices = get_latest_prices(df)
da = DiscreteAllocation(cleaned_weights, latest_prices, 
                        total_portfolio_value=portfolio_value)

allocation, leftover = da.greedy_portfolio()

print(f"\n{'='*50}")
print(f"ORDRES D'ACHAT (Budget: ${portfolio_value:,})")
print(f"{'='*50}")
print(f"{'Ticker':<8} {'Actions':>8} {'Prix':>10} {'Valeur':>12}")
print("-" * 50)

total_invested = 0
for ticker, shares in allocation.items():
    price = latest_prices[ticker]
    value = shares * price
    total_invested += value
    print(f"{ticker:<8} {shares:>8} ${price:>9.2f} ${value:>11,.2f}")

print("-" * 50)
print(f"{'Total investi':<28} ${total_invested:>11,.2f}")
print(f"{'Cash restant':<28} ${leftover:>11,.2f}")
```

## 10.2 Comparaison de Stratégies

```python
"""
Comparer différentes stratégies d'optimisation
"""
import pandas as pd
import numpy as np
from pypfopt import (
    EfficientFrontier, HRPOpt, BlackLittermanModel,
    expected_returns, risk_models, black_litterman
)

# Données
returns = df.pct_change().dropna()
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

results = {}

# ============================================
# 1. MVO - Max Sharpe
# ============================================
ef = EfficientFrontier(mu, S)
ef.max_sharpe()
ret, vol, sharpe = ef.portfolio_performance()
results['MVO Max Sharpe'] = {'return': ret, 'volatility': vol, 'sharpe': sharpe}

# ============================================
# 2. MVO - Min Volatility
# ============================================
ef = EfficientFrontier(mu, S)
ef.min_volatility()
ret, vol, sharpe = ef.portfolio_performance()
results['MVO Min Vol'] = {'return': ret, 'volatility': vol, 'sharpe': sharpe}

# ============================================
# 3. HRP
# ============================================
hrp = HRPOpt(returns)
hrp.optimize()
ret, vol, sharpe = hrp.portfolio_performance()
results['HRP'] = {'return': ret, 'volatility': vol, 'sharpe': sharpe}

# ============================================
# 4. Equal Weight
# ============================================
n = len(df.columns)
ew_ret = (returns.mean() * 252).mean()
ew_vol = np.sqrt(np.dot(np.ones(n)/n, np.dot(S, np.ones(n)/n)))
ew_sharpe = ew_ret / ew_vol
results['Equal Weight'] = {'return': ew_ret, 'volatility': ew_vol, 'sharpe': ew_sharpe}

# ============================================
# Afficher les résultats
# ============================================
print("\nCOMPARAISON DES STRATÉGIES")
print("=" * 60)
print(f"{'Stratégie':<20} {'Return':>12} {'Volatility':>12} {'Sharpe':>10}")
print("-" * 60)

for name, metrics in results.items():
    print(f"{name:<20} {metrics['return']:>11.1%} {metrics['volatility']:>11.1%} {metrics['sharpe']:>10.2f}")
```

## 10.3 Black-Litterman avec Views

```python
"""
Exemple Black-Litterman Complet
"""
import yfinance as yf
from pypfopt import (
    EfficientFrontier, BlackLittermanModel,
    expected_returns, risk_models, black_litterman
)

# Charger les données
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
df = yf.download(tickers, start='2020-01-01')['Adj Close']
spy = yf.download('SPY', start='2020-01-01')['Adj Close']

# Covariance
S = risk_models.CovarianceShrinkage(df).ledoit_wolf()

# Capitalisation boursière (milliards)
mcaps = {
    'AAPL': 2900, 'MSFT': 2800, 'GOOGL': 1900, 'AMZN': 1600,
    'META': 1200, 'NVDA': 1100, 'TSLA': 800, 'JPM': 500
}

# Aversion au risque implicite
delta = black_litterman.market_implied_risk_aversion(spy)
print(f"Aversion au risque implicite: {delta:.2f}")

# Prior (rendements implicites du marché)
prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
print("\nRendements implicites (prior):")
print(prior.round(3))

# ============================================
# DÉFINIR LES VIEWS
# ============================================
"""
Nos views:
1. NVDA va surperformer le marché de 15% (IA boom)
2. TSLA va sous-performer de 10%
3. AAPL vs MSFT: AAPL surperforme de 3%
"""

views = {
    'NVDA': prior['NVDA'] + 0.15,  # +15% vs prior
    'TSLA': prior['TSLA'] - 0.10,  # -10% vs prior
    'AAPL': prior['AAPL'] + 0.03,  # +3% vs MSFT (simplifié)
}

confidences = [0.75, 0.60, 0.50]  # Niveaux de confiance

# ============================================
# BLACK-LITTERMAN
# ============================================
bl = BlackLittermanModel(
    S,
    pi=prior,
    absolute_views=views,
    omega="idzorek",
    view_confidences=confidences,
    tau=0.05
)

# Rendements postérieurs
posterior = bl.bl_returns()
print("\nRendements postérieurs (Black-Litterman):")
print(posterior.round(3))

# Comparer prior vs posterior
comparison = pd.DataFrame({
    'Prior': prior,
    'Posterior': posterior,
    'Différence': posterior - prior
})
print("\nComparaison Prior vs Posterior:")
print(comparison.round(3))

# ============================================
# OPTIMISER AVEC LE POSTERIOR
# ============================================
ef_bl = EfficientFrontier(posterior, S)
ef_bl.max_sharpe()

print("\nPortefeuille Black-Litterman optimisé:")
for ticker, weight in ef_bl.clean_weights().items():
    if weight > 0.01:
        print(f"  {ticker}: {weight:.1%}")

ef_bl.portfolio_performance(verbose=True)
```

## 10.4 Visualisation

```python
"""
Visualisations avec PyPortfolioOpt
"""
import matplotlib.pyplot as plt
from pypfopt import plotting

# ============================================
# 1. Frontière Efficiente
# ============================================
from pypfopt import CLA

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# CLA permet de tracer la frontière complète
cla = CLA(mu, S)
cla.max_sharpe()

fig, ax = plt.subplots(figsize=(10, 6))
plotting.plot_efficient_frontier(cla, ax=ax, show_assets=True)
plt.title("Frontière Efficiente")
plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150)
plt.show()

# ============================================
# 2. Matrice de Corrélation
# ============================================
fig, ax = plt.subplots(figsize=(10, 8))
plotting.plot_covariance(S, ax=ax, show_tickers=True)
plt.title("Matrice de Covariance")
plt.tight_layout()
plt.savefig("covariance_matrix.png", dpi=150)
plt.show()

# ============================================
# 3. Poids du Portefeuille
# ============================================
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned = ef.clean_weights()

fig, ax = plt.subplots(figsize=(10, 6))
plotting.plot_weights(cleaned, ax=ax)
plt.title("Poids du Portefeuille Max Sharpe")
plt.tight_layout()
plt.savefig("portfolio_weights.png", dpi=150)
plt.show()

# ============================================
# 4. Dendrogramme HRP
# ============================================
returns = df.pct_change().dropna()
hrp = HRPOpt(returns)
hrp.optimize()

fig, ax = plt.subplots(figsize=(12, 6))
plotting.plot_dendrogram(hrp, ax=ax, show_tickers=True)
plt.title("Dendrogramme HRP")
plt.tight_layout()
plt.savefig("hrp_dendrogram.png", dpi=150)
plt.show()
```

---

# ANNEXE: GLOSSAIRE

| Terme | Anglais | Définition |
|-------|---------|------------|
| **MVO** | Mean-Variance Optimization | Optimisation Moyenne-Variance de Markowitz |
| **EF** | Efficient Frontier | Frontière Efficiente - ensemble des portefeuilles optimaux |
| **MVP** | Minimum Variance Portfolio | Portefeuille de variance minimale |
| **Sharpe Ratio** | Sharpe Ratio | Rendement excédentaire par unité de risque |
| **CAPM** | Capital Asset Pricing Model | Modèle d'évaluation des actifs financiers |
| **HRP** | Hierarchical Risk Parity | Parité de risque hiérarchique |
| **CVaR** | Conditional Value at Risk | Valeur à risque conditionnelle (Expected Shortfall) |
| **VaR** | Value at Risk | Valeur à risque |
| **CLA** | Critical Line Algorithm | Algorithme de la ligne critique |
| **B-L** | Black-Litterman | Modèle Black-Litterman |
| **OAS** | Oracle Approximating Shrinkage | Shrinkage approximant l'oracle |
| **L-W** | Ledoit-Wolf | Méthode de shrinkage Ledoit-Wolf |
| **EMA** | Exponential Moving Average | Moyenne mobile exponentielle |

---

# FIN DU GUIDE

Ce guide couvre l'ensemble des fonctionnalités de PyPortfolioOpt.
Pour plus d'informations, consultez la documentation officielle:
https://pyportfolioopt.readthedocs.io/
