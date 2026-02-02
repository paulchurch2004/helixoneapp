# Guide Complet de la Librairie ARCH pour Python
## Modèles ARCH/GARCH, Tests de Racine Unitaire, Bootstrap et Applications Financières

**Version**: Guide exhaustif pour Helix One  
**Source**: Librairie `arch` de Kevin Sheppard  
**Auteur de la librairie**: Kevin Sheppard (Université d'Oxford)

---

## Table des Matières

1. [Introduction et Concepts Fondamentaux](#1-introduction-et-concepts-fondamentaux)
2. [Installation et Configuration](#2-installation-et-configuration)
3. [Modèles de Volatilité Univariés](#3-modèles-de-volatilité-univariés)
4. [Distributions des Erreurs](#4-distributions-des-erreurs)
5. [Modèles de Moyenne](#5-modèles-de-moyenne)
6. [Prévision de Volatilité](#6-prévision-de-volatilité)
7. [Tests de Racine Unitaire](#7-tests-de-racine-unitaire)
8. [Tests de Cointégration](#8-tests-de-cointégration)
9. [Méthodes de Bootstrap](#9-méthodes-de-bootstrap)
10. [Comparaisons Multiples](#10-comparaisons-multiples)
11. [Estimation de Covariance Long Terme](#11-estimation-de-covariance-long-terme)
12. [Applications Pratiques Complètes](#12-applications-pratiques-complètes)
13. [Glossaire et Référence API](#13-glossaire-et-référence-api)

---

## 1. Introduction et Concepts Fondamentaux

### 1.1 Qu'est-ce que la Librairie ARCH ?

La librairie **ARCH** (Autoregressive Conditional Heteroskedasticity, c'est-à-dire Hétéroscédasticité Conditionnelle Autorégressive) est une bibliothèque Python professionnelle pour l'économétrie financière. Elle fournit des outils pour :

- **Modélisation de la volatilité** : ARCH (Autoregressive Conditional Heteroskedasticity), GARCH (Generalized ARCH, c'est-à-dire ARCH Généralisé), EGARCH (Exponential GARCH), TARCH (Threshold ARCH), etc.
- **Tests de racine unitaire** : ADF (Augmented Dickey-Fuller, c'est-à-dire Dickey-Fuller Augmenté), PP (Phillips-Perron), KPSS (Kwiatkowski-Phillips-Schmidt-Shin), etc.
- **Bootstrap** : IID (Independent and Identically Distributed, c'est-à-dire Indépendant et Identiquement Distribué), Stationnaire, par Blocs Circulaires
- **Tests de cointégration** : Engle-Granger, Phillips-Ouliaris
- **Comparaisons multiples** : SPA (Superior Predictive Ability, c'est-à-dire Capacité Prédictive Supérieure), MCS (Model Confidence Set, c'est-à-dire Ensemble de Confiance de Modèles)

### 1.2 Pourquoi Modéliser la Volatilité ?

La **volatilité** mesure l'amplitude des variations de prix d'un actif. Elle est cruciale pour :

1. **Gestion des risques** : La VaR (Value-at-Risk, c'est-à-dire Valeur à Risque) dépend directement de la volatilité
2. **Pricing d'options** : Le modèle Black-Scholes utilise σ (sigma, la volatilité)
3. **Allocation de portefeuille** : Optimisation moyenne-variance de Markowitz
4. **Trading algorithmique** : Ajustement dynamique des positions

#### Exemple Intuitif : Clustering de Volatilité

Les rendements financiers présentent un phénomène appelé "clustering de volatilité" : les grandes variations tendent à être suivies par de grandes variations (positives ou négatives), et les petites par de petites.

```
Jour 1: +3%  ← Grande variation
Jour 2: -2%  ← Grande variation (cluster)
Jour 3: -4%  ← Grande variation (cluster continue)
Jour 4: +0.1% ← Petite variation
Jour 5: -0.2% ← Petite variation (nouveau cluster)
```

Les modèles ARCH/GARCH capturent cette dépendance temporelle de la variance.

### 1.3 Concepts Mathématiques Clés

#### Hétéroscédasticité

- **Homoscédasticité** : Variance constante au cours du temps (σ² = constante)
- **Hétéroscédasticité** : Variance variable au cours du temps (σ²_t change)

#### Processus ARCH(q)

Le modèle ARCH(q) (introduit par Engle, 1982, Prix Nobel 2003) :

```
r_t = μ + ε_t
ε_t = σ_t × z_t    où z_t ~ N(0,1)
σ²_t = ω + α₁ε²_{t-1} + α₂ε²_{t-2} + ... + α_qε²_{t-q}
```

**Interprétation** : La variance actuelle dépend des chocs passés au carré.

**Exemple** : Si q=1, σ²_t = ω + α₁ε²_{t-1}
- Si hier il y a eu un grand choc (ε_{t-1} grand), alors σ²_t sera grand aujourd'hui
- C'est le clustering de volatilité !

#### Processus GARCH(p,q)

Le modèle GARCH(p,q) (Bollerslev, 1986) ajoute des lags (retards) de la variance :

```
σ²_t = ω + Σ(αᵢ × ε²_{t-i}) + Σ(βⱼ × σ²_{t-j})
         i=1 à q              j=1 à p
```

**Interprétation des paramètres** :
- **ω (omega)** : Variance de base (intercept)
- **αᵢ (alpha)** : Réaction aux chocs récents (news coefficient)
- **βⱼ (beta)** : Persistance de la volatilité (decay coefficient)

**Exemple GARCH(1,1)** (le plus utilisé) :
```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

- Si α + β proche de 1 : forte persistance (volatilité met du temps à revenir à la moyenne)
- Si α grand : forte réaction aux nouveaux chocs
- Si β grand : effet mémoire long

---

## 2. Installation et Configuration

### 2.1 Installation

```bash
# Via pip (PyPI)
pip install arch

# Via conda (conda-forge)
conda install arch-py -c conda-forge

# Installation depuis GitHub (dernière version)
pip install git+https://github.com/bashtage/arch.git
```

### 2.2 Dépendances

| Package | Version Minimum | Description |
|---------|-----------------|-------------|
| Python | 3.9+ | Langage de programmation |
| NumPy | 1.19+ | Calcul numérique (tableaux, algèbre linéaire) |
| SciPy | 1.5+ | Fonctions scientifiques (optimisation, statistiques) |
| Pandas | 1.1+ | Manipulation de données (DataFrames) |
| statsmodels | 0.12+ | Modèles statistiques (OLS, etc.) |
| matplotlib | 3+ | Visualisation (optionnel) |

### 2.3 Imports Standard

```python
# Imports principaux
from arch import arch_model
from arch.univariate import (
    ConstantMean, ZeroMean, ARX, HARX,     # Modèles de moyenne
    ARCH, GARCH, EGARCH, TARCH, FIGARCH,   # Modèles de volatilité
    Normal, StudentsT, SkewStudent, GeneralizedError  # Distributions
)

# Tests de racine unitaire
from arch.unitroot import ADF, DFGLS, PhillipsPerron, KPSS, ZivotAndrews, VarianceRatio

# Cointégration
from arch.unitroot.cointegration import engle_granger, phillips_ouliaris

# Bootstrap
from arch.bootstrap import (
    IIDBootstrap, StationaryBootstrap, 
    CircularBlockBootstrap, MovingBlockBootstrap,
    optimal_block_length
)

# Comparaisons multiples
from arch.bootstrap import SPA, StepM, MCS

# Covariance long terme
from arch.covariance.kernel import Bartlett, Parzen, QuadraticSpectral

# Données incluses
import arch.data.sp500      # S&P 500
import arch.data.frenchdata # Facteurs Fama-French
import arch.data.wti        # Pétrole WTI
```

---

## 3. Modèles de Volatilité Univariés

### 3.1 Création Rapide avec `arch_model()`

La fonction `arch_model()` est le point d'entrée principal :

```python
from arch import arch_model
import arch.data.sp500

# Charger les données S&P 500
data = arch.data.sp500.load()
returns = 100 * data['Adj Close'].pct_change().dropna()

# Créer un modèle GARCH(1,1) avec moyenne constante
am = arch_model(returns, vol='GARCH', p=1, q=1)
res = am.fit()
print(res.summary())
```

#### Paramètres de `arch_model()`

| Paramètre | Options | Description |
|-----------|---------|-------------|
| `mean` | `'Constant'`, `'Zero'`, `'AR'`, `'ARX'`, `'HAR'`, `'HARX'`, `'LS'` | Modèle de moyenne |
| `lags` | `int` ou `list[int]` | Lags pour les modèles AR/HAR |
| `vol` | `'GARCH'`, `'ARCH'`, `'EGARCH'`, `'FIGARCH'`, `'APARCH'`, `'HARCH'` | Processus de volatilité |
| `p` | `int` | Ordre des lags de volatilité (ou lags d'innovations pour HARCH) |
| `o` | `int` | Ordre asymétrique (pour GJR-GARCH, TARCH) |
| `q` | `int` | Ordre des lags d'innovations |
| `power` | `float` | Puissance (2.0 = variance, 1.0 = écart-type pour TARCH) |
| `dist` | `'normal'`, `'t'`, `'skewt'`, `'ged'` | Distribution des erreurs |

### 3.2 Modèle GARCH(p,q)

Le modèle GARCH standard :

```
σ²_t = ω + Σ(αᵢ × ε²_{t-i}) + Σ(βⱼ × σ²_{t-j})
```

```python
# GARCH(1,1) - Le plus utilisé
am = arch_model(returns, vol='GARCH', p=1, q=1)
res = am.fit(disp='off')

# Accéder aux paramètres estimés
print(f"omega (ω): {res.params['omega']:.6f}")
print(f"alpha[1] (α): {res.params['alpha[1]']:.6f}")
print(f"beta[1] (β): {res.params['beta[1]']:.6f}")

# Persistance de la volatilité
persistence = res.params['alpha[1]'] + res.params['beta[1]']
print(f"Persistance (α + β): {persistence:.4f}")
# Si proche de 1 : haute persistance (volatilité met du temps à revenir à la moyenne)

# Variance inconditionnelle (long terme)
omega = res.params['omega']
unconditional_var = omega / (1 - persistence)
print(f"Variance inconditionnelle: {unconditional_var:.4f}")
print(f"Volatilité inconditionnelle: {np.sqrt(unconditional_var):.4f}%")
```

**Exemple numérique** :
- ω = 0.01, α = 0.05, β = 0.90
- Persistance = 0.95 (très persistant)
- Variance inconditionnelle = 0.01 / (1 - 0.95) = 0.2
- Volatilité inconditionnelle = √0.2 ≈ 0.45 (45% annualisé)

### 3.3 GJR-GARCH (Asymétrie)

Le modèle GJR-GARCH (Glosten-Jagannathan-Runkle) capture l'**effet de levier** : les chocs négatifs ont souvent un impact plus grand sur la volatilité que les chocs positifs.

```
σ²_t = ω + α × ε²_{t-1} + γ × ε²_{t-1} × I_{[ε_{t-1}<0]} + β × σ²_{t-1}
```

Où I est une fonction indicatrice :
- I = 1 si ε_{t-1} < 0 (choc négatif)
- I = 0 si ε_{t-1} ≥ 0 (choc positif)

```python
# GJR-GARCH(1,1,1) avec le paramètre 'o'
am = arch_model(returns, vol='GARCH', p=1, o=1, q=1)
res = am.fit(disp='off')

print(res.summary())

# gamma > 0 indique un effet de levier
gamma = res.params['gamma[1]']
alpha = res.params['alpha[1]']

print(f"\nImpact d'un choc positif: α = {alpha:.4f}")
print(f"Impact d'un choc négatif: α + γ = {alpha + gamma:.4f}")

if gamma > 0:
    print("→ Effet de levier détecté : les chocs négatifs augmentent plus la volatilité")
```

**Interprétation** :
- Si γ = 0 : pas d'asymétrie (équivalent à GARCH standard)
- Si γ > 0 : les mauvaises nouvelles (rendements négatifs) augmentent plus la volatilité
- Typiquement sur les marchés actions : γ ≈ 0.1 à 0.3

### 3.4 TARCH / ZARCH (Modèle en Écart-Type)

Le modèle TARCH (Threshold ARCH) modélise la **volatilité** (σ) plutôt que la variance (σ²) :

```
σ_t = ω + α × |ε_{t-1}| + γ × |ε_{t-1}| × I_{[ε_{t-1}<0]} + β × σ_{t-1}
```

```python
# TARCH avec power=1.0
am = arch_model(returns, vol='GARCH', p=1, o=1, q=1, power=1.0)
res = am.fit(disp='off')
print(res.summary())
```

#### Modèles Power GARCH

La généralisation avec puissance arbitraire δ :

```
σ^δ_t = ω + α × |ε_{t-1}|^δ + γ × |ε_{t-1}|^δ × I_{[ε_{t-1}<0]} + β × σ^δ_{t-1}
```

- δ = 2 : GARCH standard (variance)
- δ = 1 : TARCH (écart-type)
- δ = 1.5 : compromis souvent optimal empiriquement

```python
# Power GARCH avec delta = 1.5
am = arch_model(returns, vol='GARCH', p=1, o=1, q=1, power=1.5)
res = am.fit(disp='off')
```

### 3.5 EGARCH (Exponentiel)

Le modèle EGARCH (Nelson, 1991) modélise le log de la variance :

```
log(σ²_t) = ω + α × g(z_{t-1}) + β × log(σ²_{t-1})

où g(z) = θ × z + γ × (|z| - E[|z|])
```

**Avantages** :
- Pas de contraintes de positivité (log peut être négatif, mais σ² sera toujours positif)
- Asymétrie naturellement intégrée via θ

```python
# EGARCH(1,1,1)
am = arch_model(returns, vol='EGARCH', p=1, o=1, q=1)
res = am.fit(disp='off')
print(res.summary())
```

### 3.6 FIGARCH (Mémoire Longue)

Le modèle FIGARCH (Fractionally Integrated GARCH) capture la mémoire longue dans la volatilité :

```python
# FIGARCH(1,d,1) - d est le paramètre d'intégration fractionnaire
am = arch_model(returns, vol='FIGARCH', p=1, q=1)
res = am.fit(disp='off')

# Le paramètre 'd' mesure la mémoire longue
d = res.params.get('d', None)
if d is not None:
    print(f"Paramètre d: {d:.4f}")
    print("0 < d < 0.5 : mémoire longue mais stationnarité")
```

### 3.7 HARCH (Heterogeneous ARCH)

Le modèle HARCH utilise des moyennes de carrés de résidus sur différentes fenêtres :

```python
# HARCH avec fenêtres de 1, 5, et 22 jours
am = arch_model(returns, vol='HARCH', p=[1, 5, 22])
res = am.fit(disp='off')
print(res.summary())
```

**Application** : Modèle HAR-RV (Heterogeneous Autoregressive Realized Volatility) pour la volatilité réalisée.

### 3.8 APARCH (Asymmetric Power ARCH)

Le modèle APARCH combine asymétrie et puissance flexible :

```
σ^δ_t = ω + Σαᵢ(|ε_{t-i}| - γᵢε_{t-i})^δ + Σβⱼσ^δ_{t-j}
```

```python
from arch.univariate import APARCH, ConstantMean

# Construction manuelle
am = ConstantMean(returns)
am.volatility = APARCH(p=1, o=1, q=1)
res = am.fit(disp='off')
print(res.summary())
```

### 3.9 Comparaison des Modèles

```python
import pandas as pd
from arch import arch_model

# Comparer plusieurs spécifications
models = {
    'GARCH(1,1)': arch_model(returns, vol='GARCH', p=1, q=1),
    'GJR-GARCH(1,1,1)': arch_model(returns, vol='GARCH', p=1, o=1, q=1),
    'TARCH(1,1,1)': arch_model(returns, vol='GARCH', p=1, o=1, q=1, power=1.0),
    'EGARCH(1,1,1)': arch_model(returns, vol='EGARCH', p=1, o=1, q=1),
}

results = {}
for name, model in models.items():
    res = model.fit(disp='off')
    results[name] = {
        'Log-Likelihood': res.loglikelihood,
        'AIC': res.aic,  # Akaike Information Criterion
        'BIC': res.bic,  # Bayesian Information Criterion
        'Num Params': res.num_params
    }

comparison = pd.DataFrame(results).T
print(comparison)

# Le meilleur modèle a le plus petit AIC/BIC
best_model = comparison['AIC'].idxmin()
print(f"\nMeilleur modèle selon AIC: {best_model}")
```

---

## 4. Distributions des Erreurs

### 4.1 Distribution Normale (Gaussienne)

Distribution par défaut, symétrique avec queues légères.

```python
from arch.univariate import Normal

# La distribution normale n'a pas de paramètres supplémentaires
am = arch_model(returns, dist='normal')  # ou dist='gaussian'
res = am.fit(disp='off')

# Quantiles pour VaR
from scipy.stats import norm
q_01 = norm.ppf(0.01)  # -2.326 (quantile 1%)
q_05 = norm.ppf(0.05)  # -1.645 (quantile 5%)
```

### 4.2 Distribution t de Student

Distribution à queues épaisses (fat tails), capturer les événements extrêmes.

```python
# t de Student avec degrés de liberté estimés
am = arch_model(returns, dist='t')  # ou dist='studentst'
res = am.fit(disp='off')

# Paramètre nu (ν) = degrés de liberté
nu = res.params['nu']
print(f"Degrés de liberté estimés: {nu:.2f}")

# Interprétation :
# ν < 4 : variance infinie (très fat tails)
# ν ≈ 4-8 : fat tails (typique pour les rendements financiers)
# ν > 30 : proche de la normale

# Quantiles pour VaR
from scipy.stats import t
q_01 = t.ppf(0.01, df=nu)
q_05 = t.ppf(0.05, df=nu)
print(f"Quantile 1% (t): {q_01:.3f} vs Normal: {norm.ppf(0.01):.3f}")
```

### 4.3 Distribution t de Student Asymétrique (Skew-t)

Ajoute l'asymétrie à la distribution t.

```python
# Skew-t de Hansen
am = arch_model(returns, dist='skewt')  # ou dist='skewstudent'
res = am.fit(disp='off')

# Paramètres
eta = res.params['eta']   # Degrés de liberté (η > 2)
lam = res.params['lambda']  # Asymétrie (λ ∈ [-1, 1])

print(f"Degrés de liberté (η): {eta:.2f}")
print(f"Asymétrie (λ): {lam:.4f}")

# Interprétation de λ :
# λ < 0 : queue gauche plus épaisse (plus de rendements négatifs extrêmes)
# λ = 0 : symétrique (équivalent à t standard)
# λ > 0 : queue droite plus épaisse
```

### 4.4 Distribution d'Erreur Généralisée (GED)

```python
# GED avec paramètre de forme
am = arch_model(returns, dist='ged')  # ou dist='generalized error'
res = am.fit(disp='off')

# Paramètre nu (ν) = paramètre de forme
nu = res.params['nu']
print(f"Paramètre de forme (ν): {nu:.2f}")

# Interprétation :
# ν = 1 : distribution de Laplace (très fat tails)
# ν = 2 : distribution normale
# ν > 2 : queues plus légères que la normale
```

### 4.5 Comparaison des Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, gennorm

# Comparer les distributions
x = np.linspace(-5, 5, 1000)

fig, ax = plt.subplots(figsize=(12, 6))

# Normale
ax.plot(x, norm.pdf(x), label='Normale', linewidth=2)

# t de Student (ν=6)
ax.plot(x, t.pdf(x, df=6), label='t-Student (ν=6)', linewidth=2)

# t de Student (ν=3)
ax.plot(x, t.pdf(x, df=3), label='t-Student (ν=3)', linewidth=2, linestyle='--')

# GED (ν=1.5)
scale = np.sqrt(gennorm(1.5).var())
ax.plot(x, gennorm.pdf(x * scale, 1.5) * scale, label='GED (ν=1.5)', linewidth=2)

ax.set_xlabel('z')
ax.set_ylabel('Densité')
ax.set_title('Comparaison des Distributions')
ax.legend()
ax.set_xlim(-5, 5)
plt.tight_layout()
plt.savefig('distributions_comparison.png', dpi=150)
plt.show()
```

---

## 5. Modèles de Moyenne

### 5.1 Moyenne Constante

Le modèle le plus simple : r_t = μ + ε_t

```python
from arch.univariate import ConstantMean, GARCH

# Construction explicite
am = ConstantMean(returns)
am.volatility = GARCH(p=1, q=1)
res = am.fit(disp='off')

print(f"Moyenne estimée (μ): {res.params['mu']:.4f}")
```

### 5.2 Moyenne Zéro

Pour les résidus déjà centrés : r_t = ε_t

```python
from arch.univariate import ZeroMean

am = ZeroMean(returns)
am.volatility = GARCH(p=1, q=1)
res = am.fit(disp='off')
```

### 5.3 Modèle AR(p) - Autorégressif

r_t = μ + φ₁r_{t-1} + φ₂r_{t-2} + ... + φₚr_{t-p} + ε_t

```python
from arch.univariate import ARX

# AR(3) sans régresseurs exogènes
am = ARX(returns, lags=[1, 2, 3])
am.volatility = GARCH(p=1, q=1)
res = am.fit(disp='off')
print(res.summary())

# Ou avec arch_model
am = arch_model(returns, mean='AR', lags=3, vol='GARCH', p=1, q=1)
res = am.fit(disp='off')
```

### 5.4 Modèle ARX (AR avec Variables Exogènes)

r_t = μ + Σφᵢr_{t-i} + Σγⱼx_{t,j} + ε_t

```python
import numpy as np

# Créer une variable exogène (ex: VIX)
np.random.seed(42)
vix = 15 + 5 * np.random.randn(len(returns))

# ARX(1) avec VIX comme variable exogène
am = ARX(returns, x=vix, lags=1)
am.volatility = GARCH(p=1, q=1)
res = am.fit(disp='off')
print(res.summary())
```

### 5.5 Modèle HAR (Heterogeneous AR)

Le modèle HAR est populaire pour la volatilité réalisée :

r_t = μ + φ₁r̄_{t-1:t-1} + φ₅r̄_{t-1:t-5} + φ₂₂r̄_{t-1:t-22} + ε_t

où r̄_{t-a:t-b} est la moyenne des rendements de t-a à t-b.

```python
from arch.univariate import HARX

# HAR avec lags moyennés sur 1, 5, et 22 jours
am = HARX(returns, lags=[1, 5, 22])
am.volatility = GARCH(p=1, q=1)
res = am.fit(disp='off')
print(res.summary())

# Ou avec arch_model
am = arch_model(returns, mean='HAR', lags=[1, 5, 22])
res = am.fit(disp='off')
```

### 5.6 Modèle LS (Moindres Carrés)

Régression linéaire pure sur des régresseurs exogènes :

```python
from arch.univariate import LS

# Régression avec constante et variable exogène
am = LS(returns, x=vix)
am.volatility = GARCH(p=1, q=1)
res = am.fit(disp='off')
```

### 5.7 ARCH-in-Mean

Le modèle ARCH-M inclut la volatilité dans l'équation de moyenne :

r_t = μ + δ × f(σ_t) + ε_t

où f peut être σ_t, σ²_t, ou log(σ²_t).

```python
from arch.univariate import ARCHInMean

# ARCH-M avec volatilité dans la moyenne
# form='var' : utilise σ²
# form='vol' : utilise σ
# form='log' : utilise log(σ²)
am = ARCHInMean(returns, form='vol')
am.volatility = GARCH(p=1, q=1)
res = am.fit(disp='off')
print(res.summary())

# Le paramètre delta mesure la prime de risque
delta = res.params.get('delta', None)
if delta:
    print(f"\nPrime de risque (δ): {delta:.4f}")
    print("δ > 0 signifie que les investisseurs demandent un rendement plus élevé quand la volatilité augmente")
```

---

## 6. Prévision de Volatilité

### 6.1 Prévision de Base

```python
# Estimer le modèle
am = arch_model(returns, vol='GARCH', p=1, q=1)
res = am.fit(disp='off')

# Prévision par défaut (1 pas)
forecasts = res.forecast()

# Attributs de l'objet forecast
print("Prévision de la moyenne:")
print(forecasts.mean.tail())

print("\nPrévision de la variance des résidus (σ²):")
print(forecasts.residual_variance.tail())

print("\nPrévision de la variance du processus:")
print(forecasts.variance.tail())
```

### 6.2 Prévisions Multi-Horizons

```python
# Prévision sur 5 jours
forecasts = res.forecast(horizon=5)

print("Variance prévue pour les 5 prochains jours:")
print(forecasts.variance.tail())

# Colonnes: h.1, h.2, ..., h.5
# h.1 = 1 jour ahead, h.5 = 5 jours ahead
```

### 6.3 Méthodes de Prévision

#### Analytique (par défaut)

Utilise les formules fermées quand disponibles :

```python
# Pour GARCH(1,1), les prévisions analytiques sont disponibles
forecasts = res.forecast(horizon=10, method='analytic')
```

#### Simulation

Simule des trajectoires et moyenne :

```python
# Simulation avec 1000 trajectoires
forecasts = res.forecast(horizon=10, method='simulation', simulations=1000)

# Accéder aux trajectoires simulées
sims = forecasts.simulations
print(f"Forme des simulations: {sims.residual_variances.shape}")
# (T - start, simulations, horizon)
```

#### Bootstrap

Utilise les résidus standardisés historiques :

```python
# Bootstrap
forecasts = res.forecast(horizon=10, method='bootstrap', simulations=1000)
```

### 6.4 Prévision Rolling/Recursive

```python
import numpy as np

# Configuration
train_size = len(returns) - 252  # Garder 1 an pour test
forecasts_list = []

# Rolling window
for i in range(252):
    # Estimer sur les données jusqu'à t
    train_data = returns.iloc[:train_size + i]
    model = arch_model(train_data, vol='GARCH', p=1, q=1)
    res = model.fit(disp='off', show_warning=False)
    
    # Prévoir 1 jour ahead
    fcast = res.forecast(horizon=1)
    forecasts_list.append({
        'date': returns.index[train_size + i],
        'forecast_var': fcast.variance.values[-1, 0],
        'actual_return': returns.iloc[train_size + i]
    })

forecasts_df = pd.DataFrame(forecasts_list)
forecasts_df.set_index('date', inplace=True)
forecasts_df['forecast_vol'] = np.sqrt(forecasts_df['forecast_var'])

print(forecasts_df.head(10))
```

### 6.5 Value-at-Risk (VaR)

```python
import numpy as np
from scipy.stats import norm, t

# Estimer un modèle avec distribution t
am = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='skewt')
res = am.fit(disp='off', last_obs='2020-12-31')

# Prévisions pour 2021
forecasts = res.forecast(start='2021-01-01', align='target')

# Extraire moyenne et variance conditionnelles
cond_mean = forecasts.mean['2021':].dropna()
cond_var = forecasts.variance['2021':].dropna()

# Calculer les quantiles de la distribution
# Pour la distribution Skew-t
eta, lam = res.params['eta'], res.params['lambda']
q_01 = am.distribution.ppf([0.01, 0.05], [eta, lam])

print(f"Quantiles Skew-t (1%, 5%): {q_01}")

# VaR paramétrique
VaR_01 = -cond_mean.values - np.sqrt(cond_var.values) * q_01[0]
VaR_05 = -cond_mean.values - np.sqrt(cond_var.values) * q_01[1]

# DataFrame des VaR
var_df = pd.DataFrame({
    'VaR_1%': VaR_01.flatten(),
    'VaR_5%': VaR_05.flatten()
}, index=cond_var.index)

print("\nVaR prévue:")
print(var_df.head())
```

### 6.6 Filtered Historical Simulation (FHS)

```python
# Calculer les résidus standardisés sur l'échantillon d'estimation
std_resids = (returns[:res.model._fit_indices[1]] - res.params['mu']) / res.conditional_volatility
std_resids = std_resids.dropna()

# Quantiles empiriques
q_empirical = std_resids.quantile([0.01, 0.05])
print(f"Quantiles empiriques: {q_empirical.values}")

# VaR avec FHS
VaR_FHS_01 = -cond_mean.values - np.sqrt(cond_var.values) * q_empirical.iloc[0]
VaR_FHS_05 = -cond_mean.values - np.sqrt(cond_var.values) * q_empirical.iloc[1]
```

---

## 7. Tests de Racine Unitaire

### 7.1 Pourquoi Tester la Stationnarité ?

Une série temporelle est **stationnaire** si ses propriétés statistiques (moyenne, variance, autocorrélations) ne changent pas au cours du temps.

**Problèmes avec les séries non-stationnaires** :
- Les régressions peuvent donner des résultats "spurious" (fallacieux)
- Les tests statistiques standards ne sont pas valides
- Les prévisions sont instables

**Exemples** :
- Prix d'actions : NON stationnaire (tendance + racine unitaire)
- Rendements d'actions : généralement stationnaires
- Taux d'intérêt : souvent avec racine unitaire

### 7.2 Test ADF (Augmented Dickey-Fuller)

Le test le plus utilisé. Hypothèses :
- H₀ : La série a une racine unitaire (non stationnaire)
- H₁ : La série est stationnaire

```python
from arch.unitroot import ADF
import arch.data.default

# Charger les données
data = arch.data.default.load()
default_premium = data['BAA'] - data['AAA']  # Spread de crédit

# Test ADF avec sélection automatique des lags
adf = ADF(default_premium)
print(adf.summary())

# Résultats clés
print(f"\nStatistique de test: {adf.stat:.4f}")
print(f"P-value: {adf.pvalue:.4f}")
print(f"Lags utilisés: {adf.lags}")

# Interprétation
if adf.pvalue < 0.05:
    print("→ Rejet de H₀ : la série est stationnaire")
else:
    print("→ Non-rejet de H₀ : la série a peut-être une racine unitaire")
```

#### Options du test ADF

```python
# Spécifier le nombre de lags
adf = ADF(default_premium, lags=5)

# Changer la méthode de sélection des lags
# 'aic', 'bic', 't-stat'
adf = ADF(default_premium, method='aic')

# Changer les termes déterministes
# 'n' : pas de constante ni tendance
# 'c' : constante seulement
# 'ct' : constante et tendance linéaire
# 'ctt' : constante, tendance linéaire et quadratique
adf = ADF(default_premium, trend='ct')
print(adf.summary())
```

### 7.3 Test DF-GLS (Dickey-Fuller GLS)

Version améliorée du ADF avec meilleure puissance :

```python
from arch.unitroot import DFGLS

dfgls = DFGLS(default_premium)
print(dfgls.summary())

# Avec tendance
dfgls_ct = DFGLS(default_premium, trend='ct')
print(dfgls_ct.summary())
```

### 7.4 Test Phillips-Perron

Alternative au ADF qui corrige l'autocorrélation différemment :

```python
from arch.unitroot import PhillipsPerron

# Test PP
pp = PhillipsPerron(default_premium)
print(pp.summary())

# Spécifier le nombre de lags pour la correction de Newey-West
pp = PhillipsPerron(default_premium, lags=12)

# Type de test : 'tau' (t-stat) ou 'rho' (coefficient)
pp_rho = PhillipsPerron(default_premium, test_type='rho')
print(pp_rho.summary())
```

### 7.5 Test KPSS

**Attention** : Hypothèses inversées !
- H₀ : La série est stationnaire
- H₁ : La série a une racine unitaire

```python
from arch.unitroot import KPSS

kpss = KPSS(default_premium)
print(kpss.summary())

# Avec tendance
kpss_ct = KPSS(default_premium, trend='ct')
print(kpss_ct.summary())

# Interprétation
if kpss.pvalue < 0.05:
    print("→ Rejet de H₀ : la série n'est PAS stationnaire")
else:
    print("→ Non-rejet de H₀ : la série semble stationnaire")
```

### 7.6 Test de Zivot-Andrews (Rupture Structurelle)

Teste la racine unitaire en permettant une rupture structurelle :

```python
from arch.unitroot import ZivotAndrews

za = ZivotAndrews(default_premium)
print(za.summary())

# Date de rupture estimée
print(f"Date de rupture: {za.break_date}")
```

### 7.7 Test de Ratio de Variance

Pour tester si une série est un random walk pur vs prévisibilité :

```python
from arch.unitroot import VarianceRatio
import arch.data.frenchdata

# Charger les rendements du marché
ff = arch.data.frenchdata.load()
market_excess = ff.iloc[:, 0]

# Comparer variance 1-mois vs 12-mois
vr = VarianceRatio(market_excess, 12)
print(vr.summary())

# Si rejet : la série n'est pas un random walk pur
# VR < 1 : autocorrélation positive (momentum)
# VR > 1 : autocorrélation négative (mean reversion)
```

### 7.8 Stratégie de Test Complète

```python
def test_stationarity(series, name="Series"):
    """Test complet de stationnarité avec plusieurs tests."""
    print(f"=== Tests de stationnarité pour {name} ===\n")
    
    # ADF
    adf = ADF(series, trend='c')
    print(f"ADF Test:")
    print(f"  Statistique: {adf.stat:.4f}")
    print(f"  P-value: {adf.pvalue:.4f}")
    print(f"  Conclusion: {'Stationnaire' if adf.pvalue < 0.05 else 'Racine unitaire'}\n")
    
    # KPSS
    kpss = KPSS(series, trend='c')
    print(f"KPSS Test:")
    print(f"  Statistique: {kpss.stat:.4f}")
    print(f"  P-value: {kpss.pvalue:.4f}")
    print(f"  Conclusion: {'Racine unitaire' if kpss.pvalue < 0.05 else 'Stationnaire'}\n")
    
    # PP
    pp = PhillipsPerron(series, trend='c')
    print(f"Phillips-Perron Test:")
    print(f"  Statistique: {pp.stat:.4f}")
    print(f"  P-value: {pp.pvalue:.4f}")
    print(f"  Conclusion: {'Stationnaire' if pp.pvalue < 0.05 else 'Racine unitaire'}\n")
    
    # Résumé
    adf_result = 'S' if adf.pvalue < 0.05 else 'NS'
    kpss_result = 'NS' if kpss.pvalue < 0.05 else 'S'
    pp_result = 'S' if pp.pvalue < 0.05 else 'NS'
    
    print(f"Résumé: ADF={adf_result}, KPSS={kpss_result}, PP={pp_result}")
    
    if adf_result == kpss_result == pp_result:
        print(f"→ Conclusion unanime: {'Stationnaire' if adf_result == 'S' else 'Non stationnaire'}")
    else:
        print("→ Résultats mixtes : investigation supplémentaire nécessaire")

# Exemple
test_stationarity(default_premium, "Default Premium")
```

---

## 8. Tests de Cointégration

### 8.1 Concept de Cointégration

Deux séries I(1) (intégrées d'ordre 1, c'est-à-dire avec racine unitaire) sont **cointégrées** si une combinaison linéaire est I(0) (stationnaire).

**Exemple** : Prix spot et prix à terme d'une matière première
- Chaque prix individuellement a une racine unitaire
- Mais leur différence (basis) est stationnaire

### 8.2 Test d'Engle-Granger

Test en deux étapes :
1. Estimer la régression Y = α + βX + ε
2. Tester si les résidus ε ont une racine unitaire

```python
from arch.unitroot.cointegration import engle_granger
import arch.data.crude
import numpy as np

# Charger les prix du pétrole
crude = arch.data.crude.load()
log_wti = np.log(crude['WTI'])
log_brent = np.log(crude['Brent'])

# Test de cointégration
eg_test = engle_granger(log_wti, log_brent, trend='c')
print(eg_test.summary())

# Résultats
print(f"\nStatistique de test: {eg_test.stat:.4f}")
print(f"P-value: {eg_test.pvalue:.4f}")

if eg_test.pvalue < 0.05:
    print("→ Rejet de H₀ : les séries sont cointégrées")
else:
    print("→ Non-rejet de H₀ : pas de preuve de cointégration")

# Vecteur cointégrant
print(f"\nVecteur cointégrant: {eg_test.cointegrating_vector}")
```

### 8.3 Test de Phillips-Ouliaris

Alternative au test d'Engle-Granger avec correction pour l'autocorrélation :

```python
from arch.unitroot.cointegration import phillips_ouliaris

# Quatre types de tests disponibles
# 'Zt' : similaire à PP
# 'Za' : alternative
# 'Pu' : ratio de variance univarié
# 'Pz' : ratio de variance multivarié

po_zt = phillips_ouliaris(log_wti, log_brent, trend='c', test_type='Zt')
print(po_zt.summary())

po_pz = phillips_ouliaris(log_wti, log_brent, trend='c', test_type='Pz')
print(po_pz.summary())
```

### 8.4 Application : Pairs Trading

```python
import numpy as np
import pandas as pd
from arch.unitroot.cointegration import engle_granger
from arch.unitroot import ADF

def find_cointegrated_pairs(prices_df, significance=0.05):
    """
    Trouver les paires cointégrées dans un DataFrame de prix.
    
    Parameters
    ----------
    prices_df : DataFrame
        Prix (colonnes = actifs)
    significance : float
        Seuil de significativité
    
    Returns
    -------
    list of tuples
        Paires cointégrées avec leurs statistiques
    """
    n = prices_df.shape[1]
    tickers = prices_df.columns.tolist()
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            y = np.log(prices_df.iloc[:, i])
            x = np.log(prices_df.iloc[:, j])
            
            # Vérifier que les deux séries sont I(1)
            adf_y = ADF(y, trend='c')
            adf_x = ADF(x, trend='c')
            
            if adf_y.pvalue > 0.10 and adf_x.pvalue > 0.10:
                # Les deux sont probablement I(1)
                eg = engle_granger(y, x, trend='c')
                
                if eg.pvalue < significance:
                    pairs.append({
                        'pair': (tickers[i], tickers[j]),
                        'pvalue': eg.pvalue,
                        'stat': eg.stat,
                        'coef': eg.cointegrating_vector[1]
                    })
    
    return sorted(pairs, key=lambda x: x['pvalue'])

# Exemple avec données simulées
np.random.seed(42)
n_obs = 500

# Créer des séries cointégrées
common_trend = np.cumsum(np.random.randn(n_obs)) * 0.5
prices = pd.DataFrame({
    'A': 100 * np.exp(common_trend + 0.1 * np.cumsum(np.random.randn(n_obs))),
    'B': 100 * np.exp(0.8 * common_trend + 0.1 * np.cumsum(np.random.randn(n_obs))),
    'C': 100 * np.exp(np.cumsum(np.random.randn(n_obs)) * 0.3),
    'D': 100 * np.exp(common_trend + 0.05 * np.cumsum(np.random.randn(n_obs)))
})

cointegrated = find_cointegrated_pairs(prices)
print("Paires cointégrées trouvées:")
for p in cointegrated:
    print(f"  {p['pair']}: p-value={p['pvalue']:.4f}, coef={p['coef']:.4f}")
```

---

## 9. Méthodes de Bootstrap

### 9.1 Bootstrap IID

Pour données indépendantes et identiquement distribuées :

```python
from arch.bootstrap import IIDBootstrap
import numpy as np
import arch.data.frenchdata

# Charger les données
ff = arch.data.frenchdata.load()
excess_market = ff.iloc[:, 0]

# Fonction pour calculer le ratio de Sharpe
def sharpe_ratio(x):
    """Calcule le ratio de Sharpe annualisé."""
    mu = 12 * x.mean()        # Moyenne annualisée
    sigma = np.sqrt(12 * x.var())  # Vol annualisée
    return np.array([mu, sigma, mu / sigma])

# Bootstrap IID
bs = IIDBootstrap(excess_market, seed=42)

# Appliquer la fonction
results = bs.apply(sharpe_ratio, 2500)  # 2500 réplications
print(f"Forme des résultats: {results.shape}")  # (2500, 3)

# Statistiques
print(f"\nMoyenne bootstrap: {results.mean(axis=0)}")
print(f"Écart-type bootstrap: {results.std(axis=0)}")
```

### 9.2 Intervalles de Confiance

```python
# IC par différentes méthodes
ci_basic = bs.conf_int(sharpe_ratio, 1000, method='basic')
ci_percentile = bs.conf_int(sharpe_ratio, 1000, method='percentile')
ci_bca = bs.conf_int(sharpe_ratio, 1000, method='bca')  # Bias-corrected accelerated
ci_studentized = bs.conf_int(sharpe_ratio, 1000, method='studentized')

print("Intervalles de confiance (95%):")
print(f"Basic:       [{ci_basic[0, 2]:.3f}, {ci_basic[1, 2]:.3f}]")
print(f"Percentile:  [{ci_percentile[0, 2]:.3f}, {ci_percentile[1, 2]:.3f}]")
print(f"BCa:         [{ci_bca[0, 2]:.3f}, {ci_bca[1, 2]:.3f}]")
```

### 9.3 Covariance des Paramètres

```python
# Estimer la matrice de covariance
cov = bs.cov(sharpe_ratio, 1000)
print("Matrice de covariance:")
print(cov)

# Erreurs standard
se = np.sqrt(np.diag(cov))
print(f"\nErreurs standard: {se}")
```

### 9.4 Bootstrap Stationnaire

Pour séries temporelles avec dépendance :

```python
from arch.bootstrap import StationaryBootstrap

# Le paramètre est la longueur moyenne des blocs
bs_stat = StationaryBootstrap(12, excess_market, seed=42)

# Les mêmes méthodes sont disponibles
ci = bs_stat.conf_int(sharpe_ratio, 1000, method='percentile')
print(f"IC Stationnaire: [{ci[0, 2]:.3f}, {ci[1, 2]:.3f}]")
```

### 9.5 Bootstrap par Blocs Circulaires

```python
from arch.bootstrap import CircularBlockBootstrap

# Blocs de longueur fixe avec wrap-around
bs_circ = CircularBlockBootstrap(12, excess_market, seed=42)
ci = bs_circ.conf_int(sharpe_ratio, 1000, method='percentile')
print(f"IC Circulaire: [{ci[0, 2]:.3f}, {ci[1, 2]:.3f}]")
```

### 9.6 Bootstrap par Blocs Mobiles

```python
from arch.bootstrap import MovingBlockBootstrap

# Blocs de longueur fixe sans wrap-around
bs_moving = MovingBlockBootstrap(12, excess_market, seed=42)
ci = bs_moving.conf_int(sharpe_ratio, 1000, method='percentile')
print(f"IC Moving Block: [{ci[0, 2]:.3f}, {ci[1, 2]:.3f}]")
```

### 9.7 Longueur Optimale des Blocs

```python
from arch.bootstrap import optimal_block_length

# Estimer la longueur optimale
opt = optimal_block_length(excess_market**2)  # Sur les carrés pour le Sharpe
print("Longueur optimale des blocs:")
print(opt)

# Utiliser la longueur optimale
opt_len = opt['stationary'].values[0]
bs_optimal = StationaryBootstrap(opt_len, excess_market, seed=42)
```

### 9.8 Bootstrap pour Modèles de Régression

```python
import statsmodels.api as sm
import arch.data.binary

# Charger les données
binary = arch.data.binary.load().dropna()

# Préparer les données
endog = binary[['admit']]
exog = sm.add_constant(binary[['gre', 'gpa']])

# Wrapper pour le modèle Probit
def probit_params(endog, exog):
    return sm.Probit(endog, exog).fit(disp=0).params

# Bootstrap
bs = IIDBootstrap(endog=endog, exog=exog, seed=42)
ci = bs.conf_int(probit_params, 1000, method='bca')

print("Intervalles de confiance Probit:")
for i, name in enumerate(['const', 'gre', 'gpa']):
    print(f"  {name}: [{ci[0, i]:.4f}, {ci[1, i]:.4f}]")
```

### 9.9 Bootstrap pour Échantillons Indépendants

```python
from arch.bootstrap import IndependentSamplesBootstrap

# Deux échantillons de tailles différentes
np.random.seed(42)
treatment = 0.2 + np.random.randn(200)  # Groupe traitement
control = np.random.randn(800)           # Groupe contrôle

def mean_diff(x, y):
    return x.mean() - y.mean()

bs = IndependentSamplesBootstrap(treatment, control, seed=42)
ci = bs.conf_int(mean_diff, 1000, method='percentile')
print(f"IC différence de moyennes: [{ci[0, 0]:.3f}, {ci[1, 0]:.3f}]")
```

---

## 10. Comparaisons Multiples

### 10.1 Test SPA (Superior Predictive Ability)

Le test SPA vérifie si un modèle est significativement meilleur qu'un benchmark :

```python
from arch.bootstrap import SPA
import numpy as np

np.random.seed(42)

# Simuler des pertes (MSE par exemple)
n_obs = 500
n_models = 100

# Benchmark
benchmark_losses = np.random.randn(n_obs)**2

# Modèles alternatifs (certains légèrement meilleurs)
model_losses = np.zeros((n_obs, n_models))
for i in range(n_models):
    # Les derniers modèles sont progressivement meilleurs
    improvement = i / n_models * 0.1
    model_losses[:, i] = (np.random.randn(n_obs) * (1 - improvement))**2

# Test SPA
spa = SPA(benchmark_losses, model_losses, seed=42)
spa.compute()

print("P-values du test SPA:")
print(spa.pvalues)

# Interprétation
# Si p-value < 0.05 : au moins un modèle bat significativement le benchmark
```

### 10.2 StepM (Stepwise Multiple Testing)

Identifie QUELS modèles sont supérieurs :

```python
from arch.bootstrap import StepM
import pandas as pd

# Convertir en DataFrame pour les noms
model_losses_df = pd.DataFrame(
    model_losses, 
    columns=[f'model_{i}' for i in range(n_models)]
)

stepm = StepM(benchmark_losses, model_losses_df, seed=42)
stepm.compute()

print(f"Nombre de modèles supérieurs: {len(stepm.superior_models)}")
print(f"Modèles supérieurs: {stepm.superior_models[:10]}...")  # 10 premiers
```

### 10.3 MCS (Model Confidence Set)

Trouve l'ensemble des meilleurs modèles indistinguables :

```python
from arch.bootstrap import MCS

# Réduire pour la démo
losses_subset = model_losses_df.iloc[:, ::10]  # 1 modèle sur 10

mcs = MCS(losses_subset, size=0.10, seed=42)
mcs.compute()

print("P-values MCS:")
print(mcs.pvalues)

print(f"\nModèles inclus dans le MCS (taille 10%):")
print(mcs.included)

print(f"\nModèles exclus:")
print(mcs.excluded)
```

### 10.4 Application : Sélection de Modèles de Volatilité

```python
import numpy as np
import pandas as pd
from arch import arch_model
from arch.bootstrap import MCS

# Données
import arch.data.sp500
data = arch.data.sp500.load()
returns = 100 * data['Adj Close'].pct_change().dropna()

# Définir les modèles à comparer
models_specs = {
    'GARCH(1,1)': {'vol': 'GARCH', 'p': 1, 'q': 1},
    'GARCH(1,1)-t': {'vol': 'GARCH', 'p': 1, 'q': 1, 'dist': 't'},
    'GJR(1,1,1)': {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1},
    'GJR(1,1,1)-t': {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 't'},
    'EGARCH(1,1,1)': {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1},
    'EGARCH(1,1,1)-t': {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 't'},
}

# Estimer et calculer les pertes (QLIKE)
def qlike_loss(actual_var, forecast_var):
    """QLIKE loss function pour comparaison de volatilité."""
    return np.log(forecast_var) + actual_var / forecast_var

# Utiliser les rendements au carré comme proxy de la variance réalisée
realized_var = returns**2

losses_dict = {}
for name, spec in models_specs.items():
    try:
        am = arch_model(returns, **spec)
        res = am.fit(disp='off')
        
        # Variance conditionnelle
        cond_var = res.conditional_volatility**2
        
        # Pertes QLIKE
        losses_dict[name] = qlike_loss(realized_var.values, cond_var.values)
    except Exception as e:
        print(f"Erreur pour {name}: {e}")

losses_df = pd.DataFrame(losses_dict).dropna()
print(f"Forme des pertes: {losses_df.shape}")

# MCS
mcs = MCS(losses_df, size=0.10, seed=42)
mcs.compute()

print("\nModel Confidence Set (90%):")
print(f"Inclus: {mcs.included}")
print(f"\nP-values:")
print(mcs.pvalues.sort_values())
```

---

## 11. Estimation de Covariance Long Terme

### 11.1 Estimateur de Bartlett (Newey-West)

Pour des séries autocorrélées, l'estimateur OLS standard de la variance est biaisé. L'estimateur de Newey-West corrige ce biais :

```python
from arch.covariance.kernel import Bartlett
import arch.data.nasdaq
import numpy as np

# Charger les données
data = arch.data.nasdaq.load()
returns = data[['Adj Close']].pct_change().dropna()

# Estimateur de Bartlett (= Newey-West)
cov_est = Bartlett(returns**2)  # Sur les carrés pour la variance

# Covariance long terme
print(f"Covariance long terme: {cov_est.cov.long_run}")

# Bandwidth utilisé
print(f"Bandwidth: {cov_est.bandwidth}")
```

### 11.2 Autres Noyaux (Kernels)

```python
from arch.covariance.kernel import Parzen, QuadraticSpectral

# Parzen kernel
cov_parzen = Parzen(returns**2)
print(f"Covariance (Parzen): {cov_parzen.cov.long_run}")

# Quadratic Spectral
cov_qs = QuadraticSpectral(returns**2)
print(f"Covariance (QS): {cov_qs.cov.long_run}")
```

### 11.3 Sélection Automatique de la Bandwidth

```python
# La bandwidth est automatiquement sélectionnée par défaut
cov_est = Bartlett(returns**2)
print(f"Bandwidth automatique: {cov_est.bandwidth}")

# Spécifier manuellement
cov_manual = Bartlett(returns**2, bandwidth=10)
print(f"Bandwidth manuel: {cov_manual.bandwidth}")
```

---

## 12. Applications Pratiques Complètes

### 12.1 Modélisation Complète de la Volatilité

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import arch.data.sp500

# 1. Charger et préparer les données
data = arch.data.sp500.load()
returns = 100 * data['Adj Close'].pct_change().dropna()

print("=== Statistiques Descriptives ===")
print(returns.describe())
print(f"Skewness: {returns.skew():.4f}")
print(f"Kurtosis: {returns.kurtosis():.4f}")

# 2. Test de stationnarité
from arch.unitroot import ADF
adf = ADF(returns)
print(f"\n=== Test ADF ===")
print(f"Statistique: {adf.stat:.4f}, P-value: {adf.pvalue:.4f}")

# 3. Estimation du modèle
print("\n=== Estimation GJR-GARCH(1,1,1) avec Skew-t ===")
am = arch_model(returns, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='skewt')
res = am.fit(disp='off')
print(res.summary())

# 4. Diagnostic
print("\n=== Diagnostics ===")
std_resid = res.std_resid

# Test de Ljung-Box sur les résidus standardisés
from scipy.stats import jarque_bera
jb_stat, jb_pval = jarque_bera(std_resid.dropna())
print(f"Jarque-Bera: stat={jb_stat:.2f}, p-value={jb_pval:.4f}")

# Kurtosis des résidus standardisés (devrait être proche de la distribution choisie)
print(f"Kurtosis résidus std: {std_resid.kurtosis():.4f}")

# 5. Prévision
forecasts = res.forecast(horizon=22)  # ~1 mois
print("\n=== Prévision de volatilité (22 jours) ===")
print(f"Dernière volatilité: {np.sqrt(forecasts.variance.values[-1, 0]):.4f}")
print(f"Volatilité moyenne prévue: {np.sqrt(forecasts.variance.values[-1, :].mean()):.4f}")

# 6. VaR
cond_vol = res.conditional_volatility
eta, lam = res.params['eta'], res.params['lambda']
q_05 = am.distribution.ppf(0.05, [eta, lam])
VaR_05 = -res.params['mu'] - cond_vol * q_05
print(f"\nVaR 5% dernière observation: {VaR_05.iloc[-1]:.4f}%")

# 7. Visualisation
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Rendements
axes[0].plot(returns.index, returns, alpha=0.7)
axes[0].set_title('Rendements S&P 500')
axes[0].set_ylabel('Rendement (%)')

# Volatilité conditionnelle
axes[1].plot(cond_vol.index, cond_vol, color='orange')
axes[1].set_title('Volatilité Conditionnelle')
axes[1].set_ylabel('Volatilité (%)')

# Résidus standardisés
axes[2].plot(std_resid.index, std_resid, alpha=0.7)
axes[2].axhline(y=2, color='r', linestyle='--', alpha=0.5)
axes[2].axhline(y=-2, color='r', linestyle='--', alpha=0.5)
axes[2].set_title('Résidus Standardisés')
axes[2].set_ylabel('z')

plt.tight_layout()
plt.savefig('volatility_analysis.png', dpi=150)
plt.show()
```

### 12.2 Backtest de VaR

```python
import numpy as np
import pandas as pd
from arch import arch_model
import arch.data.sp500

# Charger les données
data = arch.data.sp500.load()
returns = 100 * data['Adj Close'].pct_change().dropna()

# Paramètres
train_start = '2000-01-01'
train_end = '2015-12-31'
test_start = '2016-01-01'
test_end = '2018-12-31'

# Données d'entraînement et test
train = returns[train_start:train_end]
test = returns[test_start:test_end]

# Estimation initiale
am = arch_model(train, vol='GARCH', p=1, o=1, q=1, dist='t')
res = am.fit(disp='off')

# Rolling VaR
var_forecasts = []
full_data = returns[train_start:test_end]

for t in range(len(train), len(full_data)):
    # Estimer sur les données jusqu'à t-1
    train_data = full_data.iloc[:t]
    model = arch_model(train_data, vol='GARCH', p=1, o=1, q=1, dist='t')
    res = model.fit(disp='off', show_warning=False)
    
    # Prévoir 1 jour
    fcast = res.forecast(horizon=1)
    cond_var = fcast.variance.values[-1, 0]
    cond_vol = np.sqrt(cond_var)
    
    # VaR paramétrique
    nu = res.params['nu']
    from scipy.stats import t
    q_01 = t.ppf(0.01, df=nu)
    q_05 = t.ppf(0.05, df=nu)
    
    var_forecasts.append({
        'date': full_data.index[t],
        'return': full_data.iloc[t],
        'VaR_1%': -res.params['mu'] - cond_vol * q_01,
        'VaR_5%': -res.params['mu'] - cond_vol * q_05,
    })

var_df = pd.DataFrame(var_forecasts).set_index('date')

# Calcul des violations
var_df['violation_1%'] = var_df['return'] < -var_df['VaR_1%']
var_df['violation_5%'] = var_df['return'] < -var_df['VaR_5%']

# Statistiques
n_test = len(var_df)
print("=== Backtest de VaR ===")
print(f"Période de test: {test_start} à {test_end}")
print(f"Nombre d'observations: {n_test}")
print(f"\nVaR 1%:")
print(f"  Violations attendues: {n_test * 0.01:.1f}")
print(f"  Violations observées: {var_df['violation_1%'].sum()}")
print(f"  Taux de violation: {100 * var_df['violation_1%'].mean():.2f}%")

print(f"\nVaR 5%:")
print(f"  Violations attendues: {n_test * 0.05:.1f}")
print(f"  Violations observées: {var_df['violation_5%'].sum()}")
print(f"  Taux de violation: {100 * var_df['violation_5%'].mean():.2f}%")

# Test de Kupiec (binomial)
from scipy.stats import binom

def kupiec_test(n, violations, alpha):
    """Test de Kupiec pour la couverture inconditionnelle."""
    expected = n * alpha
    pval = 2 * min(
        binom.cdf(violations, n, alpha),
        1 - binom.cdf(violations - 1, n, alpha)
    )
    return pval

p_kupiec_1 = kupiec_test(n_test, var_df['violation_1%'].sum(), 0.01)
p_kupiec_5 = kupiec_test(n_test, var_df['violation_5%'].sum(), 0.05)

print(f"\nTest de Kupiec:")
print(f"  VaR 1%: p-value = {p_kupiec_1:.4f}")
print(f"  VaR 5%: p-value = {p_kupiec_5:.4f}")
```

### 12.3 Simulation de Scénarios de Volatilité

```python
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, SkewStudent

# Créer un modèle pour simulation
np.random.seed(42)

# Paramètres GARCH(1,1)
omega = 0.05
alpha = 0.10
beta = 0.85

# Modèle
vol = GARCH(p=1, q=1)
dist = SkewStudent(seed=42)
sim_mod = ConstantMean(None, volatility=vol, distribution=dist)

# Paramètres [mu, omega, alpha, beta, eta, lambda]
params = np.array([0.05, omega, alpha, beta, 8.0, -0.1])

# Simuler
sim_data = sim_mod.simulate(params, nobs=2520)  # ~10 ans

print("=== Données Simulées ===")
print(sim_data.head())

# Réestimer le modèle sur les données simulées
am = arch_model(sim_data['data'], vol='GARCH', p=1, q=1, dist='skewt')
res = am.fit(disp='off')
print("\n=== Paramètres Réestimés ===")
print(res.params)

# Comparaison
comparison = pd.DataFrame({
    'Vrai': params,
    'Estimé': res.params.values
}, index=['mu', 'omega', 'alpha[1]', 'beta[1]', 'eta', 'lambda'])
print("\n=== Comparaison ===")
print(comparison)
```

### 12.4 Ratio de Sharpe avec Bootstrap

```python
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
import arch.data.frenchdata

# Charger les données
ff = arch.data.frenchdata.load()
excess_market = ff.iloc[:, 0]

# Fonction Sharpe ratio avec IC
def sharpe_analysis(returns, block_length=None, n_bootstrap=5000):
    """
    Analyse complète du ratio de Sharpe avec bootstrap.
    
    Parameters
    ----------
    returns : Series
        Rendements mensuels
    block_length : float, optional
        Longueur des blocs (auto si None)
    n_bootstrap : int
        Nombre de réplications bootstrap
    
    Returns
    -------
    dict
        Résultats de l'analyse
    """
    # Sharpe ratio point estimate
    mu = 12 * returns.mean()
    sigma = np.sqrt(12 * returns.var())
    sr = mu / sigma
    
    # Longueur optimale des blocs
    if block_length is None:
        opt = optimal_block_length(returns**2)
        block_length = opt['stationary'].values[0]
    
    # Bootstrap
    def sharpe_func(x):
        m = 12 * x.mean()
        s = np.sqrt(12 * x.var())
        return np.array([m, s, m / s])
    
    bs = StationaryBootstrap(block_length, returns, seed=42)
    
    # Intervalles de confiance
    ci_percentile = bs.conf_int(sharpe_func, n_bootstrap, method='percentile')
    ci_bca = bs.conf_int(sharpe_func, n_bootstrap, method='bca')
    
    # Covariance
    cov = bs.cov(sharpe_func, n_bootstrap)
    se = np.sqrt(np.diag(cov))
    
    return {
        'mean': mu,
        'volatility': sigma,
        'sharpe_ratio': sr,
        'se_sharpe': se[2],
        'ci_percentile': ci_percentile[:, 2],
        'ci_bca': ci_bca[:, 2],
        'block_length': block_length,
        't_stat': sr / se[2]
    }

# Analyse
results = sharpe_analysis(excess_market)

print("=== Analyse du Ratio de Sharpe ===")
print(f"Moyenne annualisée: {results['mean']:.4f}")
print(f"Volatilité annualisée: {results['volatility']:.4f}")
print(f"Ratio de Sharpe: {results['sharpe_ratio']:.4f}")
print(f"Erreur standard: {results['se_sharpe']:.4f}")
print(f"T-statistique: {results['t_stat']:.4f}")
print(f"\nIC 95% (Percentile): [{results['ci_percentile'][0]:.4f}, {results['ci_percentile'][1]:.4f}]")
print(f"IC 95% (BCa): [{results['ci_bca'][0]:.4f}, {results['ci_bca'][1]:.4f}]")
print(f"\nLongueur optimale des blocs: {results['block_length']:.2f}")

# Test si Sharpe > 0
if results['ci_bca'][0] > 0:
    print("\n→ Le ratio de Sharpe est significativement positif (IC ne contient pas 0)")
else:
    print("\n→ Impossible de rejeter H₀: Sharpe = 0")
```

---

## 13. Glossaire et Référence API

### 13.1 Glossaire des Termes

| Terme | Anglais | Définition |
|-------|---------|------------|
| **ARCH** | Autoregressive Conditional Heteroskedasticity | Hétéroscédasticité conditionnelle autorégressive - modèle où la variance dépend des chocs passés |
| **GARCH** | Generalized ARCH | ARCH généralisé - ajoute des lags de la variance elle-même |
| **GJR-GARCH** | Glosten-Jagannathan-Runkle GARCH | GARCH asymétrique - les chocs négatifs ont un impact différent |
| **EGARCH** | Exponential GARCH | GARCH exponentiel - modélise le log de la variance |
| **TARCH** | Threshold ARCH | ARCH à seuil - modélise la volatilité (pas la variance) |
| **FIGARCH** | Fractionally Integrated GARCH | GARCH à mémoire longue |
| **HARCH** | Heterogeneous ARCH | ARCH hétérogène - plusieurs horizons temporels |
| **APARCH** | Asymmetric Power ARCH | ARCH asymétrique avec puissance flexible |
| **ADF** | Augmented Dickey-Fuller | Test de racine unitaire augmenté |
| **PP** | Phillips-Perron | Alternative au test ADF |
| **KPSS** | Kwiatkowski-Phillips-Schmidt-Shin | Test de stationnarité (hypothèses inversées) |
| **DF-GLS** | Dickey-Fuller GLS | ADF amélioré avec détrending GLS |
| **VaR** | Value-at-Risk | Valeur à risque - perte maximale à un certain niveau de confiance |
| **HAR** | Heterogeneous AR | Autorégressif hétérogène |
| **IID** | Independent and Identically Distributed | Indépendant et identiquement distribué |
| **MCS** | Model Confidence Set | Ensemble de confiance de modèles |
| **SPA** | Superior Predictive Ability | Capacité prédictive supérieure |
| **FHS** | Filtered Historical Simulation | Simulation historique filtrée |
| **BCa** | Bias-Corrected and accelerated | Corrigé du biais et accéléré (méthode d'IC bootstrap) |
| **AIC** | Akaike Information Criterion | Critère d'information d'Akaike |
| **BIC** | Bayesian Information Criterion | Critère d'information bayésien |
| **GED** | Generalized Error Distribution | Distribution d'erreur généralisée |
| **LLF** | Log-Likelihood Function | Fonction de log-vraisemblance |
| **MSE** | Mean Squared Error | Erreur quadratique moyenne |
| **MAD** | Mean Absolute Deviation | Écart absolu moyen |
| **OLS** | Ordinary Least Squares | Moindres carrés ordinaires |
| **QLIKE** | Quasi-Likelihood | Quasi-vraisemblance (fonction de perte pour volatilité) |

### 13.2 Symboles Mathématiques

| Symbole | Nom | Signification |
|---------|-----|---------------|
| σ² | sigma carré | Variance |
| σ | sigma | Écart-type (volatilité) |
| μ | mu | Moyenne |
| ω | omega | Terme constant dans GARCH |
| α | alpha | Coefficient des chocs passés |
| β | beta | Coefficient de persistance |
| γ | gamma | Coefficient d'asymétrie |
| ε | epsilon | Choc/innovation/résidu |
| ν (nu) | nu | Degrés de liberté (distribution t) |
| λ | lambda | Paramètre d'asymétrie (Skew-t) |
| δ | delta | Puissance (Power GARCH) |
| η | eta | Degrés de liberté (Skew-t) |

### 13.3 Référence API Rapide

#### Création de Modèles

```python
# Méthode rapide
from arch import arch_model
am = arch_model(returns, mean='Constant', vol='GARCH', p=1, o=0, q=1, 
                dist='normal', rescale=True)

# Méthode détaillée
from arch.univariate import ConstantMean, GARCH, Normal
am = ConstantMean(returns)
am.volatility = GARCH(p=1, o=0, q=1)
am.distribution = Normal()
```

#### Estimation

```python
res = am.fit(
    update_freq=5,      # Fréquence affichage
    disp='off',         # Désactiver l'affichage
    starting_values=None,  # Valeurs initiales
    cov_type='robust',  # Type de covariance
    show_warning=True   # Afficher les avertissements
)
```

#### Résultats

```python
res.params           # Paramètres estimés
res.std_err          # Erreurs standard
res.tvalues          # T-statistiques
res.pvalues          # P-values
res.loglikelihood    # Log-vraisemblance
res.aic              # AIC
res.bic              # BIC
res.conditional_volatility  # Série de volatilité
res.resid            # Résidus
res.std_resid        # Résidus standardisés
res.summary()        # Résumé complet
res.plot()           # Graphiques
```

#### Prévision

```python
forecasts = res.forecast(
    horizon=5,              # Horizon de prévision
    start=None,             # Date de début
    align='origin',         # 'origin' ou 'target'
    method='analytic',      # 'analytic', 'simulation', 'bootstrap'
    simulations=1000,       # Nombre de simulations
    random_state=None       # Graine aléatoire
)

forecasts.mean              # Prévision moyenne
forecasts.variance          # Prévision variance
forecasts.residual_variance # Prévision variance résidus
forecasts.simulations       # Trajectoires simulées
```

#### Tests de Racine Unitaire

```python
from arch.unitroot import ADF, DFGLS, PhillipsPerron, KPSS, ZivotAndrews, VarianceRatio

# ADF
adf = ADF(y, lags=None, trend='c', method='aic')
adf.stat, adf.pvalue, adf.lags, adf.summary()

# KPSS (hypothèses inversées !)
kpss = KPSS(y, lags=None, trend='c')

# PP
pp = PhillipsPerron(y, lags=None, trend='c', test_type='tau')

# DF-GLS
dfgls = DFGLS(y, lags=None, trend='c')

# Zivot-Andrews
za = ZivotAndrews(y, lags=None, trend='c')
za.break_date  # Date de rupture

# Variance Ratio
vr = VarianceRatio(y, lags=12, overlap=True)
```

#### Cointégration

```python
from arch.unitroot.cointegration import engle_granger, phillips_ouliaris

# Engle-Granger
eg = engle_granger(y, x, trend='c', lags=None)
eg.stat, eg.pvalue, eg.cointegrating_vector, eg.summary()

# Phillips-Ouliaris
po = phillips_ouliaris(y, x, trend='c', test_type='Zt')
```

#### Bootstrap

```python
from arch.bootstrap import (
    IIDBootstrap, 
    StationaryBootstrap, 
    CircularBlockBootstrap, 
    MovingBlockBootstrap,
    IndependentSamplesBootstrap,
    optimal_block_length
)

# Créer le bootstrap
bs = StationaryBootstrap(block_size, data, seed=42)

# Méthodes
bs.apply(func, reps=1000)                    # Appliquer une fonction
bs.conf_int(func, reps, method='percentile') # IC
bs.cov(func, reps)                           # Covariance
bs.reset()                                    # Réinitialiser

# Méthodes d'IC
# 'basic', 'percentile', 'norm', 'bca', 'studentized'
```

#### Comparaisons Multiples

```python
from arch.bootstrap import SPA, StepM, MCS

# SPA
spa = SPA(benchmark_losses, model_losses, seed=42)
spa.compute()
spa.pvalues  # Dict avec 'lower', 'consistent', 'upper'

# StepM
stepm = StepM(benchmark_losses, model_losses_df, seed=42)
stepm.compute()
stepm.superior_models  # Liste des modèles supérieurs

# MCS
mcs = MCS(losses_df, size=0.10, seed=42)
mcs.compute()
mcs.included   # Modèles dans le MCS
mcs.excluded   # Modèles exclus
mcs.pvalues    # P-values
```

---

## Références Bibliographiques

1. **Engle, R.F.** (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation". *Econometrica*, 50(4), 987-1007.

2. **Bollerslev, T.** (1986). "Generalized Autoregressive Conditional Heteroskedasticity". *Journal of Econometrics*, 31(3), 307-327.

3. **Glosten, L.R., Jagannathan, R., & Runkle, D.E.** (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks". *Journal of Finance*, 48(5), 1779-1801.

4. **Nelson, D.B.** (1991). "Conditional Heteroskedasticity in Asset Returns: A New Approach". *Econometrica*, 59(2), 347-370.

5. **Dickey, D.A., & Fuller, W.A.** (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root". *Journal of the American Statistical Association*, 74(366), 427-431.

6. **Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y.** (1992). "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root". *Journal of Econometrics*, 54(1-3), 159-178.

7. **Engle, R.F., & Granger, C.W.J.** (1987). "Co-Integration and Error Correction: Representation, Estimation, and Testing". *Econometrica*, 55(2), 251-276.

8. **Politis, D.N., & Romano, J.P.** (1994). "The Stationary Bootstrap". *Journal of the American Statistical Association*, 89(428), 1303-1313.

9. **White, H.** (2000). "A Reality Check for Data Snooping". *Econometrica*, 68(5), 1097-1126.

10. **Hansen, P.R.** (2005). "A Test for Superior Predictive Ability". *Journal of Business & Economic Statistics*, 23(4), 365-380.

---

**Note**: Ce guide a été créé pour Helix One. Pour la documentation officielle complète, consultez [arch.readthedocs.io](https://arch.readthedocs.io/).
