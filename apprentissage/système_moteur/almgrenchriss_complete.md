# Guide Complet : Modèle d'Exécution Optimale Almgren-Chriss
## Documentation Technique pour HelixOne

---

## Table des Matières

1. [Introduction au Problème d'Exécution Optimale](#1-introduction)
2. [Fondements Théoriques](#2-fondements-théoriques)
3. [Modèle Single Asset (Actif Unique)](#3-modèle-single-asset)
4. [Modèle Multi-Assets (Actifs Multiples)](#4-modèle-multi-assets)
5. [Calibration des Paramètres](#5-calibration-des-paramètres)
6. [Implémentation Python Complète](#6-implémentation-python)
7. [Analyse de Sensibilité](#7-analyse-de-sensibilité)
8. [Application Avancée : Cross-Market Trading](#8-cross-market-trading)
9. [Extensions et Limites](#9-extensions-et-limites)
10. [Glossaire](#10-glossaire)

---

## 1. Introduction au Problème d'Exécution Optimale {#1-introduction}

### 1.1 Contexte

Lorsqu'un investisseur institutionnel souhaite liquider une position importante (par exemple, vendre 1 million d'actions), il fait face à un dilemme fondamental :

- **Vendre rapidement** : Réduit le risque de marché (le prix peut chuter pendant la liquidation) mais génère un **impact de marché** (market impact) important qui fait baisser le prix de vente
- **Vendre lentement** : Minimise l'impact de marché mais expose à un **risque de volatilité** (volatility risk) plus élevé

Le modèle **Almgren-Chriss** (du nom de Robert Almgren et Neil Chriss, 2001) fournit un cadre mathématique pour trouver la **trajectoire de trading optimale** qui minimise une combinaison du risque de volatilité et des coûts de transaction.

### 1.2 Référence Académique

> **Almgren, R. and Chriss, N., 2001. "Optimal execution of portfolio transactions."**
> The Journal of Risk, 3(2), pp.5-39.

Cette publication est considérée comme l'un des articles fondateurs de la finance quantitative moderne et est largement utilisée dans l'industrie pour l'exécution algorithmique d'ordres.

### 1.3 Applications Pratiques

| Application | Description | Exemple |
|-------------|-------------|---------|
| **Liquidation de fonds** | Vente ordonnée d'un portefeuille | Fermeture d'un hedge fund |
| **Rebalancement** | Ajustement de positions | ETF (Exchange-Traded Fund, fonds indiciel coté) rééquilibrant son portefeuille |
| **Acquisition** | Achat progressif d'une position | Accumulation discrète d'actions |
| **VWAP/TWAP** | Algorithmes d'exécution | Ordres algorithmiques institutionnels |

---

## 2. Fondements Théoriques {#2-fondements-théoriques}

### 2.1 Impact de Marché (Market Impact)

L'**impact de marché** désigne l'effet qu'un ordre de trading a sur le prix d'un actif. Il se décompose en deux types :

#### 2.1.1 Impact Permanent (Permanent Impact)

L'**impact permanent** (noté **γ** - gamma) représente le changement de prix qui persiste après l'exécution de l'ordre.

**Exemple concret** :
> Vous vendez 100 000 actions d'une entreprise. Cette vente massive signale au marché qu'un acteur important se désengage, ce qui fait baisser définitivement le prix de 0.50€. Ce changement de prix reste même après que vous ayez fini de vendre.

**Formule** : Si on vend `n` unités, le prix baisse de `γ × n` de façon permanente.

#### 2.1.2 Impact Temporaire (Temporary Impact)

L'**impact temporaire** (noté **η** - eta) représente le changement de prix causé par la pression d'achat/vente immédiate, qui disparaît ensuite.

**Exemple concret** :
> Vous passez un ordre de vente de 50 000 actions en une journée. La pression vendeuse fait baisser le prix de 0.30€ pendant cette journée. Le lendemain, le prix remonte car la pression a disparu.

**Formule** : Si on vend à un taux `v` (actions/jour), le coût temporaire est `η × v`.

### 2.2 Le Trade-off Fondamental

```
                    RISQUE
                      ↑
     Vente rapide     │      ● Haut risque
     (gros impact)    │        (coûts élevés, mais position fermée vite)
                      │
                      │
                      │            ○ Zone optimale
                      │              (Almgren-Chriss)
                      │
     Vente lente      │
     (petit impact)   │      ● Bas coûts
                      │        (mais exposition longue au marché)
                      └─────────────────────────→ COÛTS
```

### 2.3 Fonction de Coût

Le modèle Almgren-Chriss optimise une fonction d'**utilité moyenne-variance** (mean-variance utility) :

```
U = E[Coût] + λ × Var[Coût]
```

Où :
- **E[Coût]** : Espérance mathématique du coût total de trading
- **Var[Coût]** : Variance du coût total (mesure du risque)
- **λ** (lambda) : Paramètre d'**aversion au risque** (risk aversion)
  - λ petit → Agent peu averse au risque → Vente lente
  - λ grand → Agent très averse au risque → Vente rapide

---

## 3. Modèle Single Asset (Actif Unique) {#3-modèle-single-asset}

### 3.1 Notations et Paramètres

| Symbole | Nom | Description | Unité typique |
|---------|-----|-------------|---------------|
| **X** | Position initiale | Nombre d'actions à liquider | Actions |
| **T** | Horizon | Temps total pour liquider | Jours |
| **N** | Nombre de périodes | Nombre d'opportunités de trading | - |
| **τ** (tau) | Durée d'une période | τ = T/N | Jours |
| **λ** (lambda) | Aversion au risque | Plus élevé = plus prudent | - |
| **σ** (sigma) | Volatilité | Écart-type des rendements | $/jour |
| **ε** (epsilon) | Coût fixe | Demi-spread bid/ask + frais | $/action |
| **η** (eta) | Impact temporaire | Coût proportionnel au taux | $/action² |
| **γ** (gamma) | Impact permanent | Dépréciation permanente du prix | $/action² |

### 3.2 Trajectoire de Trading

Une **trajectoire de trading** est un vecteur `(x₀, x₁, ..., xₙ)` où `xₖ` représente le nombre d'actions détenues au temps `tₖ = k × τ`.

**Contraintes** :
- `x₀ = X` (on commence avec X actions)
- `xₙ = 0` (on finit avec 0 actions)

La **liste de trades** est `nₖ = xₖ₋₁ - xₖ` (nombre d'actions vendues à la période k).

### 3.3 Formules de la Trajectoire Optimale

#### Étape 1 : Calcul des paramètres intermédiaires

```
η̃ = η - (γ × τ) / 2
```

Où **η̃** (eta tilde) est l'impact temporaire ajusté.

```
κ̃² = (λ × σ²) / η̃
```

Où **κ̃²** (kappa tilde squared) combine aversion au risque et volatilité.

#### Étape 2 : Calcul de κ (kappa)

```
κ = (1/τ) × arccosh(κ̃² × τ² / 2 + 1)
```

#### Étape 3 : Trajectoire optimale

Pour chaque période k = 0, ..., N :

```
xₖ = [sinh(κ × (T - tₖ)) / sinh(κ × T)] × X
```

Où **sinh** est le sinus hyperbolique : `sinh(x) = (eˣ - e⁻ˣ) / 2`

### 3.4 Implémentation Python

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class AlmgrenChriss1D:
    """
    Modèle Almgren-Chriss pour l'exécution optimale d'un actif unique.
    
    Le modèle minimise une combinaison du risque de volatilité et des coûts
    de transaction dus à l'impact de marché permanent et temporaire.
    """
    
    def __init__(self, params):
        """
        Initialise le modèle avec les paramètres de marché.
        
        Parameters
        ----------
        params : dict
            Dictionnaire contenant :
            - 'lambda' : Aversion au risque (float)
            - 'sigma' : Volatilité du prix (float)
            - 'epsilon' : Coût fixe par action (float)
            - 'eta' : Paramètre d'impact temporaire (float)
            - 'gamma' : Paramètre d'impact permanent (float)
            - 'tau' : Durée d'une période en jours (float)
        """
        # Extraction des paramètres
        self._lambda = params['lambda']      # Aversion au risque
        self._sigma = params['sigma']        # Volatilité (écart-type)
        self._epsilon = params['epsilon']    # Demi-spread + frais
        self._eta = params['eta']            # Impact temporaire
        self._gamma = params['gamma']        # Impact permanent
        self._tau = params['tau']            # Durée d'une période
        
        # Calcul de l'impact temporaire ajusté
        # η̃ = η - γτ/2
        self._eta_tilda = self._eta - 0.5 * self._gamma * self._tau
        
        # Vérification : η̃ doit être positif pour que le problème soit convexe
        # (garantit une solution unique)
        assert self._eta_tilda > 0, "η̃ doit être > 0 pour un problème quadratique"
        
        # Calcul de κ̃² = λσ² / η̃
        self._kappa_tilda_squared = (self._lambda * self._sigma**2) / self._eta_tilda
        
        # Calcul de κ via la formule : κ = (1/τ) × arccosh(κ̃²τ²/2 + 1)
        self._kappa = np.arccosh(
            0.5 * (self._kappa_tilda_squared * self._tau**2) + 1
        ) / self._tau
        
    def trajectory(self, X, T):
        """
        Calcule la trajectoire de liquidation optimale.
        
        Parameters
        ----------
        X : int
            Position initiale (nombre d'actions à liquider)
        T : int
            Horizon de temps (nombre de périodes)
            
        Returns
        -------
        np.array
            Vecteur de taille T+1 contenant le nombre d'actions
            détenues à chaque période [x₀, x₁, ..., xₜ]
        """
        trajectory = []
        
        for t in range(T):
            # Formule : xₜ = sinh(κ(T-t)) / sinh(κT) × X
            x = int(
                np.sinh(self._kappa * (T - t)) / 
                np.sinh(self._kappa * T) * X
            )
            trajectory.append(x)
            
        # Position finale = 0
        trajectory.append(0)
        
        return np.array(trajectory)
    
    def strategy(self, X, T):
        """
        Calcule la liste des trades optimaux (nombre d'actions à vendre par période).
        
        Parameters
        ----------
        X : int
            Position initiale
        T : int
            Horizon de temps
            
        Returns
        -------
        np.array
            Vecteur de taille T contenant le nombre d'actions
            à vendre à chaque période [n₁, n₂, ..., nₜ]
        """
        # nₖ = xₖ₋₁ - xₖ (différence entre positions consécutives)
        return -np.diff(self.trajectory(X, T))
    
    def expected_cost(self, X, T):
        """
        Calcule le coût espéré de la stratégie.
        
        Returns
        -------
        float
            Coût espéré en unités monétaires
        """
        n = self.strategy(X, T)
        
        # Coût permanent : γ × Σnₖ × (X - Σⱼ<ₖ nⱼ)
        cumsum = np.cumsum(n)
        permanent_cost = self._gamma * np.sum(n * (X - np.concatenate([[0], cumsum[:-1]])))
        
        # Coût temporaire : η × Σ(nₖ/τ)²
        temporary_cost = self._eta * np.sum((n / self._tau)**2) * self._tau
        
        # Coût fixe : ε × X
        fixed_cost = self._epsilon * X
        
        return permanent_cost + temporary_cost + fixed_cost
    
    def variance_cost(self, X, T):
        """
        Calcule la variance du coût de la stratégie.
        
        Returns
        -------
        float
            Variance du coût
        """
        traj = self.trajectory(X, T)
        
        # Variance = σ² × τ × Σxₖ²
        return self._sigma**2 * self._tau * np.sum(traj[:-1]**2)


# =============================================================================
# EXEMPLE D'UTILISATION : Liquidation d'actions Google (GOOG)
# =============================================================================

def example_single_asset():
    """
    Exemple complet : liquidation de 250 000 actions GOOG sur 3 mois.
    """
    
    # --- 1. Téléchargement des données ---
    print("=" * 60)
    print("MODÈLE ALMGREN-CHRISS - ACTIF UNIQUE")
    print("=" * 60)
    
    ticker = yf.Ticker('GOOG')
    data = ticker.history(period='3mo')
    
    print(f"\nDonnées téléchargées : {len(data)} jours de trading")
    print(f"Prix de clôture moyen : ${data['Close'].mean():.2f}")
    print(f"Volume moyen journalier : {data['Volume'].mean():,.0f} actions")
    
    # --- 2. Calibration des paramètres ---
    
    # Volume et spread moyens
    average_daily_volume = np.mean(data['Volume'])
    average_daily_spread = np.mean(data['High'] - data['Low'])
    
    # Volatilité (écart-type des prix de clôture)
    sigma = np.std(data['Close'])
    
    # Coût fixe (demi-spread)
    epsilon = average_daily_spread / 2
    
    # Impact temporaire (calibration empirique)
    # η = Spread / (0.01 × Volume moyen)
    eta = average_daily_spread / (0.01 * average_daily_volume)
    
    # Impact permanent (calibration empirique)
    # γ = Spread / (0.1 × Volume moyen)
    gamma = average_daily_spread / (0.1 * average_daily_volume)
    
    # Période = 1 jour
    tau = 1
    
    params = {
        'lambda': 1e-8,    # Aversion au risque (faible)
        'sigma': sigma,
        'epsilon': epsilon,
        'eta': eta,
        'gamma': gamma,
        'tau': tau
    }
    
    print("\n--- Paramètres calibrés ---")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k:10} = {v:.6e}")
        else:
            print(f"  {k:10} = {v}")
    
    # --- 3. Calcul de la trajectoire optimale ---
    
    X = 250_000  # 250 000 actions à liquider
    T = len(data)  # Sur toute la période (≈ 63 jours)
    
    model = AlmgrenChriss1D(params)
    trajectory = model.trajectory(X, T)
    strategy = model.strategy(X, T)
    
    print(f"\n--- Résultats ---")
    print(f"Position initiale : {X:,} actions")
    print(f"Horizon : {T} jours")
    print(f"Actions vendues jour 1 : {strategy[0]:,.0f}")
    print(f"Actions vendues jour {T//2} : {strategy[T//2]:,.0f}")
    print(f"Actions vendues jour {T-1} : {strategy[-1]:,.0f}")
    
    # --- 4. Visualisation ---
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Graphique 1 : Trajectoire
    axes[0].plot(range(T + 1), trajectory, 'o-', markersize=4, linewidth=1.5)
    axes[0].set_title(
        f'Trajectoire de Liquidation Optimale ({X:,} actions en {T} jours)',
        fontsize=14
    )
    axes[0].set_xlabel('Temps (jours)', fontsize=12)
    axes[0].set_ylabel('Nombre d\'actions détenues', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(range(T + 1), trajectory, alpha=0.3)
    
    # Graphique 2 : Stratégie (actions vendues par jour)
    axes[1].bar(range(1, T + 1), strategy, color='steelblue', alpha=0.7)
    axes[1].set_title('Actions Vendues par Jour', fontsize=14)
    axes[1].set_xlabel('Jour', fontsize=12)
    axes[1].set_ylabel('Actions vendues', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('almgren_chriss_single_asset.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, trajectory, strategy


if __name__ == "__main__":
    example_single_asset()
```

### 3.5 Interprétation des Résultats

La trajectoire optimale a une forme caractéristique :

```
Actions
détenues
    │
  X │●
    │ ●
    │  ●
    │   ●
    │     ●
    │       ●
    │          ●
    │              ●
    │                    ●
  0 │─────────────────────────●
    └─────────────────────────────→ Temps
    0                           T
```

- **Début** : Vente agressive (pour réduire l'exposition au risque)
- **Fin** : Vente plus douce (l'exposition résiduelle est faible)

Cette forme en **"fonction sinh décroissante"** est caractéristique du modèle Almgren-Chriss.

---

## 4. Modèle Multi-Assets (Actifs Multiples) {#4-modèle-multi-assets}

### 4.1 Extension au Cas Multi-Dimensionnel

Lorsqu'on liquide plusieurs actifs simultanément, on doit prendre en compte :

1. **La corrélation entre actifs** : Si GOOG et FB sont corrélés, leurs prix bougent ensemble
2. **L'impact croisé** : Vendre GOOG peut affecter le prix de FB (et vice versa)

### 4.2 Nouveaux Paramètres Matriciels

| Symbole | Dimension | Description |
|---------|-----------|-------------|
| **X** | m × 1 | Vecteur de positions initiales |
| **C** | m × m | Matrice de variance-covariance (remplace σ²) |
| **H** | m × m | Matrice d'impact temporaire (remplace η) |
| **Γ** | m × m | Matrice d'impact permanent (remplace γ) |

#### 4.2.1 Matrice de Covariance C

La matrice **C** capture la volatilité de chaque actif et leurs corrélations :

```
C = | Var(Asset1)      Cov(Asset1,2) |
    | Cov(Asset2,1)    Var(Asset2)   |
```

**Exemple** :
```
C = | 100   50 |
    |  50   80 |
```
- Variance de l'actif 1 : 100
- Variance de l'actif 2 : 80
- Covariance (corrélation) : 50 (les actifs sont positivement corrélés)

#### 4.2.2 Matrice d'Impact Γ (Gamma)

La matrice **Γ** représente l'impact permanent :

```
Γ = | γ₁₁   γ₁₂ |
    | γ₂₁   γ₂₂ |
```

- **γᵢᵢ** : Impact de la vente de l'actif i sur son propre prix
- **γᵢⱼ** (i ≠ j) : **Impact croisé** - effet de la vente de l'actif j sur le prix de l'actif i

**Exemple d'impact croisé** :
> Vous vendez des actions Apple (AAPL). Cette vente fait aussi baisser le prix de Microsoft (MSFT) car les deux entreprises sont dans le même secteur et les investisseurs perçoivent un signal négatif pour le secteur tech.

#### 4.2.3 Matrice d'Impact H

La matrice **H** représente l'impact temporaire avec la même structure que Γ.

### 4.3 Formules pour le Cas Multi-Actifs

#### Étape 1 : Décomposition des matrices

```
Γˢ = (Γᵀ + Γ) / 2      # Partie symétrique de Γ
Γᴬ = (Γ - Γᵀ) / 2      # Partie anti-symétrique de Γ
Hˢ = (Hᵀ + H) / 2      # Partie symétrique de H
```

#### Étape 2 : Calcul de H̃ (H tilde)

```
H̃ = Hˢ - (τ/2) × Γˢ
```

#### Étape 3 : Racine carrée matricielle

```
H̃^(1/2) × H̃^(1/2) = H̃
```

#### Étape 4 : Matrices transformées

```
A = (H̃^(1/2))⁻¹ × C × (H̃^(1/2))⁻¹
B = (H̃^(1/2))⁻¹ × Γᴬ × (H̃^(1/2))⁻¹
```

#### Étape 5 : Système linéaire

Pour le cas diagonal (H et Γ diagonales), on utilise la décomposition en valeurs propres :

```
λA = U × D × Uᵀ
```

Où U est orthogonale et D diagonale avec les valeurs propres κ̃₁², ..., κ̃ₘ².

### 4.4 Implémentation Python Multi-Assets

```python
from scipy.linalg import sqrtm
import numpy as np

def decompose(A):
    """
    Décompose une matrice en parties symétrique et anti-symétrique.
    
    Pour toute matrice A : A = Aˢ + Aᴬ
    où Aˢ = (A + Aᵀ)/2 et Aᴬ = (A - Aᵀ)/2
    
    Parameters
    ----------
    A : np.array
        Matrice carrée à décomposer
        
    Returns
    -------
    tuple
        (Aˢ, Aᴬ) : parties symétrique et anti-symétrique
    """
    A_symmetric = 0.5 * (A + A.T)
    A_antisymmetric = 0.5 * (A - A.T)
    return (A_symmetric, A_antisymmetric)


def is_diagonal(A):
    """
    Vérifie si une matrice est diagonale.
    
    Parameters
    ----------
    A : np.array
        Matrice à vérifier
        
    Returns
    -------
    bool
        True si la matrice est diagonale
    """
    i, j = np.nonzero(A)
    return np.all(i == j)


class AlmgrenChriss:
    """
    Modèle Almgren-Chriss pour l'exécution optimale de plusieurs actifs.
    
    Ce modèle étend le cas unidimensionnel en prenant en compte :
    - La corrélation entre actifs (matrice de covariance C)
    - L'impact croisé (matrices H et Γ non-diagonales)
    
    Deux modes de résolution sont supportés :
    - Mode diagonal : H et Γ diagonales (solution analytique)
    - Mode général : H et Γ quelconques (résolution numérique)
    """
    
    def __init__(self, params):
        """
        Initialise le modèle multi-actifs.
        
        Parameters
        ----------
        params : dict
            Dictionnaire contenant :
            - 'lambda' : Aversion au risque (float)
            - 'C' : Matrice de covariance (np.array, m×m)
            - 'epsilon' : Coût fixe moyen (float)
            - 'H' : Matrice d'impact temporaire (np.array, m×m)
            - 'Gamma' : Matrice d'impact permanent (np.array, m×m)
            - 'tau' : Durée d'une période (float)
        """
        # Extraction des paramètres
        self._lambda = params['lambda']
        self._C = params['C']                # Matrice de covariance
        self._epsilon = params['epsilon']
        self._H = params['H']                # Impact temporaire
        self._Gamma = params['Gamma']        # Impact permanent
        self._tau = params['tau']
        
        # Nombre d'actifs
        self._dims = self._C.shape[0]
        
        # Décomposition des matrices en parties symétriques/anti-symétriques
        self._H_s, self._H_a = decompose(self._H)
        self._Gamma_s, self._Gamma_a = decompose(self._Gamma)
        
        # Calcul de H̃ = Hˢ - (τ/2)Γˢ
        self._H_tilda = self._H_s - 0.5 * self._Gamma_s * self._tau
        
        # Vérification : H̃ doit être définie positive (toutes les valeurs propres > 0)
        eigenvalues = np.linalg.eigvals(self._H_tilda)
        assert np.all(eigenvalues > 0), f"H̃ doit être définie positive. Valeurs propres: {eigenvalues}"
        
        # Racine carrée matricielle de H̃
        self._H_tilda_sqrt = sqrtm(self._H_tilda)
        self._H_tilda_sqrt_inv = np.linalg.inv(self._H_tilda_sqrt)
        
        # Matrices transformées A et B
        self._A = self._H_tilda_sqrt_inv @ self._C @ self._H_tilda_sqrt_inv
        self._B = self._H_tilda_sqrt_inv @ self._Gamma_a @ self._H_tilda_sqrt_inv
        
        # Si le problème est diagonal, on peut utiliser une solution analytique
        if is_diagonal(self._H_tilda):
            # Décomposition en valeurs/vecteurs propres de λA
            self._kappa_tilda_squareds, self._U = np.linalg.eig(self._lambda * self._A)
            
            # Calcul des κⱼ pour chaque valeur propre
            self._kappas = np.arccosh(
                0.5 * (self._kappa_tilda_squareds * self._tau**2) + 1
            ) / self._tau

    def trajectory(self, X, T, general=False):
        """
        Calcule la trajectoire de liquidation optimale pour plusieurs actifs.
        
        Parameters
        ----------
        X : np.array
            Vecteur de positions initiales (m × 1)
        T : int
            Horizon de temps (nombre de périodes)
        general : bool
            Si True, utilise la résolution numérique générale.
            Si False, utilise la solution analytique (nécessite H et Γ diagonales).
            
        Returns
        -------
        np.array
            Matrice de trajectoires (m × T+1)
            Chaque ligne correspond à un actif
        """
        trajectories = []
        
        if not general:
            # =================================================================
            # CAS DIAGONAL : Solution analytique via valeurs propres
            # =================================================================
            if not is_diagonal(self._H_tilda):
                raise ValueError("Mode non-général nécessite H̃ diagonale")
            
            # Transformation initiale : z₀ = Uᵀ × H̃^(1/2) × X
            z0 = self._U.T @ self._H_tilda_sqrt @ X
            
            for t in range(T + 1):
                # Pour chaque temps t, calcul de z via sinh
                # zⱼₜ = sinh(κⱼ(T-t)) / sinh(κⱼT) × zⱼ₀
                z = np.sinh(self._kappas * (T - t)) / np.sinh(self._kappas * T) * z0
                
                # Transformation inverse : x = (H̃^(1/2))⁻¹ × U × z
                x = np.floor(self._H_tilda_sqrt_inv @ self._U @ z)
                trajectories.append(x)
                
        else:
            # =================================================================
            # CAS GÉNÉRAL : Résolution numérique du système linéaire
            # =================================================================
            if self._dims != 2:
                raise ValueError("Le mode général n'est implémenté que pour 2 actifs")
            
            # Transformation initiale
            y0 = self._H_tilda_sqrt @ X
            
            # Construction du système linéaire Ax = b
            # Le système a 2(T+1) inconnues : y₁₀, y₂₀, y₁₁, y₂₁, ..., y₁ₜ, y₂ₜ
            
            n_vars = 2 * (T + 1)
            rhs = np.zeros(n_vars)  # Second membre
            
            # --- Conditions initiales ---
            # y₁₀ = y0[0] et y₂₀ = y0[1]
            rhs[0] = y0[0]
            rhs[1] = y0[1]
            
            init1 = np.zeros(n_vars)
            init1[0] = 1  # Coefficient pour y₁₀
            
            init2 = np.zeros(n_vars)
            init2[1] = 1  # Coefficient pour y₂₀
            
            system = [init1, init2]
            
            # --- Équations de récurrence pour k = 0, ..., T-2 ---
            # (yₖ₋₁ - 2yₖ + yₖ₊₁) / τ² = λAyₖ + B(yₖ₋₁ - yₖ₊₁) / (2τ)
            
            for k in range(0, T - 1):
                a = 1 / self._tau**2
                b = 1 / (2 * self._tau)
                c = -2 / self._tau**2
                l = self._lambda
                A = self._A
                B = self._B
                
                # Équation 1 (pour composante 1)
                eq1_coeff = [
                    a - b * B[0, 0],      # yₖ₋₁ composante 1
                    -b * B[0, 1],          # yₖ₋₁ composante 2
                    c - l * A[0, 0],       # yₖ composante 1
                    -l * A[0, 1],          # yₖ composante 2
                    a + b * B[0, 0],       # yₖ₊₁ composante 1
                    b * B[0, 1]            # yₖ₊₁ composante 2
                ]
                
                # Équation 2 (pour composante 2)
                eq2_coeff = [
                    -b * B[1, 0],          # yₖ₋₁ composante 1
                    a - b * B[1, 1],       # yₖ₋₁ composante 2
                    -l * A[1, 0],          # yₖ composante 1
                    c - l * A[1, 1],       # yₖ composante 2
                    b * B[1, 0],           # yₖ₊₁ composante 1
                    a + b * B[1, 1]        # yₖ₊₁ composante 2
                ]
                
                # Placement des coefficients dans le système
                row1 = np.array([0] * (2 * k) + eq1_coeff + [0] * (2 * (T - k - 2)))
                row2 = np.array([0] * (2 * k) + eq2_coeff + [0] * (2 * (T - k - 2)))
                
                system.append(row1)
                system.append(row2)
            
            # --- Conditions finales ---
            # y₁ₙ = 0 et y₂ₙ = 0
            final1 = np.zeros(n_vars)
            final1[-2] = 1  # Coefficient pour y₁ₙ
            
            final2 = np.zeros(n_vars)
            final2[-1] = 1  # Coefficient pour y₂ₙ
            
            system.append(final1)
            system.append(final2)
            
            # --- Résolution du système ---
            solution = np.linalg.solve(np.array(system), rhs)
            y = solution.reshape(T + 1, 2)
            
            # --- Transformation inverse ---
            for yk in y:
                x = self._H_tilda_sqrt_inv @ yk
                trajectories.append(x)
        
        return np.array(trajectories).T

    def strategy(self, X, T, general=False):
        """
        Calcule la liste des trades optimaux pour plusieurs actifs.
        
        Returns
        -------
        np.array
            Matrice des trades (m × T)
        """
        return -np.diff(self.trajectory(X, T, general), axis=1)


# =============================================================================
# EXEMPLE D'UTILISATION : Liquidation simultanée de GOOG et META
# =============================================================================

def example_multi_assets():
    """
    Exemple : liquidation de positions en GOOG et META (anciennement FB).
    """
    
    print("=" * 60)
    print("MODÈLE ALMGREN-CHRISS - MULTI-ACTIFS")
    print("=" * 60)
    
    # --- 1. Téléchargement des données ---
    
    goog = yf.Ticker('GOOG')
    data_goog = goog.history(period='3mo')
    
    meta = yf.Ticker('META')  # Anciennement FB
    data_meta = meta.history(period='3mo')
    
    # Alignement des données (mêmes dates)
    common_dates = data_goog.index.intersection(data_meta.index)
    data_goog = data_goog.loc[common_dates]
    data_meta = data_meta.loc[common_dates]
    
    print(f"\nDonnées téléchargées : {len(common_dates)} jours communs")
    
    # --- 2. Calibration des paramètres ---
    
    avg_volume_goog = np.mean(data_goog['Volume'])
    avg_volume_meta = np.mean(data_meta['Volume'])
    avg_spread_goog = np.mean(data_goog['High'] - data_goog['Low'])
    avg_spread_meta = np.mean(data_meta['High'] - data_meta['Low'])
    
    # Matrice de covariance
    C = np.cov(data_goog['Close'], data_meta['Close'])
    
    # Coût fixe moyen
    epsilon = (avg_spread_goog + avg_spread_meta) / 2
    
    # Matrices d'impact (diagonales pour cet exemple)
    H = np.array([
        [avg_spread_goog / (0.01 * avg_volume_goog), 0],
        [0, avg_spread_meta / (0.01 * avg_volume_meta)]
    ])
    
    Gamma = np.array([
        [avg_spread_goog / (0.1 * avg_volume_goog), 0],
        [0, avg_spread_meta / (0.1 * avg_volume_meta)]
    ])
    
    params = {
        'lambda': 1e-8,
        'C': C,
        'epsilon': epsilon,
        'H': H,
        'Gamma': Gamma,
        'tau': 1
    }
    
    print("\n--- Paramètres calibrés ---")
    print(f"Matrice de covariance C :\n{C}")
    print(f"\nCorrélation GOOG-META : {C[0,1] / np.sqrt(C[0,0] * C[1,1]):.3f}")
    
    # --- 3. Calcul des trajectoires ---
    
    X = np.array([25_000, 25_000])  # 25 000 actions de chaque
    T = len(common_dates)
    
    model = AlmgrenChriss(params)
    trajectory = model.trajectory(X, T, general=False)
    
    print(f"\n--- Résultats ---")
    print(f"Position initiale : GOOG={X[0]:,}, META={X[1]:,}")
    print(f"Horizon : {T} jours")
    
    # --- 4. Visualisation ---
    
    plt.figure(figsize=(12, 7))
    plt.plot(range(T + 1), trajectory[0], 'o-', ms=4, label='GOOG', color='blue')
    plt.plot(range(T + 1), trajectory[1], 'o-', ms=4, label='META', color='orange')
    plt.title(f'Trajectoire de Liquidation Optimale Multi-Actifs', fontsize=14)
    plt.xlabel('Temps (jours)', fontsize=12)
    plt.ylabel('Nombre d\'actions détenues', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig('almgren_chriss_multi_assets.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, trajectory


if __name__ == "__main__":
    example_multi_assets()
```

---

## 5. Calibration des Paramètres {#5-calibration-des-paramètres}

### 5.1 Sources de Données

| Paramètre | Source | Méthode |
|-----------|--------|---------|
| **σ (volatilité)** | Prix historiques | Écart-type des rendements |
| **ε (spread)** | Données Level 2 / Estimé | (Ask - Bid) / 2 |
| **η (impact temp.)** | Données de marché | Régression sur trades passés |
| **γ (impact perm.)** | Données de marché | Analyse de l'impact résiduel |

### 5.2 Formules de Calibration Empiriques

Les formules utilisées dans l'implémentation sont des approximations empiriques courantes :

```python
# Impact temporaire η
eta = average_spread / (0.01 * average_daily_volume)

# Impact permanent γ  
gamma = average_spread / (0.1 * average_daily_volume)
```

**Interprétation** :
- L'impact temporaire est environ **10 fois plus important** que l'impact permanent
- Les deux sont inversement proportionnels au volume (marché plus liquide = moins d'impact)

### 5.3 Calibration de la Matrice de Covariance

```python
import numpy as np

def calibrate_covariance(returns_df):
    """
    Calibre la matrice de covariance à partir des rendements.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame des rendements journaliers (colonnes = actifs)
        
    Returns
    -------
    np.array
        Matrice de covariance
    """
    return np.cov(returns_df.T)


def calibrate_covariance_robust(returns_df, method='ledoit_wolf'):
    """
    Calibration robuste de la matrice de covariance.
    
    Utilise des estimateurs shrinkage (rétrécissement) pour réduire
    l'erreur d'estimation, surtout quand le nombre d'actifs est grand.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame des rendements
    method : str
        'ledoit_wolf' : Estimateur de Ledoit-Wolf
        'oas' : Oracle Approximating Shrinkage (OAS)
        
    Returns
    -------
    np.array
        Matrice de covariance estimée
    """
    from sklearn.covariance import LedoitWolf, OAS
    
    if method == 'ledoit_wolf':
        estimator = LedoitWolf()
    elif method == 'oas':
        estimator = OAS()
    else:
        raise ValueError(f"Méthode inconnue : {method}")
    
    estimator.fit(returns_df)
    return estimator.covariance_
```

### 5.4 Choix du Paramètre λ (Aversion au Risque)

Le choix de **λ** est crucial et dépend du profil de risque du trader :

| Profil | Valeur de λ | Comportement |
|--------|-------------|--------------|
| **Très tolérant** | 10⁻¹¹ | Vente très lente, presque linéaire |
| **Tolérant** | 10⁻⁹ | Vente modérément progressive |
| **Neutre** | 10⁻⁸ | Équilibre risque/coût |
| **Averse** | 10⁻⁷ | Vente rapide en début de période |
| **Très averse** | 10⁻⁴ | Vente très agressive dès le début |

**Méthode pratique pour choisir λ** :

```python
def find_optimal_lambda(X, T, params, target_time_fraction=0.5):
    """
    Trouve le λ tel qu'une fraction donnée de la position soit
    liquidée à la moitié du temps.
    
    Parameters
    ----------
    X : int
        Position initiale
    T : int
        Horizon de temps
    params : dict
        Paramètres du modèle (sans lambda)
    target_time_fraction : float
        Fraction de temps après laquelle mesurer
        
    Returns
    -------
    float
        Valeur optimale de λ
    """
    from scipy.optimize import minimize_scalar
    
    target_position = X * 0.5  # 50% de la position
    target_time = int(T * target_time_fraction)
    
    def objective(log_lambda):
        params_copy = params.copy()
        params_copy['lambda'] = 10**log_lambda
        model = AlmgrenChriss1D(params_copy)
        traj = model.trajectory(X, T)
        return (traj[target_time] - target_position)**2
    
    result = minimize_scalar(objective, bounds=(-12, -4), method='bounded')
    return 10**result.x
```

---

## 6. Implémentation Python Complète {#6-implémentation-python}

### 6.1 Code Complet avec Documentation

```python
"""
=============================================================================
ALMGREN-CHRISS OPTIMAL EXECUTION MODEL
=============================================================================

Implémentation complète du modèle d'exécution optimale Almgren-Chriss.

Référence :
-----------
Almgren, R. and Chriss, N., 2001. "Optimal execution of portfolio transactions."
The Journal of Risk, 3(2), pp.5-39.

Auteur original du notebook : Joshua Paul Jacob
Adaptation et documentation : HelixOne

Usage :
-------
    # Cas single asset (actif unique)
    model_1d = AlmgrenChriss1D(params)
    trajectory = model_1d.trajectory(X=100000, T=30)
    
    # Cas multi-assets (plusieurs actifs)
    model = AlmgrenChriss(params)
    trajectory = model.trajectory(X=np.array([50000, 50000]), T=30)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from typing import Dict, Tuple, Optional, Union
import warnings


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def decompose(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Décompose une matrice en parties symétrique et anti-symétrique.
    
    Théorème : Toute matrice A peut s'écrire A = Aˢ + Aᴬ
    où Aˢ = (A + Aᵀ)/2 est symétrique et Aᴬ = (A - Aᵀ)/2 est anti-symétrique.
    
    Parameters
    ----------
    A : np.ndarray
        Matrice carrée à décomposer
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (Aˢ, Aᴬ) : parties symétrique et anti-symétrique
        
    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> A_s, A_a = decompose(A)
    >>> np.allclose(A, A_s + A_a)
    True
    """
    A_symmetric = 0.5 * (A + A.T)
    A_antisymmetric = 0.5 * (A - A.T)
    return (A_symmetric, A_antisymmetric)


def is_diagonal(A: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Vérifie si une matrice est (essentiellement) diagonale.
    
    Parameters
    ----------
    A : np.ndarray
        Matrice à vérifier
    tol : float
        Tolérance pour les éléments non-diagonaux
        
    Returns
    -------
    bool
        True si tous les éléments hors diagonale sont < tol
    """
    return np.all(np.abs(A - np.diag(np.diagonal(A))) < tol)


def is_positive_definite(A: np.ndarray) -> bool:
    """
    Vérifie si une matrice est définie positive.
    
    Une matrice est définie positive si toutes ses valeurs propres sont > 0.
    
    Parameters
    ----------
    A : np.ndarray
        Matrice symétrique à vérifier
        
    Returns
    -------
    bool
        True si la matrice est définie positive
    """
    eigenvalues = np.linalg.eigvalsh(A)  # eigvalsh pour matrices symétriques
    return np.all(eigenvalues > 0)


# =============================================================================
# MODÈLE SINGLE ASSET
# =============================================================================

class AlmgrenChriss1D:
    """
    Modèle Almgren-Chriss pour l'exécution optimale d'un actif unique.
    
    Ce modèle trouve la trajectoire de trading qui minimise :
        E[Coût] + λ × Var[Coût]
    
    où le coût inclut :
    - L'impact permanent : changement de prix durable après chaque trade
    - L'impact temporaire : pression sur le prix pendant le trade
    - Les coûts fixes : spread bid-ask et commissions
    
    Attributes
    ----------
    _lambda : float
        Paramètre d'aversion au risque
    _sigma : float
        Volatilité de l'actif
    _epsilon : float
        Coût fixe par action (demi-spread)
    _eta : float
        Paramètre d'impact temporaire
    _gamma : float
        Paramètre d'impact permanent
    _tau : float
        Durée d'une période de trading
    _kappa : float
        Paramètre dérivé utilisé dans les formules
    """
    
    def __init__(self, params: Dict):
        """
        Initialise le modèle avec les paramètres de marché.
        
        Parameters
        ----------
        params : Dict
            Dictionnaire contenant les clés :
            - 'lambda' : Aversion au risque (float, typiquement 10⁻⁸)
            - 'sigma' : Volatilité (float, en $/jour ou %)
            - 'epsilon' : Demi-spread (float, en $)
            - 'eta' : Impact temporaire (float)
            - 'gamma' : Impact permanent (float)
            - 'tau' : Durée période (float, typiquement 1 jour)
            
        Raises
        ------
        AssertionError
            Si η̃ ≤ 0 (le problème n'est plus convexe)
        """
        # Extraction des paramètres
        self._lambda = params['lambda']
        self._sigma = params['sigma']
        self._epsilon = params['epsilon']
        self._eta = params['eta']
        self._gamma = params['gamma']
        self._tau = params['tau']
        
        # Calcul de l'impact temporaire ajusté : η̃ = η - γτ/2
        self._eta_tilda = self._eta - 0.5 * self._gamma * self._tau
        
        # Vérification de convexité
        assert self._eta_tilda > 0, (
            f"η̃ = {self._eta_tilda:.6e} doit être > 0 pour garantir "
            "une solution unique. Augmentez η ou diminuez γ."
        )
        
        # Calcul de κ̃² = λσ²/η̃
        self._kappa_tilda_squared = (
            (self._lambda * self._sigma**2) / self._eta_tilda
        )
        
        # Calcul de κ via arccosh
        # κ = (1/τ) × arccosh(κ̃²τ²/2 + 1)
        self._kappa = (
            np.arccosh(0.5 * self._kappa_tilda_squared * self._tau**2 + 1) 
            / self._tau
        )
        
    def trajectory(self, X: int, T: int) -> np.ndarray:
        """
        Calcule la trajectoire de liquidation optimale.
        
        La trajectoire optimale est donnée par :
            xₖ = sinh(κ(T-tₖ)) / sinh(κT) × X
        
        Cette formule produit une courbe convexe décroissante :
        - Vente agressive au début (quand l'exposition au risque est élevée)
        - Vente plus douce vers la fin
        
        Parameters
        ----------
        X : int
            Position initiale (nombre d'actions à liquider)
        T : int
            Horizon de temps (nombre de périodes)
            
        Returns
        -------
        np.ndarray
            Vecteur de taille T+1 : [x₀, x₁, ..., xₜ]
            où xₖ = nombre d'actions détenues à la période k
        """
        trajectory = []
        
        for t in range(T):
            # Formule de la trajectoire optimale
            x = int(
                np.sinh(self._kappa * (T - t)) / 
                np.sinh(self._kappa * T) * X
            )
            trajectory.append(x)
        
        # Position finale = 0
        trajectory.append(0)
        
        return np.array(trajectory)
    
    def strategy(self, X: int, T: int) -> np.ndarray:
        """
        Calcule la liste des trades (nombre d'actions à vendre par période).
        
        Parameters
        ----------
        X : int
            Position initiale
        T : int
            Horizon de temps
            
        Returns
        -------
        np.ndarray
            Vecteur de taille T : [n₁, n₂, ..., nₜ]
            où nₖ = xₖ₋₁ - xₖ (actions vendues à la période k)
        """
        return -np.diff(self.trajectory(X, T))
    
    def expected_cost(self, X: int, T: int) -> float:
        """
        Calcule l'espérance du coût total de la stratégie.
        
        Le coût total comprend :
        - Coût permanent : ½γ × X²
        - Coût temporaire : (η/τ) × Σnₖ²
        - Coût fixe : ε × X
        
        Parameters
        ----------
        X : int
            Position initiale
        T : int
            Horizon de temps
            
        Returns
        -------
        float
            Coût espéré total
        """
        n = self.strategy(X, T)
        
        # Coût permanent (approximation pour trajectoires régulières)
        permanent_cost = 0.5 * self._gamma * X**2
        
        # Coût temporaire
        temporary_cost = (self._eta / self._tau) * np.sum(n**2)
        
        # Coût fixe
        fixed_cost = self._epsilon * X
        
        return permanent_cost + temporary_cost + fixed_cost
    
    def variance_cost(self, X: int, T: int) -> float:
        """
        Calcule la variance du coût total.
        
        La variance provient de l'incertitude sur les prix futurs :
            Var = σ²τ × Σxₖ²
        
        Parameters
        ----------
        X : int
            Position initiale
        T : int
            Horizon de temps
            
        Returns
        -------
        float
            Variance du coût total
        """
        traj = self.trajectory(X, T)
        return self._sigma**2 * self._tau * np.sum(traj[:-1]**2)
    
    def efficient_frontier(
        self, 
        X: int, 
        T: int, 
        lambdas: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la frontière efficiente (coût espéré vs variance).
        
        Parameters
        ----------
        X : int
            Position initiale
        T : int
            Horizon de temps
        lambdas : np.ndarray, optional
            Valeurs de λ à tester. Par défaut : logspace(-12, -4, 50)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (expected_costs, variances) pour chaque λ
        """
        if lambdas is None:
            lambdas = np.logspace(-12, -4, 50)
        
        expected_costs = []
        variances = []
        
        params_copy = {
            'sigma': self._sigma,
            'epsilon': self._epsilon,
            'eta': self._eta,
            'gamma': self._gamma,
            'tau': self._tau
        }
        
        for lam in lambdas:
            params_copy['lambda'] = lam
            try:
                model = AlmgrenChriss1D(params_copy)
                expected_costs.append(model.expected_cost(X, T))
                variances.append(model.variance_cost(X, T))
            except AssertionError:
                # η̃ négatif pour ce λ
                continue
        
        return np.array(expected_costs), np.array(variances)
    
    def __repr__(self) -> str:
        return (
            f"AlmgrenChriss1D(\n"
            f"  λ = {self._lambda:.2e}\n"
            f"  σ = {self._sigma:.4f}\n"
            f"  ε = {self._epsilon:.4f}\n"
            f"  η = {self._eta:.2e}\n"
            f"  γ = {self._gamma:.2e}\n"
            f"  τ = {self._tau}\n"
            f"  κ = {self._kappa:.4f}\n"
            f")"
        )


# =============================================================================
# MODÈLE MULTI-ASSETS
# =============================================================================

class AlmgrenChriss:
    """
    Modèle Almgren-Chriss pour l'exécution optimale de plusieurs actifs.
    
    Extension du modèle 1D prenant en compte :
    - La corrélation entre actifs via la matrice de covariance C
    - L'impact croisé via les matrices H et Γ non-diagonales
    
    Deux modes de résolution :
    - Diagonal : Solution analytique quand H et Γ sont diagonales
    - Général : Résolution numérique d'un système linéaire
    
    Attributes
    ----------
    _dims : int
        Nombre d'actifs
    _C : np.ndarray
        Matrice de covariance (m × m)
    _H : np.ndarray
        Matrice d'impact temporaire (m × m)
    _Gamma : np.ndarray
        Matrice d'impact permanent (m × m)
    """
    
    def __init__(self, params: Dict):
        """
        Initialise le modèle multi-actifs.
        
        Parameters
        ----------
        params : Dict
            Dictionnaire contenant :
            - 'lambda' : Aversion au risque (float)
            - 'C' : Matrice de covariance (np.ndarray, m×m)
            - 'epsilon' : Coût fixe moyen (float)
            - 'H' : Matrice d'impact temporaire (np.ndarray, m×m)
            - 'Gamma' : Matrice d'impact permanent (np.ndarray, m×m)
            - 'tau' : Durée période (float)
            
        Raises
        ------
        AssertionError
            Si H̃ n'est pas définie positive
        """
        self._lambda = params['lambda']
        self._C = params['C']
        self._epsilon = params['epsilon']
        self._H = params['H']
        self._Gamma = params['Gamma']
        self._tau = params['tau']
        
        self._dims = self._C.shape[0]
        
        # Décomposition symétrique/anti-symétrique
        self._H_s, self._H_a = decompose(self._H)
        self._Gamma_s, self._Gamma_a = decompose(self._Gamma)
        
        # Calcul de H̃ = Hˢ - (τ/2)Γˢ
        self._H_tilda = self._H_s - 0.5 * self._Gamma_s * self._tau
        
        # Vérification : H̃ doit être définie positive
        assert is_positive_definite(self._H_tilda), (
            "H̃ doit être définie positive. "
            f"Valeurs propres : {np.linalg.eigvalsh(self._H_tilda)}"
        )
        
        # Racine carrée matricielle
        self._H_tilda_sqrt = sqrtm(self._H_tilda)
        self._H_tilda_sqrt_inv = np.linalg.inv(self._H_tilda_sqrt)
        
        # Matrices transformées
        self._A = self._H_tilda_sqrt_inv @ self._C @ self._H_tilda_sqrt_inv
        self._B = self._H_tilda_sqrt_inv @ self._Gamma_a @ self._H_tilda_sqrt_inv
        
        # Solution analytique si diagonal
        if is_diagonal(self._H_tilda):
            self._kappa_tilda_squareds, self._U = np.linalg.eig(
                self._lambda * self._A
            )
            self._kappas = (
                np.arccosh(0.5 * self._kappa_tilda_squareds * self._tau**2 + 1) 
                / self._tau
            )

    def trajectory(
        self, 
        X: np.ndarray, 
        T: int, 
        general: bool = False
    ) -> np.ndarray:
        """
        Calcule la trajectoire de liquidation optimale pour plusieurs actifs.
        
        Parameters
        ----------
        X : np.ndarray
            Vecteur de positions initiales (m,)
        T : int
            Horizon de temps
        general : bool
            True pour la résolution numérique générale
            False pour la solution analytique (nécessite H, Γ diagonales)
            
        Returns
        -------
        np.ndarray
            Matrice (m × T+1) des trajectoires
            Ligne i = trajectoire de l'actif i
        """
        trajectories = []
        
        if not general:
            # CAS DIAGONAL
            if not is_diagonal(self._H_tilda):
                raise ValueError(
                    "Mode non-général nécessite H̃ diagonale. "
                    "Utilisez general=True."
                )
            
            z0 = self._U.T @ self._H_tilda_sqrt @ X
            
            for t in range(T + 1):
                z = (
                    np.sinh(self._kappas * (T - t)) / 
                    np.sinh(self._kappas * T) * z0
                )
                x = np.floor(self._H_tilda_sqrt_inv @ self._U @ z)
                trajectories.append(x)
                
        else:
            # CAS GÉNÉRAL (système linéaire)
            if self._dims != 2:
                raise ValueError(
                    "Le mode général n'est implémenté que pour 2 actifs"
                )
            
            y0 = self._H_tilda_sqrt @ X
            n_vars = 2 * (T + 1)
            rhs = np.zeros(n_vars)
            
            # Conditions initiales
            rhs[0] = y0[0]
            rhs[1] = y0[1]
            
            system = [
                np.eye(n_vars)[0],  # y₁₀ = y0[0]
                np.eye(n_vars)[1]   # y₂₀ = y0[1]
            ]
            
            # Équations de récurrence
            for k in range(0, T - 1):
                a = 1 / self._tau**2
                b = 1 / (2 * self._tau)
                c = -2 / self._tau**2
                l = self._lambda
                A, B = self._A, self._B
                
                eq1_coeff = [
                    a - b*B[0,0], -b*B[0,1],
                    c - l*A[0,0], -l*A[0,1],
                    a + b*B[0,0], b*B[0,1]
                ]
                eq2_coeff = [
                    -b*B[1,0], a - b*B[1,1],
                    -l*A[1,0], c - l*A[1,1],
                    b*B[1,0], a + b*B[1,1]
                ]
                
                row1 = [0]*(2*k) + eq1_coeff + [0]*(2*(T-k-2))
                row2 = [0]*(2*k) + eq2_coeff + [0]*(2*(T-k-2))
                
                system.append(np.array(row1))
                system.append(np.array(row2))
            
            # Conditions finales
            final1, final2 = np.zeros(n_vars), np.zeros(n_vars)
            final1[-2], final2[-1] = 1, 1
            system.extend([final1, final2])
            
            # Résolution
            solution = np.linalg.solve(np.array(system), rhs)
            y = solution.reshape(T + 1, 2)
            
            for yk in y:
                trajectories.append(self._H_tilda_sqrt_inv @ yk)
        
        return np.array(trajectories).T

    def strategy(
        self, 
        X: np.ndarray, 
        T: int, 
        general: bool = False
    ) -> np.ndarray:
        """
        Calcule la liste des trades optimaux.
        
        Returns
        -------
        np.ndarray
            Matrice (m × T) des trades
        """
        return -np.diff(self.trajectory(X, T, general), axis=1)
    
    def __repr__(self) -> str:
        return (
            f"AlmgrenChriss(dims={self._dims}, λ={self._lambda:.2e})"
        )


# =============================================================================
# FONCTIONS DE CALIBRATION
# =============================================================================

def calibrate_from_ticker(
    ticker: str,
    period: str = '3mo',
    lambda_val: float = 1e-8
) -> Dict:
    """
    Calibre les paramètres du modèle à partir d'un ticker Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Symbole du ticker (ex: 'AAPL', 'GOOG')
    period : str
        Période de données (ex: '1mo', '3mo', '1y')
    lambda_val : float
        Valeur de l'aversion au risque
        
    Returns
    -------
    Dict
        Paramètres calibrés pour AlmgrenChriss1D
    """
    data = yf.Ticker(ticker).history(period=period)
    
    if len(data) < 10:
        raise ValueError(f"Pas assez de données pour {ticker}")
    
    avg_volume = np.mean(data['Volume'])
    avg_spread = np.mean(data['High'] - data['Low'])
    sigma = np.std(data['Close'])
    
    return {
        'lambda': lambda_val,
        'sigma': sigma,
        'epsilon': avg_spread / 2,
        'eta': avg_spread / (0.01 * avg_volume),
        'gamma': avg_spread / (0.1 * avg_volume),
        'tau': 1,
        'data': data  # Inclure les données pour référence
    }


def calibrate_multi_assets(
    tickers: list,
    period: str = '3mo',
    lambda_val: float = 1e-8
) -> Dict:
    """
    Calibre les paramètres pour plusieurs actifs.
    
    Parameters
    ----------
    tickers : list
        Liste de symboles (ex: ['GOOG', 'META'])
    period : str
        Période de données
    lambda_val : float
        Aversion au risque
        
    Returns
    -------
    Dict
        Paramètres calibrés pour AlmgrenChriss
    """
    data_list = []
    for t in tickers:
        data = yf.Ticker(t).history(period=period)
        data_list.append(data)
    
    # Alignement sur dates communes
    common_idx = data_list[0].index
    for d in data_list[1:]:
        common_idx = common_idx.intersection(d.index)
    
    data_aligned = [d.loc[common_idx] for d in data_list]
    
    n = len(tickers)
    closes = np.array([d['Close'].values for d in data_aligned])
    
    # Matrice de covariance
    C = np.cov(closes)
    
    # Matrices diagonales d'impact
    H = np.zeros((n, n))
    Gamma = np.zeros((n, n))
    
    for i, d in enumerate(data_aligned):
        avg_vol = np.mean(d['Volume'])
        avg_spread = np.mean(d['High'] - d['Low'])
        H[i, i] = avg_spread / (0.01 * avg_vol)
        Gamma[i, i] = avg_spread / (0.1 * avg_vol)
    
    avg_spreads = [np.mean(d['High'] - d['Low']) for d in data_aligned]
    
    return {
        'lambda': lambda_val,
        'C': C,
        'epsilon': np.mean(avg_spreads) / 2,
        'H': H,
        'Gamma': Gamma,
        'tau': 1,
        'data': data_aligned
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_trajectory(
    model: Union[AlmgrenChriss1D, AlmgrenChriss],
    X: Union[int, np.ndarray],
    T: int,
    title: Optional[str] = None,
    labels: Optional[list] = None
):
    """
    Visualise la trajectoire de liquidation optimale.
    
    Parameters
    ----------
    model : AlmgrenChriss1D ou AlmgrenChriss
        Modèle calibré
    X : int ou np.ndarray
        Position(s) initiale(s)
    T : int
        Horizon
    title : str, optional
        Titre du graphique
    labels : list, optional
        Labels des actifs (pour multi-assets)
    """
    if isinstance(model, AlmgrenChriss1D):
        trajectory = model.trajectory(X, T)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Trajectoire
        axes[0].plot(range(T+1), trajectory, 'o-', ms=4, lw=1.5)
        axes[0].fill_between(range(T+1), trajectory, alpha=0.3)
        axes[0].set_ylabel('Actions détenues', fontsize=12)
        axes[0].set_title(
            title or f'Trajectoire Optimale ({X:,} actions en {T} jours)',
            fontsize=14
        )
        axes[0].grid(True, alpha=0.3)
        
        # Stratégie
        strategy = model.strategy(X, T)
        axes[1].bar(range(1, T+1), strategy, alpha=0.7)
        axes[1].set_xlabel('Jour', fontsize=12)
        axes[1].set_ylabel('Actions vendues', fontsize=12)
        axes[1].set_title('Actions Vendues par Jour', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        
    else:
        trajectory = model.trajectory(X, T)
        
        plt.figure(figsize=(12, 7))
        
        for i in range(trajectory.shape[0]):
            label = labels[i] if labels else f'Actif {i+1}'
            plt.plot(range(T+1), trajectory[i], 'o-', ms=4, label=label)
        
        plt.xlabel('Temps (jours)', fontsize=12)
        plt.ylabel('Actions détenues', fontsize=12)
        plt.title(title or 'Trajectoire Multi-Actifs Optimale', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_sensitivity(
    params: Dict,
    X: int,
    T: int,
    param_name: str,
    param_values: list,
    model_class=AlmgrenChriss1D
):
    """
    Analyse de sensibilité : effet d'un paramètre sur la trajectoire.
    
    Parameters
    ----------
    params : Dict
        Paramètres de base
    X : int
        Position initiale
    T : int
        Horizon
    param_name : str
        Nom du paramètre à varier ('lambda', 'gamma', 'eta')
    param_values : list
        Valeurs à tester
    """
    plt.figure(figsize=(12, 7))
    
    for val in param_values:
        params_copy = params.copy()
        params_copy[param_name] = val
        
        try:
            model = model_class(params_copy)
            traj = model.trajectory(X, T)
            
            if isinstance(val, float) and val < 0.01:
                label = f'${param_name}$ = {val:.0e}'
            else:
                label = f'${param_name}$ = {val}'
            
            plt.plot(range(T+1), traj, 'o-', ms=4, label=label)
        except (AssertionError, ValueError) as e:
            print(f"Paramètre {param_name}={val} invalide: {e}")
    
    plt.xlabel('Temps (jours)', fontsize=12)
    plt.ylabel('Actions détenues', fontsize=12)
    plt.title(f'Sensibilité au paramètre {param_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

---

## 7. Analyse de Sensibilité {#7-analyse-de-sensibilité}

### 7.1 Effet du Paramètre λ (Aversion au Risque)

```python
def analyze_lambda_sensitivity():
    """
    Analyse l'effet de λ sur la trajectoire optimale.
    """
    params = calibrate_from_ticker('GOOG')
    X, T = 250_000, 63
    
    lambdas = [1e-2, 1e-7, 1e-8, 1e-9, 1e-11]
    
    plt.figure(figsize=(12, 7))
    
    for lam in lambdas:
        params_copy = params.copy()
        params_copy['lambda'] = lam
        
        try:
            model = AlmgrenChriss1D(params_copy)
            traj = model.trajectory(X, T)
            plt.plot(range(T+1), traj, 'o-', ms=4, label=f'λ = {lam:.0e}')
        except AssertionError:
            continue
    
    plt.title('Effet de l\'Aversion au Risque (λ)', fontsize=14)
    plt.xlabel('Temps (jours)')
    plt.ylabel('Actions détenues')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

**Observations** :

| Valeur de λ | Comportement |
|-------------|--------------|
| **λ → 0** (10⁻¹¹) | Trajectoire quasi-linéaire ("equal-packet") : on vend la même quantité chaque jour |
| **λ moyen** (10⁻⁸) | Compromis : vente plus rapide au début, ralentit vers la fin |
| **λ → ∞** (10⁻²) | Vente très agressive : on liquide presque tout dès le premier jour |

### 7.2 Effet du Paramètre γ (Impact Permanent)

```python
def analyze_gamma_sensitivity():
    """
    Analyse l'effet de γ (impact permanent) sur la trajectoire.
    """
    params = calibrate_from_ticker('GOOG')
    X, T = 250_000, 63
    
    gammas = [0.0001, 0.002, 0.005, 0.008]
    
    plt.figure(figsize=(12, 7))
    
    for gamma in gammas:
        params_copy = params.copy()
        params_copy['gamma'] = gamma
        
        try:
            model = AlmgrenChriss1D(params_copy)
            traj = model.trajectory(X, T)
            plt.plot(range(T+1), traj, 'o-', ms=4, label=f'γ = {gamma}')
        except AssertionError:
            continue
    
    plt.title('Effet de l\'Impact Permanent (γ)', fontsize=14)
    plt.xlabel('Temps (jours)')
    plt.ylabel('Actions détenues')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

**Observations** :

- **γ élevé** → Vente plus rapide (mieux vaut vendre avant que le prix ne baisse trop)
- **γ faible** → Vente plus étalée (l'impact permanent est négligeable)

### 7.3 Effet du Paramètre η (Impact Temporaire)

```python
def analyze_eta_sensitivity():
    """
    Analyse l'effet de η (impact temporaire) sur la trajectoire.
    """
    params = calibrate_from_ticker('GOOG')
    X, T = 250_000, 63
    
    etas = [0.001, 0.01, 0.1, 1]
    
    plt.figure(figsize=(12, 7))
    
    for eta in etas:
        params_copy = params.copy()
        params_copy['eta'] = eta
        
        try:
            model = AlmgrenChriss1D(params_copy)
            traj = model.trajectory(X, T)
            plt.plot(range(T+1), traj, 'o-', ms=4, label=f'η = {eta}')
        except AssertionError:
            continue
    
    plt.title('Effet de l\'Impact Temporaire (η)', fontsize=14)
    plt.xlabel('Temps (jours)')
    plt.ylabel('Actions détenues')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

**Observations** :

- **η élevé** → Trajectoire linéaire (vendre peu chaque jour pour éviter l'impact temporaire)
- **η faible** → Vente plus concentrée (l'impact temporaire est négligeable)

### 7.4 Effet de la Corrélation (Multi-Assets)

Dans le cas multi-actifs, la corrélation entre actifs a un effet surprenant :

```python
def analyze_correlation_effect():
    """
    Analyse l'effet de la corrélation sur les trajectoires multi-actifs.
    
    Observation clé : Une forte corrélation peut mener à des stratégies
    où on prend temporairement une position SHORT sur un actif !
    """
    params = calibrate_multi_assets(['GOOG', 'META'])
    X = np.array([25_000, 25_000])
    T = 63
    
    correlations = [0, 100, 1000]  # Valeurs de covariance
    
    fig, axes = plt.subplots(1, len(correlations), figsize=(16, 5))
    
    for i, cov in enumerate(correlations):
        params_copy = params.copy()
        C = params['C'].copy()
        C[0, 1] = cov
        C[1, 0] = cov
        params_copy['C'] = C
        
        model = AlmgrenChriss(params_copy)
        traj = model.trajectory(X, T, general=True)
        
        axes[i].plot(range(T+1), traj[0], 'o-', ms=3, label='GOOG')
        axes[i].plot(range(T+1), traj[1], 'o-', ms=3, label='META')
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i].set_title(f'Covariance = {cov}')
        axes[i].set_xlabel('Jours')
        axes[i].set_ylabel('Actions')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Effet de la Corrélation sur les Trajectoires', fontsize=14)
    plt.tight_layout()
    plt.show()
```

**Observation clé** : Quand la corrélation est forte, la stratégie optimale peut impliquer de **shorter** (vendre à découvert) un actif temporairement, puis de racheter. C'est une forme d'arbitrage qui exploite la corrélation.

---

## 8. Application Avancée : Cross-Market Trading {#8-cross-market-trading}

### 8.1 Contexte

Le modèle Almgren-Chriss peut être appliqué au **trading cross-market** (multi-marchés), où le même actif est coté sur plusieurs bourses.

**Exemple** : Carnival Corporation (CCL) est coté sur :
- **NYSE** (New York Stock Exchange) : ticker CCL
- **LSE** (London Stock Exchange) : ticker CCL.L

### 8.2 Impact Croisé

Quand on vend CCL sur le NYSE, cela affecte aussi le prix sur le LSE (et vice versa). Les matrices H et Γ ne sont plus diagonales :

```
          NYSE    LSE
H = NYSE [ η₁₁    η₁₂ ]  ← Impact temporaire
    LSE  [ η₂₁    η₂₂ ]

          NYSE    LSE
Γ = NYSE [ γ₁₁    γ₁₂ ]  ← Impact permanent
    LSE  [ γ₂₁    γ₂₂ ]
```

### 8.3 Implémentation

```python
def cross_market_example():
    """
    Exemple de liquidation cross-market : CCL sur NYSE et LSE.
    
    Scénario :
    - Un hedge fund détient 100 000 actions CCL sur NYSE
    - Et 15 000 actions CCL sur LSE
    - Il doit liquider en 15 jours
    
    L'impact croisé (vendre sur NYSE affecte le prix sur LSE)
    crée une stratégie de trading optimale non triviale.
    """
    
    # --- Téléchargement des données ---
    
    nyse_ccl = yf.Ticker('CCL')
    data_nyse = nyse_ccl.history(period='15d')
    
    lse_ccl = yf.Ticker('CCL.L')
    data_lse = lse_ccl.history(period='15d')
    
    # Alignement
    common_dates = data_nyse.index.intersection(data_lse.index)
    data_nyse = data_nyse.loc[common_dates]
    data_lse = data_lse.loc[common_dates]
    
    # --- Calibration avec impact croisé ---
    
    avg_vol_nyse = np.mean(data_nyse['Volume'])
    avg_vol_lse = np.mean(data_lse['Volume'])
    avg_spread_nyse = np.mean(data_nyse['High'] - data_nyse['Low'])
    avg_spread_lse = np.mean(data_lse['High'] - data_lse['Low'])
    
    # Matrice de covariance (fortement corrélée car même entreprise)
    C = np.cov(data_nyse['Close'], data_lse['Close'])
    
    # Matrices d'impact NON-DIAGONALES (impact croisé)
    H = np.array([
        [avg_spread_nyse / (0.01 * avg_vol_nyse), 
         avg_spread_nyse / (0.01 * avg_vol_lse)],
        [avg_spread_nyse / (0.01 * avg_vol_lse), 
         avg_spread_lse / (0.01 * avg_vol_lse)]
    ])
    
    # Le "7" représente un impact croisé asymétrique
    # (vendre sur LSE a plus d'impact sur NYSE que l'inverse)
    Gamma = np.array([
        [avg_spread_nyse / (0.1 * avg_vol_nyse), 
         avg_spread_nyse / (0.1 * avg_vol_lse)],
        [7 * avg_spread_lse / (0.1 * avg_vol_nyse), 
         avg_spread_lse / (0.1 * avg_vol_lse)]
    ])
    
    params = {
        'lambda': 1e-8,
        'C': C,
        'epsilon': (avg_spread_nyse + avg_spread_lse) / 2,
        'H': H,
        'Gamma': Gamma,
        'tau': 1
    }
    
    # --- Positions initiales ---
    X = np.array([100_000, 15_000])  # NYSE, LSE
    T = len(common_dates)
    
    # --- Calcul de la trajectoire ---
    model = AlmgrenChriss(params)
    trajectory = model.trajectory(X, T, general=True)
    
    # --- Visualisation ---
    plt.figure(figsize=(12, 7))
    plt.plot(range(T+1), trajectory[0], 'o-', ms=5, label='NYSE:CCL')
    plt.plot(range(T+1), trajectory[1], 'o-', ms=5, label='LSE:CCL')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('Stratégie Cross-Market Optimale (CCL)', fontsize=14)
    plt.xlabel('Temps (jours)', fontsize=12)
    plt.ylabel('Actions détenues', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return trajectory


# Exécution
trajectory = cross_market_example()
```

### 8.4 Interprétation et Avertissement

La stratégie optimale dans ce cas peut impliquer :
1. **Acheter** sur un marché tout en **vendant** sur l'autre
2. Exploiter les différences d'impact croisé

**⚠️ AVERTISSEMENT LÉGAL** : 

Cette stratégie est en réalité une forme de **manipulation de marché cross-market**, qui est **illégale** dans la plupart des juridictions. Elle est présentée ici à titre pédagogique uniquement.

Les régulateurs (SEC aux États-Unis, FCA au Royaume-Uni, AMF en France) surveillent activement ce type de comportement :

> "La manipulation cross-market consiste à passer des ordres sur un instrument financier dans l'intention d'affecter le prix du même instrument ou d'un instrument similaire sur un autre marché."

---

## 9. Extensions et Limites {#9-extensions-et-limites}

### 9.1 Extensions Possibles

| Extension | Description | Difficulté |
|-----------|-------------|------------|
| **Impact non-linéaire** | h(v) = η × v^α au lieu de η × v | Moyenne |
| **Volatilité variable** | σ(t) dépend du temps | Facile |
| **Contraintes de trading** | Min/max volumes par période | Moyenne |
| **Temps continu** | Limite τ → 0 | Mathématiquement complexe |
| **Dark pools** | Inclure des venues alternatives | Élevée |
| **Exécution stochastique** | Incertitude sur l'exécution | Élevée |

### 9.2 Limites du Modèle

1. **Hypothèse de linéarité** : L'impact de marché réel est souvent non-linéaire (convexe ou concave)

2. **Paramètres constants** : En réalité, σ, η, γ varient au cours de la journée et selon les conditions de marché

3. **Pas de coûts d'opportunité** : Le modèle ne considère pas les gains manqués si le prix monte pendant la liquidation

4. **Information parfaite** : On suppose connaître σ, η, γ à l'avance

5. **Marché unique** : Version de base ne gère pas les dark pools, internaliseurs, etc.

### 9.3 Alternatives et Compléments

| Modèle | Caractéristique | Usage |
|--------|-----------------|-------|
| **Bertsimas-Lo** | Modèle de prix discret | Plus réaliste pour petites positions |
| **Obizhaeva-Wang** | Impact transitoire | Prend en compte la résilience du marché |
| **Cartea-Jaimungal** | Modèle de contrôle stochastique | Gestion du risque d'inventaire |
| **Guéant-Lehalle** | Market making optimal | Pour les teneurs de marché |

---

## 10. Glossaire {#10-glossaire}

### Termes de Microstructure de Marché

| Terme | Définition |
|-------|------------|
| **Bid** | Prix le plus élevé qu'un acheteur est prêt à payer |
| **Ask (Offer)** | Prix le plus bas qu'un vendeur est prêt à accepter |
| **Spread** | Différence Ask - Bid |
| **Market Impact** | Effet d'un ordre sur le prix du marché |
| **Slippage** | Différence entre prix attendu et prix obtenu |
| **VWAP** | Volume Weighted Average Price (Prix moyen pondéré par le volume) |
| **TWAP** | Time Weighted Average Price (Prix moyen pondéré par le temps) |
| **Dark Pool** | Système de trading privé hors bourse |
| **Liquidity** | Facilité à acheter/vendre sans impact de prix |

### Termes Mathématiques

| Terme | Définition |
|-------|------------|
| **sinh** | Sinus hyperbolique : sinh(x) = (eˣ - e⁻ˣ) / 2 |
| **arccosh** | Fonction réciproque du cosinus hyperbolique |
| **Matrice définie positive** | Matrice dont toutes les valeurs propres sont > 0 |
| **Racine carrée matricielle** | Matrice A^(1/2) telle que A^(1/2) × A^(1/2) = A |
| **Valeurs propres** | Solutions λ de l'équation Ax = λx |

### Paramètres du Modèle

| Symbole | Nom | Interprétation |
|---------|-----|----------------|
| **λ** | Lambda | Aversion au risque (plus élevé = plus prudent) |
| **σ** | Sigma | Volatilité (risque de variation du prix) |
| **ε** | Epsilon | Coût fixe par action (demi-spread) |
| **η** | Eta | Impact temporaire (coût de vitesse) |
| **γ** | Gamma | Impact permanent (dépréciation du prix) |
| **τ** | Tau | Durée d'une période de trading |
| **κ** | Kappa | Paramètre dérivé pour la trajectoire optimale |

---

## Références

1. **Almgren, R. and Chriss, N. (2001)**. "Optimal execution of portfolio transactions." *The Journal of Risk*, 3(2), pp.5-39.

2. **Almgren, R. (2003)**. "Optimal execution with nonlinear impact functions and trading-enhanced risk." *Applied Mathematical Finance*, 10(1), pp.1-18.

3. **Bertsimas, D. and Lo, A.W. (1998)**. "Optimal control of execution costs." *Journal of Financial Markets*, 1(1), pp.1-50.

4. **Huberman, G. and Stanzl, W. (2004)**. "Price Manipulation and Quasi-Arbitrage." *Econometrica*, 72(4), pp.1247-1275.

5. **Cartea, Á. and Jaimungal, S. (2015)**. "Optimal execution with limit and market orders." *Quantitative Finance*, 15(8), pp.1279-1291.

---

*Guide créé pour HelixOne - Documentation technique Finance Quantitative*
*Basé sur l'implémentation de Joshua Paul Jacob*
