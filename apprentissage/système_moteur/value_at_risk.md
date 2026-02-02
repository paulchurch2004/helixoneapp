# Guide Complet : Value at Risk (VaR)

## Analyse du Risque Financier avec Python

*Basé sur le cours MH8331 de N. Privault - NTU Singapore*

---

# Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Glossaire des Acronymes](#2-glossaire-des-acronymes)
3. [Mesures de Risque](#3-mesures-de-risque)
4. [Mesures de Risque Cohérentes](#4-mesures-de-risque-cohérentes)
5. [Quantiles et Fonctions de Répartition](#5-quantiles-et-fonctions-de-répartition)
6. [Value at Risk (VaR)](#6-value-at-risk-var)
7. [VaR Gaussienne](#7-var-gaussienne)
8. [Calculs Numériques et Implémentation](#8-calculs-numériques-et-implémentation)
9. [Code Complet et Classes Utilitaires](#9-code-complet-et-classes-utilitaires)
10. [Exercices Résolus](#10-exercices-résolus)

---

# 1. Introduction et Contexte

## 1.1 Qu'est-ce que la Value at Risk ?

La **VaR (Value at Risk, Valeur à Risque)** est l'une des mesures de risque les plus fondamentales et les plus utilisées en finance. Elle estime la **perte potentielle maximale** sur un investissement donné, sur un horizon temporel spécifié, avec un certain niveau de confiance.

**Exemple concret** :
> "La VaR à 95% sur 1 jour de notre portefeuille est de 1 million d'euros"
> 
> Signifie : Il y a 95% de chances que la perte sur 1 jour ne dépasse pas 1 million d'euros.
> Autrement dit : Il y a 5% de chances de perdre plus de 1 million d'euros.

## 1.2 Objectifs des Mesures de Risque

Les mesures de risque ont deux objectifs principaux :

1. **Quantifier le risque** : Fournir une mesure numérique du risque encouru
2. **Déterminer les réserves de capital** : Calculer le niveau adéquat de capital à maintenir

## 1.3 Modélisation des Pertes

Les pertes potentielles sont modélisées par une **variable aléatoire X**.

**Convention de signe** :
- X > 0 : Perte (on perd de l'argent)
- X < 0 : Gain (on gagne de l'argent)

Le **capital requis** pour faire face au risque est défini par :

```
C_X = V_X - L_X
```

Où :
- **V_X** : Estimation "raisonnable" supérieure de la perte potentielle
- **L_X** : Passifs (liabilities) de l'entreprise

---

# 2. Glossaire des Acronymes

| Acronyme | Anglais | Français |
|----------|---------|----------|
| **VaR** | Value at Risk | Valeur à Risque |
| **CDF** | Cumulative Distribution Function | Fonction de Répartition (FR) |
| **PDF** | Probability Density Function | Fonction de Densité de Probabilité |
| **CTE** | Conditional Tail Expectation | Espérance Conditionnelle de Queue |
| **CVaR** | Conditional Value at Risk | Valeur à Risque Conditionnelle |
| **ES** | Expected Shortfall | Perte Attendue au-delà de la VaR |
| **ECDF** | Empirical CDF | Fonction de Répartition Empirique |
| **i.i.d.** | Independent and Identically Distributed | Indépendants et Identiquement Distribués |
| **ROC** | Receiver Operating Characteristic | Caractéristique de Fonctionnement du Récepteur |
| **KS** | Kolmogorov-Smirnov | Test de Kolmogorov-Smirnov |
| **MLE** | Maximum Likelihood Estimation | Estimation par Maximum de Vraisemblance |
| **SSE** | Sum of Squared Errors | Somme des Erreurs au Carré |

---

# 3. Mesures de Risque

## 3.1 Définition Formelle

**Définition** : Une mesure de risque est une application qui assigne une valeur V_X à une variable aléatoire de perte X.

```python
"""
=============================================================================
MESURES DE RISQUE - IMPLÉMENTATION PYTHON
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Union, Tuple, Callable
import yfinance as yf
from dataclasses import dataclass

# Configuration matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


@dataclass
class RiskMeasureResult:
    """
    Classe pour stocker les résultats d'une mesure de risque.
    
    Attributes:
        name: Nom de la mesure de risque
        value: Valeur calculée
        confidence_level: Niveau de confiance (si applicable)
        description: Description de la mesure
    """
    name: str
    value: float
    confidence_level: float = None
    description: str = ""
    
    def __repr__(self):
        if self.confidence_level:
            return f"{self.name} @ {self.confidence_level:.1%}: {self.value:.6f}"
        return f"{self.name}: {self.value:.6f}"
```

## 3.2 Prime d'Espérance (Expected Value Premium)

La **prime d'espérance** est définie par :

```
E_X = E[X] + α × E[X] = (1 + α) × E[X]
```

Où α ≥ 0 est le **chargement de sécurité** (safety loading).

**Cas particulier** : Pour α = 0, on obtient la **prime pure** (pure premium) : E_X = E[X]

```python
def expected_value_premium(
    X: np.ndarray, 
    alpha: float = 0.0
) -> RiskMeasureResult:
    """
    Calcule la prime d'espérance (Expected Value Premium).
    
    Formule : E_X = (1 + α) × E[X]
    
    Args:
        X: Array des pertes (valeurs positives = pertes)
        alpha: Chargement de sécurité (α ≥ 0)
               - α = 0 : Prime pure (pure premium)
               - α > 0 : Prime chargée pour couvrir le risque
    
    Returns:
        RiskMeasureResult avec la valeur de la prime
    
    Exemple:
        >>> pertes = np.array([100, 200, 150, 300, 50])
        >>> result = expected_value_premium(pertes, alpha=0.1)
        >>> print(result)
        Expected Value Premium: 176.000000
        
        Interprétation : Avec un chargement de 10%, la prime est de 176€
        pour des pertes moyennes de 160€.
    """
    if alpha < 0:
        raise ValueError("Le chargement α doit être ≥ 0")
    
    expected_loss = np.mean(X)
    premium = (1 + alpha) * expected_loss
    
    return RiskMeasureResult(
        name="Expected Value Premium",
        value=premium,
        description=f"E[X]={expected_loss:.4f}, α={alpha}"
    )


# Exemple d'utilisation
print("=== Prime d'Espérance ===")
pertes_exemple = np.array([100, 200, 150, 300, 50, 180, 220, 90])
for alpha in [0.0, 0.1, 0.2]:
    result = expected_value_premium(pertes_exemple, alpha)
    print(f"  α = {alpha}: {result.value:.2f}€")
```

## 3.3 Prime d'Écart-Type (Standard Deviation Premium)

La **prime d'écart-type** est définie par :

```
SD_X = E[X] + α × √Var[X] = E[X] + α × σ_X
```

**Interprétation** : On ajoute α fois l'écart-type à la perte moyenne pour couvrir la variabilité.

```python
def standard_deviation_premium(
    X: np.ndarray, 
    alpha: float = 1.0
) -> RiskMeasureResult:
    """
    Calcule la prime d'écart-type (Standard Deviation Premium).
    
    Formule : SD_X = E[X] + α × σ_X
    
    Où σ_X = √Var[X] est l'écart-type des pertes.
    
    Args:
        X: Array des pertes
        alpha: Coefficient de l'écart-type (α ≥ 0)
               - α = 1 : On ajoute 1 écart-type
               - α = 2 : On ajoute 2 écarts-types (couverture ~95%)
    
    Returns:
        RiskMeasureResult avec la valeur de la prime
    
    Exemple:
        >>> pertes = np.array([100, 200, 150, 300, 50])
        >>> result = standard_deviation_premium(pertes, alpha=2)
        >>> print(result)
        
        Avec α=2, on couvre environ 95% des cas si X est gaussienne.
    """
    if alpha < 0:
        raise ValueError("Le coefficient α doit être ≥ 0")
    
    mean_loss = np.mean(X)
    std_loss = np.std(X, ddof=1)  # ddof=1 pour l'estimateur non-biaisé
    premium = mean_loss + alpha * std_loss
    
    return RiskMeasureResult(
        name="Standard Deviation Premium",
        value=premium,
        description=f"E[X]={mean_loss:.4f}, σ={std_loss:.4f}, α={alpha}"
    )


# Exemple d'utilisation
print("\n=== Prime d'Écart-Type ===")
for alpha in [1.0, 1.645, 2.0, 2.576]:
    result = standard_deviation_premium(pertes_exemple, alpha)
    print(f"  α = {alpha:.3f}: {result.value:.2f}€")
```

## 3.4 CTE (Conditional Tail Expectation, Espérance Conditionnelle de Queue)

La **CTE** est l'espérance des pertes sachant qu'on est en perte (X < 0 pour les rendements).

**Formule** :
```
CTE_X = E[X | X < 0] = E[X × 1_{X<0}] / P(X < 0)
```

**Interprétation** : C'est la perte moyenne quand on perd de l'argent.

```python
def conditional_tail_expectation(
    X: np.ndarray, 
    threshold: float = 0.0
) -> RiskMeasureResult:
    """
    Calcule la CTE (Conditional Tail Expectation, Espérance Conditionnelle de Queue).
    
    Formule : CTE = E[X | X < threshold]
    
    Cette mesure représente la perte moyenne sachant qu'on dépasse
    un certain seuil de perte.
    
    Args:
        X: Array des rendements (négatifs = pertes)
        threshold: Seuil en dessous duquel on calcule la moyenne
                   (par défaut 0 = toutes les pertes)
    
    Returns:
        RiskMeasureResult avec la CTE
    
    Exemple:
        Pour des rendements journaliers :
        >>> returns = np.array([-0.02, 0.01, -0.03, 0.02, -0.01, 0.015])
        >>> cte = conditional_tail_expectation(returns, threshold=0)
        >>> print(f"CTE = {cte.value:.4f}")
        CTE = -0.0200
        
        Signifie : En moyenne, quand on perd, on perd 2%.
    """
    # Filtrer les valeurs sous le seuil
    tail_values = X[X < threshold]
    
    if len(tail_values) == 0:
        return RiskMeasureResult(
            name="Conditional Tail Expectation",
            value=np.nan,
            description=f"Aucune valeur sous le seuil {threshold}"
        )
    
    cte = np.mean(tail_values)
    prob_tail = len(tail_values) / len(X)
    
    # Vérification avec la formule : E[X × 1_{X<threshold}] / P(X < threshold)
    truncated_expectation = np.mean(X * (X < threshold))
    cte_formula = truncated_expectation / prob_tail
    
    return RiskMeasureResult(
        name="Conditional Tail Expectation",
        value=cte,
        description=f"P(X<{threshold})={prob_tail:.4f}, E[X|X<{threshold}]={cte:.6f}"
    )


def plot_cte_visualization(
    returns: np.ndarray, 
    threshold: float = 0.0,
    title: str = "Rendements et CTE"
) -> None:
    """
    Visualise les rendements et la CTE.
    
    Args:
        returns: Array des rendements
        threshold: Seuil pour la CTE
        title: Titre du graphique
    """
    cte_result = conditional_tail_expectation(returns, threshold)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Créer un index temporel
    x = np.arange(len(returns))
    
    # Barres pour les rendements
    colors = ['red' if r < threshold else 'blue' for r in returns]
    ax.bar(x, returns, color=colors, alpha=0.7, width=0.8)
    
    # Ligne horizontale pour la CTE
    ax.axhline(y=cte_result.value, color='darkred', linestyle='--', 
               linewidth=2.5, label=f'CTE = {cte_result.value:.4f}')
    
    # Ligne à zéro
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Temps')
    ax.set_ylabel('Rendement')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{cte_result}")
    print(f"Nombre de pertes: {np.sum(returns < threshold)}/{len(returns)}")
```

---

# 4. Mesures de Risque Cohérentes

## 4.1 Définition

Une mesure de risque V est dite **cohérente** si elle satisfait les quatre propriétés suivantes pour toutes variables aléatoires X et Y :

| Propriété | Formule | Interprétation |
|-----------|---------|----------------|
| **Monotonicité** | X ≤ Y ⇒ V_X ≤ V_Y | Plus la perte est grande, plus le risque est élevé |
| **Homogénéité positive** | V_{λX} = λ × V_X pour λ > 0 | Doubler la position double le risque |
| **Invariance par translation** | V_{μ+X} = μ + V_X pour μ > 0 | Ajouter un montant fixe l'ajoute au risque |
| **Sous-additivité** | V_{X+Y} ≤ V_X + V_Y | La diversification réduit le risque |

```python
class CoherentRiskMeasure:
    """
    Classe de base pour les mesures de risque cohérentes.
    
    Une mesure de risque cohérente satisfait :
    1. Monotonicité : X ≤ Y ⇒ V(X) ≤ V(Y)
    2. Homogénéité positive : V(λX) = λV(X) pour λ > 0
    3. Invariance par translation : V(μ + X) = μ + V(X)
    4. Sous-additivité : V(X + Y) ≤ V(X) + V(Y)
    
    Note importante : La VaR n'est PAS cohérente car elle n'est 
    pas sous-additive en général !
    """
    
    def __init__(self, name: str = "Risk Measure"):
        self.name = name
    
    def compute(self, X: np.ndarray) -> float:
        """Calcule la mesure de risque. À implémenter dans les sous-classes."""
        raise NotImplementedError
    
    def check_monotonicity(
        self, 
        X: np.ndarray, 
        Y: np.ndarray
    ) -> bool:
        """
        Vérifie la monotonicité : X ≤ Y ⇒ V(X) ≤ V(Y)
        
        Exemple:
            Si les pertes de X sont toujours ≤ aux pertes de Y,
            alors le risque de X doit être ≤ au risque de Y.
        """
        if np.all(X <= Y):
            return self.compute(X) <= self.compute(Y)
        return True  # Condition non applicable
    
    def check_positive_homogeneity(
        self, 
        X: np.ndarray, 
        lambda_: float
    ) -> Tuple[bool, float]:
        """
        Vérifie l'homogénéité positive : V(λX) = λV(X)
        
        Exemple:
            Si on double toutes les positions (λ=2),
            le risque doit exactement doubler.
        """
        if lambda_ <= 0:
            raise ValueError("λ doit être > 0")
        
        V_X = self.compute(X)
        V_lambdaX = self.compute(lambda_ * X)
        expected = lambda_ * V_X
        
        is_homogeneous = np.isclose(V_lambdaX, expected, rtol=1e-6)
        error = abs(V_lambdaX - expected)
        
        return is_homogeneous, error
    
    def check_translation_invariance(
        self, 
        X: np.ndarray, 
        mu: float
    ) -> Tuple[bool, float]:
        """
        Vérifie l'invariance par translation : V(μ + X) = μ + V(X)
        
        Exemple:
            Si on ajoute 100€ de perte certaine à tous les scénarios,
            le risque doit augmenter de exactement 100€.
        """
        V_X = self.compute(X)
        V_muX = self.compute(mu + X)
        expected = mu + V_X
        
        is_invariant = np.isclose(V_muX, expected, rtol=1e-6)
        error = abs(V_muX - expected)
        
        return is_invariant, error
    
    def check_subadditivity(
        self, 
        X: np.ndarray, 
        Y: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Vérifie la sous-additivité : V(X + Y) ≤ V(X) + V(Y)
        
        Cette propriété capture l'idée de diversification :
        le risque combiné ne doit pas dépasser la somme des risques individuels.
        
        ATTENTION : La VaR viole cette propriété !
        
        Exemple:
            >>> # Deux positions indépendantes
            >>> X = np.array([0, 0, 0, 100])  # Perte rare
            >>> Y = np.array([0, 0, 0, 100])  # Perte rare
            >>> # VaR_95(X) = VaR_95(Y) = 0
            >>> # Mais VaR_95(X+Y) peut être > 0 !
        """
        V_X = self.compute(X)
        V_Y = self.compute(Y)
        V_XY = self.compute(X + Y)
        
        is_subadditive = V_XY <= V_X + V_Y + 1e-10  # Tolérance numérique
        excess = max(0, V_XY - (V_X + V_Y))
        
        return is_subadditive, excess
    
    def full_coherence_check(
        self, 
        X: np.ndarray, 
        Y: np.ndarray = None,
        lambda_: float = 2.0,
        mu: float = 10.0
    ) -> dict:
        """
        Vérifie toutes les propriétés de cohérence.
        
        Args:
            X: Premier échantillon de pertes
            Y: Deuxième échantillon (généré aléatoirement si None)
            lambda_: Facteur d'échelle pour l'homogénéité
            mu: Translation pour l'invariance
        
        Returns:
            Dictionnaire avec les résultats de chaque test
        """
        if Y is None:
            Y = np.random.randn(len(X)) * np.std(X) + np.mean(X)
        
        results = {
            'measure_name': self.name,
            'V_X': self.compute(X),
            'V_Y': self.compute(Y),
        }
        
        # Test homogénéité
        homo_ok, homo_err = self.check_positive_homogeneity(X, lambda_)
        results['homogeneity'] = {'passed': homo_ok, 'error': homo_err}
        
        # Test translation
        trans_ok, trans_err = self.check_translation_invariance(X, mu)
        results['translation_invariance'] = {'passed': trans_ok, 'error': trans_err}
        
        # Test sous-additivité
        sub_ok, sub_excess = self.check_subadditivity(X, Y)
        results['subadditivity'] = {'passed': sub_ok, 'excess': sub_excess}
        
        # Résumé
        results['is_coherent'] = homo_ok and trans_ok and sub_ok
        
        return results
```

## 4.2 Exemple de Non-Cohérence de la VaR

La VaR n'est **PAS sous-additive** en général. Voici un contre-exemple classique :

```python
def demonstrate_var_non_subadditivity():
    """
    Démontre que la VaR n'est pas sous-additive avec un exemple de Bernoulli.
    
    On considère deux variables de Bernoulli indépendantes :
    - X ∈ {0, 1} avec P(X=1) = 2%, P(X=0) = 98%
    - Y ∈ {0, 1} avec P(Y=1) = 2%, P(Y=0) = 98%
    
    À p = 97.5% :
    - VaR_X = VaR_Y = 0 (car P(X ≤ 0) = 98% > 97.5%)
    
    Pour X + Y ∈ {0, 1, 2} :
    - P(X+Y = 0) = 0.98² = 96.04%
    - P(X+Y = 1) = 2 × 0.02 × 0.98 = 3.92%
    - P(X+Y = 2) = 0.02² = 0.04%
    
    Donc VaR_{X+Y}^{97.5%} = 1 car P(X+Y ≤ 0) = 96.04% < 97.5%
    
    Conclusion : VaR(X+Y) = 1 > VaR(X) + VaR(Y) = 0 + 0 = 0
    La VaR n'est pas sous-additive !
    """
    print("=" * 70)
    print("DÉMONSTRATION : La VaR n'est PAS sous-additive")
    print("=" * 70)
    
    # Paramètres
    p_loss = 0.02  # Probabilité de perte
    confidence = 0.975  # Niveau de confiance
    n_simulations = 1_000_000
    
    # Simulation de X et Y (Bernoulli indépendants)
    np.random.seed(42)
    X = np.random.binomial(1, p_loss, n_simulations)
    Y = np.random.binomial(1, p_loss, n_simulations)
    
    # Calcul des VaR empiriques
    var_X = np.percentile(X, confidence * 100)
    var_Y = np.percentile(Y, confidence * 100)
    var_XY = np.percentile(X + Y, confidence * 100)
    
    print(f"\nParamètres:")
    print(f"  - P(perte) = {p_loss:.1%}")
    print(f"  - Niveau de confiance = {confidence:.1%}")
    print(f"  - Nombre de simulations = {n_simulations:,}")
    
    print(f"\nDistribution de X et Y:")
    print(f"  - P(X=0) = P(Y=0) = {(1-p_loss)**1:.4f}")
    print(f"  - P(X=1) = P(Y=1) = {p_loss:.4f}")
    
    print(f"\nDistribution de X + Y:")
    print(f"  - P(X+Y=0) = {(1-p_loss)**2:.4f}")
    print(f"  - P(X+Y=1) = {2*p_loss*(1-p_loss):.4f}")
    print(f"  - P(X+Y=2) = {p_loss**2:.6f}")
    
    print(f"\nVaR à {confidence:.1%}:")
    print(f"  - VaR(X)   = {var_X:.0f}")
    print(f"  - VaR(Y)   = {var_Y:.0f}")
    print(f"  - VaR(X+Y) = {var_XY:.0f}")
    
    print(f"\nTest de sous-additivité:")
    print(f"  - VaR(X) + VaR(Y) = {var_X + var_Y:.0f}")
    print(f"  - VaR(X+Y)        = {var_XY:.0f}")
    
    if var_XY > var_X + var_Y:
        print(f"\n❌ VIOLATION : VaR(X+Y) > VaR(X) + VaR(Y)")
        print(f"   La VaR n'est PAS une mesure cohérente !")
    else:
        print(f"\n✓ Sous-additivité respectée dans cet exemple")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Distribution de X
    axes[0].bar([0, 1], [np.mean(X == 0), np.mean(X == 1)], color='blue', alpha=0.7)
    axes[0].axvline(x=var_X, color='red', linestyle='--', linewidth=2, label=f'VaR={var_X:.0f}')
    axes[0].set_title('Distribution de X')
    axes[0].set_xlabel('Valeur')
    axes[0].set_ylabel('Probabilité')
    axes[0].legend()
    
    # Distribution de Y
    axes[1].bar([0, 1], [np.mean(Y == 0), np.mean(Y == 1)], color='green', alpha=0.7)
    axes[1].axvline(x=var_Y, color='red', linestyle='--', linewidth=2, label=f'VaR={var_Y:.0f}')
    axes[1].set_title('Distribution de Y')
    axes[1].set_xlabel('Valeur')
    axes[1].legend()
    
    # Distribution de X + Y
    vals, counts = np.unique(X + Y, return_counts=True)
    axes[2].bar(vals, counts/n_simulations, color='purple', alpha=0.7)
    axes[2].axvline(x=var_XY, color='red', linestyle='--', linewidth=2, label=f'VaR={var_XY:.0f}')
    axes[2].set_title('Distribution de X + Y')
    axes[2].set_xlabel('Valeur')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('var_non_subadditivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return var_X, var_Y, var_XY


# Exécution de la démonstration
demonstrate_var_non_subadditivity()
```

---

# 5. Quantiles et Fonctions de Répartition

## 5.1 CDF (Cumulative Distribution Function, Fonction de Répartition)

**Définition** : La CDF d'une variable aléatoire X est la fonction :

```
F_X(x) = P(X ≤ x), x ∈ ℝ
```

**Propriétés** :
1. F_X est **non-décroissante**
2. F_X est **continue à droite**
3. lim_{x→+∞} F_X(x) = 1
4. lim_{x→-∞} F_X(x) = 0

```python
class CumulativeDistributionFunction:
    """
    Classe pour travailler avec les fonctions de répartition (CDF).
    
    La CDF (Cumulative Distribution Function, Fonction de Répartition)
    F_X(x) = P(X ≤ x) est fondamentale pour calculer les quantiles et la VaR.
    """
    
    def __init__(self, data: np.ndarray = None, distribution: str = None, **params):
        """
        Initialise la CDF à partir de données ou d'une distribution théorique.
        
        Args:
            data: Données empiriques (pour CDF empirique)
            distribution: Nom de la distribution ('normal', 'exponential', 'student', etc.)
            **params: Paramètres de la distribution (mu, sigma, df, lambda_, etc.)
        
        Exemple:
            >>> # CDF empirique
            >>> cdf = CumulativeDistributionFunction(data=returns)
            >>> 
            >>> # CDF normale théorique
            >>> cdf = CumulativeDistributionFunction(
            ...     distribution='normal', mu=0, sigma=1
            ... )
        """
        self.data = data
        self.distribution = distribution
        self.params = params
        
        if distribution == 'normal':
            self.mu = params.get('mu', 0)
            self.sigma = params.get('sigma', 1)
            self._cdf = lambda x: stats.norm.cdf(x, loc=self.mu, scale=self.sigma)
            self._ppf = lambda p: stats.norm.ppf(p, loc=self.mu, scale=self.sigma)
            
        elif distribution == 'exponential':
            self.lambda_ = params.get('lambda_', 1)
            self._cdf = lambda x: stats.expon.cdf(x, scale=1/self.lambda_)
            self._ppf = lambda p: stats.expon.ppf(p, scale=1/self.lambda_)
            
        elif distribution == 'student':
            self.df = params.get('df', 5)
            self._cdf = lambda x: stats.t.cdf(x, df=self.df)
            self._ppf = lambda p: stats.t.ppf(p, df=self.df)
            
        elif distribution == 'pareto':
            self.gamma = params.get('gamma', 2)
            self.theta = params.get('theta', 1)
            self._cdf = lambda x: 1 - (self.theta / (self.theta + x))**self.gamma if x >= 0 else 0
            self._ppf = lambda p: self.theta * ((1 - p)**(-1/self.gamma) - 1)
            
        elif data is not None:
            # CDF empirique
            self.sorted_data = np.sort(data)
            self.n = len(data)
            
        else:
            raise ValueError("Fournir soit 'data' soit 'distribution'")
    
    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Évalue F_X(x) = P(X ≤ x).
        
        Args:
            x: Point(s) où évaluer la CDF
            
        Returns:
            Valeur(s) de la CDF
        """
        if self.distribution:
            return self._cdf(x)
        else:
            # CDF empirique : proportion d'observations ≤ x
            if np.isscalar(x):
                return np.mean(self.data <= x)
            return np.array([np.mean(self.data <= xi) for xi in x])
    
    def quantile(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calcule le p-quantile (inverse généralisée de la CDF).
        
        Le p-quantile q_X^p est défini comme :
        q_X^p = inf{x ∈ ℝ : F_X(x) ≥ p}
        
        Args:
            p: Niveau(x) de probabilité dans (0, 1)
            
        Returns:
            Quantile(s) correspondant(s)
            
        Exemple:
            >>> cdf = CumulativeDistributionFunction(distribution='normal', mu=0, sigma=1)
            >>> q_95 = cdf.quantile(0.95)
            >>> print(f"Quantile 95% de N(0,1): {q_95:.4f}")
            Quantile 95% de N(0,1): 1.6449
        """
        if self.distribution:
            return self._ppf(p)
        else:
            # Quantile empirique
            if np.isscalar(p):
                idx = int(np.ceil(p * self.n)) - 1
                idx = max(0, min(idx, self.n - 1))
                return self.sorted_data[idx]
            return np.array([self.quantile(pi) for pi in p])
    
    def plot(
        self, 
        x_range: Tuple[float, float] = None,
        n_points: int = 1000,
        show_quantiles: list = None,
        title: str = None
    ) -> None:
        """
        Trace le graphique de la CDF.
        
        Args:
            x_range: Intervalle (x_min, x_max) pour le tracé
            n_points: Nombre de points
            show_quantiles: Liste de quantiles à afficher (ex: [0.05, 0.5, 0.95])
            title: Titre du graphique
        """
        if x_range is None:
            if self.data is not None:
                margin = 0.1 * (self.data.max() - self.data.min())
                x_range = (self.data.min() - margin, self.data.max() + margin)
            else:
                x_range = (-4, 4)
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = self.evaluate(x)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.data is not None and self.distribution is None:
            # CDF empirique en escalier
            ax.step(x, y, where='post', color='blue', linewidth=2, label='CDF empirique')
        else:
            ax.plot(x, y, color='blue', linewidth=2, label=f'CDF {self.distribution}')
        
        # Afficher les quantiles demandés
        if show_quantiles:
            for p in show_quantiles:
                q = self.quantile(p)
                ax.axhline(y=p, color='gray', linestyle=':', alpha=0.5)
                ax.axvline(x=q, color='red', linestyle='--', alpha=0.7)
                ax.plot(q, p, 'ro', markersize=8)
                ax.annotate(f'q_{{{p}}} = {q:.3f}', 
                           xy=(q, p), xytext=(q + 0.2, p + 0.05),
                           fontsize=10)
        
        ax.set_xlabel('x')
        ax.set_ylabel('F_X(x) = P(X ≤ x)')
        ax.set_title(title or f'Fonction de Répartition (CDF)')
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Exemples d'utilisation
print("=== Quantiles de Distributions Courantes ===\n")

# Distribution Normale N(0, 1)
cdf_normal = CumulativeDistributionFunction(distribution='normal', mu=0, sigma=1)
print("Distribution Normale N(0, 1):")
for p in [0.90, 0.95, 0.99]:
    q = cdf_normal.quantile(p)
    print(f"  Quantile {p:.0%}: {q:.6f}")

# Distribution Exponentielle λ = 1
print("\nDistribution Exponentielle (λ = 1):")
cdf_exp = CumulativeDistributionFunction(distribution='exponential', lambda_=1)
for p in [0.90, 0.95, 0.99]:
    q = cdf_exp.quantile(p)
    print(f"  Quantile {p:.0%}: {q:.6f}")

# Distribution Student t(5)
print("\nDistribution Student t(df=5):")
cdf_student = CumulativeDistributionFunction(distribution='student', df=5)
for p in [0.90, 0.95, 0.99]:
    q = cdf_student.quantile(p)
    print(f"  Quantile {p:.0%}: {q:.6f}")
```

## 5.2 ECDF (Empirical CDF, Fonction de Répartition Empirique)

**Définition** : Pour un échantillon {x₁, x₂, ..., x_N}, la CDF empirique est :

```
F_N(x) = (1/N) × Σᵢ 𝟙{xᵢ ≤ x}
```

Où 𝟙{A} est la fonction indicatrice de l'événement A.

```python
def plot_empirical_cdf(
    data: np.ndarray,
    theoretical_dist: str = None,
    title: str = "CDF Empirique vs Théorique"
) -> None:
    """
    Trace la CDF empirique et optionnellement la compare à une CDF théorique.
    
    L'ECDF (Empirical Cumulative Distribution Function) estime la vraie CDF
    à partir des données observées.
    
    Args:
        data: Échantillon de données
        theoretical_dist: Distribution théorique ('normal', 'student', etc.)
        title: Titre du graphique
    
    Exemple:
        >>> returns = np.random.normal(0, 0.02, 1000)
        >>> plot_empirical_cdf(returns, theoretical_dist='normal')
    """
    n = len(data)
    sorted_data = np.sort(data)
    
    # ECDF
    ecdf_y = np.arange(1, n + 1) / n
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Tracé de l'ECDF en escalier
    ax.step(sorted_data, ecdf_y, where='post', color='blue', 
            linewidth=2, label=f'ECDF (n={n})')
    
    # CDF théorique si demandée
    if theoretical_dist == 'normal':
        mu, sigma = np.mean(data), np.std(data)
        x_theo = np.linspace(sorted_data.min(), sorted_data.max(), 500)
        y_theo = stats.norm.cdf(x_theo, loc=mu, scale=sigma)
        ax.plot(x_theo, y_theo, 'r--', linewidth=2, 
                label=f'N({mu:.4f}, {sigma:.4f}²)')
    
    elif theoretical_dist == 'student':
        # Estimation des paramètres par MLE
        df, loc, scale = stats.t.fit(data)
        x_theo = np.linspace(sorted_data.min(), sorted_data.max(), 500)
        y_theo = stats.t.cdf(x_theo, df=df, loc=loc, scale=scale)
        ax.plot(x_theo, y_theo, 'r--', linewidth=2, 
                label=f't(df={df:.1f})')
    
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

---

# 6. Value at Risk (VaR)

## 6.1 Définition Formelle

**Définition** : La Value at Risk de X au niveau p ∈ (0, 1) est le p-quantile de X :

```
VaR_X^p = inf{x ∈ ℝ : P(X ≤ x) ≥ p}
```

**Interprétation** :
- VaR_X^{95%} = x signifie : "Il y a 95% de chances que X ne dépasse pas x"
- Ou de façon équivalente : "Il y a 5% de chances que X dépasse x"

```python
class ValueAtRisk:
    """
    Classe complète pour le calcul de la Value at Risk (VaR).
    
    La VaR (Value at Risk, Valeur à Risque) est la mesure de risque
    la plus utilisée en finance. Elle répond à la question :
    
    "Quelle est la perte maximale qu'on ne dépassera pas avec une
    probabilité de p% sur un horizon donné ?"
    
    Méthodes de calcul :
    1. Historique : Basée sur les données passées
    2. Paramétrique : Suppose une distribution (souvent normale)
    3. Monte Carlo : Simulation de scénarios
    
    Propriétés de la VaR :
    ✓ Monotone
    ✓ Homogène positive
    ✓ Invariante par translation
    ✗ PAS sous-additive (donc pas cohérente !)
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialise le calculateur de VaR.
        
        Args:
            confidence_level: Niveau de confiance p ∈ (0, 1)
                             - 0.95 (95%) : Standard industriel
                             - 0.99 (99%) : Bâle II/III réglementaire
        
        Exemple:
            >>> var_calc = ValueAtRisk(confidence_level=0.99)
            >>> var = var_calc.historical(returns)
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level doit être dans (0, 1)")
        
        self.p = confidence_level
    
    def historical(
        self, 
        returns: np.ndarray,
        method: str = 'academic'
    ) -> RiskMeasureResult:
        """
        Calcule la VaR historique (non-paramétrique).
        
        Méthode : On utilise directement les quantiles empiriques des données.
        
        Deux conventions existent :
        
        1. "Academic" (académique) : 
           VaR = quantile des pertes au niveau p
           Utilisé quand X représente directement les pertes (X > 0 = perte)
        
        2. "Practitioner" (praticien) :
           VaR = -quantile des rendements au niveau (1-p)
           Utilisé quand X représente les rendements (X < 0 = perte)
        
        Args:
            returns: Array des rendements (convention: négatif = perte)
            method: 'academic' ou 'practitioner'
        
        Returns:
            RiskMeasureResult avec la VaR
        
        Exemple:
            >>> returns = np.array([-0.02, 0.01, -0.05, 0.02, -0.01, ...])
            >>> var = ValueAtRisk(0.95).historical(returns, method='practitioner')
            >>> print(f"VaR 95% = {var.value:.2%}")
            
            Si VaR = 3%, cela signifie qu'il y a 95% de chances
            de ne pas perdre plus de 3% sur la période.
        """
        if method == 'academic':
            # Les pertes sont -returns (car rendement négatif = perte)
            losses = -returns
            var = np.percentile(losses, self.p * 100)
        elif method == 'practitioner':
            # VaR = -quantile(returns, 1-p)
            var = -np.percentile(returns, (1 - self.p) * 100)
        else:
            raise ValueError("method doit être 'academic' ou 'practitioner'")
        
        # Vérification : proportion sous la VaR
        if method == 'practitioner':
            prop_below = np.mean(returns < -var)
        else:
            prop_below = np.mean(-returns < var)
        
        return RiskMeasureResult(
            name="Historical VaR",
            value=var,
            confidence_level=self.p,
            description=f"Method={method}, P(loss<VaR)≈{1-prop_below:.2%}"
        )
    
    def parametric_normal(
        self, 
        returns: np.ndarray = None,
        mu: float = None,
        sigma: float = None
    ) -> RiskMeasureResult:
        """
        Calcule la VaR paramétrique sous hypothèse de normalité.
        
        Si X ~ N(μ, σ²), alors :
        
        VaR_X^p = μ + σ × Φ⁻¹(p)
        
        Où Φ⁻¹ est l'inverse de la CDF normale standard.
        
        Pour les pertes (rendements négatifs), avec convention praticien :
        
        VaR = -μ + σ × Φ⁻¹(p)
        
        Args:
            returns: Rendements pour estimer μ et σ
            mu: Moyenne (si fournie directement)
            sigma: Écart-type (si fourni directement)
        
        Returns:
            RiskMeasureResult avec la VaR gaussienne
        
        Avantages :
            + Formule fermée, rapide à calculer
            + Facile à communiquer
        
        Inconvénients :
            - Sous-estime le risque de queue (fat tails)
            - Les rendements ne sont pas vraiment normaux
        
        Exemple:
            >>> var = ValueAtRisk(0.99).parametric_normal(returns)
            >>> # Équivalent à : -mean + std × 2.326
        """
        if returns is not None:
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)
        elif mu is None or sigma is None:
            raise ValueError("Fournir soit 'returns' soit 'mu' et 'sigma'")
        
        # Quantile de la loi normale standard
        z_p = stats.norm.ppf(self.p)
        
        # VaR (convention praticien : perte = -rendement)
        var = -mu + sigma * z_p
        
        return RiskMeasureResult(
            name="Parametric (Normal) VaR",
            value=var,
            confidence_level=self.p,
            description=f"μ={mu:.6f}, σ={sigma:.6f}, z_{self.p}={z_p:.4f}"
        )
    
    def parametric_student(
        self, 
        returns: np.ndarray,
        df: float = None
    ) -> RiskMeasureResult:
        """
        Calcule la VaR paramétrique sous hypothèse de distribution Student-t.
        
        La distribution Student-t a des queues plus épaisses que la normale,
        ce qui capture mieux les événements extrêmes.
        
        Si X ~ t(df, μ, σ), alors :
        
        VaR_X^p = μ + σ × t_{df}^{-1}(p)
        
        Args:
            returns: Rendements
            df: Degrés de liberté (estimé si None)
        
        Returns:
            RiskMeasureResult avec la VaR Student
        
        Note:
            - df petit (3-5) : Queues très épaisses
            - df grand (>30) : Proche de la normale
        """
        if df is None:
            # Estimation MLE des paramètres
            df, loc, scale = stats.t.fit(returns)
        else:
            loc = np.mean(returns)
            scale = np.std(returns, ddof=1)
        
        # Quantile de la Student
        t_p = stats.t.ppf(self.p, df=df)
        
        # VaR
        var = -loc + scale * t_p
        
        return RiskMeasureResult(
            name="Parametric (Student-t) VaR",
            value=var,
            confidence_level=self.p,
            description=f"df={df:.2f}, loc={loc:.6f}, scale={scale:.6f}"
        )
    
    def monte_carlo(
        self, 
        returns: np.ndarray,
        n_simulations: int = 100_000,
        distribution: str = 'normal'
    ) -> RiskMeasureResult:
        """
        Calcule la VaR par simulation Monte Carlo.
        
        Méthode :
        1. Estimer les paramètres de la distribution à partir des données
        2. Simuler n_simulations scénarios
        3. Calculer le quantile empirique des simulations
        
        Args:
            returns: Rendements historiques pour calibration
            n_simulations: Nombre de simulations
            distribution: 'normal' ou 'student'
        
        Returns:
            RiskMeasureResult avec la VaR Monte Carlo
        
        Avantages :
            + Flexible (peut modéliser des distributions complexes)
            + Capture les dépendances (pour portefeuilles)
        
        Inconvénients :
            - Coûteux en calcul
            - Dépend du modèle choisi
        """
        np.random.seed(42)  # Pour reproductibilité
        
        if distribution == 'normal':
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)
            simulated = np.random.normal(mu, sigma, n_simulations)
            
        elif distribution == 'student':
            df, loc, scale = stats.t.fit(returns)
            simulated = stats.t.rvs(df=df, loc=loc, scale=scale, size=n_simulations)
        
        else:
            raise ValueError(f"Distribution '{distribution}' non supportée")
        
        # VaR = -quantile(1-p) des rendements simulés
        var = -np.percentile(simulated, (1 - self.p) * 100)
        
        return RiskMeasureResult(
            name=f"Monte Carlo VaR ({distribution})",
            value=var,
            confidence_level=self.p,
            description=f"n_sim={n_simulations:,}"
        )
    
    def compare_methods(
        self, 
        returns: np.ndarray,
        plot: bool = True
    ) -> pd.DataFrame:
        """
        Compare les différentes méthodes de calcul de la VaR.
        
        Args:
            returns: Rendements
            plot: Afficher le graphique de comparaison
        
        Returns:
            DataFrame avec les VaR par méthode
        """
        results = {
            'Historical': self.historical(returns, method='practitioner').value,
            'Normal': self.parametric_normal(returns).value,
            'Student-t': self.parametric_student(returns).value,
            'Monte Carlo (Normal)': self.monte_carlo(returns, distribution='normal').value,
            'Monte Carlo (Student)': self.monte_carlo(returns, distribution='student').value,
        }
        
        df = pd.DataFrame({
            'Method': results.keys(),
            f'VaR {self.p:.0%}': results.values()
        })
        
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogramme des rendements avec VaR
            ax1 = axes[0]
            ax1.hist(returns, bins=50, density=True, alpha=0.7, color='blue', label='Rendements')
            
            colors = ['red', 'green', 'orange', 'purple', 'brown']
            for i, (method, var) in enumerate(results.items()):
                ax1.axvline(x=-var, color=colors[i], linestyle='--', 
                           linewidth=2, label=f'{method}: {var:.4f}')
            
            ax1.set_xlabel('Rendement')
            ax1.set_ylabel('Densité')
            ax1.set_title(f'Distribution des Rendements et VaR {self.p:.0%}')
            ax1.legend(fontsize=8)
            
            # Barplot comparatif
            ax2 = axes[1]
            bars = ax2.bar(range(len(results)), list(results.values()), color=colors)
            ax2.set_xticks(range(len(results)))
            ax2.set_xticklabels(results.keys(), rotation=45, ha='right')
            ax2.set_ylabel(f'VaR {self.p:.0%}')
            ax2.set_title('Comparaison des Méthodes')
            
            # Ajouter les valeurs sur les barres
            for bar, val in zip(bars, results.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('var_comparison.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        return df
```

## 6.2 Propriétés de la VaR

```python
def verify_var_properties(returns: np.ndarray, p: float = 0.95) -> None:
    """
    Vérifie les propriétés mathématiques de la VaR.
    
    La VaR satisfait :
    ✓ Monotonicité : X ≤ Y ⇒ VaR(X) ≤ VaR(Y)
    ✓ Homogénéité positive : VaR(λX) = λ × VaR(X) pour λ > 0
    ✓ Invariance par translation : VaR(μ + X) = μ + VaR(X)
    ✗ Sous-additivité : VaR(X+Y) ≤ VaR(X) + VaR(Y) [FAUX en général]
    
    Args:
        returns: Rendements
        p: Niveau de confiance
    """
    print("=" * 70)
    print(f"VÉRIFICATION DES PROPRIÉTÉS DE LA VaR (p = {p:.0%})")
    print("=" * 70)
    
    var_calc = ValueAtRisk(confidence_level=p)
    
    # Fonction helper pour calculer la VaR
    def var(X):
        return -np.percentile(X, (1-p) * 100)
    
    # VaR de base
    var_X = var(returns)
    print(f"\nVaR(X) = {var_X:.6f}")
    
    # 1. Homogénéité positive
    print("\n1. HOMOGÉNÉITÉ POSITIVE : VaR(λX) = λ × VaR(X)")
    for lambda_ in [0.5, 2.0, 3.0]:
        var_lambdaX = var(lambda_ * returns)
        expected = lambda_ * var_X
        error = abs(var_lambdaX - expected)
        status = "✓" if error < 1e-10 else "✗"
        print(f"   λ = {lambda_}: VaR(λX) = {var_lambdaX:.6f}, "
              f"λ×VaR(X) = {expected:.6f}, erreur = {error:.2e} {status}")
    
    # 2. Invariance par translation
    print("\n2. INVARIANCE PAR TRANSLATION : VaR(μ + X) = μ + VaR(X)")
    for mu in [-0.01, 0.0, 0.01, 0.05]:
        var_muX = var(mu + returns)
        expected = mu + var_X
        error = abs(var_muX - expected)
        status = "✓" if error < 1e-10 else "✗"
        print(f"   μ = {mu:+.2f}: VaR(μ+X) = {var_muX:.6f}, "
              f"μ+VaR(X) = {expected:.6f}, erreur = {error:.2e} {status}")
    
    # 3. NON sous-additivité (contre-exemple)
    print("\n3. SOUS-ADDITIVITÉ : VaR(X+Y) ≤ VaR(X) + VaR(Y)")
    print("   [Cette propriété N'EST PAS satisfaite en général]")
    
    # Créer un contre-exemple avec Bernoulli
    np.random.seed(123)
    X_bernoulli = np.random.binomial(1, 0.02, 10000)
    Y_bernoulli = np.random.binomial(1, 0.02, 10000)
    
    var_X_b = np.percentile(X_bernoulli, p * 100)
    var_Y_b = np.percentile(Y_bernoulli, p * 100)
    var_XY_b = np.percentile(X_bernoulli + Y_bernoulli, p * 100)
    
    print(f"\n   Contre-exemple (Bernoulli, p=2%) à {p:.1%}:")
    print(f"   VaR(X) = {var_X_b:.0f}")
    print(f"   VaR(Y) = {var_Y_b:.0f}")
    print(f"   VaR(X) + VaR(Y) = {var_X_b + var_Y_b:.0f}")
    print(f"   VaR(X+Y) = {var_XY_b:.0f}")
    
    if var_XY_b > var_X_b + var_Y_b:
        print(f"\n   ❌ VIOLATION : {var_XY_b:.0f} > {var_X_b + var_Y_b:.0f}")
        print("   La VaR N'EST PAS une mesure de risque cohérente !")
    else:
        print(f"\n   ✓ Sous-additivité respectée dans cet exemple particulier")
```

---

# 7. VaR Gaussienne

## 7.1 Formule Analytique

Si X ~ N(μ, σ²), alors la VaR au niveau p est :

```
VaR_X^p = μ + σ × Φ⁻¹(p)
```

Où Φ⁻¹(p) est le **quantile de la loi normale standard**.

**Valeurs courantes de Φ⁻¹(p)** :

| Niveau p | Φ⁻¹(p) |
|----------|--------|
| 90% | 1.2816 |
| 95% | 1.6449 |
| 97.5% | 1.9600 |
| 99% | 2.3263 |
| 99.5% | 2.5758 |
| 99.9% | 3.0902 |

```python
def gaussian_var_formula(
    mu: float, 
    sigma: float, 
    p: float
) -> float:
    """
    Calcule la VaR gaussienne par formule fermée.
    
    Pour X ~ N(μ, σ²) :
    VaR_X^p = μ + σ × Φ⁻¹(p)
    
    Pour la perte L = -X (rendement négatif = perte) :
    VaR_L^p = -μ + σ × Φ⁻¹(p)
    
    Args:
        mu: Espérance des rendements E[X]
        sigma: Écart-type des rendements σ_X
        p: Niveau de confiance
    
    Returns:
        VaR au niveau p
    
    Exemple:
        >>> # Rendements : moyenne 0.1% par jour, volatilité 2%
        >>> var_99 = gaussian_var_formula(mu=0.001, sigma=0.02, p=0.99)
        >>> print(f"VaR 99% = {var_99:.4f} = {var_99*100:.2f}%")
        VaR 99% = 0.0456 = 4.56%
        
        Signifie : Il y a 99% de chances de ne pas perdre plus de 4.56% en un jour.
    """
    z_p = stats.norm.ppf(p)  # Quantile normal
    var = -mu + sigma * z_p  # Convention : perte positive
    return var


def gaussian_var_table():
    """
    Affiche un tableau des quantiles normaux et VaR pour différents niveaux.
    """
    levels = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
    
    print("\n" + "=" * 60)
    print("TABLEAU DES QUANTILES NORMAUX ET VaR GAUSSIENNE")
    print("=" * 60)
    print(f"\n{'Niveau p':<12} {'Φ⁻¹(p)':<12} {'VaR (σ=1)':<12} {'VaR (σ=2%)':<12}")
    print("-" * 48)
    
    for p in levels:
        z = stats.norm.ppf(p)
        var_sigma1 = z  # μ=0, σ=1
        var_sigma2pct = 0.02 * z  # μ=0, σ=2%
        print(f"{p:<12.1%} {z:<12.4f} {var_sigma1:<12.4f} {var_sigma2pct:<12.4f}")
    
    print("\nNote : VaR = -μ + σ × Φ⁻¹(p) avec μ = 0")


gaussian_var_table()
```

## 7.2 Propriété de Sous-Additivité pour les Gaussiennes

**Théorème** : La VaR EST sous-additive sur les variables gaussiennes (même corrélées).

**Preuve** : Pour X, Y gaussiennes :
- σ_{X+Y} = √(σ_X² + σ_Y² + 2ρσ_Xσ_Y) ≤ σ_X + σ_Y (par Cauchy-Schwarz)
- Donc VaR(X+Y) = μ_{X+Y} + σ_{X+Y} × z_p ≤ μ_X + μ_Y + (σ_X + σ_Y) × z_p = VaR(X) + VaR(Y)

```python
def demonstrate_gaussian_subadditivity():
    """
    Démontre que la VaR est sous-additive pour les variables gaussiennes.
    
    Théorème : Pour X ~ N(μ_X, σ_X²) et Y ~ N(μ_Y, σ_Y²) :
    
    VaR(X+Y) ≤ VaR(X) + VaR(Y)
    
    Car : σ_{X+Y} = √(σ_X² + σ_Y² + 2ρσ_Xσ_Y) ≤ σ_X + σ_Y
    (par l'inégalité de Cauchy-Schwarz)
    """
    print("=" * 70)
    print("SOUS-ADDITIVITÉ DE LA VaR POUR LES GAUSSIENNES")
    print("=" * 70)
    
    np.random.seed(42)
    n = 100_000
    p = 0.95
    
    # Paramètres
    mu_X, sigma_X = 0.001, 0.02
    mu_Y, sigma_Y = 0.002, 0.03
    
    for rho in [-0.5, 0.0, 0.5, 0.9]:
        print(f"\nCorrélation ρ = {rho}")
        
        # Génération de variables corrélées
        # X = μ_X + σ_X × Z1
        # Y = μ_Y + σ_Y × (ρ×Z1 + √(1-ρ²)×Z2)
        Z1 = np.random.normal(0, 1, n)
        Z2 = np.random.normal(0, 1, n)
        
        X = mu_X + sigma_X * Z1
        Y = mu_Y + sigma_Y * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
        
        # Calcul des VaR empiriques
        var_X = -np.percentile(X, (1-p) * 100)
        var_Y = -np.percentile(Y, (1-p) * 100)
        var_XY = -np.percentile(X + Y, (1-p) * 100)
        
        # Vérification analytique
        sigma_XY = np.sqrt(sigma_X**2 + sigma_Y**2 + 2*rho*sigma_X*sigma_Y)
        var_XY_theo = -(mu_X + mu_Y) + sigma_XY * stats.norm.ppf(p)
        
        print(f"  VaR(X) = {var_X:.6f}")
        print(f"  VaR(Y) = {var_Y:.6f}")
        print(f"  VaR(X) + VaR(Y) = {var_X + var_Y:.6f}")
        print(f"  VaR(X+Y) empirique = {var_XY:.6f}")
        print(f"  VaR(X+Y) théorique = {var_XY_theo:.6f}")
        
        if var_XY <= var_X + var_Y + 1e-6:
            print(f"  ✓ Sous-additivité : {var_XY:.6f} ≤ {var_X + var_Y:.6f}")
        else:
            print(f"  ✗ Violation : {var_XY:.6f} > {var_X + var_Y:.6f}")


demonstrate_gaussian_subadditivity()
```

---

# 8. Calculs Numériques et Implémentation

## 8.1 Téléchargement et Analyse de Données Réelles

```python
def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Télécharge les données boursières depuis Yahoo Finance.
    
    Args:
        ticker: Symbole du titre (ex: "^HSI" pour Hang Seng, "AAPL" pour Apple)
        start_date: Date de début (format "YYYY-MM-DD")
        end_date: Date de fin
    
    Returns:
        DataFrame avec les prix et rendements
    
    Exemple:
        >>> data = download_stock_data("^GSPC", "2020-01-01", "2024-01-01")
        >>> print(data.head())
    """
    print(f"Téléchargement de {ticker}...")
    
    # Téléchargement
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if stock.empty:
        raise ValueError(f"Aucune donnée pour {ticker}")
    
    # Calcul des rendements
    # Rendement simple : R_t = (P_t - P_{t-1}) / P_{t-1}
    stock['Returns'] = stock['Adj Close'].pct_change()
    
    # Rendement log : r_t = ln(P_t / P_{t-1})
    stock['Log_Returns'] = np.log(stock['Adj Close'] / stock['Adj Close'].shift(1))
    
    # Supprimer les NaN
    stock = stock.dropna()
    
    print(f"  → {len(stock)} observations du {stock.index[0].date()} au {stock.index[-1].date()}")
    print(f"  → Rendement moyen : {stock['Returns'].mean():.4%}")
    print(f"  → Volatilité : {stock['Returns'].std():.4%}")
    
    return stock


def complete_var_analysis(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: str = "2024-01-01",
    confidence_levels: list = [0.95, 0.99]
) -> dict:
    """
    Analyse complète de la VaR pour un actif.
    
    Args:
        ticker: Symbole boursier
        start_date: Date de début
        end_date: Date de fin
        confidence_levels: Niveaux de confiance à analyser
    
    Returns:
        Dictionnaire avec tous les résultats
    
    Exemple:
        >>> results = complete_var_analysis("^HSI", "2019-01-01", "2024-01-01")
        >>> print(results['summary'])
    """
    # Téléchargement
    data = download_stock_data(ticker, start_date, end_date)
    returns = data['Returns'].values
    
    results = {
        'ticker': ticker,
        'n_observations': len(returns),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'var_results': {}
    }
    
    # Calcul des VaR pour chaque niveau de confiance
    for p in confidence_levels:
        var_calc = ValueAtRisk(confidence_level=p)
        
        results['var_results'][p] = {
            'historical': var_calc.historical(returns, method='practitioner').value,
            'normal': var_calc.parametric_normal(returns).value,
            'student': var_calc.parametric_student(returns).value,
        }
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Prix
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['Adj Close'], color='blue', linewidth=1)
    ax1.set_title(f'{ticker} - Prix de Clôture Ajusté')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Prix')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rendements
    ax2 = axes[0, 1]
    ax2.plot(data.index, returns, color='blue', linewidth=0.5, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    # Ajouter les VaR
    for p in confidence_levels:
        var_hist = results['var_results'][p]['historical']
        ax2.axhline(y=-var_hist, color='red' if p == 0.95 else 'orange', 
                   linestyle='--', linewidth=2, label=f'VaR {p:.0%} = {var_hist:.4f}')
    
    ax2.set_title(f'{ticker} - Rendements Journaliers')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rendement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogramme des rendements
    ax3 = axes[1, 0]
    ax3.hist(returns, bins=100, density=True, alpha=0.7, color='blue', label='Empirique')
    
    # Superposer la densité normale
    x_range = np.linspace(returns.min(), returns.max(), 500)
    normal_pdf = stats.norm.pdf(x_range, loc=np.mean(returns), scale=np.std(returns))
    ax3.plot(x_range, normal_pdf, 'r-', linewidth=2, label='Normal')
    
    # Ajouter les VaR
    for p in confidence_levels:
        var_hist = results['var_results'][p]['historical']
        ax3.axvline(x=-var_hist, color='darkred', linestyle='--', 
                   linewidth=2, label=f'VaR {p:.0%}')
    
    ax3.set_title('Distribution des Rendements')
    ax3.set_xlabel('Rendement')
    ax3.set_ylabel('Densité')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. QQ-Plot
    ax4 = axes[1, 1]
    stats.probplot(returns, dist="norm", plot=ax4)
    ax4.set_title('QQ-Plot (Normal)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker.replace("^", "")}_var_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Résumé
    print("\n" + "=" * 70)
    print(f"RÉSUMÉ DE L'ANALYSE VaR - {ticker}")
    print("=" * 70)
    
    print(f"\nStatistiques descriptives:")
    print(f"  Nombre d'observations : {results['n_observations']:,}")
    print(f"  Rendement moyen : {results['mean_return']:.4%}")
    print(f"  Volatilité (σ) : {results['std_return']:.4%}")
    print(f"  Rendement min : {results['min_return']:.4%}")
    print(f"  Rendement max : {results['max_return']:.4%}")
    
    print(f"\nValue at Risk:")
    for p in confidence_levels:
        print(f"\n  Niveau {p:.0%}:")
        for method, value in results['var_results'][p].items():
            print(f"    {method:<12}: {value:.4%}")
    
    return results


# Exemple d'exécution
if __name__ == "__main__":
    # Analyse du Hang Seng Index
    results_hsi = complete_var_analysis(
        ticker="^HSI",
        start_date="2019-01-01",
        end_date="2024-01-01",
        confidence_levels=[0.95, 0.99]
    )
```

## 8.2 Backtesting de la VaR

```python
def backtest_var(
    returns: np.ndarray,
    confidence_level: float = 0.99,
    window_size: int = 250,
    method: str = 'historical'
) -> dict:
    """
    Backtesting de la VaR : vérifie si le modèle est bien calibré.
    
    Principe :
    - Calculer la VaR chaque jour sur une fenêtre glissante
    - Compter le nombre de "violations" (jours où la perte dépasse la VaR)
    - Le taux de violations devrait être proche de (1 - p)
    
    Test de Kupiec :
    - H0 : Le modèle est bien calibré
    - Statistique LR = 2 × [n_viol × ln(n_viol/n_expected) + (n-n_viol) × ln((n-n_viol)/(n-n_expected))]
    - Suit une χ²(1) sous H0
    
    Args:
        returns: Rendements
        confidence_level: Niveau de confiance
        window_size: Taille de la fenêtre d'estimation
        method: 'historical' ou 'normal'
    
    Returns:
        Dictionnaire avec les résultats du backtest
    
    Exemple:
        >>> bt = backtest_var(returns, confidence_level=0.99, window_size=250)
        >>> print(f"Taux de violations: {bt['violation_rate']:.2%}")
        >>> print(f"P-value Kupiec: {bt['kupiec_pvalue']:.4f}")
    """
    n = len(returns)
    n_test = n - window_size
    
    if n_test <= 0:
        raise ValueError("Pas assez de données pour le backtest")
    
    var_estimates = np.zeros(n_test)
    violations = np.zeros(n_test, dtype=bool)
    
    var_calc = ValueAtRisk(confidence_level=confidence_level)
    
    for t in range(n_test):
        # Fenêtre d'estimation : [t, t + window_size)
        window = returns[t:t + window_size]
        
        # Calcul de la VaR
        if method == 'historical':
            var_t = var_calc.historical(window, method='practitioner').value
        elif method == 'normal':
            var_t = var_calc.parametric_normal(window).value
        else:
            raise ValueError(f"Méthode '{method}' non supportée")
        
        var_estimates[t] = var_t
        
        # Vérifier si violation
        actual_return = returns[t + window_size]
        violations[t] = actual_return < -var_t
    
    # Statistiques
    n_violations = np.sum(violations)
    violation_rate = n_violations / n_test
    expected_rate = 1 - confidence_level
    expected_violations = expected_rate * n_test
    
    # Test de Kupiec (LR test)
    if n_violations > 0 and n_violations < n_test:
        lr_stat = 2 * (
            n_violations * np.log(n_violations / expected_violations) +
            (n_test - n_violations) * np.log((n_test - n_violations) / (n_test - expected_violations))
        )
        kupiec_pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)
    else:
        lr_stat = np.nan
        kupiec_pvalue = np.nan
    
    results = {
        'n_test': n_test,
        'n_violations': n_violations,
        'violation_rate': violation_rate,
        'expected_rate': expected_rate,
        'expected_violations': expected_violations,
        'kupiec_lr_stat': lr_stat,
        'kupiec_pvalue': kupiec_pvalue,
        'var_estimates': var_estimates,
        'violations': violations
    }
    
    # Affichage
    print("\n" + "=" * 60)
    print(f"BACKTEST DE LA VaR - Méthode: {method}")
    print("=" * 60)
    print(f"\nParamètres:")
    print(f"  Niveau de confiance : {confidence_level:.1%}")
    print(f"  Fenêtre d'estimation : {window_size} jours")
    print(f"  Période de test : {n_test} jours")
    
    print(f"\nRésultats:")
    print(f"  Violations observées : {n_violations}")
    print(f"  Violations attendues : {expected_violations:.1f}")
    print(f"  Taux de violations : {violation_rate:.2%}")
    print(f"  Taux attendu : {expected_rate:.2%}")
    
    print(f"\nTest de Kupiec:")
    print(f"  Statistique LR : {lr_stat:.4f}")
    print(f"  P-value : {kupiec_pvalue:.4f}")
    
    if kupiec_pvalue > 0.05:
        print(f"  → ✓ Modèle bien calibré (p > 0.05)")
    else:
        print(f"  → ✗ Modèle mal calibré (p < 0.05)")
    
    return results


def plot_backtest_results(
    returns: np.ndarray,
    backtest_results: dict,
    window_size: int = 250
) -> None:
    """
    Visualise les résultats du backtest.
    """
    var_estimates = backtest_results['var_estimates']
    violations = backtest_results['violations']
    
    # Période de test
    test_returns = returns[window_size:]
    dates = np.arange(len(test_returns))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 1. Rendements vs VaR
    ax1 = axes[0]
    ax1.plot(dates, test_returns, 'b-', linewidth=0.5, alpha=0.7, label='Rendements')
    ax1.plot(dates, -var_estimates, 'r-', linewidth=1.5, label='VaR')
    ax1.fill_between(dates, -var_estimates, test_returns.min(), alpha=0.1, color='red')
    
    # Marquer les violations
    violation_dates = dates[violations]
    violation_returns = test_returns[violations]
    ax1.scatter(violation_dates, violation_returns, color='red', s=50, zorder=5, 
               label=f'Violations ({len(violation_dates)})')
    
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Rendement')
    ax1.set_title('Backtest de la VaR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Violations cumulées
    ax2 = axes[1]
    cumulative_violations = np.cumsum(violations)
    expected_violations = (1 - 0.99) * np.arange(1, len(violations) + 1)
    
    ax2.plot(dates, cumulative_violations, 'r-', linewidth=2, label='Violations observées')
    ax2.plot(dates, expected_violations, 'k--', linewidth=1.5, label='Violations attendues')
    ax2.fill_between(dates, 
                     expected_violations - 2*np.sqrt(expected_violations),
                     expected_violations + 2*np.sqrt(expected_violations),
                     alpha=0.2, color='gray', label='Bande ±2σ')
    
    ax2.set_xlabel('Jours')
    ax2.set_ylabel('Violations cumulées')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('var_backtest.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

# 9. Code Complet et Classes Utilitaires

## 9.1 Module Complet

```python
"""
=============================================================================
MODULE COMPLET : VALUE AT RISK (VaR)
=============================================================================

Ce module fournit une implémentation complète des méthodes de calcul
de la Value at Risk (VaR) et des outils associés.

Auteur: Basé sur le cours MH8331 de N. Privault (NTU Singapore)
Date: Janvier 2026

Classes principales:
- ValueAtRisk : Calcul de la VaR (historique, paramétrique, Monte Carlo)
- CumulativeDistributionFunction : Manipulation des CDF
- RiskMeasureResult : Stockage des résultats

Fonctions principales:
- expected_value_premium() : Prime d'espérance
- standard_deviation_premium() : Prime d'écart-type
- conditional_tail_expectation() : CTE
- gaussian_var_formula() : VaR gaussienne analytique
- backtest_var() : Backtesting de la VaR
- complete_var_analysis() : Analyse complète

Exemple d'utilisation:
    >>> from var_module import ValueAtRisk, complete_var_analysis
    >>> 
    >>> # Analyse rapide
    >>> results = complete_var_analysis("^GSPC", "2020-01-01", "2024-01-01")
    >>> 
    >>> # Calcul manuel
    >>> import numpy as np
    >>> returns = np.random.normal(0.001, 0.02, 1000)
    >>> var_calc = ValueAtRisk(confidence_level=0.99)
    >>> var_hist = var_calc.historical(returns)
    >>> var_norm = var_calc.parametric_normal(returns)
    >>> print(f"VaR Historique: {var_hist.value:.4%}")
    >>> print(f"VaR Normale: {var_norm.value:.4%}")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Union, Tuple, List, Optional
from dataclasses import dataclass
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


# [Inclure toutes les classes et fonctions définies précédemment]
# ...


def demo():
    """
    Démonstration complète du module VaR.
    """
    print("=" * 70)
    print("DÉMONSTRATION DU MODULE VALUE AT RISK")
    print("=" * 70)
    
    # 1. Génération de données simulées
    print("\n1. GÉNÉRATION DE DONNÉES SIMULÉES")
    np.random.seed(42)
    n = 1000
    mu = 0.0005  # 0.05% par jour
    sigma = 0.02  # 2% de volatilité
    returns = np.random.normal(mu, sigma, n)
    print(f"   {n} rendements simulés ~ N({mu}, {sigma}²)")
    
    # 2. Calcul des mesures de risque basiques
    print("\n2. MESURES DE RISQUE BASIQUES")
    evp = expected_value_premium(returns, alpha=0.1)
    sdp = standard_deviation_premium(returns, alpha=2)
    cte = conditional_tail_expectation(returns, threshold=0)
    print(f"   {evp}")
    print(f"   {sdp}")
    print(f"   {cte}")
    
    # 3. Calcul de la VaR
    print("\n3. VALUE AT RISK")
    var_calc = ValueAtRisk(confidence_level=0.99)
    
    var_hist = var_calc.historical(returns, method='practitioner')
    var_norm = var_calc.parametric_normal(returns)
    var_student = var_calc.parametric_student(returns)
    var_mc = var_calc.monte_carlo(returns, n_simulations=100000)
    
    print(f"   {var_hist}")
    print(f"   {var_norm}")
    print(f"   {var_student}")
    print(f"   {var_mc}")
    
    # 4. VaR gaussienne analytique
    print("\n4. VaR GAUSSIENNE ANALYTIQUE")
    var_theo = gaussian_var_formula(mu, sigma, 0.99)
    print(f"   VaR théorique (μ={mu}, σ={sigma}, p=99%): {var_theo:.6f}")
    
    # 5. Vérification des propriétés
    print("\n5. VÉRIFICATION DES PROPRIÉTÉS")
    verify_var_properties(returns, p=0.95)
    
    # 6. Démonstration non-sous-additivité
    print("\n6. NON-SOUS-ADDITIVITÉ")
    demonstrate_var_non_subadditivity()
    
    # 7. Sous-additivité gaussienne
    print("\n7. SOUS-ADDITIVITÉ GAUSSIENNE")
    demonstrate_gaussian_subadditivity()
    
    print("\n" + "=" * 70)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 70)


if __name__ == "__main__":
    demo()
```

---

# 10. Exercices Résolus

## Exercice 1 : Distribution de Pareto

**Énoncé** : Soit X une variable aléatoire de distribution Pareto avec densité :
```
f_X(x) = γ × θ^γ / (θ + x)^{γ+1}, x ≥ 0
```

a) Calculer la CDF F_X(x)
b) Calculer la VaR^p_X pour θ = 40, γ = 2, p = 99%

```python
def exercice_pareto():
    """
    Résolution de l'exercice sur la distribution de Pareto.
    """
    print("=" * 70)
    print("EXERCICE : DISTRIBUTION DE PARETO")
    print("=" * 70)
    
    # Paramètres
    theta = 40
    gamma = 2
    p = 0.99
    
    # a) CDF
    # F_X(x) = ∫₀ˣ γ θ^γ / (θ + y)^{γ+1} dy
    # F_X(x) = 1 - (θ / (θ + x))^γ
    
    print("\na) CDF de la distribution de Pareto:")
    print(f"   F_X(x) = 1 - (θ / (θ + x))^γ")
    print(f"   F_X(x) = 1 - ({theta} / ({theta} + x))^{gamma}")
    
    # b) VaR
    # VaR^p = F_X^{-1}(p) = θ × ((1-p)^{-1/γ} - 1)
    var_pareto = theta * ((1 - p)**(-1/gamma) - 1)
    
    print(f"\nb) VaR au niveau p = {p:.0%}:")
    print(f"   VaR^p = θ × ((1-p)^{{-1/γ}} - 1)")
    print(f"   VaR^{{99%}} = {theta} × ((1-0.99)^{{-1/{gamma}}} - 1)")
    print(f"   VaR^{{99%}} = {theta} × ({(1-p)**(-1/gamma):.4f} - 1)")
    print(f"   VaR^{{99%}} = {var_pareto:.4f}")
    
    # Vérification numérique
    cdf_pareto = CumulativeDistributionFunction(
        distribution='pareto', gamma=gamma, theta=theta
    )
    var_numerical = cdf_pareto.quantile(p)
    print(f"\n   Vérification numérique: {var_numerical:.4f}")
    
    # Graphique
    x = np.linspace(0, 500, 1000)
    F_x = 1 - (theta / (theta + x))**gamma
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, F_x, 'b-', linewidth=2, label='CDF Pareto')
    plt.axhline(y=p, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=var_pareto, color='red', linestyle='--', linewidth=2, 
               label=f'VaR 99% = {var_pareto:.2f}')
    plt.xlabel('x')
    plt.ylabel('F_X(x)')
    plt.title(f'Distribution de Pareto (γ={gamma}, θ={theta})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return var_pareto


exercice_pareto()
```

## Exercice 2 : Distribution Exponentielle

**Énoncé** : Soit X ~ Exp(λ) avec P(X ≤ x) = 1 - e^{-λx}.

a) Calculer VaR^p_X et VaR^{95%}_X pour λ = 1
b) Calculer le capital requis si les passifs sont estimés par E[X]

```python
def exercice_exponentielle():
    """
    Résolution de l'exercice sur la distribution exponentielle.
    """
    print("=" * 70)
    print("EXERCICE : DISTRIBUTION EXPONENTIELLE")
    print("=" * 70)
    
    # Paramètres
    lambda_ = 1
    p = 0.95
    
    # a) VaR
    # P(X ≤ x) = 1 - e^{-λx} = p
    # e^{-λx} = 1 - p
    # -λx = ln(1-p)
    # x = -ln(1-p) / λ = -ln(1-p) × E[X]
    
    EX = 1 / lambda_  # Espérance de l'exponentielle
    var_exp = -np.log(1 - p) / lambda_
    
    print(f"\na) VaR de la distribution exponentielle (λ = {lambda_}):")
    print(f"   VaR^p = -ln(1-p) / λ = E[X] × ln(1/(1-p))")
    print(f"   VaR^{{95%}} = -{np.log(1-p):.6f} / {lambda_}")
    print(f"   VaR^{{95%}} = {var_exp:.6f}")
    print(f"   VaR^{{95%}} ≈ {var_exp/EX:.3f} × E[X]")
    
    # b) Capital requis
    # C_X = V_X - L_X où L_X = E[X]
    capital = var_exp - EX
    
    print(f"\nb) Capital requis:")
    print(f"   E[X] = 1/λ = {EX:.4f}")
    print(f"   C_X = VaR - E[X] = {var_exp:.4f} - {EX:.4f}")
    print(f"   C_X = {capital:.4f}")
    
    # Graphique
    x = np.linspace(0, 6, 500)
    F_x = 1 - np.exp(-lambda_ * x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, F_x, 'b-', linewidth=2, label='CDF Exponentielle')
    plt.axhline(y=p, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=var_exp, color='red', linestyle='--', linewidth=2,
               label=f'VaR 95% = {var_exp:.3f}')
    plt.axvline(x=EX, color='green', linestyle='--', linewidth=2,
               label=f'E[X] = {EX:.1f}')
    plt.xlabel('x')
    plt.ylabel('F_X(x)')
    plt.title(f'Distribution Exponentielle (λ={lambda_})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return var_exp, capital


exercice_exponentielle()
```

---

# Annexes

## A.1 Formules Récapitulatives

| Mesure | Formule | Note |
|--------|---------|------|
| **Prime d'espérance** | E_X = (1+α)×E[X] | α = chargement |
| **Prime d'écart-type** | SD_X = E[X] + α×σ_X | σ_X = √Var[X] |
| **CTE** | E[X \| X < 0] | Queue gauche |
| **VaR** | inf{x : P(X≤x) ≥ p} | p-quantile |
| **VaR gaussienne** | μ + σ×Φ⁻¹(p) | X ~ N(μ,σ²) |
| **VaR exponentielle** | -ln(1-p)/λ | X ~ Exp(λ) |
| **VaR Pareto** | θ×((1-p)^{-1/γ}-1) | Pareto(γ,θ) |

## A.2 Quantiles Normaux Standards

```python
print("\nQuantiles de la loi normale standard N(0,1):")
print("-" * 40)
for p in [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]:
    z = stats.norm.ppf(p)
    print(f"  Φ⁻¹({p:.3f}) = {z:.6f}")
```

## A.3 Références

1. Privault, N. "Notes on Financial Risk and Analytics", NTU Singapore, 2026
2. Embrechts, P. & Hofert, M. "A note on generalized inverses", Mathematical Methods of Operations Research, 2013
3. Hardy, M. "An Introduction to Risk Measures for Actuarial Applications", SOA Study Note, 2006
4. Mina, J. & Xiao, J.Y. "Return to RiskMetrics: The Evolution of a Standard", RiskMetrics Group, 2001

---

*Document généré avec tous les acronymes développés et exemples de code fonctionnel.*

*Version: Janvier 2026*
