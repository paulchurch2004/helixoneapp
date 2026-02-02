# 🏦 HELIXONE - Guide d'Implémentation Complet

> **Mission** : Construire une plateforme de gestion d'actifs rivalisant avec BlackRock Aladdin
> **Sources** : Stanford CME 241 (RL Finance) + Best Practices Industrie
> **Pour** : Claude Agent - Implémentation automatisée

---

# 📑 TABLE DES MATIÈRES

## PARTIE A : FONDATIONS (Stanford RL)
1. [Core - Distributions & Processus](#partie-a1--core)
2. [MDP - Processus de Décision](#partie-a2--mdp)
3. [Algorithmes RL](#partie-a3--algorithmes-rl)

## PARTIE B : RISK MANAGEMENT
4. [Utility & Risk Metrics](#partie-b1--utility--risk-metrics)
5. [Factor Risk Models](#partie-b2--factor-risk-models)
6. [VaR & Stress Testing](#partie-b3--var--stress-testing)

## PARTIE C : PORTFOLIO MANAGEMENT
7. [Optimisation Statique (Markowitz)](#partie-c1--optimisation-statique)
8. [Optimisation Dynamique (Merton)](#partie-c2--optimisation-dynamique)
9. [Black-Litterman](#partie-c3--black-litterman)
10. [Rebalancing & Constraints](#partie-c4--rebalancing)

## PARTIE D : FIXED INCOME
11. [Bond Pricing & Analytics](#partie-d1--bond-pricing)
12. [Duration & Convexity](#partie-d2--duration--convexity)
13. [Yield Curve Models](#partie-d3--yield-curve)

## PARTIE E : DERIVATIVES
14. [Options Pricing](#partie-e1--options-pricing)
15. [Greeks & Hedging](#partie-e2--greeks--hedging)
16. [American Options (RL)](#partie-e3--american-options-rl)

## PARTIE F : EXECUTION
17. [Optimal Execution (Almgren-Chriss)](#partie-f1--optimal-execution)
18. [Market Making (Avellaneda-Stoikov)](#partie-f2--market-making)
19. [TWAP/VWAP](#partie-f3--twapvwap)

## PARTIE G : PERFORMANCE & ATTRIBUTION
20. [Performance Metrics](#partie-g1--performance-metrics)
21. [Brinson Attribution](#partie-g2--brinson-attribution)
22. [Factor Attribution](#partie-g3--factor-attribution)

## PARTIE H : INFRASTRUCTURE
23. [Real-time Risk Engine](#partie-h1--real-time-engine)
24. [Data Pipeline](#partie-h2--data-pipeline)
25. [API Design](#partie-h3--api-design)

---

# 🏗️ ARCHITECTURE GLOBALE

## Structure du Projet

```
helixone/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── constants.py
│
├── helixone/
│   ├── __init__.py
│   │
│   ├── core/                       # PARTIE A
│   │   ├── __init__.py
│   │   ├── distributions.py        # Distributions probabilistes
│   │   ├── markov_process.py       # Processus de Markov
│   │   ├── mdp.py                  # MDP
│   │   ├── function_approx.py      # Approximation de fonctions
│   │   └── rl_base.py              # Classes de base RL
│   │
│   ├── risk/                       # PARTIE B
│   │   ├── __init__.py
│   │   ├── utility.py              # Fonctions d'utilité
│   │   ├── metrics.py              # VaR, CVaR, Sharpe, etc.
│   │   ├── factor_models.py        # Modèles factoriels
│   │   ├── stress_testing.py       # Stress tests
│   │   └── monitoring.py           # Monitoring temps réel
│   │
│   ├── portfolio/                  # PARTIE C
│   │   ├── __init__.py
│   │   ├── optimization.py         # Markowitz, MVO
│   │   ├── merton.py               # Allocation dynamique
│   │   ├── black_litterman.py      # Black-Litterman
│   │   ├── rebalancing.py          # Rebalancing
│   │   └── constraints.py          # Contraintes
│   │
│   ├── fixed_income/               # PARTIE D
│   │   ├── __init__.py
│   │   ├── bonds.py                # Pricing obligations
│   │   ├── duration.py             # Duration/Convexité
│   │   ├── yield_curve.py          # Courbe des taux
│   │   └── credit.py               # Risque crédit
│   │
│   ├── derivatives/                # PARTIE E
│   │   ├── __init__.py
│   │   ├── black_scholes.py        # Black-Scholes
│   │   ├── greeks.py               # Greeks
│   │   ├── american_options.py     # Options américaines
│   │   ├── monte_carlo.py          # MC pricing
│   │   └── hedging.py              # Stratégies de couverture
│   │
│   ├── execution/                  # PARTIE F
│   │   ├── __init__.py
│   │   ├── almgren_chriss.py       # Exécution optimale
│   │   ├── market_making.py        # Market making
│   │   ├── twap_vwap.py            # TWAP/VWAP
│   │   └── impact_models.py        # Modèles d'impact
│   │
│   ├── performance/                # PARTIE G
│   │   ├── __init__.py
│   │   ├── metrics.py              # Métriques de performance
│   │   ├── attribution.py          # Attribution Brinson
│   │   └── factor_attribution.py   # Attribution factorielle
│   │
│   ├── algorithms/                 # Algorithmes RL
│   │   ├── __init__.py
│   │   ├── dp.py                   # Programmation dynamique
│   │   ├── monte_carlo.py          # Monte Carlo RL
│   │   ├── td_learning.py          # TD(0), TD(λ)
│   │   ├── q_learning.py           # Q-Learning, DQN
│   │   ├── sarsa.py                # SARSA
│   │   ├── policy_gradient.py      # REINFORCE, A2C
│   │   └── actor_critic.py         # Actor-Critic avancé
│   │
│   ├── data/                       # PARTIE H
│   │   ├── __init__.py
│   │   ├── market_data.py          # Données de marché
│   │   ├── loaders.py              # Chargeurs de données
│   │   └── cache.py                # Cache
│   │
│   ├── engine/                     # Moteur temps réel
│   │   ├── __init__.py
│   │   ├── risk_engine.py          # Calculs de risque
│   │   ├── pricing_engine.py       # Pricing
│   │   └── execution_engine.py     # Exécution
│   │
│   ├── api/                        # API REST
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── math_utils.py
│       ├── date_utils.py
│       └── validation.py
│
├── tests/
│   ├── __init__.py
│   ├── test_core/
│   ├── test_risk/
│   ├── test_portfolio/
│   └── ...
│
└── examples/
    ├── portfolio_optimization.py
    ├── risk_analysis.py
    └── trading_strategy.py
```

## Requirements

```text
# requirements.txt

# Core
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# Machine Learning
torch>=1.9.0
scikit-learn>=0.24.0

# Finance
yfinance>=0.1.70
pandas-datareader>=0.10.0

# Visualization
matplotlib>=3.4.0
plotly>=5.0.0
seaborn>=0.11.0

# API
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0

# Database
sqlalchemy>=1.4.0
redis>=3.5.0

# Utils
python-dateutil>=2.8.0
typing_extensions>=3.10.0
numba>=0.54.0  # Pour accélération

# Testing
pytest>=6.2.0
pytest-cov>=2.12.0
```

---

# PARTIE A : FONDATIONS (Stanford RL)

---

## PARTIE A.1 : CORE

### distributions.py

```python
"""
HelixOne - Distributions de probabilité
Source: Stanford CME 241

Implémente les distributions nécessaires pour:
- Simulation de processus stochastiques
- Modélisation d'incertitude
- Algorithmes RL
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, Callable, Mapping, Sequence, 
    Tuple, Iterator, Optional, List
)
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, lognorm
from collections import defaultdict

# Type variables
A = TypeVar('A')  # Type générique pour outcome
S = TypeVar('S')  # State type
X = TypeVar('X')  # Generic type


class Distribution(ABC, Generic[A]):
    """
    Classe de base abstraite pour toutes les distributions.
    
    Une distribution représente une variable aléatoire et permet:
    - D'échantillonner des valeurs
    - De calculer des espérances
    - De transformer (map/flatmap)
    """
    
    @abstractmethod
    def sample(self) -> A:
        """Tire un échantillon de la distribution."""
        pass
    
    def sample_n(self, n: int) -> List[A]:
        """Tire n échantillons indépendants."""
        return [self.sample() for _ in range(n)]
    
    @abstractmethod
    def expectation(self, f: Callable[[A], float]) -> float:
        """
        Calcule E[f(X)] où X suit cette distribution.
        
        Args:
            f: Fonction à appliquer
            
        Returns:
            Espérance de f(X)
        """
        pass
    
    def mean(self) -> float:
        """Espérance E[X] (si A est numérique)."""
        return self.expectation(lambda x: float(x))
    
    def variance(self) -> float:
        """Variance Var[X] (si A est numérique)."""
        mu = self.mean()
        return self.expectation(lambda x: (float(x) - mu) ** 2)
    
    def std(self) -> float:
        """Écart-type."""
        return np.sqrt(self.variance())
    
    def map(self, f: Callable[[A], X]) -> Distribution[X]:
        """
        Transforme la distribution par une fonction.
        Si X ~ D, retourne la distribution de f(X).
        """
        return MappedDistribution(self, f)
    
    def apply(self, f: Callable[[A], Distribution[X]]) -> Distribution[X]:
        """
        Flatmap / bind monadique.
        Permet de chaîner des distributions dépendantes.
        """
        return FlatMappedDistribution(self, f)


class Constant(Distribution[A]):
    """Distribution dégénérée (valeur constante avec probabilité 1)."""
    
    def __init__(self, value: A):
        self.value = value
    
    def sample(self) -> A:
        return self.value
    
    def expectation(self, f: Callable[[A], float]) -> float:
        return f(self.value)
    
    def __repr__(self) -> str:
        return f"Constant({self.value})"


class Categorical(Distribution[A]):
    """
    Distribution catégorique (discrète finie).
    
    Exemple:
        >>> d = Categorical({'head': 0.5, 'tail': 0.5})
        >>> d.sample()
        'head'
    """
    
    def __init__(self, probabilities: Mapping[A, float]):
        """
        Args:
            probabilities: Dict {outcome: probability}
                          Les probabilités sont normalisées automatiquement.
        """
        self.probabilities = dict(probabilities)
        
        # Filtrer les probabilités nulles
        self.probabilities = {k: v for k, v in self.probabilities.items() if v > 0}
        
        # Normaliser
        total = sum(self.probabilities.values())
        if total <= 0:
            raise ValueError("Les probabilités doivent être positives")
        
        if not np.isclose(total, 1.0):
            self.probabilities = {k: v / total for k, v in self.probabilities.items()}
        
        # Pour échantillonnage efficace
        self._outcomes = list(self.probabilities.keys())
        self._probs = np.array([self.probabilities[o] for o in self._outcomes])
    
    def sample(self) -> A:
        idx = np.random.choice(len(self._outcomes), p=self._probs)
        return self._outcomes[idx]
    
    def expectation(self, f: Callable[[A], float]) -> float:
        return sum(p * f(x) for x, p in self.probabilities.items())
    
    def probability(self, outcome: A) -> float:
        """Retourne P(X = outcome)."""
        return self.probabilities.get(outcome, 0.0)
    
    def support(self) -> List[A]:
        """Retourne le support (outcomes possibles)."""
        return self._outcomes.copy()
    
    def __repr__(self) -> str:
        return f"Categorical({self.probabilities})"


class Bernoulli(Distribution[bool]):
    """Distribution de Bernoulli."""
    
    def __init__(self, p: float):
        """
        Args:
            p: Probabilité de succès (True)
        """
        assert 0 <= p <= 1, "p doit être dans [0, 1]"
        self.p = p
    
    def sample(self) -> bool:
        return np.random.random() < self.p
    
    def expectation(self, f: Callable[[bool], float]) -> float:
        return self.p * f(True) + (1 - self.p) * f(False)


class Uniform(Distribution[float]):
    """Distribution uniforme continue."""
    
    def __init__(self, low: float = 0.0, high: float = 1.0):
        assert low < high
        self.low = low
        self.high = high
    
    def sample(self) -> float:
        return np.random.uniform(self.low, self.high)
    
    def expectation(self, f: Callable[[float], float], n_samples: int = 10000) -> float:
        samples = np.random.uniform(self.low, self.high, n_samples)
        return np.mean([f(x) for x in samples])
    
    def mean(self) -> float:
        return (self.low + self.high) / 2
    
    def variance(self) -> float:
        return (self.high - self.low) ** 2 / 12


class Gaussian(Distribution[float]):
    """Distribution normale (gaussienne)."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Args:
            mean: Moyenne μ
            std: Écart-type σ (>0)
        """
        assert std > 0, "std doit être positif"
        self._mean = mean
        self._std = std
    
    def sample(self) -> float:
        return np.random.normal(self._mean, self._std)
    
    def expectation(self, f: Callable[[float], float], n_samples: int = 10000) -> float:
        samples = np.random.normal(self._mean, self._std, n_samples)
        return np.mean([f(x) for x in samples])
    
    def mean(self) -> float:
        return self._mean
    
    def variance(self) -> float:
        return self._std ** 2
    
    def pdf(self, x: float) -> float:
        """Densité de probabilité."""
        return norm.pdf(x, self._mean, self._std)
    
    def cdf(self, x: float) -> float:
        """Fonction de répartition."""
        return norm.cdf(x, self._mean, self._std)
    
    def quantile(self, p: float) -> float:
        """Quantile (inverse de CDF)."""
        return norm.ppf(p, self._mean, self._std)


class LogNormal(Distribution[float]):
    """
    Distribution log-normale.
    Si X ~ LogNormal(μ, σ), alors ln(X) ~ Normal(μ, σ).
    
    Utilisée pour modéliser les prix d'actifs (GBM).
    """
    
    def __init__(self, mu: float, sigma: float):
        """
        Args:
            mu: Moyenne du log
            sigma: Écart-type du log
        """
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma
    
    def sample(self) -> float:
        return np.exp(np.random.normal(self.mu, self.sigma))
    
    def expectation(self, f: Callable[[float], float], n_samples: int = 10000) -> float:
        samples = self.sample_n(n_samples)
        return np.mean([f(x) for x in samples])
    
    def mean(self) -> float:
        return np.exp(self.mu + 0.5 * self.sigma ** 2)
    
    def variance(self) -> float:
        return (np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2)
    
    def median(self) -> float:
        return np.exp(self.mu)


class Exponential(Distribution[float]):
    """Distribution exponentielle."""
    
    def __init__(self, rate: float):
        """
        Args:
            rate: Paramètre λ (rate > 0)
        """
        assert rate > 0
        self.rate = rate
    
    def sample(self) -> float:
        return np.random.exponential(1.0 / self.rate)
    
    def expectation(self, f: Callable[[float], float], n_samples: int = 10000) -> float:
        samples = np.random.exponential(1.0 / self.rate, n_samples)
        return np.mean([f(x) for x in samples])
    
    def mean(self) -> float:
        return 1.0 / self.rate
    
    def variance(self) -> float:
        return 1.0 / (self.rate ** 2)


class Poisson(Distribution[int]):
    """Distribution de Poisson."""
    
    def __init__(self, rate: float):
        """
        Args:
            rate: Paramètre λ (intensité)
        """
        assert rate > 0
        self.rate = rate
    
    def sample(self) -> int:
        return np.random.poisson(self.rate)
    
    def expectation(self, f: Callable[[int], float], n_samples: int = 10000) -> float:
        samples = np.random.poisson(self.rate, n_samples)
        return np.mean([f(x) for x in samples])
    
    def mean(self) -> float:
        return self.rate
    
    def variance(self) -> float:
        return self.rate


class MappedDistribution(Distribution[X]):
    """Distribution résultant de l'application d'une fonction."""
    
    def __init__(self, base: Distribution[A], f: Callable[[A], X]):
        self.base = base
        self.f = f
    
    def sample(self) -> X:
        return self.f(self.base.sample())
    
    def expectation(self, g: Callable[[X], float], n_samples: int = 10000) -> float:
        return np.mean([g(self.f(self.base.sample())) for _ in range(n_samples)])


class FlatMappedDistribution(Distribution[X]):
    """Distribution résultant d'un flatmap (bind monadique)."""
    
    def __init__(self, base: Distribution[A], f: Callable[[A], Distribution[X]]):
        self.base = base
        self.f = f
    
    def sample(self) -> X:
        intermediate = self.base.sample()
        return self.f(intermediate).sample()
    
    def expectation(self, g: Callable[[X], float], n_samples: int = 10000) -> float:
        return np.mean([g(self.sample()) for _ in range(n_samples)])


class Mixture(Distribution[A]):
    """
    Mélange de distributions.
    
    Exemple:
        >>> d = Mixture([
        ...     (0.3, Gaussian(0, 1)),
        ...     (0.7, Gaussian(5, 2))
        ... ])
    """
    
    def __init__(self, components: List[Tuple[float, Distribution[A]]]):
        """
        Args:
            components: Liste de (poids, distribution)
        """
        total_weight = sum(w for w, _ in components)
        self.components = [(w / total_weight, d) for w, d in components]
        self._weights = np.array([w for w, _ in self.components])
    
    def sample(self) -> A:
        idx = np.random.choice(len(self.components), p=self._weights)
        return self.components[idx][1].sample()
    
    def expectation(self, f: Callable[[A], float], n_samples: int = 10000) -> float:
        return sum(w * d.expectation(f, n_samples) for w, d in self.components)


class SampledDistribution(Distribution[A]):
    """
    Distribution définie par une fonction d'échantillonnage.
    Utile pour des distributions complexes.
    """
    
    def __init__(self, sampler: Callable[[], A], expectation_samples: int = 10000):
        self.sampler = sampler
        self.expectation_samples = expectation_samples
    
    def sample(self) -> A:
        return self.sampler()
    
    def expectation(self, f: Callable[[A], float], n_samples: int = None) -> float:
        n = n_samples or self.expectation_samples
        return np.mean([f(self.sample()) for _ in range(n)])


# ============================================
# Distributions multivariées
# ============================================

class MultivariateGaussian(Distribution[np.ndarray]):
    """Distribution normale multivariée."""
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Args:
            mean: Vecteur moyenne (n,)
            cov: Matrice de covariance (n, n)
        """
        self._mean = np.asarray(mean)
        self._cov = np.asarray(cov)
        self._dim = len(mean)
        
        # Décomposition de Cholesky pour échantillonnage
        self._cholesky = np.linalg.cholesky(self._cov)
    
    def sample(self) -> np.ndarray:
        z = np.random.standard_normal(self._dim)
        return self._mean + self._cholesky @ z
    
    def expectation(self, f: Callable[[np.ndarray], float], n_samples: int = 10000) -> float:
        samples = [self.sample() for _ in range(n_samples)]
        return np.mean([f(x) for x in samples])
    
    def mean(self) -> np.ndarray:
        return self._mean.copy()
    
    def covariance(self) -> np.ndarray:
        return self._cov.copy()
    
    @property
    def dim(self) -> int:
        return self._dim


# ============================================
# Utilitaires
# ============================================

def empirical_distribution(samples: Sequence[A]) -> Categorical[A]:
    """
    Crée une distribution empirique à partir d'échantillons.
    """
    counts = defaultdict(int)
    for s in samples:
        counts[s] += 1
    n = len(samples)
    return Categorical({k: v / n for k, v in counts.items()})


def choose(n: int, distribution: Distribution[A]) -> Distribution[List[A]]:
    """
    Tire n échantillons IID d'une distribution.
    Retourne une distribution de listes.
    """
    return SampledDistribution(lambda: distribution.sample_n(n))
```

### markov_process.py

```python
"""
HelixOne - Processus de Markov et Processus de Récompense Markoviens
Source: Stanford CME 241

Fondations pour:
- Modélisation de dynamiques de marché
- Simulation de scénarios
- Base pour MDP et RL
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, Mapping, Callable, Iterator, 
    Tuple, Optional, Sequence, Set, List
)
from dataclasses import dataclass
import numpy as np

from .distributions import Distribution, Categorical, Constant

S = TypeVar('S')  # Type d'état


@dataclass(frozen=True)
class State(Generic[S]):
    """Wrapper pour un état (peut être terminal ou non)."""
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    """État non terminal."""
    pass


@dataclass(frozen=True) 
class Terminal(State[S]):
    """État terminal (absorbant)."""
    pass


@dataclass
class TransitionStep(Generic[S]):
    """Une étape de transition."""
    state: S
    next_state: S
    reward: float = 0.0


class MarkovProcess(ABC, Generic[S]):
    """
    Processus de Markov (chaîne de Markov).
    
    Définit la dynamique de transition entre états
    sans notion de récompense ou d'action.
    """
    
    @abstractmethod
    def transition(self, state: S) -> Distribution[S]:
        """
        Distribution de l'état suivant étant donné l'état actuel.
        
        Args:
            state: État actuel
            
        Returns:
            Distribution sur les états suivants possibles
        """
        pass
    
    def is_terminal(self, state: S) -> bool:
        """Vérifie si l'état est terminal (absorbant)."""
        return False
    
    def simulate(self, start: S, n_steps: int) -> List[S]:
        """
        Simule une trajectoire.
        
        Args:
            start: État initial
            n_steps: Nombre de pas
            
        Returns:
            Liste des états [s_0, s_1, ..., s_n]
        """
        trajectory = [start]
        state = start
        
        for _ in range(n_steps):
            if self.is_terminal(state):
                break
            state = self.transition(state).sample()
            trajectory.append(state)
        
        return trajectory
    
    def simulate_trace(self, start: S) -> Iterator[S]:
        """
        Génère une trace (épisode) jusqu'à un état terminal.
        
        Yields:
            États successifs
        """
        state = start
        yield state
        
        while not self.is_terminal(state):
            state = self.transition(state).sample()
            yield state
    
    def traces(self, start_distribution: Distribution[S]) -> Iterator[List[S]]:
        """
        Génère des traces infinies à partir d'une distribution initiale.
        
        Args:
            start_distribution: Distribution des états initiaux
            
        Yields:
            Traces (listes d'états)
        """
        while True:
            start = start_distribution.sample()
            yield list(self.simulate_trace(start))


class FiniteMarkovProcess(MarkovProcess[S]):
    """
    Processus de Markov avec ensemble d'états fini.
    
    Représenté par une matrice de transition.
    """
    
    def __init__(
        self,
        transition_map: Mapping[S, Distribution[S]],
        terminal_states: Optional[Set[S]] = None
    ):
        """
        Args:
            transition_map: Dict {state: Distribution[next_state]}
            terminal_states: Ensemble des états terminaux
        """
        self.transition_map = dict(transition_map)
        self.terminal_states = terminal_states or set()
        self._states = list(self.transition_map.keys())
    
    def transition(self, state: S) -> Distribution[S]:
        if state in self.terminal_states:
            return Constant(state)
        return self.transition_map[state]
    
    def is_terminal(self, state: S) -> bool:
        return state in self.terminal_states
    
    @property
    def states(self) -> List[S]:
        """Liste de tous les états."""
        return self._states.copy()
    
    @property
    def n_states(self) -> int:
        return len(self._states)
    
    def get_transition_matrix(self) -> Tuple[np.ndarray, List[S]]:
        """
        Retourne la matrice de transition et l'ordre des états.
        
        Returns:
            P: Matrice (n x n) où P[i,j] = P(s_j | s_i)
            states: Liste ordonnée des états
        """
        states = self._states
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        P = np.zeros((n, n))
        
        for s in states:
            i = state_to_idx[s]
            if self.is_terminal(s):
                P[i, i] = 1.0
            else:
                dist = self.transition_map[s]
                if isinstance(dist, Categorical):
                    for s_next, prob in dist.probabilities.items():
                        j = state_to_idx[s_next]
                        P[i, j] = prob
        
        return P, states
    
    def stationary_distribution(self) -> Optional[Distribution[S]]:
        """
        Calcule la distribution stationnaire (si elle existe).
        
        Returns:
            Distribution stationnaire ou None si n'existe pas
        """
        P, states = self.get_transition_matrix()
        n = len(states)
        
        # Trouver le vecteur propre pour valeur propre 1
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        
        # Trouver l'indice de la valeur propre la plus proche de 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        
        if not np.isclose(eigenvalues[idx], 1.0):
            return None
        
        # Extraire et normaliser le vecteur propre
        pi = np.real(eigenvectors[:, idx])
        pi = pi / np.sum(pi)
        
        # Vérifier que c'est une distribution valide
        if np.any(pi < -1e-10):
            return None
        
        pi = np.maximum(pi, 0)
        pi = pi / np.sum(pi)
        
        return Categorical({s: p for s, p in zip(states, pi) if p > 0})


class MarkovRewardProcess(MarkovProcess[S], Generic[S]):
    """
    Processus de Récompense Markovien (MRP).
    
    Ajoute une fonction de récompense au processus de Markov.
    """
    
    @abstractmethod
    def transition_reward(self, state: S) -> Distribution[Tuple[S, float]]:
        """
        Distribution conjointe (état suivant, récompense).
        
        Args:
            state: État actuel
            
        Returns:
            Distribution sur (next_state, reward)
        """
        pass
    
    def transition(self, state: S) -> Distribution[S]:
        """Marginale sur l'état suivant."""
        return self.transition_reward(state).map(lambda x: x[0])
    
    def reward(self, state: S) -> Distribution[float]:
        """Distribution de la récompense."""
        return self.transition_reward(state).map(lambda x: x[1])
    
    def simulate_reward(
        self, 
        start: S, 
        n_steps: int
    ) -> List[TransitionStep[S]]:
        """
        Simule avec les récompenses.
        
        Returns:
            Liste de TransitionStep
        """
        trajectory = []
        state = start
        
        for _ in range(n_steps):
            if self.is_terminal(state):
                break
            
            next_state, reward = self.transition_reward(state).sample()
            trajectory.append(TransitionStep(state, next_state, reward))
            state = next_state
        
        return trajectory
    
    def simulate_trace_reward(
        self, 
        start: S
    ) -> Iterator[Tuple[S, Optional[float]]]:
        """
        Génère une trace avec récompenses.
        
        Yields:
            (state, reward) tuples
        """
        state = start
        yield (state, None)  # Premier état, pas de récompense
        
        while not self.is_terminal(state):
            next_state, reward = self.transition_reward(state).sample()
            yield (next_state, reward)
            state = next_state


class FiniteMarkovRewardProcess(MarkovRewardProcess[S]):
    """
    MRP avec ensemble d'états fini.
    """
    
    def __init__(
        self,
        transition_reward_map: Mapping[S, Distribution[Tuple[S, float]]],
        gamma: float = 1.0,
        terminal_states: Optional[Set[S]] = None
    ):
        """
        Args:
            transition_reward_map: Dict {state: Distribution[(next_state, reward)]}
            gamma: Facteur d'actualisation
            terminal_states: États terminaux
        """
        self.transition_reward_map = dict(transition_reward_map)
        self.gamma = gamma
        self.terminal_states = terminal_states or set()
        self._states = list(self.transition_reward_map.keys())
    
    def transition_reward(self, state: S) -> Distribution[Tuple[S, float]]:
        if state in self.terminal_states:
            return Constant((state, 0.0))
        return self.transition_reward_map[state]
    
    def is_terminal(self, state: S) -> bool:
        return state in self.terminal_states
    
    @property
    def states(self) -> List[S]:
        return self._states.copy()
    
    @property
    def non_terminal_states(self) -> List[S]:
        return [s for s in self._states if s not in self.terminal_states]
    
    def get_value_function_exact(self) -> Mapping[S, float]:
        """
        Calcule la fonction de valeur exacte.
        V = R + γPV  =>  V = (I - γP)^{-1} R
        
        Returns:
            Dict {state: value}
        """
        non_terminal = self.non_terminal_states
        n = len(non_terminal)
        
        if n == 0:
            return {s: 0.0 for s in self._states}
        
        state_to_idx = {s: i for i, s in enumerate(non_terminal)}
        
        # Construire P et R
        P = np.zeros((n, n))
        R = np.zeros(n)
        
        for s in non_terminal:
            i = state_to_idx[s]
            dist = self.transition_reward_map[s]
            
            if isinstance(dist, Categorical):
                for (s_next, reward), prob in dist.probabilities.items():
                    R[i] += prob * reward
                    if s_next in state_to_idx:
                        j = state_to_idx[s_next]
                        P[i, j] += prob
        
        # Résoudre (I - γP)V = R
        I = np.eye(n)
        V_array = np.linalg.solve(I - self.gamma * P, R)
        
        # Construire le résultat
        result = {s: 0.0 for s in self.terminal_states}
        for s, v in zip(non_terminal, V_array):
            result[s] = v
        
        return result
    
    def get_value_function_iterative(
        self, 
        tolerance: float = 1e-6,
        max_iterations: int = 1000
    ) -> Mapping[S, float]:
        """
        Calcule V par itération de Bellman.
        
        V(s) <- R(s) + γ Σ P(s'|s) V(s')
        """
        V = {s: 0.0 for s in self._states}
        
        for _ in range(max_iterations):
            delta = 0.0
            V_new = V.copy()
            
            for s in self.non_terminal_states:
                dist = self.transition_reward_map[s]
                
                # Calculer E[r + γV(s')]
                def bellman_target(transition: Tuple[S, float]) -> float:
                    s_next, reward = transition
                    return reward + self.gamma * V[s_next]
                
                V_new[s] = dist.expectation(bellman_target)
                delta = max(delta, abs(V_new[s] - V[s]))
            
            V = V_new
            
            if delta < tolerance:
                break
        
        return V
```


### mdp.py

```python
"""
HelixOne - Processus de Décision Markoviens (MDP)
Source: Stanford CME 241

Le MDP est le cadre fondamental pour:
- Optimisation de portefeuille
- Exécution optimale
- Trading algorithmique
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, Mapping, Set, Tuple, 
    Callable, Iterator, List, Optional, Dict
)
from dataclasses import dataclass
import numpy as np

from .distributions import Distribution, Categorical, Constant
from .markov_process import MarkovRewardProcess, TransitionStep

S = TypeVar('S')  # State type
A = TypeVar('A')  # Action type


class Policy(ABC, Generic[S, A]):
    """
    Politique: stratégie de décision.
    Mappe états vers distributions d'actions.
    """
    
    @abstractmethod
    def act(self, state: S) -> Distribution[A]:
        """
        Retourne la distribution d'actions pour un état.
        
        Args:
            state: État courant
            
        Returns:
            Distribution sur les actions
        """
        pass
    
    def __call__(self, state: S) -> Distribution[A]:
        return self.act(state)
    
    def action(self, state: S) -> A:
        """Échantillonne une action."""
        return self.act(state).sample()


class DeterministicPolicy(Policy[S, A]):
    """Politique déterministe: π(s) = a."""
    
    def __init__(self, policy_map: Mapping[S, A]):
        self.policy_map = dict(policy_map)
    
    def act(self, state: S) -> Distribution[A]:
        return Constant(self.policy_map[state])
    
    def __getitem__(self, state: S) -> A:
        return self.policy_map[state]
    
    def __setitem__(self, state: S, action: A):
        self.policy_map[state] = action


class StochasticPolicy(Policy[S, A]):
    """Politique stochastique: π(a|s)."""
    
    def __init__(self, policy_map: Mapping[S, Distribution[A]]):
        self.policy_map = dict(policy_map)
    
    def act(self, state: S) -> Distribution[A]:
        return self.policy_map[state]
    
    @staticmethod
    def from_action_probabilities(
        action_probs: Mapping[S, Mapping[A, float]]
    ) -> StochasticPolicy[S, A]:
        """Crée depuis un dict de probabilités."""
        return StochasticPolicy({
            s: Categorical(probs) for s, probs in action_probs.items()
        })


class UniformPolicy(Policy[S, A]):
    """Politique uniforme sur les actions disponibles."""
    
    def __init__(self, action_space: Callable[[S], Set[A]]):
        self.action_space = action_space
    
    def act(self, state: S) -> Distribution[A]:
        actions = list(self.action_space(state))
        return Categorical({a: 1.0 / len(actions) for a in actions})


class EpsilonGreedyPolicy(Policy[S, A]):
    """
    Politique ε-greedy basée sur Q-values.
    
    Avec probabilité ε: action aléatoire
    Sinon: action gloutonne (argmax Q)
    """
    
    def __init__(
        self,
        q_values: Callable[[S, A], float],
        action_space: Callable[[S], Set[A]],
        epsilon: float = 0.1
    ):
        self.q_values = q_values
        self.action_space = action_space
        self.epsilon = epsilon
    
    def act(self, state: S) -> Distribution[A]:
        actions = list(self.action_space(state))
        n = len(actions)
        
        # Trouver action gloutonne
        q_vals = {a: self.q_values(state, a) for a in actions}
        best_action = max(q_vals, key=q_vals.get)
        
        # Distribution ε-greedy
        probs = {}
        for a in actions:
            if a == best_action:
                probs[a] = 1 - self.epsilon + self.epsilon / n
            else:
                probs[a] = self.epsilon / n
        
        return Categorical(probs)


class SoftmaxPolicy(Policy[S, A]):
    """
    Politique softmax (Boltzmann).
    π(a|s) ∝ exp(Q(s,a) / τ)
    """
    
    def __init__(
        self,
        q_values: Callable[[S, A], float],
        action_space: Callable[[S], Set[A]],
        temperature: float = 1.0
    ):
        self.q_values = q_values
        self.action_space = action_space
        self.temperature = temperature
    
    def act(self, state: S) -> Distribution[A]:
        actions = list(self.action_space(state))
        q_vals = np.array([self.q_values(state, a) for a in actions])
        
        # Softmax avec stabilité numérique
        q_vals = q_vals - np.max(q_vals)
        exp_q = np.exp(q_vals / self.temperature)
        probs = exp_q / np.sum(exp_q)
        
        return Categorical({a: p for a, p in zip(actions, probs)})


class MDP(ABC, Generic[S, A]):
    """
    Processus de Décision Markovien.
    
    Définit:
    - Espace d'états S
    - Espace d'actions A
    - Dynamique de transition P(s'|s,a)
    - Fonction de récompense R(s,a)
    - Facteur d'actualisation γ
    """
    
    @abstractmethod
    def actions(self, state: S) -> Set[A]:
        """Actions disponibles dans un état."""
        pass
    
    @abstractmethod
    def transition(self, state: S, action: A) -> Distribution[Tuple[S, float]]:
        """
        Distribution de transition (état suivant, récompense).
        
        Returns:
            Distribution sur (next_state, reward)
        """
        pass
    
    @property
    @abstractmethod
    def gamma(self) -> float:
        """Facteur d'actualisation."""
        pass
    
    def is_terminal(self, state: S) -> bool:
        """Vérifie si l'état est terminal."""
        return len(self.actions(state)) == 0
    
    def step(self, state: S, action: A) -> Tuple[S, float, bool]:
        """
        Effectue un pas de simulation.
        
        Returns:
            (next_state, reward, done)
        """
        next_state, reward = self.transition(state, action).sample()
        done = self.is_terminal(next_state)
        return next_state, reward, done
    
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        """
        Applique une politique pour obtenir un MRP.
        
        MDP + Policy = MRP
        """
        return PolicyMRP(self, policy)


class PolicyMRP(MarkovRewardProcess[S]):
    """MRP induit par un MDP et une politique."""
    
    def __init__(self, mdp: MDP[S, A], policy: Policy[S, A]):
        self.mdp = mdp
        self.policy = policy
    
    def transition_reward(self, state: S) -> Distribution[Tuple[S, float]]:
        """Marginalise sur les actions."""
        action_dist = self.policy.act(state)
        
        def sample_transition() -> Tuple[S, float]:
            action = action_dist.sample()
            return self.mdp.transition(state, action).sample()
        
        from .distributions import SampledDistribution
        return SampledDistribution(sample_transition)
    
    def is_terminal(self, state: S) -> bool:
        return self.mdp.is_terminal(state)


class FiniteMDP(MDP[S, A]):
    """
    MDP avec espaces d'états et d'actions finis.
    """
    
    def __init__(
        self,
        transition_map: Mapping[S, Mapping[A, Distribution[Tuple[S, float]]]],
        gamma: float = 0.99,
        terminal_states: Optional[Set[S]] = None
    ):
        """
        Args:
            transition_map: Dict {s: {a: Distribution[(s', r)]}}
            gamma: Facteur d'actualisation
            terminal_states: États terminaux
        """
        self.transition_map = {
            s: dict(a_map) for s, a_map in transition_map.items()
        }
        self._gamma = gamma
        self.terminal_states = terminal_states or set()
        
        # Extraire tous les états et actions
        self._states = list(self.transition_map.keys())
        self._actions = set()
        for s_map in self.transition_map.values():
            self._actions.update(s_map.keys())
        self._actions = list(self._actions)
    
    def actions(self, state: S) -> Set[A]:
        if state in self.terminal_states:
            return set()
        return set(self.transition_map.get(state, {}).keys())
    
    def transition(self, state: S, action: A) -> Distribution[Tuple[S, float]]:
        return self.transition_map[state][action]
    
    @property
    def gamma(self) -> float:
        return self._gamma
    
    def is_terminal(self, state: S) -> bool:
        return state in self.terminal_states
    
    @property
    def states(self) -> List[S]:
        return self._states.copy()
    
    @property
    def all_actions(self) -> List[A]:
        return self._actions.copy()
    
    @property
    def n_states(self) -> int:
        return len(self._states)
    
    @property
    def n_actions(self) -> int:
        return len(self._actions)
    
    @property
    def non_terminal_states(self) -> List[S]:
        return [s for s in self._states if s not in self.terminal_states]


class TabularMDP(MDP[int, int]):
    """
    MDP tabulaire avec états et actions indexés par entiers.
    Optimisé pour les algorithmes de DP.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        transitions: np.ndarray,  # (S, A, S) -> prob
        rewards: np.ndarray,      # (S, A) -> reward
        gamma: float = 0.99,
        terminal_states: Optional[Set[int]] = None
    ):
        """
        Args:
            n_states: Nombre d'états
            n_actions: Nombre d'actions
            transitions: P[s, a, s'] = P(s' | s, a)
            rewards: R[s, a] = récompense espérée
            gamma: Facteur d'actualisation
        """
        self._n_states = n_states
        self._n_actions = n_actions
        self.P = transitions
        self.R = rewards
        self._gamma = gamma
        self.terminal_states = terminal_states or set()
        
        # Vérifications
        assert transitions.shape == (n_states, n_actions, n_states)
        assert rewards.shape == (n_states, n_actions)
        assert np.allclose(transitions.sum(axis=2), 1.0)
    
    def actions(self, state: int) -> Set[int]:
        if state in self.terminal_states:
            return set()
        return set(range(self._n_actions))
    
    def transition(self, state: int, action: int) -> Distribution[Tuple[int, float]]:
        probs = self.P[state, action]
        reward = self.R[state, action]
        
        outcomes = {}
        for s_next in range(self._n_states):
            if probs[s_next] > 0:
                outcomes[(s_next, reward)] = probs[s_next]
        
        return Categorical(outcomes)
    
    @property
    def gamma(self) -> float:
        return self._gamma
    
    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    @property
    def n_actions(self) -> int:
        return self._n_actions
    
    @property
    def states(self) -> List[int]:
        return list(range(self._n_states))
    
    @property
    def non_terminal_states(self) -> List[int]:
        return [s for s in range(self._n_states) if s not in self.terminal_states]
    
    def get_q_value(self, V: np.ndarray, state: int, action: int) -> float:
        """Calcule Q(s, a) = R(s,a) + γ Σ P(s'|s,a) V(s')."""
        return self.R[state, action] + self._gamma * np.dot(self.P[state, action], V)
    
    def get_all_q_values(self, V: np.ndarray) -> np.ndarray:
        """Calcule Q pour tous les (s, a)."""
        # Q[s, a] = R[s, a] + γ Σ P[s, a, s'] V[s']
        return self.R + self._gamma * np.einsum('ijk,k->ij', self.P, V)
    
    def get_greedy_policy(self, V: np.ndarray) -> np.ndarray:
        """
        Retourne la politique gloutonne.
        
        Returns:
            policy[s] = argmax_a Q(s, a)
        """
        Q = self.get_all_q_values(V)
        return np.argmax(Q, axis=1)
    
    def get_greedy_policy_matrix(self, V: np.ndarray) -> np.ndarray:
        """
        Retourne la politique gloutonne sous forme matricielle.
        
        Returns:
            policy[s, a] = 1 si a = argmax, 0 sinon
        """
        Q = self.get_all_q_values(V)
        policy = np.zeros((self._n_states, self._n_actions))
        best_actions = np.argmax(Q, axis=1)
        policy[np.arange(self._n_states), best_actions] = 1.0
        return policy
    
    def get_policy_transition_matrix(self, policy: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de transition induite par une politique.
        
        Args:
            policy: Soit policy[s] = a (déterministe)
                   Soit policy[s, a] = π(a|s) (stochastique)
        
        Returns:
            P_π[s, s'] = Σ_a π(a|s) P(s'|s, a)
        """
        if policy.ndim == 1:
            # Politique déterministe
            P_pi = np.zeros((self._n_states, self._n_states))
            for s in range(self._n_states):
                P_pi[s] = self.P[s, policy[s]]
        else:
            # Politique stochastique
            P_pi = np.einsum('sa,sas->ss', policy, self.P)
        
        return P_pi
    
    def get_policy_reward_vector(self, policy: np.ndarray) -> np.ndarray:
        """
        Calcule le vecteur de récompense induit par une politique.
        
        Returns:
            R_π[s] = Σ_a π(a|s) R(s, a)
        """
        if policy.ndim == 1:
            return self.R[np.arange(self._n_states), policy]
        else:
            return np.einsum('sa,sa->s', policy, self.R)


# ============================================
# Value Functions
# ============================================

@dataclass
class ValueFunctions(Generic[S, A]):
    """Conteneur pour fonctions de valeur."""
    v: Dict[S, float]  # V(s)
    q: Dict[Tuple[S, A], float]  # Q(s, a)


def compute_v_from_q(
    q: Mapping[Tuple[S, A], float],
    policy: Policy[S, A],
    states: List[S]
) -> Dict[S, float]:
    """Calcule V à partir de Q et π."""
    v = {}
    for s in states:
        action_dist = policy.act(s)
        if isinstance(action_dist, Categorical):
            v[s] = sum(
                p * q.get((s, a), 0.0)
                for a, p in action_dist.probabilities.items()
            )
        else:
            # Approximation Monte Carlo
            v[s] = np.mean([q.get((s, action_dist.sample()), 0.0) for _ in range(100)])
    return v


def compute_q_from_v(
    v: Mapping[S, float],
    mdp: FiniteMDP[S, A]
) -> Dict[Tuple[S, A], float]:
    """Calcule Q à partir de V."""
    q = {}
    for s in mdp.states:
        for a in mdp.actions(s):
            # Q(s,a) = E[r + γV(s')]
            dist = mdp.transition(s, a)
            
            def q_target(transition: Tuple[S, float]) -> float:
                s_next, reward = transition
                return reward + mdp.gamma * v.get(s_next, 0.0)
            
            q[(s, a)] = dist.expectation(q_target)
    return q
```

---

## PARTIE A.2 : ALGORITHMES RL

### dp.py (Programmation Dynamique)

```python
"""
HelixOne - Algorithmes de Programmation Dynamique
Source: Stanford CME 241 - Chapter 5

Algorithmes pour MDP avec modèle connu:
- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Value Iteration
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from ..core.mdp import TabularMDP, FiniteMDP, Policy, DeterministicPolicy


def policy_evaluation_tabular(
    mdp: TabularMDP,
    policy: np.ndarray,
    tolerance: float = 1e-6,
    max_iterations: int = 1000
) -> np.ndarray:
    """
    Évaluation de politique itérative (tabulaire).
    
    V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
    
    Args:
        mdp: MDP tabulaire
        policy: policy[s] = a (déterministe) ou policy[s,a] = π(a|s)
        tolerance: Critère de convergence
        max_iterations: Nombre max d'itérations
    
    Returns:
        V: Fonction de valeur (n_states,)
    """
    n_s = mdp.n_states
    V = np.zeros(n_s)
    
    # Convertir politique si déterministe
    if policy.ndim == 1:
        pi_matrix = np.zeros((n_s, mdp.n_actions))
        pi_matrix[np.arange(n_s), policy.astype(int)] = 1.0
    else:
        pi_matrix = policy
    
    # Matrices de transition et récompense sous π
    P_pi = mdp.get_policy_transition_matrix(pi_matrix)
    R_pi = mdp.get_policy_reward_vector(pi_matrix)
    
    for iteration in range(max_iterations):
        V_new = R_pi + mdp.gamma * P_pi @ V
        
        delta = np.max(np.abs(V_new - V))
        V = V_new
        
        if delta < tolerance:
            break
    
    return V


def policy_evaluation_exact(
    mdp: TabularMDP,
    policy: np.ndarray
) -> np.ndarray:
    """
    Évaluation de politique exacte (résolution matricielle).
    
    V = (I - γP^π)^{-1} R^π
    """
    n_s = mdp.n_states
    
    if policy.ndim == 1:
        pi_matrix = np.zeros((n_s, mdp.n_actions))
        pi_matrix[np.arange(n_s), policy.astype(int)] = 1.0
    else:
        pi_matrix = policy
    
    P_pi = mdp.get_policy_transition_matrix(pi_matrix)
    R_pi = mdp.get_policy_reward_vector(pi_matrix)
    
    I = np.eye(n_s)
    V = np.linalg.solve(I - mdp.gamma * P_pi, R_pi)
    
    return V


def policy_improvement(mdp: TabularMDP, V: np.ndarray) -> np.ndarray:
    """
    Amélioration de politique gloutonne.
    
    π'(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
    
    Returns:
        policy: Nouvelle politique (déterministe)
    """
    return mdp.get_greedy_policy(V)


def policy_iteration(
    mdp: TabularMDP,
    initial_policy: Optional[np.ndarray] = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    use_exact_evaluation: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Itération de politique.
    
    1. Évaluer la politique courante
    2. Améliorer la politique
    3. Répéter jusqu'à convergence
    
    Returns:
        policy: Politique optimale
        V: Fonction de valeur optimale
        iterations: Nombre d'itérations
    """
    n_s = mdp.n_states
    
    # Initialisation
    if initial_policy is None:
        policy = np.zeros(n_s, dtype=int)
    else:
        policy = initial_policy.copy()
    
    for iteration in range(max_iterations):
        # Évaluation
        if use_exact_evaluation:
            V = policy_evaluation_exact(mdp, policy)
        else:
            V = policy_evaluation_tabular(mdp, policy, tolerance)
        
        # Amélioration
        new_policy = policy_improvement(mdp, V)
        
        # Vérifier convergence
        if np.array_equal(new_policy, policy):
            return policy, V, iteration + 1
        
        policy = new_policy
    
    return policy, V, max_iterations


def value_iteration(
    mdp: TabularMDP,
    tolerance: float = 1e-6,
    max_iterations: int = 1000
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Itération de valeur.
    
    V(s) <- max_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
    
    Returns:
        policy: Politique optimale
        V: Fonction de valeur optimale
        iterations: Nombre d'itérations
    """
    n_s = mdp.n_states
    V = np.zeros(n_s)
    
    for iteration in range(max_iterations):
        # Backup de Bellman optimal
        Q = mdp.get_all_q_values(V)
        V_new = np.max(Q, axis=1)
        
        # Vérifier convergence
        delta = np.max(np.abs(V_new - V))
        V = V_new
        
        if delta < tolerance:
            break
    
    # Extraire politique
    policy = mdp.get_greedy_policy(V)
    
    return policy, V, iteration + 1


def backward_induction(
    mdp: TabularMDP,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Induction rétrograde pour horizon fini.
    
    V_t(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V_{t+1}(s')]
    
    Args:
        mdp: MDP
        horizon: Nombre de pas de temps
    
    Returns:
        policy: policy[t, s] = action optimale au temps t, état s
        V: V[t, s] = valeur optimale
    """
    n_s = mdp.n_states
    
    # Initialisation
    V = np.zeros((horizon + 1, n_s))
    policy = np.zeros((horizon, n_s), dtype=int)
    
    # V_T = 0 (condition terminale, déjà fait)
    
    # Rétropropagation
    for t in range(horizon - 1, -1, -1):
        Q = mdp.R + mdp.gamma * np.einsum('ijk,k->ij', mdp.P, V[t + 1])
        V[t] = np.max(Q, axis=1)
        policy[t] = np.argmax(Q, axis=1)
    
    return policy, V


def prioritized_sweeping(
    mdp: TabularMDP,
    tolerance: float = 1e-6,
    max_iterations: int = 10000,
    theta: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prioritized Sweeping - met à jour les états par ordre de priorité.
    Plus efficace que value iteration classique.
    """
    import heapq
    
    n_s = mdp.n_states
    V = np.zeros(n_s)
    
    # File de priorité (max-heap via négation)
    pq = []
    
    # Initialiser avec tous les états
    for s in range(n_s):
        Q = mdp.get_all_q_values(V)[s]
        priority = np.max(Q) - V[s]
        if abs(priority) > theta:
            heapq.heappush(pq, (-abs(priority), s))
    
    for _ in range(max_iterations):
        if not pq:
            break
        
        _, s = heapq.heappop(pq)
        
        # Mise à jour
        Q = mdp.get_all_q_values(V)[s]
        old_v = V[s]
        V[s] = np.max(Q)
        
        if abs(V[s] - old_v) < tolerance:
            continue
        
        # Ajouter les prédécesseurs
        for s_pred in range(n_s):
            for a in range(mdp.n_actions):
                if mdp.P[s_pred, a, s] > 0:
                    Q_pred = mdp.get_all_q_values(V)[s_pred]
                    priority = np.max(Q_pred) - V[s_pred]
                    if abs(priority) > theta:
                        heapq.heappush(pq, (-abs(priority), s_pred))
    
    policy = mdp.get_greedy_policy(V)
    return policy, V
```

### td_learning.py (Temporal Difference)

```python
"""
HelixOne - Algorithmes TD Learning
Source: Stanford CME 241 - Chapter 11

Algorithmes de prédiction sans modèle:
- TD(0)
- TD(λ) avec traces d'éligibilité
- n-step TD
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from collections import defaultdict


class TDPrediction:
    """
    TD(0) pour estimation de V^π.
    
    V(s) <- V(s) + α[r + γV(s') - V(s)]
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 0.1,
        initial_value: float = 0.0
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.initial_value = initial_value
        self.V: Dict = defaultdict(lambda: initial_value)
    
    def update(self, state, reward: float, next_state, done: bool):
        """
        Mise à jour TD(0).
        
        Args:
            state: État courant
            reward: Récompense reçue
            next_state: État suivant
            done: Episode terminé?
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.V[next_state]
        
        td_error = target - self.V[state]
        self.V[state] += self.alpha * td_error
        
        return td_error
    
    def get_value(self, state) -> float:
        return self.V[state]


class TDLambda:
    """
    TD(λ) avec traces d'éligibilité.
    
    Interpole entre TD(0) (λ=0) et Monte Carlo (λ=1).
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 0.1,
        lambda_: float = 0.9,
        initial_value: float = 0.0,
        trace_type: str = 'accumulating'  # ou 'replacing'
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.initial_value = initial_value
        self.trace_type = trace_type
        
        self.V: Dict = defaultdict(lambda: initial_value)
        self.eligibility: Dict = defaultdict(float)
    
    def start_episode(self):
        """Réinitialise les traces au début d'un épisode."""
        self.eligibility.clear()
    
    def update(self, state, reward: float, next_state, done: bool):
        """
        Mise à jour TD(λ).
        """
        # Erreur TD
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.V[next_state]
        
        td_error = target - self.V[state]
        
        # Mise à jour trace d'éligibilité
        if self.trace_type == 'accumulating':
            self.eligibility[state] += 1
        else:  # replacing
            self.eligibility[state] = 1
        
        # Mise à jour de toutes les valeurs
        for s in list(self.eligibility.keys()):
            self.V[s] += self.alpha * td_error * self.eligibility[s]
            self.eligibility[s] *= self.gamma * self.lambda_
            
            # Nettoyer les traces négligeables
            if self.eligibility[s] < 1e-10:
                del self.eligibility[s]
        
        return td_error
    
    def get_value(self, state) -> float:
        return self.V[state]


class NStepTD:
    """
    n-step TD.
    
    G_t:t+n = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(S_{t+n})
    """
    
    def __init__(
        self,
        n: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        initial_value: float = 0.0
    ):
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.initial_value = initial_value
        
        self.V: Dict = defaultdict(lambda: initial_value)
        
        # Buffer pour stocker les transitions
        self.states: List = []
        self.rewards: List = []
    
    def start_episode(self):
        """Réinitialise le buffer."""
        self.states.clear()
        self.rewards.clear()
    
    def update(self, state, reward: float, next_state, done: bool):
        """
        Stocke la transition et effectue les mises à jour nécessaires.
        """
        self.states.append(state)
        self.rewards.append(reward)
        
        if done:
            self.states.append(next_state)
        
        # Mettre à jour si on a assez de steps
        T = len(self.rewards)
        
        for t in range(max(0, T - self.n), T if done else T - self.n + 1):
            # Calculer le n-step return
            G = 0.0
            for i in range(min(self.n, T - t)):
                G += (self.gamma ** i) * self.rewards[t + i]
            
            # Ajouter bootstrap si pas à la fin
            if t + self.n < T or not done:
                bootstrap_state = self.states[min(t + self.n, len(self.states) - 1)]
                G += (self.gamma ** self.n) * self.V[bootstrap_state]
            
            # Mise à jour
            state_t = self.states[t]
            self.V[state_t] += self.alpha * (G - self.V[state_t])
    
    def get_value(self, state) -> float:
        return self.V[state]


class MonteCarloPrediction:
    """
    Monte Carlo pour estimation de V^π.
    
    V(s) <- V(s) + α[G - V(s)]
    
    où G est le return complet de l'épisode.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 0.1,
        first_visit: bool = True
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.first_visit = first_visit
        
        self.V: Dict = defaultdict(float)
        self.returns: Dict = defaultdict(list)  # Pour moyenne incrémentale
        self.visit_count: Dict = defaultdict(int)
    
    def update_episode(self, episode: List[Tuple]):
        """
        Met à jour après un épisode complet.
        
        Args:
            episode: Liste de (state, action, reward)
        """
        G = 0.0
        visited = set()
        
        # Parcourir l'épisode à l'envers
        for t in range(len(episode) - 1, -1, -1):
            state, _, reward = episode[t]
            G = reward + self.gamma * G
            
            if self.first_visit and state in visited:
                continue
            
            visited.add(state)
            
            # Mise à jour incrémentale
            self.visit_count[state] += 1
            alpha = 1.0 / self.visit_count[state] if self.alpha is None else self.alpha
            self.V[state] += alpha * (G - self.V[state])
    
    def get_value(self, state) -> float:
        return self.V[state]
```

 self.gamma * next_q
            
            # Loss et mise à jour
            loss = F.mse_loss(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
            self.optimizer.step()
            
            # Mise à jour réseau cible
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            return loss.item()
        
        def decay_epsilon(self):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        def save(self, path: str):
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']


    class DuelingDQN(nn.Module):
        """
        Dueling DQN Architecture.
        
        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        """
        
        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dim: int = 64
        ):
            super().__init__()
            
            # Couche partagée
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Branche Value
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Branche Advantage
            self.advantage = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )
        
        def forward(self, x):
            features = self.feature(x)
            value = self.value(features)
            advantage = self.advantage(features)
            
            # Combiner V et A
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q

except ImportError:
    print("PyTorch non disponible. DQN désactivé.")
```

### policy_gradient.py

```python
"""
HelixOne - Policy Gradient Algorithms
Source: Stanford CME 241 - Chapter 14

Algorithmes:
- REINFORCE
- Actor-Critic
- A2C
- PPO
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal
    
    
    class PolicyNetwork(nn.Module):
        """Réseau de politique pour actions discrètes."""
        
        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dims: List[int] = [64, 64]
        ):
            super().__init__()
            
            layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, n_actions))
            layers.append(nn.Softmax(dim=-1))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
        
        def get_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.forward(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)
        
        def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            probs = self.forward(state)
            dist = Categorical(probs)
            return dist.log_prob(action)
    
    
    class ValueNetwork(nn.Module):
        """Réseau de valeur (Critic)."""
        
        def __init__(
            self,
            state_dim: int,
            hidden_dims: List[int] = [64, 64]
        ):
            super().__init__()
            
            layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    
    class GaussianPolicy(nn.Module):
        """Politique gaussienne pour actions continues."""
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int] = [64, 64],
            log_std_min: float = -20,
            log_std_max: float = 2
        ):
            super().__init__()
            
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max
            
            layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            
            self.features = nn.Sequential(*layers)
            self.mean_head = nn.Linear(prev_dim, action_dim)
            self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        def forward(self, x):
            features = self.features(x)
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
        
        def get_action(self, state: np.ndarray, deterministic: bool = False):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean, log_std = self.forward(state_tensor)
            std = log_std.exp()
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.rsample()
            
            log_prob = Normal(mean, std).log_prob(action).sum(-1)
            return action.squeeze().detach().numpy(), log_prob
    
    
    class REINFORCE:
        """
        REINFORCE (Monte Carlo Policy Gradient).
        
        ∇J(θ) = E[Σ_t ∇log π(a_t|s_t) * G_t]
        """
        
        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dims: List[int] = [64, 64],
            learning_rate: float = 0.001,
            gamma: float = 0.99,
            baseline: bool = True  # Use baseline to reduce variance
        ):
            self.gamma = gamma
            self.baseline = baseline
            
            self.policy = PolicyNetwork(state_dim, n_actions, hidden_dims)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
            
            if baseline:
                self.value_net = ValueNetwork(state_dim, hidden_dims)
                self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
            return self.policy.get_action(state)
        
        def compute_returns(self, rewards: List[float]) -> torch.Tensor:
            """Calcule les returns G_t."""
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            return torch.FloatTensor(returns)
        
        def update(
            self,
            states: List[np.ndarray],
            actions: List[int],
            rewards: List[float],
            log_probs: List[torch.Tensor]
        ) -> Tuple[float, float]:
            """
            Mise à jour après un épisode.
            
            Returns:
                policy_loss, value_loss
            """
            returns = self.compute_returns(rewards)
            states_tensor = torch.FloatTensor(np.array(states))
            
            # Normaliser les returns
            returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            if self.baseline:
                # Mettre à jour le baseline (value network)
                values = self.value_net(states_tensor).squeeze()
                value_loss = F.mse_loss(values, returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                # Avantages
                with torch.no_grad():
                    values = self.value_net(states_tensor).squeeze()
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = returns_normalized
                value_loss = torch.tensor(0.0)
            
            # Policy loss
            log_probs_tensor = torch.stack(log_probs)
            policy_loss = -(log_probs_tensor * advantages.detach()).mean()
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            return policy_loss.item(), value_loss.item() if self.baseline else 0.0
    
    
    class ActorCritic:
        """
        Actor-Critic avec mise à jour TD.
        
        Réduit la variance par rapport à REINFORCE.
        """
        
        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dims: List[int] = [64, 64],
            actor_lr: float = 0.001,
            critic_lr: float = 0.001,
            gamma: float = 0.99
        ):
            self.gamma = gamma
            
            self.actor = PolicyNetwork(state_dim, n_actions, hidden_dims)
            self.critic = ValueNetwork(state_dim, hidden_dims)
            
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
            return self.actor.get_action(state)
        
        def update(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            log_prob: torch.Tensor
        ) -> Tuple[float, float]:
            """Mise à jour à chaque pas."""
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Valeurs
            value = self.critic(state_tensor)
            next_value = self.critic(next_state_tensor) if not done else torch.tensor([[0.0]])
            
            # TD target et advantage
            td_target = reward + self.gamma * next_value.detach()
            advantage = td_target - value.detach()
            
            # Critic loss
            critic_loss = F.mse_loss(value, td_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Actor loss
            actor_loss = -(log_prob * advantage)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            return actor_loss.item(), critic_loss.item()
    
    
    class A2C:
        """
        Advantage Actor-Critic (A2C).
        
        Avec entropy bonus pour l'exploration.
        """
        
        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dims: List[int] = [64, 64],
            learning_rate: float = 0.001,
            gamma: float = 0.99,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5
        ):
            self.gamma = gamma
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
            self.max_grad_norm = max_grad_norm
            
            self.actor = PolicyNetwork(state_dim, n_actions, hidden_dims)
            self.critic = ValueNetwork(state_dim, hidden_dims)
            
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=learning_rate
            )
        
        def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            dist = Categorical(probs)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action), dist.entropy()
        
        def update(
            self,
            states: List[np.ndarray],
            actions: List[int],
            rewards: List[float],
            dones: List[bool],
            log_probs: List[torch.Tensor],
            entropies: List[torch.Tensor],
            next_state: np.ndarray
        ) -> Tuple[float, float, float]:
            """Mise à jour après n steps."""
            states_tensor = torch.FloatTensor(np.array(states))
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Calculer les returns avec bootstrapping
            values = self.critic(states_tensor).squeeze()
            next_value = self.critic(next_state_tensor).squeeze().detach()
            
            returns = []
            R = next_value if not dones[-1] else 0
            for r, done in zip(reversed(rewards), reversed(dones)):
                R = r + self.gamma * R * (1 - done)
                returns.insert(0, R)
            returns = torch.FloatTensor(returns)
            
            # Advantages
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Losses
            log_probs_tensor = torch.stack(log_probs)
            entropies_tensor = torch.stack(entropies)
            
            actor_loss = -(log_probs_tensor * advantages).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy_loss = -entropies_tensor.mean()
            
            total_loss = (
                actor_loss +
                self.value_coef * critic_loss +
                self.entropy_coef * entropy_loss
            )
            
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
            
            return actor_loss.item(), critic_loss.item(), entropy_loss.item()
    
    
    class PPO:
        """
        Proximal Policy Optimization (PPO).
        
        Plus stable que les méthodes policy gradient classiques.
        """
        
        def __init__(
            self,
            state_dim: int,
            n_actions: int,
            hidden_dims: List[int] = [64, 64],
            learning_rate: float = 0.0003,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            n_epochs: int = 10,
            batch_size: int = 64
        ):
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_epsilon = clip_epsilon
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
            self.max_grad_norm = max_grad_norm
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            
            self.actor = PolicyNetwork(state_dim, n_actions, hidden_dims)
            self.critic = ValueNetwork(state_dim, hidden_dims)
            
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=learning_rate
            )
        
        def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            dist = Categorical(probs)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action), value.squeeze()
        
        def compute_gae(
            self,
            rewards: List[float],
            values: List[torch.Tensor],
            dones: List[bool],
            next_value: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Calcule les GAE (Generalized Advantage Estimation)."""
            advantages = []
            returns = []
            gae = 0
            
            values = [v.item() for v in values] + [next_value.item()]
            
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            return torch.FloatTensor(advantages), torch.FloatTensor(returns)
        
        def update(
            self,
            states: List[np.ndarray],
            actions: List[int],
            old_log_probs: List[torch.Tensor],
            rewards: List[float],
            dones: List[bool],
            values: List[torch.Tensor],
            next_state: np.ndarray
        ) -> Tuple[float, float, float]:
            """Mise à jour PPO."""
            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.LongTensor(actions)
            old_log_probs_tensor = torch.stack(old_log_probs).detach()
            
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).squeeze()
            
            # Calculer GAE
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Mini-batch updates
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0
            n_updates = 0
            
            for _ in range(self.n_epochs):
                indices = np.random.permutation(len(states))
                
                for start in range(0, len(states), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    batch_states = states_tensor[batch_indices]
                    batch_actions = actions_tensor[batch_indices]
                    batch_old_log_probs = old_log_probs_tensor[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    
                    # Nouvelles probabilités
                    probs = self.actor(batch_states)
                    dist = Categorical(probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # PPO clipped objective
                    ratio = (new_log_probs - batch_old_log_probs).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    values_pred = self.critic(batch_states).squeeze()
                    critic_loss = F.mse_loss(values_pred, batch_returns)
                    
                    # Total loss
                    loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy += entropy.item()
                    n_updates += 1
            
            return (
                total_actor_loss / n_updates,
                total_critic_loss / n_updates,
                total_entropy / n_updates
            )

except ImportError:
    print("PyTorch non disponible. Policy Gradient désactivé.")
```

---

# PARTIE B : RISK MANAGEMENT

---

## PARTIE B.1 : UTILITY & RISK METRICS

### utility.py

```python
"""
HelixOne - Fonctions d'utilité et métriques de risque
Source: Stanford CME 241 - Chapter 7

Fonctions d'utilité:
- CARA (Constant Absolute Risk Aversion)
- CRRA (Constant Relative Risk Aversion)
- Quadratique

Métriques:
- VaR (Value at Risk)
- CVaR / ES (Expected Shortfall)
- Sharpe, Sortino, Calmar
- Maximum Drawdown
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from typing import Union, Optional, Callable
from scipy.optimize import brentq


class UtilityFunction(ABC):
    """Classe de base pour fonctions d'utilité."""
    
    @abstractmethod
    def __call__(self, wealth: float) -> float:
        """Calcule l'utilité."""
        pass
    
    @abstractmethod
    def derivative(self, wealth: float) -> float:
        """Utilité marginale U'(w)."""
        pass
    
    @abstractmethod
    def second_derivative(self, wealth: float) -> float:
        """U''(w)."""
        pass
    
    def absolute_risk_aversion(self, wealth: float) -> float:
        """
        Coefficient d'aversion absolue au risque.
        A(w) = -U''(w) / U'(w)
        """
        return -self.second_derivative(wealth) / self.derivative(wealth)
    
    def relative_risk_aversion(self, wealth: float) -> float:
        """
        Coefficient d'aversion relative au risque.
        R(w) = w * A(w) = -w * U''(w) / U'(w)
        """
        return wealth * self.absolute_risk_aversion(wealth)
    
    def certainty_equivalent(
        self,
        wealth_samples: np.ndarray
    ) -> float:
        """
        Équivalent certain.
        CE tel que U(CE) = E[U(W)]
        """
        expected_utility = np.mean([self(w) for w in wealth_samples])
        return self.inverse(expected_utility)
    
    @abstractmethod
    def inverse(self, utility: float) -> float:
        """U^{-1}(u)."""
        pass
    
    def risk_premium(self, wealth_samples: np.ndarray) -> float:
        """Prime de risque π = E[W] - CE."""
        return np.mean(wealth_samples) - self.certainty_equivalent(wealth_samples)


class CARA(UtilityFunction):
    """
    Constant Absolute Risk Aversion.
    
    U(w) = -exp(-α * w)
    
    Propriétés:
    - A(w) = α (constant)
    - Indépendant de la richesse initiale
    - Pour W ~ N(μ, σ²): CE = μ - ασ²/2
    """
    
    def __init__(self, alpha: float):
        """
        Args:
            alpha: Coefficient d'aversion absolue (α > 0)
        """
        assert alpha > 0, "alpha doit être positif"
        self.alpha = alpha
    
    def __call__(self, wealth: float) -> float:
        return -np.exp(-self.alpha * wealth)
    
    def derivative(self, wealth: float) -> float:
        return self.alpha * np.exp(-self.alpha * wealth)
    
    def second_derivative(self, wealth: float) -> float:
        return -self.alpha ** 2 * np.exp(-self.alpha * wealth)
    
    def inverse(self, utility: float) -> float:
        return -np.log(-utility) / self.alpha
    
    def certainty_equivalent_gaussian(self, mean: float, variance: float) -> float:
        """
        Formule exacte pour richesse normale.
        CE = μ - ασ²/2
        """
        return mean - self.alpha * variance / 2
    
    def optimal_allocation(
        self,
        risk_free_rate: float,
        risky_return: float,
        risky_volatility: float
    ) -> float:
        """
        Allocation optimale dans l'actif risqué.
        π* = (μ - r) / (α * σ²)
        """
        excess_return = risky_return - risk_free_rate
        return excess_return / (self.alpha * risky_volatility ** 2)


class CRRA(UtilityFunction):
    """
    Constant Relative Risk Aversion.
    
    U(w) = w^{1-γ} / (1-γ)  si γ ≠ 1
    U(w) = ln(w)            si γ = 1
    
    Propriétés:
    - R(w) = γ (constant)
    - Homothétique
    - Utilisée pour le problème de Merton
    """
    
    def __init__(self, gamma: float):
        """
        Args:
            gamma: Coefficient d'aversion relative (γ > 0)
        """
        assert gamma > 0, "gamma doit être positif"
        self.gamma = gamma
    
    def __call__(self, wealth: float) -> float:
        assert wealth > 0, "La richesse doit être positive pour CRRA"
        if np.isclose(self.gamma, 1.0):
            return np.log(wealth)
        return (wealth ** (1 - self.gamma)) / (1 - self.gamma)
    
    def derivative(self, wealth: float) -> float:
        return wealth ** (-self.gamma)
    
    def second_derivative(self, wealth: float) -> float:
        return -self.gamma * wealth ** (-self.gamma - 1)
    
    def inverse(self, utility: float) -> float:
        if np.isclose(self.gamma, 1.0):
            return np.exp(utility)
        return ((1 - self.gamma) * utility) ** (1 / (1 - self.gamma))
    
    def optimal_allocation(
        self,
        risk_free_rate: float,
        risky_return: float,
        risky_volatility: float
    ) -> float:
        """
        Allocation optimale (Merton).
        π* = (μ - r) / (γ * σ²)
        """
        excess_return = risky_return - risk_free_rate
        return excess_return / (self.gamma * risky_volatility ** 2)


class LogUtility(CRRA):
    """Utilité logarithmique (CRRA avec γ=1)."""
    
    def __init__(self):
        super().__init__(gamma=1.0)


class QuadraticUtility(UtilityFunction):
    """
    Utilité quadratique.
    U(w) = w - (b/2) * w²
    
    Note: Valide seulement pour w < 1/b (sinon U décroissante)
    """
    
    def __init__(self, b: float):
        assert b > 0
        self.b = b
    
    def __call__(self, wealth: float) -> float:
        return wealth - (self.b / 2) * wealth ** 2
    
    def derivative(self, wealth: float) -> float:
        return 1 - self.b * wealth
    
    def second_derivative(self, wealth: float) -> float:
        return -self.b
    
    def inverse(self, utility: float) -> float:
        # Résoudre w - (b/2)w² = u
        discriminant = 1 - 2 * self.b * utility
        if discriminant < 0:
            raise ValueError("Pas de solution")
        return (1 - np.sqrt(discriminant)) / self.b


# ============================================
# Métriques de Risque
# ============================================

def compute_returns(prices: np.ndarray, log_returns: bool = True) -> np.ndarray:
    """
    Calcule les rendements.
    
    Args:
        prices: Série de prix
        log_returns: Si True, rendements logarithmiques
    
    Returns:
        Rendements
    """
    if log_returns:
        return np.diff(np.log(prices))
    else:
        return np.diff(prices) / prices[:-1]


def compute_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Value at Risk.
    
    Perte potentielle avec probabilité (1 - confidence).
    
    Args:
        returns: Rendements
        confidence: Niveau de confiance (ex: 0.95)
        method: 'historical', 'parametric', ou 'cornish_fisher'
    
    Returns:
        VaR (valeur positive = perte)
    """
    if method == 'historical':
        return -np.percentile(returns, (1 - confidence) * 100)
    
    elif method == 'parametric':
        mu = np.mean(returns)
        sigma = np.std(returns)
        return -(mu + norm.ppf(1 - confidence) * sigma)
    
    elif method == 'cornish_fisher':
        # Ajustement pour skewness et kurtosis
        mu = np.mean(returns)
        sigma = np.std(returns)
        skew = ((returns - mu) ** 3).mean() / sigma ** 3
        kurt = ((returns - mu) ** 4).mean() / sigma ** 4 - 3
        
        z = norm.ppf(1 - confidence)
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        return -(mu + z_cf * sigma)
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")


def compute_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Conditional VaR (Expected Shortfall).
    
    Perte moyenne conditionnelle à dépasser le VaR.
    CVaR = E[L | L > VaR]
    
    Returns:
        CVaR (valeur positive)
    """
    var = compute_var(returns, confidence, method='historical')
    losses = -returns
    return np.mean(losses[losses >= var])


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Ratio de Sharpe annualisé.
    
    SR = (E[R] - r_f) / σ(R) * √periods_per_year
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Ratio de Sortino.
    
    Comme Sharpe mais ne pénalise que la volatilité baissière.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < target_return]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-10
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std


def compute_max_drawdown(prices: np.ndarray) -> float:
    """
    Maximum Drawdown.
    
    Plus grande perte depuis un sommet.
    """
    cummax = np.maximum.accumulate(prices)
    drawdown = (cummax - prices) / cummax
    return np.max(drawdown)


def compute_calmar_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Ratio de Calmar.
    
    Rendement annualisé / Max Drawdown
    """
    # Reconstruire les prix
    prices = np.cumprod(1 + returns)
    max_dd = compute_max_drawdown(prices)
    
    if max_dd == 0:
        return np.inf
    
    annual_return = np.mean(returns) * periods_per_year
    return (annual_return - risk_free_rate) / max_dd


def compute_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Information Ratio.
    
    IR = (R_p - R_b) / TrackingError
    """
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(active_returns) / tracking_error


def compute_beta(
    returns: np.ndarray,
    market_returns: np.ndarray
) -> float:
    """
    Beta par rapport au marché.
    
    β = Cov(R, R_m) / Var(R_m)
    """
    covariance = np.cov(returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance > 0 else 0.0


def compute_alpha(
    returns: np.ndarray,
    market_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Alpha de Jensen (CAPM).
    
    α = R_p - [r_f + β(R_m - r_f)]
    """
    rf_period = risk_free_rate / periods_per_year
    beta = compute_beta(returns, market_returns)
    
    expected_return = rf_period + beta * (np.mean(market_returns) - rf_period)
    actual_return = np.mean(returns)
    
    return (actual_return - expected_return) * periods_per_year


def compute_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Omega Ratio.
    
    Ω = E[max(R - L, 0)] / E[max(L - R, 0)]
    """
    gains = np.maximum(returns - threshold, 0)
    losses = np.maximum(threshold - returns, 0)
    
    expected_gains = np.mean(gains)
    expected_losses = np.mean(losses)
    
    if expected_losses == 0:
        return np.inf
    
    return expected_gains / expected_losses


def compute_tail_ratio(
    returns: np.ndarray,
    percentile: float = 0.05
) -> float:
    """
    Tail Ratio.
    
    Mesure l'asymétrie des queues de distribution.
    """
    upper = np.percentile(returns, (1 - percentile) * 100)
    lower = np.percentile(returns, percentile * 100)
    
    if lower == 0:
        return np.inf
    
    return abs(upper) / abs(lower)
```


## PARTIE B.2 : FACTOR RISK MODELS

### factor_models.py

```python
"""
HelixOne - Modèles de Risque Factoriels
Style: Barra, Axioma

Les facteurs expliquent les rendements et risques des actifs.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class FactorExposure:
    """Exposition d'un actif aux facteurs."""
    asset_id: str
    exposures: Dict[str, float]  # factor_name -> beta


@dataclass
class FactorRiskModel:
    """
    Modèle de risque factoriel.
    
    R_i = α_i + Σ_k β_ik * F_k + ε_i
    
    Où:
    - R_i: rendement de l'actif i
    - β_ik: exposition de l'actif i au facteur k
    - F_k: rendement du facteur k
    - ε_i: risque idiosyncratique
    """
    
    def __init__(
        self,
        factor_names: List[str],
        factor_covariance: np.ndarray,
        factor_returns: Optional[np.ndarray] = None
    ):
        """
        Args:
            factor_names: Noms des facteurs
            factor_covariance: Matrice de covariance des facteurs
            factor_returns: Rendements historiques des facteurs (optionnel)
        """
        self.factor_names = factor_names
        self.n_factors = len(factor_names)
        self.factor_cov = factor_covariance
        self.factor_returns = factor_returns
        
        # Index des facteurs
        self.factor_idx = {name: i for i, name in enumerate(factor_names)}
    
    def get_asset_risk(
        self,
        exposures: np.ndarray,
        idiosyncratic_var: float
    ) -> float:
        """
        Variance totale d'un actif.
        
        Var(R_i) = β_i' Σ_F β_i + σ_i²
        """
        systematic_var = exposures @ self.factor_cov @ exposures
        return systematic_var + idiosyncratic_var
    
    def get_portfolio_risk(
        self,
        weights: np.ndarray,
        exposures_matrix: np.ndarray,  # (n_assets, n_factors)
        idiosyncratic_vars: np.ndarray  # (n_assets,)
    ) -> float:
        """
        Variance du portefeuille.
        
        Var(R_p) = w' B Σ_F B' w + w' D w
        
        Où:
        - B: matrice d'expositions (n_assets x n_factors)
        - Σ_F: covariance des facteurs
        - D: matrice diagonale des variances idiosyncratiques
        """
        # Risque factoriel
        portfolio_exposures = weights @ exposures_matrix
        factor_risk = portfolio_exposures @ self.factor_cov @ portfolio_exposures
        
        # Risque idiosyncratique
        idio_risk = np.sum(weights ** 2 * idiosyncratic_vars)
        
        return factor_risk + idio_risk
    
    def decompose_risk(
        self,
        weights: np.ndarray,
        exposures_matrix: np.ndarray,
        idiosyncratic_vars: np.ndarray
    ) -> Dict[str, float]:
        """
        Décompose le risque par facteur.
        
        Returns:
            Dict avec contribution de chaque facteur et risque idio
        """
        total_var = self.get_portfolio_risk(weights, exposures_matrix, idiosyncratic_vars)
        total_std = np.sqrt(total_var)
        
        # Exposition du portefeuille à chaque facteur
        portfolio_exposures = weights @ exposures_matrix
        
        # Contribution marginale de chaque facteur
        contributions = {}
        for i, factor in enumerate(self.factor_names):
            # Contribution = β_p,k * Σ_{j} Σ_F[k,j] * β_p,j / σ_p
            factor_contrib = portfolio_exposures[i] * (
                self.factor_cov[i] @ portfolio_exposures
            ) / total_std
            contributions[factor] = factor_contrib
        
        # Risque idiosyncratique
        idio_risk = np.sqrt(np.sum(weights ** 2 * idiosyncratic_vars))
        contributions['idiosyncratic'] = idio_risk
        
        return contributions


class StatisticalFactorModel:
    """
    Modèle factoriel statistique (PCA).
    
    Extrait les facteurs des données de rendements.
    """
    
    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self.factor_loadings = None
        self.factor_returns = None
        self.explained_variance = None
    
    def fit(self, returns: np.ndarray) -> 'StatisticalFactorModel':
        """
        Estime les facteurs par PCA.
        
        Args:
            returns: Matrice de rendements (n_periods, n_assets)
        """
        from sklearn.decomposition import PCA
        
        # Standardiser
        returns_std = (returns - returns.mean(axis=0)) / returns.std(axis=0)
        
        # PCA
        pca = PCA(n_components=self.n_factors)
        self.factor_returns = pca.fit_transform(returns_std)
        self.factor_loadings = pca.components_.T  # (n_assets, n_factors)
        self.explained_variance = pca.explained_variance_ratio_
        
        return self
    
    def get_factor_covariance(self) -> np.ndarray:
        """Covariance des facteurs."""
        return np.cov(self.factor_returns.T)
    
    def get_idiosyncratic_variance(self, returns: np.ndarray) -> np.ndarray:
        """Variance résiduelle par actif."""
        predicted = self.factor_returns @ self.factor_loadings.T
        residuals = returns - predicted
        return np.var(residuals, axis=0)


class FundamentalFactorModel:
    """
    Modèle factoriel fondamental.
    
    Facteurs basés sur des caractéristiques des entreprises:
    - Value (B/P, E/P)
    - Size (Market Cap)
    - Momentum
    - Quality
    - Volatility
    - etc.
    """
    
    # Facteurs standards
    STANDARD_FACTORS = [
        'market',
        'size',
        'value',
        'momentum',
        'quality',
        'volatility',
        'growth',
        'dividend_yield'
    ]
    
    def __init__(self, factors: List[str] = None):
        self.factors = factors or self.STANDARD_FACTORS
        self.factor_returns_history = None
        self.factor_cov = None
    
    def compute_factor_exposures(
        self,
        market_cap: np.ndarray,
        book_to_price: np.ndarray,
        momentum_12m: np.ndarray,
        volatility: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Calcule les expositions factorielles.
        
        Returns:
            Matrice (n_assets, n_factors) d'expositions z-scorées
        """
        exposures = []
        
        # Size: log market cap (inversé, small = positive)
        size = -self._zscore(np.log(market_cap))
        exposures.append(size)
        
        # Value: book-to-price
        value = self._zscore(book_to_price)
        exposures.append(value)
        
        # Momentum
        mom = self._zscore(momentum_12m)
        exposures.append(mom)
        
        # Volatility (low vol = positive)
        vol = -self._zscore(volatility)
        exposures.append(vol)
        
        return np.column_stack(exposures)
    
    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        """Z-score robuste avec winsorisation."""
        x_clean = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
        return (x_clean - np.mean(x_clean)) / np.std(x_clean)
    
    def estimate_factor_returns(
        self,
        asset_returns: np.ndarray,
        exposures: np.ndarray
    ) -> np.ndarray:
        """
        Estime les rendements factoriels par régression cross-sectionnelle.
        
        Pour chaque période t:
        R_t = X_t * f_t + ε_t
        f_t = (X'X)^{-1} X' R_t
        """
        n_periods = asset_returns.shape[0]
        n_factors = exposures.shape[1]
        
        factor_returns = np.zeros((n_periods, n_factors))
        
        for t in range(n_periods):
            # WLS regression
            X = exposures
            y = asset_returns[t]
            
            # OLS: f = (X'X)^{-1} X' y
            XtX_inv = np.linalg.inv(X.T @ X + 1e-6 * np.eye(n_factors))
            factor_returns[t] = XtX_inv @ X.T @ y
        
        self.factor_returns_history = factor_returns
        self.factor_cov = np.cov(factor_returns.T)
        
        return factor_returns


# ============================================
# Risk Attribution
# ============================================

def marginal_contribution_to_risk(
    weights: np.ndarray,
    covariance: np.ndarray
) -> np.ndarray:
    """
    Contribution marginale au risque (MCR).
    
    MCR_i = ∂σ_p / ∂w_i = (Σw)_i / σ_p
    """
    portfolio_var = weights @ covariance @ weights
    portfolio_std = np.sqrt(portfolio_var)
    
    return (covariance @ weights) / portfolio_std


def component_contribution_to_risk(
    weights: np.ndarray,
    covariance: np.ndarray
) -> np.ndarray:
    """
    Contribution composante au risque (CCR).
    
    CCR_i = w_i * MCR_i
    
    Propriété: Σ CCR_i = σ_p
    """
    mcr = marginal_contribution_to_risk(weights, covariance)
    return weights * mcr


def percent_contribution_to_risk(
    weights: np.ndarray,
    covariance: np.ndarray
) -> np.ndarray:
    """
    Contribution en pourcentage au risque.
    
    PCR_i = CCR_i / σ_p = w_i * MCR_i / σ_p
    
    Propriété: Σ PCR_i = 100%
    """
    portfolio_std = np.sqrt(weights @ covariance @ weights)
    ccr = component_contribution_to_risk(weights, covariance)
    return ccr / portfolio_std
```

## PARTIE B.3 : VAR & STRESS TESTING

### stress_testing.py

```python
"""
HelixOne - Stress Testing Framework
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    REVERSE = "reverse"


@dataclass
class Scenario:
    """Un scénario de stress."""
    name: str
    type: ScenarioType
    factor_shocks: Dict[str, float]  # factor_name -> shock (%)
    description: str = ""


@dataclass
class StressTestResult:
    """Résultat d'un stress test."""
    scenario: Scenario
    portfolio_pnl: float
    portfolio_return: float
    var_breach: bool
    asset_pnls: Dict[str, float]
    factor_contributions: Dict[str, float]


class StressTestFramework:
    """
    Framework de stress testing.
    
    Supporte:
    - Scénarios historiques (2008, COVID, etc.)
    - Scénarios hypothétiques
    - Reverse stress testing
    """
    
    # Scénarios historiques prédéfinis
    HISTORICAL_SCENARIOS = {
        'gfc_2008': Scenario(
            name='Global Financial Crisis 2008',
            type=ScenarioType.HISTORICAL,
            factor_shocks={
                'equity': -0.40,
                'credit': -0.20,
                'rates': -0.02,
                'volatility': 0.80,
                'liquidity': -0.30
            },
            description='Lehman Brothers collapse, Sep-Oct 2008'
        ),
        'covid_2020': Scenario(
            name='COVID-19 Crash 2020',
            type=ScenarioType.HISTORICAL,
            factor_shocks={
                'equity': -0.35,
                'credit': -0.15,
                'rates': -0.01,
                'volatility': 0.65,
                'oil': -0.50
            },
            description='COVID-19 market crash, March 2020'
        ),
        'taper_tantrum_2013': Scenario(
            name='Taper Tantrum 2013',
            type=ScenarioType.HISTORICAL,
            factor_shocks={
                'rates': 0.01,
                'emerging_markets': -0.15,
                'credit': -0.05,
                'equity': -0.05
            },
            description='Fed tapering announcement, May-June 2013'
        ),
        'black_monday_1987': Scenario(
            name='Black Monday 1987',
            type=ScenarioType.HISTORICAL,
            factor_shocks={
                'equity': -0.22,
                'volatility': 1.50
            },
            description='Stock market crash, October 19, 1987'
        )
    }
    
    def __init__(self):
        self.scenarios: List[Scenario] = []
        self.results: List[StressTestResult] = []
    
    def add_scenario(self, scenario: Scenario):
        """Ajoute un scénario."""
        self.scenarios.append(scenario)
    
    def add_historical_scenario(self, name: str):
        """Ajoute un scénario historique prédéfini."""
        if name in self.HISTORICAL_SCENARIOS:
            self.scenarios.append(self.HISTORICAL_SCENARIOS[name])
    
    def create_hypothetical_scenario(
        self,
        name: str,
        factor_shocks: Dict[str, float],
        description: str = ""
    ) -> Scenario:
        """Crée un scénario hypothétique."""
        scenario = Scenario(
            name=name,
            type=ScenarioType.HYPOTHETICAL,
            factor_shocks=factor_shocks,
            description=description
        )
        self.scenarios.append(scenario)
        return scenario
    
    def run_stress_test(
        self,
        portfolio_weights: np.ndarray,
        asset_factor_exposures: np.ndarray,  # (n_assets, n_factors)
        factor_names: List[str],
        current_portfolio_value: float,
        var_limit: float
    ) -> List[StressTestResult]:
        """
        Exécute tous les stress tests.
        
        Returns:
            Liste de résultats
        """
        results = []
        
        for scenario in self.scenarios:
            # Construire le vecteur de chocs
            factor_shock_vector = np.zeros(len(factor_names))
            for i, factor in enumerate(factor_names):
                if factor in scenario.factor_shocks:
                    factor_shock_vector[i] = scenario.factor_shocks[factor]
            
            # Calculer l'impact sur chaque actif
            # ΔR_asset = exposures @ factor_shocks
            asset_returns = asset_factor_exposures @ factor_shock_vector
            
            # P&L du portefeuille
            portfolio_return = portfolio_weights @ asset_returns
            portfolio_pnl = current_portfolio_value * portfolio_return
            
            # Contributions par facteur
            portfolio_exposures = portfolio_weights @ asset_factor_exposures
            factor_contributions = {
                factor: portfolio_exposures[i] * scenario.factor_shocks.get(factor, 0)
                for i, factor in enumerate(factor_names)
            }
            
            # P&L par actif
            asset_pnls = {
                f'asset_{i}': portfolio_weights[i] * current_portfolio_value * ret
                for i, ret in enumerate(asset_returns)
            }
            
            result = StressTestResult(
                scenario=scenario,
                portfolio_pnl=portfolio_pnl,
                portfolio_return=portfolio_return,
                var_breach=abs(portfolio_pnl) > var_limit,
                asset_pnls=asset_pnls,
                factor_contributions=factor_contributions
            )
            results.append(result)
        
        self.results = results
        return results
    
    def reverse_stress_test(
        self,
        portfolio_weights: np.ndarray,
        asset_factor_exposures: np.ndarray,
        factor_names: List[str],
        target_loss: float,
        current_portfolio_value: float
    ) -> Dict[str, float]:
        """
        Reverse stress test: trouve les chocs qui produisent une perte cible.
        
        Minimise ||factor_shocks||² tel que portfolio_loss = target_loss
        """
        from scipy.optimize import minimize
        
        portfolio_exposures = portfolio_weights @ asset_factor_exposures
        target_return = -target_loss / current_portfolio_value
        
        # Résoudre: exposures @ shocks = target_return
        # avec régularisation pour le choc minimum
        def objective(shocks):
            return np.sum(shocks ** 2)
        
        def constraint(shocks):
            return portfolio_exposures @ shocks - target_return
        
        result = minimize(
            objective,
            x0=np.zeros(len(factor_names)),
            constraints={'type': 'eq', 'fun': constraint},
            method='SLSQP'
        )
        
        return {factor: shock for factor, shock in zip(factor_names, result.x)}
    
    def summary_report(self) -> str:
        """Génère un rapport de synthèse."""
        if not self.results:
            return "Aucun résultat disponible"
        
        lines = ["=" * 60]
        lines.append("STRESS TEST SUMMARY REPORT")
        lines.append("=" * 60)
        
        for result in self.results:
            lines.append(f"\nScenario: {result.scenario.name}")
            lines.append(f"Type: {result.scenario.type.value}")
            lines.append(f"P&L: {result.portfolio_pnl:,.2f}")
            lines.append(f"Return: {result.portfolio_return:.2%}")
            lines.append(f"VaR Breach: {'YES ⚠️' if result.var_breach else 'No'}")
            
            lines.append("\nFactor Contributions:")
            for factor, contrib in sorted(
                result.factor_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ):
                if abs(contrib) > 0.001:
                    lines.append(f"  {factor}: {contrib:.2%}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class MonteCarloVaR:
    """
    VaR par simulation Monte Carlo.
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        time_horizon: int = 1,  # jours
        confidence: float = 0.95
    ):
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.confidence = confidence
    
    def compute_var(
        self,
        portfolio_weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        portfolio_value: float
    ) -> Tuple[float, float, np.ndarray]:
        """
        Calcule VaR et CVaR par Monte Carlo.
        
        Returns:
            var: Value at Risk
            cvar: Conditional VaR
            simulated_pnl: Distribution des P&L simulés
        """
        n_assets = len(portfolio_weights)
        
        # Paramètres pour l'horizon
        mu = expected_returns * self.time_horizon
        sigma = covariance_matrix * self.time_horizon
        
        # Simulation
        cholesky = np.linalg.cholesky(sigma)
        z = np.random.standard_normal((self.n_simulations, n_assets))
        simulated_returns = mu + z @ cholesky.T
        
        # P&L du portefeuille
        portfolio_returns = simulated_returns @ portfolio_weights
        simulated_pnl = portfolio_value * portfolio_returns
        
        # VaR et CVaR
        var = -np.percentile(simulated_pnl, (1 - self.confidence) * 100)
        cvar = -np.mean(simulated_pnl[simulated_pnl <= -var])
        
        return var, cvar, simulated_pnl
    
    def compute_component_var(
        self,
        portfolio_weights: np.ndarray,
        covariance_matrix: np.ndarray,
        portfolio_var: float
    ) -> np.ndarray:
        """
        Component VaR: contribution de chaque actif au VaR.
        """
        portfolio_std = np.sqrt(portfolio_weights @ covariance_matrix @ portfolio_weights)
        marginal_var = (covariance_matrix @ portfolio_weights) / portfolio_std
        return portfolio_weights * marginal_var * (portfolio_var / portfolio_std)
```

---

# PARTIE C : PORTFOLIO MANAGEMENT

---

## PARTIE C.1 : OPTIMISATION STATIQUE (MARKOWITZ)

### optimization.py

```python
"""
HelixOne - Optimisation de Portefeuille
Markowitz, MVO, Black-Litterman
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Résultat d'une optimisation."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str


class MeanVarianceOptimizer:
    """
    Optimiseur Mean-Variance (Markowitz).
    """
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.0
    ):
        """
        Args:
            expected_returns: Vecteur des rendements espérés (n_assets,)
            covariance_matrix: Matrice de covariance (n_assets, n_assets)
            risk_free_rate: Taux sans risque annuel
        """
        self.mu = expected_returns
        self.cov = covariance_matrix
        self.rf = risk_free_rate
        self.n_assets = len(expected_returns)
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Rendement espéré du portefeuille."""
        return weights @ self.mu
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Volatilité du portefeuille."""
        return np.sqrt(weights @ self.cov @ weights)
    
    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """Ratio de Sharpe."""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.rf) / vol if vol > 0 else 0
    
    def min_variance_portfolio(
        self,
        allow_short: bool = False
    ) -> OptimizationResult:
        """
        Portefeuille de variance minimale globale (GMV).
        """
        def objective(w):
            return w @ self.cov @ w
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = None if allow_short else [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP'
        )
        
        weights = result.x
        return OptimizationResult(
            weights=weights,
            expected_return=self.portfolio_return(weights),
            volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
            success=result.success,
            message=result.message
        )
    
    def max_sharpe_portfolio(
        self,
        allow_short: bool = False
    ) -> OptimizationResult:
        """
        Portefeuille tangent (max Sharpe).
        """
        def neg_sharpe(w):
            ret = w @ self.mu
            vol = np.sqrt(w @ self.cov @ w)
            return -(ret - self.rf) / vol if vol > 1e-10 else 0
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = None if allow_short else [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(
            neg_sharpe,
            x0=np.ones(self.n_assets) / self.n_assets,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP'
        )
        
        weights = result.x
        return OptimizationResult(
            weights=weights,
            expected_return=self.portfolio_return(weights),
            volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
            success=result.success,
            message=result.message
        )
    
    def target_return_portfolio(
        self,
        target_return: float,
        allow_short: bool = False
    ) -> OptimizationResult:
        """
        Portefeuille de variance minimale pour un rendement cible.
        """
        def objective(w):
            return w @ self.cov @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w @ self.mu - target_return}
        ]
        bounds = None if allow_short else [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP'
        )
        
        weights = result.x
        return OptimizationResult(
            weights=weights,
            expected_return=self.portfolio_return(weights),
            volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
            success=result.success,
            message=result.message
        )
    
    def target_risk_portfolio(
        self,
        target_volatility: float,
        allow_short: bool = False
    ) -> OptimizationResult:
        """
        Portefeuille de rendement maximum pour une volatilité cible.
        """
        def neg_return(w):
            return -w @ self.mu
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sqrt(w @ self.cov @ w) - target_volatility}
        ]
        bounds = None if allow_short else [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(
            neg_return,
            x0=np.ones(self.n_assets) / self.n_assets,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP'
        )
        
        weights = result.x
        return OptimizationResult(
            weights=weights,
            expected_return=self.portfolio_return(weights),
            volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
            success=result.success,
            message=result.message
        )
    
    def efficient_frontier(
        self,
        n_points: int = 50,
        allow_short: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule la frontière efficiente.
        
        Returns:
            returns: Rendements de chaque portefeuille
            volatilities: Volatilités
            weights: Poids (n_points, n_assets)
        """
        # Trouver les bornes
        gmv = self.min_variance_portfolio(allow_short)
        max_ret_idx = np.argmax(self.mu)
        max_return = self.mu[max_ret_idx]
        
        target_returns = np.linspace(gmv.expected_return, max_return, n_points)
        
        returns = []
        volatilities = []
        all_weights = []
        
        for target in target_returns:
            result = self.target_return_portfolio(target, allow_short)
            if result.success:
                returns.append(result.expected_return)
                volatilities.append(result.volatility)
                all_weights.append(result.weights)
        
        return np.array(returns), np.array(volatilities), np.array(all_weights)
    
    def risk_parity_portfolio(self) -> OptimizationResult:
        """
        Portefeuille Risk Parity (Equal Risk Contribution).
        
        Chaque actif contribue équitablement au risque total.
        """
        def risk_contribution_error(w):
            vol = np.sqrt(w @ self.cov @ w)
            marginal_contrib = (self.cov @ w) / vol
            risk_contrib = w * marginal_contrib
            target_contrib = vol / self.n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 1) for _ in range(self.n_assets)]  # Poids minimum 1%
        
        result = minimize(
            risk_contribution_error,
            x0=np.ones(self.n_assets) / self.n_assets,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP'
        )
        
        weights = result.x
        return OptimizationResult(
            weights=weights,
            expected_return=self.portfolio_return(weights),
            volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.sharpe_ratio(weights),
            success=result.success,
            message=result.message
        )


class BlackLitterman:
    """
    Modèle de Black-Litterman.
    
    Combine équilibre de marché avec vues subjectives.
    """
    
    def __init__(
        self,
        market_caps: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.0
    ):
        """
        Args:
            market_caps: Capitalisations boursières
            covariance_matrix: Matrice de covariance
            risk_aversion: Coefficient d'aversion au risque
            tau: Incertitude sur les rendements d'équilibre
            risk_free_rate: Taux sans risque
        """
        self.market_caps = market_caps
        self.cov = covariance_matrix
        self.delta = risk_aversion
        self.tau = tau
        self.rf = risk_free_rate
        self.n_assets = len(market_caps)
        
        # Poids de marché
        self.market_weights = market_caps / market_caps.sum()
        
        # Rendements d'équilibre implicites
        self.equilibrium_returns = self.delta * self.cov @ self.market_weights
    
    def posterior_returns(
        self,
        views_matrix: np.ndarray,    # P: (n_views, n_assets)
        view_returns: np.ndarray,     # Q: (n_views,)
        view_confidence: np.ndarray   # Diagonale de Ω
    ) -> np.ndarray:
        """
        Calcule les rendements postérieurs.
        
        Args:
            views_matrix: Matrice P des vues
            view_returns: Vecteur Q des rendements attendus
            view_confidence: Confiance dans chaque vue (variance)
        
        Returns:
            Rendements postérieurs
        """
        P = views_matrix
        Q = view_returns
        omega = np.diag(view_confidence)
        
        pi = self.equilibrium_returns
        tau_cov = self.tau * self.cov
        
        # Formule de Black-Litterman
        # E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]
        
        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_omega = np.linalg.inv(omega)
        
        M = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
        posterior = M @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)
        
        return posterior
    
    def posterior_covariance(
        self,
        views_matrix: np.ndarray,
        view_confidence: np.ndarray
    ) -> np.ndarray:
        """Covariance postérieure."""
        P = views_matrix
        omega = np.diag(view_confidence)
        tau_cov = self.tau * self.cov
        
        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_omega = np.linalg.inv(omega)
        
        M = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
        
        return self.cov + M
    
    def optimal_weights(
        self,
        views_matrix: np.ndarray,
        view_returns: np.ndarray,
        view_confidence: np.ndarray
    ) -> np.ndarray:
        """
        Poids optimaux avec les vues.
        """
        posterior_ret = self.posterior_returns(views_matrix, view_returns, view_confidence)
        posterior_cov = self.posterior_covariance(views_matrix, view_confidence)
        
        # Poids = (δΣ_post)^{-1} * E[R]_post
        weights = np.linalg.inv(self.delta * posterior_cov) @ posterior_ret
        weights = weights / np.sum(weights)  # Normaliser
        
        return weights
    
    @staticmethod
    def absolute_view(asset_idx: int, n_assets: int, expected_return: float, confidence: float):
        """
        Crée une vue absolue: "L'actif i aura un rendement de x%".
        
        Returns:
            P, Q, omega pour cette vue
        """
        P = np.zeros((1, n_assets))
        P[0, asset_idx] = 1
        Q = np.array([expected_return])
        omega = np.array([confidence])
        return P, Q, omega
    
    @staticmethod
    def relative_view(
        long_idx: int,
        short_idx: int,
        n_assets: int,
        outperformance: float,
        confidence: float
    ):
        """
        Crée une vue relative: "L'actif i surperformera l'actif j de x%".
        
        Returns:
            P, Q, omega pour cette vue
        """
        P = np.zeros((1, n_assets))
        P[0, long_idx] = 1
        P[0, short_idx] = -1
        Q = np.array([outperformance])
        omega = np.array([confidence])
        return P, Q, omega
```

Je vais continuer... Le fichier devient très long. Laisse-moi finaliser avec les parties restantes :

,
        n_inventory_states: int = 21,
        n_price_states: int = 21,
        n_actions: int = 5
    ):
        self.Q = Q
        self.T = T
        self.n_inv_states = n_inventory_states
        self.n_price_states = n_price_states
        self.n_actions = n_actions
        
        # Grilles
        self.inv_grid = np.linspace(0, Q, n_inventory_states)
        self.action_grid = np.linspace(0, Q / T * 2, n_actions)  # 0 à 2x TWAP
    
    def get_state(self, inventory: float, price_change: float, time: int) -> Tuple:
        """Encode l'état."""
        inv_idx = np.argmin(np.abs(self.inv_grid - inventory))
        # Discrétiser price_change
        price_idx = min(self.n_price_states - 1, 
                       max(0, int((price_change + 0.05) / 0.1 * self.n_price_states)))
        return (inv_idx, price_idx, time)
    
    def step(
        self,
        state: Tuple,
        action: int,
        current_price: float,
        volatility: float,
        eta: float
    ) -> Tuple[Tuple, float, bool]:
        """
        Effectue un pas d'exécution.
        
        Returns:
            next_state, reward, done
        """
        inv_idx, price_idx, time = state
        inventory = self.inv_grid[inv_idx]
        trade = min(self.action_grid[action], inventory)
        
        # Impact et exécution
        temp_impact = eta * trade
        exec_price = current_price - temp_impact
        revenue = trade * exec_price
        
        # Coût de risque
        new_inventory = inventory - trade
        risk_cost = 0.01 * volatility**2 * new_inventory**2
        
        reward = revenue - risk_cost
        
        # Nouveau prix (simulation)
        price_change = np.random.normal(0, volatility)
        new_price = current_price + price_change
        
        # Nouvel état
        next_state = self.get_state(new_inventory, price_change, time + 1)
        done = (time + 1 >= self.T) or (new_inventory < 1e-6)
        
        # Pénalité si non liquidé
        if done and new_inventory > 1e-6:
            reward -= new_inventory * current_price * 0.05
        
        return next_state, reward, done
```

---

## PARTIE F.2 : MARKET MAKING (AVELLANEDA-STOIKOV)

### market_making.py

```python
"""
HelixOne - Market Making Optimal
Avellaneda-Stoikov Model
Source: Stanford CME 241 - Chapter 10
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


@dataclass
class MarketMakingParams:
    """Paramètres du market maker."""
    gamma: float      # Aversion au risque
    sigma: float      # Volatilité du prix
    kappa: float      # Intensité d'arrivée des ordres
    T: float          # Horizon
    dt: float = 0.001 # Pas de temps
    max_inventory: int = 10


class AvellanedaStoikov:
    """
    Market Making Optimal selon Avellaneda-Stoikov.
    
    Le market maker maximise:
    E[-exp(-γ * W_T)]
    
    En affichant des prix bid/ask optimaux.
    """
    
    def __init__(self, params: MarketMakingParams):
        self.p = params
    
    def reservation_price(self, mid: float, inventory: int, t: float) -> float:
        """
        Prix de réservation (indifférence).
        
        r(t, q) = S - q * γ * σ² * (T - t)
        
        Intuition: Si long (q>0), on veut vendre → prix plus bas
        """
        p = self.p
        return mid - inventory * p.gamma * p.sigma**2 * (p.T - t)
    
    def optimal_spread(self, t: float) -> float:
        """
        Spread optimal bid-ask.
        
        δ(t) = γσ²(T-t) + (2/γ) * ln(1 + γ/κ)
        """
        p = self.p
        time_component = p.gamma * p.sigma**2 * (p.T - t)
        intensity_component = (2 / p.gamma) * np.log(1 + p.gamma / p.kappa)
        return time_component + intensity_component
    
    def optimal_quotes(
        self, 
        mid: float, 
        inventory: int, 
        t: float
    ) -> Tuple[float, float]:
        """
        Prix bid et ask optimaux.
        
        Returns:
            (bid_price, ask_price)
        """
        r = self.reservation_price(mid, inventory, t)
        half_spread = self.optimal_spread(t) / 2
        
        return r - half_spread, r + half_spread
    
    def simulate_session(
        self,
        S0: float = 100.0,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Simule une session de market making.
        
        Returns:
            Dict avec historiques
        """
        if seed is not None:
            np.random.seed(seed)
        
        p = self.p
        n_steps = int(p.T / p.dt)
        
        # Historiques
        times = np.zeros(n_steps + 1)
        prices = np.zeros(n_steps + 1)
        inventories = np.zeros(n_steps + 1, dtype=int)
        cash = np.zeros(n_steps + 1)
        bids = np.zeros(n_steps)
        asks = np.zeros(n_steps)
        
        prices[0] = S0
        
        for i in range(n_steps):
            t = i * p.dt
            times[i] = t
            S = prices[i]
            q = inventories[i]
            
            # Quotes optimaux
            bid, ask = self.optimal_quotes(S, q, t)
            bids[i] = bid
            asks[i] = ask
            
            # Arrivée des ordres (Poisson)
            # Prob d'exécution dépend de la distance au mid
            bid_intensity = p.kappa * np.exp(-p.kappa * (S - bid)) if bid < S else 0
            ask_intensity = p.kappa * np.exp(-p.kappa * (ask - S)) if ask > S else 0
            
            bid_prob = 1 - np.exp(-bid_intensity * p.dt)
            ask_prob = 1 - np.exp(-ask_intensity * p.dt)
            
            new_q = q
            new_cash = cash[i]
            
            # Exécution bid (on achète)
            if np.random.random() < bid_prob and q < p.max_inventory:
                new_q += 1
                new_cash -= bid
            
            # Exécution ask (on vend)
            if np.random.random() < ask_prob and q > -p.max_inventory:
                new_q -= 1
                new_cash += ask
            
            # Evolution du prix mid
            prices[i + 1] = S + p.sigma * np.sqrt(p.dt) * np.random.randn()
            inventories[i + 1] = new_q
            cash[i + 1] = new_cash
        
        times[-1] = p.T
        
        # PnL final
        final_pnl = cash[-1] + inventories[-1] * prices[-1]
        
        return {
            'times': times,
            'prices': prices,
            'inventories': inventories,
            'cash': cash,
            'bids': bids,
            'asks': asks,
            'final_pnl': final_pnl,
            'n_trades': np.sum(np.abs(np.diff(inventories)))
        }


class MarketMakingRL:
    """
    Market Making avec Reinforcement Learning.
    
    Permet d'apprendre des stratégies adaptatives.
    """
    
    def __init__(
        self,
        max_inventory: int = 10,
        n_spread_actions: int = 5,
        spread_range: Tuple[float, float] = (0.01, 0.10)
    ):
        self.max_inventory = max_inventory
        self.n_spread_actions = n_spread_actions
        self.spread_grid = np.linspace(spread_range[0], spread_range[1], n_spread_actions)
        
        # États: (inventory, time_bucket, volatility_regime)
        self.n_inv_states = 2 * max_inventory + 1
        self.n_time_buckets = 10
        self.n_vol_regimes = 3  # low, medium, high
    
    def get_state(
        self, 
        inventory: int, 
        time_fraction: float,
        realized_vol: float,
        vol_thresholds: Tuple[float, float] = (0.01, 0.02)
    ) -> Tuple:
        """Encode l'état."""
        inv_idx = inventory + self.max_inventory
        time_idx = min(self.n_time_buckets - 1, int(time_fraction * self.n_time_buckets))
        
        if realized_vol < vol_thresholds[0]:
            vol_idx = 0
        elif realized_vol < vol_thresholds[1]:
            vol_idx = 1
        else:
            vol_idx = 2
        
        return (inv_idx, time_idx, vol_idx)
    
    def get_quotes(
        self,
        mid_price: float,
        action: int,
        inventory: int
    ) -> Tuple[float, float]:
        """
        Calcule bid/ask à partir de l'action.
        
        Action = indice du spread
        On ajuste aussi selon l'inventaire (skew)
        """
        base_spread = self.spread_grid[action]
        
        # Skew basé sur l'inventaire
        skew = 0.001 * inventory  # Si long, on veut vendre → ask plus attractif
        
        half_spread = base_spread / 2
        bid = mid_price - half_spread - skew
        ask = mid_price + half_spread - skew
        
        return bid, ask
```

---

# PARTIE G : PERFORMANCE & ATTRIBUTION

---

## PARTIE G.1 : PERFORMANCE METRICS

### performance/metrics.py

```python
"""
HelixOne - Métriques de Performance
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    """Rapport de performance complet."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    best_day: float
    worst_day: float
    avg_win: float
    avg_loss: float
    skewness: float
    kurtosis: float


def compute_performance_report(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> PerformanceReport:
    """
    Génère un rapport de performance complet.
    
    Args:
        returns: Série de rendements
        risk_free_rate: Taux sans risque annuel
        periods_per_year: Nombre de périodes par an
    """
    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period
    
    # Rendements
    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    
    # Risque
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    
    # Ratios
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)
    
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-10
    sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std
    
    # Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_drawdown = np.max(drawdowns)
    
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else np.inf
    
    # VaR / CVaR
    var_95 = -np.percentile(returns, 5)
    cvar_95 = -np.mean(returns[returns <= -var_95])
    
    # Win/Loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    
    total_wins = np.sum(wins) if len(wins) > 0 else 0
    total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-10
    profit_factor = total_wins / total_losses
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    # Distribution
    skewness = ((returns - returns.mean()) ** 3).mean() / returns.std() ** 3
    kurtosis = ((returns - returns.mean()) ** 4).mean() / returns.std() ** 4 - 3
    
    return PerformanceReport(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        var_95=var_95,
        cvar_95=cvar_95,
        win_rate=win_rate,
        profit_factor=profit_factor,
        best_day=np.max(returns),
        worst_day=np.min(returns),
        avg_win=avg_win,
        avg_loss=avg_loss,
        skewness=skewness,
        kurtosis=kurtosis
    )


def rolling_sharpe(
    returns: np.ndarray,
    window: int = 252,
    risk_free_rate: float = 0.0
) -> np.ndarray:
    """Sharpe ratio glissant."""
    result = np.full(len(returns), np.nan)
    rf_period = risk_free_rate / 252
    
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        excess = window_returns - rf_period
        result[i] = np.sqrt(252) * np.mean(excess) / np.std(window_returns)
    
    return result


def drawdown_analysis(returns: np.ndarray) -> Dict:
    """
    Analyse détaillée des drawdowns.
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    
    # Identifier les périodes de drawdown
    in_drawdown = drawdowns > 0
    drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0] + 1
    drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0] + 1
    
    # Ajuster si on commence/finit en drawdown
    if in_drawdown[0]:
        drawdown_starts = np.insert(drawdown_starts, 0, 0)
    if in_drawdown[-1]:
        drawdown_ends = np.append(drawdown_ends, len(returns))
    
    # Analyser chaque drawdown
    dd_details = []
    for start, end in zip(drawdown_starts, drawdown_ends[:len(drawdown_starts)]):
        depth = np.max(drawdowns[start:end])
        duration = end - start
        
        # Trouver le point bas
        trough_idx = start + np.argmax(drawdowns[start:end])
        
        dd_details.append({
            'start': start,
            'trough': trough_idx,
            'end': end,
            'depth': depth,
            'duration': duration,
            'recovery_time': end - trough_idx
        })
    
    # Trier par profondeur
    dd_details.sort(key=lambda x: x['depth'], reverse=True)
    
    return {
        'drawdown_series': drawdowns,
        'max_drawdown': np.max(drawdowns),
        'avg_drawdown': np.mean(drawdowns[drawdowns > 0]) if np.any(drawdowns > 0) else 0,
        'drawdown_periods': dd_details[:10],  # Top 10
        'time_in_drawdown': np.mean(in_drawdown)
    }
```

## PARTIE G.2 : BRINSON ATTRIBUTION

### performance/attribution.py

```python
"""
HelixOne - Performance Attribution
Brinson-Fachler Model
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BrinsonAttribution:
    """Résultat d'attribution Brinson."""
    allocation_effect: float      # Effet de l'allocation sectorielle
    selection_effect: float       # Effet de la sélection de titres
    interaction_effect: float     # Effet d'interaction
    total_active_return: float    # Rendement actif total
    
    sector_details: Dict[str, Dict[str, float]]  # Détail par secteur


def brinson_attribution(
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    sector_names: List[str]
) -> BrinsonAttribution:
    """
    Attribution de performance Brinson-Fachler.
    
    Décompose le rendement actif en:
    - Allocation: choisir les bons secteurs
    - Sélection: choisir les bons titres dans chaque secteur
    - Interaction: effet croisé
    
    Args:
        portfolio_weights: Poids du portefeuille par secteur
        benchmark_weights: Poids du benchmark par secteur
        portfolio_returns: Rendements du portefeuille par secteur
        benchmark_returns: Rendements du benchmark par secteur
        sector_names: Noms des secteurs
    
    Returns:
        BrinsonAttribution avec la décomposition
    """
    # Rendement total du benchmark
    R_b = np.sum(benchmark_weights * benchmark_returns)
    
    # Rendement total du portefeuille
    R_p = np.sum(portfolio_weights * portfolio_returns)
    
    # Différences de poids et rendements
    weight_diff = portfolio_weights - benchmark_weights
    return_diff = portfolio_returns - benchmark_returns
    
    # Effets par secteur
    allocation_by_sector = weight_diff * (benchmark_returns - R_b)
    selection_by_sector = benchmark_weights * return_diff
    interaction_by_sector = weight_diff * return_diff
    
    # Effets totaux
    allocation_total = np.sum(allocation_by_sector)
    selection_total = np.sum(selection_by_sector)
    interaction_total = np.sum(interaction_by_sector)
    
    # Vérification
    total_active = R_p - R_b
    assert np.isclose(
        allocation_total + selection_total + interaction_total,
        total_active,
        atol=1e-10
    ), "Attribution ne somme pas au rendement actif"
    
    # Détails par secteur
    sector_details = {}
    for i, sector in enumerate(sector_names):
        sector_details[sector] = {
            'portfolio_weight': portfolio_weights[i],
            'benchmark_weight': benchmark_weights[i],
            'portfolio_return': portfolio_returns[i],
            'benchmark_return': benchmark_returns[i],
            'allocation_effect': allocation_by_sector[i],
            'selection_effect': selection_by_sector[i],
            'interaction_effect': interaction_by_sector[i],
            'total_contribution': (
                allocation_by_sector[i] +
                selection_by_sector[i] +
                interaction_by_sector[i]
            )
        }
    
    return BrinsonAttribution(
        allocation_effect=allocation_total,
        selection_effect=selection_total,
        interaction_effect=interaction_total,
        total_active_return=total_active,
        sector_details=sector_details
    )


def multi_period_attribution(
    portfolio_weights_series: np.ndarray,  # (T, n_sectors)
    benchmark_weights_series: np.ndarray,
    portfolio_returns_series: np.ndarray,
    benchmark_returns_series: np.ndarray,
    sector_names: List[str]
) -> Dict:
    """
    Attribution multi-périodes avec linking.
    
    Utilise la méthode de Carino pour lier les attributions.
    """
    T = len(portfolio_returns_series)
    
    # Attribution période par période
    period_attributions = []
    for t in range(T):
        attr = brinson_attribution(
            portfolio_weights_series[t],
            benchmark_weights_series[t],
            portfolio_returns_series[t],
            benchmark_returns_series[t],
            sector_names
        )
        period_attributions.append(attr)
    
    # Rendements cumulés
    R_p_cumul = np.prod(1 + np.sum(portfolio_weights_series * portfolio_returns_series, axis=1)) - 1
    R_b_cumul = np.prod(1 + np.sum(benchmark_weights_series * benchmark_returns_series, axis=1)) - 1
    
    # Linking avec Carino
    def carino_factor(R):
        if np.abs(R) < 1e-10:
            return 1.0
        return np.log(1 + R) / R
    
    k = carino_factor(R_p_cumul) / carino_factor(R_b_cumul) if R_b_cumul != 0 else 1
    
    # Agréger les effets
    total_allocation = sum(a.allocation_effect for a in period_attributions)
    total_selection = sum(a.selection_effect for a in period_attributions)
    total_interaction = sum(a.interaction_effect for a in period_attributions)
    
    # Ajuster avec le facteur de linking
    linked_allocation = k * total_allocation
    linked_selection = k * total_selection
    linked_interaction = k * total_interaction
    
    return {
        'cumulative_portfolio_return': R_p_cumul,
        'cumulative_benchmark_return': R_b_cumul,
        'cumulative_active_return': R_p_cumul - R_b_cumul,
        'linked_allocation_effect': linked_allocation,
        'linked_selection_effect': linked_selection,
        'linked_interaction_effect': linked_interaction,
        'period_attributions': period_attributions
    }


def factor_attribution(
    portfolio_returns: np.ndarray,
    factor_returns: np.ndarray,  # (T, n_factors)
    factor_names: List[str]
) -> Dict:
    """
    Attribution par facteurs de risque.
    
    R_p = α + Σ β_k * F_k + ε
    """
    from sklearn.linear_model import LinearRegression
    
    # Régression
    model = LinearRegression()
    model.fit(factor_returns, portfolio_returns)
    
    alpha = model.intercept_
    betas = model.coef_
    
    # R² et résidus
    predicted = model.predict(factor_returns)
    residuals = portfolio_returns - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((portfolio_returns - portfolio_returns.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    # Contribution de chaque facteur
    contributions = {}
    for i, name in enumerate(factor_names):
        factor_contrib = betas[i] * factor_returns[:, i]
        contributions[name] = {
            'beta': betas[i],
            'avg_factor_return': np.mean(factor_returns[:, i]),
            'contribution': np.mean(factor_contrib),
            'contribution_pct': np.mean(factor_contrib) / np.mean(portfolio_returns) * 100
        }
    
    return {
        'alpha': alpha,
        'r_squared': r_squared,
        'residual_vol': np.std(residuals),
        'factor_contributions': contributions
    }
```

---

# PARTIE H : INFRASTRUCTURE

---

## PARTIE H.1 : REAL-TIME RISK ENGINE

### engine/risk_engine.py

```python
"""
HelixOne - Moteur de Risque Temps Réel
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor


@dataclass
class Position:
    """Position sur un instrument."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_price)
    
    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.avg_price - 1) * np.sign(self.quantity)


@dataclass
class RiskLimits:
    """Limites de risque."""
    max_var: float = 1_000_000
    max_position_size: float = 5_000_000
    max_sector_exposure: float = 0.25
    max_single_name: float = 0.05
    max_leverage: float = 2.0
    max_drawdown: float = 0.10


@dataclass
class RiskAlert:
    """Alerte de risque."""
    timestamp: datetime
    level: str  # 'WARNING', 'CRITICAL'
    category: str
    message: str
    current_value: float
    limit_value: float


class RiskEngine:
    """
    Moteur de risque temps réel.
    
    Fonctionnalités:
    - Calcul de VaR en continu
    - Monitoring des limites
    - Alertes automatiques
    - Stress testing à la demande
    """
    
    def __init__(
        self,
        limits: RiskLimits,
        covariance_matrix: np.ndarray,
        factor_exposures: np.ndarray,
        factor_covariance: np.ndarray,
        update_frequency: float = 1.0  # secondes
    ):
        self.limits = limits
        self.cov = covariance_matrix
        self.factor_exposures = factor_exposures
        self.factor_cov = factor_covariance
        self.update_frequency = update_frequency
        
        self.positions: Dict[str, Position] = {}
        self.alerts: deque = deque(maxlen=1000)
        self.risk_history: deque = deque(maxlen=10000)
        
        self._running = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def update_position(self, position: Position):
        """Met à jour une position."""
        with self._lock:
            self.positions[position.symbol] = position
    
    def update_price(self, symbol: str, price: float):
        """Met à jour le prix d'un instrument."""
        with self._lock:
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def get_portfolio_weights(self) -> np.ndarray:
        """Calcule les poids du portefeuille."""
        with self._lock:
            total_value = sum(p.market_value for p in self.positions.values())
            if total_value == 0:
                return np.zeros(len(self.positions))
            
            return np.array([
                p.market_value / total_value
                for p in self.positions.values()
            ])
    
    def compute_portfolio_var(
        self,
        confidence: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calcule la VaR du portefeuille.
        
        Méthode: Variance-Covariance (paramétrique)
        """
        weights = self.get_portfolio_weights()
        if len(weights) == 0:
            return 0.0
        
        portfolio_value = sum(p.market_value for p in self.positions.values())
        portfolio_vol = np.sqrt(weights @ self.cov @ weights)
        
        from scipy.stats import norm
        z = norm.ppf(confidence)
        
        var = portfolio_value * z * portfolio_vol * np.sqrt(horizon)
        return var
    
    def compute_factor_risk(self) -> Dict[str, float]:
        """Décompose le risque par facteur."""
        weights = self.get_portfolio_weights()
        if len(weights) == 0:
            return {}
        
        portfolio_exposures = weights @ self.factor_exposures
        total_var = portfolio_exposures @ self.factor_cov @ portfolio_exposures
        total_std = np.sqrt(total_var)
        
        # Contribution marginale par facteur
        contributions = {}
        factor_names = ['market', 'size', 'value', 'momentum', 'volatility']
        
        for i, name in enumerate(factor_names):
            contrib = portfolio_exposures[i] * (self.factor_cov[i] @ portfolio_exposures) / total_std
            contributions[name] = contrib
        
        return contributions
    
    def check_limits(self) -> List[RiskAlert]:
        """Vérifie toutes les limites de risque."""
        alerts = []
        now = datetime.now()
        
        # VaR
        var = self.compute_portfolio_var()
        if var > self.limits.max_var:
            alerts.append(RiskAlert(
                timestamp=now,
                level='CRITICAL' if var > 1.2 * self.limits.max_var else 'WARNING',
                category='VaR',
                message=f'VaR ({var:,.0f}) exceeds limit ({self.limits.max_var:,.0f})',
                current_value=var,
                limit_value=self.limits.max_var
            ))
        
        # Taille des positions
        for symbol, pos in self.positions.items():
            if abs(pos.market_value) > self.limits.max_position_size:
                alerts.append(RiskAlert(
                    timestamp=now,
                    level='WARNING',
                    category='Position Size',
                    message=f'{symbol} position ({pos.market_value:,.0f}) exceeds limit',
                    current_value=abs(pos.market_value),
                    limit_value=self.limits.max_position_size
                ))
        
        # Concentration
        weights = self.get_portfolio_weights()
        for i, (symbol, pos) in enumerate(self.positions.items()):
            if abs(weights[i]) > self.limits.max_single_name:
                alerts.append(RiskAlert(
                    timestamp=now,
                    level='WARNING',
                    category='Concentration',
                    message=f'{symbol} weight ({weights[i]:.1%}) exceeds limit',
                    current_value=abs(weights[i]),
                    limit_value=self.limits.max_single_name
                ))
        
        return alerts
    
    def run_stress_test(self, scenario: Dict[str, float]) -> float:
        """
        Exécute un stress test.
        
        Args:
            scenario: Dict {factor_name: shock}
        
        Returns:
            P&L du scénario
        """
        weights = self.get_portfolio_weights()
        portfolio_value = sum(p.market_value for p in self.positions.values())
        
        # Construire le vecteur de chocs
        factor_names = ['market', 'size', 'value', 'momentum', 'volatility']
        shock_vector = np.array([scenario.get(f, 0) for f in factor_names])
        
        # Impact sur le portefeuille
        portfolio_exposures = weights @ self.factor_exposures
        portfolio_return = portfolio_exposures @ shock_vector
        
        return portfolio_value * portfolio_return
    
    def get_risk_summary(self) -> Dict:
        """Génère un résumé du risque."""
        return {
            'portfolio_value': sum(p.market_value for p in self.positions.values()),
            'total_pnl': sum(p.pnl for p in self.positions.values()),
            'var_95': self.compute_portfolio_var(0.95),
            'var_99': self.compute_portfolio_var(0.99),
            'factor_risk': self.compute_factor_risk(),
            'n_positions': len(self.positions),
            'active_alerts': len([a for a in self.alerts if a.level == 'CRITICAL'])
        }
```

---

# 📝 GUIDE D'IMPLÉMENTATION POUR CLAUDE AGENT

## Ordre d'implémentation recommandé

### Phase 1: Fondations (Semaine 1-2)
1. `core/distributions.py` ✅
2. `core/markov_process.py` ✅
3. `core/mdp.py` ✅
4. `algorithms/dp.py` ✅
5. Tests unitaires pour Phase 1

### Phase 2: Risque (Semaine 3-4)
1. `risk/utility.py` ✅
2. `risk/metrics.py` (VaR, Sharpe, etc.) ✅
3. `risk/factor_models.py` ✅
4. `risk/stress_testing.py` ✅
5. Tests unitaires pour Phase 2

### Phase 3: Portfolio (Semaine 5-6)
1. `portfolio/optimization.py` (Markowitz) ✅
2. `portfolio/black_litterman.py` ✅
3. `portfolio/merton.py` ✅
4. Tests unitaires pour Phase 3

### Phase 4: Algorithmes RL (Semaine 7-8)
1. `algorithms/td_learning.py` ✅
2. `algorithms/q_learning.py` ✅
3. `algorithms/policy_gradient.py` ✅
4. Tests et validation

### Phase 5: Exécution (Semaine 9-10)
1. `execution/almgren_chriss.py` ✅
2. `execution/market_making.py` ✅
3. Backtesting

### Phase 6: Performance (Semaine 11-12)
1. `performance/metrics.py` ✅
2. `performance/attribution.py` ✅
3. Reporting

### Phase 7: Infrastructure (Semaine 13-14)
1. `engine/risk_engine.py` ✅
2. API REST
3. Intégration

## Commandes pour Claude Agent

```bash
# Créer la structure
mkdir -p helixone/{core,risk,portfolio,execution,algorithms,performance,engine,data,api,utils}
touch helixone/__init__.py

# Installer les dépendances
pip install numpy scipy pandas torch scikit-learn matplotlib fastapi

# Exécuter les tests
pytest tests/ -v

# Lancer le serveur API
uvicorn helixone.api:app --reload
```

## Tests à créer

```python
# tests/test_core.py
def test_categorical_distribution():
    from helixone.core.distributions import Categorical
    d = Categorical({'a': 0.5, 'b': 0.5})
    samples = d.sample_n(1000)
    assert abs(samples.count('a') / 1000 - 0.5) < 0.1

def test_merton_solution():
    from helixone.portfolio.merton import MertonParams, MertonSolution
    params = MertonParams(r=0.03, mu=0.08, sigma=0.2, gamma=2, rho=0.04, T=np.inf)
    sol = MertonSolution(params)
    assert 0 < sol.optimal_risky_fraction < 2

def test_var_calculation():
    from helixone.risk.utility import compute_var
    returns = np.random.normal(0, 0.02, 1000)
    var = compute_var(returns, 0.95)
    assert var > 0
```

---

*Document complet pour HelixOne - Concurrent d'Aladdin*
*Basé sur Stanford CME 241 + Best Practices Industrie*
*~3500 lignes de code prêt à l'emploi*
