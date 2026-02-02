# 📚 GUIDE COMPLET STANFORD RL POUR HELIXONE
## Reinforcement Learning for Finance - Code Intégral

**Généré le**: 2026-01-29 08:51:08  
**Repository**: https://github.com/TikhonJelvis/RL-book  
**Auteurs**: Ashwin Rao (Stanford) & Tikhon Jelvis  
**Cours**: Stanford CME 241 - RL for Stochastic Control Problems in Finance

---

## ⚠️ NOTE IMPORTANTE

Ce guide contient **L'INTÉGRALITÉ DU CODE** de la bibliothèque Stanford RL.
Chaque fichier est documenté avec:
- Description et concepts clés
- Code source complet
- Notes d'intégration pour HelixOne

**Pour tout acronyme ou terme technique**, voir le GLOSSAIRE ci-dessous.

---

## 📋 TABLE DES MATIÈRES

### PARTIE 1 - FONDATIONS (Core Library)
- [distribution.py](#distribution)
- [iterate.py](#iterate)
- [markov_decision_process.py](#markov-decision-process)
- [markov_process.py](#markov-process)
- [policy.py](#policy)
- [returns.py](#returns)

### PARTIE 2 - APPROXIMATION DE FONCTIONS
- [chapter10/simple_inventory_mrp_func_approx.py](#simple-inventory-mrp-func-approx)
- [chapter5/func_approx_simple_examples.py](#func-approx-simple-examples)
- [chapter5/tabular_simple_examples.py](#tabular-simple-examples)
- [function_approx.py](#function-approx)

### PARTIE 3 - PROGRAMMATION DYNAMIQUE
- [approximate_dynamic_programming.py](#approximate-dynamic-programming)
- [dynamic_programming.py](#dynamic-programming)
- [finite_horizon.py](#finite-horizon)

### PARTIE 4 - MONTE CARLO
- [experience_replay.py](#experience-replay)
- [monte_carlo.py](#monte-carlo)

### PARTIE 5 - TEMPORAL DIFFERENCE
- [chapter10/mc_td_experience_replay.py](#mc-td-experience-replay)
- [td_lambda.py](#td-lambda)

### PARTIE 6 - POLICY GRADIENT
- [policy_gradient.py](#policy-gradient)

### PARTIE 7 - APPLICATIONS FINANCE
- [appendix2/efficient_frontier.py](#efficient-frontier)
- [chapter12/optimal_exercise_rl.py](#optimal-exercise-rl)
- [chapter13/asset_alloc_pg.py](#asset-alloc-pg)
- [chapter13/asset_alloc_reinforce.py](#asset-alloc-reinforce)
- [chapter2/stock_price_mp.py](#stock-price-mp)
- [chapter2/stock_price_simulations.py](#stock-price-simulations)
- [chapter7/asset_alloc_discrete.py](#asset-alloc-discrete)
- [chapter7/merton_solution_graph.py](#merton-solution-graph)
- [chapter8/max_exp_utility.py](#max-exp-utility)
- [chapter8/optimal_exercise_bi.py](#optimal-exercise-bi)
- [chapter8/optimal_exercise_bin_tree.py](#optimal-exercise-bin-tree)
- [chapter9/optimal_order_execution.py](#optimal-order-execution)
- [chapter9/order_book.py](#order-book)

### PARTIE 8 - EXEMPLES ET PROBLÈMES
- [chapter1/probability.py](#probability)
- [chapter10/memory_function.py](#memory-function)
- [chapter10/prediction_utils.py](#prediction-utils)
- [chapter10/random_walk_mrp.py](#random-walk-mrp)
- [chapter10/simple_inventory_mrp.py](#simple-inventory-mrp)
- [chapter11/control_utils.py](#control-utils)
- [chapter11/simple_inventory_mdp_cap.py](#simple-inventory-mdp-cap)
- [chapter11/windy_grid.py](#windy-grid)
- [chapter11/windy_grid_convergence.py](#windy-grid-convergence)
- [chapter12/laguerre.py](#laguerre)
- [chapter12/random_walk_lstd.py](#random-walk-lstd)
- [chapter12/vampire.py](#vampire)
- [chapter14/epsilon_greedy.py](#epsilon-greedy)
- [chapter14/gradient_bandits.py](#gradient-bandits)
- [chapter14/mab_base.py](#mab-base)
- [chapter14/mab_graphs_gen.py](#mab-graphs-gen)
- [chapter14/plot_mab_graphs.py](#plot-mab-graphs)
- [chapter14/ts_bernoulli.py](#ts-bernoulli)
- [chapter14/ts_gaussian.py](#ts-gaussian)
- [chapter14/ucb1.py](#ucb1)
- [chapter15/ams.py](#ams)
- [chapter2/simple_inventory_mp.py](#simple-inventory-mp)
- [chapter2/simple_inventory_mrp.py](#simple-inventory-mrp)
- [chapter3/simple_inventory_mdp_cap.py](#simple-inventory-mdp-cap)
- [chapter3/simple_inventory_mdp_nocap.py](#simple-inventory-mdp-nocap)
- [chapter4/clearance_pricing_mdp.py](#clearance-pricing-mdp)
- [gen_utils/common_funcs.py](#common-funcs)
- [gen_utils/plot_funcs.py](#plot-funcs)
- [problems/Final-Winter2021/windy_grid.py](#windy-grid)
- [problems/Final-Winter2021/windy_grid_outline.py](#windy-grid-outline)
- [problems/Midterm-Winter2021/career_optimization.py](#career-optimization)
- [problems/Midterm-Winter2021/grid_maze.py](#grid-maze)

### PARTIE 9 - TESTS UNITAIRES
- [chapter10/test_lambda_return.py](#test-lambda-return)
- [chapter12/test_batch_rl_prediction.py](#test-batch-rl-prediction)
- [chapter12/test_lspi.py](#test-lspi)
- [chapter12/test_q_learning_experience_replay.py](#test-q-learning-experience-replay)
- [test_approx_dp_clearance.py](#test-approx-dp-clearance)
- [test_approx_dp_inventory.py](#test-approx-dp-inventory)
- [test_approximate_dynamic_programming.py](#test-approximate-dynamic-programming)
- [test_distribution.py](#test-distribution)
- [test_dynamic_programming.py](#test-dynamic-programming)
- [test_finite_horizon.py](#test-finite-horizon)
- [test_function_approx.py](#test-function-approx)
- [test_iterate.py](#test-iterate)
- [test_markov_process.py](#test-markov-process)
- [test_monte_carlo.py](#test-monte-carlo)
- [test_td.py](#test-td)

### PARTIE 10 - UTILITAIRES
- [td.py](#td)

---


# 📖 GLOSSAIRE COMPLET DES ACRONYMES ET TERMES

## Acronymes Principaux

| Acronyme | Signification Complète | Explication Simple |
|----------|------------------------|-------------------|
| **RL** | Reinforcement Learning (Apprentissage par Renforcement) | L'agent apprend en interagissant avec son environnement, recevant des récompenses |
| **MDP** | Markov Decision Process (Processus de Décision Markovien) | Framework mathématique pour décisions séquentielles avec incertitude |
| **MP** | Markov Process (Processus de Markov) | Séquence d'états où chaque état ne dépend que du précédent |
| **MRP** | Markov Reward Process (Processus de Markov avec Récompenses) | MP avec signal de récompense à chaque transition |
| **DP** | Dynamic Programming (Programmation Dynamique) | Résolution par décomposition récursive en sous-problèmes |
| **ADP** | Approximate Dynamic Programming (DP Approximative) | DP avec approximation pour grands espaces d'états |
| **TD** | Temporal Difference (Différence Temporelle) | Apprentissage par bootstrap (estimer à partir d'estimations) |
| **MC** | Monte Carlo | Méthodes basées sur échantillonnage d'épisodes complets |
| **SARSA** | State-Action-Reward-State-Action | Algorithme TD on-policy: utilise l'action réellement prise |
| **DQN** | Deep Q-Network (Réseau Q Profond) | Q-Learning avec réseaux de neurones profonds |
| **LSPI** | Least Squares Policy Iteration | Itération de politique par moindres carrés |
| **LSTD** | Least Squares TD (TD par Moindres Carrés) | TD avec solution analytique par moindres carrés |
| **MAB** | Multi-Armed Bandit (Bandit Manchot Multi-Bras) | Problème d'exploration vs exploitation |
| **UCB** | Upper Confidence Bound (Borne de Confiance Supérieure) | Stratégie d'exploration optimiste |
| **TS** | Thompson Sampling (Échantillonnage de Thompson) | Exploration bayésienne par échantillonnage |
| **DNN** | Deep Neural Network (Réseau de Neurones Profond) | Réseau avec plusieurs couches cachées |
| **SGD** | Stochastic Gradient Descent (Descente de Gradient Stochastique) | Optimisation par mini-batches |
| **Adam** | Adaptive Moment Estimation | Optimiseur adaptatif (momentum + RMSprop) |

## Termes Finance

| Terme | Signification | Exemple |
|-------|---------------|---------|
| **VWAP** | Volume Weighted Average Price (Prix Moyen Pondéré par Volume) | Benchmark d'exécution: VWAP = Σ(prix × volume) / Σvolume |
| **TWAP** | Time Weighted Average Price (Prix Moyen Pondéré par Temps) | Exécution uniforme dans le temps |
| **LOB** | Limit Order Book (Carnet d'Ordres à Cours Limité) | Structure bid/ask avec ordres en attente |
| **GBM** | Geometric Brownian Motion (Mouvement Brownien Géométrique) | dS = μSdt + σSdW, modèle de prix d'actifs |
| **SDE** | Stochastic Differential Equation (Équation Différentielle Stochastique) | Équation avec terme aléatoire |
| **HJB** | Hamilton-Jacobi-Bellman | Équation du contrôle optimal continu |

## Symboles Mathématiques

| Symbole | Nom | Signification | Valeurs Typiques |
|---------|-----|---------------|-----------------|
| **γ** (gamma) | Discount Factor (Facteur d'Actualisation) | Importance des récompenses futures | 0.9, 0.99, 0.999 |
| **α** (alpha) | Learning Rate (Taux d'Apprentissage) | Vitesse de mise à jour | 0.01, 0.001, 0.0001 |
| **ε** (epsilon) | Exploration Rate (Taux d'Exploration) | Probabilité d'action aléatoire | 0.1, 0.05, décroissant |
| **λ** (lambda) | Eligibility Trace Decay | Décroissance traces d'éligibilité | 0, 0.5, 0.9, 1 |
| **V(s)** | Value Function (Fonction de Valeur) | E[Σ γ^t r_t | s_0 = s] | - |
| **Q(s,a)** | Action-Value Function | E[Σ γ^t r_t | s_0 = s, a_0 = a] | - |
| **π** (pi) | Policy (Politique) | Stratégie: π(a|s) = P(action a dans état s) | - |
| **G_t** | Return (Retour) | Σ_{k=0}^∞ γ^k r_{t+k+1} | - |
| **δ_t** | TD Error (Erreur TD) | r + γV(s') - V(s) | - |

## Exemples Concrets

### MDP (Markov Decision Process)
```
Exemple: Trading d'actions
- États (S): (prix, position, cash, temps)
- Actions (A): {acheter, vendre, attendre} × quantité
- Transitions P(s'|s,a): modèle de marché stochastique
- Récompenses R: profit/perte réalisé
- γ: facteur d'actualisation (préférence temporelle)
- Objectif: maximiser E[Σ γ^t * profit_t]
```

### TD(0) vs Monte Carlo
```
Monte Carlo (attend fin d'épisode):
  V(s) ← V(s) + α[G_t - V(s)]
  où G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... (retour réel)

TD(0) (met à jour immédiatement):
  V(s) ← V(s) + α[r + γV(s') - V(s)]
  où r + γV(s') est la cible bootstrappée

Avantage TD: apprentissage online, pas besoin d'attendre
Avantage MC: pas de biais (utilise retours réels)
```

### Q-Learning vs SARSA
```
Q-Learning (off-policy):
  Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
  Utilise l'action OPTIMALE pour la cible (même si pas prise)

SARSA (on-policy):
  Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
  Utilise l'action RÉELLEMENT PRISE a'

Q-Learning: converge vers politique optimale
SARSA: converge vers politique ε-greedy optimale
```

### ε-greedy Exploration
```
Avec probabilité ε: action ALÉATOIRE (exploration)
Avec probabilité 1-ε: action OPTIMALE (exploitation)

Exemple Trading:
- Exploitation: suivre la stratégie apprise (souvent profitable)
- Exploration: essayer une nouvelle action (découvrir mieux?)

ε décroissant: commence avec beaucoup d'exploration, puis exploite
```

---


---

# PARTIE 1 - FONDATIONS (Core Library)

================================================================================

## 📄 distribution.py {#distribution}

**Titre**: Distributions de Probabilité

**Description**: Module fondamental définissant les distributions de probabilité

**Lignes de code**: 334

**Concepts clés**:
- Distribution[A] - Classe abstraite de base pour échantillonnage
- SampledDistribution - Distribution définie par fonction sampler
- FiniteDistribution - Distribution discrète avec table de probabilités
- Categorical - Sélection parmi outcomes discrets avec probabilités
- Gaussian, Poisson, Beta, Gamma, Uniform - Distributions standard
- Bernoulli, Constant, Choose, Range - Distributions utilitaires

**🎯 Utilisation HelixOne**: Base pour modéliser l'incertitude des prix, volumes, et transitions

### Code Source Complet

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
import random
from typing import (Callable, Dict, Generic, Iterator, Iterable,
                    Mapping, Optional, Sequence, Tuple, TypeVar)

A = TypeVar('A')

B = TypeVar('B')


class Distribution(ABC, Generic[A]):
    '''A probability distribution that we can sample.

    '''
    @abstractmethod
    def sample(self) -> A:
        '''Return a random sample from this distribution.

        '''
        pass

    def sample_n(self, n: int) -> Sequence[A]:
        '''Return n samples from this distribution.'''
        return [self.sample() for _ in range(n)]

    @abstractmethod
    def expectation(
        self,
        f: Callable[[A], float]
    ) -> float:
        '''Return the expecation of f(X) where X is the
        random variable for the distribution and f is an
        arbitrary function from X to float

        '''
        pass

    def map(
        self,
        f: Callable[[A], B]
    ) -> Distribution[B]:
        '''Apply a function to the outcomes of this distribution.'''
        return SampledDistribution(lambda: f(self.sample()))

    def apply(
        self,
        f: Callable[[A], Distribution[B]]
    ) -> Distribution[B]:
        '''Apply a function that returns a distribution to the outcomes of
        this distribution. This lets us express *dependent random
        variables*.

        '''
        def sample():
            a = self.sample()
            b_dist = f(a)
            return b_dist.sample()

        return SampledDistribution(sample)


class SampledDistribution(Distribution[A]):
    '''A distribution defined by a function to sample it.

    '''
    sampler: Callable[[], A]
    expectation_samples: int

    def __init__(
        self,
        sampler: Callable[[], A],
        expectation_samples: int = 10000
    ):
        self.sampler = sampler
        self.expectation_samples = expectation_samples

    def sample(self) -> A:
        return self.sampler()

    def expectation(
        self,
        f: Callable[[A], float]
    ) -> float:
        '''Return a sampled approximation of the expectation of f(X) for some f.

        '''
        return sum(f(self.sample()) for _ in
                   range(self.expectation_samples)) / self.expectation_samples


class Uniform(SampledDistribution[float]):
    '''Sample a uniform float between 0 and 1.

    '''
    def __init__(self, expectation_samples: int = 10000):
        super().__init__(
            sampler=lambda: random.uniform(0, 1),
            expectation_samples=expectation_samples
        )


class Poisson(SampledDistribution[int]):
    '''A poisson distribution with the given parameter.

    '''

    λ: float

    def __init__(self, λ: float, expectation_samples: int = 10000):
        self.λ = λ
        super().__init__(
            sampler=lambda: np.random.poisson(lam=self.λ),
            expectation_samples=expectation_samples
        )


class Gaussian(SampledDistribution[float]):
    '''A Gaussian distribution with the given μ and σ.'''

    μ: float
    σ: float

    def __init__(self, μ: float, σ: float, expectation_samples: int = 10000):
        self.μ = μ
        self.σ = σ
        super().__init__(
            sampler=lambda: np.random.normal(loc=self.μ, scale=self.σ),
            expectation_samples=expectation_samples
        )


class Gamma(SampledDistribution[float]):
    '''A Gamma distribution with the given α and β.'''

    α: float
    β: float

    def __init__(self, α: float, β: float, expectation_samples: int = 10000):
        self.α = α
        self.β = β
        super().__init__(
            sampler=lambda: np.random.gamma(shape=self.α, scale=1/self.β),
            expectation_samples=expectation_samples
        )


class Beta(SampledDistribution[float]):
    '''A Beta distribution with the given α and β.'''

    α: float
    β: float

    def __init__(self, α: float, β: float, expectation_samples: int = 10000):
        self.α = α
        self.β = β
        super().__init__(
            sampler=lambda: np.random.beta(a=self.α, b=self.β),
            expectation_samples=expectation_samples
        )


class FiniteDistribution(Distribution[A], ABC):
    '''A probability distribution with a finite number of outcomes, which
    means we can render it as a PDF or CDF table.

    '''
    @abstractmethod
    def table(self) -> Mapping[A, float]:
        '''Returns a tabular representation of the probability density
        function (PDF) for this distribution.

        '''
        pass

    def probability(self, outcome: A) -> float:
        '''Returns the probability of the given outcome according to this
        distribution.

        '''
        return self.table()[outcome]

    def map(self, f: Callable[[A], B]) -> FiniteDistribution[B]:
        '''Return a new distribution that is the result of applying a function
        to each element of this distribution.

        '''
        result: Dict[B, float] = defaultdict(float)

        for x, p in self:
            result[f(x)] += p

        return Categorical(result)

    def sample(self) -> A:
        outcomes = list(self.table().keys())
        weights = list(self.table().values())
        return random.choices(outcomes, weights=weights)[0]

    # TODO: Can we get rid of f or make it optional? Right now, I
    # don't think that's possible with mypy.
    def expectation(self, f: Callable[[A], float]) -> float:
        '''Calculate the expected value of the distribution, using the given
        function to turn the outcomes into numbers.

        '''
        return sum(p * f(x) for x, p in self)

    def __iter__(self) -> Iterator[Tuple[A, float]]:
        return iter(self.table().items())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FiniteDistribution):
            return self.table() == other.table()
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.table())


@dataclass(frozen=True)
class Constant(FiniteDistribution[A]):
    '''A distribution that has a single outcome with probability 1.

    '''
    value: A

    def sample(self) -> A:
        return self.value

    def table(self) -> Mapping[A, float]:
        return {self.value: 1}

    def probability(self, outcome: A) -> float:
        return 1. if outcome == self.value else 0.


@dataclass(frozen=True)
class Bernoulli(FiniteDistribution[bool]):
    '''A distribution with two outcomes. Returns True with probability p
    and False with probability 1 - p.

    '''
    p: float

    def sample(self) -> bool:
        return random.uniform(0, 1) <= self.p

    def table(self) -> Mapping[bool, float]:
        return {True: self.p, False: 1 - self.p}

    def probability(self, outcome: bool) -> float:
        return self.p if outcome else 1 - self.p


@dataclass
class Range(FiniteDistribution[int]):
    '''Select a random integer in the range [low, high), with low
    inclusive and high exclusive. (This works exactly the same as the
    normal range function, but differently from random.randit.)

    '''
    low: int
    high: int

    def __init__(self, a: int, b: Optional[int] = None):
        if b is None:
            b = a
            a = 0

        assert b > a

        self.low = a
        self.high = b

    def sample(self) -> int:
        return random.randint(self.low, self.high - 1)

    def table(self) -> Mapping[int, float]:
        length = self.high - self.low
        return {x: 1 / length for x in range(self.low, self.high)}


class Choose(FiniteDistribution[A]):
    '''Select an element of the given list uniformly at random.

    '''

    options: Sequence[A]
    _table: Optional[Mapping[A, float]] = None

    def __init__(self, options: Iterable[A]):
        self.options = list(options)

    def sample(self) -> A:
        return random.choice(self.options)

    def table(self) -> Mapping[A, float]:
        if self._table is None:
            counter = Counter(self.options)
            length = len(self.options)
            self._table = {x: counter[x] / length for x in counter}

        return self._table

    def probability(self, outcome: A) -> float:
        return self.table().get(outcome, 0.0)


class Categorical(FiniteDistribution[A]):
    '''Select from a finite set of outcomes with the specified
    probabilities.

    '''

    probabilities: Mapping[A, float]

    def __init__(self, distribution: Mapping[A, float]):
        total = sum(distribution.values())
        # Normalize probabilities to sum to 1
        self.probabilities = {outcome: probability / total
                              for outcome, probability in distribution.items()}

    def table(self) -> Mapping[A, float]:
        return self.probabilities

    def probability(self, outcome: A) -> float:
        return self.probabilities.get(outcome, 0.)
```

--------------------------------------------------------------------------------

## 📄 iterate.py {#iterate}

**Titre**: Utilitaires d'Itération

**Description**: Fonctions utilitaires pour l'itération et la convergence

**Lignes de code**: 120

**Concepts clés**:
- iterate - Applique une fonction répétitivement
- converge - Itère jusqu'à convergence (tolerance)
- accumulate - Accumule les résultats intermédiaires
- last - Retourne le dernier élément d'un itérateur

**🎯 Utilisation HelixOne**: Utilitaires pour boucles d'apprentissage

### Code Source Complet

```python
'''Finding fixed points of functions using iterators.'''
import itertools
from typing import Callable, Iterable, Iterator, Optional, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')


# It would be more efficient if you iterated in place instead of
# returning a copy of the value each time, but the functional version
# of the code is a lot cleaner and easier to work with.
def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    '''Find the fixed point of a function f by applying it to its own
    result, yielding each intermediate value.

    That is, for a function f, iterate(f, x) will give us a generator
    producing:

    x, f(x), f(f(x)), f(f(f(x)))...

    '''
    state = start

    while True:
        yield state
        state = step(state)


def last(values: Iterator[X]) -> Optional[X]:
    '''Return the last value of the given iterator.

    Returns None if the iterator is empty.

    If the iterator does not end, this function will loop forever.
    '''
    try:
        *_, last_element = values
        return last_element
    except ValueError:
        return None


def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    '''Read from an iterator until two consecutive values satisfy the
    given done function or the input iterator ends.

    Raises an error if the input iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.

    '''
    a = next(values, None)
    if a is None:
        return

    yield a

    for b in values:
        yield b
        if done(a, b):
            return

        a = b


def converged(values: Iterator[X],
              done: Callable[[X, X], bool]) -> X:
    '''Return the final value of the given iterator when its values
    converge according to the done function.

    Raises an error if the iterator is empty.

    Will loop forever if the input iterator doesn't end *or* converge.
    '''
    result = last(converge(values, done))

    if result is None:
        raise ValueError("converged called on an empty iterator")

    return result


def accumulate(
        iterable: Iterable[X],
        func: Callable[[Y, X], Y],
        *,
        initial: Optional[Y]
) -> Iterator[Y]:
    '''Make an iterator that returns accumulated sums, or accumulated
    results of other binary functions (specified via the optional func
    argument).

    If func is supplied, it should be a function of two
    arguments. Elements of the input iterable may be any type that can
    be accepted as arguments to func. (For example, with the default
    operation of addition, elements may be any addable type including
    Decimal or Fraction.)

    Usually, the number of elements output matches the input
    iterable. However, if the keyword argument initial is provided,
    the accumulation leads off with the initial value so that the
    output has one more element than the input iterable.

    '''
    if initial is not None:
        iterable = itertools.chain([initial], iterable)  # type: ignore

    return itertools.accumulate(iterable, func)  # type: ignore


if __name__ == '__main__':
    import numpy as np
    x = 0.0
    values = converge(
        iterate(lambda y: np.cos(y), x),
        lambda a, b: np.abs(a - b) < 1e-3
    )
    for i, v in enumerate(values):
        print(f"{i}: {v:.4f}")
```

--------------------------------------------------------------------------------

## 📄 markov_decision_process.py {#markov-decision-process}

**Titre**: Processus de Décision Markovien (MDP)

**Description**: Framework MDP complet pour la prise de décision séquentielle

**Lignes de code**: 182

**Concepts clés**:
- MarkovDecisionProcess[S,A] - MDP avec états S et actions A
- actions(state) - Actions disponibles dans un état
- step(state, action) → Distribution[(next_state, reward)]
- apply_policy(π) - Convertit MDP + politique → MRP
- simulate_actions - Génère trajectoires avec politique
- FiniteMarkovDecisionProcess - MDP avec espaces finis

**🎯 Utilisation HelixOne**: Framework central pour exécution d'ordres et allocation

### Code Source Complet

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (DefaultDict, Dict, Iterable, Generic, Mapping,
                    Tuple, Sequence, TypeVar, Set)

from rl.distribution import (Categorical, Distribution, FiniteDistribution)

from rl.markov_process import (
    FiniteMarkovRewardProcess, MarkovRewardProcess, StateReward, State,
    NonTerminal, Terminal)
from rl.policy import FinitePolicy, Policy

A = TypeVar('A')
S = TypeVar('S')


@dataclass(frozen=True)
class TransitionStep(Generic[S, A]):
    '''A single step in the simulation of an MDP, containing:

    state -- the state we start from
    action -- the action we took at that state
    next_state -- the state we ended up in after the action
    reward -- the instantaneous reward we got for this transition
    '''
    state: NonTerminal[S]
    action: A
    next_state: State[S]
    reward: float

    def add_return(self, γ: float, return_: float) -> ReturnStep[S, A]:
        '''Given a γ and the return from 'next_state', this annotates the
        transition with a return for 'state'.

        '''
        return ReturnStep(
            self.state,
            self.action,
            self.next_state,
            self.reward,
            return_=self.reward + γ * return_
        )


@dataclass(frozen=True)
class ReturnStep(TransitionStep[S, A]):
    '''A Transition that also contains the total *return* for its starting
    state.

    '''
    return_: float


class MarkovDecisionProcess(ABC, Generic[S, A]):
    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        mdp = self

        class RewardProcess(MarkovRewardProcess[S]):
            def transition_reward(
                self,
                state: NonTerminal[S]
            ) -> Distribution[Tuple[State[S], float]]:
                actions: Distribution[A] = policy.act(state)
                return actions.apply(lambda a: mdp.step(state, a))

        return RewardProcess()

    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        pass

    @abstractmethod
    def step(
        self,
        state: NonTerminal[S],
        action: A
    ) -> Distribution[Tuple[State[S], float]]:
        pass

    def simulate_actions(
            self,
            start_states: Distribution[NonTerminal[S]],
            policy: Policy[S, A]
    ) -> Iterable[TransitionStep[S, A]]:
        '''Simulate this MDP with the given policy, yielding the
        sequence of (states, action, next state, reward) 4-tuples
        encountered in the simulation trace.

        '''
        state: State[S] = start_states.sample()

        while isinstance(state, NonTerminal):
            action_distribution = policy.act(state)

            action = action_distribution.sample()
            next_distribution = self.step(state, action)

            next_state, reward = next_distribution.sample()
            yield TransitionStep(state, action, next_state, reward)
            state = next_state

    def action_traces(
            self,
            start_states: Distribution[NonTerminal[S]],
            policy: Policy[S, A]
    ) -> Iterable[Iterable[TransitionStep[S, A]]]:
        '''Yield an infinite number of traces as returned by
        simulate_actions.

        '''
        while True:
            yield self.simulate_actions(start_states, policy)


ActionMapping = Mapping[A, StateReward[S]]
StateActionMapping = Mapping[NonTerminal[S], ActionMapping[A, S]]


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    '''A Markov Decision Process with finite state and action spaces.

    '''

    mapping: StateActionMapping[S, A]
    non_terminal_states: Sequence[NonTerminal[S]]

    def __init__(
        self,
        mapping: Mapping[S, Mapping[A, FiniteDistribution[Tuple[S, float]]]]
    ):
        non_terminals: Set[S] = set(mapping.keys())
        self.mapping = {NonTerminal(s): {a: Categorical(
            {(NonTerminal(s1) if s1 in non_terminals else Terminal(s1), r): p
             for (s1, r), p in v}
        ) for a, v in d.items()} for s, d in mapping.items()}
        self.non_terminal_states = list(self.mapping.keys())

    def __repr__(self) -> str:
        display = ""
        for s, d in self.mapping.items():
            display += f"From State {s.state}:\n"
            for a, d1 in d.items():
                display += f"  With Action {a}:\n"
                for (s1, r), p in d1:
                    opt = "Terminal " if isinstance(s1, Terminal) else ""
                    display += f"    To [{opt}State {s1.state} and "\
                        + f"Reward {r:.3f}] with Probability {p:.3f}\n"
        return display

    def step(self, state: NonTerminal[S], action: A) -> StateReward[S]:
        action_map: ActionMapping[A, S] = self.mapping[state]
        return action_map[action]

    def apply_finite_policy(self, policy: FinitePolicy[S, A])\
            -> FiniteMarkovRewardProcess[S]:

        transition_mapping: Dict[S, FiniteDistribution[Tuple[S, float]]] = {}

        for state in self.mapping:
            action_map: ActionMapping[A, S] = self.mapping[state]
            outcomes: DefaultDict[Tuple[S, float], float]\
                = defaultdict(float)
            actions = policy.act(state)
            for action, p_action in actions:
                for (s1, r), p in action_map[action]:
                    outcomes[(s1.state, r)] += p_action * p

            transition_mapping[state.state] = Categorical(outcomes)

        return FiniteMarkovRewardProcess(transition_mapping)

    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        '''All the actions allowed for the given state.

        This will be empty for terminal states.

        '''
        return self.mapping[state].keys()
```

--------------------------------------------------------------------------------

## 📄 markov_process.py {#markov-process}

**Titre**: Processus de Markov (MP/MRP)

**Description**: Implémentation des processus de Markov avec et sans récompenses

**Lignes de code**: 317

**Concepts clés**:
- State[S] = Terminal | NonTerminal - États avec distinction finale
- MarkovProcess - Processus avec transition(state) → Distribution[State]
- MarkovRewardProcess - Ajoute transition_reward avec récompenses
- FiniteMarkovProcess - Version tabulaire pour DP exacte
- TransitionStep, ReturnStep - Structures pour trajectoires
- Matrice de transition et distribution stationnaire

**🎯 Utilisation HelixOne**: Modéliser les prix d'actifs comme processus stochastiques

### Code Source Complet

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import graphviz
import numpy as np
from pprint import pprint
from typing import (Callable, Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, TypeVar, Set)

from rl.distribution import (Categorical, Distribution, FiniteDistribution,
                             SampledDistribution)

S = TypeVar('S')
X = TypeVar('X')


class State(ABC, Generic[S]):
    state: S

    def on_non_terminal(
        self,
        f: Callable[[NonTerminal[S]], X],
        default: X
    ) -> X:
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S
        
    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


class MarkovProcess(ABC, Generic[S]):
    '''A Markov process with states of type S.
    '''
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        '''Given a state of the process, returns a distribution of
        the next states.  Returning None means we are in a terminal state.
        '''

    def simulate(
        self,
        start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[State[S]]:
        '''Run a simulation trace of this Markov process, generating the
        states visited during the trace.

        This yields the start state first, then continues yielding
        subsequent states forever or until we hit a terminal state.
        '''

        state: State[S] = start_state_distribution.sample()
        yield state

        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state

    def traces(
            self,
            start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[Iterable[State[S]]]:
        '''Yield simulation traces (the output of `simulate'), sampling a
        start state from the given distribution each time.

        '''
        while True:
            yield self.simulate(start_state_distribution)


Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]


class FiniteMarkovProcess(MarkovProcess[S]):
    '''A Markov Process with a finite state space.

    Having a finite state space lets us use tabular methods to work
    with the process (ie dynamic programming).

    '''

    non_terminal_states: Sequence[NonTerminal[S]]
    transition_map: Transition[S]

    def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
        non_terminals: Set[S] = set(transition_map.keys())
        self.transition_map = {
            NonTerminal(s): Categorical(
                {(NonTerminal(s1) if s1 in non_terminals else Terminal(s1)): p
                 for s1, p in v}
            ) for s, v in transition_map.items()
        }
        self.non_terminal_states = list(self.transition_map.keys())

    def __repr__(self) -> str:
        display = ""

        for s, d in self.transition_map.items():
            display += f"From State {s.state}:\n"
            for s1, p in d:
                opt = "Terminal " if isinstance(s1, Terminal) else ""
                display += f"  To {opt}State {s1.state} with Probability {p:.3f}\n"

        return display

    def get_transition_matrix(self) -> np.ndarray:
        sz = len(self.non_terminal_states)
        mat = np.zeros((sz, sz))

        for i, s1 in enumerate(self.non_terminal_states):
            for j, s2 in enumerate(self.non_terminal_states):
                mat[i, j] = self.transition(s1).probability(s2)

        return mat

    def transition(self, state: NonTerminal[S])\
            -> FiniteDistribution[State[S]]:
        return self.transition_map[state]

    def get_stationary_distribution(self) -> FiniteDistribution[S]:
        eig_vals, eig_vecs = np.linalg.eig(self.get_transition_matrix().T)
        index_of_first_unit_eig_val = np.where(
            np.abs(eig_vals - 1) < 1e-8)[0][0]
        eig_vec_of_unit_eig_val = np.real(
            eig_vecs[:, index_of_first_unit_eig_val])
        return Categorical({
            self.non_terminal_states[i].state: ev
            for i, ev in enumerate(eig_vec_of_unit_eig_val /
                                   sum(eig_vec_of_unit_eig_val))
        })

    def display_stationary_distribution(self):
        pprint({
            s: round(p, 3)
            for s, p in self.get_stationary_distribution()
        })

    def generate_image(self) -> graphviz.Digraph:
        d = graphviz.Digraph()

        for s in self.transition_map.keys():
            d.node(str(s))

        for s, v in self.transition_map.items():
            for s1, p in v:
                d.edge(str(s), str(s1), label=str(p))

        return d


# Reward processes
@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    state: NonTerminal[S]
    next_state: State[S]
    reward: float

    def add_return(self, γ: float, return_: float) -> ReturnStep[S]:
        '''Given a γ and the return from 'next_state', this annotates the
        transition with a return for 'state'.

        '''
        return ReturnStep(
            self.state,
            self.next_state,
            self.reward,
            return_=self.reward + γ * return_
        )


@dataclass(frozen=True)
class ReturnStep(TransitionStep[S]):
    return_: float


class MarkovRewardProcess(MarkovProcess[S]):
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        '''Transitions the Markov Reward Process, ignoring the generated
        reward (which makes this just a normal Markov Process).

        '''
        distribution = self.transition_reward(state)

        def next_state(distribution=distribution):
            next_s, _ = distribution.sample()
            return next_s

        return SampledDistribution(next_state)

    @abstractmethod
    def transition_reward(self, state: NonTerminal[S])\
            -> Distribution[Tuple[State[S], float]]:
        '''Given a state, returns a distribution of the next state
        and reward from transitioning between the states.

        '''

    def simulate_reward(
        self,
        start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[TransitionStep[S]]:
        '''Simulate the MRP, yielding an Iterable of
        (state, next state, reward) for each sampled transition.
        '''

        state: State[S] = start_state_distribution.sample()
        reward: float = 0.

        while isinstance(state, NonTerminal):
            next_distribution = self.transition_reward(state)

            next_state, reward = next_distribution.sample()
            yield TransitionStep(state, next_state, reward)

            state = next_state

    def reward_traces(
            self,
            start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[Iterable[TransitionStep[S]]]:
        '''Yield simulation traces (the output of `simulate_reward'), sampling
        a start state from the given distribution each time.

        '''
        while True:
            yield self.simulate_reward(start_state_distribution)


StateReward = FiniteDistribution[Tuple[State[S], float]]
RewardTransition = Mapping[NonTerminal[S], StateReward[S]]


class FiniteMarkovRewardProcess(FiniteMarkovProcess[S],
                                MarkovRewardProcess[S]):

    transition_reward_map: RewardTransition[S]
    reward_function_vec: np.ndarray

    def __init__(
        self,
        transition_reward_map: Mapping[S, FiniteDistribution[Tuple[S, float]]]
    ):
        transition_map: Dict[S, FiniteDistribution[S]] = {}

        for state, trans in transition_reward_map.items():
            probabilities: Dict[S, float] = defaultdict(float)
            for (next_state, _), probability in trans:
                probabilities[next_state] += probability

            transition_map[state] = Categorical(probabilities)

        super().__init__(transition_map)

        nt: Set[S] = set(transition_reward_map.keys())
        self.transition_reward_map = {
            NonTerminal(s): Categorical(
                {(NonTerminal(s1) if s1 in nt else Terminal(s1), r): p
                 for (s1, r), p in v}
            ) for s, v in transition_reward_map.items()
        }

        self.reward_function_vec = np.array([
            sum(probability * reward for (_, reward), probability in
                self.transition_reward_map[state])
            for state in self.non_terminal_states
        ])

    def __repr__(self) -> str:
        display = ""
        for s, d in self.transition_reward_map.items():
            display += f"From State {s.state}:\n"
            for (s1, r), p in d:
                opt = "Terminal " if isinstance(s1, Terminal) else ""
                display +=\
                    f"  To [{opt}State {s1.state} and Reward {r:.3f}]"\
                    + f" with Probability {p:.3f}\n"
        return display

    def transition_reward(self, state: NonTerminal[S]) -> StateReward[S]:
        return self.transition_reward_map[state]

    def get_value_function_vec(self, gamma: float) -> np.ndarray:
        return np.linalg.solve(
            np.eye(len(self.non_terminal_states)) -
            gamma * self.get_transition_matrix(),
            self.reward_function_vec
        )

    def display_reward_function(self):
        pprint({
            self.non_terminal_states[i]: round(r, 3)
            for i, r in enumerate(self.reward_function_vec)
        })

    def display_value_function(self, gamma: float):
        pprint({
            self.non_terminal_states[i]: round(v, 3)
            for i, v in enumerate(self.get_value_function_vec(gamma))
        })
```

--------------------------------------------------------------------------------

## 📄 policy.py {#policy}

**Titre**: Politiques (Stratégies)

**Description**: Définition des politiques de l'agent

**Lignes de code**: 109

**Concepts clés**:
- Policy[S,A] - Mapping état → Distribution[Action]
- DeterministicPolicy - Une seule action par état (argmax)
- FinitePolicy - Politique tabulaire sur espace fini
- RandomPolicy - Politique uniforme aléatoire

**🎯 Utilisation HelixOne**: Représenter les stratégies de trading

### Code Source Complet

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Mapping, TypeVar

from rl.distribution import Choose, Constant, Distribution, FiniteDistribution
from rl.markov_process import NonTerminal

A = TypeVar('A')
S = TypeVar('S')


class Policy(ABC, Generic[S, A]):
    '''A policy is a function that specifies what we should do (the
    action) at a given state of our MDP.

    '''
    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        '''A distribution of actions to take from the given non-terminal
        state.

        '''


@dataclass(frozen=True)
class UniformPolicy(Policy[S, A]):
    valid_actions: Callable[[S], Iterable[A]]

    def act(self, state: NonTerminal[S]) -> Choose[A]:
        return Choose(self.valid_actions(state.state))


@dataclass(frozen=True)
class RandomPolicy(Policy[S, A]):
    '''A policy that randomly selects one of several specified policies
    each action.

    Given the right inputs, this could simulate things like ε-greedy
    policies::

        RandomPolicy()

    '''
    policy_choices: Distribution[Policy[S, A]]

    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        policy: Policy[S, A] = self.policy_choices.sample()
        return policy.act(state)


@dataclass(frozen=True)
class DeterministicPolicy(Policy[S, A]):
    action_for: Callable[[S], A]

    def act(self, state: NonTerminal[S]) -> Constant[A]:
        return Constant(self.action_for(state.state))


class Always(DeterministicPolicy[S, A]):
    '''A constant policy: always return the same (specified) action for
    every possible state.

    '''
    action: A

    def __init__(self, action: A):
        self.action = action
        super().__init__(lambda _: action)


@dataclass(frozen=True)
class FinitePolicy(Policy[S, A]):
    ''' A policy where the state and action spaces are finite.

    '''
    policy_map: Mapping[S, FiniteDistribution[A]]

    def __repr__(self) -> str:
        display = ""
        for s, d in self.policy_map.items():
            display += f"For State {s}:\n"
            for a, p in d:
                display += f"  Do Action {a} with Probability {p:.3f}\n"
        return display

    def act(self, state: NonTerminal[S]) -> FiniteDistribution[A]:
        return self.policy_map[state.state]


class FiniteDeterministicPolicy(FinitePolicy[S, A]):
    '''A deterministic policy where the state and action spaces are
    finite.

    '''
    action_for: Mapping[S, A]

    def __init__(self, action_for: Mapping[S, A]):
        self.action_for = action_for
        super().__init__(policy_map={s: Constant(a) for s, a in
                                     self.action_for.items()})

    def __repr__(self) -> str:
        display = ""
        for s, a in self.action_for.items():
            display += f"For State {s}: Do Action {a}\n"
        return display
```

--------------------------------------------------------------------------------

## 📄 returns.py {#returns}

**Titre**: Calcul des Retours

**Description**: Calcul des retours (returns) pour RL

**Lignes de code**: 61

**Concepts clés**:
- returns - Calcule G_t = Σ γ^k * r_{t+k}
- Retours actualisés pour épisodes complets

**🎯 Utilisation HelixOne**: Calcul des performances de stratégies

### Code Source Complet

```python
import itertools
import math
from typing import Iterable, Iterator, TypeVar, overload

import rl.markov_process as mp
import rl.markov_decision_process as mdp
import rl.iterate as iterate


S = TypeVar('S')
A = TypeVar('A')


@overload
def returns(
        trace: Iterable[mp.TransitionStep[S]],
        γ: float,
        tolerance: float
) -> Iterator[mp.ReturnStep[S]]:
    ...


@overload
def returns(
        trace: Iterable[mdp.TransitionStep[S, A]],
        γ: float,
        tolerance: float
) -> Iterator[mdp.ReturnStep[S, A]]:
    ...


def returns(trace, γ, tolerance):
    '''Given an iterator of states and rewards, calculate the return of
    the first N states.

    Arguments:
    rewards -- instantaneous rewards
    γ -- the discount factor (0 < γ ≤ 1)
    tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance

    '''
    trace = iter(trace)

    max_steps = round(math.log(tolerance) / math.log(γ)) if γ < 1 else None
    if max_steps is not None:
        trace = itertools.islice(trace, max_steps * 2)

    *transitions, last_transition = list(trace)

    return_steps = iterate.accumulate(
        reversed(transitions),
        func=lambda next, curr: curr.add_return(γ, next.return_),
        initial=last_transition.add_return(γ, 0)
    )
    return_steps = reversed(list(return_steps))

    if max_steps is not None:
        return_steps = itertools.islice(return_steps, max_steps)

    return return_steps
```

--------------------------------------------------------------------------------

# PARTIE 2 - APPROXIMATION DE FONCTIONS

================================================================================

## 📄 chapter10/simple_inventory_mrp_func_approx.py {#simple-inventory-mrp-func-approx}

**Titre**: Simple Inventory Mrp Func Approx

**Description**: Module Simple Inventory Mrp Func Approx

**Lignes de code**: 99

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Iterable, Callable
from rl.function_approx import AdamGradient
from rl.function_approx import LinearFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.distribution import Choose
from rl.markov_decision_process import NonTerminal
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.chapter10.prediction_utils import (
    mc_prediction_learning_rate,
    td_prediction_learning_rate
)
import numpy as np
from itertools import islice


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

gamma: float = 0.9

si_mrp: SimpleInventoryMRPFinite = SimpleInventoryMRPFinite(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)
nt_states: Sequence[NonTerminal[InventoryState]] = si_mrp.non_terminal_states
true_vf: np.ndarray = si_mrp.get_value_function_vec(gamma=gamma)

mc_episode_length_tol: float = 1e-6
num_episodes = 10000

td_episode_length: int = 100
initial_learning_rate: float = 0.03
half_life: float = 1000.0
exponent: float = 0.5

ffs: Sequence[Callable[[NonTerminal[InventoryState]], float]] = \
    [(lambda x, s=s: float(x.state == s.state)) for s in nt_states]

mc_ag: AdamGradient = AdamGradient(
    learning_rate=0.05,
    decay1=0.9,
    decay2=0.999
)

td_ag: AdamGradient = AdamGradient(
    learning_rate=0.003,
    decay1=0.9,
    decay2=0.999
)

mc_func_approx: LinearFunctionApprox[NonTerminal[InventoryState]] = \
    LinearFunctionApprox.create(
        feature_functions=ffs,
        adam_gradient=mc_ag
    )

td_func_approx: LinearFunctionApprox[NonTerminal[InventoryState]] = \
    LinearFunctionApprox.create(
        feature_functions=ffs,
        adam_gradient=td_ag
    )

it_mc: Iterable[ValueFunctionApprox[InventoryState]] = \
    mc_prediction_learning_rate(
        mrp=si_mrp,
        start_state_distribution=Choose(nt_states),
        gamma=gamma,
        episode_length_tolerance=mc_episode_length_tol,
        initial_func_approx=mc_func_approx
    )

it_td: Iterable[ValueFunctionApprox[InventoryState]] = \
    td_prediction_learning_rate(
        mrp=si_mrp,
        start_state_distribution=Choose(nt_states),
        gamma=gamma,
        episode_length=td_episode_length,
        initial_func_approx=td_func_approx
    )

mc_episodes: int = 3000
for i, mc_vf in enumerate(islice(it_mc, mc_episodes)):
    mc_rmse: float = np.sqrt(sum(
        (mc_vf(s) - true_vf[i]) ** 2 for i, s in enumerate(nt_states)
    ) / len(nt_states))
    print(f"MC: Iteration = {i:d}, RMSE = {mc_rmse:.3f}")

td_experiences: int = 300000
for i, td_vf in enumerate(islice(it_td, td_experiences)):
    td_rmse: float = np.sqrt(sum(
        (td_vf(s) - true_vf[i]) ** 2 for i, s in enumerate(nt_states)
    ) / len(nt_states))
    print(f"TD: Iteration = {i:d}, RMSE = {td_rmse:.3f}")
```

--------------------------------------------------------------------------------

## 📄 chapter5/func_approx_simple_examples.py {#func-approx-simple-examples}

**Titre**: Func Approx Simple Examples

**Description**: Module Func Approx Simple Examples

**Lignes de code**: 143

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Tuple, Sequence, Iterator, List
from scipy.stats import norm
import numpy as np
from rl.function_approx import LinearFunctionApprox, DNNApprox, \
    AdamGradient, DNNSpec
from itertools import islice
from rl.gen_utils.plot_funcs import plot_list_of_curves

Triple = Tuple[float, float, float]
Aug_Triple = Tuple[float, float, float, float]
DataSeq = Sequence[Tuple[Triple, float]]


def example_model_data_generator() -> Iterator[Tuple[Triple, float]]:

    coeffs: Aug_Triple = (2., 10., 4., -6.)
    d = norm(loc=0., scale=0.3)

    while True:
        pt: np.ndarray = np.random.randn(3)
        x_val: Triple = (pt[0], pt[1], pt[2])
        y_val: float = coeffs[0] + np.dot(coeffs[1:], pt) + \
            d.rvs(size=1)[0]
        yield (x_val, y_val)


def data_seq_generator(
    data_generator: Iterator[Tuple[Triple, float]],
    num_pts: int
) -> Iterator[DataSeq]:
    while True:
        pts: DataSeq = list(islice(data_generator, num_pts))
        yield pts


def feature_functions():
    return [lambda _: 1., lambda x: x[0], lambda x: x[1], lambda x: x[2]]


def adam_gradient():
    return AdamGradient(
        learning_rate=0.1,
        decay1=0.9,
        decay2=0.999
    )


def get_linear_model() -> LinearFunctionApprox[Triple]:
    ffs = feature_functions()
    ag = adam_gradient()
    return LinearFunctionApprox.create(
         feature_functions=ffs,
         adam_gradient=ag,
         regularization_coeff=0.,
         direct_solve=True
    )


def get_dnn_model() -> DNNApprox[Triple]:
    ffs = feature_functions()
    ag = adam_gradient()

    def relu(arg: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: x if x > 0. else 0.)(arg)

    def relu_deriv(res: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: 1. if x > 0. else 0.)(res)

    def identity(arg: np.ndarray) -> np.ndarray:
        return arg

    def identity_deriv(res: np.ndarray) -> np.ndarray:
        return np.ones_like(res)

    ds = DNNSpec(
        neurons=[2],
        bias=True,
        hidden_activation=relu,
        hidden_activation_deriv=relu_deriv,
        output_activation=identity,
        output_activation_deriv=identity_deriv
    )

    return DNNApprox.create(
        feature_functions=ffs,
        dnn_spec=ds,
        adam_gradient=ag,
        regularization_coeff=0.05
    )


if __name__ == '__main__':
    training_num_pts: int = 1000
    test_num_pts: int = 10000
    training_iterations: int = 200
    data_gen: Iterator[Tuple[Triple, float]] = example_model_data_generator()
    training_data_gen: Iterator[DataSeq] = data_seq_generator(
        data_gen,
        training_num_pts
    )
    test_data: DataSeq = list(islice(data_gen, test_num_pts))

    direct_solve_lfa: LinearFunctionApprox[Triple] = \
        get_linear_model().solve(next(training_data_gen))
    direct_solve_rmse: float = direct_solve_lfa.rmse(test_data)
    print(f"Linear Model Direct Solve RMSE = {direct_solve_rmse:.3f}")
    print("-----------------------------")

    print("Linear Model SGD")
    print("----------------")
    linear_model_rmse_seq: List[float] = []
    for lfa in islice(
        get_linear_model().iterate_updates(training_data_gen),
        training_iterations
    ):
        this_rmse: float = lfa.rmse(test_data)
        linear_model_rmse_seq.append(this_rmse)
        iter: int = len(linear_model_rmse_seq)
        print(f"Iteration {iter:d}: RMSE = {this_rmse:.3f}")

    print("DNN Model SGD")
    print("-------------")
    dnn_model_rmse_seq: List[float] = []
    for dfa in islice(
        get_dnn_model().iterate_updates(training_data_gen),
        training_iterations
    ):
        this_rmse: float = dfa.rmse(test_data)
        dnn_model_rmse_seq.append(this_rmse)
        iter: int = len(dnn_model_rmse_seq)
        print(f"Iteration {iter:d}: RMSE = {this_rmse:.3f}")

    x_vals = range(training_iterations)
    plot_list_of_curves(
        list_of_x_vals=[x_vals, x_vals],
        list_of_y_vals=[linear_model_rmse_seq, dnn_model_rmse_seq],
        list_of_colors=["b-", "r--"],
        list_of_curve_labels=["Linear Model", "Deep Neural Network Model"],
        x_label="Iterations of Gradient Descent",
        y_label="Root Mean Square Error",
        title="RMSE across Iterations of Gradient Descent"
    )
```

--------------------------------------------------------------------------------

## 📄 chapter5/tabular_simple_examples.py {#tabular-simple-examples}

**Titre**: Tabular Simple Examples

**Description**: Module Tabular Simple Examples

**Lignes de code**: 40

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Tuple, Sequence, Iterator, List
import numpy as np
from scipy.stats import norm
from itertools import islice
from rl.function_approx import Tabular

Triple = Tuple[float, float, float]
Aug_Triple = Tuple[float, float, float, float]
DataSeq = Sequence[Tuple[Triple, float]]


def example_model_data_generator() -> Iterator[DataSeq]:

    coeffs: Aug_Triple = (2., 10., 4., -6.)
    values = np.linspace(-10.0, 10.0, 21)
    pts: Sequence[Triple] = [(x, y, z) for x in values for y in values
                             for z in values]
    d = norm(loc=0., scale=2.0)

    while True:
        res: List[Tuple[Triple, float]] = []
        for pt in pts:
            x_val: Triple = (pt[0], pt[1], pt[2])
            y_val: float = coeffs[0] + np.dot(coeffs[1:], pt) + \
                d.rvs(size=1)[0]
            res.append((x_val, y_val))
        yield res


if __name__ == '__main__':
    training_iterations: int = 30
    data_gen: Iterator[DataSeq] = example_model_data_generator()
    test_data: DataSeq = list(next(data_gen))

    tabular: Tabular[Triple] = Tabular()
    for xy_seq in islice(data_gen, training_iterations):
        tabular = tabular.update(xy_seq)
        this_rmse: float = tabular.rmse(test_data)
        print(f"RMSE = {this_rmse:.3f}")
```

--------------------------------------------------------------------------------

## 📄 function_approx.py {#function-approx}

**Titre**: Framework d'Approximation de Fonctions

**Description**: Approximation de V(s) et Q(s,a) pour grands espaces

**Lignes de code**: 947

**Concepts clés**:
- FunctionApprox[X] - Interface abstraite X → ℝ
- Dynamic - Lookup exact (DP tabulaire)
- Tabular - Table avec learning rate α(n)
- LinearFunctionApprox - Combinaison linéaire de features
- DNNApprox - Réseau de neurones profond
- AdamGradient - Optimiseur Adam (adaptive moment)
- Weights - Poids avec cache Adam
- DNNSpec - Spécification architecture DNN

**🎯 Utilisation HelixOne**: CRITIQUE - Approximer les fonctions de valeur

### Code Source Complet

```python
'''An interface for different kinds of function approximations
(tabular, linear, DNN... etc), with several implementations.'''

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace, field
import itertools
import numpy as np
# from operator import itemgetter
# from scipy.interpolate import splrep, BSpline
from typing import (Callable, Dict, Generic, Iterator, Iterable, List,
                    Mapping, Optional, Sequence, Tuple, TypeVar, overload)

import rl.iterate as iterate

X = TypeVar('X')
F = TypeVar('F', bound='FunctionApprox')
SMALL_NUM = 1e-6


class FunctionApprox(ABC, Generic[X]):
    '''Interface for function approximations.
    An object of this class approximates some function X ↦ ℝ in a way
    that can be evaluated at specific points in X and updated with
    additional (X, ℝ) points.
    '''

    @abstractmethod
    def __add__(self: F, other: F) -> F:
        pass

    @abstractmethod
    def __mul__(self: F, scalar: float) -> F:
        pass

    @abstractmethod
    def objective_gradient(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], np.ndarray]
    ) -> Gradient[F]:
        '''Computes the gradient of an objective function of the self
        FunctionApprox with respect to the parameters in the internal
        representation of the FunctionApprox. The gradient is output
        in the form of a Gradient[FunctionApprox] whose internal parameters are
        equal to the gradient values. The argument `obj_deriv_out_fun'
        represents the derivative of the objective with respect to the output
        (evaluate) of the FunctionApprox, when evaluated at a Sequence of
        x values and a Sequence of y values (to be obtained from 'xy_vals_seq')
        '''

    @abstractmethod
    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Computes expected value of y for each x in
        x_values_seq (with the probability distribution
        function of y|x estimated as FunctionApprox)
        '''

    def __call__(self, x_value: X) -> float:
        return self.evaluate([x_value]).item()

    @abstractmethod
    def update_with_gradient(
        self: F,
        gradient: Gradient[F]
    ) -> F:
        '''Update the internal parameters of self FunctionApprox using the
        input gradient that is presented as a Gradient[FunctionApprox]
        '''

    def update(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> F:

        '''Update the internal parameters of the FunctionApprox
        based on incremental data provided in the form of (x,y)
        pairs as a xy_vals_seq data structure
        '''
        def deriv_func(x: Sequence[X], y: Sequence[float]) -> np.ndarray:
            return self.evaluate(x) - np.array(y)

        return self.update_with_gradient(
            self.objective_gradient(xy_vals_seq, deriv_func)
        )

    @abstractmethod
    def solve(
        self: F,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> F:
        '''Assuming the entire data set of (x,y) pairs is available
        in the form of the given input xy_vals_seq data structure,
        solve for the internal parameters of the FunctionApprox
        such that the internal parameters are fitted to xy_vals_seq.
        Since this is a best-fit, the internal parameters are fitted
        to within the input error_tolerance (where applicable, since
        some methods involve a direct solve for the fit that don't
        require an error_tolerance)
        '''

    @abstractmethod
    def within(self: F, other: F, tolerance: float) -> bool:
        '''Is this function approximation within a given tolerance of
        another function approximation of the same type?
        '''

    def iterate_updates(
        self: F,
        xy_seq_stream: Iterator[Iterable[Tuple[X, float]]]
    ) -> Iterator[F]:
        '''Given a stream (Iterator) of data sets of (x,y) pairs,
        perform a series of incremental updates to the internal
        parameters (using update method), with each internal
        parameter update done for each data set of (x,y) pairs in the
        input stream of xy_seq_stream
        '''
        return iterate.accumulate(
            xy_seq_stream,
            lambda fa, xy: fa.update(xy),
            initial=self
        )

    def rmse(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> float:
        '''The Root-Mean-Squared-Error between FunctionApprox's
        predictions (from evaluate) and the associated (supervisory)
        y values
        '''
        x_seq, y_seq = zip(*xy_vals_seq)
        errors: np.ndarray = self.evaluate(x_seq) - np.array(y_seq)
        return np.sqrt(np.mean(errors * errors))

    def argmax(self, xs: Iterable[X]) -> X:
        '''Return the input X that maximizes the function being approximated.
        Arguments:
          xs -- list of inputs to evaluate and maximize, cannot be empty
        Returns the X that maximizes the function this approximates.
        '''
        args: Sequence[X] = list(xs)
        return args[np.argmax(self.evaluate(args))]


@dataclass(frozen=True)
class Gradient(Generic[F]):
    function_approx: F

    @overload
    def __add__(self, x: Gradient[F]) -> Gradient[F]:
        ...

    @overload
    def __add__(self, x: F) -> F:
        ...

    def __add__(self, x):
        if isinstance(x, Gradient):
            return Gradient(self.function_approx + x.function_approx)

        return self.function_approx + x

    def __mul__(self: Gradient[F], x: float) -> Gradient[F]:
        return Gradient(self.function_approx * x)

    def zero(self) -> Gradient[F]:
        return Gradient(self.function_approx * 0.0)


@dataclass(frozen=True)
class Dynamic(FunctionApprox[X]):
    '''A FunctionApprox that works exactly the same as exact dynamic
    programming. Each update for a value in X replaces the previous
    value at X altogether.

    Fields:
    values_map -- mapping from X to its approximated value
    '''

    values_map: Mapping[X, float]

    def __add__(self, other: Dynamic[X]) -> Dynamic[X]:
        d: Dict[X, float] = {}
        for key in set.union(
            set(self.values_map.keys()),
            set(other.values_map.keys())
        ):
            d[key] = self.values_map.get(key, 0.) + \
                other.values_map.get(key, 0.)
        return Dynamic(values_map=d)

    def __mul__(self, scalar: float) -> Dynamic[X]:
        return Dynamic(
            values_map={x: scalar * y for x, y in self.values_map.items()}
        )

    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], np.ndarray]
    ) -> Gradient[Dynamic[X]]:
        x_vals, y_vals = zip(*xy_vals_seq)
        obj_deriv_out: np.ndarray = obj_deriv_out_fun(x_vals, y_vals)
        d: Dict[X, float] = {}
        for x, o in zip(x_vals, obj_deriv_out):
            d[x] = o
        return Gradient(Dynamic(values_map=d))

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Evaluate the function approximation by looking up the value in the
        mapping for each state.

        Will raise an error if an X value has not been seen before and
        was not initialized.

        '''
        return np.array([self.values_map.get(x, 0.0) for x in x_values_seq])

    def update_with_gradient(
        self,
        gradient: Gradient[Dynamic[X]]
    ) -> Dynamic[X]:
        d: Dict[X, float] = dict(self.values_map)
        for key, val in gradient.function_approx.values_map.items():
            d[key] = d.get(key, 0.) - val
        return replace(
            self,
            values_map=d
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> Dynamic[X]:
        return replace(self, values_map=dict(xy_vals_seq))

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        '''This approximation is within a tolerance of another if the value
        for each X in both approximations is within the given
        tolerance.

        Raises an error if the other approximation is missing states
        that this approximation has.

        '''
        if not isinstance(other, Dynamic):
            return False

        return all(abs(self.values_map[s] - other.values_map.get(s, 0.))
                   <= tolerance for s in self.values_map)


@dataclass(frozen=True)
class Tabular(FunctionApprox[X]):
    '''Approximates a function with a discrete domain (`X'), without any
    interpolation. The value for each `X' is maintained as a weighted
    mean of observations by recency (managed by
    `count_to_weight_func').

    In practice, this means you can use this to approximate a function
    with a learning rate α(n) specified by count_to_weight_func.

    If `count_to_weight_func' always returns 1, this behaves the same
    way as `Dynamic'.

    Fields:
    values_map -- mapping from X to its approximated value
    counts_map -- how many times a given X has been updated
    count_to_weight_func -- function for how much to weigh an update
      to X based on the number of times that X has been updated

    '''

    values_map: Mapping[X, float] = field(default_factory=lambda: {})
    counts_map: Mapping[X, int] = field(default_factory=lambda: {})
    count_to_weight_func: Callable[[int], float] = \
        field(default_factory=lambda: lambda n: 1.0 / n)

    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], float]
    ) -> Gradient[Tabular[X]]:
        x_vals, y_vals = zip(*xy_vals_seq)
        obj_deriv_out: np.ndarray = obj_deriv_out_fun(x_vals, y_vals)
        sums_map: Dict[X, float] = defaultdict(float)
        counts_map: Dict[X, int] = defaultdict(int)
        for x, o in zip(x_vals, obj_deriv_out):
            sums_map[x] += o
            counts_map[x] += 1
        return Gradient(replace(
            self,
            values_map={x: sums_map[x] / counts_map[x] for x in sums_map},
            counts_map=counts_map
        ))

    def __add__(self, other: Tabular[X]) -> Tabular[X]:
        values_map: Dict[X, float] = {}
        counts_map: Dict[X, int] = {}
        for key in set.union(
                set(self.values_map.keys()),
                set(other.values_map.keys())
        ):
            values_map[key] = self.values_map.get(key, 0.) + \
                other.values_map.get(key, 0.)
            counts_map[key] = counts_map.get(key, 0) + \
                other.counts_map.get(key, 0)
        return replace(
            self,
            values_map=values_map,
            counts_map=counts_map
        )

    def __mul__(self, scalar: float) -> Tabular[X]:
        return replace(
            self,
            values_map={x: scalar * y for x, y in self.values_map.items()}
        )

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        '''Evaluate the function approximation by looking up the value in the
        mapping for each state.

        if an X value has not been seen before and hence not initialized,
        returns 0

        '''
        return np.array([self.values_map.get(x, 0.) for x in x_values_seq])

    def update_with_gradient(
        self,
        gradient: Gradient[Tabular[X]]
    ) -> Tabular[X]:
        '''Update the approximation with the given gradient.
        Each X keeps a count n of how many times it was updated, and
        each subsequent update is scaled by count_to_weight_func(n),
        which defines our learning rate.

        '''
        values_map: Dict[X, float] = dict(self.values_map)
        counts_map: Dict[X, int] = dict(self.counts_map)
        for key in gradient.function_approx.values_map:
            counts_map[key] = counts_map.get(key, 0) + \
                gradient.function_approx.counts_map[key]
            weight: float = self.count_to_weight_func(counts_map[key])
            values_map[key] = values_map.get(key, 0.) - \
                weight * gradient.function_approx.values_map[key]
        return replace(
            self,
            values_map=values_map,
            counts_map=counts_map
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> Tabular[X]:
        values_map: Dict[X, float] = {}
        counts_map: Dict[X, int] = {}
        for x, y in xy_vals_seq:
            counts_map[x] = counts_map.get(x, 0) + 1
            weight: float = self.count_to_weight_func(counts_map[x])
            values_map[x] = weight * y + (1 - weight) * values_map.get(x, 0.)
        return replace(
            self,
            values_map=values_map,
            counts_map=counts_map
        )

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, Tabular):
            return all(abs(self.values_map[s] - other.values_map.get(s, 0.))
                       <= tolerance for s in self.values_map)
        return False


# @dataclass(frozen=True)
# class BSplineApprox(FunctionApprox[X]):
#     feature_function: Callable[[X], float]
#     degree: int
#     knots: np.ndarray = field(default_factory=lambda: np.array([]))
#     coeffs: np.ndarray = field(default_factory=lambda: np.array([]))
# 
#     def get_feature_values(self, x_values_seq: Iterable[X]) -> Sequence[float]:
#         return [self.feature_function(x) for x in x_values_seq]
# 
#     def representational_gradient(self, x_value: X) -> BSplineApprox[X]:
#         feature_val: float = self.feature_function(x_value)
#         eps: float = 1e-6
#         one_hots: np.array = np.eye(len(self.coeffs))
#         return replace(
#             self,
#             coeffs=np.array([(
#                 BSpline(
#                     self.knots,
#                     c + one_hots[i] * eps,
#                     self.degree
#                 )(feature_val) -
#                 BSpline(
#                     self.knots,
#                     c - one_hots[i] * eps,
#                     self.degree
#                 )(feature_val)
#             ) / (2 * eps) for i, c in enumerate(self.coeffs)]))
# 
#     def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
#         spline_func: Callable[[Sequence[float]], np.ndarray] = \
#             BSpline(self.knots, self.coeffs, self.degree)
#         return spline_func(self.get_feature_values(x_values_seq))
# 
#     def update(
#         self,
#         xy_vals_seq: Iterable[Tuple[X, float]]
#     ) -> BSplineApprox[X]:
#         x_vals, y_vals = zip(*xy_vals_seq)
#         feature_vals: Sequence[float] = self.get_feature_values(x_vals)
#         sorted_pairs: Sequence[Tuple[float, float]] = \
#             sorted(zip(feature_vals, y_vals), key=itemgetter(0))
#         new_knots, new_coeffs, _ = splrep(
#             [f for f, _ in sorted_pairs],
#             [y for _, y in sorted_pairs],
#             k=self.degree
#         )
#         return replace(
#             self,
#             knots=new_knots,
#             coeffs=new_coeffs
#         )
# 
#     def solve(
#         self,
#         xy_vals_seq: Iterable[Tuple[X, float]],
#         error_tolerance: Optional[float] = None
#     ) -> BSplineApprox[X]:
#         return self.update(xy_vals_seq)
# 
#     def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
#         if isinstance(other, BSplineApprox):
#             return \
#                 np.all(np.abs(self.knots - other.knots) <= tolerance).item() \
#                 and \
#                 np.all(np.abs(self.coeffs - other.coeffs) <= tolerance).item()
# 
#         return False
# 
# 
@dataclass(frozen=True)
class AdamGradient:
    learning_rate: float
    decay1: float
    decay2: float

    @staticmethod
    def default_settings() -> AdamGradient:
        return AdamGradient(
            learning_rate=0.001,
            decay1=0.9,
            decay2=0.999
        )


@dataclass(frozen=True)
class Weights:
    adam_gradient: AdamGradient
    time: int
    weights: np.ndarray
    adam_cache1: np.ndarray
    adam_cache2: np.ndarray

    @staticmethod
    def create(
        weights: np.ndarray,
        adam_gradient: AdamGradient = AdamGradient.default_settings(),
        adam_cache1: Optional[np.ndarray] = None,
        adam_cache2: Optional[np.ndarray] = None
    ) -> Weights:
        return Weights(
            adam_gradient=adam_gradient,
            time=0,
            weights=weights,
            adam_cache1=np.zeros_like(
                weights
            ) if adam_cache1 is None else adam_cache1,
            adam_cache2=np.zeros_like(
                weights
            ) if adam_cache2 is None else adam_cache2
        )

    def update(self, gradient: np.ndarray) -> Weights:
        time: int = self.time + 1
        new_adam_cache1: np.ndarray = self.adam_gradient.decay1 * \
            self.adam_cache1 + (1 - self.adam_gradient.decay1) * gradient
        new_adam_cache2: np.ndarray = self.adam_gradient.decay2 * \
            self.adam_cache2 + (1 - self.adam_gradient.decay2) * gradient ** 2
        corrected_m: np.ndarray = new_adam_cache1 / \
            (1 - self.adam_gradient.decay1 ** time)
        corrected_v: np.ndarray = new_adam_cache2 / \
            (1 - self.adam_gradient.decay2 ** time)

        new_weights: np.ndarray = self.weights - \
            self.adam_gradient.learning_rate * corrected_m / \
            (np.sqrt(corrected_v) + SMALL_NUM)

        return replace(
            self,
            time=time,
            weights=new_weights,
            adam_cache1=new_adam_cache1,
            adam_cache2=new_adam_cache2,
        )

    def within(self, other: Weights, tolerance: float) -> bool:
        return np.all(np.abs(self.weights - other.weights) <= tolerance).item()


@dataclass(frozen=True)
class LinearFunctionApprox(FunctionApprox[X]):

    feature_functions: Sequence[Callable[[X], float]]
    regularization_coeff: float
    weights: Weights
    direct_solve: bool

    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        adam_gradient: AdamGradient = AdamGradient.default_settings(),
        regularization_coeff: float = 0.,
        weights: Optional[Weights] = None,
        direct_solve: bool = True
    ) -> LinearFunctionApprox[X]:
        return LinearFunctionApprox(
            feature_functions=feature_functions,
            regularization_coeff=regularization_coeff,
            weights=Weights.create(
                adam_gradient=adam_gradient,
                weights=np.zeros(len(feature_functions))
            ) if weights is None else weights,
            direct_solve=direct_solve
        )

    def get_feature_values(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.array(
            [[f(x) for f in self.feature_functions] for x in x_values_seq]
        )

    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], float]
    ) -> Gradient[LinearFunctionApprox[X]]:
        x_vals, y_vals = zip(*xy_vals_seq)
        obj_deriv_out: np.ndarray = obj_deriv_out_fun(x_vals, y_vals)
        features: np.ndarray = self.get_feature_values(x_vals)
        gradient: np.ndarray = \
            features.T.dot(obj_deriv_out) / len(obj_deriv_out) \
            + self.regularization_coeff * self.weights.weights
        return Gradient(replace(
            self,
            weights=replace(
                self.weights,
                weights=gradient
            )
        ))

    def __add__(self, other: LinearFunctionApprox[X]) -> \
            LinearFunctionApprox[X]:
        return replace(
            self,
            weights=replace(
                self.weights,
                weights=self.weights.weights + other.weights.weights
            )
        )

    def __mul__(self, scalar: float) -> LinearFunctionApprox[X]:
        return replace(
            self,
            weights=replace(
                self.weights,
                weights=self.weights.weights * scalar
            )
        )

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.dot(
            self.get_feature_values(x_values_seq),
            self.weights.weights
        )

    def update_with_gradient(
        self,
        gradient: Gradient[LinearFunctionApprox[X]]
    ) -> LinearFunctionApprox[X]:
        return replace(
            self,
            weights=self.weights.update(
                gradient.function_approx.weights.weights
            )
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> LinearFunctionApprox[X]:
        if self.direct_solve:
            x_vals, y_vals = zip(*xy_vals_seq)
            feature_vals: np.ndarray = self.get_feature_values(x_vals)
            feature_vals_T: np.ndarray = feature_vals.T
            left: np.ndarray = np.dot(feature_vals_T, feature_vals) \
                + feature_vals.shape[0] * self.regularization_coeff * \
                np.eye(len(self.weights.weights))
            right: np.ndarray = np.dot(feature_vals_T, y_vals)
            ret = replace(
                self,
                weights=Weights.create(
                    adam_gradient=self.weights.adam_gradient,
                    weights=np.linalg.solve(left, right)
                )
            )
        else:
            tol: float = 1e-6 if error_tolerance is None else error_tolerance

            def done(
                a: LinearFunctionApprox[X],
                b: LinearFunctionApprox[X],
                tol: float = tol
            ) -> bool:
                return a.within(b, tol)

            ret = iterate.converged(
                self.iterate_updates(itertools.repeat(list(xy_vals_seq))),
                done=done
            )

        return ret

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, LinearFunctionApprox):
            return self.weights.within(other.weights, tolerance)

        return False


@dataclass(frozen=True)
class DNNSpec:
    neurons: Sequence[int]
    bias: bool
    hidden_activation: Callable[[np.ndarray], np.ndarray]
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray]
    output_activation: Callable[[np.ndarray], np.ndarray]
    output_activation_deriv: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DNNApprox(FunctionApprox[X]):

    feature_functions: Sequence[Callable[[X], float]]
    dnn_spec: DNNSpec
    regularization_coeff: float
    weights: Sequence[Weights]

    @staticmethod
    def create(
        feature_functions: Sequence[Callable[[X], float]],
        dnn_spec: DNNSpec,
        adam_gradient: AdamGradient = AdamGradient.default_settings(),
        regularization_coeff: float = 0.,
        weights: Optional[Sequence[Weights]] = None
    ) -> DNNApprox[X]:
        if weights is None:
            inputs: Sequence[int] = [len(feature_functions)] + \
                [n + (1 if dnn_spec.bias else 0)
                 for i, n in enumerate(dnn_spec.neurons)]
            outputs: Sequence[int] = list(dnn_spec.neurons) + [1]
            wts = [Weights.create(
                weights=np.random.randn(output, inp) / np.sqrt(inp),
                adam_gradient=adam_gradient
            ) for inp, output in zip(inputs, outputs)]
        else:
            wts = weights

        return DNNApprox(
            feature_functions=feature_functions,
            dnn_spec=dnn_spec,
            regularization_coeff=regularization_coeff,
            weights=wts
        )

    def get_feature_values(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return np.array(
            [[f(x) for f in self.feature_functions] for x in x_values_seq]
        )

    def forward_propagation(
        self,
        x_values_seq: Iterable[X]
    ) -> Sequence[np.ndarray]:
        """
        :param x_values_seq: a n-length iterable of input points
        :return: list of length (L+2) where the first (L+1) values
                 each represent the 2-D input arrays (of size n x |i_l|),
                 for each of the (L+1) layers (L of which are hidden layers),
                 and the last value represents the output of the DNN (as a
                 1-D array of length n)
        """
        inp: np.ndarray = self.get_feature_values(x_values_seq)
        ret: List[np.ndarray] = [inp]
        for w in self.weights[:-1]:
            out: np.ndarray = self.dnn_spec.hidden_activation(
                np.dot(inp, w.weights.T)
            )
            if self.dnn_spec.bias:
                inp = np.insert(out, 0, 1., axis=1)
            else:
                inp = out
            ret.append(inp)
        ret.append(
            self.dnn_spec.output_activation(
                np.dot(inp, self.weights[-1].weights.T)
            )[:, 0]
        )
        return ret

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        return self.forward_propagation(x_values_seq)[-1]

    def backward_propagation(
        self,
        fwd_prop: Sequence[np.ndarray],
        obj_deriv_out: np.ndarray
    ) -> Sequence[np.ndarray]:
        """
        :param fwd_prop represents the result of forward propagation (without
        the final output), a sequence of L 2-D np.ndarrays of the DNN.
        : param obj_deriv_out represents the derivative of the objective
        function with respect to the linear predictor of the final layer.

        :return: list (of length L+1) of |o_l| x |i_l| 2-D arrays,
                 i.e., same as the type of self.weights.weights
        This function computes the gradient (with respect to weights) of
        the objective where the output layer activation function
        is the canonical link function of the conditional distribution of y|x
        """
        deriv: np.ndarray = obj_deriv_out.reshape(1, -1)
        back_prop: List[np.ndarray] = [np.dot(deriv, fwd_prop[-1]) /
                                       deriv.shape[1]]
        # L is the number of hidden layers, n is the number of points
        # layer l deriv represents dObj/ds_l where s_l = i_l . weights_l
        # (s_l is the result of applying layer l without the activation func)
        for i in reversed(range(len(self.weights) - 1)):
            # deriv_l is a 2-D array of dimension |o_l| x n
            # The recursive formulation of deriv is as follows:
            # deriv_{l-1} = (weights_l^T inner deriv_l) haddamard g'(s_{l-1}),
            # which is ((|i_l| x |o_l|) inner (|o_l| x n)) haddamard
            # (|i_l| x n), which is (|i_l| x n) = (|o_{l-1}| x n)
            # Note: g'(s_{l-1}) is expressed as hidden layer activation
            # derivative as a function of o_{l-1} (=i_l).
            deriv = np.dot(self.weights[i + 1].weights.T, deriv) * \
                self.dnn_spec.hidden_activation_deriv(fwd_prop[i + 1].T)
            # If self.dnn_spec.bias is True, then i_l = o_{l-1} + 1, in which
            # case # the first row of the calculated deriv is removed to yield
            # a 2-D array of dimension |o_{l-1}| x n.
            if self.dnn_spec.bias:
                deriv = deriv[1:]
            # layer l gradient is deriv_l inner fwd_prop[l], which is
            # of dimension (|o_l| x n) inner (n x (|i_l|) = |o_l| x |i_l|
            back_prop.append(np.dot(deriv, fwd_prop[i]) / deriv.shape[1])
        return back_prop[::-1]

    def objective_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        obj_deriv_out_fun: Callable[[Sequence[X], Sequence[float]], float]
    ) -> Gradient[DNNApprox[X]]:
        x_vals, y_vals = zip(*xy_vals_seq)
        obj_deriv_out: np.ndarray = obj_deriv_out_fun(x_vals, y_vals)
        fwd_prop: Sequence[np.ndarray] = self.forward_propagation(x_vals)[:-1]
        gradient: Sequence[np.ndarray] = \
            [x + self.regularization_coeff * self.weights[i].weights
             for i, x in enumerate(self.backward_propagation(
                 fwd_prop=fwd_prop,
                 obj_deriv_out=obj_deriv_out
             ))]
        return Gradient(replace(
            self,
            weights=[replace(w, weights=g) for
                     w, g in zip(self.weights, gradient)]
        ))

    def __add__(self, other: DNNApprox[X]) -> DNNApprox[X]:
        return replace(
            self,
            weights=[replace(w, weights=w.weights + o.weights) for
                     w, o in zip(self.weights, other.weights)]
        )

    def __mul__(self, scalar: float) -> DNNApprox[X]:
        return replace(
            self,
            weights=[replace(w, weights=w.weights * scalar)
                     for w in self.weights]
        )

    def update_with_gradient(
        self,
        gradient: Gradient[DNNApprox[X]]
    ) -> DNNApprox[X]:
        return replace(
            self,
            weights=[w.update(g.weights) for w, g in
                     zip(self.weights, gradient.function_approx.weights)]
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> DNNApprox[X]:
        tol: float = 1e-6 if error_tolerance is None else error_tolerance

        def done(
            a: DNNApprox[X],
            b: DNNApprox[X],
            tol: float = tol
        ) -> bool:
            return a.within(b, tol)

        return iterate.converged(
            self.iterate_updates(itertools.repeat(list(xy_vals_seq))),
            done=done
        )

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, DNNApprox):
            return all(w1.within(w2, tolerance)
                       for w1, w2 in zip(self.weights, other.weights))
        else:
            return False


def learning_rate_schedule(
    initial_learning_rate: float,
    half_life: float,
    exponent: float
) -> Callable[[int], float]:
    def lr_func(n: int) -> float:
        return initial_learning_rate * (1 + (n - 1) / half_life) ** -exponent
    return lr_func


if __name__ == '__main__':

    from scipy.stats import norm
    from pprint import pprint

    alpha = 2.0
    beta_1 = 10.0
    beta_2 = 4.0
    beta_3 = -6.0
    beta = (beta_1, beta_2, beta_3)

    x_pts = np.arange(-10.0, 10.5, 0.5)
    y_pts = np.arange(-10.0, 10.5, 0.5)
    z_pts = np.arange(-10.0, 10.5, 0.5)
    pts: Sequence[Tuple[float, float, float]] = \
        [(x, y, z) for x in x_pts for y in y_pts for z in z_pts]

    def superv_func(pt):
        return alpha + np.dot(beta, pt)

    n = norm(loc=0., scale=2.)
    xy_vals_seq: Sequence[Tuple[Tuple[float, float, float], float]] = \
        [(x, superv_func(x) + n.rvs(size=1)[0]) for x in pts]

    ag = AdamGradient(
        learning_rate=0.5,
        decay1=0.9,
        decay2=0.999
    )
    ffs = [
        lambda _: 1.,
        lambda x: x[0],
        lambda x: x[1],
        lambda x: x[2]
    ]

    lfa = LinearFunctionApprox.create(
         feature_functions=ffs,
         adam_gradient=ag,
         regularization_coeff=0.001,
         direct_solve=True
    )

    lfa_ds = lfa.solve(xy_vals_seq)
    print("Direct Solve")
    pprint(lfa_ds.weights)
    errors: np.ndarray = lfa_ds.evaluate(pts) - \
        np.array([y for _, y in xy_vals_seq])
    print("Mean Squared Error")
    pprint(np.mean(errors * errors))
    print()

    print("Linear Gradient Solve")
    for _ in range(100):
        print("Weights")
        pprint(lfa.weights)
        errors: np.ndarray = lfa.evaluate(pts) - \
            np.array([y for _, y in xy_vals_seq])
        print("Mean Squared Error")
        pprint(np.mean(errors * errors))
        lfa = lfa.update(xy_vals_seq)
        print()

    ds = DNNSpec(
        neurons=[2],
        bias=True,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda x: np.ones_like(x),
        output_activation=lambda x: x,
        output_activation_deriv=lambda x: np.ones_like(x)
    )

    dnna = DNNApprox.create(
        feature_functions=ffs,
        dnn_spec=ds,
        adam_gradient=ag,
        regularization_coeff=0.01
    )
    print("DNN Gradient Solve")
    for _ in range(100):
        print("Weights")
        pprint(dnna.weights)
        errors: np.ndarray = dnna.evaluate(pts) - \
            np.array([y for _, y in xy_vals_seq])
        print("Mean Squared Error")
        pprint(np.mean(errors * errors))
        dnna = dnna.update(xy_vals_seq)
        print()
```

--------------------------------------------------------------------------------

# PARTIE 3 - PROGRAMMATION DYNAMIQUE

================================================================================

## 📄 approximate_dynamic_programming.py {#approximate-dynamic-programming}

**Titre**: Programmation Dynamique Exacte

**Description**: Algorithmes DP classiques pour espaces finis

**Lignes de code**: 324

**Concepts clés**:
- evaluate_mrp - Évaluation de politique (résout V = R + γPV)
- greedy_policy_from_vf - Amélioration de politique
- policy_iteration - Itération de politique complète
- value_iteration - Itération de valeur (Bellman optimality)
- Convergence garantie pour espaces finis

**🎯 Utilisation HelixOne**: Baseline pour petits problèmes

### Code Source Complet

```python
'''Approximate dynamic programming algorithms are variations on
dynamic programming algorithms that can work with function
approximations rather than exact representations of the process's
state space.

'''

from typing import Iterator, Tuple, TypeVar, Sequence, List
from operator import itemgetter
import numpy as np

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
from rl.iterate import iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition, NonTerminal, State)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess,
                                        StateActionMapping)
from rl.policy import DeterministicPolicy

S = TypeVar('S')
A = TypeVar('A')

# A representation of a value function for a finite MDP with states of
# type S
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]


def extended_vf(vf: ValueFunctionApprox[S], s: State[S]) -> float:
    return s.on_non_terminal(vf, 0.0)


def evaluate_finite_mrp(
        mrp: FiniteMarkovRewardProcess[S],
        γ: float,
        approx_0: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:

    '''Iteratively calculate the value function for the give finite Markov
    Reward Process, using the given FunctionApprox to approximate the
    value function at each step.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        vs: np.ndarray = v.evaluate(mrp.non_terminal_states)
        updated: np.ndarray = mrp.reward_function_vec + γ * \
            mrp.get_transition_matrix().dot(vs)
        return v.update(zip(mrp.non_terminal_states, updated))

    return iterate(update, approx_0)


def evaluate_mrp(
    mrp: MarkovRewardProcess[S],
    γ: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int
) -> Iterator[ValueFunctionApprox[S]]:

    '''Iteratively calculate the value function for the given Markov Reward
    Process, using the given FunctionApprox to approximate the value function
    at each step for a random sample of the process' non-terminal states.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        nt_states: Sequence[NonTerminal[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(s, mrp.transition_reward(s).expectation(return_))
             for s in nt_states]
        )

    return iterate(update, approx_0)


def value_iteration_finite(
    mdp: FiniteMarkovDecisionProcess[S, A],
    γ: float,
    approx_0: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given finite
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(
                s,
                max(mdp.mapping[s][a].expectation(return_)
                    for a in mdp.actions(s))
            ) for s in mdp.non_terminal_states]
        )

    return iterate(update, approx_0)


def value_iteration(
    mdp: MarkovDecisionProcess[S, A],
    γ: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int
) -> Iterator[ValueFunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step for a random sample of the process'
    non-terminal states.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        nt_states: Sequence[NonTerminal[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(s, max(mdp.step(s, a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in nt_states]
        )

    return iterate(update, approx_0)


def backward_evaluate_finite(
    step_f0_pairs: Sequence[Tuple[RewardTransition[S],
                                  ValueFunctionApprox[S]]],
    γ: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate the given finite Markov Reward Process using backwards
    induction, given that the process stops after limit time steps.

    '''

    v: List[ValueFunctionApprox[S]] = []

    for i, (step, approx0) in enumerate(reversed(step_f0_pairs)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(v[i-1], s1) if i > 0 else 0.)

        v.append(
            approx0.solve([(s, res.expectation(return_))
                           for s, res in step.items()])
        )

    return reversed(v)


MRP_FuncApprox_Distribution = Tuple[MarkovRewardProcess[S],
                                    ValueFunctionApprox[S],
                                    NTStateDistribution[S]]


def backward_evaluate(
    mrp_f0_mu_triples: Sequence[MRP_FuncApprox_Distribution[S]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate the given finite Markov Reward Process using backwards
    induction, given that the process stops after limit time steps, using
    the given FunctionApprox for each time step for a random sample of the
    time step's states.

    '''
    v: List[ValueFunctionApprox[S]] = []

    for i, (mrp, approx0, mu) in enumerate(reversed(mrp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(v[i-1], s1) if i > 0 else 0.)

        v.append(
            approx0.solve(
                [(s, mrp.transition_reward(s).expectation(return_))
                 for s in mu.sample_n(num_state_samples)],
                error_tolerance
            )
        )

    return reversed(v)


def back_opt_vf_and_policy_finite(
    step_f0s: Sequence[Tuple[StateActionMapping[S, A],
                             ValueFunctionApprox[S]]],
    γ: float,
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step

    '''
    vp: List[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]] = []

    for i, (step, approx0) in enumerate(reversed(step_f0s)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)

        this_v = approx0.solve(
            [(s, max(res.expectation(return_)
                     for a, res in actions_map.items()))
             for s, actions_map in step.items()]
        )

        def deter_policy(state: S) -> A:
            return max(
                ((res.expectation(return_), a) for a, res in
                 step[NonTerminal(state)].items()),
                key=itemgetter(0)
            )[1]

        vp.append((this_v, DeterministicPolicy(deter_policy)))

    return reversed(vp)


MDP_FuncApproxV_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    ValueFunctionApprox[S],
    NTStateDistribution[S]
]


def back_opt_vf_and_policy(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxV_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step, using the given FunctionApprox for each time step
    for a random sample of the time step's states.

    '''
    vp: List[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)

        this_v = approx0.solve(
            [(s, max(mdp.step(s, a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in mu.sample_n(num_state_samples)],
            error_tolerance
        )

        def deter_policy(state: S) -> A:
            return max(
                ((mdp.step(NonTerminal(state), a).expectation(return_), a)
                 for a in mdp.actions(NonTerminal(state))),
                key=itemgetter(0)
            )[1]

        vp.append((this_v, DeterministicPolicy(deter_policy)))

    return reversed(vp)


MDP_FuncApproxQ_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    QValueFunctionApprox[S, A],
    NTStateDistribution[S]
]


def back_opt_qvf(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxQ_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    '''Use backwards induction to find the optimal q-value function  policy at
    each time step, using the given FunctionApprox (for Q-Value) for each time
    step for a random sample of the time step's states.

    '''
    horizon: int = len(mdp_f0_mu_triples)
    qvf: List[QValueFunctionApprox[S, A]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            next_return: float = max(
                qvf[i-1]((s1, a)) for a in
                mdp_f0_mu_triples[horizon - i][0].actions(s1)
            ) if i > 0 and isinstance(s1, NonTerminal) else 0.
            return r + γ * next_return

        this_qvf = approx0.solve(
            [((s, a), mdp.step(s, a).expectation(return_))
             for s in mu.sample_n(num_state_samples) for a in mdp.actions(s)],
            error_tolerance
        )

        qvf.append(this_qvf)

    return reversed(qvf)
```

--------------------------------------------------------------------------------

## 📄 dynamic_programming.py {#dynamic-programming}

**Titre**: Programmation Dynamique Exacte

**Description**: Algorithmes DP classiques pour espaces finis

**Lignes de code**: 197

**Concepts clés**:
- evaluate_mrp - Évaluation de politique (résout V = R + γPV)
- greedy_policy_from_vf - Amélioration de politique
- policy_iteration - Itération de politique complète
- value_iteration - Itération de valeur (Bellman optimality)
- Convergence garantie pour espaces finis

**🎯 Utilisation HelixOne**: Baseline pour petits problèmes

### Code Source Complet

```python
import operator
from typing import Mapping, Iterator, TypeVar, Tuple, Dict

import numpy as np

from rl.distribution import Categorical, Choose
from rl.iterate import converged, iterate
from rl.markov_process import NonTerminal, State
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        FiniteMarkovRewardProcess)
from rl.policy import FinitePolicy, FiniteDeterministicPolicy

A = TypeVar('A')
S = TypeVar('S')

DEFAULT_TOLERANCE = 1e-5

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[NonTerminal[S], float]


def extended_vf(v: V[S], s: State[S]) -> float:
    def non_terminal_vf(st: NonTerminal[S], v=v) -> float:
        return v[st]
    return s.on_non_terminal(non_terminal_vf, 0.0)


def evaluate_mrp(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> Iterator[np.ndarray]:
    '''Iteratively calculate the value function for the give Markov reward
    process.

    '''
    def update(v: np.ndarray) -> np.ndarray:
        return mrp.reward_function_vec + gamma * \
            mrp.get_transition_matrix().dot(v)

    v_0: np.ndarray = np.zeros(len(mrp.non_terminal_states))

    return iterate(update, v_0)


def almost_equal_np_arrays(
    v1: np.ndarray,
    v2: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    '''Return whether the two value functions as np.ndarray are within the
    given tolerance of each other.

    '''
    return max(abs(v1 - v2)) < tolerance


def evaluate_mrp_result(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float
) -> V[S]:
    v_star: np.ndarray = converged(
        evaluate_mrp(mrp, gamma=gamma),
        done=almost_equal_np_arrays
    )
    return {s: v_star[i] for i, s in enumerate(mrp.non_terminal_states)}


def greedy_policy_from_vf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: V[S],
    gamma: float
) -> FiniteDeterministicPolicy[S, A]:
    greedy_policy_dict: Dict[S, A] = {}

    for s in mdp.non_terminal_states:
        q_values: Iterator[Tuple[A, float]] = \
            ((a, mdp.mapping[s][a].expectation(
                lambda s_r: s_r[1] + gamma * extended_vf(vf, s_r[0])
            )) for a in mdp.actions(s))
        greedy_policy_dict[s.state] = \
            max(q_values, key=operator.itemgetter(1))[0]

    return FiniteDeterministicPolicy(greedy_policy_dict)


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    matrix_method_for_mrp_eval: bool = False
) -> Iterator[Tuple[V[S], FinitePolicy[S, A]]]:
    '''Calculate the value function (V*) of the given MDP by improving
    the policy repeatedly after evaluating the value function for a policy
    '''

    def update(vf_policy: Tuple[V[S], FinitePolicy[S, A]])\
            -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:

        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        policy_vf: V[S] = {mrp.non_terminal_states[i]: v for i, v in
                           enumerate(mrp.get_value_function_vec(gamma))}\
            if matrix_method_for_mrp_eval else evaluate_mrp_result(mrp, gamma)
        improved_pi: FiniteDeterministicPolicy[S, A] = greedy_policy_from_vf(
            mdp,
            policy_vf,
            gamma
        )

        return policy_vf, improved_pi

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    pi_0: FinitePolicy[S, A] = FinitePolicy(
        {s.state: Choose(mdp.actions(s)) for s in mdp.non_terminal_states}
    )
    return iterate(update, (v_0, pi_0))


def almost_equal_vf_pis(
    x1: Tuple[V[S], FinitePolicy[S, A]],
    x2: Tuple[V[S], FinitePolicy[S, A]]
) -> bool:
    return max(
        abs(x1[0][s] - x2[0][s]) for s in x1[0]
    ) < DEFAULT_TOLERANCE


def policy_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    return converged(policy_iteration(mdp, gamma), done=almost_equal_vf_pis)


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Iterator[V[S]]:
    '''Calculate the value function (V*) of the given MDP by applying the
    update function repeatedly until the values converge.

    '''
    def update(v: V[S]) -> V[S]:
        return {s: max(mdp.mapping[s][a].expectation(
            lambda s_r: s_r[1] + gamma * extended_vf(v, s_r[0])
        ) for a in mdp.actions(s)) for s in v}

    v_0: V[S] = {s: 0.0 for s in mdp.non_terminal_states}
    return iterate(update, v_0)


def almost_equal_vfs(
    v1: V[S],
    v2: V[S],
    tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    '''Return whether the two value function tables are within the given
    tolerance of each other.

    '''
    return max(abs(v1[s] - v2[s]) for s in v1) < tolerance


def value_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    opt_vf: V[S] = converged(
        value_iteration(mdp, gamma),
        done=almost_equal_vfs
    )
    opt_policy: FiniteDeterministicPolicy[S, A] = greedy_policy_from_vf(
        mdp,
        opt_vf,
        gamma
    )

    return opt_vf, opt_policy


if __name__ == '__main__':

    from pprint import pprint

    transition_reward_map = {
        1: Categorical({(1, 7.0): 0.6, (2, 5.0): 0.3, (3, 2.0): 0.1}),
        2: Categorical({(1, -2.0): 0.1, (2, 4.0): 0.2, (3, 0.0): 0.7}),
        3: Categorical({(1, 3.0): 0.2, (2, 8.0): 0.6, (3, 4.0): 0.2})
    }
    gamma = 0.9

    fmrp = FiniteMarkovRewardProcess(transition_reward_map)
    fmrp.display_stationary_distribution()
    fmrp.display_reward_function()
    fmrp.display_value_function(gamma=gamma)
    pprint(evaluate_mrp_result(fmrp, gamma=gamma))
```

--------------------------------------------------------------------------------

## 📄 finite_horizon.py {#finite-horizon}

**Titre**: Problèmes à Horizon Fini

**Description**: MDP avec nombre d'étapes fixé

**Lignes de code**: 219

**Concepts clés**:
- Backward induction pour horizon fini
- V_t(s) dépend du temps restant
- Optimal stopping problems

**🎯 Utilisation HelixOne**: Options américaines, exécution avec deadline

### Code Source Complet

```python
from __future__ import annotations

from itertools import groupby
import dataclasses
from dataclasses import dataclass
from operator import itemgetter
from typing import Dict, List, Generic, Sequence, Tuple, TypeVar, Iterator

from rl.distribution import FiniteDistribution
from rl.dynamic_programming import V, extended_vf
from rl.markov_process import (FiniteMarkovRewardProcess, RewardTransition,
                               StateReward, NonTerminal, Terminal, State)
from rl.markov_decision_process import (
    ActionMapping, FiniteMarkovDecisionProcess, StateActionMapping)
from rl.policy import FiniteDeterministicPolicy

S = TypeVar('S')


@dataclass(frozen=True)
class WithTime(Generic[S]):
    '''A wrapper that augments a state of type S with a time field.

    '''
    state: S
    time: int = 0

    def step_time(self) -> WithTime[S]:
        return dataclasses.replace(self, time=self.time + 1)


RewardOutcome = FiniteDistribution[Tuple[WithTime[S], float]]


# Finite-horizon Markov reward processes
def finite_horizon_MRP(
    process: FiniteMarkovRewardProcess[S],
    limit: int
) -> FiniteMarkovRewardProcess[WithTime[S]]:
    '''Turn a normal FiniteMarkovRewardProcess into one with a finite horizon
    that stops after 'limit' steps.

    Note that this makes the data representation of the process
    larger, since we end up having distinct sets and transitions for
    every single time step up to the limit.

    '''
    transition_map: Dict[WithTime[S], RewardOutcome] = {}

    # Non-terminal states
    for time in range(limit):

        for s in process.non_terminal_states:
            result: StateReward[S] = process.transition_reward(s)
            s_time = WithTime(state=s.state, time=time)

            transition_map[s_time] = result.map(
                lambda sr: (WithTime(state=sr[0].state, time=time + 1), sr[1])
            )

    return FiniteMarkovRewardProcess(transition_map)


# TODO: Better name...
def unwrap_finite_horizon_MRP(
    process: FiniteMarkovRewardProcess[WithTime[S]]
) -> Sequence[RewardTransition[S]]:
    '''Given a finite-horizon process, break the transition between each
    time step (starting with 0) into its own data structure. This
    representation makes it easier to implement backwards
    induction.

    '''
    def time(x: WithTime[S]) -> int:
        return x.time

    def single_without_time(
        s_r: Tuple[State[WithTime[S]], float]
    ) -> Tuple[State[S], float]:
        if isinstance(s_r[0], NonTerminal):
            ret: Tuple[State[S], float] = (
                NonTerminal(s_r[0].state.state),
                s_r[1]
            )
        else:
            ret = (Terminal(s_r[0].state.state), s_r[1])
        return ret

    def without_time(arg: StateReward[WithTime[S]]) -> StateReward[S]:
        return arg.map(single_without_time)

    return [{NonTerminal(s.state): without_time(
        process.transition_reward(NonTerminal(s))
    ) for s in states} for _, states in groupby(
        sorted(
            (nt.state for nt in process.non_terminal_states),
            key=time
        ),
        key=time
    )]


def evaluate(
    steps: Sequence[RewardTransition[S]],
    gamma: float
) -> Iterator[V[S]]:
    '''Evaluate the given finite Markov reward process using backwards
    induction, given that the process stops after limit time steps.

    '''

    v: List[V[S]] = []

    for step in reversed(steps):
        v.append({s: res.expectation(
            lambda s_r: s_r[1] + gamma * (
                extended_vf(v[-1], s_r[0]) if len(v) > 0 else 0.
            )
        ) for s, res in step.items()})

    return reversed(v)


# Finite-horizon Markov decision processes

A = TypeVar('A')


def finite_horizon_MDP(
    process: FiniteMarkovDecisionProcess[S, A],
    limit: int
) -> FiniteMarkovDecisionProcess[WithTime[S], A]:
    '''Turn a normal FiniteMarkovDecisionProcess into one with a finite
    horizon that stops after 'limit' steps.

    Note that this makes the data representation of the process
    larger, since we end up having distinct sets and transitions for
    every single time step up to the limit.

    '''
    mapping: Dict[WithTime[S], Dict[A, FiniteDistribution[
        Tuple[WithTime[S], float]]]] = {}

    # Non-terminal states
    for time in range(0, limit):
        for s in process.non_terminal_states:
            s_time = WithTime(state=s.state, time=time)
            mapping[s_time] = {a: result.map(
                lambda sr: (WithTime(state=sr[0].state, time=time + 1), sr[1])
            ) for a, result in process.mapping[s].items()}

    return FiniteMarkovDecisionProcess(mapping)


def unwrap_finite_horizon_MDP(
    process: FiniteMarkovDecisionProcess[WithTime[S], A]
) -> Sequence[StateActionMapping[S, A]]:
    '''Unwrap a finite Markov decision process into a sequence of
    transitions between each time step (starting with 0). This
    representation makes it easier to implement backwards induction.

    '''
    def time(x: WithTime[S]) -> int:
        return x.time

    def single_without_time(
        s_r: Tuple[State[WithTime[S]], float]
    ) -> Tuple[State[S], float]:
        if isinstance(s_r[0], NonTerminal):
            ret: Tuple[State[S], float] = (
                NonTerminal(s_r[0].state.state),
                s_r[1]
            )
        else:
            ret = (Terminal(s_r[0].state.state), s_r[1])
        return ret

    def without_time(arg: ActionMapping[A, WithTime[S]]) -> \
            ActionMapping[A, S]:
        return {a: sr_distr.map(single_without_time)
                for a, sr_distr in arg.items()}

    return [{NonTerminal(s.state): without_time(
        process.mapping[NonTerminal(s)]
    ) for s in states} for _, states in groupby(
        sorted(
            (nt.state for nt in process.non_terminal_states),
            key=time
        ),
        key=time
    )]


def optimal_vf_and_policy(
    steps: Sequence[StateActionMapping[S, A]],
    gamma: float
) -> Iterator[Tuple[V[S], FiniteDeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step

    '''
    v_p: List[Tuple[V[S], FiniteDeterministicPolicy[S, A]]] = []

    for step in reversed(steps):
        this_v: Dict[NonTerminal[S], float] = {}
        this_a: Dict[S, A] = {}
        for s, actions_map in step.items():
            action_values = ((res.expectation(
                lambda s_r: s_r[1] + gamma * (
                    extended_vf(v_p[-1][0], s_r[0]) if len(v_p) > 0 else 0.
                )
            ), a) for a, res in actions_map.items())
            v_star, a_star = max(action_values, key=itemgetter(0))
            this_v[s] = v_star
            this_a[s.state] = a_star
        v_p.append((this_v, FiniteDeterministicPolicy(this_a)))

    return reversed(v_p)
```

--------------------------------------------------------------------------------

# PARTIE 4 - MONTE CARLO

================================================================================

## 📄 experience_replay.py {#experience-replay}

**Titre**: Experience Replay

**Description**: Buffer de replay pour apprentissage off-policy

**Lignes de code**: 48

**Concepts clés**:
- ExperienceReplayBuffer - Stocke (s,a,r,s') transitions
- Sampling uniforme ou prioritaire
- Brise les corrélations temporelles

**🎯 Utilisation HelixOne**: Stabilisation de l'apprentissage

### Code Source Complet

```python
from typing import Generic, Iterable, Iterator, List, TypeVar, Callable, \
    Sequence
from rl.distribution import Categorical

T = TypeVar('T')


class ExperienceReplayMemory(Generic[T]):
    saved_transitions: List[T]
    time_weights_func: Callable[[int], float]
    weights: List[float]
    weights_sum: float

    def __init__(
        self,
        time_weights_func: Callable[[int], float] = lambda _: 1.0,
    ):
        self.saved_transitions = []
        self.time_weights_func = time_weights_func
        self.weights = []
        self.weights_sum = 0.0

    def add_data(self, transition: T) -> None:
        self.saved_transitions.append(transition)
        weight: float = self.time_weights_func(len(self.saved_transitions) - 1)
        self.weights.append(weight)
        self.weights_sum += weight

    def sample_mini_batch(self, mini_batch_size: int) -> Sequence[T]:
        num_transitions: int = len(self.saved_transitions)
        return Categorical(
            {tr: self.weights[num_transitions - 1 - i] / self.weights_sum
             for i, tr in enumerate(self.saved_transitions)}
        ).sample_n(min(mini_batch_size, num_transitions))

    def replay(
        self,
        transitions: Iterable[T],
        mini_batch_size: int
    ) -> Iterator[Sequence[T]]:

        for transition in transitions:
            self.add_data(transition)
            yield self.sample_mini_batch(mini_batch_size)

        while True:
            yield self.sample_mini_batch(mini_batch_size)
```

--------------------------------------------------------------------------------

## 📄 monte_carlo.py {#monte-carlo}

**Titre**: Méthodes Monte Carlo

**Description**: Algorithmes MC pour évaluation et contrôle

**Lignes de code**: 143

**Concepts clés**:
- mc_prediction - Évaluation MC: V(s) ← V(s) + α[G - V(s)]
- mc_control - Contrôle MC avec ε-greedy
- Utilise épisodes complets (pas de bootstrap)

**🎯 Utilisation HelixOne**: Évaluation avec historique complet

### Code Source Complet

```python
'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from typing import Iterable, Iterator, TypeVar, Callable
from rl.distribution import Categorical
from rl.approximate_dynamic_programming import (ValueFunctionApprox,
                                                QValueFunctionApprox,
                                                NTStateDistribution)
from rl.iterate import last
from rl.markov_decision_process import MarkovDecisionProcess, Policy, \
    TransitionStep, NonTerminal
from rl.policy import DeterministicPolicy, RandomPolicy, UniformPolicy
import rl.markov_process as mp
from rl.returns import returns
import itertools

S = TypeVar('S')
A = TypeVar('A')


def mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx_0: ValueFunctionApprox[S],
    γ: float,
    episode_length_tolerance: float = 1e-6
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MRP
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1), default: 1
      episode_length_tolerance -- stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated value
    function after each episode.

    '''
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    f = approx_0
    yield f

    for episode in episodes:
        f = last(f.iterate_updates(
            [(step.state, step.return_)] for step in episode
        ))
        yield f


def batch_mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx: ValueFunctionApprox[S],
    γ: float,
    episode_length_tolerance: float = 1e-6,
    convergence_tolerance: float = 1e-5
) -> ValueFunctionApprox[S]:
    '''traces is a finite iterable'''
    return_steps: Iterable[mp.ReturnStep[S]] = \
        itertools.chain.from_iterable(
            returns(trace, γ, episode_length_tolerance) for trace in traces
        )
    return approx.solve(
        [(step.state, step.return_) for step in return_steps],
        convergence_tolerance
    )


def greedy_policy_from_qvf(
    q: QValueFunctionApprox[S, A],
    actions: Callable[[NonTerminal[S]], Iterable[A]]
) -> DeterministicPolicy[S, A]:
    '''Return the policy that takes the optimal action at each state based
    on the given approximation of the process's Q function.

    '''
    def optimal_action(s: S) -> A:
        _, a = q.argmax((NonTerminal(s), a) for a in actions(NonTerminal(s)))
        return a
    return DeterministicPolicy(optimal_action)


def epsilon_greedy_policy(
    q: QValueFunctionApprox[S, A],
    mdp: MarkovDecisionProcess[S, A],
    ε: float = 0.0
) -> Policy[S, A]:
    def explore(s: S, mdp=mdp) -> Iterable[A]:
        return mdp.actions(NonTerminal(s))
    return RandomPolicy(Categorical(
        {UniformPolicy(explore): ε,
         greedy_policy_from_qvf(q, mdp.actions): 1 - ε}
    ))


def glie_mc_control(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    approx_0: QValueFunctionApprox[S, A],
    γ: float,
    ϵ_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-6
) -> Iterator[QValueFunctionApprox[S, A]]:
    '''Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      mdp -- the Markov Decision Process to evaluate
      states -- distribution of states to start episodes from
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 ≤ γ ≤ 1)
      ϵ_as_func_of_episodes -- a function from the number of episodes
      to epsilon. epsilon is the fraction of the actions where we explore
      rather than following the optimal policy
      episode_length_tolerance -- stop iterating once γᵏ ≤ tolerance

    Returns an iterator with updates to the approximated Q function
    after each episode.

    '''
    q: QValueFunctionApprox[S, A] = approx_0
    p: Policy[S, A] = epsilon_greedy_policy(q, mdp, 1.0)
    yield q

    num_episodes: int = 0
    while True:
        trace: Iterable[TransitionStep[S, A]] = \
            mdp.simulate_actions(states, p)
        num_episodes += 1
        for step in returns(trace, γ, episode_length_tolerance):
            q = q.update([((step.state, step.action), step.return_)])
        p = epsilon_greedy_policy(q, mdp, ϵ_as_func_of_episodes(num_episodes))
        yield q
```

--------------------------------------------------------------------------------

# PARTIE 5 - TEMPORAL DIFFERENCE

================================================================================

## 📄 chapter10/mc_td_experience_replay.py {#mc-td-experience-replay}

**Titre**: Experience Replay

**Description**: Buffer de replay pour apprentissage off-policy

**Lignes de code**: 187

**Concepts clés**:
- ExperienceReplayBuffer - Stocke (s,a,r,s') transitions
- Sampling uniforme ou prioritaire
- Brise les corrélations temporelles

**🎯 Utilisation HelixOne**: Stabilisation de l'apprentissage

### Code Source Complet

```python
from typing import Sequence, TypeVar, Tuple, Mapping, Iterator, Dict
from rl.markov_process import TransitionStep, ReturnStep, \
    NonTerminal, Terminal, FiniteMarkovRewardProcess
from rl.function_approx import Tabular
from rl.distribution import Categorical
from rl.returns import returns
import rl.iterate as iterate
from rl.function_approx import learning_rate_schedule
import itertools
import collections
import numpy as np
import rl.monte_carlo as mc
import rl.td as td

S = TypeVar('S')


def get_fixed_episodes_from_sr_pairs_seq(
    sr_pairs_seq: Sequence[Sequence[Tuple[S, float]]],
    terminal_state: S
) -> Sequence[Sequence[TransitionStep[S]]]:
    return [[TransitionStep(
        state=NonTerminal(s),
        reward=r,
        next_state=NonTerminal(trace[i+1][0])
        if i < len(trace) - 1 else Terminal(terminal_state)
    ) for i, (s, r) in enumerate(trace)] for trace in sr_pairs_seq]


def get_return_steps_from_fixed_episodes(
    fixed_episodes: Sequence[Sequence[TransitionStep[S]]],
    gamma: float
) -> Sequence[ReturnStep[S]]:
    return list(itertools.chain.from_iterable(returns(episode, gamma, 1e-8)
                                              for episode in fixed_episodes))


def get_mean_returns_from_return_steps(
    returns_seq: Sequence[ReturnStep[S]]
) -> Mapping[NonTerminal[S], float]:
    def by_state(ret: ReturnStep[S]) -> S:
        return ret.state.state

    sorted_returns_seq: Sequence[ReturnStep[S]] = sorted(
        returns_seq,
        key=by_state
    )
    return {NonTerminal(s): np.mean([r.return_ for r in l])
            for s, l in itertools.groupby(
                sorted_returns_seq,
                key=by_state
            )}


def get_episodes_stream(
    fixed_episodes: Sequence[Sequence[TransitionStep[S]]]
) -> Iterator[Sequence[TransitionStep[S]]]:
    num_episodes: int = len(fixed_episodes)
    while True:
        yield fixed_episodes[np.random.randint(num_episodes)]


def mc_prediction(
    episodes_stream: Iterator[Sequence[TransitionStep[S]]],
    gamma: float,
    num_episodes: int
) -> Mapping[NonTerminal[S], float]:
    return iterate.last(itertools.islice(
        mc.mc_prediction(
            traces=episodes_stream,
            approx_0=Tabular(),
            γ=gamma,
            episode_length_tolerance=1e-10
        ),
        num_episodes
    )).values_map


def fixed_experiences_from_fixed_episodes(
    fixed_episodes: Sequence[Sequence[TransitionStep[S]]]
) -> Sequence[TransitionStep[S]]:
    return list(itertools.chain.from_iterable(fixed_episodes))


def finite_mrp(
    fixed_experiences: Sequence[TransitionStep[S]]
) -> FiniteMarkovRewardProcess[S]:
    def by_state(tr: TransitionStep[S]) -> S:
        return tr.state.state

    d: Mapping[S, Sequence[Tuple[S, float]]] = \
        {s: [(t.next_state.state, t.reward) for t in l] for s, l in
         itertools.groupby(
             sorted(fixed_experiences, key=by_state),
             key=by_state
         )}
    mrp: Dict[S, Categorical[Tuple[S, float]]] = \
        {s: Categorical({x: y / len(l) for x, y in
                         collections.Counter(l).items()})
         for s, l in d.items()}
    return FiniteMarkovRewardProcess(mrp)


def get_experiences_stream(
    fixed_experiences: Sequence[TransitionStep[S]]
) -> Iterator[TransitionStep[S]]:
    num_experiences: int = len(fixed_experiences)
    while True:
        yield fixed_experiences[np.random.randint(num_experiences)]


def td_prediction(
    experiences_stream: Iterator[TransitionStep[S]],
    gamma: float,
    num_experiences: int
) -> Mapping[NonTerminal[S], float]:
    return iterate.last(itertools.islice(
        td.td_prediction(
            transitions=experiences_stream,
            approx_0=Tabular(count_to_weight_func=learning_rate_schedule(
                initial_learning_rate=0.01,
                half_life=10000,
                exponent=0.5
            )),
            γ=gamma
        ),
        num_experiences
    )).values_map


if __name__ == '__main__':
    from pprint import pprint

    given_data: Sequence[Sequence[Tuple[str, float]]] = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    gamma: float = 0.9
    num_mc_episodes: int = 100000
    num_td_experiences: int = 1000000

    fixed_episodes: Sequence[Sequence[TransitionStep[str]]] = \
        get_fixed_episodes_from_sr_pairs_seq(
            sr_pairs_seq=given_data,
            terminal_state='T'
        )

    returns_seq: Sequence[ReturnStep[str]] = \
        get_return_steps_from_fixed_episodes(
            fixed_episodes=fixed_episodes,
            gamma=gamma
        )

    mean_returns: Mapping[NonTerminal[str], float] = \
        get_mean_returns_from_return_steps(returns_seq)
    pprint(mean_returns)

    episodes: Iterator[Sequence[TransitionStep[str]]] = \
        get_episodes_stream(fixed_episodes)

    mc_pred: Mapping[NonTerminal[str], float] = mc_prediction(
        episodes_stream=episodes,
        gamma=gamma,
        num_episodes=num_mc_episodes
    )
    pprint(mc_pred)

    fixed_experiences: Sequence[TransitionStep[str]] = \
        fixed_experiences_from_fixed_episodes(fixed_episodes)

    fmrp: FiniteMarkovRewardProcess[str] = finite_mrp(fixed_experiences)
    fmrp.display_value_function(gamma)

    experiences: Iterator[TransitionStep[str]] = \
        get_experiences_stream(fixed_experiences)

    td_pred: Mapping[NonTerminal[str], float] = td_prediction(
        experiences_stream=experiences,
        gamma=gamma,
        num_experiences=num_td_experiences
    )
    pprint(td_pred)
```

--------------------------------------------------------------------------------

## 📄 td_lambda.py {#td-lambda}

**Titre**: TD(λ) et Traces d'Éligibilité

**Description**: TD avec traces d'éligibilité pour crédit temporel

**Lignes de code**: 105

**Concepts clés**:
- Eligibility traces e(s) - mémoire des états visités
- TD(λ) - compromis entre TD(0) et MC
- λ=0 → TD(0), λ=1 → MC
- Forward view vs Backward view

**🎯 Utilisation HelixOne**: Meilleur crédit assignment pour trading

### Code Source Complet

```python
'''lambda-return and TD(lambda) methods for working with prediction and control

'''

from typing import Iterable, Iterator, TypeVar, List, Sequence
from rl.function_approx import Gradient
import rl.markov_process as mp
from rl.markov_decision_process import NonTerminal
import numpy as np
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.approximate_dynamic_programming import extended_vf

S = TypeVar('S')


def lambda_return_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: ValueFunctionApprox[S],
        γ: float,
        lambd: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Value Function Prediction using the lambda-return method given a
    sequence of traces.

    Each value this function yields represents the approximated value
    function for the MRP after an additional episode

    Arguments:
      traces -- a sequence of traces
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      lambd -- lambda parameter (0 <= lambd <= 1)
    '''
    func_approx: ValueFunctionApprox[S] = approx_0
    yield func_approx

    for trace in traces:
        gp: List[float] = [1.]
        lp: List[float] = [1.]
        predictors: List[NonTerminal[S]] = []
        partials: List[List[float]] = []
        weights: List[List[float]] = []
        trace_seq: Sequence[mp.TransitionStep[S]] = list(trace)
        for t, tr in enumerate(trace_seq):
            for i, partial in enumerate(partials):
                partial.append(
                    partial[-1] +
                    gp[t - i] * (tr.reward - func_approx(tr.state)) +
                    (gp[t - i] * γ * extended_vf(func_approx, tr.next_state)
                     if t < len(trace_seq) - 1 else 0.)
                )
                weights[i].append(
                    weights[i][-1] * lambd if t < len(trace_seq)
                    else lp[t - i]
                )
            predictors.append(tr.state)
            partials.append([tr.reward +
                             (γ * extended_vf(func_approx, tr.next_state)
                              if t < len(trace_seq) - 1 else 0.)])
            weights.append([1. - (lambd if t < len(trace_seq) else 0.)])
            gp.append(gp[-1] * γ)
            lp.append(lp[-1] * lambd)
        responses: Sequence[float] = [np.dot(p, w) for p, w in
                                      zip(partials, weights)]
        for p, r in zip(predictors, responses):
            func_approx = func_approx.update([(p, r)])
        yield func_approx


def td_lambda_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: ValueFunctionApprox[S],
        γ: float,
        lambd: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate an MRP using TD(lambda) using the given sequence of traces.

    Each value this function yields represents the approximated value function
    for the MRP after an additional transition within each trace

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)
      lambd -- lambda parameter (0 <= lambd <= 1)
    '''
    func_approx: ValueFunctionApprox[S] = approx_0
    yield func_approx

    for trace in traces:
        el_tr: Gradient[ValueFunctionApprox[S]] = Gradient(func_approx).zero()
        for step in trace:
            x: NonTerminal[S] = step.state
            y: float = step.reward + γ * \
                extended_vf(func_approx, step.next_state)
            el_tr = el_tr * (γ * lambd) + func_approx.objective_gradient(
                xy_vals_seq=[(x, y)],
                obj_deriv_out_fun=lambda x1, y1: np.ones(len(x1))
            )
            func_approx = func_approx.update_with_gradient(
                el_tr * (func_approx(x) - y)
            )
            yield func_approx
```

--------------------------------------------------------------------------------

# PARTIE 6 - POLICY GRADIENT

================================================================================

## 📄 policy_gradient.py {#policy-gradient}

**Titre**: Policy Gradient Methods

**Description**: Optimisation directe de la politique

**Lignes de code**: 236

**Concepts clés**:
- reinforce - REINFORCE: ∇J(θ) = E[∇log π(a|s) * G_t]
- actor_critic_td - Actor-Critic avec TD
- Gradient ascent sur J(θ) = E[Σ γ^t r_t]
- Avantage: politiques continues et stochastiques

**🎯 Utilisation HelixOne**: Allocation d'actifs avec actions continues

### Code Source Complet

```python
from typing import Iterator, Iterable, Sequence, TypeVar
from dataclasses import dataclass
from rl.distribution import Gaussian
from rl.function_approx import FunctionApprox, Gradient
from rl.returns import returns
from rl.policy import Policy
from rl.markov_process import NonTerminal
from rl.markov_decision_process import MarkovDecisionProcess, TransitionStep
from rl.approximate_dynamic_programming import NTStateDistribution
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox
import numpy as np

S = TypeVar('S')


@dataclass(frozen=True)
class GaussianPolicyFromApprox(Policy[S, float]):
    function_approx: FunctionApprox[NonTerminal[S]]
    stdev: float

    def act(self, state: NonTerminal[S]) -> Gaussian:
        return Gaussian(
            μ=self.function_approx(state),
            σ=self.stdev
        )


def reinforce_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    episode_length_tolerance: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    while True:
        policy: Policy[S, float] = GaussianPolicyFromApprox(
            function_approx=policy_mean_approx,
            stdev=policy_stdev
        )
        trace: Iterable[TransitionStep[S, float]] = mdp.simulate_actions(
            start_states=start_states_distribution,
            policy=policy
        )
        gamma_prod: float = 1.0
        for step in returns(trace, gamma, episode_length_tolerance):
            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)
            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(step.state, step.action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * step.return_
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            gamma_prod *= gamma
        yield policy_mean_approx


def actor_critic_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    q_value_func_approx0: QValueFunctionApprox[S, float],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    max_episode_length: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    q: QValueFunctionApprox[S, float] = q_value_func_approx0
    while True:
        steps: int = 0
        gamma_prod: float = 1.0
        state: NonTerminal[S] = start_states_distribution.sample()
        action: float = Gaussian(
            μ=policy_mean_approx(state),
            σ=policy_stdev
        ).sample()
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                next_action: float = Gaussian(
                    μ=policy_mean_approx(next_state),
                    σ=policy_stdev
                ).sample()
                q = q.update([(
                    (state, action),
                    reward + gamma * q((next_state, next_action))
                )])
                action = next_action
            else:
                q = q.update([((state, action), reward)])

            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)

            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(state, action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * q((state, action))
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            yield policy_mean_approx
            gamma_prod *= gamma
            steps += 1
            state = next_state


def actor_critic_advantage_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    q_value_func_approx0: QValueFunctionApprox[S, float],
    value_func_approx0: ValueFunctionApprox[S],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    max_episode_length: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    q: QValueFunctionApprox[S, float] = q_value_func_approx0
    v: ValueFunctionApprox[S] = value_func_approx0
    while True:
        steps: int = 0
        gamma_prod: float = 1.0
        state: NonTerminal[S] = start_states_distribution.sample()
        action: float = Gaussian(
            μ=policy_mean_approx(state),
            σ=policy_stdev
        ).sample()
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                next_action: float = Gaussian(
                    μ=policy_mean_approx(next_state),
                    σ=policy_stdev
                ).sample()
                q = q.update([(
                    (state, action),
                    reward + gamma * q((next_state, next_action))
                )])
                v = v.update([(state, reward + gamma * v(next_state))])
                action = next_action
            else:
                q = q.update([((state, action), reward)])
                v = v.update([(state, reward)])

            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)

            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(state, action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * (q((state, action)) - v(state))
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            yield policy_mean_approx
            gamma_prod *= gamma
            steps += 1
            state = next_state


def actor_critic_td_error_gaussian(
    mdp: MarkovDecisionProcess[S, float],
    policy_mean_approx0: FunctionApprox[NonTerminal[S]],
    value_func_approx0: ValueFunctionApprox[S],
    start_states_distribution: NTStateDistribution[S],
    policy_stdev: float,
    gamma: float,
    max_episode_length: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    policy_mean_approx: FunctionApprox[NonTerminal[S]] = policy_mean_approx0
    yield policy_mean_approx
    vf: ValueFunctionApprox[S] = value_func_approx0
    while True:
        steps: int = 0
        gamma_prod: float = 1.0
        state: NonTerminal[S] = start_states_distribution.sample()
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            action: float = Gaussian(
                μ=policy_mean_approx(state),
                σ=policy_stdev
            ).sample()
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                td_target: float = reward + gamma * vf(next_state)
            else:
                td_target = reward
            td_error: float = td_target - vf(state)
            vf = vf.update([(state, td_target)])

            def obj_deriv_out(
                states: Sequence[NonTerminal[S]],
                actions: Sequence[float]
            ) -> np.ndarray:
                return (policy_mean_approx.evaluate(states) -
                        np.array(actions)) / (policy_stdev * policy_stdev)

            grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                policy_mean_approx.objective_gradient(
                    xy_vals_seq=[(state, action)],
                    obj_deriv_out_fun=obj_deriv_out
            )
            scaled_grad: Gradient[FunctionApprox[NonTerminal[S]]] = \
                grad * gamma_prod * td_error
            policy_mean_approx = \
                policy_mean_approx.update_with_gradient(scaled_grad)
            yield policy_mean_approx
            gamma_prod *= gamma
            steps += 1
            state = next_state
```

--------------------------------------------------------------------------------

# PARTIE 7 - APPLICATIONS FINANCE

================================================================================

## 📄 appendix2/efficient_frontier.py {#efficient-frontier}

**Titre**: Frontière Efficiente (Markowitz)

**Description**: Optimisation moyenne-variance

**Lignes de code**: 88

**Concepts clés**:
- min w'Σw s.t. w'μ = r, Σw = 1
- Frontière efficiente
- Sharpe ratio optimal

**🎯 Utilisation HelixOne**: Benchmark allocation

### Code Source Complet

```python
import pandas_datareader as pdr
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def percentage_formatter(x, pos):
    return "%.1f%%" % (x * 100)


def get_historical_prices(tickers, start, end):
    return pd.concat(
        [pdr.get_data_yahoo(ticker, start, end)["Adj Close"]
         for ticker in tickers],
        axis=1,
        keys=tickers
    )


def get_parabola(a, b, c):
    return lambda r: (a - 2 * b * r + c * r * r) / (a * c - b * b)


if __name__ == '__main__':

    days = 1
    tickers = ["IBM", "GOOG", "AAPL", "TGT", "GS", "MS", "AMZN",
               "MSFT", "WMT", "NKE", "UNH", "PG", "DB", "C", "FB", "NVDA"]
    start = datetime(2017, 9, 17)
    end = datetime(2020, 9, 17)
    prices = get_historical_prices(tickers, start, end)
    print(prices)
    percent_change = prices.pct_change(periods=days)
    factor = 252. / days
    mean = percent_change.mean() * factor
    cov = percent_change.cov() * factor
    stdev = np.sqrt(np.diagonal(cov))
    # print(mean)
    # print(cov)
    # print(stdev)
    ones = np.ones(len(tickers))
    inv_cov = np.linalg.inv(cov)
    x = np.dot(mean, inv_cov)
    a = np.dot(x, mean)
    b = np.sum(x)
    c = np.sum(inv_cov)

    r0 = b / c
    sigma2_0 = 1 / c

    r1 = a / b
    sigma2_1 = a / (b * b)

    x_max = max(np.sqrt(sigma2_1), max(stdev))
    y_max = max(r1, max(mean))

    mean_pts = np.arange(-0.5, y_max + 0.05, 0.001)
    parabola = get_parabola(a, b, c)
    stdev_pts = np.sqrt(parabola(mean_pts))

    _, ax = plt.subplots()
    ax.set_xlabel(
        "Standard Deviation of Returns (Annualized)",
        fontsize=20
    )
    ax.set_ylabel("Mean Returns (Annualized)", fontsize=20)
    ax.set_title(
        "Historical Returns Mean versus Standard Deviation",
        fontsize=30
    )
    formatter = FuncFormatter(percentage_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid()
    plt.xlim(xmin=0.15, xmax=x_max + 0.02)
    plt.ylim(ymin=-0.15, ymax=y_max + 0.02)
    plt.scatter(stdev_pts, mean_pts)
    plt.scatter(stdev, mean)
    plt.scatter(np.sqrt(sigma2_0), r0, marker='x', c=0.1, s=100)
    plt.annotate("GMVP", xy=(np.sqrt(sigma2_0), r0), fontsize=15)
    plt.scatter(np.sqrt(sigma2_1), r1, marker='x', c=0.1, s=100)
    plt.annotate("SEP", xy=(np.sqrt(sigma2_1), r1), fontsize=15)
    for t, x, y in zip(tickers, stdev, mean):
        plt.annotate(t, xy=(x, y))
    plt.show()
```

--------------------------------------------------------------------------------

## 📄 chapter12/optimal_exercise_rl.py {#optimal-exercise-rl}

**Titre**: Options Américaines via RL (LSPI)

**Description**: Pricing par Least Squares Policy Iteration

**Lignes de code**: 483

**Concepts clés**:
- LSPI: résout Q = Φw par moindres carrés
- Features: polynômes de Laguerre
- Longstaff-Schwartz method
- MC + régression pour continuation value

**🎯 Utilisation HelixOne**: Pricing scalable d'options exotiques

### Code Source Complet

```python
from typing import Callable, Sequence, Tuple, List
import numpy as np
import random
from scipy.stats import norm
from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox, DNNSpec, AdamGradient, Weights
from random import randrange
from numpy.polynomial.laguerre import lagval
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_process import NonTerminal
from rl.gen_utils.plot_funcs import plot_list_of_curves

TrainingDataType = Tuple[int, float, float]


def european_put_price(
    spot_price: float,
    expiry: float,
    rate: float,
    vol: float,
    strike: float
) -> float:
    sigma_sqrt: float = vol * np.sqrt(expiry)
    d1: float = (np.log(spot_price / strike) +
                 (rate + vol ** 2 / 2.) * expiry) \
        / sigma_sqrt
    d2: float = d1 - sigma_sqrt
    return strike * np.exp(-rate * expiry) * norm.cdf(-d2) \
        - spot_price * norm.cdf(-d1)


def training_sim_data(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float
) -> Sequence[TrainingDataType]:
    ret: List[TrainingDataType] = []
    dt: float = expiry / num_steps
    spot: float = spot_price
    vol2: float = vol * vol

    mean2: float = spot * spot
    var: float = mean2 * spot_price_frac * spot_price_frac
    log_mean: float = np.log(mean2 / np.sqrt(var + mean2))
    log_stdev: float = np.sqrt(np.log(var / mean2 + 1))

    for _ in range(num_paths):
        price: float = np.random.lognormal(log_mean, log_stdev)
        for step in range(num_steps):
            m: float = np.log(price) + (rate - vol2 / 2) * dt
            v: float = vol2 * dt
            next_price: float = np.exp(np.random.normal(m, np.sqrt(v)))
            ret.append((step, price, next_price))
            price = next_price
    return ret


def fitted_lspi_put_option(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float,
    strike: float,
    training_iters: int
) -> LinearFunctionApprox[Tuple[float, float]]:

    num_laguerre: int = 4
    epsilon: float = 1e-3

    ident: np.ndarray = np.eye(num_laguerre)
    features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
    features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    features += [
        lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
        lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
        lambda t_s: (t_s[0] / expiry) ** 2
    ]

    training_data: Sequence[TrainingDataType] = training_sim_data(
        expiry=expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        spot_price=spot_price,
        spot_price_frac=spot_price_frac,
        rate=rate,
        vol=vol
    )

    dt: float = expiry / num_steps
    gamma: float = np.exp(-rate * dt)
    num_features: int = len(features)
    states: Sequence[Tuple[float, float]] = [(i * dt, s) for
                                             i, s, _ in training_data]
    next_states: Sequence[Tuple[float, float]] = \
        [((i + 1) * dt, s1) for i, _, s1 in training_data]
    feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                         for x in states])
    next_feature_vals: np.ndarray = np.array([[f(x) for f in features]
                                              for x in next_states])
    non_terminal: np.ndarray = np.array(
        [i < num_steps - 1 for i, _, _ in training_data]
    )
    exer: np.ndarray = np.array([max(strike - s1, 0)
                                 for _, s1 in next_states])
    wts: np.ndarray = np.zeros(num_features)
    for _ in range(training_iters):
        a_inv: np.ndarray = np.eye(num_features) / epsilon
        b_vec: np.ndarray = np.zeros(num_features)
        cont: np.ndarray = np.dot(next_feature_vals, wts)
        cont_cond: np.ndarray = non_terminal * (cont > exer)
        for i in range(len(training_data)):
            phi1: np.ndarray = feature_vals[i]
            phi2: np.ndarray = phi1 - \
                cont_cond[i] * gamma * next_feature_vals[i]
            temp: np.ndarray = a_inv.T.dot(phi2)
            a_inv -= np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
            b_vec += phi1 * (1 - cont_cond[i]) * exer[i] * gamma
        wts = a_inv.dot(b_vec)

    return LinearFunctionApprox.create(
        feature_functions=features,
        weights=Weights.create(wts)
    )


def fitted_dql_put_option(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float,
    strike: float,
    training_iters: int
) -> DNNApprox[Tuple[float, float]]:

    reg_coeff: float = 1e-2
    neurons: Sequence[int] = [6]

#     features: List[Callable[[Tuple[float, float]], float]] = [
#         lambda t_s: 1.,
#         lambda t_s: t_s[0] / expiry,
#         lambda t_s: t_s[1] / strike,
#         lambda t_s: t_s[0] * t_s[1] / (expiry * strike)
#     ]

    num_laguerre: int = 2
    ident: np.ndarray = np.eye(num_laguerre)
    features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
    features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    features += [
        lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
        lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
        lambda t_s: (t_s[0] / expiry) ** 2
    ]

    ds: DNNSpec = DNNSpec(
        neurons=neurons,
        bias=True,
        hidden_activation=lambda x: np.log(1 + np.exp(-x)),
        hidden_activation_deriv=lambda y: np.exp(-y) - 1,
        output_activation=lambda x: x,
        output_activation_deriv=lambda y: np.ones_like(y)
    )

    fa: DNNApprox[Tuple[float, float]] = DNNApprox.create(
        feature_functions=features,
        dnn_spec=ds,
        adam_gradient=AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        ),
        regularization_coeff=reg_coeff
    )

    dt: float = expiry / num_steps
    gamma: float = np.exp(-rate * dt)
    training_data: Sequence[TrainingDataType] = training_sim_data(
        expiry=expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        spot_price=spot_price,
        spot_price_frac=spot_price_frac,
        rate=rate,
        vol=vol
    )
    for _ in range(training_iters):
        t_ind, s, s1 = training_data[randrange(len(training_data))]
        t = t_ind * dt
        x_val: Tuple[float, float] = (t, s)
        val: float = max(strike - s1, 0)
        if t_ind < num_steps - 1:
            val = max(val, fa.evaluate([(t + dt, s1)])[0])
        y_val: float = gamma * val
        fa = fa.update([(x_val, y_val)])
        # for w in fa.weights:
        #     pprint(w.weights)
    return fa


def scoring_sim_data(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    rate: float,
    vol: float
) -> np.ndarray:
    paths: np.ndarray = np.empty([num_paths, num_steps + 1])
    dt: float = expiry / num_steps
    vol2: float = vol * vol
    for i in range(num_paths):
        paths[i, 0] = spot_price
        for step in range(num_steps):
            m: float = np.log(paths[i, step]) + (rate - vol2 / 2) * dt
            v: float = vol2 * dt
            paths[i, step + 1] = np.exp(np.random.normal(m, np.sqrt(v)))
    return paths


def continuation_curve(
    func: FunctionApprox[Tuple[float, float]],
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return func.evaluate([(t, p) for p in prices])


def exercise_curve(
    strike: float,
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return np.array([max(strike - p, 0) for p in prices])


def put_option_exercise_boundary(
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    num_steps: int,
    strike: float
) -> Tuple[Sequence[float], Sequence[float]]:
    x: List[float] = []
    y: List[float] = []
    prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
    for step in range(num_steps):
        t: float = step * expiry / num_steps
        cp: np.ndarray = continuation_curve(
            func=func,
            t=t,
            prices=prices
        )
        ep: np.ndarray = exercise_curve(
            strike=strike,
            t=t,
            prices=prices
        )
        ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
        if len(ll) > 0:
            x.append(t)
            y.append(max(ll))
    final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
    x.append(expiry)
    y.append(max(p for p, e in final if e > 0))
    return x, y


def option_price(
    scoring_data: np.ndarray,
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    rate: float,
    strike: float
) -> float:
    num_paths: int = scoring_data.shape[0]
    num_steps: int = scoring_data.shape[1] - 1
    prices: np.ndarray = np.zeros(num_paths)
    dt: float = expiry / num_steps

    for i, path in enumerate(scoring_data):
        step: int = 0
        while step <= num_steps:
            t: float = step * dt
            exercise_price: float = max(strike - path[step], 0)
            continue_price: float = func.evaluate([(t, path[step])])[0] \
                if step < num_steps else 0.
            step += 1
            if exercise_price >= continue_price:
                prices[i] = np.exp(-rate * t) * exercise_price
                step = num_steps + 1

    return np.average(prices)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    spot_price_val: float = 100.0
    strike_val: float = 100.0
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_scoring_paths: int = 10000
    num_steps_scoring: int = 100

    num_steps_lspi: int = 20
    num_training_paths_lspi: int = 1000
    spot_price_frac_lspi: float = 0.3
    training_iters_lspi: int = 8

    num_steps_dql: int = 20
    num_training_paths_dql: int = 1000
    spot_price_frac_dql: float = 0.02
    training_iters_dql: int = 100000

    random.seed(100)
    np.random.seed(100)

    flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_put_option(
        expiry=expiry_val,
        num_steps=num_steps_lspi,
        num_paths=num_training_paths_lspi,
        spot_price=spot_price_val,
        spot_price_frac=spot_price_frac_lspi,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val,
        training_iters=training_iters_lspi
    )

    print("Fitted LSPI Model")

    fdql: DNNApprox[Tuple[float, float]] = fitted_dql_put_option(
        expiry=expiry_val,
        num_steps=num_steps_dql,
        num_paths=num_training_paths_dql,
        spot_price=spot_price_val,
        spot_price_frac=spot_price_frac_dql,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val,
        training_iters=training_iters_dql
    )

    print("Fitted DQL Model")

    for step in [0, int(num_steps_lspi / 2), num_steps_lspi - 1]:
        t = step * expiry_val / num_steps_lspi
        prices = np.arange(120.0)
        exer_curve = exercise_curve(
            strike=strike_val,
            t=t,
            prices=prices
        )
        cont_curve_lspi = continuation_curve(
            func=flspi,
            t=t,
            prices=prices
        )
        plt.plot(
            prices,
            exer_curve,
            "b",
            prices,
            cont_curve_lspi,
            "r",
        )
        plt.title(f"LSPI Curves for Time = {t:.3f}")
        plt.show()

    for step in [0, int(num_steps_dql / 2), num_steps_dql - 1]:
        t = step * expiry_val / num_steps_dql
        prices = np.arange(120.0)
        exer_curve = exercise_curve(
            strike=strike_val,
            t=t,
            prices=prices
        )
        cont_curve_dql = continuation_curve(
            func=fdql,
            t=t,
            prices=prices
        )
        plt.plot(
            prices,
            exer_curve,
            "b",
            prices,
            cont_curve_dql,
            "g",
        )
        plt.title(f"DQL Curves for Time = {t:.3f}")
        plt.show()

    european_price: float = european_put_price(
        spot_price=spot_price_val,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        strike=strike_val
    )

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=lambda _, x: max(strike_val - x, 0),
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=100
    )

    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    bin_tree_price: float = vf_seq[0][NonTerminal(0)]
    bin_tree_ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, False)
    bin_tree_x, bin_tree_y = zip(*bin_tree_ex_boundary)

    lspi_x, lspi_y = put_option_exercise_boundary(
        func=flspi,
        expiry=expiry_val,
        num_steps=num_steps_lspi,
        strike=strike_val
    )
    dql_x, dql_y = put_option_exercise_boundary(
        func=fdql,
        expiry=expiry_val,
        num_steps=num_steps_dql,
        strike=strike_val
    )
    plot_list_of_curves(
        list_of_x_vals=[lspi_x, dql_x, bin_tree_x],
        list_of_y_vals=[lspi_y, dql_y, bin_tree_y],
        list_of_colors=["b", "r", "g"],
        list_of_curve_labels=["LSPI", "DQL", "Binary Tree"],
        x_label="Time",
        y_label="Underlying Price",
        title="LSPI, DQL, Binary Tree Exercise Boundaries"
    )

    scoring_data: np.ndarray = scoring_sim_data(
        expiry=expiry_val,
        num_steps=num_steps_scoring,
        num_paths=num_scoring_paths,
        spot_price=spot_price_val,
        rate=rate_val,
        vol=vol_val
    )

    print(f"European Put Price = {european_price:.3f}")
    print(f"Binary Tree Price = {bin_tree_price:.3f}")

    lspi_opt_price: float = option_price(
        scoring_data=scoring_data,
        func=flspi,
        expiry=expiry_val,
        rate=rate_val,
        strike=strike_val,
    )
    print(f"LSPI Option Price = {lspi_opt_price:.3f}")

    dql_opt_price: float = option_price(
        scoring_data=scoring_data,
        func=fdql,
        expiry=expiry_val,
        rate=rate_val,
        strike=strike_val,
    )
    print(f"DQL Option Price = {dql_opt_price:.3f}")
```

--------------------------------------------------------------------------------

## 📄 chapter13/asset_alloc_pg.py {#asset-alloc-pg}

**Titre**: Allocation via Policy Gradient

**Description**: Allocation avec actions continues via PG

**Lignes de code**: 387

**Concepts clés**:
- Politique paramétrique π_θ(a|s)
- REINFORCE pour allocation
- Actions continues (fractions)

**🎯 Utilisation HelixOne**: Allocation optimale continue

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, Iterator, List
from rl.distribution import Distribution, SampledDistribution, Gaussian
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State, Terminal
from rl.function_approx import AdamGradient, FunctionApprox, DNNSpec, \
    DNNApprox
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.policy_gradient import reinforce_gaussian, actor_critic_gaussian, \
    actor_critic_advantage_gaussian, actor_critic_td_error_gaussian
from rl.gen_utils.plot_funcs import plot_list_of_curves
import itertools
import numpy as np

AssetAllocState = Tuple[int, float]


@dataclass(frozen=True)
class AssetAllocPG:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]]
    policy_mean_dnn_spec: DNNSpec
    policy_stdev: float
    initial_wealth_distribution: Distribution[float]

    def time_steps(self) -> int:
        return len(self.risky_return_distributions)

    def get_mdp(self) -> MarkovDecisionProcess[AssetAllocState, float]:
        """
        State is (Wealth W_t, Time t), Action is investment in risky asset x_t
        Investment in riskless asset is W_t - x_t
        """

        steps: int = self.time_steps()
        distrs: Sequence[Distribution[float]] = self.risky_return_distributions
        rates: Sequence[float] = self.riskless_returns
        utility_f: Callable[[float], float] = self.utility_func

        class AssetAllocMDP(MarkovDecisionProcess[AssetAllocState, float]):

            def step(
                self,
                state: NonTerminal[AssetAllocState],
                action: float
            ) -> SampledDistribution[Tuple[State[AssetAllocState], float]]:

                def sr_sampler_func(
                    state=state,
                    action=action
                ) -> Tuple[State[AssetAllocState], float]:
                    time, wealth = state.state
                    next_wealth: float = action * (1 + distrs[time].sample()) \
                        + (wealth - action) * (1 + rates[time])
                    reward: float = utility_f(next_wealth) \
                        if time == steps - 1 else 0.
                    next_pair: AssetAllocState = (time + 1, next_wealth)
                    next_state: State[AssetAllocState] = \
                        Terminal(next_pair) if time == steps - 1 \
                        else NonTerminal(next_pair)
                    return (next_state, reward)

                return SampledDistribution(sampler=sr_sampler_func)

            def actions(self, state: NonTerminal[AssetAllocState]) \
                    -> Sequence[float]:
                return []

        return AssetAllocMDP()

    def start_states_distribution(self) -> \
            SampledDistribution[NonTerminal[AssetAllocState]]:

        def start_states_distribution_func() -> NonTerminal[AssetAllocState]:
            wealth: float = self.initial_wealth_distribution.sample()
            return NonTerminal((0, wealth))

        return SampledDistribution(sampler=start_states_distribution_func)

    def policy_mean_approx(self) -> \
            FunctionApprox[NonTerminal[AssetAllocState]]:
        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.003,
            decay1=0.9,
            decay2=0.999
        )
        ffs: List[Callable[[NonTerminal[AssetAllocState]], float]] = []
        for f in self.policy_feature_funcs:
            def this_f(st: NonTerminal[AssetAllocState], f=f) -> float:
                return f(st.state)
            ffs.append(this_f)
        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=self.policy_mean_dnn_spec,
            adam_gradient=adam_gradient
        )

    def reinforce(self) -> \
            Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        return reinforce_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            episode_length_tolerance=1e-5
        )

    def vf_adam_gradient(self) -> AdamGradient:
        return AdamGradient(
            learning_rate=0.003,
            decay1=0.9,
            decay2=0.999
        )

    def q_value_func_approx(
        self,
        feature_functions: Sequence[Callable[
            [Tuple[AssetAllocState, float]], float]],
        dnn_spec: DNNSpec
    ) -> QValueFunctionApprox[AssetAllocState, float]:
        adam_gradient: AdamGradient = self.vf_adam_gradient()
        ffs: List[Callable[[Tuple[NonTerminal[
            AssetAllocState], float]], float]] = []
        for f in feature_functions:
            def this_f(
                pair: Tuple[NonTerminal[AssetAllocState], float],
                f=f
            ) -> float:
                return f((pair[0].state, pair[1]))
            ffs.append(this_f)

        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=dnn_spec,
            adam_gradient=adam_gradient
        )

    def value_funcion_approx(
        self,
        feature_functions: Sequence[Callable[[AssetAllocState], float]],
        dnn_spec: DNNSpec
    ) -> ValueFunctionApprox[AssetAllocState]:
        adam_gradient: AdamGradient = self.vf_adam_gradient()
        ffs: List[Callable[[NonTerminal[AssetAllocState]], float]] = []
        for vf in feature_functions:
            def this_vf(
                state: NonTerminal[AssetAllocState],
                vf=vf
            ) -> float:
                return vf(state.state)
            ffs.append(this_vf)

        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=dnn_spec,
            adam_gradient=adam_gradient
        )

    def actor_critic(
        self,
        feature_functions: Sequence[Callable[
            [Tuple[AssetAllocState, float]], float]],
        q_value_dnn_spec: DNNSpec
    ) -> Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        q_value_func_approx0: QValueFunctionApprox[AssetAllocState, float] = \
            self.q_value_func_approx(feature_functions, q_value_dnn_spec)

        return actor_critic_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            q_value_func_approx0=q_value_func_approx0,
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            max_episode_length=self.time_steps()
        )

    def actor_critic_advantage(
        self,
        q_feature_functions: Sequence[Callable[
            [Tuple[AssetAllocState, float]], float]],
        q_dnn_spec: DNNSpec,
        v_feature_functions: Sequence[Callable[[AssetAllocState], float]],
        v_dnn_spec: DNNSpec
    ) -> Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        q_value_func_approx0: QValueFunctionApprox[AssetAllocState, float] = \
            self.q_value_func_approx(q_feature_functions, q_dnn_spec)
        value_func_approx0: ValueFunctionApprox[AssetAllocState] = \
            self.value_funcion_approx(v_feature_functions, v_dnn_spec)
        return actor_critic_advantage_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            q_value_func_approx0=q_value_func_approx0,
            value_func_approx0=value_func_approx0,
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            max_episode_length=self.time_steps()
        )

    def actor_critic_td_error(
        self,
        feature_functions: Sequence[Callable[[AssetAllocState], float]],
        q_value_dnn_spec: DNNSpec
    ) -> Iterator[FunctionApprox[NonTerminal[AssetAllocState]]]:
        value_func_approx0: ValueFunctionApprox[AssetAllocState] = \
            self.value_funcion_approx(feature_functions, q_value_dnn_spec)
        return actor_critic_td_error_gaussian(
            mdp=self.get_mdp(),
            policy_mean_approx0=self.policy_mean_approx(),
            value_func_approx0=value_func_approx0,
            start_states_distribution=self.start_states_distribution(),
            policy_stdev=self.policy_stdev,
            gamma=1.0,
            max_episode_length=self.time_steps()
        )


if __name__ == '__main__':

    steps: int = 5
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.1
    policy_stdev: float = 0.5

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        print(f"Time {t:d}: Optimal Risky Allocation = {alloc:.3f}")
        print()

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]] = \
        [
            lambda w_t: (1 + r) ** w_t[1]
        ]
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)
    policy_mean_dnn_spec: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: x,
        output_activation_deriv=lambda y: np.ones_like(y)
    )

    aad: AssetAllocPG = AssetAllocPG(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        policy_feature_funcs=policy_feature_funcs,
        policy_mean_dnn_spec=policy_mean_dnn_spec,
        policy_stdev=policy_stdev,
        initial_wealth_distribution=init_wealth_distr
    )

    reinforce_policies: Iterator[FunctionApprox[
        NonTerminal[AssetAllocState]]] = aad.reinforce()

    q_ffs: Sequence[Callable[[Tuple[AssetAllocState, float]], float]] = \
        [
            lambda _: 1.,
            lambda wt_x: float(wt_x[0][1]),
            lambda wt_x: wt_x[0][0] * (1 + r) ** (- wt_x[0][1]),
            lambda wt_x: wt_x[1] * (1 + r) ** (- wt_x[0][1]),
            lambda wt_x: (wt_x[1] * (1 + r) ** (- wt_x[0][1])) ** 2,
        ]
    dnn_qvf_spec: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    actor_critic_policies: Iterator[FunctionApprox[
        NonTerminal[AssetAllocState]]] = aad.actor_critic(
            feature_functions=q_ffs,
            q_value_dnn_spec=dnn_qvf_spec
        )

    v_ffs: Sequence[Callable[[AssetAllocState], float]] = \
        [
            lambda _: 1.,
            lambda w_t: float(w_t[1]),
            lambda w_t: w_t[0] * (1 + r) ** (- w_t[1])
        ]
    dnn_vf_spec: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    actor_critic_adv_policies: Iterator[FunctionApprox[
        NonTerminal[AssetAllocState]]] = aad.actor_critic_advantage(
            q_feature_functions=q_ffs,
            q_dnn_spec=dnn_qvf_spec,
            v_feature_functions=v_ffs,
            v_dnn_spec=dnn_vf_spec
        )
    actor_critic_error_policies: Iterator[FunctionApprox[
        NonTerminal[AssetAllocState]]] = aad.actor_critic_td_error(
            feature_functions=v_ffs,
            q_value_dnn_spec=dnn_vf_spec
        )

    num_episodes: int = 50000

    x: Sequence[int] = range(num_episodes)
    y0: Sequence[float] = [base_alloc * (1 + r) ** (1 - steps)] * num_episodes
    y1: Sequence[float] = [p(NonTerminal((init_wealth, 0))) for p in
                           itertools.islice(reinforce_policies, num_episodes)]
    y2: Sequence[float] = [p(NonTerminal((init_wealth, 0))) for p in
                           itertools.islice(
                               actor_critic_policies,
                               0,
                               num_episodes * steps,
                               steps
                           )]
    y3: Sequence[float] = [p(NonTerminal((init_wealth, 0))) for p in
                           itertools.islice(
                               actor_critic_adv_policies,
                               0,
                               num_episodes * steps,
                               steps
                           )]
    y4: Sequence[float] = [p(NonTerminal((init_wealth, 0))) for p in
                           itertools.islice(
                               actor_critic_error_policies,
                               0,
                               num_episodes * steps,
                               steps
                            )]
    plot_period: int = 200
    start: int = 50
    x_vals = [[i * plot_period for i in
               range(start, int(num_episodes / plot_period))]] * 4
    y_vals = []
    for y in [y0, y1, y2, y4]:
        y_vals.append([np.mean(y[i * plot_period:(i + 1) * plot_period])
                       for i in range(start, int(num_episodes / plot_period))])
    print(x_vals)
    print(y_vals)

    plot_list_of_curves(
        x_vals,
        y_vals,
        ["k--", "r-x", "g-.", "b-"],
        ["True", "REINFORCE", "Actor-Critic", "Actor-Critic with TD Error"],
        "Iteration",
        "Action",
        "Action for Initial Wealth at Time 0"
    )

    print("Policy Gradient Solution")
    print("------------------------")
    print()

    opt_policies: Sequence[FunctionApprox[NonTerminal[AssetAllocState]]] = \
        list(itertools.islice(actor_critic_error_policies, 10000 * steps))
    for t in range(steps):
        opt_alloc: float = np.mean([p(NonTerminal((init_wealth, t)))
                                   for p in opt_policies])
        print(f"Time {t:d}: Optimal Risky Allocation = {opt_alloc:.3f}")
        print()
```

--------------------------------------------------------------------------------

## 📄 chapter13/asset_alloc_reinforce.py {#asset-alloc-reinforce}

**Titre**: Asset Alloc Reinforce

**Description**: Module Asset Alloc Reinforce

**Lignes de code**: 82

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Callable, Iterator
from rl.function_approx import FunctionApprox, DNNSpec
from rl.markov_process import NonTerminal
from rl.distribution import Gaussian
from rl.chapter13.asset_alloc_pg import AssetAllocPG, AssetAllocState
import numpy as np
import itertools


steps: int = 5
μ: float = 0.13
σ: float = 0.2
r: float = 0.07
a: float = 1.0
init_wealth: float = 1.0
init_wealth_stdev: float = 0.1
policy_stdev: float = 0.5

excess: float = μ - r
var: float = σ * σ
base_alloc: float = excess / (a * var)

print("Analytical Solution")
print("-------------------")
print()

for t in range(steps):
    left: int = steps - t
    growth: float = (1 + r) ** (left - 1)
    alloc: float = base_alloc / growth
    print(f"Time {t:d}: Optimal Risky Allocation = {alloc:.3f}")
    print()

risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
riskless_ret: Sequence[float] = [r for _ in range(steps)]
utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
policy_feature_funcs: Sequence[Callable[[AssetAllocState], float]] = \
    [
        lambda w_t: (1 + r) ** w_t[1]
    ]
init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)
policy_mean_dnn_spec: DNNSpec = DNNSpec(
    neurons=[],
    bias=False,
    hidden_activation=lambda x: x,
    hidden_activation_deriv=lambda y: np.ones_like(y),
    output_activation=lambda x: x,
    output_activation_deriv=lambda y: np.ones_like(y)
)

aad: AssetAllocPG = AssetAllocPG(
    risky_return_distributions=risky_ret,
    riskless_returns=riskless_ret,
    utility_func=utility_function,
    policy_feature_funcs=policy_feature_funcs,
    policy_mean_dnn_spec=policy_mean_dnn_spec,
    policy_stdev=policy_stdev,
    initial_wealth_distribution=init_wealth_distr
)

reinforce_policies: Iterator[FunctionApprox[
    NonTerminal[AssetAllocState]]] = aad.reinforce()

print("REINFORCE Solution")
print("------------------")
print()

num_episodes: int = 10000
averaging_episodes: int = 10000

policies: Sequence[FunctionApprox[NonTerminal[AssetAllocState]]] = \
    list(itertools.islice(
        reinforce_policies,
        num_episodes,
        num_episodes + averaging_episodes
    ))
for t in range(steps):
    opt_alloc: float = np.mean([p(NonTerminal((init_wealth, t)))
                               for p in policies])
    print(f"Time {t:d}: Optimal Risky Allocation = {opt_alloc:.3f}")
    print()
```

--------------------------------------------------------------------------------

## 📄 chapter2/stock_price_mp.py {#stock-price-mp}

**Titre**: Stock Price Mp

**Description**: Module Stock Price Mp

**Lignes de code**: 194

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
import itertools
from rl.distribution import Categorical, Constant
from rl.markov_process import MarkovProcess, NonTerminal, State
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.chapter2.stock_price_simulations import\
    plot_single_trace_all_processes
from rl.chapter2.stock_price_simulations import\
    plot_distribution_at_time_all_processes


@dataclass(frozen=True)
class StateMP1:
    price: int


@dataclass
class StockPriceMP1(MarkovProcess[StateMP1]):

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition(
        self,
        state: NonTerminal[StateMP1]
    ) -> Categorical[State[StateMP1]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(StateMP1(state.state.price + 1)): up_p,
            NonTerminal(StateMP1(state.state.price - 1)): 1 - up_p
        })


@dataclass(frozen=True)
class StateMP2:
    price: int
    is_prev_move_up: Optional[bool]


handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass
class StockPriceMP2(MarkovProcess[StateMP2]):

    alpha2: float = 0.75  # strength of reverse-pull (value in [0,1])

    def up_prob(self, state: StateMP2) -> float:
        return 0.5 * (1 + self.alpha2 * handy_map[state.is_prev_move_up])

    def transition(
        self,
        state: NonTerminal[StateMP2]
    ) -> Categorical[State[StateMP2]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(StateMP2(state.state.price + 1, True)): up_p,
            NonTerminal(StateMP2(state.state.price - 1, False)): 1 - up_p
        })


@dataclass(frozen=True)
class StateMP3:
    num_up_moves: int
    num_down_moves: int


@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):

    alpha3: float = 1.0  # strength of reverse-pull (non-negative value)

    def up_prob(self, state: StateMP3) -> float:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3)(
            state.num_down_moves / total
        ) if total else 0.5

    def transition(
        self,
        state: NonTerminal[StateMP3]
    ) -> Categorical[State[StateMP3]]:
        up_p = self.up_prob(state.state)
        return Categorical({
            NonTerminal(StateMP3(
                state.state.num_up_moves + 1, state.state.num_down_moves
            )): up_p,
            NonTerminal(StateMP3(
                state.state.num_up_moves, state.state.num_down_moves + 1
            )): 1 - up_p
        })


def process1_price_traces(
    start_price: int,
    level_param: int,
    alpha1: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP1(level_param=level_param, alpha1=alpha1)
    start_state_distribution = Constant(
        NonTerminal(StateMP1(price=start_price))
    )
    return np.vstack([
        np.fromiter((s.state.price for s in itertools.islice(
            mp.simulate(start_state_distribution),
            time_steps + 1
        )), float) for _ in range(num_traces)])


def process2_price_traces(
    start_price: int,
    alpha2: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP2(alpha2=alpha2)
    start_state_distribution = Constant(
        NonTerminal(StateMP2(price=start_price, is_prev_move_up=None))
    )
    return np.vstack([
        np.fromiter((s.state.price for s in itertools.islice(
            mp.simulate(start_state_distribution),
            time_steps + 1
        )), float) for _ in range(num_traces)])


def process3_price_traces(
    start_price: int,
    alpha3: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP3(alpha3=alpha3)
    start_state_distribution = Constant(
        NonTerminal(StateMP3(num_up_moves=0, num_down_moves=0))
    )
    return np.vstack([np.fromiter(
        (start_price + s.state.num_up_moves - s.state.num_down_moves for s in
         itertools.islice(
             mp.simulate(start_state_distribution),
             time_steps + 1
         )),
        float
    ) for _ in range(num_traces)])


if __name__ == '__main__':
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100
    num_traces: int = 1000

    process1_traces: np.ndarray = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_steps,
        num_traces=num_traces
    )
    process2_traces: np.ndarray = process2_price_traces(
        start_price=start_price,
        alpha2=alpha2,
        time_steps=time_steps,
        num_traces=num_traces
    )
    process3_traces: np.ndarray = process3_price_traces(
        start_price=start_price,
        alpha3=alpha3,
        time_steps=time_steps,
        num_traces=num_traces
    )

    trace1 = process1_traces[0]
    trace2 = process2_traces[0]
    trace3 = process3_traces[0]

    plot_single_trace_all_processes(trace1, trace2, trace3)

    plot_distribution_at_time_all_processes(
        process1_traces,
        process2_traces,
        process3_traces
    )
```

--------------------------------------------------------------------------------

## 📄 chapter2/stock_price_simulations.py {#stock-price-simulations}

**Titre**: Stock Price Simulations

**Description**: Module Stock Price Simulations

**Lignes de code**: 232

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Optional, Mapping, Sequence, Tuple
from collections import Counter
import numpy as np
from numpy.random import binomial
import itertools
from operator import itemgetter
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func


@dataclass
class Process1:
    @dataclass
    class State:
        price: int

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: State) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process1.State(price=state.price + up_move * 2 - 1)


handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass
class Process2:
    @dataclass
    class State:
        price: int
        is_prev_move_up: Optional[bool]

    alpha2: float = 0.75  # strength of reverse-pull (value in [0,1])

    def up_prob(self, state: State) -> float:
        return 0.5 * (1 + self.alpha2 * handy_map[state.is_prev_move_up])

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process2.State(
            price=state.price + up_move * 2 - 1,
            is_prev_move_up=bool(up_move)
        )


@dataclass
class Process3:
    @dataclass
    class State:
        num_up_moves: int
        num_down_moves: int

    alpha3: float = 1.0  # strength of reverse-pull (non-negative value)

    def up_prob(self, state: State) -> float:
        total = state.num_up_moves + state.num_down_moves
        return get_unit_sigmoid_func(self.alpha3)(
            state.num_down_moves / total
        ) if total else 0.5

    def next_state(self, state: State) -> State:
        up_move: int = binomial(1, self.up_prob(state), 1)[0]
        return Process3.State(
            num_up_moves=state.num_up_moves + up_move,
            num_down_moves=state.num_down_moves + 1 - up_move
        )


def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)


def process1_price_traces(
    start_price: int,
    level_param: int,
    alpha1: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    process = Process1(level_param=level_param, alpha1=alpha1)
    start_state = Process1.State(price=start_price)
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1
        )), float) for _ in range(num_traces)])


def process2_price_traces(
    start_price: int,
    alpha2: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    process = Process2(alpha2=alpha2)
    start_state = Process2.State(price=start_price, is_prev_move_up=None)
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1
        )), float) for _ in range(num_traces)])


def process3_price_traces(
    start_price: int,
    alpha3: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    process = Process3(alpha3=alpha3)
    start_state = Process3.State(num_up_moves=0, num_down_moves=0)
    return np.vstack([
        np.fromiter((start_price + s.num_up_moves - s.num_down_moves
                    for s in itertools.islice(simulation(process, start_state),
                                              time_steps + 1)), float)
        for _ in range(num_traces)])


def plot_single_trace_all_processes(
    process1_trace: np.ndarray,
    process2_trace: np.ndarray,
    process3_trace: np.ndarray
) -> None:

    from rl.gen_utils.plot_funcs import plot_list_of_curves

    traces_len: int = len(process1_trace)

    plot_list_of_curves(
        [range(traces_len)] * 3,
        [process1_trace, process2_trace, process3_trace],
        ["r-", "b--", "g-."],
        [
            r"Process 1 ($\alpha_1=0.25$)",
            r"Process 2 ($\alpha_2=0.75$)",
            r"Process 3 ($\alpha_3=1.0$)"
        ],
        "Time Steps",
        "Stock Price",
        "Single-Trace Simulation for Each Process"
    )


def get_terminal_histogram(
    price_traces: np.ndarray
) -> Tuple[Sequence[int], Sequence[int]]:
    pairs: Sequence[Tuple[int, int]] = sorted(
        list(Counter(price_traces[:, -1]).items()),
        key=itemgetter(0)
    )
    return [x for x, _ in pairs], [y for _, y in pairs]


def plot_distribution_at_time_all_processes(
    process1_traces: np.ndarray,
    process2_traces: np.ndarray,
    process3_traces: np.ndarray
) -> None:

    from rl.gen_utils.plot_funcs import plot_list_of_curves

    num_traces: int = len(process1_traces)
    time_steps: int = len(process1_traces[0]) - 1

    x1, y1 = get_terminal_histogram(process1_traces)
    x2, y2 = get_terminal_histogram(process2_traces)
    x3, y3 = get_terminal_histogram(process3_traces)

    plot_list_of_curves(
        [x1, x2, x3],
        [y1, y2, y3],
        ["r-", "b--", "g-."],
        [
            r"Process 1 ($\alpha_1=0.25$)",
            r"Process 2 ($\alpha_2=0.75$)",
            r"Process 3 ($\alpha_3=1.0$)"
        ],
        "Terminal Stock Price",
        "Counts",
        f"Terminal Price Counts (T={time_steps:d}, Traces={num_traces:d})"
    )


if __name__ == '__main__':
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100
    num_traces: int = 1000

    process1_traces: np.ndarray = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_steps,
        num_traces=num_traces
    )
    process2_traces: np.ndarray = process2_price_traces(
        start_price=start_price,
        alpha2=alpha2,
        time_steps=time_steps,
        num_traces=num_traces
    )
    process3_traces: np.ndarray = process3_price_traces(
        start_price=start_price,
        alpha3=alpha3,
        time_steps=time_steps,
        num_traces=num_traces
    )

    trace1 = process1_traces[0]
    trace2 = process2_traces[0]
    trace3 = process3_traces[0]

    plot_single_trace_all_processes(trace1, trace2, trace3)

    plot_distribution_at_time_all_processes(
        process1_traces,
        process2_traces,
        process3_traces
    )
```

--------------------------------------------------------------------------------

## 📄 chapter7/asset_alloc_discrete.py {#asset-alloc-discrete}

**Titre**: Allocation d'Actifs Discrète

**Description**: Allocation avec actions discrètes

**Lignes de code**: 294

**Concepts clés**:
- MDP pour allocation multi-période
- États: (wealth, time)
- Actions: fraction investie
- Maximisation utilité terminale

**🎯 Utilisation HelixOne**: Allocation avec choix discrets

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, Iterator, List
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State, Terminal
from rl.policy import DeterministicPolicy
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox
from rl.approximate_dynamic_programming import back_opt_vf_and_policy, \
    back_opt_qvf, ValueFunctionApprox, QValueFunctionApprox
from operator import itemgetter
import numpy as np


@dataclass(frozen=True)
class AssetAllocDiscrete:
    risky_return_distributions: Sequence[Distribution[float]]
    riskless_returns: Sequence[float]
    utility_func: Callable[[float], float]
    risky_alloc_choices: Sequence[float]
    feature_functions: Sequence[Callable[[Tuple[float, float]], float]]
    dnn_spec: DNNSpec
    initial_wealth_distribution: Distribution[float]

    def time_steps(self) -> int:
        return len(self.risky_return_distributions)

    def uniform_actions(self) -> Choose[float]:
        return Choose(self.risky_alloc_choices)

    def get_mdp(self, t: int) -> MarkovDecisionProcess[float, float]:
        """
        State is Wealth W_t, Action is investment in risky asset (= x_t)
        Investment in riskless asset is W_t - x_t
        """

        distr: Distribution[float] = self.risky_return_distributions[t]
        rate: float = self.riskless_returns[t]
        alloc_choices: Sequence[float] = self.risky_alloc_choices
        steps: int = self.time_steps()
        utility_f: Callable[[float], float] = self.utility_func

        class AssetAllocMDP(MarkovDecisionProcess[float, float]):

            def step(
                self,
                wealth: NonTerminal[float],
                alloc: float
            ) -> SampledDistribution[Tuple[State[float], float]]:

                def sr_sampler_func(
                    wealth=wealth,
                    alloc=alloc
                ) -> Tuple[State[float], float]:
                    next_wealth: float = alloc * (1 + distr.sample()) \
                        + (wealth.state - alloc) * (1 + rate)
                    reward: float = utility_f(next_wealth) \
                        if t == steps - 1 else 0.
                    next_state: State[float] = Terminal(next_wealth) \
                        if t == steps - 1 else NonTerminal(next_wealth)
                    return (next_state, reward)

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=1000
                )

            def actions(self, wealth: NonTerminal[float]) -> Sequence[float]:
                return alloc_choices

        return AssetAllocMDP()

    def get_qvf_func_approx(self) -> \
            DNNApprox[Tuple[NonTerminal[float], float]]:

        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        ffs: List[Callable[[Tuple[NonTerminal[float], float]], float]] = []
        for f in self.feature_functions:
            def this_f(pair: Tuple[NonTerminal[float], float], f=f) -> float:
                return f((pair[0].state, pair[1]))
            ffs.append(this_f)

        return DNNApprox.create(
            feature_functions=ffs,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )

    def get_states_distribution(self, t: int) -> \
            SampledDistribution[NonTerminal[float]]:

        actions_distr: Choose[float] = self.uniform_actions()

        def states_sampler_func() -> NonTerminal[float]:
            wealth: float = self.initial_wealth_distribution.sample()
            for i in range(t):
                distr: Distribution[float] = self.risky_return_distributions[i]
                rate: float = self.riskless_returns[i]
                alloc: float = actions_distr.sample()
                wealth = alloc * (1 + distr.sample()) + \
                    (wealth - alloc) * (1 + rate)
            return NonTerminal(wealth)

        return SampledDistribution(states_sampler_func)

    def backward_induction_qvf(self) -> \
            Iterator[QValueFunctionApprox[float, float]]:

        init_fa: DNNApprox[Tuple[NonTerminal[float], float]] = \
            self.get_qvf_func_approx()

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            DNNApprox[Tuple[NonTerminal[float], float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(i),
            init_fa,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 300
        error_tolerance: float = 1e-6

        return back_opt_qvf(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )

    def get_vf_func_approx(
        self,
        ff: Sequence[Callable[[NonTerminal[float]], float]]
    ) -> DNNApprox[NonTerminal[float]]:

        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        return DNNApprox.create(
            feature_functions=ff,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )

    def backward_induction_vf_and_pi(
        self,
        ff: Sequence[Callable[[NonTerminal[float]], float]]
    ) -> Iterator[Tuple[ValueFunctionApprox[float],
                        DeterministicPolicy[float, float]]]:

        init_fa: DNNApprox[NonTerminal[float]] = self.get_vf_func_approx(ff)

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            DNNApprox[NonTerminal[float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(i),
            init_fa,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 300
        error_tolerance: float = 1e-8

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':

    from pprint import pprint

    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    # vf_ff: Sequence[Callable[[NonTerminal[float]], float]] = [lambda _: 1., lambda w: w.state]
    # it_vf: Iterator[Tuple[DNNApprox[NonTerminal[float]], DeterministicPolicy[float, float]]] = \
    #     aad.backward_induction_vf_and_pi(vf_ff)

    # print("Backward Induction: VF And Policy")
    # print("---------------------------------")
    # print()
    # for t, (v, p) in enumerate(it_vf):
    #     print(f"Time {t:d}")
    #     print()
    #     opt_alloc: float = p.action_for(init_wealth)
    #     val: float = v(NonTerminal(init_wealth))
    #     print(f"Opt Risky Allocation = {opt_alloc:.2f}, Opt Val = {val:.3f}")
    #     print("Weights")
    #     for w in v.weights:
    #         print(w.weights)
    #     print()

    it_qvf: Iterator[QValueFunctionApprox[float, float]] = \
        aad.backward_induction_qvf()

    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        opt_alloc: float = max(
            ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(q((NonTerminal(init_wealth), ac))
                         for ac in alloc_choices)
        print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            pprint(wts.weights)
        print()

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        vval: float = - np.exp(- excess * excess * left / (2 * var)
                               - a * growth * (1 + r) * init_wealth) / a
        bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
            np.log(np.abs(a))
        w_t_wt: float = a * growth * (1 + r)
        x_t_wt: float = a * excess * growth
        x_t2_wt: float = - var * (a * growth) ** 2 / 2

        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {vval:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()
```

--------------------------------------------------------------------------------

## 📄 chapter7/merton_solution_graph.py {#merton-solution-graph}

**Titre**: Merton Solution Graph

**Description**: Module Merton Solution Graph

**Lignes de code**: 120

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from math import exp
from typing import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MertonPortfolio:
    mu: float
    sigma: float
    r: float
    rho: float
    horizon: float
    gamma: float
    epsilon: float = 1e-6

    def excess(self) -> float:
        return self.mu - self.r

    def variance(self) -> float:
        return self.sigma * self.sigma

    def allocation(self) -> float:
        return self.excess() / (self.gamma * self.variance())

    def portfolio_return(self) -> float:
        return self.r + self.allocation() * self.excess()

    def nu(self) -> float:
        return (self.rho - (1 - self.gamma) * self.portfolio_return()) / \
            self.gamma

    def f(self, time: float) -> float:
        remaining: float = self.horizon - time
        nu = self.nu()
        if nu == 0:
            ret = remaining + self.epsilon
        else:
            ret = (1 + (nu * self.epsilon - 1) * exp(-nu * remaining)) / nu
        return ret

    def fractional_consumption_rate(self, time: float) -> float:
        return 1 / self.f(time)

    def wealth_growth_rate(self, time: float) -> float:
        return self.portfolio_return() - self.fractional_consumption_rate(time)

    def expected_wealth(self, time: float) -> float:
        base: float = exp(self.portfolio_return() * time)
        nu = self.nu()
        if nu == 0:
            ret = base * (1 - time / (self.horizon + self.epsilon))
        else:
            ret = base * (1 - (1 - exp(-nu * time)) /
                          (1 + (nu * self.epsilon - 1) *
                           exp(-nu * self.horizon)))
        return ret


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves

    mu: float = 0.1
    sigma: float = 0.1
    r: float = 0.02
    rho: float = 0.01
    horizon: float = 20.0
    gamma: float = 2.0

    mp = MertonPortfolio(
        mu,
        sigma,
        r,
        rho,
        horizon,
        gamma
    )

    intervals: float = 20
    time_steps = [i * horizon / intervals for i in range(intervals)]

    optimal_consumption_rate: Sequence[float] = [
        mp.fractional_consumption_rate(i) for i in time_steps
    ]
    expected_portfolio_return: float = mp.portfolio_return()
    expected_wealth_growth: Sequence[float] = [mp.wealth_growth_rate(i)
                                               for i in time_steps]

    plot_list_of_curves(
        [time_steps] * 3,
        [
            optimal_consumption_rate,
            expected_wealth_growth,
            [expected_portfolio_return] * intervals
        ],
        ["b-", "g--", "r-."],
        [
         "Fractional Consumption Rate",
         "Expected Wealth Growth Rate",
         "Expected Portfolio Annual Return = %.1f%%" %
         (expected_portfolio_return * 100)
        ],
        x_label="Time in years",
        y_label="Annual Rate",
        title="Fractional Consumption and Expected Wealth Growth"
    )

    extended_time_steps = time_steps + [horizon]
    expected_wealth: Sequence[float] = [mp.expected_wealth(i)
                                        for i in extended_time_steps]

    plot_list_of_curves(
        [extended_time_steps],
        [expected_wealth],
        ["b"],
        ["Expected Wealth"],
        x_label="Time in Years",
        y_label="Wealth",
        title="Time-Trajectory of Expected Wealth"
    )
```

--------------------------------------------------------------------------------

## 📄 chapter8/max_exp_utility.py {#max-exp-utility}

**Titre**: Max Exp Utility

**Description**: Module Max Exp Utility

**Lignes de code**: 211

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Callable, Mapping
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import quad
import numpy as np


@dataclass(frozen=True)
class MaxExpUtility:
    """
    The goal is to compute the price and hedges for a derivative
    for a single risky asset in a single time-period setting. We
    assume that the risky asset takes on a continuum of values
    at t=1 (hedge of risky and riskless assets established
    at t = 0). This corresponds to an incomplete market scenario
    and so, there is no unique price. We determine pricing
    and hedging using the Maximum Expected Utility method and
    assume that the Utility function is CARA U(x) = (1-e^{-ax})/a,
    where a is the risk-aversion parameter. We assume the risky asset
    follows a normal distribution at t=1.
    """
    risky_spot: float  # risky asset price at t=0
    riskless_rate: float  # riskless asset price grows from 1 to 1+r
    risky_mean: float  # mean of risky asset price at t=1
    risky_stdev: float  # std dev of risky asset price at t=1
    payoff_func: Callable[[float], float]  # derivative payoff at t=1

    def complete_mkt_price_and_hedges(self) -> Mapping[str, float]:
        """
        This computes the price and hedges assuming a complete
        market, which means the risky asset takes on two values
        at t=1. 1) mean + stdev 2) mean - stdev, with equal
        probabilities. This situation can be perfectly hedged
        with a risky and a riskless asset. The following
        code provides the solution for the 2 equations and 2
        variables system
        alpha is the hedge in the risky asset units and beta
        is the hedge in the riskless asset units
        """
        x = self.risky_mean + self.risky_stdev
        z = self.risky_mean - self.risky_stdev
        v1 = self.payoff_func(x)
        v2 = self.payoff_func(z)
        alpha = (v1 - v2) / (z - x)
        beta = - 1 / (1 + self.riskless_rate) * (v1 + alpha * x)
        price = - (beta + alpha * self.risky_spot)
        return {"price": price, "alpha": alpha, "beta": beta}

    def max_exp_util_for_zero(
        self,
        c: float,
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        """
        This implements the closed-form solution when the derivative
        payoff is uniformly 0
        The input c refers to the cash one pays at t=0
        This means the net position of risky asset together with riskless
        asset is -c, i.e., alpha * risky_spot + beta = -c
        """
        ra = risk_aversion_param
        er = 1 + self.riskless_rate
        mu = self.risky_mean
        sigma = self.risky_stdev
        s0 = self.risky_spot
        alpha = (mu - s0 * er) / (ra * sigma * sigma)
        beta = - (c + alpha * self.risky_spot)
        max_val = (1 - np.exp(-ra * (-er * c + alpha * (mu - s0 * er))
                              + (ra * alpha * sigma) ** 2 / 2)) / ra
        return {"alpha": alpha, "beta": beta, "max_val": max_val}

    def max_exp_util(
        self,
        c: float,
        pf: Callable[[float], float],
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        sigma2 = self.risky_stdev * self.risky_stdev
        mu = self.risky_mean
        s0 = self.risky_spot
        er = 1 + self.riskless_rate
        factor = 1 / np.sqrt(2 * np.pi * sigma2)

        integral_lb = self.risky_mean - self.risky_stdev * 6
        integral_ub = self.risky_mean + self.risky_stdev * 6

        def eval_expectation(alpha: float, c=c) -> float:

            def integrand(rand: float, alpha=alpha, c=c) -> float:
                payoff = pf(rand) - er * c\
                         + alpha * (rand - er * s0)
                exponent = -(0.5 * (rand - mu) * (rand - mu) / sigma2
                             + risk_aversion_param * payoff)
                return (1 - factor * np.exp(exponent)) / risk_aversion_param

            return -quad(integrand, integral_lb, integral_ub)[0]

        res = minimize_scalar(eval_expectation)
        alpha_star = res["x"]
        max_val = - res["fun"]
        beta_star = - (c + alpha_star * s0)
        return {"alpha": alpha_star, "beta": beta_star, "max_val": max_val}

    def max_exp_util_price_and_hedge(
        self,
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        meu_for_zero = self.max_exp_util_for_zero(
            0.,
            risk_aversion_param
        )["max_val"]

        def prep_func(pr: float) -> float:
            return self.max_exp_util(
                pr,
                self.payoff_func,
                risk_aversion_param
            )["max_val"] - meu_for_zero

        lb = self.risky_mean - self.risky_stdev * 10
        ub = self.risky_mean + self.risky_stdev * 10
        payoff_vals = [self.payoff_func(x) for x in np.linspace(lb, ub, 1001)]
        lb_payoff = min(payoff_vals)
        ub_payoff = max(payoff_vals)

        opt_price = root_scalar(
            prep_func,
            bracket=[lb_payoff, ub_payoff],
            method="brentq"
        ).root

        hedges = self.max_exp_util(
            opt_price,
            self.payoff_func,
            risk_aversion_param
        )
        alpha = hedges["alpha"]
        beta = hedges["beta"]
        return {"price": opt_price, "alpha": alpha, "beta": beta}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    risky_spot_val: float = 100.0
    riskless_rate_val: float = 0.05
    risky_mean_val: float = 110.0
    risky_stdev_val: float = 25.0
    payoff_function: Callable[[float], float] = lambda x: - min(x - 105.0, 0)

    b1 = riskless_rate_val >= 0.
    b2 = risky_stdev_val > 0.
    x = risky_spot_val * (1 + riskless_rate_val)
    b3 = risky_mean_val > x > risky_mean_val - risky_stdev_val
    assert all([b1, b2, b3]), "Bad Inputs"

    meu: MaxExpUtility = MaxExpUtility(
        risky_spot=risky_spot_val,
        riskless_rate=riskless_rate_val,
        risky_mean=risky_mean_val,
        risky_stdev=risky_stdev_val,
        payoff_func=payoff_function
    )

    plt.xlabel("Risky Asset Price", size=20)
    plt.ylabel("Derivative Payoff and Hedges", size=20)
    plt.title("Hedging in Incomplete Market", size=30)
    lb = meu.risky_mean - meu.risky_stdev * 1.5
    ub = meu.risky_mean + meu.risky_stdev * 1.5
    x_plot_pts = np.linspace(lb, ub, 101)
    payoff_plot_pts = np.array([meu.payoff_func(x) for x in x_plot_pts])
    plt.plot(
        x_plot_pts,
        payoff_plot_pts,
        "r",
        linewidth=5,
        label="Derivative Payoff"
    )
    cm_ph = meu.complete_mkt_price_and_hedges()
    cm_plot_pts = - (cm_ph["beta"] + cm_ph["alpha"] * x_plot_pts)
    plt.plot(
        x_plot_pts,
        cm_plot_pts,
        "b",
        label="Complete Market Hedge"
    )
    print("Complete Market Price = %.3f" % cm_ph["price"])
    print("Complete Market Alpha = %.3f" % cm_ph["alpha"])
    print("Complete Market Beta = %.3f" % cm_ph["beta"])
    for risk_aversion_param, color in [(0.3, "g--"), (0.6, "y.-"), (0.9, "m+-")]:
        print("--- Risk Aversion Param = %.2f ---" % risk_aversion_param)
        meu_for_zero = meu.max_exp_util_for_zero(0., risk_aversion_param)
        print("MEU for Zero Alpha = %.3f" % meu_for_zero["alpha"])
        print("MEU for Zero Beta = %.3f" % meu_for_zero["beta"])
        print("MEU for Zero Max Val = %.3f" % meu_for_zero["max_val"])
        res2 = meu.max_exp_util_price_and_hedge(risk_aversion_param)
        print(res2)
        im_plot_pts = - (res2["beta"] + res2["alpha"] * x_plot_pts)
        plt.plot(
            x_plot_pts,
            im_plot_pts,
            color,
            label="Hedge for Risk-Aversion = %.1f" % risk_aversion_param
        )

    plt.xlim(lb, ub)
    plt.ylim(min(payoff_plot_pts), max(payoff_plot_pts))
    plt.grid(True)
    plt.legend()
    plt.show()
```

--------------------------------------------------------------------------------

## 📄 chapter8/optimal_exercise_bi.py {#optimal-exercise-bi}

**Titre**: Options Américaines - Backward Induction

**Description**: Pricing par programmation dynamique

**Lignes de code**: 258

**Concepts clés**:
- V(s,t) = max(exercise, continue)
- Backward induction depuis maturité
- Frontière d'exercice optimal

**🎯 Utilisation HelixOne**: Pricing d'options

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Iterator, List
import numpy as np
from scipy.stats import norm
from rl.distribution import SampledDistribution
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State, Terminal
from rl.policy import DeterministicPolicy
from rl.function_approx import FunctionApprox, LinearFunctionApprox
from rl.approximate_dynamic_programming import back_opt_vf_and_policy
from numpy.polynomial.laguerre import lagval


@dataclass(frozen=True)
class OptimalExerciseBI:
    '''Optimal Exercise with Backward Induction when the underlying
    price follows a lognormal process'''

    spot_price: float
    payoff: Callable[[float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int
    spot_price_frac: float

    def european_put_price(self, strike: float) -> float:
        sigma_sqrt: float = self.vol * np.sqrt(self.expiry)
        d1: float = (np.log(self.spot_price / strike) +
                     (self.rate + self.vol ** 2 / 2.) * self.expiry) \
            / sigma_sqrt
        d2: float = d1 - sigma_sqrt
        return strike * np.exp(-self.rate * self.expiry) * norm.cdf(-d2) \
            - self.spot_price * norm.cdf(-d1)

    def get_mdp(self, t: int) -> MarkovDecisionProcess[float, bool]:
        dt: float = self.expiry / self.num_steps
        exer_payoff: Callable[[float], float] = self.payoff
        r: float = self.rate
        s: float = self.vol

        class OptExerciseBIMDP(MarkovDecisionProcess[float, bool]):

            def step(
                self,
                price: NonTerminal[float],
                exer: bool
            ) -> SampledDistribution[Tuple[State[float], float]]:

                def sr_sampler_func(
                    price=price,
                    exer=exer
                ) -> Tuple[State[float], float]:
                    if exer:
                        return Terminal(0.), exer_payoff(price.state)
                    else:
                        next_price: float = np.exp(np.random.normal(
                            np.log(price.state) + (r - s * s / 2) * dt,
                            s * np.sqrt(dt)
                        ))
                        return NonTerminal(next_price), 0.

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=200
                )

            def actions(self, price: NonTerminal[float]) -> Sequence[bool]:
                return [True, False]

        return OptExerciseBIMDP()

    def get_states_distribution(
        self,
        t: int
    ) -> SampledDistribution[NonTerminal[float]]:
        spot_mean2: float = self.spot_price * self.spot_price
        spot_var: float = spot_mean2 * \
            self.spot_price_frac * self.spot_price_frac
        log_mean: float = np.log(spot_mean2 / np.sqrt(spot_var + spot_mean2))
        log_stdev: float = np.sqrt(np.log(spot_var / spot_mean2 + 1))

        time: float = t * self.expiry / self.num_steps

        def states_sampler_func() -> NonTerminal[float]:
            start: float = np.random.lognormal(log_mean, log_stdev)
            price = np.exp(np.random.normal(
                np.log(start) + (self.rate - self.vol * self.vol / 2) * time,
                self.vol * np.sqrt(time)
            ))
            return NonTerminal(price)

        return SampledDistribution(states_sampler_func)

    def get_vf_func_approx(
        self,
        t: int,
        features: Sequence[Callable[[NonTerminal[float]], float]],
        reg_coeff: float
    ) -> LinearFunctionApprox[NonTerminal[float]]:
        return LinearFunctionApprox.create(
            feature_functions=features,
            regularization_coeff=reg_coeff,
            direct_solve=True
        )

    def backward_induction_vf_and_pi(
        self,
        features: Sequence[Callable[[NonTerminal[float]], float]],
        reg_coeff: float
    ) -> Iterator[
        Tuple[FunctionApprox[NonTerminal[float]],
              DeterministicPolicy[float, bool]]
    ]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, bool],
            FunctionApprox[NonTerminal[float]],
            SampledDistribution[NonTerminal[float]]
        ]] = [(
            self.get_mdp(t=i),
            self.get_vf_func_approx(
                t=i,
                features=features,
                reg_coeff=reg_coeff
            ),
            self.get_states_distribution(t=i)
        ) for i in range(self.num_steps + 1)]

        num_state_samples: int = 1000

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=np.exp(-self.rate * self.expiry / self.num_steps),
            num_state_samples=num_state_samples,
            error_tolerance=1e-8
        )

    def optimal_value_curve(
        self,
        func: FunctionApprox[NonTerminal[float]],
        prices: Sequence[float]
    ) -> np.ndarray:
        return func.evaluate([NonTerminal(p) for p in prices])

    def exercise_curve(
        self,
        prices: Sequence[float]
    ) -> np.ndarray:
        return np.array([self.payoff(p) for p in prices])

    def put_option_exercise_boundary(
        self,
        opt_vfs: Sequence[FunctionApprox[NonTerminal[float]]],
        strike: float
    ) -> Sequence[float]:
        ret: List[float] = []
        prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
        for vf in opt_vfs[:-1]:
            cp: np.ndarray = self.optimal_value_curve(
                func=vf,
                prices=prices
            )
            ep: np.ndarray = self.exercise_curve(prices=prices)
            ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                                   if e > c]
            ret.append(max(ll) if len(ll) > 0 else 0.)
        final: Sequence[Tuple[float, float]] = \
            [(p, self.payoff(p)) for p in prices]
        ret.append(max(p for p, e in final if e > 0))
        return ret


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    spot_price_val: float = 100.0
    strike: float = 100.0
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 10
    spot_price_frac_val: float = 0.02

    opt_ex_bi: OptimalExerciseBI = OptimalExerciseBI(
        spot_price=spot_price_val,
        payoff=lambda x: max(strike - x, 0.),
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=num_steps_val,
        spot_price_frac=spot_price_frac_val
    )

    num_laguerre: int = 4
    reglr_coeff: float = 0.001

    ident: np.ndarray = np.eye(num_laguerre)
    ffs: List[Callable[[NonTerminal[float]], float]] = [lambda _: 1.]
    ffs += [(lambda s, i=i: np.log(1 + np.exp(-s.state / (2 * strike))) *
            lagval(s.state / strike, ident[i]))
            for i in range(num_laguerre)]
    it_vf = opt_ex_bi.backward_induction_vf_and_pi(
        features=ffs,
        reg_coeff=reglr_coeff
    )

    prices: np.ndarray = np.arange(120.0)

    print("Backward Induction: VF And Policy")
    print("---------------------------------")
    print()

    all_funcs: List[FunctionApprox[NonTerminal[float]]] = []
    for t, (v, p) in enumerate(it_vf):
        print(f"Time {t:d}")
        print()

        if t == 0 or t == int(num_steps_val / 2) or t == num_steps_val - 1:
            exer_curve: np.ndarray = opt_ex_bi.exercise_curve(
                prices=prices
            )
            opt_val_curve: np.ndarray = opt_ex_bi.optimal_value_curve(
                func=v,
                prices=prices
            )
            plt.plot(
                prices,
                opt_val_curve,
                "r",
                prices,
                exer_curve,
                "b"
            )
            time: float = t * expiry_val / num_steps_val
            plt.title(f"OptVal and Exercise Curves for Time = {time:.3f}")
            plt.show()

        all_funcs.append(v)

        opt_alloc: float = p.action_for(spot_price_val)
        val: float = v(NonTerminal(spot_price_val))
        print(f"Opt Action = {opt_alloc}, Opt Val = {val:.3f}")
        print()

    ex_bound: Sequence[float] = opt_ex_bi.put_option_exercise_boundary(
        all_funcs,
        strike
    )
    plt.plot(range(num_steps_val + 1), ex_bound)
    plt.title("Exercise Boundary")
    plt.show()

    print("European Put Price")
    print("------------------")
    print()
    print(opt_ex_bi.european_put_price(strike=strike))
```

--------------------------------------------------------------------------------

## 📄 chapter8/optimal_exercise_bin_tree.py {#optimal-exercise-bin-tree}

**Titre**: Optimal Exercise Bin Tree

**Description**: Module Optimal Exercise Bin Tree

**Lignes de code**: 133

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List
import numpy as np
from rl.dynamic_programming import V
from scipy.stats import norm
from rl.markov_decision_process import Terminal, NonTerminal
from rl.policy import FiniteDeterministicPolicy
from rl.distribution import Constant, Categorical
from rl.finite_horizon import optimal_vf_and_policy


@dataclass(frozen=True)
class OptimalExerciseBinTree:

    spot_price: float
    payoff: Callable[[float, float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int

    def european_price(self, is_call: bool, strike: float) -> float:
        sigma_sqrt: float = self.vol * np.sqrt(self.expiry)
        d1: float = (np.log(self.spot_price / strike) +
                     (self.rate + self.vol ** 2 / 2.) * self.expiry) \
            / sigma_sqrt
        d2: float = d1 - sigma_sqrt
        if is_call:
            ret = self.spot_price * norm.cdf(d1) - \
                strike * np.exp(-self.rate * self.expiry) * norm.cdf(d2)
        else:
            ret = strike * np.exp(-self.rate * self.expiry) * norm.cdf(-d2) - \
                self.spot_price * norm.cdf(-d1)
        return ret

    def dt(self) -> float:
        return self.expiry / self.num_steps

    def state_price(self, i: int, j: int) -> float:
        return self.spot_price * np.exp((2 * j - i) * self.vol *
                                        np.sqrt(self.dt()))

    def get_opt_vf_and_policy(self) -> \
            Iterator[Tuple[V[int], FiniteDeterministicPolicy[int, bool]]]:
        dt: float = self.dt()
        up_factor: float = np.exp(self.vol * np.sqrt(dt))
        up_prob: float = (np.exp(self.rate * dt) * up_factor - 1) / \
            (up_factor * up_factor - 1)
        return optimal_vf_and_policy(
            steps=[
                {NonTerminal(j): {
                    True: Constant(
                        (
                            Terminal(-1),
                            self.payoff(i * dt, self.state_price(i, j))
                        )
                    ),
                    False: Categorical(
                        {
                            (NonTerminal(j + 1), 0.): up_prob,
                            (NonTerminal(j), 0.): 1 - up_prob
                        }
                    )
                } for j in range(i + 1)}
                for i in range(self.num_steps + 1)
            ],
            gamma=np.exp(-self.rate * dt)
        )

    def option_exercise_boundary(
        self,
        policy_seq: Sequence[FiniteDeterministicPolicy[int, bool]],
        is_call: bool
    ) -> Sequence[Tuple[float, float]]:
        dt: float = self.dt()
        ex_boundary: List[Tuple[float, float]] = []
        for i in range(self.num_steps + 1):
            ex_points = [j for j in range(i + 1)
                         if policy_seq[i].action_for[j] and
                         self.payoff(i * dt, self.state_price(i, j)) > 0]
            if len(ex_points) > 0:
                boundary_pt = min(ex_points) if is_call else max(ex_points)
                ex_boundary.append(
                    (i * dt, self.state_price(i, boundary_pt))
                )
        return ex_boundary


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 300

    if is_call:
        opt_payoff = lambda _, x: max(x - strike, 0)
    else:
        opt_payoff = lambda _, x: max(strike - x, 0)

    opt_ex_bin_tree: OptimalExerciseBinTree = OptimalExerciseBinTree(
        spot_price=spot_price_val,
        payoff=opt_payoff,
        expiry=expiry_val,
        rate=rate_val,
        vol=vol_val,
        num_steps=num_steps_val
    )

    vf_seq, policy_seq = zip(*opt_ex_bin_tree.get_opt_vf_and_policy())
    ex_boundary: Sequence[Tuple[float, float]] = \
        opt_ex_bin_tree.option_exercise_boundary(policy_seq, is_call)
    time_pts, ex_bound_pts = zip(*ex_boundary)
    label = ("Call" if is_call else "Put") + " Option Exercise Boundary"
    plot_list_of_curves(
        list_of_x_vals=[time_pts],
        list_of_y_vals=[ex_bound_pts],
        list_of_colors=["b"],
        list_of_curve_labels=[label],
        x_label="Time",
        y_label="Underlying Price",
        title=label
    )

    european: float = opt_ex_bin_tree.european_price(is_call, strike)
    print(f"European Price = {european:.3f}")

    am_price: float = vf_seq[0][NonTerminal(0)]
    print(f"American Price = {am_price:.3f}")
```

--------------------------------------------------------------------------------

## 📄 chapter9/optimal_order_execution.py {#optimal-order-execution}

**Titre**: Exécution Optimale d'Ordres (Almgren-Chriss)

**Description**: Modèle classique d'exécution avec impact de marché

**Lignes de code**: 234

**Concepts clés**:
- Impact temporaire: η * v (proportionnel au volume)
- Impact permanent: γ * v (modifie le prix fondamental)
- Trade-off: exécuter vite (risque) vs lent (coût)
- Trajectoire optimale analytique
- MDP formulation avec RL

**🎯 Utilisation HelixOne**: CRITIQUE - Module d'exécution d'ordres

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Iterator
from rl.distribution import Distribution, SampledDistribution, Choose
from rl.function_approx import FunctionApprox, LinearFunctionApprox
from rl.markov_decision_process import MarkovDecisionProcess, \
    NonTerminal, State
from rl.policy import DeterministicPolicy
from rl.approximate_dynamic_programming import back_opt_vf_and_policy, \
    ValueFunctionApprox


@dataclass(frozen=True)
class PriceAndShares:
    price: float
    shares: int


@dataclass(frozen=True)
class OptimalOrderExecution:
    '''
    shares refers to the total number of shares N to be sold over
    T time steps.

    time_steps refers to the number of time steps T.

    avg_exec_price_diff refers to the time-sequenced functions g_t
    that gives the average reduction in the price obtained by the
    Market Order at time t due to eating into the Buy LOs. g_t is
    a function of PriceAndShares that represents the pair of Price P_t
    and MO size N_t. Sales Proceeds = N_t*(P_t - g_t(P_t, N_t)).

    price_dynamics refers to the time-sequenced functions f_t that
    represents the price dynamics: P_{t+1} ~ f_t(P_t, N_t). f_t
    outputs a distribution of prices.

    utility_func refers to the Utility of Sales proceeds function,
    incorporating any risk-aversion.

    discount_factor refers to the discount factor gamma.

    func_approx refers to the FunctionApprox required to approximate
    the Value Function for each time step.

    initial_price_distribution refers to the distribution of prices
    at time 0 (needed to generate the samples of states at each time step,
    needed in the approximate backward induction algorithm).
    '''
    shares: int
    time_steps: int
    avg_exec_price_diff: Sequence[Callable[[PriceAndShares], float]]
    price_dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]]
    utility_func: Callable[[float], float]
    discount_factor: float
    func_approx: ValueFunctionApprox[PriceAndShares]
    initial_price_distribution: Distribution[float]

    def get_mdp(self, t: int) -> MarkovDecisionProcess[PriceAndShares, int]:
        """
        State is (Price P_t, Remaining Shares R_t)
        Action is shares sold N_t
        """

        utility_f: Callable[[float], float] = self.utility_func
        price_diff: Sequence[Callable[[PriceAndShares], float]] = \
            self.avg_exec_price_diff
        dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]] = \
            self.price_dynamics
        steps: int = self.time_steps

        class OptimalExecutionMDP(MarkovDecisionProcess[PriceAndShares, int]):

            def step(
                self,
                p_r: NonTerminal[PriceAndShares],
                sell: int
            ) -> SampledDistribution[Tuple[State[PriceAndShares],
                                           float]]:

                def sr_sampler_func(
                    p_r=p_r,
                    sell=sell
                ) -> Tuple[State[PriceAndShares], float]:
                    p_s: PriceAndShares = PriceAndShares(
                        price=p_r.state.price,
                        shares=sell
                    )
                    next_price: float = dynamics[t](p_s).sample()
                    next_rem: int = p_r.state.shares - sell
                    next_state: PriceAndShares = PriceAndShares(
                        price=next_price,
                        shares=next_rem
                    )
                    reward: float = utility_f(
                        sell * (p_r.state.price - price_diff[t](p_s))
                    )
                    return (NonTerminal(next_state), reward)

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=100
                )

            def actions(self, p_s: NonTerminal[PriceAndShares]) -> \
                    Iterator[int]:
                if t == steps - 1:
                    return iter([p_s.state.shares])
                else:
                    return iter(range(p_s.state.shares + 1))

        return OptimalExecutionMDP()

    def get_states_distribution(self, t: int) -> \
            SampledDistribution[NonTerminal[PriceAndShares]]:

        def states_sampler_func() -> NonTerminal[PriceAndShares]:
            price: float = self.initial_price_distribution.sample()
            rem: int = self.shares
            for i in range(t):
                sell: int = Choose(range(rem + 1)).sample()
                price = self.price_dynamics[i](PriceAndShares(
                    price=price,
                    shares=rem
                )).sample()
                rem -= sell
            return NonTerminal(PriceAndShares(
                price=price,
                shares=rem
            ))

        return SampledDistribution(states_sampler_func)

    def backward_induction_vf_and_pi(
        self
    ) -> Iterator[Tuple[ValueFunctionApprox[PriceAndShares],
                        DeterministicPolicy[PriceAndShares, int]]]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[PriceAndShares, int],
            ValueFunctionApprox[PriceAndShares],
            SampledDistribution[NonTerminal[PriceAndShares]]
        ]] = [(
            self.get_mdp(i),
            self.func_approx,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps)]

        num_state_samples: int = 10000
        error_tolerance: float = 1e-6

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=self.discount_factor,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )


if __name__ == '__main__':

    from rl.distribution import Gaussian

    init_price_mean: float = 100.0
    init_price_stdev: float = 10.0
    num_shares: int = 100
    num_time_steps: int = 5
    alpha: float = 0.03
    beta: float = 0.05

    price_diff = [lambda p_s: beta * p_s.shares for _ in range(num_time_steps)]
    dynamics = [lambda p_s: Gaussian(
        μ=p_s.price - alpha * p_s.shares,
        σ=0.
    ) for _ in range(num_time_steps)]
    ffs = [
        lambda p_s: p_s.state.price * p_s.state.shares,
        lambda p_s: float(p_s.state.shares * p_s.state.shares)
    ]
    fa: FunctionApprox = LinearFunctionApprox.create(feature_functions=ffs)
    init_price_distrib: Gaussian = Gaussian(
        μ=init_price_mean,
        σ=init_price_stdev
    )

    ooe: OptimalOrderExecution = OptimalOrderExecution(
        shares=num_shares,
        time_steps=num_time_steps,
        avg_exec_price_diff=price_diff,
        price_dynamics=dynamics,
        utility_func=lambda x: x,
        discount_factor=1,
        func_approx=fa,
        initial_price_distribution=init_price_distrib
    )
    it_vf: Iterator[Tuple[ValueFunctionApprox[PriceAndShares],
                          DeterministicPolicy[PriceAndShares, int]]] = \
        ooe.backward_induction_vf_and_pi()

    state: PriceAndShares = PriceAndShares(
        price=init_price_mean,
        shares=num_shares
    )
    print("Backward Induction: VF And Policy")
    print("---------------------------------")
    print()
    for t, (vf, pol) in enumerate(it_vf):
        print(f"Time {t:d}")
        print()
        opt_sale: int = pol.action_for(state)
        val: float = vf(NonTerminal(state))
        print(f"Optimal Sales = {opt_sale:d}, Opt Val = {val:.3f}")
        print()
        print("Optimal Weights below:")
        print(vf.weights.weights)
        print()

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(num_time_steps):
        print(f"Time {t:d}")
        print()
        left: int = num_time_steps - t
        opt_sale_anal: float = num_shares / num_time_steps
        wt1: float = 1
        wt2: float = -(2 * beta + alpha * (left - 1)) / (2 * left)
        val_anal: float = wt1 * state.price * state.shares + \
            wt2 * state.shares * state.shares

        print(f"Optimal Sales = {opt_sale_anal:.3f}, Opt Val = {val_anal:.3f}")
        print(f"Weight1 = {wt1:.3f}")
        print(f"Weight2 = {wt2:.3f}")
        print()
```

--------------------------------------------------------------------------------

## 📄 chapter9/order_book.py {#order-book}

**Titre**: Modélisation du Carnet d'Ordres (LOB)

**Description**: Dynamique du carnet d'ordres limite

**Lignes de code**: 280

**Concepts clés**:
- Bid/Ask spread dynamics
- Market making comme MDP
- Gestion de l'inventaire
- Risque d'adverse selection

**🎯 Utilisation HelixOne**: Microstructure pour exécution avancée

### Code Source Complet

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple, Optional, List


@dataclass(frozen=True)
class DollarsAndShares:

    dollars: float
    shares: int


PriceSizePairs = Sequence[DollarsAndShares]


@dataclass(frozen=True)
class OrderBook:

    descending_bids: PriceSizePairs
    ascending_asks: PriceSizePairs

    def bid_price(self) -> float:
        return self.descending_bids[0].dollars

    def ask_price(self) -> float:
        return self.ascending_asks[0].dollars

    def mid_price(self) -> float:
        return (self.bid_price() + self.ask_price()) / 2

    def bid_ask_spread(self) -> float:
        return self.ask_price() - self.bid_price()

    def market_depth(self) -> float:
        return self.ascending_asks[-1].dollars - \
            self.descending_bids[-1].dollars

    @staticmethod
    def eat_book(
        ps_pairs: PriceSizePairs,
        shares: int
    ) -> Tuple[DollarsAndShares, PriceSizePairs]:
        '''
        Returned DollarsAndShares represents the pair of
        dollars transacted and the number of shares transacted
        on ps_pairs (with number of shares transacted being less
        than or equal to the input shares).
        Returned PriceSizePairs represents the remainder of the
        ps_pairs after the transacted number of shares have eaten into
        the input ps_pairs.
        '''
        rem_shares: int = shares
        dollars: float = 0.
        for i, d_s in enumerate(ps_pairs):
            this_price: float = d_s.dollars
            this_shares: int = d_s.shares
            dollars += this_price * min(rem_shares, this_shares)
            if rem_shares < this_shares:
                return (
                    DollarsAndShares(dollars=dollars, shares=shares),
                    [DollarsAndShares(
                        dollars=this_price,
                        shares=this_shares - rem_shares
                    )] + list(ps_pairs[i+1:])
                )
            else:
                rem_shares -= this_shares

        return (
            DollarsAndShares(dollars=dollars, shares=shares - rem_shares),
            []
        )

    def sell_limit_order(self, price: float, shares: int) -> \
            Tuple[DollarsAndShares, OrderBook]:
        index: Optional[int] = next((i for i, d_s
                                     in enumerate(self.descending_bids)
                                     if d_s.dollars < price), None)
        eligible_bids: PriceSizePairs = self.descending_bids \
            if index is None else self.descending_bids[:index]
        ineligible_bids: PriceSizePairs = [] if index is None else \
            self.descending_bids[index:]

        d_s, rem_bids = OrderBook.eat_book(eligible_bids, shares)
        new_bids: PriceSizePairs = list(rem_bids) + list(ineligible_bids)
        rem_shares: int = shares - d_s.shares

        if rem_shares > 0:
            new_asks: List[DollarsAndShares] = list(self.ascending_asks)
            index1: Optional[int] = next((i for i, d_s
                                          in enumerate(new_asks)
                                          if d_s.dollars >= price), None)
            if index1 is None:
                new_asks.append(DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            elif new_asks[index1].dollars != price:
                new_asks.insert(index1, DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            else:
                new_asks[index1] = DollarsAndShares(
                    dollars=price,
                    shares=new_asks[index1].shares + rem_shares
                )
            return d_s, OrderBook(
                ascending_asks=new_asks,
                descending_bids=new_bids
            )
        else:
            return d_s, replace(
                self,
                descending_bids=new_bids
            )

    def sell_market_order(
        self,
        shares: int
    ) -> Tuple[DollarsAndShares, OrderBook]:
        d_s, rem_bids = OrderBook.eat_book(
            self.descending_bids,
            shares
        )
        return (d_s, replace(self, descending_bids=rem_bids))

    def buy_limit_order(self, price: float, shares: int) -> \
            Tuple[DollarsAndShares, OrderBook]:
        index: Optional[int] = next((i for i, d_s
                                     in enumerate(self.ascending_asks)
                                     if d_s.dollars > price), None)
        eligible_asks: PriceSizePairs = self.ascending_asks \
            if index is None else self.ascending_asks[:index]
        ineligible_asks: PriceSizePairs = [] if index is None else \
            self.ascending_asks[index:]

        d_s, rem_asks = OrderBook.eat_book(eligible_asks, shares)
        new_asks: PriceSizePairs = list(rem_asks) + list(ineligible_asks)
        rem_shares: int = shares - d_s.shares

        if rem_shares > 0:
            new_bids: List[DollarsAndShares] = list(self.descending_bids)
            index1: Optional[int] = next((i for i, d_s
                                          in enumerate(new_bids)
                                          if d_s.dollars <= price), None)
            if index1 is None:
                new_bids.append(DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            elif new_bids[index1].dollars != price:
                new_bids.insert(index1, DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            else:
                new_bids[index1] = DollarsAndShares(
                    dollars=price,
                    shares=new_bids[index1].shares + rem_shares
                )
            return d_s, replace(
                self,
                ascending_asks=new_asks,
                descending_bids=new_bids
            )
        else:
            return d_s, replace(
                self,
                ascending_asks=new_asks
            )

    def buy_market_order(
        self,
        shares: int
    ) -> Tuple[DollarsAndShares, OrderBook]:
        d_s, rem_asks = OrderBook.eat_book(
            self.ascending_asks,
            shares
        )
        return (d_s, replace(self, ascending_asks=rem_asks))

    def pretty_print_order_book(self) -> None:
        from pprint import pprint
        print()
        print("Bids")
        pprint(self.descending_bids)
        print()
        print("Asks")
        print()
        pprint(self.ascending_asks)
        print()

    def display_order_book(self) -> None:
        import matplotlib.pyplot as plt

        bid_prices = [d_s.dollars for d_s in self.descending_bids]
        bid_shares = [d_s.shares for d_s in self.descending_bids]
        if self.descending_bids:
            plt.bar(bid_prices, bid_shares, color='blue')

        ask_prices = [d_s.dollars for d_s in self.ascending_asks]
        ask_shares = [d_s.shares for d_s in self.ascending_asks]
        if self.ascending_asks:
            plt.bar(ask_prices, ask_shares, color='red')

        all_prices = sorted(bid_prices + ask_prices)
        all_ticks = ["%d" % x for x in all_prices]
        plt.xticks(all_prices, all_ticks)
        plt.grid(axis='y')
        plt.xlabel("Prices")
        plt.ylabel("Number of Shares")
        plt.title("Order Book")
        # plt.xticks(x_pos, x)
        plt.show()


if __name__ == '__main__':

    from numpy.random import poisson

    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    ob0.pretty_print_order_book()
    ob0.display_order_book()

    print("Sell Limit Order of (107, 40)")
    print()
    d_s1, ob1 = ob0.sell_limit_order(107, 40)
    proceeds1: float = d_s1.dollars
    shares_sold1: int = d_s1.shares
    print(f"Sales Proceeds = {proceeds1:.2f}, Shares Sold = {shares_sold1:d}")
    ob1.pretty_print_order_book()
    ob1.display_order_book()

    print("Sell Market Order of 120")
    print()
    d_s2, ob2 = ob1.sell_market_order(120)
    proceeds2: float = d_s2.dollars
    shares_sold2: int = d_s2.shares
    print(f"Sales Proceeds = {proceeds2:.2f}, Shares Sold = {shares_sold2:d}")
    ob2.pretty_print_order_book()
    ob2.display_order_book()

    print("Buy Limit Order of (100, 80)")
    print()
    d_s3, ob3 = ob2.buy_limit_order(100, 80)
    bill3: float = d_s3.dollars
    shares_bought3: int = d_s3.shares
    print(f"Purchase Bill = {bill3:.2f}, Shares Bought = {shares_bought3:d}")
    ob3.pretty_print_order_book()
    ob3.display_order_book()

    print("Sell Limit Order of (104, 60)")
    print()
    d_s4, ob4 = ob3.sell_limit_order(104, 60)
    proceeds4: float = d_s4.dollars
    shares_sold4: int = d_s4.shares
    print(f"Sales Proceeds = {proceeds4:.2f}, Shares Sold = {shares_sold4:d}")
    ob4.pretty_print_order_book()
    ob4.display_order_book()

    print("Buy Market Order of 150")
    print()
    d_s5, ob5 = ob4.buy_market_order(150)
    bill5: float = d_s5.dollars
    shares_bought5: int = d_s5.shares
    print(f"Purchase Bill = {bill5:.2f}, Shares Bought = {shares_bought5:d}")
    ob5.pretty_print_order_book()
    ob5.display_order_book()
```

--------------------------------------------------------------------------------

# PARTIE 8 - EXEMPLES ET PROBLÈMES

================================================================================

## 📄 chapter1/probability.py {#probability}

**Titre**: Probability

**Description**: Module Probability

**Lignes de code**: 60

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import statistics
from typing import Generic, TypeVar


A = TypeVar("A")


class Distribution(ABC, Generic[A]):
    @abstractmethod
    def sample(self) -> A:
        pass


class OldDie(Distribution):
    def __init__(self, sides):
        self.sides = sides

    def __repr__(self):
        return f"Die(sides={self.sides})"

    def __eq__(self, other):
        if isinstance(other, Die):
            return self.sides == other.sides

        return False

    def sample(self) -> int:
        return random.randint(1, self.sides)


six_sided = OldDie(6)


def roll_dice():
    return six_sided.sample() + six_sided.sample()


@dataclass(frozen=True)
class Coin(Distribution[str]):
    def sample(self):
        return "heads" if random.random() < 0.5 else "tails"


@dataclass(frozen=True)
class Die(Distribution):
    sides: int

    def sample(self):
        return random.randint(1, self.sides)


def expected_value(d: Distribution[float], n: int) -> float:
    return statistics.mean(d.sample() for _ in range(n))


expected_value(Die(6), 100)
```

--------------------------------------------------------------------------------

## 📄 chapter10/memory_function.py {#memory-function}

**Titre**: Memory Function

**Description**: Module Memory Function

**Lignes de code**: 29

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, List
import math
import matplotlib.pyplot as plt


def plot_memory_function(theta: float, event_times: List[float]) -> None:
    step: float = 0.01
    x_vals: List[float] = [0.0]
    y_vals: List[float] = [0.0]
    for t in event_times:
        rng: Sequence[int] = range(1, int(math.floor((t - x_vals[-1]) / step)))
        x_vals += [x_vals[-1] + i * step for i in rng]
        y_vals += [y_vals[-1] * theta ** (i * step) for i in rng]
        x_vals.append(t)
        y_vals.append(y_vals[-1] * theta ** (t - x_vals[-1]) + 1.0)
    plt.plot(x_vals, y_vals)
    plt.grid()
    plt.xticks([0.0] + event_times)
    plt.xlabel("Event Timings", fontsize=15)
    plt.ylabel("Memory Funtion Values", fontsize=15)
    plt.title("Memory Function (Frequency and Recency)", fontsize=25)
    plt.show()


if __name__ == '__main__':
    theta = 0.8
    event_times = [2.0, 3.0, 4.0, 7.0, 9.0, 14.0, 15.0, 21.0]
    plot_memory_function(theta, event_times)
```

--------------------------------------------------------------------------------

## 📄 chapter10/prediction_utils.py {#prediction-utils}

**Titre**: Prediction Utils

**Description**: Module Prediction Utils

**Lignes de code**: 423

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Iterable, TypeVar, Callable, Iterator, Sequence, \
    Tuple, Mapping
from rl.function_approx import Tabular
from rl.distribution import Choose
from rl.markov_process import (MarkovRewardProcess, NonTerminal,
                               FiniteMarkovRewardProcess, TransitionStep)
import itertools
import rl.iterate as iterate
from rl.returns import returns
import rl.monte_carlo as mc
from rl.function_approx import learning_rate_schedule
import rl.td as td
import rl.td_lambda as td_lambda
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.approximate_dynamic_programming import NTStateDistribution
import numpy as np
from math import sqrt
from pprint import pprint

S = TypeVar('S')


def mrp_episodes_stream(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: NTStateDistribution[S]
) -> Iterable[Iterable[TransitionStep[S]]]:
    return mrp.reward_traces(start_state_distribution)


def fmrp_episodes_stream(
    fmrp: FiniteMarkovRewardProcess[S]
) -> Iterable[Iterable[TransitionStep[S]]]:
    return mrp_episodes_stream(fmrp, Choose(fmrp.non_terminal_states))


def mc_finite_prediction_equal_wts(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length_tolerance: float,
    initial_vf_dict: Mapping[NonTerminal[S], float]
) -> Iterator[ValueFunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp_episodes_stream(fmrp)
    return mc.mc_prediction(
        traces=episodes,
        approx_0=Tabular(values_map=initial_vf_dict),
        γ=gamma,
        episode_length_tolerance=episode_length_tolerance
    )


def mc_prediction_learning_rate(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: NTStateDistribution[S],
    gamma: float,
    episode_length_tolerance: float,
    initial_func_approx: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    return mc.mc_prediction(
        traces=episodes,
        approx_0=initial_func_approx,
        γ=gamma,
        episode_length_tolerance=episode_length_tolerance
    )


def mc_finite_prediction_learning_rate(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length_tolerance: float,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[NonTerminal[S], float]
) -> Iterator[ValueFunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp_episodes_stream(fmrp)
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return mc.mc_prediction(
        traces=episodes,
        approx_0=Tabular(
            values_map=initial_vf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        episode_length_tolerance=episode_length_tolerance
    )


def unit_experiences_from_episodes(
    episodes: Iterable[Iterable[TransitionStep[S]]],
    episode_length: int
) -> Iterable[TransitionStep[S]]:
    return itertools.chain.from_iterable(
        itertools.islice(episode, episode_length) for episode in episodes
    )


def td_prediction_learning_rate(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: NTStateDistribution[S],
    gamma: float,
    episode_length: int,
    initial_func_approx: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    td_experiences: Iterable[TransitionStep[S]] = \
        unit_experiences_from_episodes(
            episodes,
            episode_length
        )
    return td.td_prediction(
        transitions=td_experiences,
        approx_0=initial_func_approx,
        γ=gamma
    )


def td_finite_prediction_learning_rate(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[NonTerminal[S], float]
) -> Iterator[ValueFunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp_episodes_stream(fmrp)
    td_experiences: Iterable[TransitionStep[S]] = \
        unit_experiences_from_episodes(
            episodes,
            episode_length
        )
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return td.td_prediction(
        transitions=td_experiences,
        approx_0=Tabular(
            values_map=initial_vf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma
    )


def td_lambda_prediction_learning_rate(
    mrp: MarkovRewardProcess[S],
    start_state_distribution: NTStateDistribution[S],
    gamma: float,
    lambd: float,
    episode_length: int,
    initial_func_approx: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        mrp_episodes_stream(mrp, start_state_distribution)
    curtailed_episodes: Iterable[Iterable[TransitionStep[S]]] = \
        (itertools.islice(episode, episode_length) for episode in episodes)
    return td_lambda.td_lambda_prediction(
        traces=curtailed_episodes,
        approx_0=initial_func_approx,
        γ=gamma,
        lambd=lambd
    )


def td_lambda_finite_prediction_learning_rate(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    lambd: float,
    episode_length: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[NonTerminal[S], float]
) -> Iterator[ValueFunctionApprox[S]]:
    episodes: Iterable[Iterable[TransitionStep[S]]] = \
        fmrp_episodes_stream(fmrp)
    curtailed_episodes: Iterable[Iterable[TransitionStep[S]]] = \
        (itertools.islice(episode, episode_length) for episode in episodes)
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return td_lambda.td_lambda_prediction(
        traces=curtailed_episodes,
        approx_0=Tabular(
            values_map=initial_vf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        lambd=lambd
    )


def mc_finite_equal_wts_correctness(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length_tolerance: float,
    num_episodes: int,
    initial_vf_dict: Mapping[NonTerminal[S], float]
) -> None:
    mc_vfs: Iterator[ValueFunctionApprox[S]] = \
        mc_finite_prediction_equal_wts(
            fmrp=fmrp,
            gamma=gamma,
            episode_length_tolerance=episode_length_tolerance,
            initial_vf_dict=initial_vf_dict
        )
    final_mc_vf: ValueFunctionApprox[S] = \
        iterate.last(itertools.islice(mc_vfs, num_episodes))
    print(f"Equal-Weights-MC Value Function with {num_episodes:d} episodes")
    pprint({s: round(final_mc_vf(s), 3) for s in fmrp.non_terminal_states})
    print("True Value Function")
    fmrp.display_value_function(gamma=gamma)


def mc_finite_learning_rate_correctness(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length_tolerance: float,
    num_episodes: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[NonTerminal[S], float]
) -> None:
    mc_vfs: Iterator[ValueFunctionApprox[S]] = \
        mc_finite_prediction_learning_rate(
            fmrp=fmrp,
            gamma=gamma,
            episode_length_tolerance=episode_length_tolerance,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            initial_vf_dict=initial_vf_dict
        )
    final_mc_vf: ValueFunctionApprox[S] = \
        iterate.last(itertools.islice(mc_vfs, num_episodes))
    print("Decaying-Learning-Rate-MC Value Function with " +
          f"{num_episodes:d} episodes")
    pprint({s: round(final_mc_vf(s), 3) for s in fmrp.non_terminal_states})
    print("True Value Function")
    fmrp.display_value_function(gamma=gamma)


def td_finite_learning_rate_correctness(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    episode_length: int,
    num_episodes: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[NonTerminal[S], float]
) -> None:
    td_vfs: Iterator[ValueFunctionApprox[S]] = \
        td_finite_prediction_learning_rate(
            fmrp=fmrp,
            gamma=gamma,
            episode_length=episode_length,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            initial_vf_dict=initial_vf_dict
        )
    final_td_vf: ValueFunctionApprox[S] = \
        iterate.last(itertools.islice(td_vfs, episode_length * num_episodes))
    print("Decaying-Learning-Rate-TD Value Function with " +
          f"{num_episodes:d} episodes")
    pprint({s: round(final_td_vf(s), 3) for s in fmrp.non_terminal_states})
    print("True Value Function")
    fmrp.display_value_function(gamma=gamma)


def td_lambda_finite_learning_rate_correctness(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    lambd: float,
    episode_length: int,
    num_episodes: int,
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    initial_vf_dict: Mapping[NonTerminal[S], float]
) -> None:
    td_lambda_vfs: Iterator[ValueFunctionApprox[S]] = \
        td_lambda_finite_prediction_learning_rate(
            fmrp=fmrp,
            gamma=gamma,
            lambd=lambd,
            episode_length=episode_length,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            initial_vf_dict=initial_vf_dict
        )
    final_td_lambda_vf: ValueFunctionApprox[S] = \
        iterate.last(itertools.islice(
            td_lambda_vfs,
            episode_length * num_episodes
        ))
    print("Decaying-Learning-Rate-TD-Lambda Value Function with " +
          f"{num_episodes:d} episodes")
    pprint({s: round(final_td_lambda_vf(s), 3)
            for s in fmrp.non_terminal_states})
    print("True Value Function")
    fmrp.display_value_function(gamma=gamma)


def compare_td_and_mc(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    mc_episode_length_tol: float,
    num_episodes: int,
    learning_rates: Sequence[Tuple[float, float, float]],
    initial_vf_dict: Mapping[NonTerminal[S], float],
    plot_batch: int,
    plot_start: int
) -> None:
    true_vf: np.ndarray = fmrp.get_value_function_vec(gamma)
    states: Sequence[NonTerminal[S]] = fmrp.non_terminal_states
    colors: Sequence[str] = ['r', 'y', 'm', 'g', 'c', 'k', 'b']

    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))

    for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
        mc_funcs_it: Iterator[ValueFunctionApprox[S]] = \
            mc_finite_prediction_learning_rate(
                fmrp=fmrp,
                gamma=gamma,
                episode_length_tolerance=mc_episode_length_tol,
                initial_learning_rate=init_lr,
                half_life=half_life,
                exponent=exponent,
                initial_vf_dict=initial_vf_dict
            )
        mc_errors = []
        batch_mc_errs = []
        for i, mc_f in enumerate(itertools.islice(mc_funcs_it, num_episodes)):
            batch_mc_errs.append(sqrt(sum(
                (mc_f(s) - true_vf[j]) ** 2 for j, s in enumerate(states)
            ) / len(states)))
            if i % plot_batch == plot_batch - 1:
                mc_errors.append(sum(batch_mc_errs) / plot_batch)
                batch_mc_errs = []
        mc_plot = mc_errors[plot_start:]
        label = f"MC InitRate={init_lr:.3f},HalfLife" + \
            f"={half_life:.0f},Exp={exponent:.1f}"
        plt.plot(
            range(len(mc_plot)),
            mc_plot,
            color=colors[k],
            linestyle='-',
            label=label
        )

    sample_episodes: int = 1000
    td_episode_length: int = int(round(sum(
        len(list(returns(
            trace=fmrp.simulate_reward(Choose(states)),
            γ=gamma,
            tolerance=mc_episode_length_tol
        ))) for _ in range(sample_episodes)
    ) / sample_episodes))

    for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
        td_funcs_it: Iterator[ValueFunctionApprox[S]] = \
            td_finite_prediction_learning_rate(
                fmrp=fmrp,
                gamma=gamma,
                episode_length=td_episode_length,
                initial_learning_rate=init_lr,
                half_life=half_life,
                exponent=exponent,
                initial_vf_dict=initial_vf_dict
            )
        td_errors = []
        transitions_batch = plot_batch * td_episode_length
        batch_td_errs = []

        for i, td_f in enumerate(
                itertools.islice(td_funcs_it, num_episodes * td_episode_length)
        ):
            batch_td_errs.append(sqrt(sum(
                (td_f(s) - true_vf[j]) ** 2 for j, s in enumerate(states)
            ) / len(states)))
            if i % transitions_batch == transitions_batch - 1:
                td_errors.append(sum(batch_td_errs) / transitions_batch)
                batch_td_errs = []
        td_plot = td_errors[plot_start:]
        label = f"TD InitRate={init_lr:.3f},HalfLife" + \
            f"={half_life:.0f},Exp={exponent:.1f}"
        plt.plot(
            range(len(td_plot)),
            td_plot,
            color=colors[k],
            linestyle='--',
            label=label
        )

    plt.xlabel("Episode Batches", fontsize=20)
    plt.ylabel("Value Function RMSE", fontsize=20)
    plt.title(
        "RMSE of MC and TD as function of episode batches",
        fontsize=25
    )
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.show()
```

--------------------------------------------------------------------------------

## 📄 chapter10/random_walk_mrp.py {#random-walk-mrp}

**Titre**: Random Walk Mrp

**Description**: Module Random Walk Mrp

**Lignes de code**: 59

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Mapping, Dict, Tuple
from rl.distribution import Categorical
from rl.markov_process import FiniteMarkovRewardProcess


class RandomWalkMRP(FiniteMarkovRewardProcess[int]):
    '''
    This MRP's states are {0, 1, 2,...,self.barrier}
    with 0 and self.barrier as the terminal states.
    At each time step, we go from state i to state
    i+1 with probability self.p or to state i-1 with
    probability 1-self.p, for all 0 < i < self.barrier.
    The reward is 0 if we transition to a non-terminal
    state or to terminal state 0, and the reward is 1
    if we transition to terminal state self.barrier
    '''
    barrier: int
    p: float

    def __init__(
        self,
        barrier: int,
        p: float
    ):
        self.barrier = barrier
        self.p = p
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[int, Categorical[Tuple[int, float]]]:
        d: Dict[int, Categorical[Tuple[int, float]]] = {
            i: Categorical({
                (i + 1, 0. if i < self.barrier - 1 else 1.): self.p,
                (i - 1, 0.): 1 - self.p
            }) for i in range(1, self.barrier)
        }
        return d


if __name__ == '__main__':
    from rl.chapter10.prediction_utils import compare_td_and_mc

    this_barrier: int = 10
    this_p: float = 0.5
    random_walk: RandomWalkMRP = RandomWalkMRP(
        barrier=this_barrier,
        p=this_p
    )
    compare_td_and_mc(
        fmrp=random_walk,
        gamma=1.0,
        mc_episode_length_tol=1e-6,
        num_episodes=700,
        learning_rates=[(0.01, 1e8, 0.5), (0.05, 1e8, 0.5)],
        initial_vf_dict={s: 0.5 for s in random_walk.non_terminal_states},
        plot_batch=7,
        plot_start=0
    )
```

--------------------------------------------------------------------------------

## 📄 chapter10/simple_inventory_mrp.py {#simple-inventory-mrp}

**Titre**: Simple Inventory Mrp

**Description**: Module Simple Inventory Mrp

**Lignes de code**: 93

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Tuple, Mapping
from rl.markov_decision_process import NonTerminal
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.chapter10.prediction_utils import (
    mc_finite_equal_wts_correctness,
    mc_finite_learning_rate_correctness,
    td_finite_learning_rate_correctness,
    td_lambda_finite_learning_rate_correctness,
    compare_td_and_mc
)


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

si_mrp: SimpleInventoryMRPFinite = SimpleInventoryMRPFinite(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)
initial_vf_dict: Mapping[NonTerminal[InventoryState], float] = \
    {s: 0. for s in si_mrp.non_terminal_states}

gamma: float = 0.9
mc_episode_length_tol: float = 1e-6
num_episodes = 10000

td_episode_length: int = 100
initial_learning_rate: float = 0.03
half_life: float = 1000.0
exponent: float = 0.5

lambda_param = 0.3

mc_finite_equal_wts_correctness(
    fmrp=si_mrp,
    gamma=gamma,
    episode_length_tolerance=mc_episode_length_tol,
    num_episodes=num_episodes,
    initial_vf_dict=initial_vf_dict
)
mc_finite_learning_rate_correctness(
    fmrp=si_mrp,
    gamma=gamma,
    episode_length_tolerance=mc_episode_length_tol,
    num_episodes=num_episodes,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    initial_vf_dict=initial_vf_dict
)
td_finite_learning_rate_correctness(
    fmrp=si_mrp,
    gamma=gamma,
    episode_length=td_episode_length,
    num_episodes=num_episodes,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    initial_vf_dict=initial_vf_dict
)
td_lambda_finite_learning_rate_correctness(
    fmrp=si_mrp,
    gamma=gamma,
    lambd=lambda_param,
    episode_length=td_episode_length,
    num_episodes=num_episodes,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    initial_vf_dict=initial_vf_dict
)

plot_batch: int = 50
plot_start: int = 0
learning_rates: Sequence[Tuple[float, float, float]] = \
    [(0.01, 100000, 0.5), (0.03, 1000, 0.5)]

compare_td_and_mc(
    fmrp=si_mrp,
    gamma=gamma,
    mc_episode_length_tol=mc_episode_length_tol,
    num_episodes=num_episodes,
    learning_rates=learning_rates,
    initial_vf_dict=initial_vf_dict,
    plot_batch=plot_batch,
    plot_start=plot_start
)
```

--------------------------------------------------------------------------------

## 📄 chapter11/control_utils.py {#control-utils}

**Titre**: Control Utils

**Description**: Module Control Utils

**Lignes de code**: 534

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import TypeVar, Callable, Iterator, Sequence, Tuple, Mapping
from rl.function_approx import Tabular
from rl.distribution import Choose
from rl.markov_process import NonTerminal
from rl.markov_decision_process import (
    MarkovDecisionProcess, FiniteMarkovDecisionProcess,
    FiniteMarkovRewardProcess)
from rl.policy import FiniteDeterministicPolicy, FinitePolicy
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import NTStateDistribution
import itertools
import rl.iterate as iterate
from rl.returns import returns
import rl.monte_carlo as mc
import rl.td as td
from rl.function_approx import learning_rate_schedule
from rl.dynamic_programming import V, value_iteration_result
from math import sqrt
from pprint import pprint

S = TypeVar('S')
A = TypeVar('A')


def glie_mc_finite_control_equal_wts(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-5,
) -> Iterator[QValueFunctionApprox[S, A]]:
    initial_qvf_dict: Mapping[Tuple[NonTerminal[S], A], float] = {
        (s, a): 0. for s in fmdp.non_terminal_states for a in fmdp.actions(s)
    }
    return mc.glie_mc_control(
        mdp=fmdp,
        states=Choose(fmdp.non_terminal_states),
        approx_0=Tabular(values_map=initial_qvf_dict),
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        episode_length_tolerance=episode_length_tolerance
    )


def glie_mc_control_learning_rate(
    mdp: MarkovDecisionProcess[S, A],
    start_state_distribution: NTStateDistribution,
    initial_func_approx: QValueFunctionApprox[S, A],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-5
) -> Iterator[QValueFunctionApprox[S, A]]:
    return mc.glie_mc_control(
        mdp=mdp,
        states=start_state_distribution,
        approx_0=initial_func_approx,
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        episode_length_tolerance=episode_length_tolerance
    )


def glie_mc_finite_control_learning_rate(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float = 1e-5
) -> Iterator[QValueFunctionApprox[S, A]]:
    initial_qvf_dict: Mapping[Tuple[NonTerminal[S], A], float] = {
        (s, a): 0. for s in fmdp.non_terminal_states for a in fmdp.actions(s)
    }
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return mc.glie_mc_control(
        mdp=fmdp,
        states=Choose(fmdp.non_terminal_states),
        approx_0=Tabular(
            values_map=initial_qvf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        episode_length_tolerance=episode_length_tolerance
    )


def glie_sarsa_learning_rate(
    mdp: MarkovDecisionProcess[S, A],
    start_state_distribution: NTStateDistribution[S],
    initial_func_approx: QValueFunctionApprox[S, A],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    return td.glie_sarsa(
        mdp=mdp,
        states=start_state_distribution,
        approx_0=initial_func_approx,
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        max_episode_length=max_episode_length
    )


def glie_sarsa_finite_learning_rate(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    initial_qvf_dict: Mapping[Tuple[NonTerminal[S], A], float] = {
        (s, a): 0. for s in fmdp.non_terminal_states for a in fmdp.actions(s)
    }
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return td.glie_sarsa(
        mdp=fmdp,
        states=Choose(fmdp.non_terminal_states),
        approx_0=Tabular(
            values_map=initial_qvf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        ϵ_as_func_of_episodes=epsilon_as_func_of_episodes,
        max_episode_length=max_episode_length
    )


def q_learning_learning_rate(
    mdp: MarkovDecisionProcess[S, A],
    start_state_distribution: NTStateDistribution[S],
    initial_func_approx: QValueFunctionApprox[S, A],
    gamma: float,
    epsilon: float,
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    return td.q_learning(
        mdp=mdp,
        policy_from_q=lambda f, m: mc.epsilon_greedy_policy(
            q=f,
            mdp=m,
            ϵ=epsilon
        ),
        states=start_state_distribution,
        approx_0=initial_func_approx,
        γ=gamma,
        max_episode_length=max_episode_length
    )


def q_learning_finite_learning_rate(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon: float,
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    initial_qvf_dict: Mapping[Tuple[NonTerminal[S], A], float] = {
        (s, a): 0. for s in fmdp.non_terminal_states for a in fmdp.actions(s)
    }
    learning_rate_func: Callable[[int], float] = learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
    return td.q_learning(
        mdp=fmdp,
        policy_from_q=lambda f, m: mc.epsilon_greedy_policy(
            q=f,
            mdp=m,
            ϵ=epsilon
        ),
        states=Choose(fmdp.non_terminal_states),
        approx_0=Tabular(
            values_map=initial_qvf_dict,
            count_to_weight_func=learning_rate_func
        ),
        γ=gamma,
        max_episode_length=max_episode_length
    )


def get_vf_and_policy_from_qvf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    qvf: QValueFunctionApprox[S, A]
) -> Tuple[V[S], FiniteDeterministicPolicy[S, A]]:
    opt_vf: V[S] = {
        s: max(qvf((s, a)) for a in mdp.actions(s))
        for s in mdp.non_terminal_states
    }
    opt_policy: FiniteDeterministicPolicy[S, A] = \
        FiniteDeterministicPolicy({
            s.state: qvf.argmax((s, a) for a in mdp.actions(s))[1]
            for s in mdp.non_terminal_states
        })
    return opt_vf, opt_policy


def glie_mc_finite_equal_wts_correctness(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float,
    num_episodes: int
) -> None:
    qvfs: Iterator[QValueFunctionApprox[S, A]] = \
        glie_mc_finite_control_equal_wts(
            fmdp=fmdp,
            gamma=gamma,
            epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
            episode_length_tolerance=episode_length_tolerance
        )
    final_qvf: QValueFunctionApprox[S, A] = \
        iterate.last(itertools.islice(qvfs, num_episodes))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=fmdp,
        qvf=final_qvf
    )

    print(f"GLIE MC Optimal Value Function with {num_episodes:d} episodes")
    pprint(opt_vf)
    print(f"GLIE MC Optimal Policy with {num_episodes:d} episodes")
    print(opt_policy)

    true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)

    print("True Optimal Value Function")
    pprint(true_opt_vf)
    print("True Optimal Policy")
    print(true_opt_policy)


def glie_mc_finite_learning_rate_correctness(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    episode_length_tolerance: float,
    num_episodes: int
) -> None:
    qvfs: Iterator[QValueFunctionApprox[S, A]] = \
        glie_mc_finite_control_learning_rate(
            fmdp=fmdp,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            gamma=gamma,
            epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
            episode_length_tolerance=episode_length_tolerance
        )
    final_qvf: QValueFunctionApprox[S, A] = \
        iterate.last(itertools.islice(qvfs, num_episodes))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=fmdp,
        qvf=final_qvf
    )

    print(f"GLIE MC Optimal Value Function with {num_episodes:d} episodes")
    pprint(opt_vf)
    print(f"GLIE MC Optimal Policy with {num_episodes:d} episodes")
    print(opt_policy)

    true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)

    print("True Optimal Value Function")
    pprint(true_opt_vf)
    print("True Optimal Policy")
    print(true_opt_policy)


def glie_sarsa_finite_learning_rate_correctness(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int,
    num_updates: int,
) -> None:
    qvfs: Iterator[QValueFunctionApprox[S, A]] = \
        glie_sarsa_finite_learning_rate(
            fmdp=fmdp,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            gamma=gamma,
            epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
            max_episode_length=max_episode_length
        )
    final_qvf: QValueFunctionApprox[S, A] = \
        iterate.last(itertools.islice(qvfs, num_updates))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=fmdp,
        qvf=final_qvf
    )

    print(f"GLIE SARSA Optimal Value Function with {num_updates:d} updates")
    pprint(opt_vf)
    print(f"GLIE SARSA Optimal Policy with {num_updates:d} updates")
    print(opt_policy)

    true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)

    print("True Optimal Value Function")
    pprint(true_opt_vf)
    print("True Optimal Policy")
    print(true_opt_policy)


def q_learning_finite_learning_rate_correctness(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    initial_learning_rate: float,
    half_life: float,
    exponent: float,
    gamma: float,
    epsilon: float,
    max_episode_length: int,
    num_updates: int,
) -> None:
    qvfs: Iterator[QValueFunctionApprox[S, A]] = \
        q_learning_finite_learning_rate(
            fmdp=fmdp,
            initial_learning_rate=initial_learning_rate,
            half_life=half_life,
            exponent=exponent,
            gamma=gamma,
            epsilon=epsilon,
            max_episode_length=max_episode_length
        )
    final_qvf: QValueFunctionApprox[S, A] = \
        iterate.last(itertools.islice(qvfs, num_updates))
    opt_vf, opt_policy = get_vf_and_policy_from_qvf(
        mdp=fmdp,
        qvf=final_qvf
    )

    print(f"Q-Learning ptimal Value Function with {num_updates:d} updates")
    pprint(opt_vf)
    print(f"Q-Learning Optimal Policy with {num_updates:d} updates")
    print(opt_policy)

    true_opt_vf, true_opt_policy = value_iteration_result(fmdp, gamma=gamma)

    print("True Optimal Value Function")
    pprint(true_opt_vf)
    print("True Optimal Policy")
    print(true_opt_policy)


def compare_mc_sarsa_ql(
    fmdp: FiniteMarkovDecisionProcess[S, A],
    method_mask: Tuple[bool, bool, bool],
    learning_rates: Sequence[Tuple[float, float, float]],
    gamma: float,
    epsilon_as_func_of_episodes: Callable[[int], float],
    q_learning_epsilon: float,
    mc_episode_length_tol: float,
    num_episodes: int,
    plot_batch: int,
    plot_start: int
) -> None:
    true_vf: V[S] = value_iteration_result(fmdp, gamma)[0]
    states: Sequence[NonTerminal[S]] = fmdp.non_terminal_states
    colors: Sequence[str] = ['b', 'g', 'r', 'k', 'c', 'm', 'y']

    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))

    if method_mask[0]:
        for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
            mc_funcs_it: Iterator[QValueFunctionApprox[S, A]] = \
                glie_mc_finite_control_learning_rate(
                    fmdp=fmdp,
                    initial_learning_rate=init_lr,
                    half_life=half_life,
                    exponent=exponent,
                    gamma=gamma,
                    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
                    episode_length_tolerance=mc_episode_length_tol
                )
            mc_errors = []
            batch_mc_errs = []
            for i, mc_qvf in enumerate(
                    itertools.islice(mc_funcs_it, num_episodes)
            ):
                mc_vf: V[S] = {
                    s: max(mc_qvf((s, a)) for a in fmdp.actions(s))
                    for s in states
                }
                batch_mc_errs.append(sqrt(sum(
                    (mc_vf[s] - true_vf[s]) ** 2 for s in states
                ) / len(states)))
                if i % plot_batch == plot_batch - 1:
                    mc_errors.append(sum(batch_mc_errs) / plot_batch)
                    batch_mc_errs = []
            mc_plot = mc_errors[plot_start:]
            label = f"MC InitRate={init_lr:.3f},HalfLife" + \
                f"={half_life:.0f},Exp={exponent:.1f}"
            plt.plot(
                range(len(mc_plot)),
                mc_plot,
                color=colors[k],
                linestyle='-',
                label=label
            )

    sample_episodes: int = 1000
    uniform_policy: FinitePolicy[S, A] = \
        FinitePolicy(
            {s.state: Choose(fmdp.actions(s)) for s in states}
    )
    fmrp: FiniteMarkovRewardProcess[S] = \
        fmdp.apply_finite_policy(uniform_policy)
    td_episode_length: int = int(round(sum(
        len(list(returns(
            trace=fmrp.simulate_reward(Choose(states)),
            γ=gamma,
            tolerance=mc_episode_length_tol
        ))) for _ in range(sample_episodes)
    ) / sample_episodes))

    if method_mask[1]:
        for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
            sarsa_funcs_it: Iterator[QValueFunctionApprox[S, A]] = \
                glie_sarsa_finite_learning_rate(
                    fmdp=fmdp,
                    initial_learning_rate=init_lr,
                    half_life=half_life,
                    exponent=exponent,
                    gamma=gamma,
                    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
                    max_episode_length=td_episode_length,
                )
            sarsa_errors = []
            transitions_batch = plot_batch * td_episode_length
            batch_sarsa_errs = []

            for i, sarsa_qvf in enumerate(
                itertools.islice(
                    sarsa_funcs_it,
                    num_episodes * td_episode_length
                )
            ):
                sarsa_vf: V[S] = {
                    s: max(sarsa_qvf((s, a)) for a in fmdp.actions(s))
                    for s in states
                }
                batch_sarsa_errs.append(sqrt(sum(
                    (sarsa_vf[s] - true_vf[s]) ** 2 for s in states
                ) / len(states)))
                if i % transitions_batch == transitions_batch - 1:
                    sarsa_errors.append(sum(batch_sarsa_errs) /
                                        transitions_batch)
                    batch_sarsa_errs = []
            sarsa_plot = sarsa_errors[plot_start:]
            label = f"SARSA InitRate={init_lr:.3f},HalfLife" + \
                f"={half_life:.0f},Exp={exponent:.1f}"
            plt.plot(
                range(len(sarsa_plot)),
                sarsa_plot,
                color=colors[k],
                linestyle='--',
                label=label
            )

    if method_mask[2]:
        for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
            ql_funcs_it: Iterator[QValueFunctionApprox[S, A]] = \
                q_learning_finite_learning_rate(
                    fmdp=fmdp,
                    initial_learning_rate=init_lr,
                    half_life=half_life,
                    exponent=exponent,
                    gamma=gamma,
                    epsilon=q_learning_epsilon,
                    max_episode_length=td_episode_length,
                )
            ql_errors = []
            transitions_batch = plot_batch * td_episode_length
            batch_ql_errs = []

            for i, ql_qvf in enumerate(
                itertools.islice(
                    ql_funcs_it,
                    num_episodes * td_episode_length
                )
            ):
                ql_vf: V[S] = {
                    s: max(ql_qvf((s, a)) for a in fmdp.actions(s))
                    for s in states
                }
                batch_ql_errs.append(sqrt(sum(
                    (ql_vf[s] - true_vf[s]) ** 2 for s in states
                ) / len(states)))
                if i % transitions_batch == transitions_batch - 1:
                    ql_errors.append(sum(batch_ql_errs) / transitions_batch)
                    batch_ql_errs = []
            ql_plot = ql_errors[plot_start:]
            label = f"Q-Learning InitRate={init_lr:.3f},HalfLife" + \
                f"={half_life:.0f},Exp={exponent:.1f}"
            plt.plot(
                range(len(ql_plot)),
                ql_plot,
                color=colors[k],
                linestyle=':',
                label=label
            )

    plt.xlabel("Episode Batches", fontsize=20)
    plt.ylabel("Optimal Value Function RMSE", fontsize=20)
    plt.title(
        "RMSE as function of episode batches",
        fontsize=20
    )
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.show()
```

--------------------------------------------------------------------------------

## 📄 chapter11/simple_inventory_mdp_cap.py {#simple-inventory-mdp-cap}

**Titre**: Simple Inventory Mdp Cap

**Description**: Module Simple Inventory Mdp Cap

**Lignes de code**: 86

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Tuple, Callable, Sequence
from rl.chapter11.control_utils import glie_mc_finite_learning_rate_correctness
from rl.chapter11.control_utils import \
    q_learning_finite_learning_rate_correctness
from rl.chapter11.control_utils import \
    glie_sarsa_finite_learning_rate_correctness
from rl.chapter11.control_utils import compare_mc_sarsa_ql
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

si_mdp: SimpleInventoryMDPCap = SimpleInventoryMDPCap(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)

gamma: float = 0.9
mc_episode_length_tol: float = 1e-5
num_episodes = 10000

epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: k ** -0.5
q_learning_epsilon: float = 0.2

td_episode_length: int = 100
initial_learning_rate: float = 0.1
half_life: float = 10000.0
exponent: float = 1.0

glie_mc_finite_learning_rate_correctness(
    fmdp=si_mdp,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    gamma=gamma,
    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
    episode_length_tolerance=mc_episode_length_tol,
    num_episodes=num_episodes
)

glie_sarsa_finite_learning_rate_correctness(
    fmdp=si_mdp,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    gamma=gamma,
    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
    max_episode_length=td_episode_length,
    num_updates=num_episodes * td_episode_length
)

q_learning_finite_learning_rate_correctness(
    fmdp=si_mdp,
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent,
    gamma=gamma,
    epsilon=q_learning_epsilon,
    max_episode_length=td_episode_length,
    num_updates=num_episodes * td_episode_length
)

num_episodes = 500
plot_batch: int = 10
plot_start: int = 0
learning_rates: Sequence[Tuple[float, float, float]] = \
    [(0.05, 1000000, 0.5)]

compare_mc_sarsa_ql(
    fmdp=si_mdp,
    method_mask=[True, True, False],
    learning_rates=learning_rates,
    gamma=gamma,
    epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
    q_learning_epsilon=q_learning_epsilon,
    mc_episode_length_tol=mc_episode_length_tol,
    num_episodes=num_episodes,
    plot_batch=plot_batch,
    plot_start=plot_start
)
```

--------------------------------------------------------------------------------

## 📄 chapter11/windy_grid.py {#windy-grid}

**Titre**: Windy Grid

**Description**: Module Windy Grid

**Lignes de code**: 271

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Tuple, Callable, Sequence, Set, Mapping, Dict, Iterator
from rl.chapter11.control_utils import glie_sarsa_finite_learning_rate
from rl.chapter11.control_utils import q_learning_finite_learning_rate
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from dataclasses import dataclass
from rl.distribution import Categorical
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.dynamic_programming import value_iteration_result, V
import itertools
import rl.iterate as iterate

'''
Cell specifies (row, column) coordinate
'''
Cell = Tuple[int, int]
CellSet = Set[Cell]
Move = Tuple[int, int]
'''
WindSpec specifies a random vertical wind for each column.
Each random vertical wind is specified by a (p1, p2) pair
where p1 specifies probability of Downward Wind (could take you
one step lower in row coordinate unless prevented by a block or
boundary) and p2 specifies probability of Upward Wind (could take
you onw step higher in column coordinate unless prevented by a
block or boundary). If one bumps against a block or boundary, one
incurs a bump cost and doesn't move. The remaining probability
1- p1 - p2 corresponds to No Wind.
'''
WindSpec = Sequence[Tuple[float, float]]

possible_moves: Mapping[Move, str] = {
    (-1, 0): 'D',
    (1, 0): 'U',
    (0, -1): 'L',
    (0, 1): 'R'
}


@dataclass(frozen=True)
class WindyGrid:

    rows: int  # number of grid rows
    columns: int  # number of grid columns
    blocks: CellSet  # coordinates of block cells
    terminals: CellSet  # coordinates of goal cells
    wind: WindSpec  # spec of vertical random wind for the columns
    bump_cost: float  # cost of bumping against block or boundary

    def validate_spec(self) -> bool:
        b1 = self.rows >= 2
        b2 = self.columns >= 2
        b3 = all(0 <= r < self.rows and 0 <= c < self.columns
                 for r, c in self.blocks)
        b4 = len(self.terminals) >= 1
        b5 = all(0 <= r < self.rows and 0 <= c < self.columns and
                 (r, c) not in self.blocks for r, c in self.terminals)
        b6 = len(self.wind) == self.columns
        b7 = all(0. <= p1 <= 1. and 0. <= p2 <= 1. and p1 + p2 <= 1.
                 for p1, p2 in self.wind)
        b8 = self.bump_cost > 0.
        return all([b1, b2, b3, b4, b5, b6, b7, b8])

    def print_wind_and_bumps(self) -> None:
        for i, (d, u) in enumerate(self.wind):
            print(f"Column {i:d}: Down Prob = {d:.2f}, Up Prob = {u:.2f}")
        print(f"Bump Cost = {self.bump_cost:.2f}")
        print()

    @staticmethod
    def add_move_to_cell(cell: Cell, move: Move) -> Cell:
        return cell[0] + move[0], cell[1] + move[1]

    def is_valid_state(self, cell: Cell) -> bool:
        '''
        checks if a cell is a valid state of the MDP
        '''
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.columns \
            and cell not in self.blocks

    def get_all_nt_states(self) -> CellSet:
        '''
        returns all the non-terminal states
        '''
        return {(i, j) for i in range(self.rows) for j in range(self.columns)
                if (i, j) not in set.union(self.blocks, self.terminals)}

    def get_actions_and_next_states(self, nt_state: Cell) \
            -> Set[Tuple[Move, Cell]]:
        '''
        given a non-terminal state, returns the set of all possible
        (action, next_state) pairs
        '''
        temp: Set[Tuple[Move, Cell]] = {(a, WindyGrid.add_move_to_cell(
            nt_state,
            a
        )) for a in possible_moves}
        return {(a, s) for a, s in temp if self.is_valid_state(s)}

    def get_transition_probabilities(self, nt_state: Cell) \
            -> Mapping[Move, Categorical[Tuple[Cell, float]]]:
        '''
        given a non-terminal state, return a dictionary whose
        keys are the valid actions (moves) from the given state
        and the corresponding values are the associated probabilities
        (following that move) of the (next_state, reward) pairs.
        The probabilities are determined from the wind probabilities
        of the column one is in after the move. Note that if one moves
        to a goal cell (terminal state), then one ends up in that
        goal cell with 100% probability (i.e., no wind exposure in a
        goal cell).
        '''
        d: Dict[Move, Categorical[Tuple[Cell, float]]] = {}
        for a, (r, c) in self.get_actions_and_next_states(nt_state):
            if (r, c) in self.terminals:
                d[a] = Categorical({((r, c), -1.): 1.})
            else:
                down_prob, up_prob = self.wind[c]
                stay_prob: float = 1. - down_prob - up_prob
                d1: Dict[Tuple[Cell, float], float] = \
                    {((r, c), -1.): stay_prob}
                if self.is_valid_state((r - 1, c)):
                    d1[((r - 1, c), -1.)] = down_prob
                if self.is_valid_state((r + 1, c)):
                    d1[((r + 1, c), -1.)] = up_prob
                d1[((r, c), -1. - self.bump_cost)] = \
                    down_prob * (1 - self.is_valid_state((r - 1, c))) + \
                    up_prob * (1 - self.is_valid_state((r + 1, c)))
                d[a] = Categorical(d1)
        return d

    def get_finite_mdp(self) -> FiniteMarkovDecisionProcess[Cell, Move]:
        '''
        returns the FiniteMarkovDecision object for this windy grid problem
        '''
        return FiniteMarkovDecisionProcess(
            {s: self.get_transition_probabilities(s) for s in
             self.get_all_nt_states()}
        )

    def get_vi_vf_and_policy(self) -> \
            Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        '''
        Performs the Value Iteration DP algorithm returning the
        Optimal Value Function (as a V[Cell]) and the Optimal Policy
        (as a FiniteDeterministicPolicy[Cell, Move])
        '''
        return value_iteration_result(self.get_finite_mdp(), gamma=1.)

    def get_glie_sarsa_vf_and_policy(
        self,
        epsilon_as_func_of_episodes: Callable[[int], float],
        learning_rate: float,
        num_updates: int
    ) -> Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        qvfs: Iterator[QValueFunctionApprox[Cell, Move]] = \
            glie_sarsa_finite_learning_rate(
                fmdp=self.get_finite_mdp(),
                initial_learning_rate=learning_rate,
                half_life=1e8,
                exponent=1.0,
                gamma=1.0,
                epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
                max_episode_length=int(1e8)
            )
        final_qvf: QValueFunctionApprox[Cell, Move] = \
            iterate.last(itertools.islice(qvfs, num_updates))
        return get_vf_and_policy_from_qvf(
            mdp=self.get_finite_mdp(),
            qvf=final_qvf
        )

    def get_q_learning_vf_and_policy(
        self,
        epsilon: float,
        learning_rate: float,
        num_updates: int
    ) -> Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        qvfs: Iterator[QValueFunctionApprox[Cell, Move]] = \
            q_learning_finite_learning_rate(
                fmdp=self.get_finite_mdp(),
                initial_learning_rate=learning_rate,
                half_life=1e8,
                exponent=1.0,
                gamma=1.0,
                epsilon=epsilon,
                max_episode_length=int(1e8)
            )
        final_qvf: QValueFunctionApprox[Cell, Move] = \
            iterate.last(itertools.islice(qvfs, num_updates))
        return get_vf_and_policy_from_qvf(
            mdp=self.get_finite_mdp(),
            qvf=final_qvf
        )

    def print_vf_and_policy(
        self,
        vf_dict: V[Cell],
        policy: FiniteDeterministicPolicy[Cell, Move]
    ) -> None:
        display = "%5.2f"
        display1 = "%5d"
        vf_full_dict = {
            **{s.state: display % -v for s, v in vf_dict.items()},
            **{s: display % 0.0 for s in self.terminals},
            **{s: 'X' * 5 for s in self.blocks}
        }
        print("   " + " ".join([display1 % j for j in range(self.columns)]))
        for i in range(self.rows - 1, -1, -1):
            print("%2d " % i + " ".join(vf_full_dict[(i, j)]
                                        for j in range(self.columns)))
        print()
        pol_full_dict = {
            **{s: possible_moves[policy.action_for[s]]
               for s in self.get_all_nt_states()},
            **{s: 'T' for s in self.terminals},
            **{s: 'X' for s in self.blocks}
        }
        print("   " + " ".join(["%2d" % j for j in range(self.columns)]))
        for i in range(self.rows - 1, -1, -1):
            print("%2d  " % i + "  ".join(pol_full_dict[(i, j)]
                                          for j in range(self.columns)))
        print()


if __name__ == '__main__':
    wg = WindyGrid(
        rows=5,
        columns=5,
        blocks={(0, 1), (0, 2), (0, 4), (2, 3), (3, 0), (4, 0)},
        terminals={(3, 4)},
        wind=[(0., 0.9), (0.0, 0.8), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0)],
        bump_cost=4.0
    )
    valid = wg.validate_spec()
    if valid:
        wg.print_wind_and_bumps()
        vi_vf_dict, vi_policy = wg.get_vi_vf_and_policy()
        print("Value Iteration\n")
        wg.print_vf_and_policy(
            vf_dict=vi_vf_dict,
            policy=vi_policy
        )
        epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: 1. / k
        learning_rate: float = 0.03
        num_updates: int = 100000
        sarsa_vf_dict, sarsa_policy = wg.get_glie_sarsa_vf_and_policy(
            epsilon_as_func_of_episodes=epsilon_as_func_of_episodes,
            learning_rate=learning_rate,
            num_updates=num_updates
        )
        print("SARSA\n")
        wg.print_vf_and_policy(
            vf_dict=sarsa_vf_dict,
            policy=sarsa_policy
        )
        epsilon: float = 0.2
        ql_vf_dict, ql_policy = wg.get_q_learning_vf_and_policy(
            epsilon=epsilon,
            learning_rate=learning_rate,
            num_updates=num_updates
        )
        print("Q-Learning\n")
        wg.print_vf_and_policy(
            vf_dict=ql_vf_dict,
            policy=ql_policy
        )
    else:
        print("Invalid Spec of Windy Grid")
```

--------------------------------------------------------------------------------

## 📄 chapter11/windy_grid_convergence.py {#windy-grid-convergence}

**Titre**: Windy Grid Convergence

**Description**: Module Windy Grid Convergence

**Lignes de code**: 28

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from rl.chapter11.windy_grid import WindyGrid, Cell, Move
from rl.chapter11.control_utils import compare_mc_sarsa_ql
from rl.markov_decision_process import FiniteMarkovDecisionProcess

wg = WindyGrid(
    rows=5,
    columns=5,
    blocks={(0, 1), (0, 2), (0, 4), (2, 3), (3, 0), (4, 0)},
    terminals={(3, 4)},
    wind=[(0., 0.9), (0.0, 0.8), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0)],
    bump_cost=100000.0
)
valid = wg.validate_spec()
if valid:
    fmdp: FiniteMarkovDecisionProcess[Cell, Move] = wg.get_finite_mdp()
    compare_mc_sarsa_ql(
        fmdp=fmdp,
        method_mask=[False, True, True],
        learning_rates=[(0.03, 1e8, 1.0)],
        gamma=1.,
        epsilon_as_func_of_episodes=lambda k: 1. / k,
        q_learning_epsilon=0.2,
        mc_episode_length_tol=1e-5,
        num_episodes=400,
        plot_batch=10,
        plot_start=0
    )
```

--------------------------------------------------------------------------------

## 📄 chapter12/laguerre.py {#laguerre}

**Titre**: Laguerre

**Description**: Module Laguerre

**Lignes de code**: 60

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from numpy.polynomial.laguerre import lagval
from typing import Sequence, List, Callable, TypeVar, Tuple
from rl.markov_process import NonTerminal
import numpy as np

S = TypeVar('S')
A = TypeVar('A')


def laguerre_polynomials(n: int) -> Sequence[Callable[[float], float]]:
    ret: List[Callable[[float], float]] = []
    ident: np.ndarray = np.eye(n)
    for i in range(n):
        def laguerre_func(x: float, i=i) -> float:
            return lagval(x, ident[i])
        ret.append(laguerre_func)
    return ret


def laguerre_state_features(n: int) -> \
        Sequence[Callable[[NonTerminal[S]], float]]:
    ret: List[Callable[[NonTerminal[S]], float]] = []
    ident: np.ndarray = np.eye(n)
    for i in range(n):
        def laguerre_ff(x: NonTerminal[S], i=i) -> float:
            return lagval(float(x.state), ident[i])
        ret.append(laguerre_ff)
    return ret


def laguerre_state_action_features(
    num_state_features: int,
    num_action_features: int
) -> Sequence[Callable[[Tuple[NonTerminal[S], A]], float]]:
    ret: List[Callable[[Tuple[NonTerminal[S], A]], float]] = []
    states_ident: np.ndarray = np.eye(num_state_features)
    actions_ident: np.ndarray = np.eye(num_state_features)
    for i in range(num_state_features):
        def laguerre_state_ff(x: Tuple[NonTerminal[S], A], i=i) -> float:
            return lagval(float(x[0].state), states_ident[i])
        ret.append(laguerre_state_ff)
    for j in range(num_action_features):
        def laguerre_action_ff(x: Tuple[NonTerminal[S], A], j=j) -> float:
            return lagval(float(x[1]), actions_ident[j])
        ret.append(laguerre_action_ff)
    return ret


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num_polynomials = 4
    lps: Sequence[Callable[[float], float]] = \
        laguerre_polynomials(num_polynomials)
    x_vals: np.ndarray = np.arange(-2, 2, 0.1)
    for i in range(num_polynomials):
        plt.plot(x_vals, lps[i](x_vals), label="Laguerre %d" % i)
    plt.grid()
    plt.legend()
    plt.show()
```

--------------------------------------------------------------------------------

## 📄 chapter12/random_walk_lstd.py {#random-walk-lstd}

**Titre**: Temporal Difference Learning

**Description**: Algorithmes TD pour apprentissage online

**Lignes de code**: 87

**Concepts clés**:
- td_prediction - TD(0): V(s) ← V(s) + α[r + γV(s') - V(s)]
- sarsa - SARSA (on-policy): Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- q_learning - Q-Learning (off-policy): Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]
- Bootstrap: apprend d'estimations, pas besoin d'épisode complet

**🎯 Utilisation HelixOne**: Apprentissage online des stratégies

### Code Source Complet

```python
from rl.chapter10.random_walk_mrp import RandomWalkMRP
from rl.chapter12.laguerre import laguerre_state_features
from rl.td import td_prediction, least_squares_td
from rl.function_approx import LinearFunctionApprox, Tabular, \
    learning_rate_schedule
from rl.approximate_dynamic_programming import NTStateDistribution
import numpy as np
from typing import Iterable, Sequence, Callable
from rl.markov_process import TransitionStep, NonTerminal
from rl.distribution import Choose
import itertools
from rl.gen_utils.plot_funcs import plot_list_of_curves
import rl.iterate as iterate


this_barrier: int = 20
this_p: float = 0.55
random_walk: RandomWalkMRP = RandomWalkMRP(
    barrier=this_barrier,
    p=this_p
)

gamma = 1.0
true_vf: np.ndarray = random_walk.get_value_function_vec(gamma=gamma)

num_transitions: int = 10000

nt_states: Sequence[NonTerminal[int]] = random_walk.non_terminal_states
start_distribution: NTStateDistribution[int] = Choose(nt_states)
traces: Iterable[Iterable[TransitionStep[int]]] = \
    random_walk.reward_traces(start_distribution)
transitions: Iterable[TransitionStep[int]] = \
    itertools.chain.from_iterable(traces)

td_transitions: Iterable[TransitionStep[int]] = \
    itertools.islice(transitions, num_transitions)

initial_learning_rate: float = 0.5
half_life: float = 1000
exponent: float = 0.5
approx0: Tabular[NonTerminal[int]] = Tabular(
    count_to_weight_func=learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
)

td_func: Tabular[NonTerminal[int]] = \
    iterate.last(itertools.islice(
        td_prediction(
            transitions=td_transitions,
            approx_0=approx0,
            γ=gamma
        ),
        num_transitions
    ))
td_vf: np.ndarray = td_func.evaluate(nt_states)

num_polynomials: int = 5
features: Sequence[Callable[[NonTerminal[int]], float]] = \
    laguerre_state_features(num_polynomials)
lstd_transitions: Iterable[TransitionStep[int]] = \
    itertools.islice(transitions, num_transitions)
epsilon: float = 1e-4

lstd_func: LinearFunctionApprox[NonTerminal[int]] = \
    least_squares_td(
        transitions=lstd_transitions,
        feature_functions=features,
        γ=gamma,
        ε=epsilon
    )
lstd_vf: np.ndarray = lstd_func.evaluate(nt_states)

x_vals: Sequence[int] = [s.state for s in nt_states]

plot_list_of_curves(
    [x_vals, x_vals, x_vals],
    [true_vf, td_vf, lstd_vf],
    ["b-", "g.-", "r--"],
    ["True Value Function", "Tabular TD Value Function", "LSTD Value Function"],
    x_label="States",
    y_label="Value Function",
    title="Tabular TD and LSTD versus True Value Function"
)
```

--------------------------------------------------------------------------------

## 📄 chapter12/vampire.py {#vampire}

**Titre**: Vampire

**Description**: Module Vampire

**Lignes de code**: 140

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Mapping, Tuple, Iterable, Iterator, Sequence, Callable, \
    List
from rl.markov_process import NonTerminal
from rl.markov_decision_process import FiniteMarkovDecisionProcess, \
    TransitionStep
from rl.distribution import Categorical, Choose
from rl.function_approx import LinearFunctionApprox
from rl.policy import DeterministicPolicy, FiniteDeterministicPolicy
from rl.dynamic_programming import value_iteration_result, V
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.td import least_squares_policy_iteration
from numpy.polynomial.laguerre import lagval
import itertools
import rl.iterate as iterate
import numpy as np


class VampireMDP(FiniteMarkovDecisionProcess[int, int]):

    initial_villagers: int

    def __init__(self, initial_villagers: int):
        self.initial_villagers = initial_villagers
        super().__init__(self.mdp_map())

    def mdp_map(self) -> \
            Mapping[int, Mapping[int, Categorical[Tuple[int, float]]]]:
        return {s: {a: Categorical(
            {(s - a - 1, 0.): 1 - a / s, (0, float(s - a)): a / s}
        ) for a in range(s)} for s in range(1, self.initial_villagers + 1)}

    def vi_vf_and_policy(self) -> \
            Tuple[V[int], FiniteDeterministicPolicy[int, int]]:
        return value_iteration_result(self, 1.0)

    def lspi_features(
        self,
        factor1_features: int,
        factor2_features: int
    ) -> Sequence[Callable[[Tuple[NonTerminal[int], int]], float]]:
        ret: List[Callable[[Tuple[NonTerminal[int], int]], float]] = []
        ident1: np.ndarray = np.eye(factor1_features)
        ident2: np.ndarray = np.eye(factor2_features)
        for i in range(factor1_features):
            def factor1_ff(x: Tuple[NonTerminal[int], int], i=i) -> float:
                return lagval(
                    float((x[0].state - x[1]) ** 2 / x[0].state),
                    ident1[i]
                )
            ret.append(factor1_ff)
        for j in range(factor2_features):
            def factor2_ff(x: Tuple[NonTerminal[int], int], j=j) -> float:
                return lagval(
                    float((x[0].state - x[1]) * x[1] / x[0].state),
                    ident2[j]
                )
            ret.append(factor2_ff)
        return ret

    def lspi_transitions(self) -> Iterator[TransitionStep[int, int]]:
        states_distribution: Choose[NonTerminal[int]] = \
            Choose(self.non_terminal_states)
        while True:
            state: NonTerminal[int] = states_distribution.sample()
            action: int = Choose(range(state.state)). sample()
            next_state, reward = self.step(state, action).sample()
            transition: TransitionStep[int, int] = TransitionStep(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward
            )
            yield transition

    def lspi_vf_and_policy(self) -> \
            Tuple[V[int], FiniteDeterministicPolicy[int, int]]:
        transitions: Iterable[TransitionStep[int, int]] = itertools.islice(
            self.lspi_transitions(),
            20000
        )
        qvf_iter: Iterator[LinearFunctionApprox[Tuple[
            NonTerminal[int], int]]] = least_squares_policy_iteration(
                transitions=transitions,
                actions=self.actions,
                feature_functions=self.lspi_features(4, 4),
                initial_target_policy=DeterministicPolicy(
                    lambda s: int(s / 2)
                ),
                γ=1.0,
                ε=1e-5
            )
        qvf: LinearFunctionApprox[Tuple[NonTerminal[int], int]] = \
            iterate.last(
                itertools.islice(
                    qvf_iter,
                    20
                )
            )
        return get_vf_and_policy_from_qvf(self, qvf)


if __name__ == '__main__':
    from pprint import pprint
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    from rl.markov_process import NonTerminal

    villagers: int = 20
    vampire_mdp: VampireMDP = VampireMDP(villagers)
    true_vf, true_policy = vampire_mdp.vi_vf_and_policy()
    pprint(true_vf)
    print(true_policy)
    lspi_vf, lspi_policy = vampire_mdp.lspi_vf_and_policy()
    pprint(lspi_vf)
    print(lspi_policy)

    states = range(1, villagers + 1)
    true_vf_vals = [true_vf[NonTerminal(s)] for s in states]
    lspi_vf_vals = [lspi_vf[NonTerminal(s)] for s in states]
    true_policy_actions = [true_policy.action_for[s] for s in states]
    lspi_policy_actions = [lspi_policy.action_for[s] for s in states]

    plot_list_of_curves(
        [states, states],
        [true_vf_vals, lspi_vf_vals],
        ["r-", "b--"],
        ["True Optimal VF", "LSPI-Estimated Optimal VF"],
        x_label="States",
        y_label="Optimal Values",
        title="True Optimal VF versus LSPI-Estimated Optimal VF"
    )
    plot_list_of_curves(
        [states, states],
        [true_policy_actions, lspi_policy_actions],
        ["r-", "b--"],
        ["True Optimal Policy", "LSPI-Estimated Optimal Policy"],
        x_label="States",
        y_label="Optimal Actions",
        title="True Optimal Policy versus LSPI-Estimated Optimal Policy"
    )
```

--------------------------------------------------------------------------------

## 📄 chapter14/epsilon_greedy.py {#epsilon-greedy}

**Titre**: Epsilon Greedy

**Description**: Module Epsilon Greedy

**Lignes de code**: 122

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import List, Callable, Tuple, Sequence
from rl.distribution import Distribution, Gaussian, Range, Bernoulli
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty


class EpsilonGreedy(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        epsilon: float,
        epsilon_half_life: float = 1e8,
        count_init: int = 0,
        mean_init: float = 0.,
    ) -> None:
        if epsilon < 0 or epsilon > 1 or \
                epsilon_half_life <= 1 or count_init < 0:
            raise ValueError

        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.epsilon_func: Callable[[int], float] = \
            EpsilonGreedy.get_epsilon_decay_func(epsilon, epsilon_half_life)
        self.count_init: int = count_init
        self.mean_init: float = mean_init

    @staticmethod
    def get_epsilon_decay_func(
        epsilon,
        epsilon_half_life
    ) -> Callable[[int], float]:

        def epsilon_decay(
            t: int,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life
        ) -> float:
            return epsilon * 2 ** -(t / epsilon_half_life)

        return epsilon_decay

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        counts: List[int] = [self.count_init] * self.num_arms
        means: List[float] = [self.mean_init] * self.num_arms
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.time_steps):
            max_action: int = max(enumerate(means), key=itemgetter(1))[0]
            epsl: float = self.epsilon_func(i)
            action: int = max_action if Bernoulli(1 - epsl).sample() else \
                Range(self.num_arms).sample()
            reward: float = self.arm_distributions[action].sample()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    means_vars_data = [(9., 5.), (10., 2.), (0., 4.),
                       (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(means_vars_data, key=itemgetter(0))[0]
    steps = 1000
    episodes = 500
    eps = 0.12
    eps_hl = 150
    ci = 0
    mi = 0.

    arm_distrs = [Gaussian(μ=m, σ=s) for m, s in means_vars_data]
    decay_eg = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=ci,
        mean_init=mi
    )
    decay_eg_cum_regret = decay_eg.get_expected_cum_regret(mu_star)

    eg = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eg_cum_regret = eg.get_expected_cum_regret(mu_star)

    greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.0,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    greedy_cum_regret = greedy.get_expected_cum_regret(mu_star)

    plot_list_of_curves(
        [range(1, steps + 1), range(1, steps + 1), range(1, steps + 1)],
        [greedy_cum_regret, eg_cum_regret, decay_eg_cum_regret],
        ["r-", "b--", "g-."],
        ["Greedy", "$\epsilon$-Greedy", "Decaying $\epsilon$-Greedy"],
        x_label="Time Steps",
        y_label="Expected Total Regret",
        title="Total Regret"
    )
```

--------------------------------------------------------------------------------

## 📄 chapter14/gradient_bandits.py {#gradient-bandits}

**Titre**: Gradient Bandits

**Description**: Module Gradient Bandits

**Lignes de code**: 76

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Tuple, List
from rl.distribution import Distribution, Gaussian, Categorical
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty, exp


class GradientBandits(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        learning_rate: float,
        learning_rate_decay: float
    ) -> None:
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.learning_rate: float = learning_rate
        self.learning_rate_decay: float = learning_rate_decay

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        scores: List[float] = [0.] * self.num_arms
        avg_reward: float = 0.

        for i in range(self.time_steps):
            max_score: float = max(scores)
            exp_scores: Sequence[float] = [exp(s - max_score) for s in scores]
            sum_exp_scores = sum(exp_scores)
            probs: Sequence[float] = [s / sum_exp_scores for s in exp_scores]
            action: int = Categorical(
                {i: p for i, p in enumerate(probs)}
            ).sample()
            reward: float = self.arm_distributions[action].sample()
            avg_reward += (reward - avg_reward) / (i + 1)
            step_size: float = self.learning_rate *\
                (i / self.learning_rate_decay + 1) ** -0.5
            for j in range(self.num_arms):
                scores[j] += step_size * (reward - avg_reward) *\
                             ((1 if j == action else 0) - probs[j])

            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    means_vars_data = [(9., 5.), (10., 2.), (0., 4.),
                       (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(means_vars_data, key=itemgetter(0))[0]
    steps = 1000
    episodes = 500
    lr = 0.1
    lr_decay = 20.0

    arm_distrs = [Gaussian(μ=m, σ=s) for m, s in means_vars_data]
    gb = GradientBandits(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )
    # exp_cum_regret = gb.get_expected_cum_regret(mu_star)
    # print(exp_cum_regret)
    # exp_act_count = gb.get_expected_action_counts()
    # print(exp_act_count)

    gb.plot_exp_cum_regret_curve(mu_star)
```

--------------------------------------------------------------------------------

## 📄 chapter14/mab_base.py {#mab-base}

**Titre**: Mab Base

**Description**: Module Mab Base

**Lignes de code**: 71

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Tuple
from abc import ABC, abstractmethod
from rl.distribution import Distribution
from numpy import ndarray, mean, vstack, cumsum, full, bincount


class MABBase(ABC):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int
    ) -> None:
        self.arm_distributions: Sequence[Distribution[float]] = \
            arm_distributions
        self.num_arms: int = len(arm_distributions)
        self.time_steps: int = time_steps
        self.num_episodes: int = num_episodes

    @abstractmethod
    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        pass

    def get_all_rewards_actions(self) -> Sequence[Tuple[ndarray, ndarray]]:
        return [self.get_episode_rewards_actions()
                for _ in range(self.num_episodes)]

    def get_rewards_matrix(self) -> ndarray:
        return vstack([x for x, _ in self.get_all_rewards_actions()])

    def get_actions_matrix(self) -> ndarray:
        return vstack([y for _, y in self.get_all_rewards_actions()])

    def get_expected_rewards(self) -> ndarray:
        return mean(self.get_rewards_matrix(), axis=0)

    def get_expected_cum_rewards(self) -> ndarray:
        return cumsum(self.get_expected_rewards())

    def get_expected_regret(self, best_mean) -> ndarray:
        return full(self.time_steps, best_mean) - self.get_expected_rewards()

    def get_expected_cum_regret(self, best_mean) -> ndarray:
        return cumsum(self.get_expected_regret(best_mean))

    def get_action_counts(self) -> ndarray:
        return vstack([bincount(ep, minlength=self.num_arms)
                       for ep in self.get_actions_matrix()])

    def get_expected_action_counts(self) -> ndarray:
        return mean(self.get_action_counts(), axis=0)

    def plot_exp_cum_regret_curve(self, best_mean) -> None:
        import matplotlib.pyplot as plt
        x_vals = range(1, self.time_steps + 1)
        plt.plot(
            self.get_expected_cum_regret(best_mean),
            "b",
            label="Expected Total Regret"
        )
        plt.xlabel("Time Steps", fontsize=20)
        plt.ylabel("Expected Total Regret", fontsize=20)
        plt.title("Total Regret Curve", fontsize=25)
        plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
        plt.ylim(ymin=0.0)
        # plt.xticks(x_vals)
        plt.grid(True)
        # plt.legend(loc='upper left')
        plt.show()
```

--------------------------------------------------------------------------------

## 📄 chapter14/mab_graphs_gen.py {#mab-graphs-gen}

**Titre**: Mab Graphs Gen

**Description**: Module Mab Graphs Gen

**Lignes de code**: 72

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
import numpy as np


def graph_regret_curve() -> None:
    import matplotlib.pyplot as plt
    x_vals = range(1, 71)
    plt.plot(x_vals, [3*x for x in x_vals], "r", label="Greedy")
    plt.plot(x_vals, [2*x for x in x_vals], "b", label="$\epsilon$-Greedy")
    plt.plot(
        x_vals,
        [20 * np.log(x) for x in x_vals],
        "g",
        label="Decaying $\epsilon$-Greedy"
    )
    plt.xlabel("Time Steps", fontsize=25)
    plt.ylabel("Total Regret", fontsize=25)
    plt.title("Total Regret Curves", fontsize=25)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    # plt.xticks(x_vals)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()


def get_pdf(x: float, mu: float, sigma: float) -> float:
    return np.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma)) / \
        (np.sqrt(2 * np.pi) * sigma)


def graph_qestimate_pdfs() -> None:
    import matplotlib.pyplot as plt
    x_vals = np.arange(-2., 6., 0.01)
    mu_b = 1.5
    sigma_b = 2.0
    mu_r = 2.0
    sigma_r = 0.8
    mu_g = 2.5
    sigma_g = 0.3
    plt.plot(
        x_vals,
        [get_pdf(x, mu_b, sigma_b) for x in x_vals],
        "b-",
        label="$Q(a_1)$"
    )
    plt.plot(
        x_vals,
        [get_pdf(x, mu_r, sigma_r) for x in x_vals],
        "r--",
        label="$Q(a_2)$"
    )
    plt.plot(
        x_vals,
        [get_pdf(x, mu_g, sigma_g) for x in x_vals],
        "g-.",
        label="$Q(a_3)$"
    )
    plt.xlabel("Q", fontsize=25)
    plt.ylabel("Prob(Q)", fontsize=25)
    # plt.title("Total Regret Curves", fontsize=25)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    # plt.xticks(x_vals)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()


if __name__ == '__main__':
    # graph_regret_curve()
    graph_qestimate_pdfs()
```

--------------------------------------------------------------------------------

## 📄 chapter14/plot_mab_graphs.py {#plot-mab-graphs}

**Titre**: Plot Mab Graphs

**Description**: Module Plot Mab Graphs

**Lignes de code**: 277

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from operator import itemgetter
from rl.distribution import Gaussian, Bernoulli
from rl.chapter14.epsilon_greedy import EpsilonGreedy
from rl.chapter14.ucb1 import UCB1
from rl.chapter14.ts_gaussian import ThompsonSamplingGaussian
from rl.chapter14.ts_bernoulli import ThompsonSamplingBernoulli
from rl.chapter14.gradient_bandits import GradientBandits
from numpy import arange
import matplotlib.pyplot as plt


def plot_gaussian_algorithms() -> None:
    means_vars_data = [
        (0., 10.),
        (2., 20.),
        (4., 1.),
        (6., 8.),
        (8., 4.),
        (9., 6.),
        (10., 4.)]
    mu_star = max(means_vars_data, key=itemgetter(0))[0]

    steps = 500
    episodes = 500

    eps = 0.3
    eps_hl = 400

    ci = 5
    mi = mu_star * 3.

    ts_mi = 0.
    ts_si = 10.

    lr = 0.1
    lr_decay = 20.

    arm_distrs = [Gaussian(μ=m, σ=s) for m, s in means_vars_data]

    greedy_opt_init = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eps_greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=0,
        mean_init=0.
    )
    decay_eps_greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=0,
        mean_init=0.
    )
    ts = ThompsonSamplingGaussian(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        init_mean=ts_mi,
        init_stdev=ts_si
    )
    grad_bandits = GradientBandits(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )

    plot_colors = ['k', 'y', 'k--', 'y--', 'b-.']
    labels = [
        'Greedy, Optimistic Initialization',
        '$\epsilon$-Greedy',
        'Decaying $\epsilon$-Greedy',
        'Thompson Sampling',
        'Gradient Bandit'
    ]

    exp_cum_regrets = [
        greedy_opt_init.get_expected_cum_regret(mu_star),
        eps_greedy.get_expected_cum_regret(mu_star),
        decay_eps_greedy.get_expected_cum_regret(mu_star),
        ts.get_expected_cum_regret(mu_star),
        grad_bandits.get_expected_cum_regret(mu_star)
    ]

    x_vals = range(1, steps + 1)
    for i in range(len(exp_cum_regrets)):
        plt.plot(x_vals, exp_cum_regrets[i], plot_colors[i], label=labels[i])
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Expected Total Regret", fontsize=20)
    plt.title("Total Regret Curves", fontsize=25)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

    exp_act_counts = [
        greedy_opt_init.get_expected_action_counts(),
        eps_greedy.get_expected_action_counts(),
        decay_eps_greedy.get_expected_action_counts(),
        ts.get_expected_action_counts(),
        grad_bandits.get_expected_action_counts()
    ]
    index = arange(len(means_vars_data))
    spacing = 0.4
    width = (1 - spacing) / len(exp_act_counts)

    hist_plot_colors = ['r', 'b', 'g', 'k', 'y']
    for i in range(len(exp_act_counts)):
        plt.bar(
            index - (1 - spacing) / 2 + (i - 1.5) * width,
            exp_act_counts[i],
            width,
            color=hist_plot_colors[i],
            label=labels[i]
        )
    plt.xlabel("Arms", fontsize=20)
    plt.ylabel("Expected Counts of Arms", fontsize=20)
    plt.title("Arms Counts Plot", fontsize=25)
    plt.xticks(
        index - 0.3,
        ["$\mu$=%.1f,$\sigma$=%.1f" % (m, s) for m, s in means_vars_data]
    )
    plt.legend(loc='upper left', fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_bernoulli_algorithms() -> None:
    probs_data = [0.1, 0.2, 0.4, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9]
    mu_star = max(probs_data)

    steps = 500
    episodes = 500

    eps = 0.3
    eps_hl = 400

    ci = 5
    mi = mu_star * 3.

    ucb_alpha = 4.0

    lr = 0.5
    lr_decay = 20.

    arm_distrs = [Bernoulli(p) for p in probs_data]

    greedy_opt_init = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eps_greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=0,
        mean_init=0.
    )
    decay_eps_greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=0,
        mean_init=0.
    )
    ucb1 = UCB1(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        bounds_range=1.0,
        alpha=ucb_alpha
    )
    ts = ThompsonSamplingBernoulli(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes
    )
    grad_bandits = GradientBandits(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )

    plot_colors = ['k', 'y', 'k--', 'y--', 'r-.', 'c-.']
    labels = [
        'Greedy, Optimistic Initialization',
        '$\epsilon$-Greedy',
        'Decaying $\epsilon$-Greedy',
        'UCB1',
        'Thompson Sampling',
        'Gradient Bandit'
    ]

    exp_cum_regrets = [
        greedy_opt_init.get_expected_cum_regret(mu_star),
        eps_greedy.get_expected_cum_regret(mu_star),
        decay_eps_greedy.get_expected_cum_regret(mu_star),
        ucb1.get_expected_cum_regret(mu_star),
        ts.get_expected_cum_regret(mu_star),
        grad_bandits.get_expected_cum_regret(mu_star)
    ]

    x_vals = range(1, steps + 1)
    for i in range(len(exp_cum_regrets)):
        plt.plot(x_vals, exp_cum_regrets[i], plot_colors[i], label=labels[i])
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Expected Total Regret", fontsize=20)
    plt.title("Total Regret Curves", fontsize=25)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

    exp_act_counts = [
        greedy_opt_init.get_expected_action_counts(),
        eps_greedy.get_expected_action_counts(),
        decay_eps_greedy.get_expected_action_counts(),
        ucb1.get_expected_action_counts(),
        ts.get_expected_action_counts(),
        grad_bandits.get_expected_action_counts()
    ]
    index = arange(len(probs_data))
    spacing = 0.4
    width = (1 - spacing) / len(exp_act_counts)

    hist_plot_colors = ['r', 'b', 'g', 'k', 'y', "c"]
    for i in range(len(exp_act_counts)):
        plt.bar(
            index - (1 - spacing) / 2 + (i - 1.5) * width,
            exp_act_counts[i],
            width,
            color=hist_plot_colors[i],
            label=labels[i]
        )
    plt.xlabel("Arms", fontsize=20)
    plt.ylabel("Expected Counts of Arms", fontsize=20)
    plt.title("Arms Counts Plot", fontsize=25)
    plt.xticks(
        index - 0.2,
        ["$p$=%.2f" % p for p in probs_data]
    )
    plt.legend(loc='upper left', fontsize=15)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_gaussian_algorithms()
    plot_bernoulli_algorithms()
```

--------------------------------------------------------------------------------

## 📄 chapter14/ts_bernoulli.py {#ts-bernoulli}

**Titre**: Ts Bernoulli

**Description**: Module Ts Bernoulli

**Lignes de code**: 57

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Tuple, List
from rl.distribution import Bernoulli, Beta
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty


class ThompsonSamplingBernoulli(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Bernoulli],
        time_steps: int,
        num_episodes: int
    ) -> None:
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        bayes: List[Tuple[int, int]] = [(1, 1)] * self.num_arms

        for i in range(self.time_steps):
            mean_draws: Sequence[float] = \
                [Beta(α=alpha, β=beta).sample() for alpha, beta in bayes]
            action: int = max(enumerate(mean_draws), key=itemgetter(1))[0]
            reward: float = float(self.arm_distributions[action].sample())
            alpha, beta = bayes[action]
            bayes[action] = (alpha + int(reward), beta + int(1 - reward))
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    probs_data = [0.2, 0.4, 0.8, 0.5, 0.1, 0.9]
    mu_star = max(probs_data)
    steps = 1000
    episodes = 500

    arm_distrs = [Bernoulli(p) for p in probs_data]
    ts_bernoulli = ThompsonSamplingBernoulli(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes
    )
    # exp_cum_regret = ts_bernoulli.get_expected_cum_regret(mu_star)
    # print(exp_cum_regret)
    # exp_act_count = ts_bernoulli.get_expected_action_counts()
    # print(exp_act_count)

    ts_bernoulli.plot_exp_cum_regret_curve(mu_star)
```

--------------------------------------------------------------------------------

## 📄 chapter14/ts_gaussian.py {#ts-gaussian}

**Titre**: Ts Gaussian

**Description**: Module Ts Gaussian

**Lignes de code**: 80

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Tuple, List
from rl.distribution import Gaussian, Gamma
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty, sqrt


class ThompsonSamplingGaussian(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Gaussian],
        time_steps: int,
        num_episodes: int,
        init_mean: float,
        init_stdev: float
    ) -> None:
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.theta0: float = init_mean
        self.n0: int = 1
        self.alpha0: float = 1
        self.beta0: float = init_stdev * init_stdev

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        # Bayesian update based on the treatment in
        # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
        # (Section 3 on page 5, where both the mean and the
        # variance are random)
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        bayes: List[Tuple[float, int, float, float]] =\
            [(self.theta0, self.n0, self.alpha0, self.beta0)] * self.num_arms

        for i in range(self.time_steps):
            mean_draws: Sequence[float] = [Gaussian(
                μ=theta,
                σ=1 / sqrt(n * Gamma(α=alpha, β=beta).sample())
            ).sample() for theta, n, alpha, beta in bayes]
            action: int = max(enumerate(mean_draws), key=itemgetter(1))[0]
            reward: float = self.arm_distributions[action].sample()
            theta, n, alpha, beta = bayes[action]
            bayes[action] = (
                (reward + n * theta) / (n + 1),
                n + 1,
                alpha + 0.5,
                beta + 0.5 * n / (n + 1) * (reward - theta) * (reward - theta)
            )
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    means_vars_data = [(9., 5.), (10., 2.), (0., 4.),
                       (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(means_vars_data, key=itemgetter(0))[0]
    steps = 1000
    episodes = 500
    guess_mean = 0.
    guess_stdev = 10.

    arm_distrs = [Gaussian(μ=m, σ=s) for m, s in means_vars_data]
    ts_gaussian = ThompsonSamplingGaussian(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        init_mean=guess_mean,
        init_stdev=guess_stdev
    )
    # exp_cum_regret = ts_gaussian.get_expected_cum_regret(mu_star)
    # print(exp_cum_regret)
    # exp_act_count = ts_gaussian.get_expected_action_counts()
    # print(exp_act_count)

    ts_gaussian.plot_exp_cum_regret_curve(mu_star)
```

--------------------------------------------------------------------------------

## 📄 chapter14/ucb1.py {#ucb1}

**Titre**: Ucb1

**Description**: Module Ucb1

**Lignes de code**: 77

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Tuple, List
from rl.distribution import Distribution, Categorical
from math import comb
from rl.chapter14.mab_base import MABBase
from operator import itemgetter
from numpy import ndarray, empty, sqrt, log


class UCB1(MABBase):

    def __init__(
        self,
        arm_distributions: Sequence[Distribution[float]],
        time_steps: int,
        num_episodes: int,
        bounds_range: float,
        alpha: float
    ) -> None:
        if bounds_range < 0 or alpha <= 0:
            raise ValueError
        super().__init__(
            arm_distributions=arm_distributions,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.bounds_range: float = bounds_range
        self.alpha: float = alpha

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.num_arms):
            ep_rewards[i] = self.arm_distributions[i].sample()
            ep_actions[i] = i
        counts: List[int] = [1] * self.num_arms
        means: List[float] = [ep_rewards[j] for j in range(self.num_arms)]
        for i in range(self.num_arms, self.time_steps):
            ucbs: Sequence[float] = [means[j] + self.bounds_range *
                                     sqrt(0.5 * self.alpha * log(i) /
                                          counts[j])
                                     for j in range(self.num_arms)]
            action: int = max(enumerate(ucbs), key=itemgetter(1))[0]
            reward: float = self.arm_distributions[action].sample()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    binomial_count = 10
    binomial_probs = [0.4, 0.8, 0.1, 0.5, 0.9, 0.2]
    binomial_params = [(binomial_count, p) for p in binomial_probs]
    mu_star = max(n * p for n, p in binomial_params)
    steps = 1000
    episodes = 500
    this_range = binomial_count
    this_alpha = 4.0

    arm_distrs = [Categorical(
        {float(i): p ** i * (1-p) ** (n-i) * comb(n, i) for i in range(n + 1)}
    ) for n, p in binomial_params]
    ucb1 = UCB1(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        bounds_range=this_range,
        alpha=this_alpha
    )
    # exp_cum_regret = ucb1.get_expected_cum_regret(mu_star)
    # print(exp_cum_regret)
    # exp_act_count = ucb1.get_expected_action_counts()
    # print(exp_act_count)

    ucb1.plot_exp_cum_regret_curve(mu_star)
```

--------------------------------------------------------------------------------

## 📄 chapter15/ams.py {#ams}

**Titre**: Ams

**Description**: Module Ams

**Lignes de code**: 120

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Callable, Sequence, Set, TypeVar, Generic, \
    Mapping, Dict, Tuple
from rl.distribution import Distribution
import numpy as np
from operator import itemgetter

A = TypeVar('A')
S = TypeVar('S')


class AMS(Generic[S, A]):

    def __init__(
        self,
        actions_funcs: Sequence[Callable[[S], Set[A]]],
        state_distr_funcs: Sequence[Callable[[S, A], Distribution[S]]],
        expected_reward_funcs: Sequence[Callable[[S, A], float]],
        num_samples: Sequence[int],
        gamma: float
    ) -> None:
        self.num_steps: int = len(actions_funcs)
        self.actions_funcs: Sequence[Callable[[S], Set[A]]] = actions_funcs
        self.state_distr_funcs: Sequence[Callable[[S, A], Distribution[S]]] = \
            state_distr_funcs
        self.expected_reward_funcs: Sequence[Callable[[S, A], float]] = \
            expected_reward_funcs
        self.num_samples: Sequence[int] = num_samples
        self.gamma: float = gamma

    def optimal_vf_and_policy(self, t: int, s: S) -> \
            Tuple[float, A]:

        actions: Set[A] = self.actions_funcs[t](s)
        state_distr_func: Callable[[S, A], Distribution[S]] = \
            self.state_distr_funcs[t]
        expected_reward_func: Callable[[S, A], float] = \
            self.expected_reward_funcs[t]
        # sample each action once, sample each action's next state, and
        # recursively call the next state's V* estimate
        rewards: Mapping[A, float] = {a: expected_reward_func(s, a)
                                      for a in actions}
        val_sums: Dict[A, float] = {a: (self.optimal_vf_and_policy(
            t + 1,
            state_distr_func(s, a).sample()
        )[0] if t < self.num_steps - 1 else 0.) for a in actions}
        counts: Dict[A, int] = {a: 1 for a in actions}
        # loop num_samples[t] number of times (beyond the
        # len(actions) samples that have already been done above
        for i in range(len(actions), self.num_samples[t]):
            # determine the actions that dominate on the UCB Q* estimated value
            # and pick one of these dominating actions at random, call it a*
            ucb_vals: Mapping[A, float] = \
                {a: rewards[a] + self.gamma * val_sums[a] / counts[a] +
                 np.sqrt(2 * np.log(i) / counts[a]) for a in actions}
            max_actions: Sequence[A] = [a for a, u in ucb_vals.items()
                                        if u == max(ucb_vals.values())]
            a_star: A = np.random.default_rng().choice(max_actions)
            # sample a*'s next state and reward at random, and recursively
            # call the next state's V* estimate
            val_sums[a_star] += (self.optimal_vf_and_policy(
                t + 1,
                state_distr_func(s, a_star).sample()
            )[0] if t < self.num_steps - 1 else 0.)
            counts[a_star] += 1

        # return estimated V* as weighted average of the estimated Q* where
        # weights are proportioned by the number of times an action was sampled
        return (
            sum(counts[a] / self.num_samples[t] *
                (rewards[a] + self.gamma * val_sums[a] / counts[a])
                for a in actions),
            max(
                [(a, rewards[a] + self.gamma * val_sums[a] / counts[a])
                 for a in actions],
                key=itemgetter(1)
            )[0]
        )


if __name__ == '__main__':

    from rl.chapter4.clearance_pricing_mdp import ClearancePricingMDP
    from rl.distribution import Categorical
    from scipy.stats import poisson
    from pprint import pprint

    ii = 5
    steps = 3
    pairs = [(1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5)]

    ams: AMS[int, int] = AMS(
        actions_funcs=[lambda _: set(range(len(pairs)))] * steps,
        state_distr_funcs=[lambda s, a: Categorical({
            s - i: (poisson(pairs[a][1]).pmf(i) if i < s else
                    (1 - poisson(pairs[a][1]).cdf(s - 1)))
            for i in range(s + 1)
        })] * steps,
        expected_reward_funcs=[lambda s, a: sum(
            poisson(pairs[a][1]).pmf(i) * pairs[a][0] * i for i in range(s)
        ) + (1 - poisson(pairs[a][1]).cdf(s - 1)) * pairs[a][0] * s] * steps,
        num_samples=[100] * steps,
        gamma=1.0
    )

    print("AMS Optimal Value Function and Optimal Policy for t=0")
    print("------------------------------")
    print({s: ams.optimal_vf_and_policy(0, s) for s in range(ii + 1)})

    cp: ClearancePricingMDP = ClearancePricingMDP(
        initial_inventory=ii,
        time_steps=steps,
        price_lambda_pairs=pairs
    )

    print("BI Optimal Value Function and Optimal Policy for t =0")
    print("------------------------------------")
    vf, policy = next(cp.get_optimal_vf_and_policy())
    pprint(vf)
    print(policy)
```

--------------------------------------------------------------------------------

## 📄 chapter2/simple_inventory_mp.py {#simple-inventory-mp}

**Titre**: Simple Inventory Mp

**Description**: Module Simple Inventory Mp

**Lignes de code**: 64

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Mapping, Dict
from rl.distribution import Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


class SimpleInventoryMPFinite(FiniteMarkovProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[InventoryState, FiniteDistribution[InventoryState]]:
        d: Dict[InventoryState, Categorical[InventoryState]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                state_probs_map: Mapping[InventoryState, float] = {
                    InventoryState(ip - i, beta1):
                    (self.poisson_distr.pmf(i) if i < ip else
                     1 - self.poisson_distr.cdf(ip - 1))
                    for i in range(ip + 1)
                }
                d[InventoryState(alpha, beta)] = Categorical(state_probs_map)
        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0

    si_mp = SimpleInventoryMPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda
    )

    print("Transition Map")
    print("--------------")
    print(si_mp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mp.display_stationary_distribution()
```

--------------------------------------------------------------------------------

## 📄 chapter2/simple_inventory_mrp.py {#simple-inventory-mrp}

**Titre**: Simple Inventory Mrp

**Description**: Module Simple Inventory Mrp

**Lignes de code**: 137

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_process import MarkovRewardProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import State, NonTerminal
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical, \
    FiniteDistribution
import numpy as np


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


class SimpleInventoryMRP(MarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def transition_reward(
        self,
        state: NonTerminal[InventoryState]
    ) -> SampledDistribution[Tuple[State[InventoryState], float]]:

        def sample_next_state_reward(state=state) ->\
                Tuple[State[InventoryState], float]:
            demand_sample: int = np.random.poisson(self.poisson_lambda)
            ip: int = state.state.inventory_position()
            next_state: InventoryState = InventoryState(
                max(ip - demand_sample, 0),
                max(self.capacity - ip, 0)
            )
            reward: float = - self.holding_cost * state.state.on_hand\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return NonTerminal(next_state), reward

        return SampledDistribution(sample_next_state_reward)


class SimpleInventoryMRPFinite(FiniteMarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> \
            Mapping[
                InventoryState,
                FiniteDistribution[Tuple[InventoryState, float]]
            ]:
        d: Dict[InventoryState, Categorical[Tuple[InventoryState, float]]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                base_reward = - self.holding_cost * state.on_hand
                sr_probs_map: Dict[Tuple[InventoryState, float], float] =\
                    {(InventoryState(ip - i, beta1), base_reward):
                     self.poisson_distr.pmf(i) for i in range(ip)}
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                reward = base_reward - self.stockout_cost * \
                    (self.poisson_lambda - ip *
                     (1 - self.poisson_distr.pmf(ip) / probability))
                sr_probs_map[(InventoryState(0, beta1), reward)] = probability
                d[state] = Categorical(sr_probs_map)
        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    from rl.markov_process import FiniteMarkovProcess
    print("Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(
        {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
         for s, v in si_mrp.transition_map.items()}
    ))

    print("Transition Reward Map")
    print("---------------------")
    print(si_mrp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mrp.display_stationary_distribution()
    print()

    print("Reward Function")
    print("---------------")
    si_mrp.display_reward_function()
    print()

    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()
```

--------------------------------------------------------------------------------

## 📄 chapter3/simple_inventory_mdp_cap.py {#simple-inventory-mdp-cap}

**Titre**: Simple Inventory Mdp Cap

**Description**: Module Simple Inventory Mdp Cap

**Lignes de code**: 156

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[int, Categorical[Tuple[InventoryState, float]]]
]


class SimpleInventoryMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[int, Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state: InventoryState = InventoryState(alpha, beta)
                ip: int = state.inventory_position()
                base_reward: float = - self.holding_cost * alpha
                d1: Dict[int, Categorical[Tuple[InventoryState, float]]] = {}

                for order in range(self.capacity - ip + 1):
                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] =\
                        {(InventoryState(ip - i, order), base_reward):
                         self.poisson_distr.pmf(i) for i in range(ip)}

                    probability: float = 1 - self.poisson_distr.cdf(ip - 1)
                    reward: float = base_reward - self.stockout_cost * \
                        (self.poisson_lambda - ip * 
                        (1 - self.poisson_distr.pmf(ip) / probability))
                    sr_probs_dict[(InventoryState(0, order), reward)] = \
                        probability
                    d1[order] = Categorical(sr_probs_dict)

                d[state] = d1
        return d


if __name__ == '__main__':
    from pprint import pprint

    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)

    fdp: FiniteDeterministicPolicy[InventoryState, int] = \
        FiniteDeterministicPolicy(
            {InventoryState(alpha, beta): user_capacity - (alpha + beta)
             for alpha in range(user_capacity + 1)
             for beta in range(user_capacity + 1 - alpha)}
    )

    print("Deterministic Policy Map")
    print("------------------------")
    print(fdp)

    implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
        si_mdp.apply_finite_policy(fdp)
    print("Implied MP Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(
        {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
         for s, v in implied_mrp.transition_map.items()}
    ))

    print("Implied MRP Transition Reward Map")
    print("---------------------")
    print(implied_mrp)

    print("Implied MP Stationary Distribution")
    print("-----------------------")
    implied_mrp.display_stationary_distribution()
    print()

    print("Implied MRP Reward Function")
    print("---------------")
    implied_mrp.display_reward_function()
    print()

    print("Implied MRP Value Function")
    print("--------------")
    implied_mrp.display_value_function(gamma=user_gamma)
    print()

    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result

    print("Implied MRP Policy Evaluation Value Function")
    print("--------------")
    pprint(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
    print()

    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    pprint(opt_vf_pi)
    print(opt_policy_pi)
    print()

    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()
```

--------------------------------------------------------------------------------

## 📄 chapter3/simple_inventory_mdp_nocap.py {#simple-inventory-mdp-nocap}

**Titre**: Simple Inventory Mdp Nocap

**Description**: Module Simple Inventory Mdp Nocap

**Lignes de code**: 144

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from dataclasses import dataclass
from typing import Tuple, Iterator
import itertools
import numpy as np
from scipy.stats import poisson
import random

from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import MarkovRewardProcess, NonTerminal, State
from rl.policy import Policy, DeterministicPolicy
from rl.distribution import Constant, SampledDistribution


@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


@dataclass(frozen=True)
class SimpleInventoryMDPNoCap(MarkovDecisionProcess[InventoryState, int]):
    poisson_lambda: float
    holding_cost: float
    stockout_cost: float

    def step(
        self,
        state: NonTerminal[InventoryState],
        order: int
    ) -> SampledDistribution[Tuple[State[InventoryState], float]]:

        def sample_next_state_reward(
            state=state,
            order=order
        ) -> Tuple[State[InventoryState], float]:
            demand_sample: int = np.random.poisson(self.poisson_lambda)
            ip: int = state.state.inventory_position()
            next_state: InventoryState = InventoryState(
                max(ip - demand_sample, 0),
                order
            )
            reward: float = - self.holding_cost * state.state.on_hand\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return NonTerminal(next_state), reward

        return SampledDistribution(sample_next_state_reward)

    def actions(self, state: NonTerminal[InventoryState]) -> Iterator[int]:
        return itertools.count(start=0, step=1)

    def fraction_of_days_oos(
        self,
        policy: Policy[InventoryState, int],
        time_steps: int,
        num_traces: int
    ) -> float:
        impl_mrp: MarkovRewardProcess[InventoryState] =\
            self.apply_policy(policy)
        count: int = 0
        high_fractile: int = int(poisson(self.poisson_lambda).ppf(0.98))
        start: InventoryState = random.choice(
            [InventoryState(i, 0) for i in range(high_fractile + 1)])

        for _ in range(num_traces):
            steps = itertools.islice(
                impl_mrp.simulate_reward(Constant(NonTerminal(start))),
                time_steps
            )
            for step in steps:
                if step.reward < -self.holding_cost * step.state.state.on_hand:
                    count += 1

        return float(count) / (time_steps * num_traces)


class SimpleInventoryDeterministicPolicy(
        DeterministicPolicy[InventoryState, int]
):
    def __init__(self, reorder_point: int):
        self.reorder_point: int = reorder_point

        def action_for(s: InventoryState) -> int:
            return max(self.reorder_point - s.inventory_position(), 0)

        super().__init__(action_for)


class SimpleInventoryStochasticPolicy(Policy[InventoryState, int]):
    def __init__(self, reorder_point_poisson_mean: float):
        self.reorder_point_poisson_mean: float = reorder_point_poisson_mean

    def act(self, state: NonTerminal[InventoryState]) -> \
            SampledDistribution[int]:
        def action_func(state=state) -> int:
            reorder_point_sample: int = \
                np.random.poisson(self.reorder_point_poisson_mean)
            return max(
                reorder_point_sample - state.state.inventory_position(),
                0
            )
        return SampledDistribution(action_func)


if __name__ == '__main__':
    user_poisson_lambda = 2.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_reorder_point = 8
    user_reorder_point_poisson_mean = 8.0

    user_time_steps = 1000
    user_num_traces = 1000

    si_mdp_nocap = SimpleInventoryMDPNoCap(poisson_lambda=user_poisson_lambda,
                                           holding_cost=user_holding_cost,
                                           stockout_cost=user_stockout_cost)

    si_dp = SimpleInventoryDeterministicPolicy(
        reorder_point=user_reorder_point
    )

    oos_frac_dp = si_mdp_nocap.fraction_of_days_oos(policy=si_dp,
                                                    time_steps=user_time_steps,
                                                    num_traces=user_num_traces)
    print(
        f"Deterministic Policy yields {oos_frac_dp * 100:.2f}%"
        + " of Out-Of-Stock days"
    )

    si_sp = SimpleInventoryStochasticPolicy(
        reorder_point_poisson_mean=user_reorder_point_poisson_mean)

    oos_frac_sp = si_mdp_nocap.fraction_of_days_oos(policy=si_sp,
                                                    time_steps=user_time_steps,
                                                    num_traces=user_num_traces)
    print(
        f"Stochastic Policy yields {oos_frac_sp * 100:.2f}%"
        + " of Out-Of-Stock days"
    )
```

--------------------------------------------------------------------------------

## 📄 chapter4/clearance_pricing_mdp.py {#clearance-pricing-mdp}

**Titre**: Clearance Pricing Mdp

**Description**: Module Clearance Pricing Mdp

**Lignes de code**: 112

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from rl.markov_decision_process import (
    FiniteMarkovDecisionProcess, FiniteMarkovRewardProcess)
from rl.policy import FiniteDeterministicPolicy, FinitePolicy
from rl.finite_horizon import WithTime
from typing import Sequence, Tuple, Iterator
from scipy.stats import poisson
from rl.distribution import Categorical
from rl.finite_horizon import (
    finite_horizon_MRP, unwrap_finite_horizon_MRP, evaluate,
    finite_horizon_MDP, unwrap_finite_horizon_MDP, optimal_vf_and_policy)
from rl.dynamic_programming import V


class ClearancePricingMDP:

    initial_inventory: int
    time_steps: int
    price_lambda_pairs: Sequence[Tuple[float, float]]
    single_step_mdp: FiniteMarkovDecisionProcess[int, int]
    mdp: FiniteMarkovDecisionProcess[WithTime[int], int]

    def __init__(
        self,
        initial_inventory: int,
        time_steps: int,
        price_lambda_pairs: Sequence[Tuple[float, float]]
    ):
        self.initial_inventory = initial_inventory
        self.time_steps = time_steps
        self.price_lambda_pairs = price_lambda_pairs
        distrs = [poisson(l) for _, l in price_lambda_pairs]
        prices = [p for p, _ in price_lambda_pairs]
        self.single_step_mdp: FiniteMarkovDecisionProcess[int, int] =\
            FiniteMarkovDecisionProcess({
                s: {i: Categorical(
                    {(s - k, prices[i] * k):
                     (distrs[i].pmf(k) if k < s else 1 - distrs[i].cdf(s - 1))
                     for k in range(s + 1)})
                    for i in range(len(prices))}
                for s in range(initial_inventory + 1)
            })
        self.mdp = finite_horizon_MDP(self.single_step_mdp, time_steps)

    def get_vf_for_policy(
        self,
        policy: FinitePolicy[WithTime[int], int]
    ) -> Iterator[V[int]]:
        mrp: FiniteMarkovRewardProcess[WithTime[int]] \
            = self.mdp.apply_finite_policy(policy)
        return evaluate(unwrap_finite_horizon_MRP(mrp), 1.)

    def get_optimal_vf_and_policy(self)\
            -> Iterator[Tuple[V[int], FiniteDeterministicPolicy[int, int]]]:
        return optimal_vf_and_policy(unwrap_finite_horizon_MDP(self.mdp), 1.)


if __name__ == '__main__':
    from pprint import pprint
    ii = 12
    steps = 8
    pairs = [(1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5)]
    cp: ClearancePricingMDP = ClearancePricingMDP(
        initial_inventory=ii,
        time_steps=steps,
        price_lambda_pairs=pairs
    )
    print("Clearance Pricing MDP")
    print("---------------------")
    print(cp.mdp)

    def policy_func(x: int) -> int:
        return 0 if x < 2 else (1 if x < 5 else (2 if x < 8 else 3))

    stationary_policy: FiniteDeterministicPolicy[int, int] = \
        FiniteDeterministicPolicy({s: policy_func(s) for s in range(ii + 1)})

    single_step_mrp: FiniteMarkovRewardProcess[int] = \
        cp.single_step_mdp.apply_finite_policy(stationary_policy)

    vf_for_policy: Iterator[V[int]] = evaluate(
        unwrap_finite_horizon_MRP(finite_horizon_MRP(single_step_mrp, steps)),
        1.
    )

    print("Value Function for Stationary Policy")
    print("------------------------------------")
    for t, vf in enumerate(vf_for_policy):
        print(f"Time Step {t:d}")
        print("---------------")
        pprint(vf)

    print("Optimal Value Function and Optimal Policy")
    print("------------------------------------")
    prices = []
    for t, (vf, policy) in enumerate(cp.get_optimal_vf_and_policy()):
        print(f"Time Step {t:d}")
        print("---------------")
        pprint(vf)
        print(policy)
        prices.append(
            [pairs[policy.action_for[s]][0]
             for s in range(ii + 1)])

    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    heatmap = plt.imshow(np.array(prices).T, origin='lower')
    plt.colorbar(heatmap, shrink=0.5, aspect=5)
    plt.xlabel("Time Steps")
    plt.ylabel("Inventory")
    plt.show()
```

--------------------------------------------------------------------------------

## 📄 gen_utils/common_funcs.py {#common-funcs}

**Titre**: Common Funcs

**Description**: Module Common Funcs

**Lignes de code**: 43

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Callable
import numpy as np
VSML = 1e-8


def get_logistic_func(alpha: float) -> Callable[[float], float]:
    return lambda x: 1. / (1 + np.exp(-alpha * x))


def get_unit_sigmoid_func(alpha: float) -> Callable[[float], float]:
    return lambda x: 1. / (1 + (1 / np.where(x == 0, VSML, x) - 1) ** alpha)


if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves
    alpha = [2.0, 1.0, 0.5]
    colors = ["r-", "b--", "g-."]
    labels = [(r"$\alpha$ = %.1f" % a) for a in alpha]
    logistics = [get_logistic_func(a) for a in alpha]
    x_vals = np.arange(-3.0, 3.01, 0.05)
    y_vals = [f(x_vals) for f in logistics]
    plot_list_of_curves(
        [x_vals] * len(logistics),
        y_vals,
        colors,
        labels,
        title="Logistic Functions"
    )

    alpha = [2.0, 1.0, 0.5]
    colors = ["r-", "b--", "g-."]
    labels = [(r"$\alpha$ = %.1f" % a) for a in alpha]
    unit_sigmoids = [get_unit_sigmoid_func(a) for a in alpha]
    x_vals = np.arange(0.0, 1.01, 0.01)
    y_vals = [f(x_vals) for f in unit_sigmoids]
    plot_list_of_curves(
        [x_vals] * len(logistics),
        y_vals,
        colors,
        labels,
        title="Unit-Sigmoid Functions"
    )
```

--------------------------------------------------------------------------------

## 📄 gen_utils/plot_funcs.py {#plot-funcs}

**Titre**: Plot Funcs

**Description**: Module Plot Funcs

**Lignes de code**: 53

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_list_of_curves(
    list_of_x_vals,
    list_of_y_vals,
    list_of_colors,
    list_of_curve_labels,
    x_label=None,
    y_label=None,
    title=None
):
    plt.figure(figsize=(11, 7))
    for i, x_vals in enumerate(list_of_x_vals):
        plt.plot(
            x_vals,
            list_of_y_vals[i],
            list_of_colors[i],
            label=list_of_curve_labels[i]
        )
    plt.axis((
        min(map(min, list_of_x_vals)),
        max(map(max, list_of_x_vals)),
        min(map(min, list_of_y_vals)),
        max(map(max, list_of_y_vals))
    ))
    if x_label is not None:
        plt.xlabel(x_label, fontsize=20)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=20)
    if title is not None:
        plt.title(title, fontsize=25)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()


if __name__ == '__main__':
    x = np.arange(1, 100)
    y = [0.1 * x + 1.0, 0.001 * (x - 50) ** 2, np.log(x)]
    colors = ["r", "b", "g"]
    labels = ["Linear", "Quadratic", "Log"]
    plot_list_of_curves(
        [x, x, x],
        y,
        colors,
        labels,
        "X-Axis",
        "Y-Axis",
        "Test Plot"
    )
```

--------------------------------------------------------------------------------

## 📄 problems/Final-Winter2021/windy_grid.py {#windy-grid}

**Titre**: Windy Grid

**Description**: Module Windy Grid

**Lignes de code**: 348

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Tuple, Callable, Sequence, Set, Mapping, Dict
from dataclasses import dataclass
from rl.distribution import Categorical, Choose
from rl.markov_process import NonTerminal
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.dynamic_programming import value_iteration_result, V
from operator import itemgetter

'''
Cell specifies (row, column) coordinate
'''
Cell = Tuple[int, int]
CellSet = Set[Cell]
Move = Tuple[int, int]
'''
WindSpec specifies a random vectical wind for each column.
Each random vertical wind is specified by a (p1, p2) pair
where p1 specifies probability of Downward Wind (could take you
one step lower in row coordinate unless prevented by a block or
boundary) and p2 specifies probability of Upward Wind (could take
you onw step higher in column coordinate unless prevented by a
block or boundary). If one bumps against a block or boundary, one
incurs a bump cost and doesn't move. The remaining probability
1- p1 - p2 corresponds to No Wind.
'''
WindSpec = Sequence[Tuple[float, float]]

possible_moves: Mapping[Move, str] = {
    (-1, 0): 'D',
    (1, 0): 'U',
    (0, -1): 'L',
    (0, 1): 'R'
}


@dataclass(frozen=True)
class WindyGrid:

    rows: int  # number of grid rows
    columns: int  # number of grid columns
    blocks: CellSet  # coordinates of block cells
    terminals: CellSet  # coordinates of goal cells
    wind: WindSpec  # spec of vertical random wind for the columns
    bump_cost: float  # cost of bumping against block or boundary

    def validate_spec(self) -> bool:
        b1 = self.rows >= 2
        b2 = self.columns >= 2
        b3 = all(0 <= r < self.rows and 0 <= c < self.columns
                 for r, c in self.blocks)
        b4 = len(self.terminals) >= 1
        b5 = all(0 <= r < self.rows and 0 <= c < self.columns and
                 (r, c) not in self.blocks for r, c in self.terminals)
        b6 = len(self.wind) == self.columns
        b7 = all(0. <= p1 <= 1. and 0. <= p2 <= 1. and p1 + p2 <= 1.
                 for p1, p2 in self.wind)
        b8 = self.bump_cost > 0.
        return all([b1, b2, b3, b4, b5, b6, b7, b8])

    def print_wind_and_bumps(self) -> None:
        for i, (d, u) in enumerate(self.wind):
            print(f"Column {i:d}: Down Prob = {d:.2f}, Up Prob = {u:.2f}")
        print(f"Bump Cost = {self.bump_cost:.2f}")
        print()

    @staticmethod
    def add_move_to_cell(cell: Cell, move: Cell) -> Cell:
        return cell[0] + move[0], cell[1] + move[1]

    def is_valid_state(self, cell: Cell) -> bool:
        '''
        checks if a cell is a valid state of the MDP
        '''
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.columns \
            and cell not in self.blocks

    def get_all_nt_states(self) -> CellSet:
        '''
        returns all the non-terminal states
        '''
        return {(i, j) for i in range(self.rows) for j in range(self.columns)
                if (i, j) not in set.union(self.blocks, self.terminals)}

    def get_actions_and_next_states(self, nt_state: Cell) \
            -> Set[Tuple[Move, Cell]]:
        '''
        given a non-terminal state, returns the set of all possible
        (action, next_state) pairs
        '''
        temp: Set[Tuple[Move, Cell]] = {(a, WindyGrid.add_move_to_cell(
            nt_state,
            a
        )) for a in possible_moves}
        return {(a, s) for a, s in temp if self.is_valid_state(s)}

    def get_transition_probabilities(self, nt_state: Cell) \
            -> Mapping[Move, Categorical[Tuple[Cell, float]]]:
        '''
        given a non-terminal state, return a dictionary whose
        keys are the valid actions (moves) from the given state
        and the corresponding values are the associated probabilities
        (following that move) of the (next_state, reward) pairs.
        The probabilities are determined from the wind probabilities
        of the column one is in after the move. Note that if one moves
        to a goal cell (terminal state), then one ends up in that
        goal cell with 100% probability (i.e., no wind exposure in a
        goal cell).
        '''
        d: Dict[Move, Categorical[Tuple[Cell, float]]] = {}
        for a, (r, c) in self.get_actions_and_next_states(nt_state):
            if (r, c) in self.terminals:
                d[a] = Categorical({((r, c), -1.): 1.})
            else:
                down_prob, up_prob = self.wind[c]
                stay_prob: float = 1. - down_prob - up_prob
                d1: Dict[Tuple[Cell, float], float] = \
                    {((r, c), -1.): stay_prob}
                if self.is_valid_state((r - 1, c)):
                    d1[((r - 1, c), -1.)] = down_prob
                if self.is_valid_state((r + 1, c)):
                    d1[((r + 1, c), -1.)] = up_prob
                d1[((r, c), -1. - self.bump_cost)] = \
                    down_prob * (1 - self.is_valid_state((r - 1, c))) + \
                    up_prob * (1 - self.is_valid_state((r + 1, c)))
                d[a] = Categorical(d1)
        return d

    def get_finite_mdp(self) -> FiniteMarkovDecisionProcess[Cell, Move]:
        '''
        returns the FiniteMarkovDecision object for this windy grid problem
        '''
        return FiniteMarkovDecisionProcess(
            {s: self.get_transition_probabilities(s) for s in
             self.get_all_nt_states()}
        )

    def get_vi_vf_and_policy(self) -> \
            Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        '''
        Performs the Value Iteration DP algorithm returning the
        Optimal Value Function (as a V[Cell]) and the Optimal Policy
        (as a FiniteDeterministicPolicy[Cell, Move])
        '''
        return value_iteration_result(self.get_finite_mdp(), gamma=1.)

    @staticmethod
    def epsilon_greedy_action(
        nt_state: Cell,
        q: Mapping[Cell, Mapping[Move, float]],
        epsilon: float
    ) -> Move:
        '''
        given a non-terminal state, a Q-Value Function (in the form of a
        {state: {action: Expected Return}} dictionary) and epislon, return
        an action sampled from the probability distribution implied by an
        epsilon-greedy policy that is derived from the Q-Value Function.
        '''
        action_values: Mapping[Move, float] = q[nt_state]
        greedy_action: Move = max(action_values.items(), key=itemgetter(1))[0]
        return Categorical(
            {a: epsilon / len(action_values) +
             (1 - epsilon if a == greedy_action else 0.)
             for a in action_values}
        ).sample()

    def get_states_actions_dict(self) -> Mapping[Cell, Set[Move]]:
        '''
        Returns a dictionary whose keys are the non-terminal states and
        the corresponding values are the set of actions for the state
        '''
        return {s: {a for a, _ in self.get_actions_and_next_states(s)}
                for s in self.get_all_nt_states()}

    def get_sarsa_vf_and_policy(
        self,
        states_actions_dict: Mapping[Cell, Set[Move]],
        sample_func: Callable[[Cell, Move], Tuple[Cell, float]],
        episodes: int = 10000,
        step_size: float = 0.01
    ) -> Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        '''
        states_actions_dict gives us the set of possible moves from
        a non-terminal cell.
        sample_func is a function with two inputs: state and action,
        and with output as a sampled pair of (next_state, reward).
        '''
        q: Dict[Cell, Dict[Move, float]] = \
            {s: {a: 0. for a in actions} for s, actions in
             states_actions_dict.items()}
        nt_states: CellSet = {s for s in q}
        uniform_states: Choose[Cell] = Choose(nt_states)
        for episode_num in range(episodes):
            epsilon: float = 1.0 / (episode_num + 1)
            state: Cell = uniform_states.sample()
            action: Move = WindyGrid.epsilon_greedy_action(
                state,
                q,
                epsilon
            )
            while state in nt_states:
                next_state, reward = sample_func(state, action)
                if next_state in nt_states:
                    next_action: Move = WindyGrid.epsilon_greedy_action(
                        next_state,
                        q,
                        epsilon
                    )
                    q[state][action] += step_size * \
                        (reward + q[next_state][next_action] -
                         q[state][action])
                    action = next_action
                else:
                    q[state][action] += step_size * (reward - q[state][action])
                state = next_state

        vf_dict: V[Cell] = {NonTerminal(s): max(d.values()) for s, d
                            in q.items()}
        policy: FiniteDeterministicPolicy[Cell, Move] = \
            FiniteDeterministicPolicy(
                {s: max(d.items(), key=itemgetter(1))[0] for s, d in q.items()}
            )
        return vf_dict, policy

    def get_q_learning_vf_and_policy(
        self,
        states_actions_dict: Mapping[Cell, Set[Move]],
        sample_func: Callable[[Cell, Move], Tuple[Cell, float]],
        episodes: int = 10000,
        step_size: float = 0.01,
        epsilon: float = 0.1
    ) -> Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        '''
        states_actions_dict gives us the set of possible moves from
        a non-block cell.
        sample_func is a function with two inputs: state and action,
        and with output as a sampled pair of (next_state, reward).
        '''
        q: Dict[Cell, Dict[Move, float]] = \
            {s: {a: 0. for a in actions} for s, actions in
             states_actions_dict.items()}
        nt_states: CellSet = {s for s in q}
        uniform_states: Choose[Cell] = Choose(nt_states)
        for episode_num in range(episodes):
            state: Cell = uniform_states.sample()
            while state in nt_states:
                action: Move = WindyGrid.epsilon_greedy_action(
                    state,
                    q,
                    epsilon
                )
                next_state, reward = sample_func(state, action)
                q[state][action] += step_size * \
                    (reward + (max(q[next_state].values())
                               if next_state in nt_states else 0.)
                     - q[state][action])
                state = next_state

        vf_dict: V[Cell] = {NonTerminal(s): max(d.values()) for s, d
                            in q.items()}
        policy: FiniteDeterministicPolicy[Cell, Move] = \
            FiniteDeterministicPolicy(
                {s: max(d.items(), key=itemgetter(1))[0] for s, d in q.items()}
            )
        return (vf_dict, policy)

    def print_vf_and_policy(
        self,
        vf_dict: V[Cell],
        policy: FiniteDeterministicPolicy[Cell, Move]
    ) -> None:
        display = "%5.2f"
        display1 = "%5d"
        vf_full_dict = {
            **{s.state: display % -v for s, v in vf_dict.items()},
            **{s: display % 0.0 for s in self.terminals},
            **{s: 'X' * 5 for s in self.blocks}
        }
        print("   " + " ".join([display1 % j for j in range(self.columns)]))
        for i in range(self.rows - 1, -1, -1):
            print("%2d " % i + " ".join(vf_full_dict[(i, j)]
                                        for j in range(self.columns)))
        print()
        pol_full_dict = {
            **{s: possible_moves[policy.action_for[s]]
               for s in self.get_all_nt_states()},
            **{s: 'T' for s in self.terminals},
            **{s: 'X' for s in self.blocks}
        }
        print("   " + " ".join(["%2d" % j for j in range(self.columns)]))
        for i in range(self.rows - 1, -1, -1):
            print("%2d  " % i + "  ".join(pol_full_dict[(i, j)]
                                          for j in range(self.columns)))
        print()


if __name__ == '__main__':
    wg = WindyGrid(
        rows=5,
        columns=5,
        blocks={(0, 1), (0, 2), (0, 4), (2, 3), (3, 0), (4, 0)},
        terminals={(3, 4)},
        wind=[(0., 0.9), (0.0, 0.8), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0)],
        bump_cost=4.0
    )
    valid = wg.validate_spec()
    if valid:
        wg.print_wind_and_bumps()
        vi_vf_dict, vi_policy = wg.get_vi_vf_and_policy()
        print("Value Iteration\n")
        wg.print_vf_and_policy(
            vf_dict=vi_vf_dict,
            policy=vi_policy
        )
        mdp: FiniteMarkovDecisionProcess[Cell, Move] = wg.get_finite_mdp()

        def sample_func(state: Cell, action: Move) -> Tuple[Cell, float]:
            s, r = mdp.step(NonTerminal(state), action).sample()
            return s.state, r

        sarsa_vf_dict, sarsa_policy = wg.get_sarsa_vf_and_policy(
            states_actions_dict=wg.get_states_actions_dict(),
            sample_func=sample_func,
            episodes=10000,
            step_size=0.03
        )
        print("SARSA\n")
        wg.print_vf_and_policy(
            vf_dict=sarsa_vf_dict,
            policy=sarsa_policy
        )

        ql_vf_dict, ql_policy = wg.get_q_learning_vf_and_policy(
            states_actions_dict=wg.get_states_actions_dict(),
            sample_func=sample_func,
            episodes=10000,
            step_size=0.03,
            epsilon=0.2
        )
        print("Q-Learning\n")
        wg.print_vf_and_policy(
            vf_dict=ql_vf_dict,
            policy=ql_policy
        )

    else:
        print("Invalid Spec of Windy Grid")
```

--------------------------------------------------------------------------------

## 📄 problems/Final-Winter2021/windy_grid_outline.py {#windy-grid-outline}

**Titre**: Windy Grid Outline

**Description**: Module Windy Grid Outline

**Lignes de code**: 326

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Tuple, Callable, Sequence, Set, Mapping, Dict
from dataclasses import dataclass
from rl.distribution import Categorical, Choose
from rl.markov_process import NonTerminal
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.dynamic_programming import value_iteration_result, V
from operator import itemgetter

'''
Cell specifies (row, column) coordinate
'''
Cell = Tuple[int, int]
CellSet = Set[Cell]
Move = Tuple[int, int]
'''
WindSpec specifies a random vectical wind for each column.
Each random vertical wind is specified by a (p1, p2) pair
where p1 specifies probability of Downward Wind (could take you
one step lower in row coordinate unless prevented by a block or
boundary) and p2 specifies probability of Upward Wind (could take
you onw step higher in column coordinate unless prevented by a
block or boundary). If one bumps against a block or boundary, one
incurs a bump cost and doesn't move. The remaining probability
1- p1 - p2 corresponds to No Wind.
'''
WindSpec = Sequence[Tuple[float, float]]

possible_moves: Mapping[Move, str] = {
    (-1, 0): 'D',
    (1, 0): 'U',
    (0, -1): 'L',
    (0, 1): 'R'
}


@dataclass(frozen=True)
class WindyGrid:

    rows: int  # number of grid rows
    columns: int  # number of grid columns
    blocks: CellSet  # coordinates of block cells
    terminals: CellSet  # coordinates of goal cells
    wind: WindSpec  # spec of vertical random wind for the columns
    bump_cost: float  # cost of bumping against block or boundary

    def validate_spec(self) -> bool:
        b1 = self.rows >= 2
        b2 = self.columns >= 2
        b3 = all(0 <= r < self.rows and 0 <= c < self.columns
                 for r, c in self.blocks)
        b4 = len(self.terminals) >= 1
        b5 = all(0 <= r < self.rows and 0 <= c < self.columns and
                 (r, c) not in self.blocks for r, c in self.terminals)
        b6 = len(self.wind) == self.columns
        b7 = all(0. <= p1 <= 1. and 0. <= p2 <= 1. and p1 + p2 <= 1.
                 for p1, p2 in self.wind)
        b8 = self.bump_cost > 0.
        return all([b1, b2, b3, b4, b5, b6, b7, b8])

    def print_wind_and_bumps(self) -> None:
        for i, (d, u) in enumerate(self.wind):
            print(f"Column {i:d}: Down Prob = {d:.2f}, Up Prob = {u:.2f}")
        print(f"Bump Cost = {self.bump_cost:.2f}")
        print()

    @staticmethod
    def add_move_to_cell(cell: Cell, move: Cell) -> Cell:
        return cell[0] + move[0], cell[1] + move[1]

    def is_valid_state(self, cell: Cell) -> bool:
        '''
        checks if a cell is a valid state of the MDP
        '''
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.columns \
            and cell not in self.blocks

    def get_all_nt_states(self) -> CellSet:
        '''
        returns all the non-terminal states
        '''
        return {(i, j) for i in range(self.rows) for j in range(self.columns)
                if (i, j) not in set.union(self.blocks, self.terminals)}

    def get_actions_and_next_states(self, nt_state: Cell) \
            -> Set[Tuple[Move, Cell]]:
        '''
        given a non-terminal state, returns the set of all possible
        (action, next_state) pairs
        '''
        temp: Set[Tuple[Move, Cell]] = {(a, WindyGrid.add_move_to_cell(
            nt_state,
            a
        )) for a in possible_moves}
        return {(a, s) for a, s in temp if self.is_valid_state(s)}

    def get_transition_probabilities(self, nt_state: Cell) \
            -> Mapping[Move, Categorical[Tuple[Cell, float]]]:
        '''
        given a non-terminal state, return a dictionary whose
        keys are the valid actions (moves) from the given state
        and the corresponding values are the associated probabilities
        (following that move) of the (next_state, reward) pairs.
        The probabilities are determined from the wind probabilities
        of the column one is in after the move. Note that if one moves
        to a goal cell (terminal state), then one ends up in that
        goal cell with 100% probability (i.e., no wind exposure in a
        goal cell).
        '''
        d: Dict[Move, Categorical[Tuple[Cell, float]]] = {}
        for a, (r, c) in self.get_actions_and_next_states(nt_state):
            if (r, c) in self.terminals:
                d[a] = Categorical({((r, c), -1.): 1.})
            else:
                down_prob, up_prob = self.wind[c]
                stay_prob: float = 1. - down_prob - up_prob
                d1: Dict[Tuple[Cell, float], float] = \
                    {((r, c), -1.): stay_prob}
                if self.is_valid_state((r - 1, c)):
                    d1[((r - 1, c), -1.)] = down_prob
                if self.is_valid_state((r + 1, c)):
                    d1[((r + 1, c), -1.)] = up_prob
                d1[((r, c), -1. - self.bump_cost)] = \
                    down_prob * (1 - self.is_valid_state((r - 1, c))) + \
                    up_prob * (1 - self.is_valid_state((r + 1, c)))
                d[a] = Categorical(d1)
        return d

    def get_finite_mdp(self) -> FiniteMarkovDecisionProcess[Cell, Move]:
        '''
        returns the FiniteMarkovDecision object for this windy grid problem
        '''
        return FiniteMarkovDecisionProcess(
            {s: self.get_transition_probabilities(s) for s in
             self.get_all_nt_states()}
        )

    def get_vi_vf_and_policy(self) -> \
            Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        '''
        Performs the Value Iteration DP algorithm returning the
        Optimal Value Function (as a V[Cell]) and the Optimal Policy
        (as a FiniteDeterministicPolicy[Cell, Move])
        '''
        return value_iteration_result(self.get_finite_mdp(), gamma=1.)

    @staticmethod
    def epsilon_greedy_action(
        nt_state: Cell,
        q: Mapping[Cell, Mapping[Move, float]],
        epsilon: float
    ) -> Move:
        '''
        given a non-terminal state, a Q-Value Function (in the form of a
        {state: {action: Expected Return}} dictionary) and epislon, return
        an action sampled from the probability distribution implied by an
        epsilon-greedy policy that is derived from the Q-Value Function.
        '''
        action_values: Mapping[Move, float] = q[nt_state]
        greedy_action: Move = max(action_values.items(), key=itemgetter(1))[0]
        return Categorical(
            {a: epsilon / len(action_values) +
             (1 - epsilon if a == greedy_action else 0.)
             for a in action_values}
        ).sample()

    def get_states_actions_dict(self) -> Mapping[Cell, Set[Move]]:
        '''
        Returns a dictionary whose keys are the non-terminal states and
        the corresponding values are the set of actions for the state
        '''
        return {s: {a for a, _ in self.get_actions_and_next_states(s)}
                for s in self.get_all_nt_states()}

    def get_sarsa_vf_and_policy(
        self,
        states_actions_dict: Mapping[Cell, Set[Move]],
        sample_func: Callable[[Cell, Move], Tuple[Cell, float]],
        episodes: int = 10000,
        step_size: float = 0.01
    ) -> Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        '''
        states_actions_dict gives us the set of possible moves from
        a non-terminal cell.
        sample_func is a function with two inputs: state and action,
        and with output as a sampled pair of (next_state, reward).
        '''
        q: Dict[Cell, Dict[Move, float]] = \
            {s: {a: 0. for a in actions} for s, actions in
             states_actions_dict.items()}
        nt_states: CellSet = {s for s in q}
        uniform_states: Choose[Cell] = Choose(nt_states)
        for episode_num in range(episodes):
            epsilon: float = 1.0 / (episode_num + 1)
            state: Cell = uniform_states.sample()
            '''
            write your code here
            update the dictionary q initialized above according
            to the SARSA algorithm's Q-Value Function updates.
            '''

        vf_dict: V[Cell] = {NonTerminal(s): max(d.values()) for s, d
                            in q.items()}
        policy: FiniteDeterministicPolicy[Cell, Move] = \
            FiniteDeterministicPolicy(
                {s: max(d.items(), key=itemgetter(1))[0] for s, d in q.items()}
            )
        return vf_dict, policy

    def get_q_learning_vf_and_policy(
        self,
        states_actions_dict: Mapping[Cell, Set[Move]],
        sample_func: Callable[[Cell, Move], Tuple[Cell, float]],
        episodes: int = 10000,
        step_size: float = 0.01,
        epsilon: float = 0.1
    ) -> Tuple[V[Cell], FiniteDeterministicPolicy[Cell, Move]]:
        '''
        states_actions_dict gives us the set of possible moves from
        a non-block cell.
        sample_func is a function with two inputs: state and action,
        and with output as a sampled pair of (next_state, reward).
        '''
        q: Dict[Cell, Dict[Move, float]] = \
            {s: {a: 0. for a in actions} for s, actions in
             states_actions_dict.items()}
        nt_states: CellSet = {s for s in q}
        uniform_states: Choose[Cell] = Choose(nt_states)
        for episode_num in range(episodes):
            state: Cell = uniform_states.sample()
            '''
            write your code here
            update the dictionary q initialized above according
            to the Q-learning algorithm's Q-Value Function updates.
            '''

        vf_dict: V[Cell] = {NonTerminal(s): max(d.values()) for s, d
                            in q.items()}
        policy: FiniteDeterministicPolicy[Cell, Move] = \
            FiniteDeterministicPolicy(
                {s: max(d.items(), key=itemgetter(1))[0] for s, d in q.items()}
            )
        return (vf_dict, policy)

    def print_vf_and_policy(
        self,
        vf_dict: V[Cell],
        policy: FiniteDeterministicPolicy[Cell, Move]
    ) -> None:
        display = "%5.2f"
        display1 = "%5d"
        vf_full_dict = {
            **{s.state: display % -v for s, v in vf_dict.items()},
            **{s: display % 0.0 for s in self.terminals},
            **{s: 'X' * 5 for s in self.blocks}
        }
        print("   " + " ".join([display1 % j for j in range(self.columns)]))
        for i in range(self.rows - 1, -1, -1):
            print("%2d " % i + " ".join(vf_full_dict[(i, j)]
                                        for j in range(self.columns)))
        print()
        pol_full_dict = {
            **{s: possible_moves[policy.action_for[s]]
               for s in self.get_all_nt_states()},
            **{s: 'T' for s in self.terminals},
            **{s: 'X' for s in self.blocks}
        }
        print("   " + " ".join(["%2d" % j for j in range(self.columns)]))
        for i in range(self.rows - 1, -1, -1):
            print("%2d  " % i + "  ".join(pol_full_dict[(i, j)]
                                          for j in range(self.columns)))
        print()


if __name__ == '__main__':
    wg = WindyGrid(
        rows=5,
        columns=5,
        blocks={(0, 1), (0, 2), (0, 4), (2, 3), (3, 0), (4, 0)},
        terminals={(3, 4)},
        wind=[(0., 0.9), (0.0, 0.8), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0)],
        bump_cost=4.0
    )
    valid = wg.validate_spec()
    if valid:
        wg.print_wind_and_bumps()
        vi_vf_dict, vi_policy = wg.get_vi_vf_and_policy()
        print("Value Iteration\n")
        wg.print_vf_and_policy(
            vf_dict=vi_vf_dict,
            policy=vi_policy
        )
        mdp: FiniteMarkovDecisionProcess[Cell, Move] = wg.get_finite_mdp()

        def sample_func(state: Cell, action: Move) -> Tuple[Cell, float]:
            s, r = mdp.step(NonTerminal(state), action).sample()
            return s.state, r

        sarsa_vf_dict, sarsa_policy = wg.get_sarsa_vf_and_policy(
            states_actions_dict=wg.get_states_actions_dict(),
            sample_func=sample_func,
            episodes=10000,
            step_size=0.03
        )
        print("SARSA\n")
        wg.print_vf_and_policy(
            vf_dict=sarsa_vf_dict,
            policy=sarsa_policy
        )

        ql_vf_dict, ql_policy = wg.get_q_learning_vf_and_policy(
            states_actions_dict=wg.get_states_actions_dict(),
            sample_func=sample_func,
            episodes=10000,
            step_size=0.03,
            epsilon=0.2
        )
        print("Q-Learning\n")
        wg.print_vf_and_policy(
            vf_dict=ql_vf_dict,
            policy=ql_policy
        )

    else:
        print("Invalid Spec of Windy Grid")
```

--------------------------------------------------------------------------------

## 📄 problems/Midterm-Winter2021/career_optimization.py {#career-optimization}

**Titre**: Career Optimization

**Description**: Module Career Optimization

**Lignes de code**: 94

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Tuple, Mapping, Dict, Sequence, Iterable
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.dynamic_programming import value_iteration_result
from rl.distribution import Categorical
from scipy.stats import poisson

IntPair = Tuple[int, int]
CareerDecisionsMap = Mapping[int, Mapping[
    IntPair,
    Categorical[Tuple[int, float]]
]]


class CareerOptimization(FiniteMarkovDecisionProcess[int, IntPair]):

    def __init__(
        self,
        hours: int,
        wage_cap: int,
        alpha: float,
        beta: float
    ):
        self.hours = hours
        self.wage_cap = wage_cap
        self.alpha = alpha
        self.beta = beta
        super().__init__(self.get_transitions())

    def get_transitions(self) -> CareerDecisionsMap:
        d: Dict[int, Mapping[IntPair, Categorical[Tuple[int, float]]]] = {}
        for w in range(1, self.wage_cap + 1):
            d1: Dict[IntPair, Categorical[Tuple[int, float]]] = {}
            for s in range(self.hours + 1):
                for t in range(self.hours + 1 - s):
                    pd = poisson(self.alpha * t)
                    prob: float = self.beta * s / self.hours
                    r: float = w * (self.hours - s - t)
                    same_prob: float = (1 - prob) * pd.pmf(0)
                    sr_probs: Dict[Tuple[int, float], float] = {}
                    if w == self.wage_cap:
                        sr_probs[(w, r)] = 1.
                    elif w == self.wage_cap - 1:
                        sr_probs[(w, r)] = same_prob
                        sr_probs[(w + 1, r)] = 1 - same_prob
                    else:
                        sr_probs[(w, r)] = same_prob
                        sr_probs[(w + 1, r)] = prob * pd.pmf(0) + pd.pmf(1)
                        for w1 in range(w + 2, self.wage_cap):
                            sr_probs[(w1, r)] = pd.pmf(w1 - w)
                        sr_probs[(self.wage_cap, r)] = \
                            1 - pd.cdf(self.wage_cap - w - 1)
                    d1[(s, t)] = Categorical(sr_probs)
            d[w] = d1
        return d


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from pprint import pprint
    hours: int = 10
    wage_cap: int = 30
    alpha: float = 0.08
    beta: float = 0.82
    gamma: float = 0.95

    co: CareerOptimization = CareerOptimization(
        hours=hours,
        wage_cap=wage_cap,
        alpha=alpha,
        beta=beta
    )

    _, opt_det_policy = value_iteration_result(co, gamma=gamma)
    wages: Iterable[int] = range(1, co.wage_cap + 1)
    opt_actions: Mapping[int, Tuple[int, int]] = \
        {w: opt_det_policy.action_for[w]
         for w in wages}
    searching: Sequence[int] = [s for _, (s, _) in opt_actions.items()]
    learning: Sequence[int] = [l for _, (_, l) in opt_actions.items()]
    working: Sequence[int] = [co.hours - s - l for _, (s, l) in
                              opt_actions.items()]
    pprint(opt_actions)
    plt.xticks(wages)
    p1 = plt.bar(wages, searching, color='red')
    p2 = plt.bar(wages, learning, color='blue')
    p3 = plt.bar(wages, working, color='green')
    plt.legend((p1[0], p2[0], p3[0]), ('Job-Searching', 'Learning', 'Working'))
    plt.grid(axis='y')
    plt.xlabel("Hourly Wage Level")
    plt.ylabel("Hours Spent")
    plt.title("Career Optimization")
    plt.show()
```

--------------------------------------------------------------------------------

## 📄 problems/Midterm-Winter2021/grid_maze.py {#grid-maze}

**Titre**: Grid Maze

**Description**: Module Grid Maze

**Lignes de code**: 18

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
SPACE = 'SPACE'
BLOCK = 'BLOCK'
GOAL = 'GOAL'

maze_grid = {(0, 0): SPACE, (0, 1): BLOCK, (0, 2): SPACE, (0, 3): SPACE, (0, 4): SPACE, 
             (0, 5): SPACE, (0, 6): SPACE, (0, 7): SPACE, (1, 0): SPACE, (1, 1): BLOCK,
             (1, 2): BLOCK, (1, 3): SPACE, (1, 4): BLOCK, (1, 5): BLOCK, (1, 6): BLOCK, 
             (1, 7): BLOCK, (2, 0): SPACE, (2, 1): BLOCK, (2, 2): SPACE, (2, 3): SPACE, 
             (2, 4): SPACE, (2, 5): SPACE, (2, 6): BLOCK, (2, 7): SPACE, (3, 0): SPACE, 
             (3, 1): SPACE, (3, 2): SPACE, (3, 3): BLOCK, (3, 4): BLOCK, (3, 5): SPACE, 
             (3, 6): BLOCK, (3, 7): SPACE, (4, 0): SPACE, (4, 1): BLOCK, (4, 2): SPACE, 
             (4, 3): BLOCK, (4, 4): SPACE, (4, 5): SPACE, (4, 6): SPACE, (4, 7): SPACE, 
             (5, 0): BLOCK, (5, 1): BLOCK, (5, 2): SPACE, (5, 3): BLOCK, (5, 4): SPACE, 
             (5, 5): BLOCK, (5, 6): SPACE, (5, 7): BLOCK, (6, 0): SPACE, (6, 1): BLOCK, 
             (6, 2): BLOCK, (6, 3): BLOCK, (6, 4): SPACE, (6, 5): BLOCK, (6, 6): SPACE, 
             (6, 7): SPACE, (7, 0): SPACE, (7, 1): SPACE, (7, 2): SPACE, (7, 3): SPACE, 
             (7, 4): SPACE, (7, 5): BLOCK, (7, 6): BLOCK, (7, 7): GOAL}
```

--------------------------------------------------------------------------------

# PARTIE 9 - TESTS UNITAIRES

================================================================================

## 📄 chapter10/test_lambda_return.py {#test-lambda-return}

**Titre**: Test Lambda Return

**Description**: Module Test Lambda Return

**Lignes de code**: 68

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Mapping, Iterator, Iterable
from rl.distribution import Choose
from rl.function_approx import Tabular, learning_rate_schedule
from rl.markov_process import NonTerminal, TransitionStep
from rl.chapter2.simple_inventory_mrp import InventoryState
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.td_lambda import lambda_return_prediction
import rl.iterate as iterate
import itertools
from pprint import pprint


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

si_mrp: SimpleInventoryMRPFinite = SimpleInventoryMRPFinite(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)
initial_vf_dict: Mapping[NonTerminal[InventoryState], float] = \
    {s: 0. for s in si_mrp.non_terminal_states}

gamma: float = 0.9
lambda_param = 0.3
num_episodes = 10000

episode_length: int = 100
initial_learning_rate: float = 0.03
half_life: float = 1000.0
exponent: float = 0.5

approx_0: Tabular[NonTerminal[InventoryState]] = Tabular(
    values_map=initial_vf_dict,
    count_to_weight_func=learning_rate_schedule(
        initial_learning_rate=initial_learning_rate,
        half_life=half_life,
        exponent=exponent
    )
)

episodes: Iterable[Iterable[TransitionStep[InventoryState]]] = \
    si_mrp.reward_traces(Choose(si_mrp.non_terminal_states))
traces: Iterable[Iterable[TransitionStep[InventoryState]]] = \
        (itertools.islice(episode, episode_length) for episode in episodes)

vf_iter: Iterator[Tabular[NonTerminal[InventoryState]]] = \
    lambda_return_prediction(
        traces=traces,
        approx_0=approx_0,
        γ=gamma,
        lambd=lambda_param
    )

vf: Tabular[NonTerminal[InventoryState]] = \
    iterate.last(itertools.islice(vf_iter, num_episodes))

pprint(vf.values_map)
si_mrp.display_value_function(gamma=gamma)





```

--------------------------------------------------------------------------------

## 📄 chapter12/test_batch_rl_prediction.py {#test-batch-rl-prediction}

**Titre**: Test Batch Rl Prediction

**Description**: Module Test Batch Rl Prediction

**Lignes de code**: 106

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Tuple, Iterator
from rl.markov_process import TransitionStep, NonTerminal, Terminal
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.function_approx import Tabular, learning_rate_schedule
from rl.monte_carlo import batch_mc_prediction
from rl.td import td_prediction, batch_td_prediction
from rl.experience_replay import ExperienceReplayMemory
import itertools
import rl.iterate as iterate

given_data: Sequence[Sequence[Tuple[str, float]]] = [
    [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
    [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
    [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
    [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
    [('B', 8.), ('B', 2.)]
]

gamma: float = 0.9

fixed_traces: Sequence[Sequence[TransitionStep[str]]] = \
    [[TransitionStep(
        state=NonTerminal(s),
        reward=r,
        next_state=NonTerminal(trace[i+1][0])
        if i < len(trace) - 1 else Terminal('T')
    ) for i, (s, r) in enumerate(trace)] for trace in given_data]

a: NonTerminal[str] = NonTerminal('A')
b: NonTerminal[str] = NonTerminal('B')

# fa: LinearFunctionApprox[NonTerminal[str]] = LinearFunctionApprox.create(
#     feature_functions=[
#         lambda x: 1.0 if x == a else 0.,
#         lambda y: 1.0 if y == b else 0.
#     ],
#     adam_gradient=AdamGradient(
#         learning_rate=0.1,
#         decay1=0.9,
#         decay2=0.999
#     ),
#     direct_solve=False
# )

mc_fa: Tabular[NonTerminal[str]] = Tabular()

mc_vf: ValueFunctionApprox[str] = batch_mc_prediction(
    fixed_traces,
    mc_fa,
    gamma
)

print("Result of Batch MC Prediction")
print("V[A] = %.3f" % mc_vf(a))
print("V[B] = %.3f" % mc_vf(b))

fixed_transitions: Sequence[TransitionStep[str]] = \
    [t for tr in fixed_traces for t in tr]

td_fa: Tabular[NonTerminal[str]] = Tabular(
    count_to_weight_func=learning_rate_schedule(
        initial_learning_rate=0.1,
        half_life=10000,
        exponent=0.5
    )
)

exp_replay_memory: ExperienceReplayMemory[TransitionStep[str]] = \
    ExperienceReplayMemory()

replay: Iterator[Sequence[TransitionStep[str]]] = \
    exp_replay_memory.replay(fixed_transitions, 1)


def replay_transitions(replay=replay) -> Iterator[TransitionStep[str]]:
    while True:
        yield next(replay)[0]


num_iterations: int = 100000

td1_vf: ValueFunctionApprox[str] = iterate.last(
    itertools.islice(
        td_prediction(
            replay_transitions(),
            td_fa,
            gamma
        ),
        num_iterations
    )
)

print("Result of Batch TD1 Prediction")
print("V[A] = %.3f" % td1_vf(a))
print("V[B] = %.3f" % td1_vf(b))

td2_vf: ValueFunctionApprox[str] = batch_td_prediction(
    fixed_transitions,
    td_fa,
    gamma
)

print("Result of Batch TD2 Prediction")
print("V[A] = %.3f" % td2_vf(a))
print("V[B] = %.3f" % td2_vf(b))
```

--------------------------------------------------------------------------------

## 📄 chapter12/test_lspi.py {#test-lspi}

**Titre**: Test Lspi

**Description**: Module Test Lspi

**Lignes de code**: 1

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python

```

--------------------------------------------------------------------------------

## 📄 chapter12/test_q_learning_experience_replay.py {#test-q-learning-experience-replay}

**Titre**: Experience Replay

**Description**: Buffer de replay pour apprentissage off-policy

**Lignes de code**: 75

**Concepts clés**:
- ExperienceReplayBuffer - Stocke (s,a,r,s') transitions
- Sampling uniforme ou prioritaire
- Brise les corrélations temporelles

**🎯 Utilisation HelixOne**: Stabilisation de l'apprentissage

### Code Source Complet

```python
from typing import Iterator
from rl.distribution import Choose
from rl.function_approx import Tabular, learning_rate_schedule
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, \
    InventoryState
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.monte_carlo import epsilon_greedy_policy
from rl.td import q_learning_experience_replay
from rl.dynamic_programming import value_iteration_result
import rl.iterate as iterate
import itertools
from pprint import pprint


capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

si_mdp: SimpleInventoryMDPCap = SimpleInventoryMDPCap(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)

gamma: float = 0.9
epsilon: float = 0.3

initial_learning_rate: float = 0.1
learning_rate_half_life: float = 1000
learning_rate_exponent: float = 0.5

episode_length: int = 100
mini_batch_size: int = 1000
time_decay_half_life: float = 3000
num_updates: int = 10000

q_iter: Iterator[QValueFunctionApprox[InventoryState, int]] = \
    q_learning_experience_replay(
        mdp=si_mdp,
        policy_from_q=lambda f, m: epsilon_greedy_policy(
            q=f,
            mdp=m,
            ϵ=epsilon
        ),
        states=Choose(si_mdp.non_terminal_states),
        approx_0=Tabular(
            count_to_weight_func=learning_rate_schedule(
                initial_learning_rate=initial_learning_rate,
                half_life=learning_rate_half_life,
                exponent=learning_rate_exponent
            )
        ),
        γ=gamma,
        max_episode_length=episode_length,
        mini_batch_size=mini_batch_size,
        weights_decay_half_life=time_decay_half_life
    )

qvf: QValueFunctionApprox[InventoryState, int] = iterate.last(
    itertools.islice(
        q_iter,
        num_updates
    )
)
vf, pol = get_vf_and_policy_from_qvf(mdp=si_mdp, qvf=qvf)
pprint(vf)
print(pol)

true_vf, true_pol = value_iteration_result(mdp=si_mdp, gamma=gamma)
pprint(true_vf)
print(true_pol)
```

--------------------------------------------------------------------------------

## 📄 test_approx_dp_clearance.py {#test-approx-dp-clearance}

**Titre**: Test Approx Dp Clearance

**Description**: Module Test Approx Dp Clearance

**Lignes de code**: 113

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
import unittest

import numpy as np

from rl.distribution import Choose
from rl.finite_horizon import (
    unwrap_finite_horizon_MRP, finite_horizon_MRP, evaluate,
    unwrap_finite_horizon_MDP, finite_horizon_MDP, optimal_vf_and_policy)
from rl.function_approx import Dynamic, Tabular
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy

from rl.chapter4.clearance_pricing_mdp import ClearancePricingMDP

from rl.approximate_dynamic_programming import (
    backward_evaluate_finite, backward_evaluate,
    back_opt_vf_and_policy_finite, back_opt_vf_and_policy)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        ii = 10
        self.steps = 6
        pairs = [(1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5)]
        self.cp: ClearancePricingMDP = ClearancePricingMDP(
            initial_inventory=ii,
            time_steps=self.steps,
            price_lambda_pairs=pairs
        )

        def policy_func(x: int) -> int:
            return 0 if x < 2 else (1 if x < 5 else (2 if x < 8 else 3))

        stationary_policy: FiniteDeterministicPolicy[int, int] = \
            FiniteDeterministicPolicy(
                {s: policy_func(s) for s in range(ii + 1)}
            )

        self.single_step_mrp: FiniteMarkovRewardProcess[int] = \
            self.cp.single_step_mdp.apply_finite_policy(stationary_policy)

        self.mrp_seq = unwrap_finite_horizon_MRP(
            finite_horizon_MRP(self.single_step_mrp, self.steps)
        )

        self.single_step_mdp: FiniteMarkovDecisionProcess[int, int] = \
            self.cp.single_step_mdp

        self.mdp_seq = unwrap_finite_horizon_MDP(
            finite_horizon_MDP(self.single_step_mdp, self.steps)
        )

    def test_evaluate_mrp(self):
        vf = evaluate(self.mrp_seq, 1.)
        states = self.single_step_mrp.non_terminal_states
        fa_dynamic = Dynamic({s: 0.0 for s in states})
        fa_tabular = Tabular()
        distribution = Choose(states)
        approx_vf_finite = backward_evaluate_finite(
            [(self.mrp_seq[i], fa_dynamic) for i in range(self.steps)],
            1.
        )
        approx_vf = backward_evaluate(
            [(self.single_step_mrp, fa_tabular, distribution)
             for _ in range(self.steps)],
            1.,
            num_state_samples=120,
            error_tolerance=0.01
        )

        for t, (v1, v2, v3) in enumerate(zip(
                vf,
                approx_vf_finite,
                approx_vf
        )):
            states = self.mrp_seq[t].keys()
            v1_arr = np.array([v1[s] for s in states])
            v2_arr = v2.evaluate(states)
            v3_arr = v3.evaluate(states)
            self.assertLess(max(abs(v1_arr - v2_arr)), 0.001)
            self.assertLess(max(abs(v1_arr - v3_arr)), 1.0)

    def test_value_iteration(self):
        vpstar = optimal_vf_and_policy(self.mdp_seq, 1.)
        states = self.single_step_mdp.non_terminal_states
        fa_dynamic = Dynamic({s: 0.0 for s in states})
        fa_tabular = Tabular()
        distribution = Choose(states)
        approx_vpstar_finite = back_opt_vf_and_policy_finite(
            [(self.mdp_seq[i], fa_dynamic) for i in range(self.steps)],
            1.
        )
        approx_vpstar = back_opt_vf_and_policy(
            [(self.single_step_mdp, fa_tabular, distribution)
             for _ in range(self.steps)],
            1.,
            num_state_samples=120,
            error_tolerance=0.01
        )

        for t, ((v1, _), (v2, _), (v3, _)) in enumerate(zip(
                vpstar,
                approx_vpstar_finite,
                approx_vpstar
        )):
            states = self.mdp_seq[t].keys()
            v1_arr = np.array([v1[s] for s in states])
            v2_arr = v2.evaluate(states)
            v3_arr = v3.evaluate(states)
            self.assertLess(max(abs(v1_arr - v2_arr)), 0.001)
            self.assertLess(max(abs(v1_arr - v3_arr)), 1.0)
```

--------------------------------------------------------------------------------

## 📄 test_approx_dp_inventory.py {#test-approx-dp-inventory}

**Titre**: Test Approx Dp Inventory

**Description**: Module Test Approx Dp Inventory

**Lignes de code**: 120

**🎯 Utilisation HelixOne**: À analyser pour intégration

### Code Source Complet

```python
from typing import Sequence, Mapping
import unittest

import numpy as np

from rl.approximate_dynamic_programming import (
    evaluate_finite_mrp, evaluate_mrp, value_iteration_finite, value_iteration)
from rl.dynamic_programming import value_iteration_result
from rl.distribution import Choose
from rl.function_approx import Dynamic
import rl.iterate as iterate
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy

from rl.chapter3.simple_inventory_mdp_cap import (InventoryState,
                                                  SimpleInventoryMDPCap)


# @unittest.skip("Explanation (ie test is too slow)")
class TestEvaluate(unittest.TestCase):
    def setUp(self):
        user_capacity = 2
        user_poisson_lambda = 1.0
        user_holding_cost = 1.0
        user_stockout_cost = 10.0

        self.gamma = 0.9

        self.si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
            SimpleInventoryMDPCap(
                capacity=user_capacity,
                poisson_lambda=user_poisson_lambda,
                holding_cost=user_holding_cost,
                stockout_cost=user_stockout_cost
            )

        self.fdp: FiniteDeterministicPolicy[InventoryState, int] = \
            FiniteDeterministicPolicy(
                {InventoryState(alpha, beta): user_capacity - (alpha + beta)
                 for alpha in range(user_capacity + 1)
                 for beta in range(user_capacity + 1 - alpha)}
        )

        self.implied_mrp: FiniteMarkovRewardProcess[InventoryState] =\
            self.si_mdp.apply_finite_policy(self.fdp)

        self.states: Sequence[NonTerminal[InventoryState]] = \
            self.implied_mrp.non_terminal_states

    def test_evaluate_mrp(self):
        mrp_vf1: np.ndarray = self.implied_mrp.get_value_function_vec(
            self.gamma
        )
        # print({s: mrp_vf1[i] for i, s in enumerate(self.states)})

        fa = Dynamic({s: 0.0 for s in self.states})
        mrp_finite_fa = iterate.converged(
            evaluate_finite_mrp(
                self.implied_mrp,
                self.gamma,
                fa
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        mrp_vf2: np.ndarray = mrp_finite_fa.evaluate(self.states)

        self.assertLess(max(abs(mrp_vf1 - mrp_vf2)), 0.001)

        mrp_fa = iterate.converged(
            evaluate_mrp(
                self.implied_mrp,
                self.gamma,
                fa,
                Choose(self.states),
                num_state_samples=30
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )
        # print(mrp_fa.values_map)
        mrp_vf3: np.ndarray = mrp_fa.evaluate(self.states)
        self.assertLess(max(abs(mrp_vf1 - mrp_vf3)), 0.001)

    def test_value_iteration(self):
        mdp_map: Mapping[NonTerminal[InventoryState], float] = value_iteration_result(
            self.si_mdp,
            self.gamma
        )[0]
        # print(mdp_map)
        mdp_vf1: np.ndarray = np.array([mdp_map[s] for s in self.states])

        fa = Dynamic({s: 0.0 for s in self.states})
        mdp_finite_fa = iterate.converged(
            value_iteration_finite(
                self.si_mdp,
                self.gamma,
                fa
            ),
            done=lambda a, b: a.within(b, 1e-5)
        )
        # print(mdp_finite_fa.values_map)
        mdp_vf2: np.ndarray = mdp_finite_fa.evaluate(self.states)

        self.assertLess(max(abs(mdp_vf1 - mdp_vf2)), 0.01)

        mdp_fa = iterate.converged(
            value_iteration(
                self.si_mdp,
                self.gamma,
                fa,
                Choose(self.states),
                num_state_samples=30
            ),
            done=lambda a, b: a.within(b, 1e-5)
        )
        # print(mdp_fa.values_map)
        mdp_vf3: np.ndarray = mdp_fa.evaluate(self.states)
        self.assertLess(max(abs(mdp_vf1 - mdp_vf3)), 0.01)
```

--------------------------------------------------------------------------------

## 📄 test_approximate_dynamic_programming.py {#test-approximate-dynamic-programming}

**Titre**: Programmation Dynamique Exacte

**Description**: Algorithmes DP classiques pour espaces finis

**Lignes de code**: 112

**Concepts clés**:
- evaluate_mrp - Évaluation de politique (résout V = R + γPV)
- greedy_policy_from_vf - Amélioration de politique
- policy_iteration - Itération de politique complète
- value_iteration - Itération de valeur (Bellman optimality)
- Convergence garantie pour espaces finis

**🎯 Utilisation HelixOne**: Baseline pour petits problèmes

### Code Source Complet

```python
from numpy.testing import assert_allclose
import unittest

from rl.approximate_dynamic_programming import (evaluate_mrp,
                                                evaluate_finite_mrp)
from rl.distribution import Categorical, Choose
from rl.finite_horizon import (finite_horizon_MRP, evaluate,
                               unwrap_finite_horizon_MRP, WithTime)
from rl.function_approx import Dynamic
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
import rl.iterate as iterate


class FlipFlop(FiniteMarkovRewardProcess[bool]):
    '''A version of FlipFlop implemented with the FiniteMarkovProcess
    machinery.

    '''

    def __init__(self, p: float):
        transition_reward_map = {
            b: Categorical({(not b, 2.0): p, (b, 1.0): 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_reward_map)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.finite_flip_flop = FlipFlop(0.7)

    def test_evaluate_finite_mrp(self):
        start = Dynamic({s: 0.0 for s in
                         self.finite_flip_flop.non_terminal_states})
        v = iterate.converged(
            evaluate_finite_mrp(
                self.finite_flip_flop,
                γ=0.99,
                approx_0=start
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        self.assertEqual(len(v.values_map), 2)

        for s in v.values_map:
            self.assertLess(abs(v(s) - 170), 0.1)

    def test_evaluate_mrp(self):
        start = Dynamic({s: 0.0 for s in
                         self.finite_flip_flop.non_terminal_states})

        v = iterate.converged(
            evaluate_mrp(
                self.finite_flip_flop,
                γ=0.99,
                approx_0=start,
                non_terminal_states_distribution=Choose(
                    self.finite_flip_flop.non_terminal_states
                ),
                num_state_samples=5
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        self.assertEqual(len(v.values_map), 2)

        for s in v.values_map:
            self.assertLess(abs(v(s) - 170), 1.0)

        v_finite = iterate.converged(
            evaluate_finite_mrp(
                self.finite_flip_flop,
                γ=0.99,
                approx_0=start
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        assert_allclose(v.evaluate([NonTerminal(True), NonTerminal(False)]),
                        v_finite.evaluate([NonTerminal(True),
                                           NonTerminal(False)]),
                        rtol=0.01)

    def test_compare_to_backward_induction(self):
        finite_horizon = finite_horizon_MRP(self.finite_flip_flop, 10)

        start = Dynamic({s: 0.0 for s in finite_horizon.non_terminal_states})
        v = iterate.converged(
            evaluate_finite_mrp(
                finite_horizon,
                γ=1,
                approx_0=start
            ),
            done=lambda a, b: a.within(b, 1e-4)
        )

        self.assertEqual(len(v.values_map), 20)

        finite_v =\
            list(evaluate(unwrap_finite_horizon_MRP(finite_horizon), gamma=1))

        for time in range(10):
            self.assertAlmostEqual(
                v(NonTerminal(WithTime(state=True, time=time))),
                finite_v[time][NonTerminal(True)]
            )
            self.assertAlmostEqual(
                v(NonTerminal(WithTime(state=False, time=time))),
                finite_v[time][NonTerminal(False)]
            )
```

--------------------------------------------------------------------------------

## 📄 test_distribution.py {#test-distribution}

**Titre**: Distributions de Probabilité

**Description**: Module fondamental définissant les distributions de probabilité

**Lignes de code**: 159

**Concepts clés**:
- Distribution[A] - Classe abstraite de base pour échantillonnage
- SampledDistribution - Distribution définie par fonction sampler
- FiniteDistribution - Distribution discrète avec table de probabilités
- Categorical - Sélection parmi outcomes discrets avec probabilités
- Gaussian, Poisson, Beta, Gamma, Uniform - Distributions standard
- Bernoulli, Constant, Choose, Range - Distributions utilitaires

**🎯 Utilisation HelixOne**: Base pour modéliser l'incertitude des prix, volumes, et transitions

### Code Source Complet

```python
from collections import Counter
import unittest

from rl.distribution import (Bernoulli, Categorical, Choose, Constant,
                             Gaussian, SampledDistribution, Uniform)


def assert_almost_equal(test_case, dist_a, dist_b):
    '''Check that two distributions are "almost" equal (ie ignore small
    differences in floating point numbers when comparing them).

    '''
    a_table = dist_a.table()
    b_table = dist_b.table()

    assert a_table.keys() == b_table.keys()

    for outcome in a_table:
        test_case.assertAlmostEqual(a_table[outcome], b_table[outcome])


class TestDistribution(unittest.TestCase):
    def setUp(self):
        self.finite = Choose(range(0, 6))
        self.sampled = SampledDistribution(
            lambda: self.finite.sample(),
            100000
        )

    def test_expectation(self):
        expected_finite = self.finite.expectation(lambda x: x)
        expected_sampled = self.sampled.expectation(lambda x: x)
        self.assertLess(abs(expected_finite - expected_sampled), 0.02)

    def test_sample_n(self):
        samples = self.sampled.sample_n(10)
        self.assertEqual(len(samples), 10)
        self.assertTrue(all(0 <= s < 6 for s in samples))


class TestUniform(unittest.TestCase):
    def setUp(self):
        self.uniform = Uniform(100000)

    def test_expectation(self):
        expectation = self.uniform.expectation(lambda x: x)
        self.assertLess(abs(expectation - 0.5), 0.01)


class TestGaussian(unittest.TestCase):
    def setUp(self):
        self.unit = Gaussian(1.0, 1.0, 100000)
        self.large = Gaussian(10.0, 30.0, 100000)

    def test_expectation(self):
        unit_expectation = self.unit.expectation(lambda x: x)
        self.assertLess(abs(unit_expectation - 1.0), 0.1)

        large_expectation = self.large.expectation(lambda x: x)
        self.assertLess(abs(large_expectation - 10), 0.3)


class TestFiniteDistribution(unittest.TestCase):
    def setUp(self):
        self.die = Choose({1, 2, 3, 4, 5, 6})

        self.ragged = Categorical({0: 0.9, 1: 0.05, 2: 0.025, 3: 0.025})

    def test_map(self):
        plusOne = self.die.map(lambda x: x + 1)
        assert_almost_equal(self, plusOne, Choose({2, 3, 4, 5, 6, 7}))

        evenOdd = self.die.map(lambda x: x % 2 == 0)
        assert_almost_equal(self, evenOdd, Choose({True, False}))

        greaterThan4 = self.die.map(lambda x: x > 4)
        assert_almost_equal(self, greaterThan4,
                            Categorical({True: 1/3, False: 2/3}))

    def test_expectation(self):
        self.assertAlmostEqual(self.die.expectation(float), 3.5)

        even = self.die.map(lambda n: n % 2 == 0)
        self.assertAlmostEqual(even.expectation(float), 0.5)

        self.assertAlmostEqual(self.ragged.expectation(float), 0.175)


class TestConstant(unittest.TestCase):
    def test_constant(self):
        assert_almost_equal(self, Constant(42), Categorical({42: 1.}))
        self.assertAlmostEqual(Constant(42).probability(42), 1.)
        self.assertAlmostEqual(Constant(42).probability(37), 0.)


class TestBernoulli(unittest.TestCase):
    def setUp(self):
        self.fair = Bernoulli(0.5)
        self.unfair = Bernoulli(0.3)

    def test_constant(self):
        assert_almost_equal(
            self, self.fair, Categorical({True: 0.5, False: 0.5}))
        self.assertAlmostEqual(self.fair.probability(True), 0.5)
        self.assertAlmostEqual(self.fair.probability(False), 0.5)

        assert_almost_equal(self, self.unfair,
                            Categorical({True: 0.3, False: 0.7}))
        self.assertAlmostEqual(self.unfair.probability(True), 0.3)
        self.assertAlmostEqual(self.unfair.probability(False), 0.7)


class TestChoose(unittest.TestCase):
    def setUp(self):
        self.one = Choose({1})
        self.six = Choose({1, 2, 3, 4, 5, 6})
        self.repeated = Choose([1,1,1,2])

    def test_choose(self):
        assert_almost_equal(self, self.one, Constant(1))
        self.assertAlmostEqual(self.one.probability(1), 1.)
        self.assertAlmostEqual(self.one.probability(0), 0.)

        categorical_six = Categorical({x: 1/6 for x in range(1, 7)})
        assert_almost_equal(self, self.six, categorical_six)
        self.assertAlmostEqual(self.six.probability(1), 1/6)
        self.assertAlmostEqual(self.six.probability(0), 0.)

    def test_repeated(self):
        counts = Counter(self.repeated.sample_n(1000))
        self.assertLess(abs(counts[1] - 750), 50)
        self.assertLess(abs(counts[2] - 250), 50)

        table = self.repeated.table()
        self.assertAlmostEqual(table[1], 0.75)
        self.assertAlmostEqual(table[2], 0.25)

        counts = Counter(self.repeated.sample_n(1000))
        self.assertLess(abs(counts[1] - 750), 50)
        self.assertLess(abs(counts[2] - 250), 50)


class TestCategorical(unittest.TestCase):
    def setUp(self):
        self.normalized = Categorical({True: 0.3, False: 0.7})
        self.unnormalized = Categorical({True: 3., False: 7.})

    def test_categorical(self):
        assert_almost_equal(self, self.normalized, Bernoulli(0.3))
        self.assertAlmostEqual(self.normalized.probability(True), 0.3)
        self.assertAlmostEqual(self.normalized.probability(False), 0.7)
        self.assertAlmostEqual(self.normalized.probability(None), 0.)

    def test_normalization(self):
        assert_almost_equal(self, self.unnormalized, self.normalized)
        self.assertAlmostEqual(self.unnormalized.probability(True), 0.3)
        self.assertAlmostEqual(self.unnormalized.probability(False), 0.7)
        self.assertAlmostEqual(self.unnormalized.probability(None), 0.)
```

--------------------------------------------------------------------------------

## 📄 test_dynamic_programming.py {#test-dynamic-programming}

**Titre**: Programmation Dynamique Exacte

**Description**: Algorithmes DP classiques pour espaces finis

**Lignes de code**: 54

**Concepts clés**:
- evaluate_mrp - Évaluation de politique (résout V = R + γPV)
- greedy_policy_from_vf - Amélioration de politique
- policy_iteration - Itération de politique complète
- value_iteration - Itération de valeur (Bellman optimality)
- Convergence garantie pour espaces finis

**🎯 Utilisation HelixOne**: Baseline pour petits problèmes

### Code Source Complet

```python
import unittest

from rl.distribution import Categorical
from rl.dynamic_programming import evaluate_mrp_result
from rl.finite_horizon import (finite_horizon_MRP, evaluate,
                               unwrap_finite_horizon_MRP, WithTime)
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal


class FlipFlop(FiniteMarkovRewardProcess[bool]):
    '''A version of FlipFlop implemented with the FiniteMarkovProcess
    machinery.

    '''

    def __init__(self, p: float):
        transition_reward_map = {
            b: Categorical({(not b, 2.0): p, (b, 1.0): 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_reward_map)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.finite_flip_flop = FlipFlop(0.7)

    def test_evaluate_mrp(self):
        v = evaluate_mrp_result(self.finite_flip_flop, gamma=0.99)

        self.assertEqual(len(v), 2)

        for s in v:
            self.assertLess(abs(v[s] - 170), 0.1)

    def test_compare_to_backward_induction(self):
        finite_horizon = finite_horizon_MRP(self.finite_flip_flop, 10)

        v = evaluate_mrp_result(finite_horizon, gamma=1)
        self.assertEqual(len(v), 20)

        finite_v =\
            list(evaluate(unwrap_finite_horizon_MRP(finite_horizon), gamma=1))

        for time in range(10):
            self.assertAlmostEqual(
                v[NonTerminal(WithTime(state=True, time=time))],
                finite_v[time][NonTerminal(True)]
            )
            self.assertAlmostEqual(
                v[NonTerminal(WithTime(state=False, time=time))],
                finite_v[time][NonTerminal(False)]
            )
```

--------------------------------------------------------------------------------

## 📄 test_finite_horizon.py {#test-finite-horizon}

**Titre**: Problèmes à Horizon Fini

**Description**: MDP avec nombre d'étapes fixé

**Lignes de code**: 228

**Concepts clés**:
- Backward induction pour horizon fini
- V_t(s) dépend du temps restant
- Optimal stopping problems

**🎯 Utilisation HelixOne**: Options américaines, exécution avec deadline

### Code Source Complet

```python
from typing import Mapping, Sequence
import unittest

import dataclasses

from rl.distribution import Categorical
import rl.test_distribution as distribution
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_decision_process import (ActionMapping,
                                        FiniteMarkovDecisionProcess,
                                        NonTerminal, Terminal)

from rl.finite_horizon import (finite_horizon_MDP, finite_horizon_MRP,
                               WithTime, unwrap_finite_horizon_MDP,
                               unwrap_finite_horizon_MRP, evaluate,
                               optimal_vf_and_policy)


class FlipFlop(FiniteMarkovRewardProcess[bool]):
    ''' A version of FlipFlop implemented with the FiniteMarkovProcess machinery.

    '''

    def __init__(self, p: float):
        transition_reward_map = {
            b: Categorical({(not b, 2.0): p, (b, 1.0): 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_reward_map)


class TestFiniteMRP(unittest.TestCase):
    def setUp(self):
        self.finite_flip_flop = FlipFlop(0.7)

    def test_finite_horizon_MRP(self):
        finite = finite_horizon_MRP(self.finite_flip_flop, 10)

        trues = [NonTerminal(WithTime(True, time)) for time in range(10)]
        falses = [NonTerminal(WithTime(False, time)) for time in range(10)]
        non_terminal_states = set(trues + falses)
        self.assertEqual(set(finite.non_terminal_states), non_terminal_states)

        expected_transition = {}
        for state in non_terminal_states:
            t: int = state.state.time
            st: bool = state.state.state
            if t < 9:
                prob = {
                    (NonTerminal(WithTime(st, t + 1)), 1.0): 0.3,
                    (NonTerminal(WithTime(not st, t + 1)), 2.0): 0.7
                }
            else:
                prob = {
                    (Terminal(WithTime(st, t + 1)), 1.0): 0.3,
                    (Terminal(WithTime(not st, t + 1)), 2.0): 0.7
                }

            expected_transition[state] = Categorical(prob)

        for state in non_terminal_states:
            distribution.assert_almost_equal(
                self,
                finite.transition_reward(state),
                expected_transition[state])

    def test_unwrap_finite_horizon_MRP(self):
        finite = finite_horizon_MRP(self.finite_flip_flop, 10)

        def transition_for(_):
            return {
                True: Categorical({
                    (NonTerminal(True), 1.0): 0.3,
                    (NonTerminal(False), 2.0): 0.7
                }),
                False: Categorical({
                    (NonTerminal(True), 2.0): 0.7,
                    (NonTerminal(False), 1.0): 0.3
                })
            }

        unwrapped = unwrap_finite_horizon_MRP(finite)
        self.assertEqual(len(unwrapped), 10)

        expected_transitions = [transition_for(n) for n in range(10)]
        for time in range(9):
            got = unwrapped[time]
            expected = expected_transitions[time]
            distribution.assert_almost_equal(
                self, got[NonTerminal(True)],
                expected[True]
            )
            distribution.assert_almost_equal(
                self, got[NonTerminal(False)],
                expected[False]
            )

        distribution.assert_almost_equal(
            self, unwrapped[9][NonTerminal(True)],
            Categorical({
                (Terminal(True), 1.0): 0.3,
                (Terminal(False), 2.0): 0.7
            })
        )
        distribution.assert_almost_equal(
            self, unwrapped[9][NonTerminal(False)],
            Categorical({
                (Terminal(True), 2.0): 0.7,
                (Terminal(False), 1.0): 0.3
            })
        )

    def test_evaluate(self):
        process = finite_horizon_MRP(self.finite_flip_flop, 10)
        vs = list(evaluate(unwrap_finite_horizon_MRP(process), gamma=1))

        self.assertEqual(len(vs), 10)

        self.assertAlmostEqual(vs[0][NonTerminal(True)], 17)
        self.assertAlmostEqual(vs[0][NonTerminal(False)], 17)

        self.assertAlmostEqual(vs[5][NonTerminal(True)], 17 / 2)
        self.assertAlmostEqual(vs[5][NonTerminal(False)], 17 / 2)

        self.assertAlmostEqual(vs[9][NonTerminal(True)], 17 / 10)
        self.assertAlmostEqual(vs[9][NonTerminal(False)], 17 / 10)


class TestFiniteMDP(unittest.TestCase):
    def setUp(self):
        self.finite_flip_flop = FiniteMarkovDecisionProcess({
            True: {
                True: Categorical({(True, 1.0): 0.7, (False, 2.0): 0.3}),
                False: Categorical({(True, 1.0): 0.3, (False, 2.0): 0.7}),
            },
            False: {
                True: Categorical({(False, 1.0): 0.7, (True, 2.0): 0.3}),
                False: Categorical({(False, 1.0): 0.3, (True, 2.0): 0.7}),
            }
        })

    def test_finite_horizon_MDP(self):
        finite = finite_horizon_MDP(self.finite_flip_flop, limit=10)

        self.assertEqual(len(finite.non_terminal_states), 20)

        for s in finite.non_terminal_states:
            self.assertEqual(set(finite.actions(s)), {False, True})

        start = NonTerminal(WithTime(state=True, time=0))
        result = finite.mapping[start][False]
        expected_result = Categorical({
            (NonTerminal(WithTime(False, time=1)), 2.0): 0.7,
            (NonTerminal(WithTime(True, time=1)), 1.0): 0.3
        })
        distribution.assert_almost_equal(self, result, expected_result)

    def test_unwrap_finite_horizon_MDP(self):
        finite = finite_horizon_MDP(self.finite_flip_flop, 10)
        unwrapped = unwrap_finite_horizon_MDP(finite)

        self.assertEqual(len(unwrapped), 10)

        def action_mapping_for(s: WithTime[bool]) -> \
                ActionMapping[bool, WithTime[bool]]:
            same = NonTerminal(s.step_time())
            different = NonTerminal(dataclasses.replace(
                s.step_time(),
                state=not s.state
            ))

            return {
                True: Categorical({
                    (same, 1.0): 0.7,
                    (different, 2.0): 0.3
                }),
                False: Categorical({
                    (same, 1.0): 0.3,
                    (different, 2.0): 0.7
                })
            }

        for t in range(9):
            for s in True, False:
                s_time = WithTime(state=s, time=t)
                for a in True, False:
                    distribution.assert_almost_equal(
                        self,
                        finite.mapping[NonTerminal(s_time)][a],
                        action_mapping_for(s_time)[a]
                    )

        for s in True, False:
            s_time = WithTime(state=s, time=9)
            same = Terminal(s_time.step_time())
            different = Terminal(dataclasses.replace(
                s_time.step_time(),
                state=not s_time.state
            ))
            act_map = {
                True: Categorical({
                    (same, 1.0): 0.7,
                    (different, 2.0): 0.3
                }),
                False: Categorical({
                    (same, 1.0): 0.3,
                    (different, 2.0): 0.7
                })

            }
            for a in True, False:
                distribution.assert_almost_equal(
                    self,
                    finite.mapping[NonTerminal(s_time)][a],
                    act_map[a]
                )

    def test_optimal_policy(self):
        finite = finite_horizon_MDP(self.finite_flip_flop, limit=10)
        steps = unwrap_finite_horizon_MDP(finite)
        *v_ps, (_, p) = optimal_vf_and_policy(steps, gamma=1)

        for _, a in p.action_for.items():
            self.assertEqual(a, False)

        self.assertAlmostEqual(v_ps[0][0][NonTerminal(True)], 17)
        self.assertAlmostEqual(v_ps[5][0][NonTerminal(False)], 17 / 2)
```

--------------------------------------------------------------------------------

## 📄 test_function_approx.py {#test-function-approx}

**Titre**: Framework d'Approximation de Fonctions

**Description**: Approximation de V(s) et Q(s,a) pour grands espaces

**Lignes de code**: 47

**Concepts clés**:
- FunctionApprox[X] - Interface abstraite X → ℝ
- Dynamic - Lookup exact (DP tabulaire)
- Tabular - Table avec learning rate α(n)
- LinearFunctionApprox - Combinaison linéaire de features
- DNNApprox - Réseau de neurones profond
- AdamGradient - Optimiseur Adam (adaptive moment)
- Weights - Poids avec cache Adam
- DNNSpec - Spécification architecture DNN

**🎯 Utilisation HelixOne**: CRITIQUE - Approximer les fonctions de valeur

### Code Source Complet

```python
import unittest
import numpy as np

from rl.function_approx import (Dynamic)


class TestDynamic(unittest.TestCase):
    def setUp(self):
        self.dynamic_0 = Dynamic(values_map={0: 0.0, 1: 0.0, 2: 0.0})
        self.dynamic_almost_0 = Dynamic(values_map={0: 0.01, 1: 0.01, 2: 0.01})

        self.dynamic_1 = Dynamic(values_map={0: 1.0, 1: 2.0, 2: 3.0})
        self.dynamic_almost_1 = Dynamic(values_map={0: 1.01, 1: 2.01, 2: 3.01})

    def test_update(self):
        updated = self.dynamic_0.update([(0, 1.0), (1, 2.0), (2, 3.0)])
        self.assertEqual(self.dynamic_1, updated)

        partially_updated = self.dynamic_0.update([(1, 3.0)])
        expected = {0: 0.0, 1: 3.0, 2: 0.0}
        self.assertEqual(partially_updated, Dynamic(values_map=expected))

    def test_evaluate(self):
        np.testing.assert_array_almost_equal(
            self.dynamic_0.evaluate([0, 1, 2]),
            np.array([0.0, 0.0, 0.0])
        )

        np.testing.assert_array_almost_equal(
            self.dynamic_1.evaluate([0, 1, 2]),
            np.array([1.0, 2.0, 3.0])
        )

    def test_call(self):
        for i in range(0, 3):
            self.assertEqual(self.dynamic_0(i), 0.0)
            self.assertEqual(self.dynamic_1(i), float(i + 1))

    def test_within(self):
        self.assertTrue(self.dynamic_0.within(self.dynamic_0, tolerance=0.0))
        self.assertTrue(self.dynamic_0.within(self.dynamic_almost_0,
                                              tolerance=0.011))

        self.assertTrue(self.dynamic_1.within(self.dynamic_1, tolerance=0.0))
        self.assertTrue(self.dynamic_1.within(self.dynamic_almost_1,
                                              tolerance=0.011))
```

--------------------------------------------------------------------------------

## 📄 test_iterate.py {#test-iterate}

**Titre**: Utilitaires d'Itération

**Description**: Fonctions utilitaires pour l'itération et la convergence

**Lignes de code**: 47

**Concepts clés**:
- iterate - Applique une fonction répétitivement
- converge - Itère jusqu'à convergence (tolerance)
- accumulate - Accumule les résultats intermédiaires
- last - Retourne le dernier élément d'un itérateur

**🎯 Utilisation HelixOne**: Utilitaires pour boucles d'apprentissage

### Code Source Complet

```python
import itertools
import unittest

from rl.iterate import (iterate, last, converge, converged)


class TestIterate(unittest.TestCase):
    def test_iterate(self):
        ns = iterate(lambda x: x + 1, start=0)
        self.assertEqual(list(itertools.islice(ns, 5)), list(range(0, 5)))


class TestLast(unittest.TestCase):
    def test_last(self):
        self.assertEqual(last(range(0, 5)), 4)
        self.assertEqual(last(range(0, 10)), 9)

        self.assertEqual(last([]), None)


class TestConverge(unittest.TestCase):
    def test_converge(self):
        def close(a, b):
            return abs(a - b) < 0.1

        ns = (1.0 / n for n in iterate(lambda x: x + 1, start=1))
        self.assertAlmostEqual(converged(ns, close), 0.25, places=2)

        ns = (1.0 / n for n in iterate(lambda x: x + 1, start=1))
        all_ns = [1.0, 0.5, 0.33]
        for got, expected in zip(converge(ns, close), all_ns):
            self.assertAlmostEqual(got, expected, places=2)

    def test_converge_end(self):
        '''Check that converge ends the iterator at the right place when the
        underlying iterator ends before converging.

        '''
        def close(a, b):
            return abs(a - b) < 0.1

        ns = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.assertAlmostEqual(converged(iter(ns), close), 2.0)

        for got, expected in zip(converge(iter(ns), close), ns):
            self.assertAlmostEqual(got, expected)
```

--------------------------------------------------------------------------------

## 📄 test_markov_process.py {#test-markov-process}

**Titre**: Processus de Markov (MP/MRP)

**Description**: Implémentation des processus de Markov avec et sans récompenses

**Lignes de code**: 170

**Concepts clés**:
- State[S] = Terminal | NonTerminal - États avec distinction finale
- MarkovProcess - Processus avec transition(state) → Distribution[State]
- MarkovRewardProcess - Ajoute transition_reward avec récompenses
- FiniteMarkovProcess - Version tabulaire pour DP exacte
- TransitionStep, ReturnStep - Structures pour trajectoires
- Matrice de transition et distribution stationnaire

**🎯 Utilisation HelixOne**: Modéliser les prix d'actifs comme processus stochastiques

### Code Source Complet

```python
import itertools
import numpy as np
from io import StringIO
import sys
from typing import Tuple
import unittest

from rl.distribution import (
    Bernoulli,
    Categorical,
    Distribution,
    SampledDistribution,
    Constant,
)
from rl.markov_process import (
    FiniteMarkovProcess,
    MarkovProcess,
    MarkovRewardProcess,
    NonTerminal,
    State,
)


# Example classes:
class FlipFlop(MarkovProcess[bool]):
    """A simple example Markov chain with two states, flipping from one to
    the other with probability p and staying at the same state with
    probability 1 - p.

    """

    p: float

    def __init__(self, p: float):
        self.p = p

    def transition(self, state: NonTerminal[bool]) -> Distribution[State[bool]]:
        def next_state(state=state):
            switch_states = Bernoulli(self.p).sample()
            next_st: bool = not state.state if switch_states else state.state
            return NonTerminal(next_st)

        return SampledDistribution(next_state)


class FiniteFlipFlop(FiniteMarkovProcess[bool]):
    """A version of FlipFlop implemented with the FiniteMarkovProcess machinery."""

    def __init__(self, p: float):
        transition_map = {b: Categorical({not b: p, b: 1 - p}) for b in (True, False)}
        super().__init__(transition_map)


class RewardFlipFlop(MarkovRewardProcess[bool]):
    p: float

    def __init__(self, p: float):
        self.p = p

    def transition_reward(
        self, state: NonTerminal[bool]
    ) -> Distribution[Tuple[State[bool], float]]:
        def next_state(state=state):
            switch_states = Bernoulli(self.p).sample()

            st: bool = state.state
            if switch_states:
                next_s: bool = not st
                reward = 1 if st else 0.5
                return NonTerminal(next_s), reward
            else:
                return NonTerminal(st), 0.5

        return SampledDistribution(next_state)


class TestMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = FlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(
            itertools.islice(self.flip_flop.simulate(Constant(NonTerminal(True))), 10)
        )

        self.assertTrue(all(isinstance(outcome.state, bool) for outcome in trace))

        longer_trace = itertools.islice(
            self.flip_flop.simulate(Constant(NonTerminal(True))), 10000
        )
        count_trues = len(list(outcome for outcome in longer_trace if outcome.state))

        # If the code is correct, this should fail with a vanishingly
        # small probability
        self.assertTrue(1000 < count_trues < 9000)


class TestFiniteMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = FiniteFlipFlop(0.5)

        self.biased = FiniteFlipFlop(0.3)

    def test_flip_flop(self):
        trace = list(
            itertools.islice(self.flip_flop.simulate(Constant(NonTerminal(True))), 10)
        )

        self.assertTrue(all(isinstance(outcome.state, bool) for outcome in trace))

        longer_trace = itertools.islice(
            self.flip_flop.simulate(Constant(NonTerminal(True))), 10000
        )
        count_trues = len(list(outcome for outcome in longer_trace if outcome.state))

        # If the code is correct, this should fail with a vanishingly
        # small probability
        self.assertTrue(1000 < count_trues < 9000)

    def test_transition_matrix(self):
        matrix = self.flip_flop.get_transition_matrix()
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_equal(matrix, expected)

        matrix = self.biased.get_transition_matrix()
        expected = np.array([[0.7, 0.3], [0.3, 0.7]])
        np.testing.assert_array_equal(matrix, expected)

    def test_stationary_distribution(self):
        distribution = self.flip_flop.get_stationary_distribution().table()
        expected = [(True, 0.5), (False, 0.5)]
        np.testing.assert_almost_equal(list(distribution.items()), expected)

        distribution = self.biased.get_stationary_distribution().table()
        expected = [(True, 0.5), (False, 0.5)]
        np.testing.assert_almost_equal(list(distribution.items()), expected)

    def test_display(self):
        # Just test that the display functions don't error out.
        stdout, stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            self.flip_flop.display_stationary_distribution()
            self.flip_flop.generate_image()
            self.flip_flop.__repr__()
        except Exception:
            self.fail("Display functions raised an error.")
        finally:
            sys.stdout = stdout
            sys.stderr = stderr


class TestRewardMarkovProcess(unittest.TestCase):
    def setUp(self):
        self.flip_flop = RewardFlipFlop(0.5)

    def test_flip_flop(self):
        trace = list(
            itertools.islice(
                self.flip_flop.simulate_reward(Constant(NonTerminal(True))), 10
            )
        )

        self.assertTrue(all(isinstance(step.next_state.state, bool) for step in trace))

        cumulative_reward = sum(step.reward for step in trace)
        self.assertTrue(0 <= cumulative_reward <= 10)
```

--------------------------------------------------------------------------------

## 📄 test_monte_carlo.py {#test-monte-carlo}

**Titre**: Méthodes Monte Carlo

**Description**: Algorithmes MC pour évaluation et contrôle

**Lignes de code**: 50

**Concepts clés**:
- mc_prediction - Évaluation MC: V(s) ← V(s) + α[G - V(s)]
- mc_control - Contrôle MC avec ε-greedy
- Utilise épisodes complets (pas de bootstrap)

**🎯 Utilisation HelixOne**: Évaluation avec historique complet

### Code Source Complet

```python
import unittest

import random

from rl.distribution import Categorical, Choose
from rl.function_approx import Tabular
import rl.iterate as iterate
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
import rl.monte_carlo as mc


class FlipFlop(FiniteMarkovRewardProcess[bool]):
    '''A version of FlipFlop implemented with the FiniteMarkovProcess
    machinery.

    '''

    def __init__(self, p: float):
        transition_reward_map = {
            b: Categorical({(not b, 2.0): p, (b, 1.0): 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_reward_map)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.finite_flip_flop = FlipFlop(0.7)

    def test_evaluate_finite_mrp(self):
        start = Tabular({s: 0.0 for s in
                         self.finite_flip_flop.non_terminal_states})
        traces = self.finite_flip_flop.reward_traces(Choose({
            NonTerminal(True),
            NonTerminal(False)
        }))
        v = iterate.converged(
            mc.mc_prediction(traces, γ=0.99, approx_0=start),
            # Loose bound of 0.01 to speed up test.
            done=lambda a, b: a.within(b, 0.01)
        )

        self.assertEqual(len(v.values_map), 2)

        for s in v.values_map:
            # Intentionally loose bound—otherwise test is too slow.
            # Takes >1s on my machine otherwise.
            self.assertLess(abs(v(s) - 170), 1.0)
```

--------------------------------------------------------------------------------

## 📄 test_td.py {#test-td}

**Titre**: Temporal Difference Learning

**Description**: Algorithmes TD pour apprentissage online

**Lignes de code**: 126

**Concepts clés**:
- td_prediction - TD(0): V(s) ← V(s) + α[r + γV(s') - V(s)]
- sarsa - SARSA (on-policy): Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- q_learning - Q-Learning (off-policy): Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]
- Bootstrap: apprend d'estimations, pas besoin d'épisode complet

**🎯 Utilisation HelixOne**: Apprentissage online des stratégies

### Code Source Complet

```python
import unittest

import itertools
import random
from typing import cast, Iterable, Iterator, Optional, Tuple

from rl.distribution import Categorical, Choose
from rl.function_approx import Tabular
import rl.iterate as iterate
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
import rl.markov_process as mp
from rl.markov_decision_process import FiniteMarkovDecisionProcess
import rl.markov_decision_process as mdp
import rl.td as td
from rl.policy import FinitePolicy


class FlipFlop(FiniteMarkovRewardProcess[bool]):
    '''A version of FlipFlop implemented with the FiniteMarkovProcess
    machinery.

    '''

    def __init__(self, p: float):
        transition_reward_map = {
            b: Categorical({(not b, 2.0): p, (b, 1.0): 1 - p})
            for b in (True, False)
        }
        super().__init__(transition_reward_map)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        random.seed(42)

        self.finite_flip_flop = FlipFlop(0.7)

        self.finite_mdp = FiniteMarkovDecisionProcess({
            True: {
                True: Categorical({(True, 1.0): 0.7, (False, 2.0): 0.3}),
                False: Categorical({(True, 1.0): 0.3, (False, 2.0): 0.7}),
            },
            False: {
                True: Categorical({(False, 1.0): 0.7, (True, 2.0): 0.3}),
                False: Categorical({(False, 1.0): 0.3, (True, 2.0): 0.7}),
            }
        })

    def test_evaluate_finite_mrp(self) -> None:
        start = Tabular(
            {s: 0.0 for s in self.finite_flip_flop.non_terminal_states},
            count_to_weight_func=lambda _: 0.1
        )

        episode_length = 20
        episodes: Iterable[Iterable[mp.TransitionStep[bool]]] =\
            self.finite_flip_flop.reward_traces(Choose({
                NonTerminal(True),
                NonTerminal(False)
            }))
        transitions: Iterable[mp.TransitionStep[bool]] =\
            itertools.chain.from_iterable(
                itertools.islice(episode, episode_length)
                for episode in episodes
            )

        vs = td.td_prediction(transitions, γ=0.99, approx_0=start)

        v: Optional[Tabular[NonTerminal[bool]]] = iterate.last(
            itertools.islice(
                cast(Iterator[Tabular[NonTerminal[bool]]], vs),
                10000)
        )

        if v is not None:
            self.assertEqual(len(v.values_map), 2)

            for s in v.values_map:
                # Intentionally loose bound—otherwise test is too slow.
                # Takes >1s on my machine otherwise.
                self.assertLess(abs(v(s) - 170), 3.0)
        else:
            assert False

    def test_evaluate_finite_mdp(self) -> None:
        q_0: Tabular[Tuple[NonTerminal[bool], bool]] = Tabular(
            {(s, a): 0.0
             for s in self.finite_mdp.non_terminal_states
             for a in self.finite_mdp.actions(s)},
            count_to_weight_func=lambda _: 0.1
        )

        uniform_policy: FinitePolicy[bool, bool] =\
            FinitePolicy({
                s.state: Choose(self.finite_mdp.actions(s))
                for s in self.finite_mdp.non_terminal_states
            })

        transitions: Iterable[mdp.TransitionStep[bool, bool]] =\
            self.finite_mdp.simulate_actions(
                Choose(self.finite_mdp.non_terminal_states),
                uniform_policy
            )

        qs = td.q_learning_external_transitions(
            transitions,
            self.finite_mdp.actions,
            q_0,
            γ=0.99
        )

        q: Optional[Tabular[Tuple[NonTerminal[bool], bool]]] =\
            iterate.last(
                cast(Iterator[Tabular[Tuple[NonTerminal[bool], bool]]],
                     itertools.islice(qs, 20000))
            )

        if q is not None:
            self.assertEqual(len(q.values_map), 4)

            for s in [NonTerminal(True), NonTerminal(False)]:
                self.assertLess(abs(q((s, False)) - 170.0), 2)
                self.assertGreater(q((s, False)), q((s, True)))
        else:
            assert False
```

--------------------------------------------------------------------------------

# PARTIE 10 - UTILITAIRES

================================================================================

## 📄 td.py {#td}

**Titre**: Temporal Difference Learning

**Description**: Algorithmes TD pour apprentissage online

**Lignes de code**: 407

**Concepts clés**:
- td_prediction - TD(0): V(s) ← V(s) + α[r + γV(s') - V(s)]
- sarsa - SARSA (on-policy): Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- q_learning - Q-Learning (off-policy): Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) - Q(s,a)]
- Bootstrap: apprend d'estimations, pas besoin d'épisode complet

**🎯 Utilisation HelixOne**: Apprentissage online des stratégies

### Code Source Complet

```python
'''Monte Carlo methods for working with Markov Reward Process and
Markov Decision Processes.

'''

from operator import itemgetter
import itertools
from typing import Callable, Iterable, Iterator, TypeVar, Set, Sequence, Tuple

import numpy as np

from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf
from rl.distribution import Categorical
from rl.function_approx import LinearFunctionApprox, Weights
import rl.iterate as iterate
import rl.markov_process as mp
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_decision_process import TransitionStep, NonTerminal
from rl.monte_carlo import greedy_policy_from_qvf
from rl.policy import Policy, DeterministicPolicy
from rl.experience_replay import ExperienceReplayMemory

S = TypeVar('S')


def td_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: ValueFunctionApprox[S],
        γ: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate an MRP using TD(0) using the given sequence of
    transitions.

    Each value this function yields represents the approximated value
    function for the MRP after an additional transition.

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      approx_0 -- initial approximation of value function
      γ -- discount rate (0 < γ ≤ 1)

    '''
    def step(
            v: ValueFunctionApprox[S],
            transition: mp.TransitionStep[S]
    ) -> ValueFunctionApprox[S]:
        return v.update([(
            transition.state,
            transition.reward + γ * extended_vf(v, transition.next_state)
        )])
    return iterate.accumulate(transitions, step, initial=approx_0)


def batch_td_prediction(
    transitions: Iterable[mp.TransitionStep[S]],
    approx_0: ValueFunctionApprox[S],
    γ: float,
    convergence_tolerance: float = 1e-5
) -> ValueFunctionApprox[S]:
    '''transitions is a finite iterable'''

    def step(
        v: ValueFunctionApprox[S],
        tr_seq: Sequence[mp.TransitionStep[S]]
    ) -> ValueFunctionApprox[S]:
        return v.update([(
            tr.state, tr.reward + γ * extended_vf(v, tr.next_state)
        ) for tr in tr_seq])

    def done(
        a: ValueFunctionApprox[S],
        b: ValueFunctionApprox[S],
        convergence_tolerance=convergence_tolerance
    ) -> bool:
        return b.within(a, convergence_tolerance)

    return iterate.converged(
        iterate.accumulate(
            itertools.repeat(list(transitions)),
            step,
            initial=approx_0
        ),
        done=done
    )


def least_squares_td(
    transitions: Iterable[mp.TransitionStep[S]],
    feature_functions: Sequence[Callable[[NonTerminal[S]], float]],
    γ: float,
    ε: float
) -> LinearFunctionApprox[NonTerminal[S]]:
    ''' transitions is a finite iterable '''
    num_features: int = len(feature_functions)
    a_inv: np.ndarray = np.eye(num_features) / ε
    b_vec: np.ndarray = np.zeros(num_features)
    for tr in transitions:
        phi1: np.ndarray = np.array([f(tr.state) for f in feature_functions])
        if isinstance(tr.next_state, NonTerminal):
            phi2 = phi1 - γ * np.array([f(tr.next_state)
                                        for f in feature_functions])
        else:
            phi2 = phi1
        temp: np.ndarray = a_inv.T.dot(phi2)
        a_inv = a_inv - np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
        b_vec += phi1 * tr.reward

    opt_wts: np.ndarray = a_inv.dot(b_vec)
    return LinearFunctionApprox.create(
        feature_functions=feature_functions,
        weights=Weights.create(opt_wts)
    )


A = TypeVar('A')


def epsilon_greedy_action(
    q: QValueFunctionApprox[S, A],
    nt_state: NonTerminal[S],
    actions: Set[A],
    ϵ: float
) -> A:
    '''
    given a non-terminal state, a Q-Value Function (in the form of a
    FunctionApprox: (state, action) -> Value, and epislon, return
    an action sampled from the probability distribution implied by an
    epsilon-greedy policy that is derived from the Q-Value Function.
    '''
    greedy_action: A = max(
        ((a, q((nt_state, a))) for a in actions),
        key=itemgetter(1)
    )[0]
    return Categorical(
        {a: ϵ / len(actions) +
         (1 - ϵ if a == greedy_action else 0.) for a in actions}
    ).sample()


def glie_sarsa(
    mdp: MarkovDecisionProcess[S, A],
    states: NTStateDistribution[S],
    approx_0: QValueFunctionApprox[S, A],
    γ: float,
    ϵ_as_func_of_episodes: Callable[[int], float],
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    q: QValueFunctionApprox[S, A] = approx_0
    yield q
    num_episodes: int = 0
    while True:
        num_episodes += 1
        ϵ: float = ϵ_as_func_of_episodes(num_episodes)
        state: NonTerminal[S] = states.sample()
        action: A = epsilon_greedy_action(
            q=q,
            nt_state=state,
            actions=set(mdp.actions(state)),
            ϵ=ϵ
        )
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            next_state, reward = mdp.step(state, action).sample()
            if isinstance(next_state, NonTerminal):
                next_action: A = epsilon_greedy_action(
                    q=q,
                    nt_state=next_state,
                    actions=set(mdp.actions(next_state)),
                    ϵ=ϵ
                )
                q = q.update([(
                    (state, action),
                    reward + γ * q((next_state, next_action))
                )])
                action = next_action
            else:
                q = q.update([((state, action), reward)])
            yield q
            steps += 1
            state = next_state


PolicyFromQType = Callable[
    [QValueFunctionApprox[S, A], MarkovDecisionProcess[S, A]],
    Policy[S, A]
]


def q_learning(
    mdp: MarkovDecisionProcess[S, A],
    policy_from_q: PolicyFromQType,
    states: NTStateDistribution[S],
    approx_0: QValueFunctionApprox[S, A],
    γ: float,
    max_episode_length: int
) -> Iterator[QValueFunctionApprox[S, A]]:
    q: QValueFunctionApprox[S, A] = approx_0
    yield q
    while True:
        state: NonTerminal[S] = states.sample()
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            policy: Policy[S, A] = policy_from_q(q, mdp)
            action: A = policy.act(state).sample()
            next_state, reward = mdp.step(state, action).sample()
            next_return: float = max(
                q((next_state, a))
                for a in mdp.actions(next_state)
            ) if isinstance(next_state, NonTerminal) else 0.
            q = q.update([((state, action), reward + γ * next_return)])
            yield q
            steps += 1
            state = next_state


def q_learning_external_transitions(
        transitions: Iterable[TransitionStep[S, A]],
        actions: Callable[[NonTerminal[S]], Iterable[A]],
        approx_0: QValueFunctionApprox[S, A],
        γ: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    '''Return policies that try to maximize the reward based on the given
    set of experiences.

    Arguments:
      transitions -- a sequence of state, action, reward, state (S, A, R, S')
      actions -- a function returning the possible actions for a given state
      approx_0 -- initial approximation of q function
      γ -- discount rate (0 < γ ≤ 1)

    Returns:
      an itertor of approximations of the q function based on the
      transitions given as input

    '''
    def step(
            q: QValueFunctionApprox[S, A],
            transition: TransitionStep[S, A]
    ) -> QValueFunctionApprox[S, A]:
        next_return: float = max(
            q((transition.next_state, a))
            for a in actions(transition.next_state)
        ) if isinstance(transition.next_state, NonTerminal) else 0.
        return q.update([
            ((transition.state, transition.action),
             transition.reward + γ * next_return)
        ])

    return iterate.accumulate(transitions, step, initial=approx_0)
# 
# 
# def q_learning_experience_replay(
#     mdp: MarkovDecisionProcess[S, A],
#     policy_from_q: PolicyFromQType,
#     states: NTStateDistribution[S],
#     approx_0: QValueFunctionApprox[S, A],
#     γ: float,
#     max_episode_length: int,
#     mini_batch_size: int,
#     weights_decay_half_life: float
# ) -> Iterator[QValueFunctionApprox[S, A]]:
#     replay_memory: List[TransitionStep[S, A]] = []
#     decay_weights: List[float] = []
#     factor: float = 0.5 ** (1.0 / weights_decay_half_life)
#     random_gen = np.random.default_rng()
#     q: QValueFunctionApprox[S, A] = approx_0
#     yield q
#     while True:
#         state: NonTerminal[S] = states.sample()
#         steps: int = 0
#         while isinstance(state, NonTerminal) and steps < max_episode_length:
#             policy: Policy[S, A] = policy_from_q(q, mdp)
#             action: A = policy.act(state).sample()
#             next_state, reward = mdp.step(state, action).sample()
#             replay_memory.append(TransitionStep(
#                 state=state,
#                 action=action,
#                 next_state=next_state,
#                 reward=reward
#             ))
#             replay_len: int = len(replay_memory)
#             decay_weights.append(factor ** (replay_len - 1))
#             norm_factor: float = (1 - factor ** replay_len) / (1 - factor)
#             norm_decay_weights: Sequence[float] = [w / norm_factor for w in
#                                                    reversed(decay_weights)]
#             trs: Sequence[TransitionStep[S, A]] = \
#                 [replay_memory[i] for i in random_gen.choice(
#                     replay_len,
#                     min(mini_batch_size, replay_len),
#                     replace=False,
#                     p=norm_decay_weights
#                 )]
#             q = q.update(
#                 [(
#                     (tr.state, tr.action),
#                     tr.reward + γ * (
#                         max(q((tr.next_state, a))
#                             for a in mdp.actions(tr.next_state))
#                         if isinstance(tr.next_state, NonTerminal) else 0.)
#                 ) for tr in trs],
#             )
#             yield q
#             steps += 1
#             state = next_state


def q_learning_experience_replay(
    mdp: MarkovDecisionProcess[S, A],
    policy_from_q: PolicyFromQType,
    states: NTStateDistribution[S],
    approx_0: QValueFunctionApprox[S, A],
    γ: float,
    max_episode_length: int,
    mini_batch_size: int,
    weights_decay_half_life: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    exp_replay: ExperienceReplayMemory[TransitionStep[S, A]] = \
        ExperienceReplayMemory(
            time_weights_func=lambda t: 0.5 ** (t / weights_decay_half_life),
        )
    q: QValueFunctionApprox[S, A] = approx_0
    yield q
    while True:
        state: NonTerminal[S] = states.sample()
        steps: int = 0
        while isinstance(state, NonTerminal) and steps < max_episode_length:
            policy: Policy[S, A] = policy_from_q(q, mdp)
            action: A = policy.act(state).sample()
            next_state, reward = mdp.step(state, action).sample()
            exp_replay.add_data(TransitionStep(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward
            ))
            trs: Sequence[TransitionStep[S, A]] = \
                exp_replay.sample_mini_batch(mini_batch_size)
            q = q.update(
                [(
                    (tr.state, tr.action),
                    tr.reward + γ * (
                        max(q((tr.next_state, a))
                            for a in mdp.actions(tr.next_state))
                        if isinstance(tr.next_state, NonTerminal) else 0.)
                ) for tr in trs],
            )
            yield q
            steps += 1
            state = next_state


def least_squares_tdq(
    transitions: Iterable[TransitionStep[S, A]],
    feature_functions: Sequence[Callable[[Tuple[NonTerminal[S], A]], float]],
    target_policy: DeterministicPolicy[S, A],
    γ: float,
    ε: float
) -> LinearFunctionApprox[Tuple[NonTerminal[S], A]]:
    '''transitions is a finite iterable'''
    num_features: int = len(feature_functions)
    a_inv: np.ndarray = np.eye(num_features) / ε
    b_vec: np.ndarray = np.zeros(num_features)
    for tr in transitions:
        phi1: np.ndarray = np.array([f((tr.state, tr.action))
                                     for f in feature_functions])
        if isinstance(tr.next_state, NonTerminal):
            phi2 = phi1 - γ * np.array([
                f((tr.next_state, target_policy.action_for(tr.next_state.state)))
                for f in feature_functions])
        else:
            phi2 = phi1
        temp: np.ndarray = a_inv.T.dot(phi2)
        a_inv = a_inv - np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
        b_vec += phi1 * tr.reward

    opt_wts: np.ndarray = a_inv.dot(b_vec)
    return LinearFunctionApprox.create(
        feature_functions=feature_functions,
        weights=Weights.create(opt_wts)
    )


def least_squares_policy_iteration(
    transitions: Iterable[TransitionStep[S, A]],
    actions: Callable[[NonTerminal[S]], Iterable[A]],
    feature_functions: Sequence[Callable[[Tuple[NonTerminal[S], A]], float]],
    initial_target_policy: DeterministicPolicy[S, A],
    γ: float,
    ε: float
) -> Iterator[LinearFunctionApprox[Tuple[NonTerminal[S], A]]]:
    '''transitions is a finite iterable'''
    target_policy: DeterministicPolicy[S, A] = initial_target_policy
    transitions_seq: Sequence[TransitionStep[S, A]] = list(transitions)
    while True:
        q: LinearFunctionApprox[Tuple[NonTerminal[S], A]] = \
            least_squares_tdq(
                transitions=transitions_seq,
                feature_functions=feature_functions,
                target_policy=target_policy,
                γ=γ,
                ε=ε,
            )
        target_policy = greedy_policy_from_qvf(q, actions)
        yield q
```

--------------------------------------------------------------------------------


---

# 📊 STATISTIQUES FINALES

| Métrique | Valeur |
|----------|--------|
| **Fichiers Python** | 79 |
| **Lignes de code** | 13,007 |
| **Date** | 2026-01-29 08:51:08 |

---

# 🎯 GUIDE D'IMPLÉMENTATION HELIXONE

## Priorité 1 - Core (Semaine 1-2)
1. `distribution.py` - Distributions de probabilité
2. `markov_process.py` - MP/MRP (Markov Reward Process)
3. `markov_decision_process.py` - MDP (Markov Decision Process)
4. `policy.py` - Politiques
5. `function_approx.py` - Approximation (Linear, DNN)

## Priorité 2 - Algorithmes (Semaine 3-4)
1. `dynamic_programming.py` - DP (Dynamic Programming) exacte
2. `td.py` - TD (Temporal Difference) Learning
3. `monte_carlo.py` - MC (Monte Carlo) methods
4. `policy_gradient.py` - PG (Policy Gradient)

## Priorité 3 - Finance (Semaine 5-6)
1. `optimal_order_execution.py` - Almgren-Chriss (CRITIQUE)
2. `order_book.py` - LOB (Limit Order Book)
3. `asset_alloc_discrete.py` - Allocation
4. `optimal_exercise_bi.py` - Options américaines

## Architecture HelixOne Recommandée

```
helixone/
├── rl_core/
│   ├── __init__.py
│   ├── distributions.py      # Basé sur distribution.py
│   ├── markov.py            # MP, MRP, MDP
│   ├── policies.py          # Politiques
│   └── approximators.py     # FunctionApprox (Tabular, Linear, DNN)
├── algorithms/
│   ├── __init__.py
│   ├── dp.py                # DP (Dynamic Programming)
│   ├── td.py                # TD (Temporal Difference)
│   ├── mc.py                # MC (Monte Carlo)
│   └── pg.py                # PG (Policy Gradient)
├── finance/
│   ├── __init__.py
│   ├── execution.py         # Optimal Order Execution (Almgren-Chriss)
│   ├── order_book.py        # LOB (Limit Order Book) modeling
│   ├── allocation.py        # Asset Allocation (Merton, Discrete)
│   └── options.py           # American Options (BI, LSPI)
└── utils/
    ├── __init__.py
    ├── iterate.py           # Utilitaires d'itération
    └── returns.py           # Calcul des retours
```

---

# ✅ CHECKLIST AVANT IMPLÉMENTATION

- [ ] Lire le glossaire pour comprendre tous les acronymes
- [ ] Commencer par `distribution.py` (base de tout)
- [ ] Implémenter les tests unitaires en parallèle
- [ ] Valider chaque module avant de passer au suivant
- [ ] Documenter les adaptations pour HelixOne

---

**FIN DU GUIDE COMPLET STANFORD RL POUR HELIXONE**
**Total: 13,007 lignes de code documentées**
