"""
================================================================================
FOUNDATIONS OF REINFORCEMENT LEARNING WITH APPLICATIONS IN FINANCE
================================================================================

Ce fichier contient l'implémentation complète des concepts et algorithmes
du livre "Foundations of Reinforcement Learning with Applications in Finance"
par Ashwin Rao et Tikhon Jelvis.

CONVENTIONS DANS CE CODE:
- Tous les acronymes sont explicités entre parenthèses
- Les termes techniques sont accompagnés d'exemples concrets
- Le code est abondamment commenté en français

GLOSSAIRE DES ACRONYMES:
- RL (Reinforcement Learning): Apprentissage par renforcement
- MDP (Markov Decision Process): Processus de décision markovien
- MRP (Markov Reward Process): Processus de récompense markovien
- MC (Monte Carlo): Méthode de Monte Carlo
- TD (Temporal Difference): Différence temporelle
- DP (Dynamic Programming): Programmation dynamique
- ADP (Approximate Dynamic Programming): Programmation dynamique approchée
- CARA (Constant Absolute Risk Aversion): Aversion au risque absolue constante
- CRRA (Constant Relative Risk Aversion): Aversion au risque relative constante
- HJB (Hamilton-Jacobi-Bellman): Équation de Hamilton-Jacobi-Bellman
- DNN (Deep Neural Network): Réseau de neurones profond
- SARSA (State-Action-Reward-State-Action): Algorithme on-policy
- DQN (Deep Q-Network): Réseau Q profond
- LSPI (Least Squares Policy Iteration): Itération de politique par moindres carrés
- LSTD (Least Squares TD): TD par moindres carrés
- GLIE (Greedy in the Limit with Infinite Exploration): Glouton à la limite avec exploration infinie
- FIM (Fisher Information Matrix): Matrice d'information de Fisher

Auteur: Basé sur le travail de Ashwin Rao et Tikhon Jelvis
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Generic, TypeVar, Callable, Iterable, Iterator, Mapping, 
    Sequence, Tuple, Optional, Dict, List, Any
)
from collections import defaultdict
import numpy as np
from numpy.random import default_rng
import random
from functools import reduce
from operator import add
import itertools

# =============================================================================
# SECTION 1: TYPES GÉNÉRIQUES ET UTILITAIRES
# =============================================================================

# TypeVar = Variable de type générique (permet de créer des classes/fonctions
# qui fonctionnent avec différents types)
# Exemple: S peut être un int, str, tuple, etc.
S = TypeVar('S')  # Type pour les états (States)
A = TypeVar('A')  # Type pour les actions
X = TypeVar('X')  # Type générique

# Générateur de nombres aléatoires
rng = default_rng()


def iterate(step: Callable[[X], X], start: X) -> Iterator[X]:
    """
    Génère une séquence infinie en appliquant répétitivement une fonction.
    
    C'est un pattern fondamental en RL (Reinforcement Learning): on part
    d'une estimation initiale et on l'améliore itérativement.
    
    Exemple concret - Calcul de racine carrée par méthode de Newton:
        >>> def newton_step(x): return (x + 2/x) / 2  # Pour sqrt(2)
        >>> list(itertools.islice(iterate(newton_step, 1.0), 5))
        [1.0, 1.5, 1.4166..., 1.41421..., 1.41421...]
    
    Args:
        step: Fonction de transition f: X -> X
        start: Valeur initiale
        
    Yields:
        Séquence: start, step(start), step(step(start)), ...
    """
    state = start
    while True:
        yield state
        state = step(state)


def converge(values: Iterator[X], done: Callable[[X, X], bool]) -> Iterator[X]:
    """
    Itère jusqu'à convergence selon un critère donné.
    
    Utilisé pour arrêter les algorithmes itératifs quand la solution
    ne change plus significativement.
    
    Exemple - Convergence avec tolérance:
        >>> def close_enough(a, b): return abs(a - b) < 0.001
        >>> vals = iter([1.0, 1.5, 1.41, 1.414, 1.4142, 1.41421])
        >>> list(converge(vals, close_enough))
        [1.0, 1.5, 1.41, 1.414, 1.4142]
    
    Args:
        values: Itérateur de valeurs
        done: Fonction (ancien, nouveau) -> bool qui retourne True si convergé
        
    Yields:
        Valeurs jusqu'à convergence (la dernière valeur est celle où done=True)
    """
    a = next(values, None)
    if a is None:
        return
    yield a
    
    for b in values:
        yield b
        if done(a, b):
            return
        a = b


def last(it: Iterator[X]) -> Optional[X]:
    """
    Retourne le dernier élément d'un itérateur.
    
    Utile pour obtenir le résultat final d'un algorithme itératif.
    
    Args:
        it: Itérateur
        
    Returns:
        Dernier élément, ou None si l'itérateur est vide
    """
    result = None
    for x in it:
        result = x
    return result


# =============================================================================
# SECTION 2: DISTRIBUTIONS DE PROBABILITÉ
# =============================================================================

class Distribution(ABC, Generic[X]):
    """
    Classe abstraite représentant une distribution de probabilité.
    
    Une distribution permet de:
    1. Échantillonner (sample): tirer des valeurs aléatoires
    2. Calculer des espérances: E[f(X)] = ∑ p(x) * f(x)
    
    Exemple intuitif:
        Un dé à 6 faces est une distribution uniforme sur {1,2,3,4,5,6}
        - sample() retourne un nombre entre 1 et 6
        - expectation(lambda x: x) retourne 3.5 (la moyenne)
    
    Cette abstraction est fondamentale en RL car:
    - Les transitions d'états sont probabilistes
    - Les politiques peuvent être stochastiques
    - On estime les valeurs par espérance des retours
    """
    
    @abstractmethod
    def sample(self) -> X:
        """
        Tire un échantillon selon la distribution.
        
        Returns:
            Une valeur tirée aléatoirement
        """
        pass
    
    def sample_n(self, n: int) -> Sequence[X]:
        """
        Tire n échantillons i.i.d. (indépendants et identiquement distribués).
        
        i.i.d. signifie que chaque tirage est:
        - Indépendant: le résultat d'un tirage n'affecte pas les autres
        - Identiquement distribué: tous suivent la même loi
        
        Args:
            n: Nombre d'échantillons
            
        Returns:
            Liste de n échantillons
        """
        return [self.sample() for _ in range(n)]
    
    @abstractmethod
    def expectation(self, f: Callable[[X], float]) -> float:
        """
        Calcule l'espérance E[f(X)].
        
        L'espérance est la "moyenne pondérée" d'une fonction.
        
        Exemple avec un dé:
            E[X] = (1+2+3+4+5+6)/6 = 3.5
            E[X²] = (1+4+9+16+25+36)/6 = 15.17
        
        Args:
            f: Fonction à appliquer aux valeurs
            
        Returns:
            Espérance de f(X)
        """
        pass
    
    def map(self, f: Callable[[X], X]) -> Distribution[X]:
        """
        Transforme la distribution en appliquant une fonction.
        
        Si X ~ D, alors f(X) ~ map(D, f)
        
        Exemple:
            Si X est uniforme sur [0,1], alors 2*X est uniforme sur [0,2]
        
        Args:
            f: Fonction de transformation
            
        Returns:
            Nouvelle distribution
        """
        return SampledDistribution(lambda: f(self.sample()))


class SampledDistribution(Distribution[X]):
    """
    Distribution définie uniquement par sa méthode d'échantillonnage.
    
    Utile quand on peut simuler mais pas calculer analytiquement.
    
    Exemple - Distribution de la somme de deux dés:
        >>> def sum_two_dice(): return random.randint(1,6) + random.randint(1,6)
        >>> dist = SampledDistribution(sum_two_dice)
        >>> dist.sample()  # Retourne un nombre entre 2 et 12
    """
    
    def __init__(
        self, 
        sampler: Callable[[], X],
        expectation_samples: int = 10000
    ):
        """
        Args:
            sampler: Fonction sans argument qui retourne un échantillon
            expectation_samples: Nombre d'échantillons pour estimer l'espérance
        """
        self.sampler = sampler
        self.expectation_samples = expectation_samples
    
    def sample(self) -> X:
        return self.sampler()
    
    def expectation(self, f: Callable[[X], float]) -> float:
        """
        Estime l'espérance par Monte Carlo (MC).
        
        MC = Méthode de Monte Carlo: estimer une quantité en faisant
        la moyenne de nombreux échantillons aléatoires.
        
        E[f(X)] ≈ (1/N) * ∑ f(x_i) où x_i sont des échantillons
        
        Par la loi des grands nombres, cette estimation converge
        vers la vraie valeur quand N → ∞
        """
        samples = self.sample_n(self.expectation_samples)
        return sum(f(x) for x in samples) / len(samples)


@dataclass(frozen=True)
class Categorical(Distribution[X]):
    """
    Distribution catégorielle (discrète avec probabilités explicites).
    
    C'est la distribution la plus générale pour un ensemble fini de valeurs.
    
    Exemple - Dé pipé:
        >>> probas = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.5}
        >>> dice = Categorical(probas)  # Le 6 sort 50% du temps
    
    Attributs:
        probabilities: Dict mappant chaque valeur à sa probabilité
    """
    probabilities: Mapping[X, float]
    
    def sample(self) -> X:
        """Tire selon les probabilités données."""
        items = list(self.probabilities.items())
        values = [item[0] for item in items]
        probs = [item[1] for item in items]
        return random.choices(values, weights=probs, k=1)[0]
    
    def expectation(self, f: Callable[[X], float]) -> float:
        """
        Calcule exactement E[f(X)] = ∑ p(x) * f(x)
        
        C'est un calcul exact (pas une estimation).
        """
        return sum(p * f(x) for x, p in self.probabilities.items())
    
    def probability(self, value: X) -> float:
        """Retourne P(X = value)."""
        return self.probabilities.get(value, 0.0)


class Gaussian(Distribution[float]):
    """
    Distribution gaussienne (normale).
    
    La distribution la plus importante en statistiques, caractérisée par:
    - μ (mu): moyenne (centre de la distribution)
    - σ (sigma): écart-type (dispersion autour de la moyenne)
    
    Propriétés:
    - 68% des valeurs sont dans [μ-σ, μ+σ]
    - 95% des valeurs sont dans [μ-2σ, μ+2σ]
    - 99.7% des valeurs sont dans [μ-3σ, μ+3σ]
    
    Exemple - Taille humaine (en cm):
        >>> heights = Gaussian(mu=170, sigma=10)
        >>> heights.sample()  # Une taille aléatoire
    """
    
    def __init__(self, mu: float, sigma: float):
        """
        Args:
            mu: Moyenne
            sigma: Écart-type (doit être > 0)
        """
        self.mu = mu
        self.sigma = sigma
    
    def sample(self) -> float:
        return np.random.normal(self.mu, self.sigma)
    
    def expectation(self, f: Callable[[float], float]) -> float:
        """Estimation par MC car pas de forme analytique générale."""
        samples = [self.sample() for _ in range(10000)]
        return sum(f(x) for x in samples) / len(samples)


class Constant(Distribution[X]):
    """
    Distribution dégénérée (une seule valeur avec probabilité 1).
    
    Utile pour représenter des transitions déterministes.
    
    Exemple:
        >>> always_zero = Constant(0)
        >>> always_zero.sample()  # Retourne toujours 0
    """
    
    def __init__(self, value: X):
        self.value = value
    
    def sample(self) -> X:
        return self.value
    
    def expectation(self, f: Callable[[X], float]) -> float:
        return f(self.value)


class Poisson(Distribution[int]):
    """
    Distribution de Poisson.
    
    Modélise le nombre d'événements dans un intervalle de temps,
    quand ces événements arrivent à taux constant et indépendamment.
    
    Paramètre λ (lambda): taux moyen d'événements
    
    Exemples d'utilisation:
    - Nombre de clients arrivant dans un magasin par heure
    - Nombre de demandes de stock par jour
    - Nombre d'appels téléphoniques par minute
    
    Propriétés:
    - E[X] = λ (espérance = lambda)
    - Var[X] = λ (variance = lambda aussi)
    
    Exemple - Demande en inventaire:
        >>> demand = Poisson(lam=3.0)  # En moyenne 3 demandes par jour
        >>> demand.sample()  # 0, 1, 2, 3, 4, 5... (entiers non négatifs)
    """
    
    def __init__(self, lam: float):
        """
        Args:
            lam: Paramètre λ (taux moyen), doit être > 0
        """
        self.lam = lam
    
    def sample(self) -> int:
        return np.random.poisson(self.lam)
    
    def expectation(self, f: Callable[[int], float]) -> float:
        """
        Calcul par somme tronquée (les probabilités décroissent exponentiellement).
        """
        total = 0.0
        prob_sum = 0.0
        k = 0
        while prob_sum < 0.9999 or k < self.lam + 10:
            from math import factorial
            prob = np.exp(-self.lam) * (self.lam ** k) / factorial(k)
            total += prob * f(k)
            prob_sum += prob
            k += 1
            if k > 100:  # Safety limit
                break
        return total


# =============================================================================
# SECTION 3: ÉTATS ET TRANSITIONS
# =============================================================================

@dataclass(frozen=True)
class State(Generic[S]):
    """
    Classe de base pour les états dans un processus de Markov.
    
    frozen=True rend l'objet immutable (non modifiable après création),
    ce qui est important pour l'utiliser comme clé de dictionnaire.
    """
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    """
    État non-terminal: un état depuis lequel on peut encore agir.
    
    Exemple dans un jeu d'échecs:
        - Position où la partie continue = NonTerminal
        - Mat ou pat = Terminal
    """
    pass


@dataclass(frozen=True)
class Terminal(State[S]):
    """
    État terminal: un état absorbant où l'épisode se termine.
    
    Dans un état terminal:
    - Aucune action n'est possible
    - La valeur future est 0
    - L'agent ne peut plus accumuler de récompenses
    """
    pass


@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    """
    Une étape de transition: (état, récompense, état_suivant).
    
    C'est l'unité atomique d'expérience en RL.
    
    Exemple - Robot dans une grille:
        TransitionStep(
            state=NonTerminal((0, 0)),      # Position initiale
            reward=-1,                        # Coût du mouvement
            next_state=NonTerminal((0, 1))   # Nouvelle position
        )
    """
    state: NonTerminal[S]
    reward: float
    next_state: State[S]


@dataclass(frozen=True)
class ActionStep(Generic[S, A]):
    """
    Une étape avec action: (état, action, récompense, état_suivant).
    
    Utilisé dans les MDP (Markov Decision Process) où l'agent choisit des actions.
    
    Exemple - Trading:
        ActionStep(
            state=NonTerminal(portfolio),
            action='buy_10_shares',
            reward=-transaction_cost,
            next_state=NonTerminal(new_portfolio)
        )
    """
    state: NonTerminal[S]
    action: A
    reward: float
    next_state: State[S]


# =============================================================================
# SECTION 4: PROCESSUS DE MARKOV
# =============================================================================

class MarkovProcess(ABC, Generic[S]):
    """
    Processus de Markov (MP): séquence d'états où la transition
    ne dépend que de l'état actuel (propriété de Markov).
    
    PROPRIÉTÉ DE MARKOV (fondamentale!):
    P(S_{t+1} | S_t, S_{t-1}, ..., S_0) = P(S_{t+1} | S_t)
    
    En mots: "Le futur ne dépend du passé qu'à travers le présent"
    
    Exemples:
    - ✓ Position d'une pièce sur un échiquier
    - ✗ Cours d'une action (dépend de l'historique pour certaines stratégies)
    
    Comment rendre un processus markovien:
    Enrichir l'état pour inclure l'historique nécessaire.
    Exemple: état = (position, vitesse) au lieu de juste (position)
    """
    
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        """
        Distribution de l'état suivant depuis un état donné.
        
        Args:
            state: État actuel (non-terminal)
            
        Returns:
            Distribution sur les états suivants possibles
        """
        pass
    
    def simulate(self, start: NonTerminal[S]) -> Iterator[State[S]]:
        """
        Simule une trajectoire à partir d'un état initial.
        
        Génère: S_0, S_1, S_2, ... jusqu'à un état terminal.
        
        Args:
            start: État initial
            
        Yields:
            Séquence d'états
        """
        state: State[S] = start
        while isinstance(state, NonTerminal):
            yield state
            state = self.transition(state).sample()
        yield state  # État terminal final


class MarkovRewardProcess(MarkovProcess[S]):
    """
    MRP (Markov Reward Process): Processus de Markov avec récompenses.
    
    Un MRP ajoute à un MP:
    - R(s): récompense attendue en quittant l'état s
    - γ (gamma): facteur d'actualisation (discount factor)
    
    VALEUR D'UN ÉTAT V(s):
    La somme actualisée espérée des récompenses futures:
    V(s) = E[R_1 + γR_2 + γ²R_3 + ... | S_0 = s]
    
    Le discount γ ∈ [0,1] capture:
    - γ proche de 0: vision court-terme ("l'argent maintenant vaut plus")
    - γ proche de 1: vision long-terme ("le futur compte autant")
    
    ÉQUATION DE BELLMAN pour MRP:
    V(s) = R(s) + γ * Σ P(s,s') * V(s')
    
    En mots: "Valeur = Récompense immédiate + Valeur actualisée attendue du futur"
    """
    
    @abstractmethod
    def transition_reward(
        self, state: NonTerminal[S]
    ) -> Distribution[Tuple[State[S], float]]:
        """
        Distribution jointe de (état_suivant, récompense).
        
        Args:
            state: État actuel
            
        Returns:
            Distribution sur (S', R)
        """
        pass
    
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        """Transition marginale (sans la récompense)."""
        def get_state(sr: Tuple[State[S], float]) -> State[S]:
            return sr[0]
        return self.transition_reward(state).map(get_state)
    
    def simulate_reward(
        self, start: NonTerminal[S]
    ) -> Iterator[TransitionStep[S]]:
        """
        Simule une trajectoire avec les récompenses.
        
        Yields:
            Séquence de TransitionStep
        """
        state = start
        while isinstance(state, NonTerminal):
            next_state, reward = self.transition_reward(state).sample()
            yield TransitionStep(state, reward, next_state)
            state = next_state


class FiniteMarkovRewardProcess(MarkovRewardProcess[S]):
    """
    MRP fini: ensemble d'états fini avec transitions explicites.
    
    Permet des calculs exacts via algèbre linéaire.
    
    La fonction de valeur satisfait:
    V = R + γ * P * V
    
    Où:
    - V est le vecteur des valeurs [V(s_1), V(s_2), ...]
    - R est le vecteur des récompenses [R(s_1), R(s_2), ...]
    - P est la matrice de transition P[i,j] = P(S'=s_j | S=s_i)
    
    Solution analytique (si (I - γP) est inversible):
    V = (I - γP)^{-1} * R
    """
    
    def __init__(
        self, 
        transition_reward_map: Mapping[
            NonTerminal[S], 
            Categorical[Tuple[State[S], float]]
        ]
    ):
        """
        Args:
            transition_reward_map: Dict état -> distribution(état_suivant, récompense)
        """
        self.transition_reward_map = transition_reward_map
        self.non_terminal_states = list(transition_reward_map.keys())
    
    def transition_reward(
        self, state: NonTerminal[S]
    ) -> Distribution[Tuple[State[S], float]]:
        return self.transition_reward_map[state]
    
    def get_value_function_matrix(
        self, gamma: float
    ) -> Mapping[NonTerminal[S], float]:
        """
        Calcule V exactement par inversion matricielle.
        
        V = (I - γP)^{-1} * R
        
        Args:
            gamma: Facteur d'actualisation
            
        Returns:
            Dict état -> valeur
        """
        n = len(self.non_terminal_states)
        state_to_idx = {s: i for i, s in enumerate(self.non_terminal_states)}
        
        # Construire la matrice P et le vecteur R
        P = np.zeros((n, n))
        R = np.zeros(n)
        
        for s in self.non_terminal_states:
            i = state_to_idx[s]
            dist = self.transition_reward_map[s]
            
            # Calculer R(s) et P(s, s')
            for (next_state, reward), prob in dist.probabilities.items():
                R[i] += prob * reward
                if isinstance(next_state, NonTerminal):
                    j = state_to_idx[next_state]
                    P[i, j] += prob
        
        # Résoudre (I - γP) * V = R
        A = np.eye(n) - gamma * P
        V_vec = np.linalg.solve(A, R)
        
        return {s: V_vec[state_to_idx[s]] for s in self.non_terminal_states}


# =============================================================================
# SECTION 5: PROCESSUS DE DÉCISION MARKOVIEN (MDP)
# =============================================================================

class Policy(ABC, Generic[S, A]):
    """
    Politique: règle de décision qui mappe états → actions.
    
    Une politique π peut être:
    - Déterministe: π(s) = a (une action fixe par état)
    - Stochastique: π(a|s) = probabilité de choisir a dans l'état s
    
    En RL, on cherche la politique optimale π* qui maximise
    les récompenses cumulées.
    
    Exemple - Trading:
    - État: (prix_actuel, position)
    - Actions: {acheter, vendre, conserver}
    - Politique: "acheter si prix < moyenne mobile, sinon conserver"
    """
    
    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        """
        Retourne la distribution d'actions pour un état donné.
        
        Args:
            state: État actuel
            
        Returns:
            Distribution sur les actions
        """
        pass


class DeterministicPolicy(Policy[S, A]):
    """
    Politique déterministe: une action fixe par état.
    
    π(s) = a (pas de randomisation)
    
    Les politiques optimales dans les MDP finis sont toujours
    déterministes (théorème fondamental).
    """
    
    def __init__(self, action_for: Callable[[NonTerminal[S]], A]):
        """
        Args:
            action_for: Fonction état -> action
        """
        self.action_for = action_for
    
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        return Constant(self.action_for(state))


class StochasticPolicy(Policy[S, A]):
    """
    Politique stochastique: distribution d'actions par état.
    
    π(a|s) = probabilité de choisir l'action a dans l'état s
    
    Utile pour:
    - Exploration (essayer différentes actions)
    - Environnements adversariaux (être imprévisible)
    - Régularisation (éviter l'overfitting)
    """
    
    def __init__(
        self, 
        policy_map: Callable[[NonTerminal[S]], Distribution[A]]
    ):
        self.policy_map = policy_map
    
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        return self.policy_map(state)


class MarkovDecisionProcess(ABC, Generic[S, A]):
    """
    MDP (Markov Decision Process): MRP avec actions.
    
    Un MDP est défini par:
    - S: ensemble d'états
    - A: ensemble d'actions
    - P(s'|s,a): probabilités de transition
    - R(s,a): récompense pour action a dans état s
    - γ: facteur d'actualisation
    
    C'est le cadre mathématique fondamental du RL!
    
    FONCTIONS DE VALEUR:
    
    1. V^π(s) = Valeur d'un état sous politique π
       "Combien de récompenses futures espère-t-on en partant de s?"
       
    2. Q^π(s,a) = Valeur action-état sous politique π
       "Combien si on fait a dans s, puis on suit π?"
    
    ÉQUATIONS DE BELLMAN pour MDP:
    
    V^π(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_s' P(s'|s,a) * V^π(s')]
    
    Q^π(s,a) = R(s,a) + γ * Σ_s' P(s'|s,a) * V^π(s')
    
    OPTIMALITÉ:
    
    V*(s) = max_a Q*(s,a) = max_a [R(s,a) + γ * Σ_s' P(s'|s,a) * V*(s')]
    
    La politique optimale est gloutonne par rapport à Q*:
    π*(s) = argmax_a Q*(s,a)
    """
    
    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        """
        Actions disponibles dans un état.
        
        Args:
            state: État actuel
            
        Returns:
            Itérable d'actions possibles
        """
        pass
    
    @abstractmethod
    def step(
        self, 
        state: NonTerminal[S], 
        action: A
    ) -> Distribution[Tuple[State[S], float]]:
        """
        Distribution de (état_suivant, récompense) pour une action.
        
        Args:
            state: État actuel
            action: Action choisie
            
        Returns:
            Distribution sur (S', R)
        """
        pass
    
    def simulate(
        self, 
        policy: Policy[S, A], 
        start: NonTerminal[S]
    ) -> Iterator[ActionStep[S, A]]:
        """
        Simule une trajectoire avec actions.
        
        Args:
            policy: Politique à suivre
            start: État initial
            
        Yields:
            Séquence de ActionStep
        """
        state = start
        while isinstance(state, NonTerminal):
            action = policy.act(state).sample()
            next_state, reward = self.step(state, action).sample()
            yield ActionStep(state, action, reward, next_state)
            state = next_state


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    """
    MDP fini avec transitions explicites.
    
    Permet d'appliquer la DP (Dynamic Programming) exacte:
    - Policy Evaluation (évaluation de politique)
    - Policy Iteration (itération de politique)
    - Value Iteration (itération de valeur)
    """
    
    def __init__(
        self,
        mapping: Mapping[
            NonTerminal[S],
            Mapping[A, Categorical[Tuple[State[S], float]]]
        ]
    ):
        """
        Args:
            mapping: état -> action -> distribution(état_suivant, récompense)
        """
        self.mapping = mapping
        self.non_terminal_states = list(mapping.keys())
    
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        return self.mapping[state].keys()
    
    def step(
        self, 
        state: NonTerminal[S], 
        action: A
    ) -> Distribution[Tuple[State[S], float]]:
        return self.mapping[state][action]


# =============================================================================
# SECTION 6: APPROXIMATION DE FONCTIONS
# =============================================================================

class FunctionApprox(ABC, Generic[X]):
    """
    Approximateur de fonction: représente une fonction paramétrique f_w(x).
    
    En RL, on utilise des approximateurs pour:
    - V(s; w) ≈ V*(s): estimer la fonction de valeur
    - Q(s,a; w) ≈ Q*(s,a): estimer la fonction Q
    - π(a|s; θ): paramétrer une politique
    
    L'approximation est nécessaire quand l'espace d'états est:
    - Continu (infinité d'états)
    - Très grand (explosion combinatoire)
    
    Types d'approximateurs:
    1. Tabular: un paramètre par état (exact mais limité)
    2. Linear: f(x) = φ(x)ᵀw (features linéaires)
    3. DNN (Deep Neural Network): réseaux de neurones
    """
    
    @abstractmethod
    def evaluate(self, x_values: Iterable[X]) -> np.ndarray:
        """
        Évalue la fonction sur plusieurs entrées.
        
        Args:
            x_values: Points d'évaluation
            
        Returns:
            Array des valeurs f(x) pour chaque x
        """
        pass
    
    def __call__(self, x: X) -> float:
        """Évalue en un point: f(x)."""
        return self.evaluate([x])[0]
    
    @abstractmethod
    def update(
        self, 
        xy_values: Iterable[Tuple[X, float]]
    ) -> FunctionApprox[X]:
        """
        Met à jour l'approximateur avec des paires (x, y).
        
        Effectue une étape de gradient descent vers y = f(x).
        
        Args:
            xy_values: Paires (entrée, cible)
            
        Returns:
            Nouvel approximateur mis à jour
        """
        pass


class Tabular(FunctionApprox[X]):
    """
    Approximation tabulaire: stocke une valeur par entrée.
    
    C'est le cas exact (pas vraiment une approximation) utilisable
    uniquement quand l'espace d'états est petit et discret.
    
    Mise à jour:
    V(s) ← V(s) + α * (y - V(s))
    
    où α est le learning rate (taux d'apprentissage).
    
    Exemple - Tic-tac-toe:
        Il y a environ 5000 états possibles, donc tabulaire fonctionne.
    
    Contre-exemple - Échecs:
        Il y a ~10^43 états, impossible en tabulaire!
    """
    
    def __init__(
        self,
        values_map: Optional[Mapping[X, float]] = None,
        count_to_weight_func: Callable[[int], float] = lambda n: 1.0 / n
    ):
        """
        Args:
            values_map: Valeurs initiales {x: V(x)}
            count_to_weight_func: Fonction n -> α(n) pour le learning rate
                                   Par défaut: 1/n (moyenne mobile)
        """
        self.values_map: Dict[X, float] = dict(values_map or {})
        self.counts: Dict[X, int] = defaultdict(int)
        self.count_to_weight_func = count_to_weight_func
    
    def evaluate(self, x_values: Iterable[X]) -> np.ndarray:
        return np.array([self.values_map.get(x, 0.0) for x in x_values])
    
    def update(
        self, 
        xy_values: Iterable[Tuple[X, float]]
    ) -> Tabular[X]:
        """
        Mise à jour incrémentale avec learning rate adaptatif.
        
        V(x) ← V(x) + α(n) * (y - V(x))
        
        où n est le nombre de fois que x a été vu.
        """
        new_values = dict(self.values_map)
        new_counts = dict(self.counts)
        
        for x, y in xy_values:
            new_counts[x] = new_counts.get(x, 0) + 1
            alpha = self.count_to_weight_func(new_counts[x])
            old_val = new_values.get(x, 0.0)
            new_values[x] = old_val + alpha * (y - old_val)
        
        result = Tabular(new_values, self.count_to_weight_func)
        result.counts = defaultdict(int, new_counts)
        return result


class LinearFunctionApprox(FunctionApprox[X]):
    """
    Approximation linéaire: f(x) = φ(x)ᵀ · w
    
    Où:
    - φ(x): vecteur de features (caractéristiques) de l'entrée x
    - w: vecteur de poids (paramètres à apprendre)
    
    Les features φ(x) encodent les propriétés importantes de x.
    
    Exemple - Valeur d'une maison:
        x = maison
        φ(x) = [1, surface, nb_chambres, distance_centre, ...]
        f(x) = w₀ + w₁·surface + w₂·nb_chambres + ...
    
    Exemple en RL - Valeur d'un état:
        x = état du jeu
        φ(x) = [nb_pieces, controle_centre, securite_roi, ...]
        V(x) = wᵀ · φ(x)
    
    AVANTAGES:
    - Généralisation: états similaires ont valeurs similaires
    - Efficacité: O(d) paramètres au lieu de O(|S|)
    - Convergence garantie pour TD linéaire
    
    MISE À JOUR (Gradient Descent):
    Erreur: δ = y - f(x) = y - wᵀφ(x)
    Gradient: ∇_w (½δ²) = -δ · φ(x)
    Update: w ← w + α · δ · φ(x)
    """
    
    def __init__(
        self,
        feature_functions: Sequence[Callable[[X], float]],
        weights: Optional[np.ndarray] = None,
        learning_rate: float = 0.01,
        regularization: float = 0.0
    ):
        """
        Args:
            feature_functions: Liste de fonctions [φ₁, φ₂, ..., φ_d]
            weights: Poids initiaux (default: zéros)
            learning_rate: Taux d'apprentissage α
            regularization: Coefficient de régularisation L2 (ridge)
        """
        self.feature_functions = feature_functions
        self.num_features = len(feature_functions)
        self.weights = weights if weights is not None else np.zeros(self.num_features)
        self.learning_rate = learning_rate
        self.regularization = regularization
    
    def get_features(self, x: X) -> np.ndarray:
        """Calcule le vecteur de features φ(x)."""
        return np.array([f(x) for f in self.feature_functions])
    
    def evaluate(self, x_values: Iterable[X]) -> np.ndarray:
        """f(x) = φ(x)ᵀ · w pour chaque x."""
        x_list = list(x_values)
        features = np.array([self.get_features(x) for x in x_list])
        return features @ self.weights
    
    def update(
        self, 
        xy_values: Iterable[Tuple[X, float]]
    ) -> LinearFunctionApprox[X]:
        """
        Mise à jour par gradient descent.
        
        Pour chaque (x, y):
            δ = y - wᵀφ(x)
            w ← w + α·δ·φ(x) - α·λ·w  (avec régularisation)
        """
        new_weights = self.weights.copy()
        
        for x, y in xy_values:
            features = self.get_features(x)
            prediction = np.dot(features, new_weights)
            error = y - prediction
            
            # Gradient descent avec régularisation L2
            new_weights += self.learning_rate * (
                error * features - self.regularization * new_weights
            )
        
        return LinearFunctionApprox(
            self.feature_functions,
            new_weights,
            self.learning_rate,
            self.regularization
        )


# =============================================================================
# SECTION 7: PROGRAMMATION DYNAMIQUE (DP)
# =============================================================================

def policy_evaluation(
    mdp: FiniteMarkovDecisionProcess[S, A],
    policy: Policy[S, A],
    gamma: float,
    tolerance: float = 1e-6
) -> Mapping[NonTerminal[S], float]:
    """
    Policy Evaluation (Évaluation de Politique): calcule V^π.
    
    C'est la première brique de la DP (Dynamic Programming).
    
    ALGORITHME:
    1. Initialiser V(s) = 0 pour tout s
    2. Répéter jusqu'à convergence:
       Pour chaque état s:
           V(s) ← Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
    
    C'est l'application itérative de l'opérateur de Bellman B^π:
    B^π(V)(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
    
    CONVERGENCE: B^π est une contraction, donc V_n → V^π
    
    Args:
        mdp: MDP fini
        policy: Politique à évaluer
        gamma: Facteur d'actualisation
        tolerance: Seuil de convergence
        
    Returns:
        Fonction de valeur V^π
    """
    # Initialisation
    V: Dict[NonTerminal[S], float] = {
        s: 0.0 for s in mdp.non_terminal_states
    }
    
    while True:
        delta = 0.0  # Changement maximal
        new_V: Dict[NonTerminal[S], float] = {}
        
        for s in mdp.non_terminal_states:
            # Calculer la nouvelle valeur
            value = 0.0
            actions_list = list(mdp.actions(s))
            
            for a in actions_list:
                # Probabilité de l'action sous la politique
                action_dist = policy.act(s)
                if hasattr(action_dist, 'probability'):
                    action_prob = action_dist.probability(a)
                else:
                    action_prob = 1.0 / len(actions_list)
                
                # Espérance sur les transitions
                transition_dist = mdp.step(s, a)
                expected_value = transition_dist.expectation(
                    lambda sr: sr[1] + gamma * (V.get(sr[0], 0.0) if isinstance(sr[0], NonTerminal) else 0.0)
                )
                
                value += action_prob * expected_value
            
            new_V[s] = value
            delta = max(delta, abs(value - V[s]))
        
        V = new_V
        
        if delta < tolerance:
            break
    
    return V


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    tolerance: float = 1e-6
) -> Tuple[DeterministicPolicy[S, A], Mapping[NonTerminal[S], float]]:
    """
    Value Iteration (Itération de Valeur): trouve V* directement.
    
    Combine évaluation et amélioration en une seule étape.
    
    ALGORITHME:
    1. Initialiser V(s) = 0 pour tout s
    2. Répéter jusqu'à convergence:
       Pour chaque état s:
           V(s) ← max_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
    
    C'est l'application itérative de l'opérateur de Bellman optimal B*:
    B*(V)(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
    
    CONVERGENCE: V_n → V* car B* est une contraction.
    
    Args:
        mdp: MDP fini
        gamma: Facteur d'actualisation
        tolerance: Seuil de convergence
        
    Returns:
        (politique_optimale, V*)
    """
    # Initialisation
    V: Dict[NonTerminal[S], float] = {
        s: 0.0 for s in mdp.non_terminal_states
    }
    
    while True:
        delta = 0.0
        new_V: Dict[NonTerminal[S], float] = {}
        
        for s in mdp.non_terminal_states:
            # Trouver la meilleure action
            best_value = float('-inf')
            
            for a in mdp.actions(s):
                transition_dist = mdp.step(s, a)
                q_value = transition_dist.expectation(
                    lambda sr: sr[1] + gamma * (V.get(sr[0], 0.0) if isinstance(sr[0], NonTerminal) else 0.0)
                )
                best_value = max(best_value, q_value)
            
            new_V[s] = best_value
            delta = max(delta, abs(best_value - V[s]))
        
        V = new_V
        
        if delta < tolerance:
            break
    
    # Extraire la politique optimale
    def greedy_action(s: NonTerminal[S]) -> A:
        best_action = None
        best_value = float('-inf')
        
        for a in mdp.actions(s):
            transition_dist = mdp.step(s, a)
            q_value = transition_dist.expectation(
                lambda sr: sr[1] + gamma * (V.get(sr[0], 0.0) if isinstance(sr[0], NonTerminal) else 0.0)
            )
            if q_value > best_value:
                best_value = q_value
                best_action = a
        
        return best_action
    
    policy = DeterministicPolicy(greedy_action)
    
    return policy, V


# =============================================================================
# SECTION 8: MÉTHODES MONTE CARLO (MC)
# =============================================================================

def mc_prediction(
    traces: Iterable[Iterable[TransitionStep[S]]],
    approx: FunctionApprox[NonTerminal[S]],
    gamma: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    """
    MC Prediction (Prédiction Monte Carlo): estime V^π par échantillonnage.
    
    MC = Monte Carlo: estimer par la moyenne d'échantillons.
    
    IDÉE CLEF:
    V(s) = E[G_t | S_t = s]
    
    où G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... est le RETOUR (somme actualisée)
    
    ALGORITHME (Every-visit MC):
    Pour chaque épisode:
        Calculer G_t pour chaque visite de chaque état
        V(s) ← V(s) + α(G_t - V(s))
    
    PROPRIÉTÉS:
    - Non-biaisé: E[G_t] = V^π(s) exactement
    - Variance élevée: G_t varie beaucoup d'un épisode à l'autre
    - Nécessite des épisodes complets (pas online)
    
    Args:
        traces: Itérateur d'épisodes (séquences de TransitionStep)
        approx: Approximateur de valeur initial
        gamma: Facteur d'actualisation
        
    Yields:
        Approximateurs successifs (un par épisode)
    """
    for episode in traces:
        episode_list = list(episode)
        if not episode_list:
            yield approx
            continue
        
        # Calculer les retours pour chaque pas
        # G_t = R_{t+1} + γG_{t+1}
        returns: List[Tuple[NonTerminal[S], float]] = []
        G = 0.0
        
        # Parcourir l'épisode à l'envers pour calculer les retours
        for step in reversed(episode_list):
            G = step.reward + gamma * G
            returns.append((step.state, G))
        
        # Mettre à jour l'approximateur
        approx = approx.update(returns)
        yield approx


# =============================================================================
# SECTION 9: APPRENTISSAGE PAR DIFFÉRENCE TEMPORELLE (TD)
# =============================================================================

def td_prediction(
    transitions: Iterable[TransitionStep[S]],
    approx: FunctionApprox[NonTerminal[S]],
    gamma: float
) -> Iterator[FunctionApprox[NonTerminal[S]]]:
    """
    TD Prediction (Prédiction TD): estime V^π en temps réel.
    
    TD = Temporal Difference (Différence Temporelle)
    
    IDÉE CLEF (bootstrapping):
    Au lieu d'attendre le vrai retour G_t (comme MC),
    on utilise une ESTIMATION:
    
    G_t ≈ R_{t+1} + γV(S_{t+1})    (TD Target)
    
    ERREUR TD:
    δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
    
    MISE À JOUR TD(0):
    V(S_t) ← V(S_t) + α·δ_t
    
    PROPRIÉTÉS:
    - Biaisé: utilise V(S_{t+1}) qui est une estimation
    - Variance faible: pas de somme de récompenses aléatoires
    - Online: mise à jour après chaque transition
    
    Args:
        transitions: Flux de transitions (S, R, S')
        approx: Approximateur initial
        gamma: Facteur d'actualisation
        
    Yields:
        Approximateurs successifs (un par transition)
    """
    for step in transitions:
        # TD Target: R + γV(S')
        next_value = approx(step.next_state) if isinstance(step.next_state, NonTerminal) else 0.0
        td_target = step.reward + gamma * next_value
        
        # Mise à jour: V(S) vers TD Target
        approx = approx.update([(step.state, td_target)])
        yield approx


# =============================================================================
# SECTION 10: CONTRÔLE TD (SARSA, Q-LEARNING)
# =============================================================================

def sarsa(
    mdp: MarkovDecisionProcess[S, A],
    states: Distribution[NonTerminal[S]],
    approx: FunctionApprox[Tuple[NonTerminal[S], A]],
    gamma: float,
    epsilon: float = 0.1,
    num_episodes: int = 1000
) -> Iterator[FunctionApprox[Tuple[NonTerminal[S], A]]]:
    """
    SARSA: On-policy TD Control.
    
    SARSA = State-Action-Reward-State-Action
    
    Args:
        mdp: MDP
        states: Distribution des états initiaux
        approx: Approximateur Q initial
        gamma: Facteur d'actualisation
        epsilon: Paramètre d'exploration
        num_episodes: Nombre d'épisodes
        
    Yields:
        Approximateurs Q successifs
    """
    Q = approx
    
    def epsilon_greedy(s: NonTerminal[S]) -> A:
        actions = list(mdp.actions(s))
        if random.random() < epsilon:
            return random.choice(actions)
        else:
            return max(actions, key=lambda a: Q((s, a)))
    
    for _ in range(num_episodes):
        state = states.sample()
        action = epsilon_greedy(state)
        
        while isinstance(state, NonTerminal):
            next_state, reward = mdp.step(state, action).sample()
            
            if isinstance(next_state, NonTerminal):
                next_action = epsilon_greedy(next_state)
                td_target = reward + gamma * Q((next_state, next_action))
            else:
                td_target = reward
                next_action = None
            
            Q = Q.update([((state, action), td_target)])
            
            state = next_state
            action = next_action
        
        yield Q


def q_learning(
    mdp: MarkovDecisionProcess[S, A],
    states: Distribution[NonTerminal[S]],
    approx: FunctionApprox[Tuple[NonTerminal[S], A]],
    gamma: float,
    epsilon: float = 0.1,
    num_episodes: int = 1000
) -> Iterator[FunctionApprox[Tuple[NonTerminal[S], A]]]:
    """
    Q-Learning: Off-policy TD Control.
    
    L'algorithme le plus célèbre du RL!
    
    Args:
        mdp: MDP
        states: Distribution des états initiaux
        approx: Approximateur Q initial
        gamma: Facteur d'actualisation
        epsilon: Paramètre d'exploration
        num_episodes: Nombre d'épisodes
        
    Yields:
        Approximateurs Q successifs
    """
    Q = approx
    
    for _ in range(num_episodes):
        state = states.sample()
        
        while isinstance(state, NonTerminal):
            # Behaviour policy: ε-greedy
            actions = list(mdp.actions(state))
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(actions, key=lambda a: Q((state, a)))
            
            # Exécuter l'action
            next_state, reward = mdp.step(state, action).sample()
            
            # Target policy: greedy (max)
            if isinstance(next_state, NonTerminal):
                next_actions = list(mdp.actions(next_state))
                max_q = max(Q((next_state, a)) for a in next_actions)
                td_target = reward + gamma * max_q
            else:
                td_target = reward
            
            # Mise à jour Q
            Q = Q.update([((state, action), td_target)])
            
            state = next_state
        
        yield Q


# =============================================================================
# SECTION 11: APPLICATIONS FINANCIÈRES
# =============================================================================

def cara_utility(x: float, a: float) -> float:
    """
    Utilité CARA (Constant Absolute Risk Aversion).
    
    U(x) = (1 - e^{-a·x}) / a
    
    Args:
        x: Richesse
        a: Coefficient d'aversion
        
    Returns:
        Utilité de la richesse x
    """
    if abs(a) < 1e-10:
        return x
    return (1 - np.exp(-a * x)) / a


def crra_utility(x: float, gamma: float) -> float:
    """
    Utilité CRRA (Constant Relative Risk Aversion).
    
    U(x) = (x^{1-γ} - 1) / (1 - γ)  si γ ≠ 1
    U(x) = log(x)                    si γ = 1
    
    Args:
        x: Richesse (doit être > 0)
        gamma: Coefficient d'aversion relative
        
    Returns:
        Utilité de la richesse x
    """
    if x <= 0:
        return float('-inf')
    if abs(gamma - 1) < 1e-10:
        return np.log(x)
    return (x ** (1 - gamma) - 1) / (1 - gamma)


@dataclass
class MertonPortfolioProblem:
    """
    Problème de Merton: allocation optimale avec consommation.
    
    SETUP:
    - Un actif risqué: prix S_t suit dS/S = μdt + σdW
    - Un actif sans risque: rendement r
    - Un agent avec utilité CRRA et horizon T
    
    SOLUTION ANALYTIQUE (cas CRRA):
    - Allocation optimale: π* = (μ - r) / (γ σ²) (constante!)
    """
    mu: float       # Rendement espéré de l'actif risqué
    sigma: float    # Volatilité de l'actif risqué
    r: float        # Taux sans risque
    gamma: float    # Aversion au risque CRRA
    T: float        # Horizon
    
    def optimal_allocation(self) -> float:
        """
        Allocation optimale dans l'actif risqué.
        
        π* = (μ - r) / (γ σ²)
        """
        return (self.mu - self.r) / (self.gamma * self.sigma ** 2)


@dataclass(frozen=True)
class InventoryState:
    """État pour le problème de gestion d'inventaire."""
    on_hand: int
    on_order: int


class InventoryMDP(MarkovDecisionProcess[InventoryState, int]):
    """
    Gestion d'inventaire comme MDP.
    
    POLITIQUE OPTIMALE (s, S):
    - Si inventaire < s: commander jusqu'à S
    - Sinon: ne rien commander
    """
    
    def __init__(
        self,
        capacity: int,
        holding_cost: float,
        stockout_cost: float,
        poisson_lambda: float
    ):
        self.capacity = capacity
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.poisson_lambda = poisson_lambda
    
    def actions(self, state: NonTerminal[InventoryState]) -> Iterable[int]:
        s = state.state
        total_stock = s.on_hand + s.on_order
        max_order = max(0, self.capacity - total_stock)
        return range(max_order + 1)
    
    def step(
        self,
        state: NonTerminal[InventoryState],
        action: int
    ) -> Distribution[Tuple[State[InventoryState], float]]:
        s = state.state
        
        def sample_next() -> Tuple[State[InventoryState], float]:
            new_on_hand = s.on_hand + s.on_order
            demand = np.random.poisson(self.poisson_lambda)
            
            sold = min(new_on_hand, demand)
            stockout = max(0, demand - new_on_hand)
            remaining = new_on_hand - sold
            
            holding_cost = self.holding_cost * remaining
            stockout_cost = self.stockout_cost * stockout
            reward = -(holding_cost + stockout_cost)
            
            new_state = InventoryState(on_hand=remaining, on_order=action)
            
            return (NonTerminal(new_state), reward)
        
        return SampledDistribution(sample_next)


# =============================================================================
# SECTION 12: DÉMONSTRATIONS
# =============================================================================

def demo_distributions():
    """Démonstration des distributions."""
    print("=" * 60)
    print("DÉMONSTRATION: Distributions de probabilité")
    print("=" * 60)
    
    die = Categorical({1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6})
    print(f"\nDé équilibré - 5 tirages: {die.sample_n(5)}")
    print(f"Espérance E[X]: {die.expectation(lambda x: x):.2f}")
    
    gauss = Gaussian(mu=170, sigma=10)
    print(f"\nTaille humaine N(170, 10) - 5 tirages: {[f'{x:.1f}' for x in gauss.sample_n(5)]}")
    
    demand = Poisson(lam=3.0)
    print(f"\nDemande Poisson(3) - 10 tirages: {demand.sample_n(10)}")


def demo_merton():
    """Démonstration du problème de Merton."""
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: Problème de Merton")
    print("=" * 60)
    
    problem = MertonPortfolioProblem(
        mu=0.08, sigma=0.20, r=0.03, gamma=2.0, T=10.0
    )
    
    print(f"\nParamètres:")
    print(f"  μ (rendement espéré): {problem.mu:.1%}")
    print(f"  σ (volatilité): {problem.sigma:.1%}")
    print(f"  r (taux sans risque): {problem.r:.1%}")
    print(f"  γ (aversion au risque): {problem.gamma}")
    
    optimal_alloc = problem.optimal_allocation()
    print(f"\nAllocation optimale: π* = {optimal_alloc:.1%}")


def demo_gridworld():
    """Démonstration Value Iteration sur GridWorld."""
    print("\n" + "=" * 60)
    print("DÉMONSTRATION: Value Iteration sur GridWorld 3x3")
    print("=" * 60)
    
    grid_size = 3
    goal = 8
    
    def get_neighbors(pos: int) -> Dict[str, int]:
        row, col = pos // grid_size, pos % grid_size
        neighbors = {}
        if row > 0: neighbors['up'] = pos - grid_size
        if row < grid_size - 1: neighbors['down'] = pos + grid_size
        if col > 0: neighbors['left'] = pos - 1
        if col < grid_size - 1: neighbors['right'] = pos + 1
        return neighbors
    
    mapping = {}
    for pos in range(grid_size * grid_size):
        if pos == goal:
            continue
        nt = NonTerminal(pos)
        actions = {}
        neighbors = get_neighbors(pos)
        
        for action, next_pos in neighbors.items():
            if next_pos == goal:
                actions[action] = Categorical({(Terminal(next_pos), 0.0): 1.0})
            else:
                actions[action] = Categorical({(NonTerminal(next_pos), -1.0): 1.0})
        
        if not actions:
            actions['stay'] = Categorical({(NonTerminal(pos), -1.0): 1.0})
        
        mapping[nt] = actions
    
    mdp = FiniteMarkovDecisionProcess(mapping)
    policy, V = value_iteration(mdp, gamma=0.9)
    
    print("\nValeurs optimales V*(s):")
    for pos in range(grid_size * grid_size):
        if pos == goal:
            print(f"  Position {pos}: GOAL")
        else:
            print(f"  Position {pos}: V* = {V.get(NonTerminal(pos), 0):.2f}")


def main():
    """Point d'entrée principal."""
    print("=" * 70)
    print("  FOUNDATIONS OF REINFORCEMENT LEARNING WITH APPLICATIONS IN FINANCE")
    print("=" * 70)
    
    demo_distributions()
    demo_merton()
    demo_gridworld()
    
    print("\n" + "=" * 70)
    print("  FIN DES DÉMONSTRATIONS")
    print("=" * 70)


if __name__ == "__main__":
    main()