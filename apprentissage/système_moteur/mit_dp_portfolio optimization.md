# 📚 MIT 15.450 - PROGRAMMATION DYNAMIQUE POUR L'OPTIMISATION DE PORTEFEUILLE
## Avec Prédictibilité des Rendements et Contraintes de Marge

**Source**: MIT OpenCourseWare 15.450 - Analytics of Finance (supp06.m)  
**Auteur Original**: L. Kogan, 05/10/2010  
**Conversion**: MATLAB → Python pour HelixOne  
**Date**: 2026-01-29

---

## 📋 TABLE DES MATIÈRES

1. [Glossaire des Termes](#glossaire-des-termes)
2. [Description du Problème](#description-du-problème)
3. [Modèle Mathématique](#modèle-mathématique)
4. [Code MATLAB Original](#code-matlab-original)
5. [Code Python Converti](#code-python-converti)
6. [Guide d'Utilisation HelixOne](#guide-dutilisation-helixone)

---

## 📖 GLOSSAIRE DES TERMES

### Acronymes et Termes Techniques

| Terme | Signification Complète | Explication |
|-------|------------------------|-------------|
| **DP** | Dynamic Programming (Programmation Dynamique) | Résolution par récurrence arrière (backward induction) |
| **MDP** | Markov Decision Process (Processus de Décision Markovien) | Framework pour décisions séquentielles |
| **VF** | Value Function (Fonction de Valeur) | J(W,x,t) = espérance d'utilité optimale |
| **AR(1)** | AutoRegressive of order 1 | Processus où x_t dépend de x_{t-1} |
| **CRRA** | Constant Relative Risk Aversion | Utilité U(W) = W^(1-γ)/(1-γ) |
| **CARA** | Constant Absolute Risk Aversion | Utilité U(W) = -exp(-αW) |
| **θ** (theta) | Position/Allocation | Montant investi dans l'actif risqué |
| **α** (alpha) | Risk Aversion | Coefficient d'aversion au risque |
| **λ** (lambda) | Margin Constraint | Contrainte: |θ| ≤ W/λ |
| **ρ** (rho) | Autocorrelation | Persistance du signal prédictif |
| **σ** (sigma) | Volatility | Écart-type des chocs |

### Symboles Mathématiques

| Symbole | Nom | Description |
|---------|-----|-------------|
| **W_t** | Wealth | Richesse du portefeuille au temps t |
| **x_t** | Predictor | Signal prédictif (price spread) |
| **J(W,x,t)** | Value Function | Utilité espérée maximale depuis (W,x,t) |
| **θ*(W,x,t)** | Optimal Policy | Politique d'investissement optimale |
| **dt** | Time Step | Pas de temps (1/12 = mensuel) |
| **T** | Horizon | Horizon d'investissement (5 ans) |

---

## 📊 DESCRIPTION DU PROBLÈME

### Contexte

Un investisseur cherche à **maximiser son utilité terminale** en allouant dynamiquement sa richesse entre:
- Un actif **sans risque** (cash)
- Un actif **risqué** dont le rendement est **partiellement prévisible**

### Caractéristiques Clés

| Aspect | Description |
|--------|-------------|
| **Prédictibilité** | Le rendement de l'actif risqué dépend d'un signal x_t |
| **Contrainte de marge** | Position limitée: \|θ\| ≤ W/λ |
| **Utilité CARA** | U(W) = -exp(-αW), aversion au risque constante |
| **Signal AR(1)** | x_{t+1} = ρ·x_t + ε, mean-reverting |

### Pourquoi ce Problème est Important?

1. **Trading Quantitatif**: Exploiter des signaux prédictifs (momentum, mean-reversion)
2. **Gestion des Risques**: Contraintes de levier réalistes
3. **Allocation Dynamique**: Stratégie qui s'adapte à l'état du marché

---

## 📐 MODÈLE MATHÉMATIQUE

### Dynamique du Signal Prédictif

Le signal x_t suit un processus AR(1) (AutoRegressive d'ordre 1):

$$x_{t+1} = \rho \cdot x_t + \sigma \cdot \varepsilon_{t+1}$$

où:
- $\rho = e^{-0.5 \cdot dt}$ : coefficient d'autocorrélation
- $\sigma$ : volatilité des innovations
- $\varepsilon \sim N(0,1)$ : bruit blanc gaussien

### Dynamique de la Richesse

$$W_{t+1} = W_t + \theta_t \cdot (x_{t+1} - x_t)$$

où $\theta_t$ est la position (montant investi).

### Contrainte de Marge

$$|\theta_t| \leq \frac{W_t}{\lambda}$$

Cette contrainte limite le **levier** : si λ = 0.25, le levier max est 4x.

### Fonction d'Utilité (CARA)

$$U(W) = -\exp(-\alpha \cdot W)$$

où α = 4 est le coefficient d'aversion au risque absolue.

### Équation de Bellman

$$J(W, x, t) = \max_{\theta} \mathbb{E}_t \left[ J(W_{t+1}, x_{t+1}, t+1) \right]$$

avec condition terminale:
$$J(W, x, T) = U(W) = -\exp(-\alpha W)$$

### Résolution par DP (Backward Induction)

1. Initialiser J(W, x, T) = U(W)
2. Pour t = T-dt, T-2dt, ..., 0:
   - Pour chaque (W, x):
     - Calculer J(W, x, t) = max_θ E[J(W', x', t+dt)]
     - Stocker θ*(W, x, t)

---

## 💻 CODE MATLAB ORIGINAL (supp06.m)

```matlab
% Code for the DP solution of the portfolio optimization 
% problem with return predictability and margin constraints
%
% L. Kogan, 05/10/2010


clear all

% Parameters

alpha = 4;
lambda = 0.25;
dt = 1/12;
rho = exp(-0.5 * dt);
sigma = 0.10*sqrt(dt);
T = 5;

% grid 

dx = sigma/3;
sigmass = sqrt(sigma^2/(1 - rho^2)); % steady-state volatility
xmax = 4*sigmass;
Nx = 2*ceil( xmax/ dx) + 1;

Wmax = 4;
NW = 81;
Wgrid = linspace(0,Wmax,NW);
dW = Wgrid(2) - Wgrid(1);

xgrid = linspace(-xmax,xmax,Nx);
dx = 2*xmax / (Nx);

Ntheta = 81;
thetagrid = linspace(-1/lambda,1/lambda,Ntheta);
dtheta = thetagrid(2) - thetagrid(1);

% Utility function

U = -exp(-alpha*Wgrid);

% Transition matrix for x_t

TrM_x = zeros(Nx,Nx); % transition matrix from xgrid(i) to xgrid(j)

 
for i=1:Nx

    p = zeros(1,Nx);
    p(2:end-1) = dx * (1 / sqrt(2*pi*sigma^2)) *...
        exp( -1/(2*sigma^2) .* (xgrid(2:end-1) - rho*xgrid(i)).^2 );
    p(1) = normcdf((xgrid(1) + dx/2 - rho*xgrid(i))/sigma);
    p(end) = 1 - normcdf((xgrid(end) - dx/2 - rho*xgrid(i))/sigma);
    
    p = p ./ sum(p);    % normalize p to add up to 1
    TrM_x(i,:) = p;
    
end

% Bellman iterations

J_next = ones(Nx,1)*U;   % initiate the value function at T
theta_opt = zeros([ceil(T/dt) size(J_next)]); % optimal strategy
t = T;  % time
j=0;    % counter

while t > 1e-12
    
    J = J_next;

    t = t - dt
    j = j+1;
    tic 
    for ix = 1:Nx

        x = xgrid(ix);  % current x   
        p = TrM_x(ix,:); % x transition probs
        
        for iW = 1:NW  

            W = Wgrid(iW);  % current W
            % compute the expected next-period value function for all
            % possible values of control (theta) and maximize
            
            V = zeros(size(thetagrid));
            
            for ntheta=1:Ntheta
                
                theta = W*thetagrid(ntheta);  % current control
               
                W_next = W + theta*(xgrid - x);
                    % compute W next period
                % focus on the region of positive W_next separately    
                W_next = min(W_next,Wmax); % implement interpolation at the
                            % boundaries of W grid
                            
                arg = (W_next > Wgrid(1)); 
                
                J1 = zeros(size(xgrid));
                
                J1(arg) = interp2(Wgrid,xgrid',J_next,...
                             W_next(arg),xgrid(arg),'*linear');
                J1((W_next <= Wgrid(1))) = ...
                    -exp(-alpha*W_next((W_next <= Wgrid(1))));
                
                V(ntheta) = TrM_x(ix,:) * J1';        
                 
            end
            
            [J(ix,iW),n] = max(V);
            theta_opt(j,ix,iW) = thetagrid(n);
            
        end
        
    end
    
    J_next = J;  % update the value function
    
    toc
    
end

 
nW = 21;

figure(1)
hold off
 C = squeeze(theta_opt(1,:,:));
 plot(xgrid,smooth(C(:,nW),9),'b-','LineW',2 );
hold on
 C = squeeze(theta_opt(12,:,:));
 plot(xgrid,smooth(C(:,nW),9),'r-.','LineW',2 );
 C = squeeze(theta_opt(36,:,:));
 plot(xgrid,smooth(C(:,nW),9),'g-<','LineW',2,'MarkerS',2 );
 C = squeeze(theta_opt(60,:,:));
 plot(xgrid,smooth(C(:,nW),9),'m--','LineW',2 );
 axis('square');
 box off
 axis([-.5 .5 -5 5]);
 legend('T=1 ','T=12','T=36','T=60');
 xlabel('Price Spread (X)','FontS',16);
 ylabel('Optimal Policy (\theta^*)','FontS',16);
 
 tmp = num2str(Wgrid(nW));
 title('W = 1','FontS',16); 
 
 
 
figure(3)
surf(Wgrid(1:40),xgrid(1:3:end), J_next(1:3:end,1:40),'LineW',1.5);
axis('square');
box off
ylabel('Price Spread (X)','FontS',14);
xlabel('Portfolio Value (W)','FontS',14);
zlabel('Value Function (J)','FontS',14);
axis([ 0 2 -.5 .5 -1 0]);
```

---

## 🐍 CODE PYTHON CONVERTI

```python
#!/usr/bin/env python3
"""
=============================================================================
MIT 15.450 - PROGRAMMATION DYNAMIQUE POUR L'OPTIMISATION DE PORTEFEUILLE
Avec Prédictibilité des Rendements et Contraintes de Marge
=============================================================================

Ce code résout le problème d'allocation optimale de portefeuille avec:
- Un signal prédictif AR(1) pour les rendements
- Des contraintes de marge (levier limité)
- Une fonction d'utilité CARA (Constant Absolute Risk Aversion)

GLOSSAIRE:
- DP (Dynamic Programming): Programmation dynamique, résolution par backward induction
- CARA (Constant Absolute Risk Aversion): Utilité U(W) = -exp(-αW)
- AR(1) (AutoRegressive order 1): x_{t+1} = ρx_t + σε
- VF (Value Function): J(W,x,t) = utilité espérée maximale
- Margin Constraint: |θ| ≤ W/λ (limite le levier)

Auteur original: L. Kogan, MIT (05/10/2010)
Conversion Python: HelixOne
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class DPParams:
    """
    Paramètres du problème de DP (Dynamic Programming) pour l'optimisation de portefeuille.
    
    Attributs:
        alpha: Coefficient d'aversion au risque (CARA). Plus α est grand, plus l'agent est averse au risque.
               Exemple: α=4 signifie forte aversion au risque.
        
        lambda_margin: Paramètre de contrainte de marge. La position θ est limitée à |θ| ≤ W/λ.
                       Exemple: λ=0.25 signifie levier max de 4x (1/0.25 = 4).
        
        dt: Pas de temps. dt=1/12 signifie pas mensuels.
        
        rho: Coefficient d'autocorrélation du signal AR(1). 
             ρ proche de 1 = signal très persistant.
             ρ = exp(-0.5*dt) par défaut.
        
        sigma: Volatilité des innovations du signal. 
               σ = 0.10*sqrt(dt) par défaut.
        
        T: Horizon d'investissement en années. T=5 signifie 5 ans.
    """
    alpha: float = 4.0        # Aversion au risque (CARA)
    lambda_margin: float = 0.25  # Contrainte de marge (levier max = 1/λ = 4)
    dt: float = 1/12          # Pas de temps (mensuel)
    rho: float = None         # Autocorrélation AR(1), calculé si None
    sigma: float = None       # Volatilité innovations, calculé si None
    T: float = 5.0            # Horizon (années)
    
    def __post_init__(self):
        """Calcule les paramètres dérivés."""
        if self.rho is None:
            self.rho = np.exp(-0.5 * self.dt)
        if self.sigma is None:
            self.sigma = 0.10 * np.sqrt(self.dt)


@dataclass
class DPGrids:
    """
    Grilles de discrétisation pour la DP.
    
    Attributs:
        W_grid: Grille de richesse [0, W_max]
        x_grid: Grille du signal prédictif [-x_max, x_max]
        theta_grid: Grille des contrôles (positions normalisées)
        
    Note: On normalise θ par W, donc theta_grid contient θ/W ∈ [-1/λ, 1/λ]
    """
    W_grid: np.ndarray
    x_grid: np.ndarray
    theta_grid: np.ndarray
    
    @property
    def NW(self) -> int:
        """Nombre de points sur la grille de richesse."""
        return len(self.W_grid)
    
    @property
    def Nx(self) -> int:
        """Nombre de points sur la grille du signal."""
        return len(self.x_grid)
    
    @property
    def Ntheta(self) -> int:
        """Nombre de points sur la grille des contrôles."""
        return len(self.theta_grid)


@dataclass
class DPResult:
    """
    Résultat de la résolution DP.
    
    Attributs:
        J: Fonction de valeur finale J(x, W) à t=0
        theta_opt: Politique optimale θ*(t, x, W) pour chaque période
        params: Paramètres utilisés
        grids: Grilles utilisées
        computation_time: Temps de calcul total
    """
    J: np.ndarray                    # Fonction de valeur (Nx, NW)
    theta_opt: np.ndarray            # Politique optimale (Nt, Nx, NW)
    params: DPParams
    grids: DPGrids
    computation_time: float = 0.0


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def create_grids(params: DPParams, 
                 NW: int = 81, 
                 Nx: int = None,
                 Ntheta: int = 81,
                 W_max: float = 4.0) -> DPGrids:
    """
    Crée les grilles de discrétisation pour la DP.
    
    Args:
        params: Paramètres du problème
        NW: Nombre de points pour la richesse W
        Nx: Nombre de points pour le signal x (auto si None)
        Ntheta: Nombre de points pour le contrôle θ
        W_max: Richesse maximale sur la grille
    
    Returns:
        DPGrids avec les trois grilles
    
    Note: La grille x est construite pour couvrir ±4 écarts-types
          de la distribution stationnaire du processus AR(1).
    """
    # Grille de richesse: W ∈ [0, W_max]
    W_grid = np.linspace(0, W_max, NW)
    
    # Grille du signal x
    # Volatilité stationnaire (steady-state) du processus AR(1):
    # Var(x_∞) = σ² / (1 - ρ²)
    sigma_ss = np.sqrt(params.sigma**2 / (1 - params.rho**2))
    x_max = 4 * sigma_ss  # Couvrir ±4 écarts-types
    
    if Nx is None:
        dx = params.sigma / 3  # Résolution fine
        Nx = 2 * int(np.ceil(x_max / dx)) + 1
    
    x_grid = np.linspace(-x_max, x_max, Nx)
    
    # Grille des contrôles: θ/W ∈ [-1/λ, 1/λ]
    # Contrainte de marge: |θ| ≤ W/λ ⟺ |θ/W| ≤ 1/λ
    theta_max = 1 / params.lambda_margin
    theta_grid = np.linspace(-theta_max, theta_max, Ntheta)
    
    return DPGrids(W_grid=W_grid, x_grid=x_grid, theta_grid=theta_grid)


def compute_transition_matrix(x_grid: np.ndarray, 
                               rho: float, 
                               sigma: float) -> np.ndarray:
    """
    Calcule la matrice de transition pour le processus AR(1).
    
    Le signal suit: x_{t+1} = ρ·x_t + σ·ε où ε ~ N(0,1)
    
    Pour chaque x_i, on calcule P(x_{t+1} = x_j | x_t = x_i)
    
    Args:
        x_grid: Grille des valeurs de x
        rho: Coefficient d'autocorrélation
        sigma: Volatilité des innovations
    
    Returns:
        TrM: Matrice de transition (Nx, Nx) où TrM[i,j] = P(x_j | x_i)
    
    Note: Les bords de la grille utilisent la CDF pour capturer
          les probabilités de dépasser les limites.
    """
    Nx = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    TrM = np.zeros((Nx, Nx))
    
    for i in range(Nx):
        x_current = x_grid[i]
        x_mean = rho * x_current  # Espérance conditionnelle E[x_{t+1}|x_t]
        
        # Probabilités pour les points intérieurs (approximation discrète)
        p = np.zeros(Nx)
        
        # Points intérieurs: densité × dx
        p[1:-1] = dx * (1 / np.sqrt(2 * np.pi * sigma**2)) * \
                  np.exp(-1 / (2 * sigma**2) * (x_grid[1:-1] - x_mean)**2)
        
        # Bord gauche: P(x ≤ x_grid[0] + dx/2)
        p[0] = norm.cdf((x_grid[0] + dx/2 - x_mean) / sigma)
        
        # Bord droit: P(x > x_grid[-1] - dx/2)
        p[-1] = 1 - norm.cdf((x_grid[-1] - dx/2 - x_mean) / sigma)
        
        # Normaliser pour que Σp = 1
        p = p / np.sum(p)
        TrM[i, :] = p
    
    return TrM


def utility_CARA(W: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fonction d'utilité CARA (Constant Absolute Risk Aversion).
    
    U(W) = -exp(-α·W)
    
    Propriétés:
    - U'(W) = α·exp(-α·W) > 0 (croissante)
    - U''(W) = -α²·exp(-α·W) < 0 (concave)
    - ARA(W) = -U''(W)/U'(W) = α (constante)
    
    Args:
        W: Richesse (scalaire ou array)
        alpha: Coefficient d'aversion au risque
    
    Returns:
        Utilité U(W)
    
    Exemple:
        >>> utility_CARA(1.0, 4.0)
        -0.0183  # -exp(-4)
    """
    return -np.exp(-alpha * W)


# =============================================================================
# ALGORITHME DE PROGRAMMATION DYNAMIQUE
# =============================================================================

def solve_portfolio_dp(params: DPParams = None,
                       grids: DPGrids = None,
                       verbose: bool = True) -> DPResult:
    """
    Résout le problème d'optimisation de portefeuille par DP (Dynamic Programming).
    
    PROBLÈME:
    max_θ E[U(W_T)] sujet à:
    - W_{t+1} = W_t + θ_t·(x_{t+1} - x_t)
    - x_{t+1} = ρ·x_t + σ·ε_{t+1}
    - |θ_t| ≤ W_t/λ (contrainte de marge)
    
    MÉTHODE (Backward Induction):
    1. Initialiser J(W, x, T) = U(W)
    2. Pour t = T-dt, ..., 0:
       J(W, x, t) = max_θ E[J(W', x', t+dt)]
    
    Args:
        params: Paramètres du problème (défaut: DPParams())
        grids: Grilles de discrétisation (créées si None)
        verbose: Afficher la progression
    
    Returns:
        DPResult avec fonction de valeur et politique optimale
    
    Exemple:
        >>> result = solve_portfolio_dp()
        >>> print(f"Temps de calcul: {result.computation_time:.1f}s")
    """
    start_time = time.time()
    
    # Paramètres par défaut
    if params is None:
        params = DPParams()
    
    # Créer les grilles si non fournies
    if grids is None:
        grids = create_grids(params)
    
    # Raccourcis
    W_grid = grids.W_grid
    x_grid = grids.x_grid
    theta_grid = grids.theta_grid
    NW, Nx, Ntheta = grids.NW, grids.Nx, grids.Ntheta
    alpha = params.alpha
    dt = params.dt
    T = params.T
    
    # Nombre de périodes
    Nt = int(np.ceil(T / dt))
    
    # Matrice de transition pour x
    TrM_x = compute_transition_matrix(x_grid, params.rho, params.sigma)
    
    # Initialisation: J(W, x, T) = U(W) = -exp(-αW)
    # J_next[ix, iW] = J(x_grid[ix], W_grid[iW])
    J_next = np.outer(np.ones(Nx), utility_CARA(W_grid, alpha))
    
    # Stockage de la politique optimale
    theta_opt = np.zeros((Nt, Nx, NW))
    
    # Backward induction
    t = T
    period = 0
    
    while t > 1e-12:
        t = t - dt
        
        if verbose:
            print(f"Période {period+1}/{Nt}, t = {t:.4f}", end="")
            iter_start = time.time()
        
        J = np.zeros_like(J_next)
        
        for ix in range(Nx):
            x = x_grid[ix]
            p = TrM_x[ix, :]  # Probabilités de transition
            
            for iW in range(NW):
                W = W_grid[iW]
                
                # Évaluer toutes les actions possibles
                V = np.zeros(Ntheta)
                
                for itheta in range(Ntheta):
                    # Position: θ = W × (θ/W)
                    theta = W * theta_grid[itheta]
                    
                    # Richesse future: W' = W + θ·(x' - x)
                    W_next = W + theta * (x_grid - x)
                    
                    # Contraindre W' aux bornes de la grille
                    W_next = np.clip(W_next, W_grid[0], W_grid[-1])
                    
                    # Calculer J(W', x') par interpolation
                    J1 = np.zeros(Nx)
                    
                    for jx in range(Nx):
                        if W_next[jx] <= W_grid[0]:
                            # Sous la grille: utiliser utilité directe
                            J1[jx] = utility_CARA(W_next[jx], alpha)
                        else:
                            # Interpolation linéaire sur la grille W
                            J1[jx] = np.interp(W_next[jx], W_grid, J_next[jx, :])
                    
                    # Espérance: V(θ) = Σ_x' p(x'|x) × J(W', x')
                    V[itheta] = np.dot(p, J1)
                
                # Maximiser sur θ
                best_idx = np.argmax(V)
                J[ix, iW] = V[best_idx]
                theta_opt[period, ix, iW] = theta_grid[best_idx]
        
        J_next = J.copy()
        period += 1
        
        if verbose:
            print(f" ({time.time() - iter_start:.2f}s)")
    
    computation_time = time.time() - start_time
    
    if verbose:
        print(f"\nTemps total: {computation_time:.1f}s")
    
    return DPResult(
        J=J_next,
        theta_opt=theta_opt,
        params=params,
        grids=grids,
        computation_time=computation_time
    )


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_optimal_policy(result: DPResult, 
                        W_index: int = None,
                        periods: list = None,
                        smooth_window: int = 9) -> plt.Figure:
    """
    Trace la politique optimale θ*(x) pour différentes périodes.
    
    Args:
        result: Résultat de solve_portfolio_dp
        W_index: Indice de W sur la grille (défaut: W ≈ 1)
        periods: Liste des périodes à tracer (défaut: [1, 12, 36, 60])
        smooth_window: Fenêtre de lissage
    
    Returns:
        Figure matplotlib
    """
    if W_index is None:
        # Trouver l'indice correspondant à W ≈ 1
        W_index = np.argmin(np.abs(result.grids.W_grid - 1.0))
    
    if periods is None:
        Nt = result.theta_opt.shape[0]
        periods = [0, min(11, Nt-1), min(35, Nt-1), min(59, Nt-1)]
    
    x_grid = result.grids.x_grid
    W_value = result.grids.W_grid[W_index]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['b-', 'r-.', 'g-', 'm--']
    labels = [f'T-t = {p+1} mois' for p in periods]
    
    for i, period in enumerate(periods):
        if period < result.theta_opt.shape[0]:
            policy = result.theta_opt[period, :, W_index]
            
            # Lissage optionnel
            if smooth_window > 1:
                from scipy.ndimage import uniform_filter1d
                policy = uniform_filter1d(policy, smooth_window)
            
            ax.plot(x_grid, policy, colors[i % len(colors)], 
                   linewidth=2, label=labels[i])
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Signal Prédictif (x)', fontsize=14)
    ax.set_ylabel('Politique Optimale (θ*/W)', fontsize=14)
    ax.set_title(f'Politique Optimale pour W = {W_value:.2f}', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-5, 5])
    
    plt.tight_layout()
    return fig


def plot_value_function(result: DPResult) -> plt.Figure:
    """
    Trace la fonction de valeur J(W, x) en 3D.
    
    Args:
        result: Résultat de solve_portfolio_dp
    
    Returns:
        Figure matplotlib 3D
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sous-échantillonner pour une meilleure visualisation
    W_grid = result.grids.W_grid[:40]
    x_grid = result.grids.x_grid[::3]
    J = result.J[::3, :40]
    
    W_mesh, x_mesh = np.meshgrid(W_grid, x_grid)
    
    surf = ax.plot_surface(W_mesh, x_mesh, J, cmap='viridis', 
                           linewidth=0.5, antialiased=True)
    
    ax.set_xlabel('Richesse (W)', fontsize=12)
    ax.set_ylabel('Signal (x)', fontsize=12)
    ax.set_zlabel('Fonction de Valeur J(W,x)', fontsize=12)
    ax.set_title('Fonction de Valeur à t=0', fontsize=14)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return fig


# =============================================================================
# DÉMONSTRATION
# =============================================================================

def demo_portfolio_dp():
    """
    Démonstration complète de la DP pour l'optimisation de portefeuille.
    """
    print("=" * 70)
    print("MIT 15.450 - DP POUR L'OPTIMISATION DE PORTEFEUILLE")
    print("=" * 70)
    
    # Paramètres
    params = DPParams(
        alpha=4.0,         # Aversion au risque (CARA)
        lambda_margin=0.25, # Contrainte de marge (levier max 4x)
        dt=1/12,           # Pas mensuel
        T=5.0              # Horizon 5 ans
    )
    
    print(f"\nParamètres:")
    print(f"  α (alpha) = {params.alpha} (aversion au risque CARA)")
    print(f"  λ (lambda) = {params.lambda_margin} (levier max = {1/params.lambda_margin:.0f}x)")
    print(f"  dt = {params.dt:.4f} (pas mensuel)")
    print(f"  ρ (rho) = {params.rho:.4f} (autocorrélation AR(1))")
    print(f"  σ (sigma) = {params.sigma:.4f} (volatilité innovations)")
    print(f"  T = {params.T} ans ({int(params.T/params.dt)} périodes)")
    
    # Résolution (avec grilles réduites pour la démo)
    print("\nRésolution par backward induction...")
    grids = create_grids(params, NW=41, Ntheta=41)
    print(f"  Grille W: {grids.NW} points")
    print(f"  Grille x: {grids.Nx} points")
    print(f"  Grille θ: {grids.Ntheta} points")
    
    result = solve_portfolio_dp(params, grids, verbose=True)
    
    # Analyse des résultats
    print("\n" + "=" * 70)
    print("ANALYSE DES RÉSULTATS")
    print("=" * 70)
    
    # Politique optimale pour x=0 et W=1
    ix_0 = grids.Nx // 2  # x ≈ 0
    iW_1 = np.argmin(np.abs(grids.W_grid - 1.0))  # W ≈ 1
    
    print(f"\nPolitique optimale θ*/W pour x=0, W=1:")
    for t_months in [1, 12, 36, 60]:
        period = t_months - 1
        if period < result.theta_opt.shape[0]:
            theta = result.theta_opt[period, ix_0, iW_1]
            print(f"  T-t = {t_months:2d} mois: θ*/W = {theta:+.4f}")
    
    print(f"\nFonction de valeur J(W=1, x=0) = {result.J[ix_0, iW_1]:.6f}")
    
    return result


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    result = demo_portfolio_dp()
    
    # Optionnel: créer les graphiques
    try:
        fig1 = plot_optimal_policy(result)
        fig1.savefig('optimal_policy.png', dpi=150)
        print("\nGraphique sauvegardé: optimal_policy.png")
        
        fig2 = plot_value_function(result)
        fig2.savefig('value_function.png', dpi=150)
        print("Graphique sauvegardé: value_function.png")
    except Exception as e:
        print(f"\nGraphiques non générés: {e}")
```

---

## 📊 RÉSULTATS ATTENDUS

### Politique Optimale θ*(x)

La politique optimale dépend du signal x:
- **x > 0** (signal positif): Position longue (θ > 0)
- **x < 0** (signal négatif): Position courte (θ < 0)
- **x = 0**: Position nulle ou proche de zéro

| Signal x | Interprétation | Action Optimale |
|----------|----------------|-----------------|
| x >> 0 | Fort signal haussier | Long maximum (θ = W/λ) |
| x > 0 | Signal haussier modéré | Long modéré |
| x ≈ 0 | Pas de signal | Position neutre |
| x < 0 | Signal baissier modéré | Short modéré |
| x << 0 | Fort signal baissier | Short maximum (θ = -W/λ) |

### Effet du Temps Restant

| Temps Restant | Comportement |
|---------------|--------------|
| **T-t grand** | Politique plus agressive (plus de temps pour récupérer) |
| **T-t petit** | Politique plus conservatrice (moins de marge d'erreur) |

---

## 🎯 GUIDE D'UTILISATION POUR HELIXONE

### Intégration Recommandée

```
helixone/
├── optimization/
│   ├── __init__.py
│   ├── dynamic_programming.py   # Ce module
│   ├── portfolio_dp.py          # Spécialisation portefeuille
│   └── grids.py                 # Utilitaires de discrétisation
└── strategies/
    └── predictive_allocation.py # Stratégie basée sur signaux
```

### Cas d'Utilisation

| Cas | Application |
|-----|-------------|
| **Mean Reversion** | Trading sur signaux de retour à la moyenne |
| **Momentum** | Allocation basée sur signaux de tendance |
| **Risk Management** | Contraintes de levier dynamiques |
| **Backtesting** | Benchmark pour stratégies ML |

### Exemple d'Utilisation

```python
from helixone.optimization.portfolio_dp import solve_portfolio_dp, DPParams

# Paramètres personnalisés
params = DPParams(
    alpha=3.0,           # Moins averse au risque
    lambda_margin=0.5,   # Levier max 2x
    dt=1/252,            # Pas journalier
    T=1.0                # Horizon 1 an
)

# Résoudre
result = solve_portfolio_dp(params, verbose=True)

# Extraire la politique pour l'état actuel
current_x = 0.02  # Signal actuel
current_W = 1.0   # Richesse actuelle

# Trouver θ* par interpolation
ix = np.searchsorted(result.grids.x_grid, current_x)
iW = np.searchsorted(result.grids.W_grid, current_W)
theta_star = result.theta_opt[0, ix, iW]

print(f"Position optimale: θ* = {theta_star * current_W:.4f}")
```

---

## 📚 LIENS AVEC LE CODE STANFORD RL

Ce code DP s'intègre avec les modules Stanford RL:

| Module MIT | Équivalent Stanford | Connexion |
|------------|---------------------|-----------|
| `DPParams` | `markov_decision_process.py` | Structure MDP |
| `solve_portfolio_dp` | `dynamic_programming.py` | Algorithme DP |
| `compute_transition_matrix` | `markov_process.py` | Transitions |
| `DPGrids` | `function_approx.py` | Discrétisation |

---

## ✅ RÉSUMÉ

| Aspect | Détail |
|--------|--------|
| **Problème** | Allocation optimale avec signal prédictif |
| **Méthode** | DP (Dynamic Programming) par backward induction |
| **Contraintes** | Marge (levier limité), utilité CARA |
| **Complexité** | O(Nt × Nx × NW × Ntheta) |
| **Output** | Politique optimale θ*(t, x, W) |

---

**FIN DU GUIDE MIT DP PORTFOLIO OPTIMIZATION POUR HELIXONE**
