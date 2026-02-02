# 📗 MODULE 6: REINFORCEMENT LEARNING POUR LA FINANCE
## Applications du RL au Trading et à l'Investissement

---

## 📚 SOURCE PRINCIPALE
**Stanford CME 241 - Foundations of RL with Applications in Finance**
- PDF COMPLET (400+ pages): https://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf
- Site du cours: https://cme241.github.io/
- Auteur: Ashwin Rao (Stanford)

---

## 🎯 OBJECTIFS D'APPRENTISSAGE
1. Maîtriser le framework MDP (Markov Decision Process)
2. Comprendre Dynamic Programming (Value/Policy Iteration)
3. Implémenter les algorithmes TD (Temporal Difference)
4. Appliquer Policy Gradient et Actor-Critic
5. Résoudre les problèmes financiers avec RL

---

## 📂 APPLICATIONS FINANCIÈRES DU RL

### 1. Allocation Dynamique d'Actifs
- État: richesse actuelle, prix des actifs, features de marché
- Action: allocation aux différents actifs
- Récompense: utilité de la consommation

### 2. Exécution Optimale
- État: inventaire restant, temps restant, état du marché
- Action: quantité à trader
- Récompense: -coût d'exécution (impact + risque)

### 3. Pricing d'Options Américaines
- État: prix du sous-jacent, temps jusqu'à maturité
- Action: exercer ou attendre
- Récompense: payoff si exercice, continuation value sinon

### 4. Market Making
- État: inventaire, spread actuel, flux d'ordres
- Action: bid/ask quotes
- Récompense: P&L - pénalité inventaire

---

## 🔑 CONCEPTS FONDAMENTAUX

### 1. MDP (Markov Decision Process)

**Définition formelle**: (S, A, P, R, γ)
- S: ensemble des états
- A: ensemble des actions
- P(s'|s,a): probabilité de transition
- R(s,a,s'): récompense
- γ ∈ [0,1]: facteur de discount

**Équation de Bellman (Value Function)**:
```
V^π(s) = E_π[Σₜ γᵗ R_{t+1} | S_0 = s]
V^π(s) = Σₐ π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

**Équation de Bellman (Action-Value)**:
```
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]
```

**Optimalité**:
```
V*(s) = max_a Q*(s,a)
Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
```

### 2. Dynamic Programming

**Value Iteration**:
```python
def value_iteration(mdp, epsilon=1e-6):
    V = np.zeros(mdp.n_states)
    while True:
        V_new = np.zeros(mdp.n_states)
        for s in range(mdp.n_states):
            V_new[s] = max(
                sum(mdp.P[s,a,s_] * (mdp.R[s,a,s_] + mdp.gamma * V[s_])
                    for s_ in range(mdp.n_states))
                for a in range(mdp.n_actions)
            )
        if np.max(np.abs(V_new - V)) < epsilon:
            break
        V = V_new
    return V
```

**Policy Iteration**:
```python
def policy_iteration(mdp):
    policy = np.zeros(mdp.n_states, dtype=int)
    while True:
        # Policy Evaluation
        V = evaluate_policy(mdp, policy)
        # Policy Improvement
        policy_new = np.argmax([
            [sum(mdp.P[s,a,s_] * (mdp.R[s,a,s_] + mdp.gamma * V[s_])
                 for s_ in range(mdp.n_states))
             for a in range(mdp.n_actions)]
            for s in range(mdp.n_states)
        ], axis=1)
        if np.array_equal(policy, policy_new):
            break
        policy = policy_new
    return policy, V
```

### 3. Temporal Difference Learning

**TD(0) Update**:
```
V(s) ← V(s) + α[R + γV(s') - V(s)]
```

**Q-Learning (Off-policy)**:
```
Q(s,a) ← Q(s,a) + α[R + γ max_{a'} Q(s',a') - Q(s,a)]
```

**SARSA (On-policy)**:
```
Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
```

### 4. Policy Gradient

**Théorème du Gradient de la Policy**:
```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) Q^π(s,a)]
```

**REINFORCE**:
```python
def reinforce_update(trajectory, policy_net, optimizer):
    returns = compute_returns(trajectory, gamma)
    loss = 0
    for (s, a, _), G in zip(trajectory, returns):
        log_prob = torch.log(policy_net(s)[a])
        loss -= log_prob * G  # Negative for gradient ascent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Actor-Critic**:
```python
# Actor: policy network π_θ(a|s)
# Critic: value network V_φ(s)

def actor_critic_update(s, a, r, s_next, done):
    # TD error (advantage estimate)
    td_error = r + gamma * V(s_next) * (1 - done) - V(s)
    
    # Critic update
    critic_loss = td_error ** 2
    
    # Actor update
    actor_loss = -log_prob(a|s) * td_error.detach()
    
    return actor_loss + critic_loss
```

---

## 💻 IMPLÉMENTATION: EXÉCUTION OPTIMALE AVEC RL

```python
import numpy as np
import torch
import torch.nn as nn

class OptimalExecutionEnv:
    """
    Environment for optimal execution problem
    Based on Almgren-Chriss framework
    """
    def __init__(self, T=1.0, N=10, X0=1e6, sigma=0.02, 
                 eta=2.5e-6, gamma_risk=1e-6):
        self.T = T
        self.N = N
        self.dt = T / N
        self.X0 = X0
        self.sigma = sigma
        self.eta = eta  # Temporary impact
        self.gamma_risk = gamma_risk  # Risk aversion
        
    def reset(self):
        self.t = 0
        self.inventory = self.X0
        self.cash = 0
        self.price = 100.0
        return self._get_state()
    
    def _get_state(self):
        return np.array([
            self.inventory / self.X0,  # Normalized inventory
            self.t / self.N,           # Time fraction
            self.price / 100.0         # Normalized price
        ])
    
    def step(self, action):
        """
        action: fraction of remaining inventory to sell [0, 1]
        """
        x = action * self.inventory  # Shares to sell
        
        # Price impact (temporary)
        impact = self.eta * x / self.dt
        execution_price = self.price - impact
        
        # Execute trade
        self.cash += x * execution_price
        self.inventory -= x
        
        # Price evolves (random walk)
        self.price += self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.t += 1
        
        # Reward: cash received - risk penalty
        reward = x * execution_price - self.gamma_risk * (self.inventory ** 2) * self.dt
        
        done = (self.t >= self.N) or (self.inventory <= 0)
        
        if done and self.inventory > 0:
            # Liquidate remaining at market
            reward += self.inventory * (self.price - self.eta * self.inventory / self.dt)
            self.inventory = 0
        
        return self._get_state(), reward, done, {}


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Mean and log_std
        )
        
    def forward(self, state):
        output = self.network(state)
        mean = torch.sigmoid(output[..., 0])  # Action in [0, 1]
        log_std = output[..., 1].clamp(-2, 0)
        return mean, log_std
    
    def sample_action(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.clamp(0, 1), log_prob


def train_optimal_execution(n_episodes=1000):
    env = OptimalExecutionEnv()
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for episode in range(n_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            action, log_prob = policy.sample_action(state_tensor)
            
            next_state, reward, done, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}")
    
    return policy


# Train and test
if __name__ == "__main__":
    policy = train_optimal_execution(n_episodes=2000)
```

---

## 📊 COMPARAISON DES ALGORITHMES

| Algorithme | Type | Avantages | Inconvénients |
|------------|------|-----------|---------------|
| Value Iteration | DP | Convergence garantie | Nécessite modèle complet |
| Q-Learning | TD, Off-policy | Exploration libre | Peut diverger avec FA |
| SARSA | TD, On-policy | Plus stable | Moins efficace en données |
| DQN | Deep RL | Espaces d'états continus | Hyperparameters sensibles |
| PPO | Policy Gradient | Actions continues | Variance élevée |
| SAC | Actor-Critic | Sample efficient | Complexité d'implémentation |

---

## 🔗 RÉFÉRENCES
1. Rao, A. - Foundations of RL with Applications in Finance (Stanford CME 241)
2. Sutton & Barto - Reinforcement Learning: An Introduction
3. Buehler et al. (2019) - Deep Hedging
4. Kolm & Ritter (2019) - Modern Perspectives on RL in Finance
