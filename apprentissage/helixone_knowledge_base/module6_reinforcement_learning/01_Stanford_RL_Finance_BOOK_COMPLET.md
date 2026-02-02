# Foundations of Reinforcement Learning with Applications in Finance

> **Source**: https://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf
> **Auteurs**: Ashwin Rao, Tikhon Jelvis
> **Extrait pour la base de connaissances HelixOne**

---

## Table des Matières

### Preface (p.11)
### Summary of Notation (p.15)

## 1. Overview (p.17)

### 1.1. Learning Reinforcement Learning
Reinforcement Learning (RL) is emerging as a practical, powerful technique for solving a variety of complex business problems across industries that involve **Sequential Optimal Decisioning under Uncertainty**. Although RL is classified as a branch of Machine Learning (ML), it tends to be viewed and treated quite differently from other branches of ML (Supervised and Unsupervised Learning). Indeed, RL seems to hold the key to unlocking the promise of AI—machines that adapt their decisions to vagaries in observed information, while continuously steering towards the optimal outcome.

### 1.2. What You Will Learn from This Book
- Theory of Markov Decision Processes (MDPs)—a framework for Sequential Optimal Decisioning under Uncertainty
- Power of Bellman Equations
- Dynamic Programming (DP) Algorithms: Policy Iteration, Value Iteration, Backward Induction, Approximate Dynamic Programming
- Generalized Policy Iteration
- RL Algorithms: SARSA, Q-Learning, Gradient TD, DQN, LSPI, Policy Gradient, MCTS
- Multi-Armed Bandits: UCB, Thompson Sampling, Gradient Bandits
- Financial Applications:
  - Dynamic Asset-Allocation to maximize Utility of Consumption
  - Pricing and Hedging of Derivatives in an Incomplete Market
  - Optimal Exercise/Stopping of Path-Dependent American Options
  - Optimal Trade Order Execution (managing Price Impact)
  - Optimal Market-Making (Bid/Ask managing Inventory Risk)

### 1.3. Expected Background
- Python experience (numpy)
- Undergraduate-level Probability (most important foundation)
- Numerical Optimization, Statistics, Linear Algebra
- No Finance background required

### 1.4. Decluttering the Jargon
**Key Terms:**
- **Uncertainty**: Problems involving random variables evolving over time (stochastic processes)
- **Optimal Decisions**: Optimization - maximizing a well-defined quantity (the "goal")
- **Sequential**: Dynamic decisions adjusted to "changing circumstances"
- **Control**: Persistent steering towards the goal
- **Stochastic Control**: The combined framework

### 1.5. Introduction to the MDP Framework

The MDP Framework consists of:
- **Agent**: An AI algorithm
- **Environment**: Abstract entity serving up uncertain outcomes
- **State** (St ∈ S): Abstract piece of information at time t
- **Action** (At ∈ A): Activity performed by the Agent
- **Reward** (Rt ∈ D): Numerical feedback

**Transition probabilities:**
```
p(r, s'|s, a) = P[(Rt+1 = r, St+1 = s') | St = s, At = a]
```

**Return (accumulated rewards):**
```
Gt = Rt+1 + γ·Rt+2 + γ²·Rt+3 + ...
```

Where γ ∈ [0, 1] is the **discount factor**.

**Goal**: Find a Policy π : S → A that maximizes E[Gt|St = s] for all s ∈ S.

**Markov Property:**
- Next State/Reward depends only on Current State (for a given Action)
- Current State encapsulates all relevant information from history
- Current State is a sufficient statistic of the future

### 1.6. Real-World Problems That Fit the MDP Framework
- Self-driving vehicles
- Game of Chess
- Complex Logistical Operations (Warehouse)
- Humanoid robot walking
- Investment portfolio management
- Football game decisions
- Election strategy

### 1.7. The Inherent Difficulty in Solving MDPs
- Large or complex State Space
- Large or complex Action Space
- No direct feedback on "correct" Action
- Time-sequenced complexity (actions influence future states)
- Delayed consequences
- Unknown model of environment
- Exploration vs Exploitation balance

### 1.8. Value Function, Bellman Equations, DP and RL

**Value Function for policy π:**
```
V^π(s) = E_π,p[Gt|St = s] for all s ∈ S
```

**Bellman Equation (recursive):**
```
V^π(s) = Σ_{r,s'} p(r, s'|s, π(s)) · (r + γ · V^π(s'))
```

**Optimal Value Function:**
```
V*(s) = max_π V^π(s) for all s ∈ S
```

**Bellman Optimality Equation:**
```
V*(s) = max_a Σ_{r,s'} p(r, s'|s, a) · (r + γ · V*(s'))
```

**Key Problems:**
- **Prediction**: Calculate V^π(s) for a given policy
- **Control**: Calculate V* and π*

**Algorithm Types:**
- **Dynamic Programming**: Planning algorithms (requires knowing p)
- **Reinforcement Learning**: Learning algorithms (learns from interaction)

---

## MODULE I: PROCESSES AND PLANNING ALGORITHMS

## 3. Markov Processes (p.59)

### 3.1. The Concept of State in a Process
A state captures all relevant information needed to predict future behavior.

### 3.2-3.4. Markov Processes Formalism

**Markov Process Definition:**
- State Space S (countable set)
- Transition probability function: P(s'|s) for all s, s' ∈ S
- Starting state distribution
- Terminal states (optional)

### 3.8. Markov Reward Processes

**MRP adds:**
- Reward function R(s) or R(s, s')
- Discount factor γ

**Value Function for MRP:**
```
V(s) = E[Gt|St = s] = E[Rt+1 + γ·Rt+2 + γ²·Rt+3 + ... | St = s]
```

**Bellman Equation for MRP:**
```
V(s) = R(s) + γ · Σ_{s'} P(s'|s) · V(s')
```

---

## 4. Markov Decision Processes (p.93)

### 4.3. Formal Definition of MDP
- State Space S
- Action Space A
- Transition probabilities: P(s'|s, a)
- Reward function: R(s, a, s') or R(s, a)
- Discount factor γ

### 4.4. Policy
**Deterministic Policy**: π : S → A
**Stochastic Policy**: π(a|s) = P(At = a | St = s)

### 4.9. MDP Value Function for Fixed Policy

**State-Value Function:**
```
V^π(s) = E_π[Gt | St = s]
```

**Action-Value Function (Q-function):**
```
Q^π(s, a) = E_π[Gt | St = s, At = a]
```

**Relationship:**
```
V^π(s) = Σ_a π(a|s) · Q^π(s, a)
Q^π(s, a) = R(s, a) + γ · Σ_{s'} P(s'|s, a) · V^π(s')
```

### 4.10. Optimal Value Function and Optimal Policies

**Optimal State-Value:**
```
V*(s) = max_π V^π(s)
```

**Optimal Action-Value:**
```
Q*(s, a) = max_π Q^π(s, a)
```

**Optimal Policy from Q*:**
```
π*(s) = argmax_a Q*(s, a)
```

---

## 5. Dynamic Programming Algorithms (p.125)

### 5.1. Planning versus Learning
- **Planning**: Model known, compute optimal policy
- **Learning**: Model unknown, learn from experience

### 5.3. Fixed-Point Theory
Bellman operators are contractions → unique fixed point exists.

### 5.4. Policy Evaluation Algorithm
Iteratively apply Bellman equation until convergence:
```
V_{k+1}(s) = Σ_a π(a|s) · [R(s,a) + γ · Σ_{s'} P(s'|s,a) · V_k(s')]
```

### 5.5-5.7. Policy Improvement and Policy Iteration

**Greedy Policy:**
```
π'(s) = argmax_a Q^π(s, a)
```

**Policy Iteration:**
1. Initialize π
2. Policy Evaluation: Compute V^π
3. Policy Improvement: π' = greedy(V^π)
4. If π' ≠ π, set π = π' and go to step 2
5. Return π*

### 5.8. Value Iteration

**Bellman Optimality Operator:**
```
V_{k+1}(s) = max_a [R(s,a) + γ · Σ_{s'} P(s'|s,a) · V_k(s')]
```

Iterate until convergence.

### 5.13. Backward Induction (Finite Horizon)
For finite-horizon MDPs, solve backwards from terminal time T:
```
V_T(s) = R_T(s)
V_t(s) = max_a [R_t(s,a) + Σ_{s'} P(s'|s,a) · V_{t+1}(s')]
```

---

## 6. Function Approximation and ADP (p.163)

### 6.1. Why Function Approximation?
- State space too large for tabular methods
- Generalization to unseen states
- Memory efficiency

### 6.2. Linear Function Approximation
```
V(s; w) = w^T · φ(s) = Σ_i w_i · φ_i(s)
```
Where φ(s) is a feature vector.

### 6.3. Neural Network Function Approximation
Deep neural networks as universal function approximators.

**Training:**
- Forward propagation
- Loss computation
- Backpropagation
- Gradient descent update

### 6.5-6.6. Approximate DP Algorithms
- Approximate Policy Evaluation
- Approximate Value Iteration
- Fitted Value Iteration

---

## MODULE II: MODELING FINANCIAL APPLICATIONS

## 7. Utility Theory (p.199)

### 7.1. Introduction to Utility
People are typically **risk-averse**: they prefer certain outcomes over uncertain ones with same expected value.

### 7.3. Shape of Utility Function
- **Concave**: Risk-averse (most common)
- **Linear**: Risk-neutral
- **Convex**: Risk-seeking

### 7.4. Risk Premium
The amount of expected return an investor requires to accept uncertainty.

### 7.5. CARA - Constant Absolute Risk Aversion
```
U(x) = -e^{-αx} / α
```
Where α > 0 is the risk-aversion coefficient.

**Properties:**
- Absolute Risk Aversion: A(x) = -U''(x)/U'(x) = α (constant)
- Independent of wealth level

### 7.7. CRRA - Constant Relative Risk Aversion
```
U(x) = x^{1-γ} / (1-γ)  for γ ≠ 1
U(x) = log(x)           for γ = 1
```

**Properties:**
- Relative Risk Aversion: R(x) = -x·U''(x)/U'(x) = γ (constant)
- Scales with wealth

---

## 8. Dynamic Asset-Allocation and Consumption (p.211)

### 8.2. Merton's Portfolio Problem

**Setting:**
- Continuous time [0, T]
- Risk-free asset with return r
- Risky asset following GBM: dS/S = μdt + σdz
- Wealth W_t
- Consumption rate c_t
- Portfolio allocation π_t (fraction in risky asset)

**Objective:**
```
max E[∫_0^T e^{-ρt} U(c_t) dt + e^{-ρT} B(W_T)]
```

**Wealth dynamics:**
```
dW = [W(r + π(μ-r)) - c]dt + Wπσdz
```

### 8.3. Merton's Solution (CRRA Utility)

**Optimal allocation:**
```
π* = (μ - r) / (γσ²)
```
Independent of wealth and time!

**Optimal consumption:**
```
c* = ν · W
```
Where ν depends on parameters.

**Key Insight:** Separation theorem - allocation decision independent of consumption decision.

---

## 9. Derivatives Pricing and Hedging (p.235)

### 9.1. Brief Introduction to Derivatives
- **Forwards**: Agreement to buy/sell at future date
- **European Options**: Right to buy (call) or sell (put) at expiry
- **American Options**: Can exercise any time before expiry

### 9.3-9.5. Fundamental Theorems of Asset Pricing

**1st FTAP:** No-arbitrage ⟺ ∃ risk-neutral probability measure Q

**2nd FTAP:** Market complete ⟺ Q is unique

### 9.6. Derivatives Pricing

**Complete Market:**
```
Price = E^Q[e^{-rT} · Payoff]
```

**Incomplete Market:**
Price bounds from super/sub-replication or utility-based pricing.

### 9.8. American Options as MDP

**State:** (t, S_t) or path history
**Action:** Exercise or Continue
**Reward:** Payoff if exercise, 0 otherwise

**Bellman Equation:**
```
V(t, s) = max{g(s), e^{-rΔt} · E[V(t+Δt, S_{t+Δt}) | S_t = s]}
```

Where g(s) is the payoff function.

### 9.10. Pricing/Hedging in Incomplete Market as MDP

**State:** (t, S_t, inventory)
**Action:** Hedge amount
**Objective:** Minimize hedging error + risk penalty

---

## 10. Order-Book Trading Algorithms (p.271)

### 10.1. Basics of Order Book

**Order Types:**
- **Market Order**: Execute immediately at best available price
- **Limit Order**: Execute only at specified price or better

**Order Book:** List of all outstanding limit orders
- **Bid side**: Buy orders
- **Ask side**: Sell orders
- **Spread**: Ask price - Bid price

### 10.2. Optimal Execution

**Problem:** Sell X shares over time [0, T] to maximize proceeds (minimize market impact).

**Market Impact:**
- **Temporary**: Price moves during execution, then reverts
- **Permanent**: Price moves permanently

### 10.2.1. Almgren-Chriss Model

**Assumptions:**
- Linear temporary impact: h(v) = η·v
- Linear permanent impact: g(v) = γ·v
- Arithmetic random walk for price

**State:** (t, remaining_shares, current_price)
**Action:** Number of shares to sell at time t

**Optimal Solution (mean-variance):**
```
n*_t = (X/T) · sinh(κ(T-t)) / sinh(κT)
```

Where κ depends on risk aversion and impact parameters.

**Key Insight:** Risk-averse trader front-loads execution.

### 10.3. Optimal Market-Making

**Problem:** Market maker quotes bid/ask prices to maximize profit while managing inventory risk.

### 10.3.1. Avellaneda-Stoikov Model

**State:** (t, S_t, inventory_q)
**Action:** Bid spread δ^b, Ask spread δ^a

**Dynamics:**
- Mid-price follows Brownian motion
- Order arrivals are Poisson with intensity λ(δ)

**Optimal Quotes:**
```
δ^a = δ^b = (1/γ)·log(1 + γ/k) + (γσ²(T-t))/2 · (2q + 1)
```

Where:
- γ: risk aversion
- σ: volatility
- k: arrival rate parameter
- q: current inventory

**Key Insight:** Skew quotes based on inventory to mean-revert position.

---

## MODULE III: REINFORCEMENT LEARNING ALGORITHMS

## 11. Monte-Carlo and TD for Prediction (p.307)

### 11.3. Monte-Carlo (MC) Prediction

**Idea:** Estimate V^π(s) by averaging returns from visits to state s.

**First-Visit MC:**
```python
for each episode:
    generate episode following π
    for first visit to each state s:
        G = return from that point
        update: V(s) ← V(s) + α(G - V(s))
```

**Every-Visit MC:** Same but count all visits.

### 11.4. Temporal-Difference (TD) Prediction

**TD(0) Update:**
```
V(s) ← V(s) + α[r + γV(s') - V(s)]
```

**TD Target:** r + γV(s')
**TD Error:** δ = r + γV(s') - V(s)

### 11.5. TD versus MC

| Aspect | MC | TD |
|--------|----|----|
| Bias | Unbiased | Biased (bootstrap) |
| Variance | High | Low |
| Convergence | To V^π | To V^π (with conditions) |
| Data efficiency | Lower | Higher |
| Requires terminal | Yes | No |

### 11.6. TD(λ) - Eligibility Traces

**n-step Return:**
```
G_t^{(n)} = r_{t+1} + γr_{t+2} + ... + γ^{n-1}r_{t+n} + γ^n V(s_{t+n})
```

**λ-Return (weighted average of n-step returns):**
```
G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^{(n)}
```

**Eligibility Trace:**
```
e_t(s) = γλ·e_{t-1}(s) + 𝟙(S_t = s)
```

**TD(λ) Update:**
```
V(s) ← V(s) + α·δ_t·e_t(s)  for all s
```

---

## 12. Monte-Carlo and TD for Control (p.345)

### 12.2-12.3. MC Control

**GLIE (Greedy in the Limit with Infinite Exploration):**
1. All state-action pairs visited infinitely often
2. Policy converges to greedy

**MC Control with ε-greedy:**
```python
for each episode:
    generate episode using ε-greedy policy
    for each (s, a) in episode:
        G = return from that point
        Q(s, a) ← Q(s, a) + α(G - Q(s, a))
    improve policy: π(s) = ε-greedy(Q)
```

### 12.4. SARSA (On-Policy TD Control)

**Update:**
```
Q(s, a) ← Q(s, a) + α[r + γQ(s', a') - Q(s, a)]
```

Where a' is chosen by current policy from s'.

**Algorithm:**
```python
initialize Q(s, a)
for each episode:
    s = initial state
    a = ε-greedy(Q, s)
    while not terminal:
        take action a, observe r, s'
        a' = ε-greedy(Q, s')
        Q(s, a) ← Q(s, a) + α[r + γQ(s', a') - Q(s, a)]
        s, a = s', a'
```

### 12.6. Q-Learning (Off-Policy TD Control)

**Update:**
```
Q(s, a) ← Q(s, a) + α[r + γ·max_{a'} Q(s', a') - Q(s, a)]
```

**Key difference from SARSA:** Uses max over actions (greedy w.r.t. Q), not actual next action.

**Properties:**
- Off-policy: learns optimal Q* regardless of behavior policy
- More sample efficient
- Can be unstable with function approximation

---

## 13. Batch RL, Experience-Replay, DQN, LSPI (p.381)

### 13.1. Experience Replay

**Idea:** Store experiences in buffer, sample randomly for updates.

**Benefits:**
- Breaks correlation in sequential data
- Reuses data multiple times
- More stable learning

### 13.4. Deep Q-Networks (DQN)

**Key innovations:**
1. **Experience Replay Buffer**
2. **Target Network:** Separate network for TD target, updated periodically

**Loss function:**
```
L(θ) = E[(r + γ·max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))²]
```

Where θ^- is the target network parameters.

**Algorithm:**
```python
initialize replay buffer D
initialize Q-network with random weights θ
initialize target network θ^- = θ

for each episode:
    for each step:
        select action (ε-greedy)
        execute action, observe r, s'
        store (s, a, r, s') in D
        
        sample minibatch from D
        compute targets: y = r + γ·max_{a'} Q(s', a'; θ^-)
        gradient descent on (y - Q(s, a; θ))²
        
        periodically update θ^- = θ
```

### 13.5. Least-Squares Policy Iteration (LSPI)

**For linear function approximation:**
```
Q(s, a; w) = w^T · φ(s, a)
```

**LSTD for Q-function:**
Solve: Aw = b
Where:
```
A = Σ φ(s,a) · [φ(s,a) - γφ(s',π(s'))]^T
b = Σ φ(s,a) · r
```

**LSPI Algorithm:**
```python
collect data {(s_i, a_i, r_i, s'_i)}
initialize policy π
repeat:
    w = LSTDQ(data, π)  # policy evaluation
    π_new = greedy(w)    # policy improvement
until π converges
```

### 13.6. RL for American Options

**LSPI approach:**
- Features: polynomials of stock price
- Actions: exercise or continue
- State: (time, stock price, path features)

---

## 14. Policy Gradient Algorithms (p.415)

### 14.1. Motivation

**When to use Policy Gradient:**
- Large/continuous action spaces
- Stochastic policies needed
- Policy easier to represent than value function

### 14.2. Policy Gradient Theorem

**Objective:**
```
J(θ) = E_{τ~π_θ}[R(τ)] = E_{s_0}[V^{π_θ}(s_0)]
```

**Theorem:**
```
∇_θ J(θ) = E_{π_θ}[∇_θ log π_θ(a|s) · Q^{π_θ}(s, a)]
```

**Score function:** ∇_θ log π_θ(a|s)

### 14.4. REINFORCE (Monte-Carlo Policy Gradient)

**Update:**
```
θ ← θ + α · ∇_θ log π_θ(a_t|s_t) · G_t
```

**Algorithm:**
```python
for each episode:
    generate trajectory τ = (s_0, a_0, r_1, ..., s_T)
    for t = 0 to T-1:
        G_t = Σ_{k=t}^T γ^{k-t} r_{k+1}
        θ ← θ + α · ∇_θ log π_θ(a_t|s_t) · G_t
```

### 14.6. Actor-Critic

**Idea:** Use value function to reduce variance.

**Advantage function:**
```
A^π(s, a) = Q^π(s, a) - V^π(s)
```

**Actor-Critic update:**
```
Critic: w ← w + α_w · δ · ∇_w V(s; w)
Actor:  θ ← θ + α_θ · δ · ∇_θ log π_θ(a|s)
```

Where δ = r + γV(s'; w) - V(s; w) is the TD error.

### 14.8. Advanced Policy Gradient Methods

**Natural Policy Gradient:**
Uses Fisher information matrix for more stable updates.

**TRPO (Trust Region Policy Optimization):**
Constrains policy change per update.

**PPO (Proximal Policy Optimization):**
Clips objective to prevent large updates.

---

## MODULE IV: FINISHING TOUCHES

## 15. Multi-Armed Bandits (p.447)

### 15.1. Problem Definition

**Setting:**
- K arms (actions)
- Each arm has unknown reward distribution
- Goal: maximize cumulative reward

**Regret:**
```
Regret_T = T·μ* - Σ_{t=1}^T μ_{A_t}
```

Where μ* is the best arm's mean.

### 15.2. Simple Algorithms

**ε-Greedy:**
- With prob ε: explore (random arm)
- With prob 1-ε: exploit (best arm so far)

**Decaying ε:** ε_t = 1/t

### 15.4. Upper Confidence Bound (UCB)

**UCB1:**
```
A_t = argmax_a [Q_t(a) + c·√(log(t)/N_t(a))]
```

Where N_t(a) is the number of times arm a was pulled.

**Intuition:** "Optimism in the face of uncertainty"

### 15.5. Thompson Sampling

**Bayesian approach:**
1. Maintain posterior distribution for each arm's mean
2. Sample from each posterior
3. Pull arm with highest sample

**For Bernoulli bandits with Beta prior:**
```python
for each round t:
    for each arm a:
        sample θ_a ~ Beta(α_a, β_a)
    pull arm a* = argmax_a θ_a
    update: if reward=1: α_{a*} += 1, else: β_{a*} += 1
```

---

## 16. Blending Learning and Planning (p.475)

### 16.1. Model-Based RL

**Approach:**
1. Learn environment model from experience
2. Plan using learned model
3. Execute and collect more data

**Dyna Architecture:**
- Direct RL: learn from real experience
- Model learning: fit model to experience
- Planning: simulate with model, update value/policy

### 16.3. Monte-Carlo Tree Search (MCTS)

**Four phases:**
1. **Selection:** Follow tree using UCB until leaf
2. **Expansion:** Add new node
3. **Simulation:** Random rollout to terminal
4. **Backpropagation:** Update statistics

**Used in:** AlphaGo, game playing

---

## 17. Summary and Real-World Considerations (p.487)

### Key Learnings

1. **MDP Framework:** Universal language for sequential decision problems
2. **Bellman Equations:** Foundation of all DP and RL
3. **DP Algorithms:** Exact solutions when model known
4. **RL Algorithms:** Learn from experience when model unknown
5. **Function Approximation:** Scale to large state spaces
6. **Exploration-Exploitation:** Fundamental tradeoff

### Real-World Challenges

- **Sample efficiency:** RL often needs lots of data
- **Safety:** Exploration can be dangerous
- **Reward design:** Hard to specify correctly
- **Partial observability:** Real states often hidden
- **Non-stationarity:** Environment changes over time
- **Sim-to-real gap:** Simulators imperfect

---

## APPENDICES

### Appendix B: Portfolio Theory (p.501)

**Efficient Frontier:** Set of portfolios with maximum return for given risk.

**CAPM:**
```
E[r_i] - r_f = β_i · (E[r_m] - r_f)
```

### Appendix C: Stochastic Calculus Basics (p.505)

**Brownian Motion Properties:**
- Continuous paths
- Independent increments
- W_t - W_s ~ N(0, t-s)

**Ito's Lemma:**
For f(t, X_t) where dX = μdt + σdW:
```
df = (∂f/∂t + μ·∂f/∂x + ½σ²·∂²f/∂x²)dt + σ·∂f/∂x·dW
```

### Appendix D: Hamilton-Jacobi-Bellman Equation (p.513)

**Continuous-time Bellman:**
```
0 = max_a {f(x,a) + ∂V/∂t + μ·∂V/∂x + ½σ²·∂²V/∂x²}
```

### Appendix E: Black-Scholes (p.515)

**Black-Scholes PDE:**
```
∂V/∂t + rS·∂V/∂S + ½σ²S²·∂²V/∂S² - rV = 0
```

**Call option price:**
```
C = S·N(d₁) - K·e^{-rT}·N(d₂)
```

Where:
```
d₁ = [log(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

---

## Key Equations Summary

### Value Functions
```
V^π(s) = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
Q^π(s,a) = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]
```

### Bellman Equations
```
V^π(s) = Σ_a π(a|s) · Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]
V*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γV*(s')]
```

### TD Learning
```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

### Q-Learning
```
Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γ·max_a Q(S_{t+1},a) - Q(S_t,A_t)]
```

### Policy Gradient
```
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) · Q^π(s,a)]
```

---

*Document extrait du livre "Foundations of Reinforcement Learning with Applications in Finance" par Ashwin Rao et Tikhon Jelvis (Stanford University). Pour usage éducatif.*
