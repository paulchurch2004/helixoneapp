# 📐 CALCUL D'ITÔ
## Intégrale Stochastique et Lemme d'Itô

---

## 1. INTÉGRALE STOCHASTIQUE D'ITÔ

### Définition
L'intégrale stochastique ∫₀ᵗ f(s) dW_s est définie comme limite de sommes de Riemann:
```
∫₀ᵗ f(s) dW_s = lim_{n→∞} Σᵢ f(tᵢ)(W_{tᵢ₊₁} - W_{tᵢ})
```

### Propriétés Fondamentales
1. **Espérance nulle**: E[∫₀ᵗ f(s) dW_s] = 0
2. **Isométrie d'Itô**: E[(∫₀ᵗ f(s) dW_s)²] = E[∫₀ᵗ f(s)² ds]
3. **Martingale**: Le processus M_t = ∫₀ᵗ f(s) dW_s est une martingale

### Règles de Calcul
```
dW_t · dW_t = dt
dW_t · dt = 0
dt · dt = 0
```

---

## 2. LEMME D'ITÔ (FORMULE CENTRALE)

### Version Scalaire
Pour X_t satisfaisant dX_t = μ(t,X_t)dt + σ(t,X_t)dW_t et f ∈ C²:

```
df(t, X_t) = [∂f/∂t + μ·∂f/∂x + ½σ²·∂²f/∂x²] dt + σ·∂f/∂x dW_t
```

### Version Multidimensionnelle
Pour X = (X¹,...,Xⁿ) avec dXⁱ = μⁱdt + Σⱼ σⁱʲdWʲ:

```
df = [∂f/∂t + Σᵢ μⁱ·∂f/∂xⁱ + ½ΣᵢΣⱼΣₖ σⁱᵏσʲᵏ·∂²f/∂xⁱ∂xʲ] dt + Σᵢ,ⱼ σⁱʲ·∂f/∂xⁱ dWʲ
```

---

## 3. APPLICATIONS FONDAMENTALES

### Application 1: Mouvement Brownien Géométrique
**Processus**: dS_t = μS_t dt + σS_t dW_t

**Appliquer Itô avec f(x) = ln(x)**:
- ∂f/∂x = 1/x
- ∂²f/∂x² = -1/x²

```
d(ln S_t) = [μ - ½σ²] dt + σ dW_t
```

**Solution**:
```
S_t = S_0 · exp((μ - ½σ²)t + σW_t)
```

### Application 2: Processus d'Ornstein-Uhlenbeck
**Processus**: dX_t = θ(μ - X_t)dt + σdW_t

**Solution**:
```
X_t = μ + (X_0 - μ)e^{-θt} + σ∫₀ᵗ e^{-θ(t-s)} dW_s
```

### Application 3: Processus CIR (Cox-Ingersoll-Ross)
**Processus**: dr_t = κ(θ - r_t)dt + σ√r_t dW_t

Utilisé pour modéliser les taux d'intérêt (condition de Feller: 2κθ > σ²)

---

## 4. FORMULE D'ITÔ POUR LE PRODUIT

Pour deux processus d'Itô X_t et Y_t:

```
d(X_t · Y_t) = X_t dY_t + Y_t dX_t + dX_t · dY_t
```

Où dX_t · dY_t = σ_X σ_Y ρ dt (si corrélés avec ρ)

---

## 5. CODE PYTHON - VÉRIFICATION NUMÉRIQUE

```python
import numpy as np

def verify_ito_lemma():
    """Verify Ito's lemma numerically for GBM"""
    np.random.seed(42)
    
    # Parameters
    S0, mu, sigma, T, n = 100, 0.05, 0.2, 1.0, 10000
    dt = T / n
    
    # Simulate GBM
    dW = np.random.normal(0, np.sqrt(dt), n)
    W = np.cumsum(dW)
    t = np.linspace(dt, T, n)
    
    # Exact solution
    S_exact = S0 * np.exp((mu - 0.5*sigma**2)*t + sigma*W)
    
    # Euler discretization
    S_euler = np.zeros(n+1)
    S_euler[0] = S0
    for i in range(n):
        S_euler[i+1] = S_euler[i] * (1 + mu*dt + sigma*dW[i])
    
    # Compare
    print(f"Final S (exact):  {S_exact[-1]:.4f}")
    print(f"Final S (Euler):  {S_euler[-1]:.4f}")
    print(f"Error: {abs(S_exact[-1] - S_euler[-1]):.4f}")

verify_ito_lemma()
```

---

## 6. EXERCICES

### Exercice 1: Appliquer Itô
Soit X_t = W_t². Trouver dX_t.

**Solution**:
```
dX_t = 2W_t dW_t + dt
```

### Exercice 2: Processus de Variance
Pour la variance réalisée V_t = ∫₀ᵗ σ_s² ds, montrer que si σ_s suit un processus d'Itô, V_t aussi.

### Exercice 3: Formule de Black-Scholes
Dériver l'EDP de Black-Scholes en utilisant Itô et l'argument de couverture delta.

---

## 🔗 RÉFÉRENCES
- Shreve, S. (2004). Stochastic Calculus for Finance II, Chapters 4-5
- Øksendal, B. (2003). Stochastic Differential Equations, Chapter 4
