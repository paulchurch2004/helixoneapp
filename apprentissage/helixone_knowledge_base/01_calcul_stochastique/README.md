# 📘 MODULE 1: CALCUL STOCHASTIQUE
## Fondements Mathématiques pour la Finance Quantitative

---

## 📚 SOURCES PRINCIPALES
- **Shreve Vol I**: Binomial Asset Pricing - https://cms.dm.uba.ar/academico/materias/2docuat2016/analisis_cuantitativo_en_finanzas/Steve_Shreve_Stochastic_Calculus_for_Finance_I.pdf
- **Shreve Vol II**: Continuous-Time Models - https://cms.dm.uba.ar/academico/materias/2docuat2016/analisis_cuantitativo_en_finanzas/Steve_ShreveStochastic_Calculus_for_Finance_II.pdf
- **CMU Notes**: https://www.math.cmu.edu/~gautam/sj/teaching/2016-17/944-scalc-finance1/pdfs/notes.pdf

---

## 🎯 OBJECTIFS D'APPRENTISSAGE
1. Comprendre le mouvement brownien et ses propriétés
2. Maîtriser le calcul d'Itô
3. Résoudre des équations différentielles stochastiques (SDE)
4. Appliquer le théorème de Girsanov pour le changement de mesure
5. Utiliser les martingales pour la valorisation

---

## 📂 FICHIERS DU MODULE
- `brownian_motion.md` - Mouvement brownien et propriétés
- `ito_calculus.md` - Lemme d'Itô et intégrale stochastique
- `stochastic_differential_equations.md` - SDEs et solutions
- `girsanov_theorem.md` - Changement de mesure
- `martingales.md` - Théorie des martingales en finance
- `black_scholes_derivation.md` - Dérivation complète de Black-Scholes

---

## 🔑 CONCEPTS CLÉS

### Mouvement Brownien Standard W(t)
- W(0) = 0
- Incréments indépendants
- W(t) - W(s) ~ N(0, t-s)
- Trajectoires continues mais non-dérivables

### Lemme d'Itô
Pour f(t, X_t) où dX_t = μdt + σdW_t:
```
df = (∂f/∂t + μ∂f/∂x + ½σ²∂²f/∂x²)dt + σ(∂f/∂x)dW_t
```

### Équation de Black-Scholes
```
dS_t = μS_t dt + σS_t dW_t
```
Solution: S_t = S_0 exp((μ - σ²/2)t + σW_t)

---

## 💻 CODE PYTHON EXEMPLE

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths):
    """Simulate Geometric Brownian Motion"""
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    
    # Generate random increments
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    
    # Cumulative sum for Brownian motion
    W = np.cumsum(dW, axis=1)
    W = np.hstack([np.zeros((n_paths, 1)), W])
    
    # GBM formula
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    
    return t, S

# Example usage
t, paths = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1, n_steps=252, n_paths=1000)
```

---

## 📝 EXERCICES PRATIQUES

1. **Exercice 1**: Montrer que E[W_t²] = t
2. **Exercice 2**: Appliquer Itô à f(x) = x² pour X_t = W_t
3. **Exercice 3**: Dériver la formule de Black-Scholes via Itô
4. **Exercice 4**: Simuler 10000 trajectoires GBM et vérifier la distribution log-normale

---

## 🔗 LIENS AVEC AUTRES MODULES
- **→ Module 5 (Optimal Execution)**: Les SDEs modélisent la dynamique des prix
- **→ Module 6 (RL)**: Les processus stochastiques définissent l'environnement
- **→ Module 7 (Portfolio)**: Optimisation sous incertitude stochastique
