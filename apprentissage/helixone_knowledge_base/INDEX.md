# 🧠 HELIXONE KNOWLEDGE BASE
## Base de Connaissances IA Finance Quantitative

---

## 📋 SOMMAIRE

Cette base de connaissances contient tout le matériel nécessaire pour maîtriser l'IA appliquée à la finance quantitative, organisé en 8 modules.

| # | Module | Contenu Principal |
|---|--------|-------------------|
| 01 | [Calcul Stochastique](./01_calcul_stochastique/README.md) | Brownien, Itô, SDEs, Girsanov |
| 02 | [Machine Learning](./02_machine_learning/README.md) | Features, validation, modèles |
| 03 | [Deep Learning](./03_deep_learning/README.md) | LSTM, Transformer, CNN, VAE |
| 04 | [Microstructure](./04_microstructure/README.md) | Order book, market making |
| 05 | [Exécution Optimale](./05_optimal_execution/README.md) | Almgren-Chriss, TWAP/VWAP |
| 06 | [Reinforcement Learning](./06_reinforcement_learning/README.md) | MDP, Q-learning, Policy Gradient |
| 07 | [Portfolio](./07_portfolio/README.md) | Markowitz, Black-Litterman, Risk Parity |
| 08 | [Risk Management](./08_risk_management/README.md) | VaR, CVaR, Stress Testing |

---

## 🔗 SOURCES PDF À TÉLÉCHARGER

### ⭐ PRIORITÉ HAUTE

| Ressource | URL |
|-----------|-----|
| **Stanford RL Finance Book** | https://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf |
| **Shreve Vol I** | https://cms.dm.uba.ar/.../Steve_Shreve_...Finance_I.pdf |
| **Shreve Vol II** | https://cms.dm.uba.ar/.../Steve_Shreve_...Finance_II.pdf |
| **Almgren-Chriss** | https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf |
| **ENSAE ML for Finance** | https://www.master-statistique-finance.com/.../Machine%20Learning%20for%20finance_Eng.pdf |
| **Lehalle IPAM Slides** | http://helper.ipam.ucla.edu/publications/fmws2/fmws2_12928.pdf |

---

## 🤖 UTILISATION PAR CLAUDE/HELIXONE

### Recherche de Contenu
```python
import os
import glob

# Trouver tous les fichiers markdown
knowledge_dir = "helixone_knowledge_base"
md_files = glob.glob(f"{knowledge_dir}/**/*.md", recursive=True)

# Rechercher un concept
def search_concept(concept, files):
    results = []
    for f in files:
        with open(f, 'r') as file:
            content = file.read()
            if concept.lower() in content.lower():
                results.append(f)
    return results

# Exemple
print(search_concept("market impact", md_files))
```

### Charger un Module
```python
def load_module(module_name):
    path = f"helixone_knowledge_base/{module_name}/README.md"
    with open(path, 'r') as f:
        return f.read()

# Charger le module RL
rl_content = load_module("06_reinforcement_learning")
```

---

## 📚 STRUCTURE DES FICHIERS

```
helixone_knowledge_base/
├── INDEX.md                          # Ce fichier
├── 01_calcul_stochastique/
│   ├── README.md                     # Vue d'ensemble + théorie
│   └── ito_calculus.md              # Lemme d'Itô détaillé
├── 02_machine_learning/
│   └── README.md                     # ML complet avec code
├── 03_deep_learning/
│   └── README.md                     # Architectures DL
├── 04_microstructure/
│   └── README.md                     # Order book, market making
├── 05_optimal_execution/
│   └── README.md                     # Almgren-Chriss + code
├── 06_reinforcement_learning/
│   └── README.md                     # MDP, TD, Policy Gradient
├── 07_portfolio/
│   └── README.md                     # Markowitz, Black-Litterman
└── 08_risk_management/
    └── README.md                     # VaR, CVaR, stress testing
```

---

## 🎯 PARCOURS D'APPRENTISSAGE RECOMMANDÉ

### Phase 1: Fondamentaux (2-3 mois)
1. ✅ Calcul stochastique (Module 01)
2. ✅ ML fondamental (Module 02)
3. ✅ Portfolio theory (Module 07)

### Phase 2: Applications (3-4 mois)
4. ✅ Deep Learning (Module 03)
5. ✅ Microstructure (Module 04)
6. ✅ Risk Management (Module 08)

### Phase 3: Avancé (3-4 mois)
7. ✅ Exécution Optimale (Module 05)
8. ✅ Reinforcement Learning (Module 06)

### Phase 4: Intégration
9. 🚀 Projet HelixOne complet

---

## 📖 CONCEPTS CLÉS PAR MODULE

### Module 01 - Calcul Stochastique
- Mouvement brownien W(t)
- Lemme d'Itô: df = (∂f/∂t + μ∂f/∂x + ½σ²∂²f/∂x²)dt + σ∂f/∂x dW
- GBM: dS = μSdt + σSdW
- Théorème de Girsanov

### Module 02 - Machine Learning
- Feature engineering financier
- Cross-validation temporelle
- Régularisation (ElasticNet)
- Gradient boosting (LightGBM)

### Module 03 - Deep Learning
- LSTM avec attention
- Transformer pour time series
- CNN pour order book
- VAE pour anomaly detection

### Module 04 - Microstructure
- Carnet d'ordres (LOB)
- Modèle de Kyle (λ = σ_v/2σ_u)
- Market making (Avellaneda-Stoikov)
- Order flow imbalance

### Module 05 - Exécution Optimale
- Almgren-Chriss: x(t) = X₀ sinh(κ(T-t))/sinh(κT)
- Impact temporaire vs permanent
- TWAP, VWAP, IS
- Frontière efficiente coût-risque

### Module 06 - Reinforcement Learning
- MDP: (S, A, P, R, γ)
- Équation de Bellman
- Q-learning, SARSA
- Policy gradient, Actor-Critic

### Module 07 - Portfolio
- Markowitz: min w'Σw s.t. w'μ = r
- Black-Litterman
- Risk Parity
- Sharpe ratio

### Module 08 - Risk Management
- VaR_α = -quantile(1-α)
- CVaR = E[Loss | Loss > VaR]
- Mesures cohérentes
- Stress testing

---

*Base de connaissances créée pour HelixOne - Janvier 2026*
*Niveau: Professionnel - Equivalent MScT AI MaQI Polytechnique*
