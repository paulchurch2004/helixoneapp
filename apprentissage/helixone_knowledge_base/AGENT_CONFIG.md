# 🤖 HELIXONE AGENT CONFIG
## Configuration pour Claude Agent / HelixOne AI

---

## 📋 INSTRUCTIONS POUR L'AGENT

Quand tu travailles sur le projet HelixOne, utilise cette base de connaissances comme référence pour:

1. **Implémenter des algorithmes de trading**
   - Réfère-toi à `05_optimal_execution/` pour Almgren-Chriss
   - Réfère-toi à `06_reinforcement_learning/` pour les stratégies RL

2. **Construire des modèles ML**
   - Réfère-toi à `02_machine_learning/` pour les features et validation
   - Réfère-toi à `03_deep_learning/` pour les architectures

3. **Gérer les risques**
   - Réfère-toi à `08_risk_management/` pour VaR/CVaR
   - Réfère-toi à `07_portfolio/` pour l'optimisation

4. **Analyser les marchés**
   - Réfère-toi à `04_microstructure/` pour le carnet d'ordres
   - Réfère-toi à `01_calcul_stochastique/` pour les modèles de prix

---

## 🔧 COMMANDES UTILES

### Charger la base de connaissances
```python
import os
import glob

KNOWLEDGE_BASE = "helixone_knowledge_base"

def load_all_knowledge():
    """Charge tout le contenu de la base"""
    content = {}
    for md_file in glob.glob(f"{KNOWLEDGE_BASE}/**/*.md", recursive=True):
        module = os.path.dirname(md_file).split('/')[-1]
        with open(md_file, 'r') as f:
            content[module] = f.read()
    return content

def get_module(module_name):
    """Charge un module spécifique"""
    path = f"{KNOWLEDGE_BASE}/{module_name}/README.md"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None

def search_knowledge(query):
    """Recherche dans la base"""
    results = []
    for md_file in glob.glob(f"{KNOWLEDGE_BASE}/**/*.md", recursive=True):
        with open(md_file, 'r') as f:
            content = f.read()
            if query.lower() in content.lower():
                results.append({
                    'file': md_file,
                    'preview': content[:500]
                })
    return results
```

### Exemple d'utilisation
```python
# Quand tu dois implémenter de l'exécution optimale
execution_knowledge = get_module("05_optimal_execution")

# Quand tu cherches un concept
results = search_knowledge("market impact")

# Quand tu as besoin de tout
all_knowledge = load_all_knowledge()
```

---

## 📊 MAPPING TÂCHES → MODULES

| Tâche HelixOne | Module(s) à consulter |
|----------------|----------------------|
| Trading algorithm | 05, 06 |
| Price prediction | 02, 03 |
| Risk calculation | 08 |
| Portfolio optimization | 07 |
| Order book analysis | 04 |
| Backtest framework | 02, 08 |
| Market making bot | 04, 06 |
| Feature engineering | 02, 04 |
| Model training | 02, 03 |

---

## ⚡ QUICK REFERENCE

### Almgren-Chriss Optimal Trajectory
```python
kappa = np.sqrt(lambda_risk * sigma**2 / eta)
x_t = X0 * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
```

### Q-Learning Update
```python
Q[s, a] = Q[s, a] + alpha * (reward + gamma * max(Q[s_next]) - Q[s, a])
```

### Portfolio Optimization
```python
w_tangent = Sigma_inv @ (mu - rf) / (ones @ Sigma_inv @ (mu - rf))
```

### VaR/CVaR
```python
var_99 = -np.percentile(returns, 1)
cvar_99 = -returns[returns < -var_99].mean()
```

---

## 🎯 OBJECTIF HELIXONE

Construire une plateforme de trading IA complète intégrant:
- ✅ Exécution optimale (Almgren-Chriss + RL)
- ✅ Prédiction de prix (LSTM/Transformer)
- ✅ Gestion de portefeuille (Mean-Variance + Black-Litterman)
- ✅ Risk management (VaR/CVaR temps réel)
- ✅ Market making (Avellaneda-Stoikov)

**Cette base de connaissances contient TOUT ce dont tu as besoin.**
