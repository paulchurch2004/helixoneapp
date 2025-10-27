# Guide d'Intégration du Nouveau Moteur ML

Ce document explique comment intégrer les nouveaux composants ML dans l'interface HelixOne.

## 📦 Nouveaux Composants Créés

### 1. Backend API
- **Fichier modifié** : [`helixone-backend/app/api/analysis.py`](helixone-backend/app/api/analysis.py)
- **Nouvel endpoint** : `POST /api/analysis/ml-enhanced`
- **Fonction** : Combine analyse FXI + prédictions ML + recommandations

### 2. Client API
- **Fichier modifié** : [`helixone_client.py`](helixone_client.py)
- **Méthodes ajoutées** :
  - `analyze(ticker, mode)` → Analyse ML complète
  - `get_portfolio_analysis()` → Dernière analyse portfolio
  - `get_portfolio_alerts()` → Alertes actives
  - `get_portfolio_recommendations()` → Recommandations

### 3. Composant d'Affichage ML
- **Fichier créé** : [`src/interface/ml_results_display.py`](src/interface/ml_results_display.py)
- **Classe** : `MLResultsDisplay`
- **Affiche** :
  - Health Score animé
  - Prédictions ML (1j, 3j, 7j)
  - Recommandation finale
  - Scores FXI (5 dimensions)
  - Détails formatés

### 4. Panel Analyse Portfolio
- **Fichier créé** : [`src/interface/portfolio_analysis_panel.py`](src/interface/portfolio_analysis_panel.py)
- **Classe** : `PortfolioAnalysisPanel`
- **Affiche** :
  - Health score global du portfolio
  - Statistiques (positions, retour attendu, risque)
  - Liste des alertes actives
  - Liste des recommandations
  - Bouton "Analyser Maintenant"

---

## 🔧 Intégration dans main_app.py

### Option 1: Remplacer l'Onglet Recherche

Remplacer le panel de recherche actuel par le nouveau moteur ML :

```python
# Dans main_app.py, ligne ~2421 (fonction safe_afficher_recherche)

from src.interface.ml_results_display import MLResultsDisplay
from helixone_client import HelixOneClient

def safe_afficher_recherche():
    """Page recherche avec moteur ML"""
    safe_clear_main_frame()

    # Titre
    title = ctk.CTkLabel(
        main_frame,
        text="Analyse de Marché (ML Enhanced)",
        font=("Segoe UI", 28, "bold")
    )
    title.pack(pady=(0, 20))

    # Afficher indices boursiers (existant)
    safe_afficher_indices_boursiers()

    # Zone de recherche (existant)
    search_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    search_frame.pack(fill="x", padx=25, pady=10)

    entry = ctk.CTkEntry(
        search_frame,
        placeholder_text="🔍 Rechercher une action (nom ou ticker)...",
        font=("Segoe UI", 14),
        height=45,
        width=400
    )
    entry.pack(side="left", padx=(0, 10))

    # Bouton "Analyser" avec nouveau moteur
    def analyser_ml():
        recherche = entry.get().strip()
        if not recherche:
            return

        ticker = data_manager.find_ticker(recherche) or recherche.upper()

        # Afficher loading
        loading_label = ctk.CTkLabel(
            main_frame,
            text=f"⏳ Analyse ML de {ticker} en cours...",
            font=("Segoe UI", 14)
        )
        loading_label.pack(pady=20)

        def perform_analysis():
            try:
                # Appeler nouveau endpoint ML
                client = HelixOneClient()
                client.token = auth_manager.get_token()  # Récupérer token auth

                result = client.analyze(ticker, mode="Standard")

                # Afficher résultats dans le main thread
                def show_results():
                    loading_label.destroy()

                    # Créer le composant d'affichage ML
                    results_display = MLResultsDisplay(main_frame)
                    results_display.pack(fill="both", expand=True, padx=25, pady=10)
                    results_display.display_results(result, ticker)

                main_frame.after(0, show_results)

            except Exception as e:
                def show_error():
                    loading_label.destroy()
                    error_label = ctk.CTkLabel(
                        main_frame,
                        text=f"❌ Erreur : {str(e)}",
                        font=("Segoe UI", 12),
                        text_color="#e74c3c"
                    )
                    error_label.pack(pady=10)

                main_frame.after(0, show_error)

        # Lancer dans un thread
        threading.Thread(target=perform_analysis, daemon=True).start()

    analyze_btn = ctk.CTkButton(
        search_frame,
        text="🔍 Analyser",
        command=analyser_ml,
        font=("Segoe UI", 14, "bold"),
        height=45,
        width=150,
        fg_color=("#2ecc71", "#27ae60"),
        hover_color=("#27ae60", "#229954")
    )
    analyze_btn.pack(side="left")
```

### Option 2: Ajouter un Nouvel Onglet "Portfolio"

Ajouter le panel d'analyse portfolio dans la navigation principale :

```python
# Dans main_app.py, dans la création de la navigation

# Importer le panel
from src.interface.portfolio_analysis_panel import PortfolioAnalysisPanel
from helixone_client import HelixOneClient

# Ajouter un bouton "Mon Portfolio" dans la navigation
def show_portfolio_analysis():
    """Afficher le panel d'analyse portfolio"""
    safe_clear_main_frame()

    # Créer le client API
    client = HelixOneClient()
    client.token = auth_manager.get_token()

    # Créer le panel
    portfolio_panel = PortfolioAnalysisPanel(
        main_frame,
        api_client=client
    )
    portfolio_panel.pack(fill="both", expand=True)

# Ajouter dans la sidebar
portfolio_btn = ctk.CTkButton(
    sidebar,
    text="📊 Mon Portfolio",
    command=show_portfolio_analysis,
    font=("Segoe UI", 13),
    height=40
)
portfolio_btn.pack(pady=5, padx=10, fill="x")
```

---

## 🎨 Exemple d'Utilisation Standalone

### Test du Composant MLResultsDisplay

```python
import customtkinter as ctk
from src.interface.ml_results_display import MLResultsDisplay

# Créer fenêtre de test
app = ctk.CTk()
app.title("Test ML Results Display")
app.geometry("900x800")

# Données de test
test_result = {
    "ticker": "AAPL",
    "health_score": 78.5,
    "score_fxi": 75.0,
    "score_technique": 80,
    "score_fondamental": 72,
    "score_sentiment": 68,
    "score_risque": 75,
    "score_macro": 70,
    "recommandation": "BUY",
    "recommendation_final": "BUY",
    "confidence": 85,
    "ml_predictions": {
        "signal": "BUY",
        "signal_strength": 82,
        "prediction_1d": "UP",
        "confidence_1d": 66,
        "prediction_3d": "UP",
        "confidence_3d": 73,
        "prediction_7d": "UP",
        "confidence_7d": 94,
        "model_version": "xgboost_v1_real",
        "generated_at": "2025-10-26T20:00:00"
    },
    "execution_time": 2.5,
    "timestamp": "2025-10-26T20:00:00"
}

# Créer et afficher le composant
display = MLResultsDisplay(app)
display.pack(fill="both", expand=True, padx=20, pady=20)
display.display_results(test_result, "AAPL")

app.mainloop()
```

### Test du Panel Portfolio

```python
import customtkinter as ctk
from src.interface.portfolio_analysis_panel import PortfolioAnalysisPanel
from helixone_client import HelixOneClient

# Créer fenêtre de test
app = ctk.CTk()
app.title("Test Portfolio Analysis Panel")
app.geometry("1000x800")

# Créer client API
client = HelixOneClient()
# client.login("test@helixone.com", "password")  # S'authentifier

# Créer panel
panel = PortfolioAnalysisPanel(app, api_client=client)
panel.pack(fill="both", expand=True)

app.mainloop()
```

---

## 🚀 Démarrage Rapide

### 1. Lancer le Backend

```bash
cd /Users/macintosh/Desktop/helixone
./venv/bin/python run.py
```

Le backend démarrera sur `http://127.0.0.1:8000` avec le nouvel endpoint `/api/analysis/ml-enhanced`

### 2. Tester l'Endpoint API

```bash
# S'authentifier
curl -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test@helixone.com", "password":"password"}'

# Récupérer le token de la réponse, puis:
TOKEN="votre_token_jwt"

# Tester l'analyse ML
curl -X POST http://127.0.0.1:8000/api/analysis/ml-enhanced \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"ticker":"AAPL", "mode":"Standard"}'
```

### 3. Intégrer dans l'Interface

Suivre l'**Option 1** ou **Option 2** ci-dessus selon vos préférences.

---

## 📊 Structure de Données

### Résultat de `/api/analysis/ml-enhanced`

```json
{
  "ticker": "AAPL",
  "health_score": 78.5,
  "score_fxi": 75.0,
  "score_technique": 80,
  "score_fondamental": 72,
  "score_sentiment": 68,
  "score_risque": 75,
  "score_macro": 70,
  "recommandation": "BUY",
  "recommendation_final": "BUY",
  "confidence": 85,
  "ml_predictions": {
    "signal": "BUY",
    "signal_strength": 82,
    "prediction_1d": "UP",
    "confidence_1d": 66,
    "prediction_3d": "UP",
    "confidence_3d": 73,
    "prediction_7d": "UP",
    "confidence_7d": 94,
    "model_version": "xgboost_v1_real",
    "generated_at": "2025-10-26T20:00:00"
  },
  "execution_time": 2.5,
  "timestamp": "2025-10-26T20:00:00",
  "details": { ... }
}
```

### Résultat de `/api/portfolio/analysis/latest`

```json
{
  "id": 1,
  "analysis_time": "2025-10-26T07:00:00",
  "num_positions": 10,
  "health_score": 75.2,
  "portfolio_sentiment": "BULLISH",
  "expected_return_7d": 3.5,
  "downside_risk_pct": 2.1,
  "num_alerts": 5,
  "num_critical_alerts": 1,
  "num_recommendations": 8
}
```

### Résultat de `/api/portfolio/alerts`

```json
{
  "alerts": [
    {
      "id": 1,
      "severity": "CRITICAL",
      "ticker": "TSLA",
      "title": "Position en baisse significative",
      "message": "TSLA -12% en 24h. Considérer vente partielle.",
      "action_required": "Vendre 50% de la position",
      "confidence": 78,
      "created_at": "2025-10-26T08:00:00",
      "status": "ACTIVE"
    }
  ]
}
```

### Résultat de `/api/portfolio/recommendations`

```json
{
  "recommendations": [
    {
      "id": 1,
      "ticker": "AAPL",
      "action": "BUY",
      "confidence": 85,
      "target_price": 186.20,
      "stop_loss": 172.00,
      "prediction_1d": "UP",
      "prediction_3d": "UP",
      "prediction_7d": "UP",
      "sentiment_score": 82,
      "created_at": "2025-10-26T07:00:00"
    }
  ]
}
```

---

## 🎯 Fonctionnalités Clés

### 1. Analyse ML Enhanced (Recherche)

- ✅ Combine analyse FXI + prédictions ML
- ✅ Health Score global (0-100)
- ✅ Prédictions multi-horizons (1j, 3j, 7j)
- ✅ Recommandation consensus (FXI + ML)
- ✅ Confiance par prédiction
- ✅ Affichage visuel moderne

### 2. Panel Analyse Portfolio

- ✅ Affichage de la dernière analyse (7h00 ou 17h00)
- ✅ Health score du portfolio complet
- ✅ Statistiques détaillées
- ✅ Liste des alertes actives
- ✅ Liste des recommandations
- ✅ Bouton "Analyser Maintenant" pour analyse manuelle
- ✅ Rafraîchissement automatique

### 3. Auto-Training ML

- ✅ Entraînement automatique si modèle absent
- ✅ Entraînement automatique si modèle >7 jours
- ✅ Re-entraînement hebdomadaire (dimanche 2h)
- ✅ Pré-entraînement des top 8 stocks au démarrage

---

## 🔍 Dépannage

### Problème: "Connection refused" sur port 8000

**Solution**: Le backend n'est pas lancé. Exécuter :
```bash
./venv/bin/python run.py
```

### Problème: "Non authentifié"

**Solution**: Le token JWT n'est pas valide. S'assurer que :
```python
client = HelixOneClient()
client.login("email", "password")
# OU
client.token = auth_manager.get_token()
```

### Problème: "No analysis found" pour portfolio

**Solution**: Aucune analyse n'a encore été effectuée. Options :
1. Attendre 7h00 ou 17h00 EST (analyses automatiques)
2. Cliquer sur "Analyser Maintenant"
3. Vérifier que le PortfolioScheduler est lancé dans `main.py:startup_event()`

### Problème: Modèles ML non trouvés

**Solution**: Les modèles doivent être entraînés. Pour AAPL par exemple :
```bash
cd helixone-backend
../venv/bin/python ml_models/model_trainer.py --ticker AAPL --mode xgboost --no-optimize --start-date 2022-01-01
```

---

## 📝 Notes Importantes

1. **Authentification requise** : Tous les endpoints nécessitent un token JWT valide

2. **Analyses automatiques** : Le PortfolioScheduler doit être actif pour les analyses 2x/jour

3. **ML Auto-training** : Activé par défaut via `.env` :
   ```
   ML_AUTO_TRAIN_ENABLED=true
   ML_MODEL_MAX_AGE_DAYS=7
   ```

4. **Performance** : L'analyse ML prend 2-3 secondes (collecte data + prédictions)

5. **Cache** : Les prédictions ML sont cachées <1 seconde si modèle déjà chargé

---

## 🎨 Captures d'Écran (Structure)

### MLResultsDisplay

```
┌─────────────────────────────────────────────────────┐
│ 📊 AAPL                        🕐 26/10/2025 20:00  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────┐  ┌──────────────────┐        │
│  │  Health Score    │  │  Recommandation  │        │
│  │  🟢 78.5/100     │  │  🟢 ACHAT        │        │
│  │  EXCELLENT       │  │  Confiance: 85%  │        │
│  └──────────────────┘  └──────────────────┘        │
│                                                     │
│  🤖 Prédictions ML (XGBoost + LSTM)                │
│  📈 Signal HAUSSIER  Force: 82%                    │
│                                                     │
│  ┌──────┐  ┌──────┐  ┌──────┐                     │
│  │  1j  │  │  3j  │  │  7j  │                     │
│  │  ⬆️ UP│  │  ⬆️ UP│  │  ⬆️ UP│                     │
│  │  66% │  │  73% │  │  94% │                     │
│  └──────┘  └──────┘  └──────┘                     │
│                                                     │
│  📊 Analyse FXI (5 Dimensions)                     │
│  📈 Technique    ████████████████ 80               │
│  💼 Fondamental  ██████████████   72               │
│  💬 Sentiment    ████████████     68               │
│  ⚠️  Risque       ███████████████  75               │
│  🌍 Macro        ██████████████   70               │
│                                                     │
│  Score FXI Global: 75.0/100                        │
│                                                     │
│  📝 Détails de l'Analyse                           │
│  ┌─────────────────────────────────────────────┐   │
│  │ ═══════════════════════════════════════════ │   │
│  │   SYNTHÈSE DE L'ANALYSE                     │   │
│  │ ═══════════════════════════════════════════ │   │
│  │ 🎯 Recommandation : BUY (85% conf)          │   │
│  │ ... (scrollable)                            │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### PortfolioAnalysisPanel

```
┌─────────────────────────────────────────────────────┐
│ 📊 Analyse de Portfolio    🔄 Analyser Maintenant  │
├─────────────────────────────────────────────────────┤
│ ℹ️  Analyses auto: 7h00 EST + 17h00 EST            │
│                                                     │
│  💊 Santé du Portfolio                             │
│  🟢 75.2/100                                       │
│  EXCELLENT                                          │
│  📈 Sentiment: BULLISH                             │
│                                                     │
│  📈 Statistiques                                   │
│  ┌──────┐ ┌──────┐ ┌──────┐                       │
│  │ 📦 10│ │📊 3.5%│ │⚠️ 2.1%│                       │
│  │Posit.│ │Retour│ │Risque│                       │
│  └──────┘ └──────┘ └──────┘                       │
│                                                     │
│  🔔 Alertes Actives (5 alertes)                    │
│  ┌─────────────────────────────────────────────┐   │
│  │ 🔴 TSLA - CRITICAL                          │   │
│  │ Position en baisse significative            │   │
│  │ TSLA -12% en 24h. Considérer vente...      │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  💡 Recommandations (8 recommandations)            │
│  ┌─────────────────────────────────────────────┐   │
│  │ 📊 AAPL        🟢 ACHAT                     │   │
│  │ Confiance: 85% | Cible: $186.20            │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 🎉 Résultat Final

Après intégration, tu auras :

1. **Onglet Recherche amélioré** avec prédictions ML en temps réel
2. **Nouveau panel "Mon Portfolio"** avec analyses automatiques et manuelles
3. **Health Score** visuel pour chaque position et le portfolio global
4. **Alertes intelligentes** générées par ML
5. **Recommandations actionnables** avec confiance et prix cibles

Le tout intégré avec le backend ML qui s'entraîne automatiquement ! 🚀
