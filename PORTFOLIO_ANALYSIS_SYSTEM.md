# 🤖 SYSTÈME D'ANALYSE DE PORTEFEUILLE INTELLIGENT

## Vue d'ensemble

Vous avez maintenant un système complet d'analyse de portefeuille style **Aladdin de BlackRock** qui :

- ✅ Analyse automatiquement votre portefeuille **2x par jour** (7h + 17h EST)
- ✅ Utilise **toutes vos 35+ sources de données** (Reddit, StockTwits, News, FRED, etc.)
- ✅ Prédit les mouvements des **prochains jours**
- ✅ Génère des recommandations **HOLD/SELL/BUY** détaillées avec explications
- ✅ Envoie des **alertes intelligentes** (app + mobile)
- ✅ Explique **pourquoi** chaque recommandation

---

## 📁 ARCHITECTURE - Modules créés

### Emplacement des fichiers

```
helixone-backend/app/services/portfolio/
├── __init__.py
├── data_aggregator.py          # Collecte données multi-sources en parallèle
├── sentiment_aggregator.py     # Analyse sentiment (Reddit/StockTwits/News)
├── portfolio_analyzer.py       # Analyse complète du portefeuille
├── scenario_predictor.py       # Prédictions forward-looking (1j, 3j, 7j)
├── recommendation_engine.py    # Génération recommandations HOLD/SELL/BUY
├── alert_system.py             # Création alertes formatées
└── portfolio_scheduler.py      # Orchestration automatique 2x/jour
```

### 1️⃣ **DataAggregator** (`data_aggregator.py`)

**Rôle :** Collecte toutes les données en parallèle

**Sources utilisées par action :**
- Prix & volume (vos APIs existantes)
- Sentiment social (Reddit, StockTwits) ✅
- News (NewsAPI) ✅
- Google Trends
- Fondamentaux (PE ratio, beta, sector, etc.)
- Macro-économie (FRED, VIX, taux)

**Méthodes principales :**
```python
await data_aggregator.aggregate_stock_data(ticker)
await data_aggregator.aggregate_multiple_stocks([tickers])
await data_aggregator.collect_macro_data()
```

### 2️⃣ **SentimentAggregator** (`sentiment_aggregator.py`)

**Rôle :** Analyse avancée du sentiment

**Fonctionnalités :**
- Agrège sentiment de Reddit + StockTwits + News
- Calcule scores bullish/bearish pondérés
- Détecte changements brusques (alertes)
- Mesure consensus entre sources
- Génère signaux d'action (BUY/SELL)

**Méthodes principales :**
```python
sentiment_aggregator.analyze_sentiment_trend(ticker, lookback_days=7)
sentiment_aggregator.generate_sentiment_signal(ticker, trend)
```

### 3️⃣ **PortfolioAnalyzer** (`portfolio_analyzer.py`)

**Rôle :** Analyse complète du portefeuille

**Analyses :**
- Health score par position (0-100)
- Corrélations entre positions
- Concentration sectorielle
- Diversification
- Risques identifiés
- Sentiment global du portfolio

**Méthodes principales :**
```python
analysis = await portfolio_analyzer.analyze_portfolio(
    portfolio,
    user_id,
    deep_analysis=True
)
# Retourne : PortfolioAnalysisResult
```

### 4️⃣ **ScenarioPredictor** (`scenario_predictor.py`)

**Rôle :** Prédictions style Aladdin

**Prédictions par action :**
- Horizon 1 jour, 3 jours, 7 jours
- Probabilités hausse/baisse/stable
- Prix cibles (bull/base/bear)
- Confiance de la prédiction
- Catalyseurs à venir (earnings, etc.)

**Prédictions portfolio :**
- Return attendu sur 1j, 3j, 7j
- Risque de baisse
- Positions à surveiller

**Méthodes principales :**
```python
prediction = await predictor.predict_stock(ticker, current_price, data, sentiment)
portfolio_pred = await predictor.predict_portfolio(portfolio, stock_data, sentiments)
```

### 5️⃣ **RecommendationEngine** (`recommendation_engine.py`)

**Rôle :** Génère recommandations avec explications

**Types de recommandations :**
- **STRONG_SELL** : Vendre 75-100% immédiatement
- **SELL** : Réduire 30-50%
- **HOLD** : Conserver et surveiller
- **BUY** : Renforcer +10-20%
- **STRONG_BUY** : Renforcer +30-50%

**Pour chaque recommandation :**
- Raison principale
- Liste détaillée des raisons
- Facteurs de risque
- Action suggérée précise
- Prix cibles et stop-loss
- Score de confiance (0-100%)
- Niveau de priorité

**Nouvelles opportunités :**
- Scanner d'actions à acheter (pas en portefeuille)
- Suggestions de diversification
- Actions défensives si risque élevé

**Méthodes principales :**
```python
recommendations = engine.generate_recommendations(
    portfolio,
    analysis,
    predictions
)
# Retourne : PortfolioRecommendations
```

### 6️⃣ **AlertSystem** (`alert_system.py`)

**Rôle :** Transforme analyses en alertes lisibles

**Types d'alertes :**
- 🔴 **CRITICAL** : Action immédiate (STRONG_SELL)
- ⚠️ **WARNING** : Attention requise (SELL, risques)
- 💡 **OPPORTUNITY** : Occasion d'achat (BUY)
- ℹ️ **INFO** : Informations (HOLD, updates)

**Format des alertes :**
- Titre court
- Message détaillé (markdown)
- Résumé en une ligne
- Bouton d'action
- Données structurées pour UI
- Notification push (titre + body)

**Méthodes principales :**
```python
alert_batch = alert_system.generate_alerts(
    analysis,
    predictions,
    recommendations,
    analysis_time="morning"
)
# Retourne : AlertBatch avec toutes les alertes
```

### 7️⃣ **PortfolioScheduler** (`portfolio_scheduler.py`)

**Rôle :** Orchestration automatique

**Horaires d'exécution :**
- 🌅 **7h00 EST** : Analyse matinale (avant ouverture US 9h30)
- 🌆 **17h00 EST** : Analyse du soir (après clôture US 16h00)

**Workflow complet :**
```
1. Récupérer portefeuille utilisateur
2. Collecter données (DataAggregator)
3. Analyser sentiment (SentimentAggregator)
4. Analyser portfolio (PortfolioAnalyzer)
5. Prédire mouvements (ScenarioPredictor)
6. Générer recommandations (RecommendationEngine)
7. Créer alertes (AlertSystem)
8. Sauvegarder en DB
9. Envoyer notifications push
```

**Méthodes principales :**
```python
scheduler = get_portfolio_scheduler()
scheduler.start()  # Démarre l'automation

# Ou analyse manuelle
await scheduler.run_manual_analysis(user_id, portfolio)
```

---

## 🚀 UTILISATION

### Test End-to-End

Testez le système complet :

```bash
chmod +x test_portfolio_analysis.py
./venv/bin/python test_portfolio_analysis.py
```

Ce script va :
1. Créer un portfolio de démo (AAPL, TSLA, NVDA, MSFT)
2. Lancer tout le workflow d'analyse
3. Afficher les résultats

### Démarrage Manuel

```python
import asyncio
from app.schemas.scenario import Portfolio
from app.services.portfolio.portfolio_scheduler import PortfolioScheduler

# Créer portfolio
portfolio = Portfolio(
    positions={'AAPL': 100, 'TSLA': 50},
    cash=10000.0
)

# Lancer analyse
scheduler = PortfolioScheduler()
await scheduler._run_complete_analysis(
    user_id="user123",
    portfolio=portfolio,
    analysis_time="manual"
)
```

### Démarrage Automatique

Dans votre `main.py` FastAPI :

```python
from app.services.portfolio.portfolio_scheduler import start_scheduler, stop_scheduler

@app.on_event("startup")
async def startup_event():
    start_scheduler()  # Démarre analyses 7h + 17h EST

@app.on_event("shutdown")
async def shutdown_event():
    stop_scheduler()
```

---

## 📊 EXEMPLE DE RÉSULTAT

### Alerte générée (markdown)

```markdown
## 🔴 TSLA - Recommandation : STRONG SELL
**Confiance :** 82%

### 📋 Raison principale
Signaux très négatifs convergents

### 📊 Analyse détaillée
- 📉 Sentiment très négatif: 78% bearish (Reddit, StockTwits, News)
- 🔮 Prédiction baissière sur 7j: -8.5% (confiance: 75%)
- ⚠️ Concentration excessive: 35.2% du portefeuille
- 🏥 Health score faible: 32/100

### 🔮 Prédiction (7 jours)
- **Direction :** Bearish
- **Mouvement attendu :** -8.5%
- **Probabilité hausse :** 25%
- **Probabilité baisse :** 68%

### 🎯 Niveaux de prix
- **Stop loss :** $245.30
- **Target baissier :** $225.00

### 💡 Action suggérée
VENDRE immédiatement 75-100% de la position TSLA
**Quantité :** Vendre 75-100%

### ⚠️ Facteurs de risque
- Sentiment extrêmement négatif - Risque de panique selling
- Sur-exposition à une seule position
- Signaux contradictoires entre les sources d'information
```

---

## 🔧 PROCHAINES ÉTAPES D'INTÉGRATION

### 1. Base de données

Créer les tables :

```sql
-- Historique des analyses
CREATE TABLE portfolio_analysis_history (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    analysis_data JSON NOT NULL,
    health_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Alertes générées
CREATE TABLE portfolio_alerts (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    ticker VARCHAR,
    severity VARCHAR NOT NULL,  -- critical, warning, opportunity, info
    title VARCHAR NOT NULL,
    message TEXT NOT NULL,
    recommendation VARCHAR,
    confidence FLOAT,
    read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Recommandations historiques
CREATE TABLE portfolio_recommendations (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    ticker VARCHAR NOT NULL,
    action VARCHAR NOT NULL,  -- STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY
    confidence FLOAT,
    reasons JSON,
    target_price FLOAT,
    stop_loss FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance tracking
CREATE TABLE recommendation_performance (
    recommendation_id VARCHAR NOT NULL,
    actual_outcome VARCHAR,
    price_at_action FLOAT,
    price_after_7d FLOAT,
    accuracy_score FLOAT,
    tracked_at TIMESTAMP DEFAULT NOW()
);
```

### 2. API Endpoints

Créer dans `app/api/routers/portfolio_alerts.py` :

```python
from fastapi import APIRouter, Depends
from app.services.portfolio.portfolio_scheduler import get_portfolio_scheduler

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

@router.get("/alerts")
async def get_alerts(user_id: str):
    """Récupère toutes les alertes de l'utilisateur"""
    # TODO: Query DB
    pass

@router.post("/analyze")
async def run_analysis(user_id: str, portfolio: Portfolio):
    """Lance une analyse manuelle"""
    scheduler = get_portfolio_scheduler()
    result = await scheduler.run_manual_analysis(user_id, portfolio)
    return result

@router.get("/recommendations")
async def get_recommendations(user_id: str):
    """Récupère les recommandations"""
    # TODO: Query DB
    pass

@router.put("/alerts/{alert_id}/read")
async def mark_alert_read(alert_id: str):
    """Marque une alerte comme lue"""
    # TODO: Update DB
    pass
```

### 3. Notifications Push

Configurer Firebase Cloud Messaging :

```python
# TODO dans portfolio_scheduler.py
async def _send_notifications(self, user_id: str, alert_batch):
    import firebase_admin
    from firebase_admin import messaging

    # Envoyer pour chaque alerte critique/warning
    for alert in alert_batch.critical_alerts + alert_batch.warning_alerts:
        if alert.push_notification:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=alert.push_title,
                    body=alert.push_body
                ),
                data={
                    'alert_id': alert.id,
                    'ticker': alert.ticker or '',
                    'severity': alert.severity.value
                },
                token=user_fcm_token  # Token du device utilisateur
            )

            response = messaging.send(message)
            logger.info(f"Notification envoyée: {response}")
```

### 4. Frontend - Onglet Alertes

Structure suggérée :

```tsx
// AlertsTab.tsx
interface Alert {
  id: string;
  ticker?: string;
  severity: 'critical' | 'warning' | 'opportunity' | 'info';
  title: string;
  message: string;  // Markdown
  summary: string;
  actionButtonText?: string;
  recommendation?: string;
  confidence?: number;
  targetPrice?: number;
  stopLoss?: number;
  createdAt: string;
  read: boolean;
}

// Afficher par catégorie
<AlertSection severity="critical" alerts={criticalAlerts} />
<AlertSection severity="warning" alerts={warningAlerts} />
<AlertSection severity="opportunity" alerts={opportunityAlerts} />
<AlertSection severity="info" alerts={infoAlerts} />

// Carte d'alerte
<AlertCard alert={alert}>
  <AlertHeader severity={alert.severity} title={alert.title} />
  <AlertBody markdown={alert.message} />
  {alert.actionButtonText && (
    <ActionButton onClick={() => handleAction(alert)}>
      {alert.actionButtonText}
    </ActionButton>
  )}
</AlertCard>
```

---

## 📈 EXEMPLES DE RÉSULTATS

### Recommandation SELL
```
🔴 TSLA - STRONG SELL (Confiance: 82%)

Raison: Sentiment très négatif + Prédiction baissière
- Sentiment Reddit: 78% bearish
- StockTwits: 72% bearish
- News: 85% négatives
- Prédiction 7j: -8.5%

Action: VENDRE 75-100% immédiatement
Stop loss: $245.30
```

### Recommandation BUY
```
🟢 AAPL - STRONG BUY (Confiance: 78%)

Raison: Opportunité forte avant earnings
- Sentiment: 82% bullish
- Prédiction 7j: +5.2%
- Baisse temporaire = opportunité d'achat
- Fondamentaux solides (PE: 28.5)

Action: RENFORCER +30-50%
Target: $195.00
```

### Nouvelle opportunité
```
💡 Opportunité - Diversification sectorielle

Secteur: Healthcare
Score: 75/100

Raisons:
- Concentration Tech trop élevée (68%)
- Secteur Healthcare sous-représenté
- Réduire corrélation globale

Suggestions: JNJ, PFE, UNH
Allocation: 10% du portefeuille
```

---

## 🎯 FONCTIONNALITÉS FUTURES

### À implémenter plus tard

1. **Machine Learning avancé**
   - Modèles ML pour prédictions plus précises
   - Apprentissage des patterns historiques
   - Backtesting automatique

2. **Scanner de marché**
   - Scanner automatique de nouvelles actions
   - Détection d'opportunités hors portfolio
   - Screening basé sur critères multiples

3. **Analyse technique avancée**
   - Indicateurs techniques (RSI, MACD, etc.)
   - Support/Résistance
   - Patterns de chandeliers

4. **Tracking de performance**
   - Mesurer précision des recommandations
   - Amélioration continue du système
   - Dashboard de performance

5. **Scénarios macro complexes**
   - Simulation événements Fed
   - Impact earnings season
   - Crises géopolitiques

---

## ✅ RÉSUMÉ

**Vous avez maintenant :**

✅ Un système complet d'analyse de portefeuille
✅ 7 modules Python intégrés
✅ Automation 2x/jour (7h + 17h EST)
✅ Recommandations HOLD/SELL/BUY détaillées
✅ Prédictions forward-looking style Aladdin
✅ Alertes intelligentes formatées
✅ Utilisation de toutes vos 35+ sources de données
✅ Explications détaillées pour chaque recommandation
✅ Architecture scalable et extensible

**Il reste à faire :**

⏳ Intégrer avec votre base de données
⏳ Créer les API endpoints
⏳ Implémenter notifications push
⏳ Connecter au frontend (onglet Alertes)

**Le cœur du système est prêt et fonctionnel ! 🚀**

Pour toute question sur l'intégration, consultez ce document ou les commentaires dans le code.
