# 📊 Guide de Collecte de Données de Marché - HelixOne

## 🎯 Vue d'ensemble

Un système complet de collecte de données de marché a été implémenté pour alimenter le moteur de scénarios. Ce système permet de collecter, stocker et classifier des données historiques pour entraîner les modèles ML.

---

## 📁 Structure créée

### 1. **Modèles de données** (`app/models/market_data.py`)

#### Tables créées:

| Table | Description | Données stockées |
|-------|-------------|------------------|
| `market_data_ohlcv` | Prix OHLCV | Open, High, Low, Close, Volume, VWAP |
| `market_data_tick` | Tick-by-tick | Prix individuels, bid/ask, mid price |
| `market_data_quote` | Quotes temps réel | Bid/ask, spread, mid price |
| `data_collection_jobs` | Jobs de collecte | Statut, progression, résultats |
| `data_collection_schedules` | Collectes planifiées | Cron, intervalles, récurrence |
| `symbol_metadata` | Métadonnées symboles | Nom, secteur, industrie, beta |

#### Timeframes supportés:

```python
- TICK: Tick by tick
- 1s, 5s: Secondes
- 1m, 5m, 15m, 30m: Minutes
- 1h, 4h: Heures
- 1d: Journalier
- 1w: Hebdomadaire
- 1M: Mensuel
```

### 2. **Service de collecte** (`app/services/data_collector.py`)

Fonctionnalités:
- ✅ Collecte données journalières (Yahoo Finance)
- ✅ Collecte données intraday (1m, 5m, 15m, 30m, 1h)
- ✅ Collecte multi-symboles en parallèle
- ✅ Métadonnées des symboles (secteur, industrie, etc.)
- ✅ Collecte spécifique pour crises historiques
- ⏳ Tick-by-tick (à implémenter avec API premium)

### 3. **API REST** (`app/api/data_collection.py`)

Endpoints disponibles:
- `POST /api/data/collect/daily` - Collecte journalière
- `POST /api/data/collect/intraday` - Collecte intraday
- `POST /api/data/collect/crisis/{crisis_id}` - Collecte pour une crise
- `POST /api/data/collect/all-crises` - Toutes les crises
- `GET /api/data/crises` - Liste des crises disponibles
- `GET /api/data/jobs` - Liste des jobs de collecte
- `GET /api/data/coverage/{symbol}` - Couverture des données
- `POST /api/data/metadata/{symbol}` - Métadonnées d'un symbol

---

## 🚀 Utilisation

### Démarrer le serveur

```bash
cd /Users/macintosh/Desktop/helixone
HELIXONE_DEV=1 ./venv/bin/python run.py
```

### Tester l'API

#### 1. Lister les crises disponibles

```bash
curl http://127.0.0.1:8000/api/data/crises | python3 -m json.tool
```

Retourne:
```json
[
  {
    "id": "2008_crisis",
    "name": "2008 Financial Crisis",
    "start_date": "2007-10-09T00:00:00",
    "end_date": "2009-03-09T00:00:00",
    "duration_days": 517,
    "default_symbols": ["SPY", "DIA", "QQQ", "XLF", ...]
  },
  {
    "id": "covid_2020",
    "name": "COVID-19 Crash",
    ...
  }
]
```

#### 2. Collecter des données journalières

```bash
curl -X POST "http://127.0.0.1:8000/api/data/collect/daily" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2020-01-01T00:00:00",
    "end_date": "2023-12-31T00:00:00",
    "adjusted": true
  }'
```

Retourne un job ID pour suivre la progression.

#### 3. Collecter données intraday (1 minute)

```bash
curl -X POST "http://127.0.0.1:8000/api/data/collect/intraday" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "TSLA"],
    "interval": "1m",
    "period": "7d"
  }'
```

**Note**: Les données intraday Yahoo Finance sont limitées:
- 1m, 2m: 7 derniers jours
- 5m, 15m, 30m: 60 derniers jours
- 1h: 730 derniers jours

#### 4. Collecter une crise historique complète

```bash
curl -X POST "http://127.0.0.1:8000/api/data/collect/crisis/2008_crisis" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Collecte automatiquement tous les symboles pertinents pour la crise de 2008.

#### 5. Collecter TOUTES les crises

```bash
curl -X POST "http://127.0.0.1:8000/api/data/collect/all-crises" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Collecte les données pour:
- 2008 Financial Crisis (517 jours)
- COVID-19 Crash (33 jours)
- Dot-com Bubble (943 jours)
- Black Monday 1987 (8 jours)

#### 6. Vérifier la couverture des données

```bash
curl "http://127.0.0.1:8000/api/data/coverage/AAPL" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Retourne:
```json
{
  "symbol": "AAPL",
  "timeframes": {
    "1d": {
      "start": "2020-01-01T00:00:00",
      "end": "2023-12-31T00:00:00",
      "count": 1008
    },
    "1m": {
      "start": "2025-10-13T09:30:00",
      "end": "2025-10-20T16:00:00",
      "count": 2730
    }
  },
  "total_records": 3738
}
```

#### 7. Suivre un job de collecte

```bash
curl "http://127.0.0.1:8000/api/data/jobs/YOUR_JOB_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Retourne:
```json
{
  "id": "abc-123",
  "job_name": "Daily Data: AAPL, MSFT, GOOGL",
  "status": "completed",
  "progress": 100.0,
  "records_collected": 3024,
  "records_failed": 0,
  "created_at": "2025-10-20T10:00:00",
  "completed_at": "2025-10-20T10:05:23"
}
```

---

## 📦 Crises Historiques Prédéfinies

### 1. **2008 Financial Crisis** (`2008_crisis`)
- **Période**: 09/10/2007 → 09/03/2009 (517 jours)
- **Symboles**: SPY, DIA, QQQ, XLF (Financials), XLE (Energy), XLK (Tech), XLV (Healthcare), BAC, C, GS, JPM, AIG
- **Impact**: -56.7% (S&P 500)

### 2. **COVID-19 Crash** (`covid_2020`)
- **Période**: 19/02/2020 → 23/03/2020 (33 jours)
- **Symboles**: SPY, QQQ, XLE, XLV, AAPL, MSFT, AMZN, BA, DIS, AAL
- **Impact**: -33.9% (S&P 500)
- **Particularité**: Crash le plus rapide de l'histoire

### 3. **Dot-com Bubble** (`dotcom_2000`)
- **Période**: 10/03/2000 → 09/10/2002 (943 jours)
- **Symboles**: QQQ, XLK, CSCO, INTC, MSFT, ORCL, AMZN, EBAY
- **Impact**: -49.1% (NASDAQ), Tech -78%

### 4. **Black Monday 1987** (`black_monday_1987`)
- **Période**: 15/10/1987 → 22/10/1987 (8 jours)
- **Symboles**: SPY, DIA
- **Impact**: -22.6% en 1 seul jour (19/10/1987)

---

## 🔄 Workflow de collecte recommandé

### Étape 1: Collecter les métadonnées

```bash
# Collecter métadonnées pour les symboles importants
for symbol in AAPL MSFT GOOGL AMZN TSLA SPY QQQ; do
  curl -X POST "http://127.0.0.1:8000/api/data/metadata/$symbol" \
    -H "Authorization: Bearer $TOKEN"
done
```

### Étape 2: Collecter les données historiques des crises

```bash
# Collecter toutes les crises en une fois
curl -X POST "http://127.0.0.1:8000/api/data/collect/all-crises" \
  -H "Authorization: Bearer $TOKEN"
```

Durée estimée: **5-10 minutes** (dépend de la connexion internet)

### Étape 3: Collecter données récentes intraday

```bash
# Données 1 minute des 7 derniers jours
curl -X POST "http://127.0.0.1:8000/api/data/collect/intraday" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "SPY", "QQQ"],
    "interval": "1m",
    "period": "7d"
  }'
```

### Étape 4: Vérifier la couverture

```bash
# Vérifier ce qu'on a collecté
curl "http://127.0.0.1:8000/api/data/coverage/AAPL" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🗄️ Stockage des données

### Base de données SQLite

Fichier: `helixone-backend/helixone.db`

### Structure:

```
market_data_ohlcv
├── symbol: "AAPL"
├── timeframe: "1d" | "1m" | "5m" | ...
├── timestamp: datetime
├── open, high, low, close: float
├── volume: int
├── vwap: float (optional)
└── source: "yahoo" | "alphavantage" | ...

symbol_metadata
├── symbol: "AAPL"
├── name: "Apple Inc."
├── sector: "Technology"
├── industry: "Consumer Electronics"
├── market_cap: 3000000000000
├── beta: 1.25
└── ...
```

---

## 📊 Données disponibles après collecte complète

| Timeframe | Période couverte | Nombre de records (par symbol) |
|-----------|------------------|--------------------------------|
| Journalier (1d) | 2007-2025 | ~4500 jours |
| 1 heure (1h) | 2 dernières années | ~3300 heures |
| 15 minutes (15m) | 60 derniers jours | ~1560 barres |
| 5 minutes (5m) | 60 derniers jours | ~4680 barres |
| 1 minute (1m) | 7 derniers jours | ~2730 barres |

**Total par symbol**: ~16,770 enregistrements

Pour 50 symbols: **~838,500 enregistrements**

---

## 🔧 Configuration

### Limites Yahoo Finance (gratuit)

- **Rate limiting**: ~2000 requêtes/heure
- **Données intraday limitées**: 7-60 jours selon timeframe
- **Pas de tick-by-tick**: Nécessite API premium

### Prochaines sources à intégrer

#### Alpha Vantage (gratuit limité)
- 500 requêtes/jour (gratuit)
- Données fondamentales
- Indicateurs techniques calculés

#### Polygon.io (premium)
- Tick-by-tick
- Données options
- Données crypto

#### IEX Cloud (freemium)
- Données temps réel
- News et sentiment
- Données alternatives

---

## 🎯 Prochaines étapes

### Phase 1: Classification des données ✅
- [x] Modèles de données créés
- [x] Service de collecte implémenté
- [x] API REST fonctionnelle
- [x] Crises historiques prédéfinies

### Phase 2: Feature Engineering (en cours)
- [ ] Calculer les indicateurs techniques (RSI, MACD, BB)
- [ ] Extraire les patterns de crises
- [ ] Calculer les corrélations sectorielles
- [ ] Identifier les signaux précurseurs

### Phase 3: ML Training
- [ ] Préparer les datasets d'entraînement
- [ ] Entraîner le classifier de crises
- [ ] Entraîner le prédicteur d'impact
- [ ] Entraîner le modèle de corrélation
- [ ] GAN pour génération de scénarios

### Phase 4: Interface de collecte
- [ ] Panel de gestion de la collecte dans l'UI
- [ ] Visualisation de la couverture
- [ ] Planification de collectes récurrentes
- [ ] Monitoring des jobs

---

## 📝 Exemples de requêtes utiles

### Collecter données pour un portefeuille

```python
import requests

TOKEN = "your_token"
API = "http://127.0.0.1:8000/api/data"

# Portefeuille à analyser
portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Collecter 5 ans de données
response = requests.post(
    f"{API}/collect/daily",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={
        "symbols": portfolio,
        "start_date": "2019-01-01T00:00:00",
        "end_date": "2024-12-31T00:00:00",
        "adjusted": True
    }
)

job_id = response.json()["id"]
print(f"Job créé: {job_id}")

# Suivre la progression
import time
while True:
    status = requests.get(
        f"{API}/jobs/{job_id}",
        headers={"Authorization": f"Bearer {TOKEN}"}
    ).json()

    print(f"Progression: {status['progress']:.1f}% - {status['records_collected']} records")

    if status["status"] == "completed":
        break

    time.sleep(5)
```

### Analyser la couverture d'un portefeuille

```python
for symbol in portfolio:
    coverage = requests.get(
        f"{API}/coverage/{symbol}",
        headers={"Authorization": f"Bearer {TOKEN}"}
    ).json()

    print(f"\n{symbol}:")
    for tf, data in coverage["timeframes"].items():
        print(f"  {tf}: {data['count']} records ({data['start']} → {data['end']})")
```

---

## 🔗 Ressources

- **Documentation FastAPI**: http://127.0.0.1:8000/docs
- **Modèles**: `helixone-backend/app/models/market_data.py`
- **Service**: `helixone-backend/app/services/data_collector.py`
- **API**: `helixone-backend/app/api/data_collection.py`
- **Yahoo Finance Doc**: https://github.com/ranaroussi/yfinance

---

## ✅ Résumé

### Ce qui est prêt:
✅ Modèles de base de données (6 tables)
✅ Service de collecte (journalier + intraday)
✅ API REST complète (10+ endpoints)
✅ 4 crises historiques prédéfinies
✅ Métadonnées des symboles
✅ Tracking des jobs
✅ Collecte multi-symboles en parallèle

### Prêt à collecter:
- Données journalières: 2007 → aujourd'hui
- Données intraday: 7-60 derniers jours
- Crises historiques complètes
- Métadonnées: secteur, industrie, beta, etc.

### Prochaine étape:
**Feature Engineering** → Extraire les patterns et calculer les indicateurs pour l'entraînement ML!
