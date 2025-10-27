# 🔌 Guide d'Intégration Interactive Brokers (IBKR)

## 🎯 Objectifs

Intégrer IBKR pour:
1. ✅ **Voir le portefeuille en temps réel** (positions, cash, P&L)
2. ✅ **Historique des ordres** (tous les trades passés)
3. ✅ **Alertes automatiques** (dangers détectés par le moteur de scénarios)
4. ✅ **Recommandations intelligentes** (hedging, diversification)
5. ⏳ **Passage d'ordres** (optionnel, à implémenter plus tard)

---

## 📚 Options d'API IBKR

Interactive Brokers propose plusieurs APIs:

### 1. **TWS API** (Trader Workstation API) ⭐⭐⭐⭐⭐
- ✅ **API officielle** la plus complète
- ✅ Temps réel
- ✅ Portefeuille, ordres, market data
- ❌ Nécessite TWS ou IB Gateway running
- ❌ Socket connection (complexe)

### 2. **ib_insync** (Wrapper Python) ⭐⭐⭐⭐⭐ **RECOMMANDÉ**
- ✅ **Wrapper moderne** de TWS API
- ✅ Asyncio support
- ✅ Syntaxe simple et pythonique
- ✅ Bien maintenu
- ✅ Documentation excellente
- ❌ Nécessite toujours TWS/Gateway

### 3. **Client Portal API** (REST API) ⭐⭐⭐
- ✅ REST API moderne
- ✅ Pas besoin de TWS
- ❌ Moins de fonctionnalités
- ❌ Moins stable
- ❌ Documentation limitée

### 4. **Flex Web Service** ⭐⭐
- ✅ Rapports de compte
- ✅ Historique complet
- ❌ Pas de temps réel
- ❌ Délai de 24h

---

## 🚀 Solution Recommandée: ib_insync + IB Gateway

### Pourquoi?
1. **ib_insync** = API simple et puissante
2. **IB Gateway** = Version headless de TWS (pas de GUI)
3. **Stable et fiable** = Utilisé par des hedge funds
4. **Temps réel** = Updates instantanées

### Architecture

```
HelixOne Backend
    ↓
ib_insync (Python)
    ↓
IB Gateway (Java)
    ↓
Interactive Brokers Servers
```

---

## 📋 Prérequis

### 1. Compte Interactive Brokers
- ✅ Compte actif (réel ou paper trading)
- ✅ Identifiants de connexion
- ✅ TWS ou IB Gateway installé

### 2. Configuration IBKR
- ✅ Activer API dans TWS/Gateway
- ✅ Socket port: 7497 (paper) ou 7496 (live)
- ✅ Client ID: unique par connexion

### 3. Software
- ✅ Python 3.11+
- ✅ ib_insync library
- ✅ IB Gateway ou TWS

---

## 🔧 Installation

### Étape 1: Installer IB Gateway

**macOS:**
```bash
# Télécharger depuis:
# https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

# Installer le .dmg
# Lancer IB Gateway
# Connexion avec tes identifiants
# Configuration > API > Enable ActiveX and Socket Clients ✅
# Socket port: 7497 (paper) ou 7496 (live)
```

**Alternative: TWS (avec GUI)**
```bash
# Si tu préfères utiliser TWS au lieu de Gateway
# https://www.interactivebrokers.com/en/trading/tws.php
```

### Étape 2: Installer ib_insync

```bash
cd /Users/macintosh/Desktop/helixone
./venv/bin/pip install ib_insync
```

### Étape 3: Tester la connexion

```python
from ib_insync import IB, util

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # 7497 = paper trading

print("✅ Connecté à IBKR!")
print(f"Compte: {ib.managedAccounts()}")

ib.disconnect()
```

---

## 📊 Fonctionnalités à Implémenter

### 1. 💼 Récupération Portefeuille

**Ce qu'on peut obtenir:**
```python
# Positions
positions = ib.positions()
# → [(Contract, Position, avgCost, unrealizedPNL)]

# Account summary
account = ib.accountSummary()
# → NetLiquidation, TotalCashValue, GrossPositionValue, etc.

# P&L
pnl = ib.pnl()
# → DailyPnL, UnrealizedPnL, RealizedPnL
```

**Structure des données:**
```python
{
    "account_id": "U1234567",
    "net_liquidation": 100000.00,
    "cash": 25000.00,
    "stock_value": 75000.00,
    "positions": [
        {
            "symbol": "AAPL",
            "position": 100,
            "avg_cost": 150.00,
            "market_price": 175.00,
            "market_value": 17500.00,
            "unrealized_pnl": 2500.00,
            "unrealized_pnl_pct": 16.67
        },
        ...
    ],
    "daily_pnl": 1200.00,
    "total_pnl": 5000.00,
    "last_update": "2025-10-20T12:00:00"
}
```

### 2. 📝 Historique des Ordres

**Ce qu'on peut obtenir:**
```python
# Ordres récents
trades = ib.trades()
# → [Trade(order, contract, orderStatus, fills)]

# Executions (fills)
executions = ib.reqExecutions()
# → [Execution(execId, time, symbol, side, shares, price, ...)]
```

**Structure:**
```python
{
    "orders": [
        {
            "order_id": "123456",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100,
            "order_type": "LIMIT",
            "limit_price": 175.00,
            "status": "Filled",
            "filled_quantity": 100,
            "avg_fill_price": 174.50,
            "commission": 1.00,
            "timestamp": "2025-10-20T10:30:00"
        },
        ...
    ]
}
```

### 3. 🔔 Alertes en Temps Réel

**Déclencheurs d'alertes:**
1. **Perte > X%** sur une position
2. **Portfolio drawdown > Y%**
3. **Scénario de crise détecté** (via moteur)
4. **Volatilité anormale** sur une position
5. **Corrélation risque** détectée

**Types d'alertes:**
```python
{
    "alert_type": "position_loss",
    "severity": "high",  # low, medium, high, critical
    "symbol": "AAPL",
    "message": "⚠️ AAPL: Perte de 15% détectée",
    "current_pnl": -2250.00,
    "current_pnl_pct": -15.0,
    "recommendation": "Considérer un stop loss ou hedging",
    "timestamp": "2025-10-20T12:00:00"
}
```

### 4. 🤖 Recommandations Intelligentes

**Scénarios → Recommandations:**

#### Scénario 1: Position en forte baisse
```python
if position_loss > -10%:
    recommendations = [
        "Stop Loss: Placer un stop à -15%",
        "Hedging: Acheter 10 PUT options",
        "Diversification: Réduire exposition de 50%"
    ]
```

#### Scénario 2: Portfolio non diversifié
```python
if sector_concentration > 50%:
    recommendations = [
        "Trop exposé au secteur Tech (65%)",
        "Ajouter des positions défensives (Utilities, Healthcare)",
        "Suggéré: XLU (Utilities ETF), XLV (Healthcare ETF)"
    ]
```

#### Scénario 3: Crise imminente détectée
```python
if crisis_probability > 70%:
    recommendations = [
        "⚠️ Probabilité de crise élevée (75%)",
        "Hedging urgent: SQQQ (3x inverse NASDAQ)",
        "Réduire leverage",
        "Augmenter cash position à 30%"
    ]
```

---

## 🏗️ Architecture Technique

### Structure de la Base de Données

```sql
-- Table des connexions IBKR
CREATE TABLE ibkr_connections (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES users(id),
    account_id VARCHAR NOT NULL,
    connection_type VARCHAR,  -- 'paper' ou 'live'
    is_active BOOLEAN DEFAULT TRUE,
    last_connected_at DATETIME,
    created_at DATETIME DEFAULT NOW()
);

-- Table des snapshots de portefeuille
CREATE TABLE portfolio_snapshots (
    id VARCHAR PRIMARY KEY,
    connection_id VARCHAR REFERENCES ibkr_connections(id),
    account_id VARCHAR NOT NULL,
    net_liquidation FLOAT,
    cash FLOAT,
    stock_value FLOAT,
    daily_pnl FLOAT,
    total_pnl FLOAT,
    positions JSON,  -- Array de positions
    timestamp DATETIME DEFAULT NOW()
);

-- Table des positions
CREATE TABLE portfolio_positions (
    id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR REFERENCES portfolio_snapshots(id),
    symbol VARCHAR NOT NULL,
    quantity FLOAT NOT NULL,
    avg_cost FLOAT,
    market_price FLOAT,
    market_value FLOAT,
    unrealized_pnl FLOAT,
    unrealized_pnl_pct FLOAT,
    timestamp DATETIME DEFAULT NOW()
);

-- Table des ordres
CREATE TABLE ibkr_orders (
    id VARCHAR PRIMARY KEY,
    connection_id VARCHAR REFERENCES ibkr_connections(id),
    order_id VARCHAR UNIQUE NOT NULL,
    symbol VARCHAR NOT NULL,
    action VARCHAR,  -- 'BUY', 'SELL'
    quantity FLOAT,
    order_type VARCHAR,  -- 'MARKET', 'LIMIT', 'STOP'
    limit_price FLOAT,
    stop_price FLOAT,
    status VARCHAR,  -- 'Submitted', 'Filled', 'Cancelled'
    filled_quantity FLOAT,
    avg_fill_price FLOAT,
    commission FLOAT,
    submitted_at DATETIME,
    filled_at DATETIME
);

-- Table des alertes
CREATE TABLE portfolio_alerts (
    id VARCHAR PRIMARY KEY,
    connection_id VARCHAR REFERENCES ibkr_connections(id),
    alert_type VARCHAR NOT NULL,
    severity VARCHAR,
    symbol VARCHAR,
    message TEXT,
    data JSON,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT NOW()
);

-- Table des recommandations
CREATE TABLE portfolio_recommendations (
    id VARCHAR PRIMARY KEY,
    alert_id VARCHAR REFERENCES portfolio_alerts(id),
    recommendation_type VARCHAR,
    action TEXT,
    rationale TEXT,
    priority INTEGER,
    is_applied BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT NOW()
);
```

### Services Python

```python
# app/services/ibkr_service.py

class IBKRService:
    def __init__(self, account_id: str):
        self.ib = IB()
        self.account_id = account_id

    async def connect(self, host='127.0.0.1', port=7497, client_id=1):
        """Connecter à IB Gateway"""
        self.ib.connect(host, port, clientId=client_id)

    async def get_portfolio(self) -> Dict:
        """Récupérer le portefeuille complet"""
        positions = self.ib.positions()
        account_summary = self.ib.accountSummary()
        pnl = self.ib.pnl()

        return {
            "positions": [self._format_position(p) for p in positions],
            "account_summary": self._format_account(account_summary),
            "pnl": self._format_pnl(pnl)
        }

    async def get_orders(self, days=30) -> List[Dict]:
        """Récupérer l'historique des ordres"""
        trades = self.ib.trades()
        return [self._format_trade(t) for t in trades]

    async def monitor_portfolio(self, callback):
        """Surveiller le portefeuille en temps réel"""
        self.ib.positionEvent += callback
        self.ib.pnlEvent += callback

    async def check_alerts(self) -> List[Dict]:
        """Vérifier si des alertes doivent être déclenchées"""
        portfolio = await self.get_portfolio()
        alerts = []

        # Check position losses
        for position in portfolio['positions']:
            if position['unrealized_pnl_pct'] < -10:
                alerts.append({
                    "type": "position_loss",
                    "severity": "high",
                    "symbol": position['symbol'],
                    "message": f"⚠️ {position['symbol']}: Perte de {position['unrealized_pnl_pct']:.1f}%"
                })

        return alerts
```

### API Endpoints

```python
# app/api/ibkr.py

@router.post("/ibkr/connect")
async def connect_ibkr(
    account_id: str,
    connection_type: str = "paper",
    current_user: User = Depends(get_current_user)
):
    """Connecter à Interactive Brokers"""
    # Implementation

@router.get("/ibkr/portfolio")
async def get_portfolio(
    current_user: User = Depends(get_current_user)
):
    """Récupérer le portefeuille actuel"""
    # Implementation

@router.get("/ibkr/orders")
async def get_orders(
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Récupérer l'historique des ordres"""
    # Implementation

@router.get("/ibkr/alerts")
async def get_alerts(
    current_user: User = Depends(get_current_user)
):
    """Récupérer les alertes actives"""
    # Implementation

@router.post("/ibkr/analyze")
async def analyze_portfolio(
    current_user: User = Depends(get_current_user)
):
    """Analyser le portefeuille avec le moteur de scénarios"""
    portfolio = await get_portfolio()

    # Passer au moteur de scénarios
    scenario_engine = get_scenario_engine()
    results = await scenario_engine.analyze_portfolio(portfolio)

    return results
```

---

## 🔒 Sécurité

### Stockage des Identifiants

**NE JAMAIS stocker:**
- ❌ Username IBKR en clair
- ❌ Password en clair
- ❌ API tokens en clair

**Solution:**
1. **Variables d'environnement**
```bash
# .env
IBKR_USERNAME=encrypted_value
IBKR_PASSWORD=encrypted_value
IBKR_ACCOUNT_ID=encrypted_value
```

2. **Encryption des credentials**
```python
from cryptography.fernet import Fernet

def encrypt_credentials(username: str, password: str) -> Dict:
    key = os.getenv("ENCRYPTION_KEY")
    f = Fernet(key)

    return {
        "username": f.encrypt(username.encode()),
        "password": f.encrypt(password.encode())
    }
```

### Permissions

- ✅ Read-only par défaut (portefeuille, ordres)
- ⚠️ Trading permissions = opt-in explicit
- ✅ 2FA pour connexion initiale

---

## 📱 Interface Utilisateur

### Panel "Mon Portefeuille IBKR"

```
┌─────────────────────────────────────────────────────┐
│  💼 Mon Portefeuille Interactive Brokers            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Account: U1234567                                  │
│  Type: Paper Trading                                │
│  Last Update: 12:45:23                              │
│                                                     │
│  ┌─────────────────────────────────────────┐      │
│  │ 💰 Net Liquidation:    $100,000.00      │      │
│  │ 💵 Cash:               $25,000.00       │      │
│  │ 📈 Stock Value:        $75,000.00       │      │
│  │ ↗️  Daily P&L:          +$1,200.00      │      │
│  │ 💯 Total P&L:          +$5,000.00       │      │
│  └─────────────────────────────────────────┘      │
│                                                     │
│  📊 Positions (5)                                   │
│  ┌───────────────────────────────────────────────┐ │
│  │ AAPL   100 shares   $175.00   +$2,500 (+16%)│ │
│  │ MSFT   50 shares    $380.00   +$1,500 (+8%) │ │
│  │ GOOGL  30 shares    $145.00   +$800 (+7%)   │ │
│  │ TSLA   25 shares    $245.00   -$500 (-8%)  ⚠│ │
│  │ SPY    10 shares    $450.00   +$200 (+5%)   │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  🔔 Alertes (2)                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ ⚠️  TSLA: Perte de 8% détectée               │ │
│  │     💡 Recommandation: Considérer stop loss  │ │
│  │                                               │ │
│  │ ℹ️  Portfolio: Concentration Tech élevée     │ │
│  │     💡 Suggéré: Diversifier vers défensives  │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  [🔄 Rafraîchir] [⚡ Analyser avec Scénarios]      │
│  [📊 Historique] [⚙️ Paramètres]                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Plan d'Implémentation

### Phase 1: Connexion de Base (2-3 heures)
- [ ] Installer ib_insync
- [ ] Créer modèles BDD
- [ ] Service de connexion IBKR
- [ ] Test connexion paper trading

### Phase 2: Portefeuille (2-3 heures)
- [ ] Récupération positions
- [ ] Récupération account summary
- [ ] Calcul P&L
- [ ] Stockage snapshots

### Phase 3: Ordres (2 heures)
- [ ] Récupération historique
- [ ] Parsing trades
- [ ] Stockage BDD

### Phase 4: Alertes (3-4 heures)
- [ ] Système d'alertes
- [ ] Détection pertes
- [ ] Détection concentration
- [ ] Notifications

### Phase 5: Intégration Scénarios (3-4 heures)
- [ ] Connecter moteur de scénarios
- [ ] Analyse automatique
- [ ] Recommandations
- [ ] Stress tests sur portfolio réel

### Phase 6: Interface (4-5 heures)
- [ ] Panel portefeuille
- [ ] Affichage positions
- [ ] Affichage alertes
- [ ] Bouton analyse

**Total: 16-21 heures de travail**

---

## 🧪 Étapes de Test

### 1. Paper Trading D'abord
- ✅ Toujours tester avec paper trading
- ✅ Vérifier toutes les fonctionnalités
- ✅ Valider alertes et recommandations

### 2. Tests Automatisés
```python
def test_portfolio_retrieval():
    service = IBKRService("paper_account")
    portfolio = service.get_portfolio()
    assert portfolio['net_liquidation'] > 0
    assert len(portfolio['positions']) >= 0
```

### 3. Live Trading (Optionnel)
- ⚠️ Uniquement après validation complète
- ⚠️ Commencer avec small amounts
- ⚠️ Monitor de près

---

## ⏭️ Prochaines Étapes Recommandées

**Aujourd'hui:**
1. Installer IB Gateway
2. Installer ib_insync
3. Tester connexion paper trading

**Cette semaine:**
4. Implémenter récupération portefeuille
5. Créer système d'alertes basique
6. Interface UI basique

**Prochaines semaines:**
7. Intégration moteur de scénarios
8. Recommandations intelligentes
9. Tests complets paper trading

---

**Tu veux qu'on commence maintenant?** 🚀

Je peux:
1. T'aider à installer IB Gateway
2. Configurer la connexion
3. Commencer l'implémentation du service IBKR

Dis-moi par quoi tu veux commencer!
