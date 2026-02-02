# 📕 MODULE 4: MICROSTRUCTURE DES MARCHÉS
## Order Books, Market Making et Price Formation

---

## 📚 SOURCES PRINCIPALES
- **Lehalle - IPAM UCLA Slides**: http://helper.ipam.ucla.edu/publications/fmws2/fmws2_12928.pdf
- **Market Microstructure in Practice** (Lehalle & Laruelle)
- **Cont, Stoikov & Talreja**: https://www.columbia.edu/~ww2040/orderbook.pdf

---

## 🎯 OBJECTIFS
1. Comprendre la structure du carnet d'ordres (LOB)
2. Analyser le processus de formation des prix
3. Modéliser le market making optimal
4. Quantifier l'impact de marché

---

## 🔑 CONCEPTS FONDAMENTAUX

### 1. Le Carnet d'Ordres (Limit Order Book)

**Structure**:
```
         ASKS (ventes)
    ┌─────────────────────┐
    │ 100.05  │    500    │  ← Best Ask (meilleure vente)
    │ 100.06  │   1200    │
    │ 100.07  │    800    │
    └─────────────────────┘
         MID = 100.025
    ┌─────────────────────┐
    │ 100.00  │    700    │  ← Best Bid (meilleur achat)
    │  99.99  │   1500    │
    │  99.98  │    300    │
    └─────────────────────┘
         BIDS (achats)
```

**Définitions clés**:
- **Spread** = Best Ask - Best Bid
- **Mid-price** = (Best Ask + Best Bid) / 2
- **Depth** = Volume disponible à chaque niveau
- **Imbalance** = (Vol_bid - Vol_ask) / (Vol_bid + Vol_ask)

### 2. Types d'Ordres

| Type | Description | Impact |
|------|-------------|--------|
| **Market Order** | Exécution immédiate au meilleur prix | Consomme la liquidité |
| **Limit Order** | Prix spécifié, attend exécution | Fournit la liquidité |
| **Stop Order** | Déclenché quand prix atteint seuil | Peut amplifier mouvements |
| **Iceberg** | Cache la taille réelle | Réduit l'impact |
| **TWAP** | Ordres étalés dans le temps | Minimise l'impact |

### 3. Modèle de Kyle (1985)

**Setup**:
- Informed trader connaît la valeur V
- Noise traders soumettent u ~ N(0, σ_u²)
- Market maker fixe le prix

**Équilibre**:
```
P = μ + λ(x + u)
```
où:
- x: ordre de l'informed trader
- λ = σ_v / (2σ_u): impact de marché ("Kyle's lambda")

**Stratégie optimale de l'informed**:
```
x = β(V - μ)    où β = σ_u / σ_v
```

### 4. Modèle de Glosten-Milgrom (1985)

**Spread comme protection contre sélection adverse**:
```
Ask = E[V | Buy] > E[V]
Bid = E[V | Sell] < E[V]
```

**Spread proportionnel à l'asymétrie d'information**

---

## 💻 MODÉLISATION DU CARNET D'ORDRES

### Modèle de Cont-Stoikov-Talreja (Queue-Reactive)

```python
import numpy as np
from collections import defaultdict

class LimitOrderBook:
    """
    Simple Limit Order Book implementation
    """
    def __init__(self, tick_size=0.01):
        self.tick_size = tick_size
        self.bids = defaultdict(int)  # price -> quantity
        self.asks = defaultdict(int)
        self.trade_history = []
    
    def add_limit_order(self, side, price, quantity):
        """Add a limit order to the book"""
        price = round(price / self.tick_size) * self.tick_size
        if side == 'buy':
            self.bids[price] += quantity
        else:
            self.asks[price] += quantity
    
    def submit_market_order(self, side, quantity):
        """Execute a market order"""
        remaining = quantity
        executed_qty = 0
        avg_price = 0
        
        if side == 'buy':
            prices = sorted(self.asks.keys())
            for price in prices:
                if remaining <= 0:
                    break
                available = self.asks[price]
                filled = min(remaining, available)
                self.asks[price] -= filled
                if self.asks[price] == 0:
                    del self.asks[price]
                remaining -= filled
                executed_qty += filled
                avg_price += filled * price
        else:  # sell
            prices = sorted(self.bids.keys(), reverse=True)
            for price in prices:
                if remaining <= 0:
                    break
                available = self.bids[price]
                filled = min(remaining, available)
                self.bids[price] -= filled
                if self.bids[price] == 0:
                    del self.bids[price]
                remaining -= filled
                executed_qty += filled
                avg_price += filled * price
        
        if executed_qty > 0:
            avg_price /= executed_qty
            self.trade_history.append({
                'side': side, 
                'quantity': executed_qty, 
                'price': avg_price
            })
        
        return executed_qty, avg_price
    
    @property
    def best_bid(self):
        return max(self.bids.keys()) if self.bids else None
    
    @property
    def best_ask(self):
        return min(self.asks.keys()) if self.asks else None
    
    @property
    def mid_price(self):
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self):
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    def imbalance(self, levels=1):
        """Calculate order book imbalance"""
        bid_vol = sum(self.bids[p] for p in sorted(self.bids.keys(), reverse=True)[:levels])
        ask_vol = sum(self.asks[p] for p in sorted(self.asks.keys())[:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0
        return (bid_vol - ask_vol) / total
    
    def __str__(self):
        s = "=== ORDER BOOK ===\n"
        s += "ASKS:\n"
        for price in sorted(self.asks.keys(), reverse=True)[:5]:
            s += f"  {price:.2f}: {self.asks[price]}\n"
        s += f"--- Spread: {self.spread:.2f} ---\n"
        s += "BIDS:\n"
        for price in sorted(self.bids.keys(), reverse=True)[:5]:
            s += f"  {price:.2f}: {self.bids[price]}\n"
        return s


class MarketMaker:
    """
    Simple market making strategy (Avellaneda-Stoikov style)
    """
    def __init__(self, gamma=0.1, sigma=0.02, k=1.5):
        """
        gamma: risk aversion
        sigma: volatility
        k: order arrival intensity
        """
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.inventory = 0
        self.cash = 0
        self.pnl_history = []
    
    def compute_quotes(self, mid_price, time_remaining):
        """
        Compute optimal bid/ask quotes
        Based on Avellaneda-Stoikov (2008)
        """
        # Reservation price (adjust for inventory)
        reservation = mid_price - self.gamma * self.sigma**2 * time_remaining * self.inventory
        
        # Optimal spread
        spread = self.gamma * self.sigma**2 * time_remaining + (2/self.gamma) * np.log(1 + self.gamma/self.k)
        
        bid = reservation - spread / 2
        ask = reservation + spread / 2
        
        return bid, ask
    
    def update(self, execution_price, side, quantity):
        """Update state after execution"""
        if side == 'buy':  # We bought (filled our bid)
            self.inventory += quantity
            self.cash -= execution_price * quantity
        else:  # We sold (filled our ask)
            self.inventory -= quantity
            self.cash += execution_price * quantity


# Simulation example
def simulate_market_making(n_steps=1000):
    """Simulate market making strategy"""
    np.random.seed(42)
    
    # Initialize
    mm = MarketMaker(gamma=0.1, sigma=0.02, k=1.5)
    price = 100.0
    T = 1.0  # 1 day
    dt = T / n_steps
    
    inventory_history = [0]
    pnl_history = [0]
    
    for t in range(n_steps):
        time_remaining = T - t * dt
        
        # Compute quotes
        bid, ask = mm.compute_quotes(price, time_remaining)
        
        # Random order arrivals
        if np.random.random() < 0.3:  # Sell order hits our bid
            mm.update(bid, 'buy', 1)
        if np.random.random() < 0.3:  # Buy order hits our ask
            mm.update(ask, 'sell', 1)
        
        # Price evolution (random walk)
        price += 0.02 * np.sqrt(dt) * np.random.randn()
        
        # Track
        inventory_history.append(mm.inventory)
        mark_to_market = mm.cash + mm.inventory * price
        pnl_history.append(mark_to_market)
    
    return inventory_history, pnl_history


if __name__ == "__main__":
    # Test LOB
    lob = LimitOrderBook()
    lob.add_limit_order('buy', 99.95, 100)
    lob.add_limit_order('buy', 99.90, 200)
    lob.add_limit_order('sell', 100.05, 150)
    lob.add_limit_order('sell', 100.10, 100)
    print(lob)
    print(f"Imbalance: {lob.imbalance():.2f}")
    
    # Execute market order
    qty, price = lob.submit_market_order('buy', 175)
    print(f"Executed buy: {qty} @ {price:.2f}")
    print(lob)
```

---

## 📊 MESURES DE QUALITÉ DE MARCHÉ

### Liquidité
- **Spread relatif**: (Ask - Bid) / Mid
- **Profondeur**: Volume aux meilleurs prix
- **Résilience**: Vitesse de retour à l'équilibre

### Impact de Prix
```python
def estimate_market_impact(trades, prices, window=10):
    """
    Estimate temporary and permanent market impact
    """
    impacts = []
    for i, trade in enumerate(trades):
        if i < window or i + window >= len(prices):
            continue
        
        # Price before trade
        price_before = prices[i-1]
        
        # Price immediately after
        price_after = prices[i]
        
        # Price after decay period
        price_later = prices[i + window]
        
        # Temporary impact (reverts)
        temp_impact = (price_after - price_before) - (price_later - price_before)
        
        # Permanent impact
        perm_impact = price_later - price_before
        
        impacts.append({
            'temporary': temp_impact / trade['signed_volume'],
            'permanent': perm_impact / trade['signed_volume']
        })
    
    return impacts
```

---

## 🔗 LIENS AVEC AUTRES MODULES
- **→ Module 5 (Execution)**: Le modèle de LOB informe l'impact de marché
- **→ Module 6 (RL)**: Market making comme problème RL
- **→ Module 2 (ML)**: Prédiction de mouvement basée sur LOB features

---

## 🔗 RÉFÉRENCES
1. Lehalle, C.-A. & Laruelle, S. - Market Microstructure in Practice
2. Kyle, A.S. (1985) - Continuous Auctions and Insider Trading
3. Avellaneda, M. & Stoikov, S. (2008) - High-Frequency Trading in a Limit Order Book
4. Cont, R., Stoikov, S. & Talreja, R. (2010) - A Stochastic Model for Order Book Dynamics
