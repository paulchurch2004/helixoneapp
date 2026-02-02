# Guide Complet : PyLOB - Limit Order Book

## Simulation d'un Carnet d'Ordres en Python

*Basé sur le projet PyLOB de Ash Booth & Alex Bodnaru*

---

# Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Glossaire des Acronymes](#2-glossaire-des-acronymes)
3. [Architecture du LOB](#3-architecture-du-lob)
4. [Types d'Ordres](#4-types-dordres)
5. [Mécanisme de Matching](#5-mécanisme-de-matching)
6. [Structure de la Base de Données](#6-structure-de-la-base-de-données)
7. [Implémentation Python](#7-implémentation-python)
8. [Exemples d'Utilisation](#8-exemples-dutilisation)
9. [Visualisation du Carnet d'Ordres](#9-visualisation-du-carnet-dordres)
10. [Stratégies de Trading Automatisé](#10-stratégies-de-trading-automatisé)

---

# 1. Introduction et Contexte

## 1.1 Qu'est-ce qu'un LOB (Limit Order Book, Carnet d'Ordres à Cours Limité) ?

Un **LOB (Limit Order Book, Carnet d'Ordres)** est le mécanisme central utilisé par la plupart des bourses financières modernes pour faire correspondre les ordres d'achat et de vente.

**Principe fondamental** : Le LOB maintient deux listes triées :
- **Bids (Offres d'achat)** : Ordres d'acheteurs, triés par prix décroissant
- **Asks (Offres de vente)** : Ordres de vendeurs, triés par prix croissant

```
         CARNET D'ORDRES (LOB)
         
    ASKS (Ventes)          BIDS (Achats)
    ============           ============
    103.00 x 5             99.00 x 10
    102.00 x 8             98.00 x 15
    101.00 x 12   <--->    97.00 x 20
    ↑ Meilleure Ask        ↑ Meilleur Bid
    (Best Ask)             (Best Bid)
    
    Spread = 101.00 - 99.00 = 2.00
```

## 1.2 Données de Niveau 2 (Level 2 Data)

Les données **Level 2** (ou Market Depth) fournissent la vue complète du carnet d'ordres, contrairement aux données Level 1 qui ne montrent que le meilleur bid/ask.

| Niveau | Informations |
|--------|--------------|
| **Level 1** | Meilleur bid, meilleur ask, dernier prix |
| **Level 2** | Tous les ordres à tous les niveaux de prix |
| **Level 3** | Level 2 + identification des participants |

## 1.3 Priorité Prix-Temps

PyLOB implémente la règle **Price-Time Priority** (Priorité Prix-Temps) :

1. **Priorité de prix** : Les meilleurs prix sont servis en premier
   - Pour les bids : prix le plus élevé d'abord
   - Pour les asks : prix le plus bas d'abord

2. **Priorité de temps** : À prix égal, le premier arrivé est servi en premier (FIFO - First In, First Out)

**Exemple** :
```
Deux ordres d'achat à 100€ :
- Ordre A : 100€ x 10 unités (arrivé à 09:00:01)
- Ordre B : 100€ x 5 unités (arrivé à 09:00:02)

Si un vendeur vend 8 unités à 100€ :
→ L'ordre A est exécuté en totalité (10 unités, mais seulement 8 dispo)
→ L'ordre A reçoit 8 unités
→ L'ordre B attend (n'est pas touché car A a la priorité temporelle)
```

## 1.4 Hypothèses Simplificatrices

PyLOB fait plusieurs hypothèses :

1. **Latence nulle** : Un ordre est traité instantanément
2. **Mono-thread** : Pas de gestion de concurrence
3. **Pas de frais de transaction** (configurables)
4. **Pas de restrictions de position**

---

# 2. Glossaire des Acronymes

| Acronyme | Anglais | Français |
|----------|---------|----------|
| **LOB** | Limit Order Book | Carnet d'Ordres à Cours Limité |
| **BBO** | Best Bid and Offer | Meilleure Offre d'Achat et de Vente |
| **FIFO** | First In, First Out | Premier Entré, Premier Sorti |
| **MO** | Market Order | Ordre au Marché |
| **LO** | Limit Order | Ordre à Cours Limité |
| **GTC** | Good Till Cancelled | Valable Jusqu'à Annulation |
| **IOC** | Immediate Or Cancel | Immédiat ou Annulé |
| **FOK** | Fill Or Kill | Exécuté ou Annulé |
| **HFT** | High-Frequency Trading | Trading Haute Fréquence |
| **MM** | Market Maker | Teneur de Marché |
| **VWAP** | Volume Weighted Average Price | Prix Moyen Pondéré par le Volume |
| **TWAP** | Time Weighted Average Price | Prix Moyen Pondéré par le Temps |
| **PnL** | Profit and Loss | Profits et Pertes |
| **OMS** | Order Management System | Système de Gestion des Ordres |
| **FIX** | Financial Information eXchange | Protocole d'Échange d'Informations Financières |
| **NBBO** | National Best Bid and Offer | Meilleure Offre Nationale |
| **TOB** | Top of Book | Sommet du Carnet |
| **DOB** | Depth of Book | Profondeur du Carnet |

---

# 3. Architecture du LOB

## 3.1 Structure Conceptuelle

```
                    ┌─────────────────────────────────────┐
                    │         LIMIT ORDER BOOK            │
                    │              (LOB)                  │
                    └─────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │   BID BOOK    │       │   ASK BOOK    │       │    TRADES     │
    │  (Acheteurs)  │       │  (Vendeurs)   │       │  (Historique) │
    └───────────────┘       └───────────────┘       └───────────────┘
            │                       │                       │
            │    ┌─────────────────┐│                       │
            └───►│ MATCHING ENGINE │◄───────────────────────┘
                 │ (Moteur de      │
                 │  Correspondance)│
                 └─────────────────┘
```

## 3.2 Composants Principaux

```python
"""
=============================================================================
ARCHITECTURE DU LOB - COMPOSANTS PRINCIPAUX
=============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
import sqlite3
import math


class OrderSide(Enum):
    """
    Côté de l'ordre (Side).
    
    BID = Achat : Le trader veut ACHETER l'instrument
    ASK = Vente : Le trader veut VENDRE l'instrument
    """
    BID = "bid"  # Achat
    ASK = "ask"  # Vente


class OrderType(Enum):
    """
    Type d'ordre.
    
    LIMIT : Ordre à cours limité - exécuté seulement au prix spécifié ou mieux
    MARKET : Ordre au marché - exécuté immédiatement au meilleur prix disponible
    """
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class Order:
    """
    Représente un ordre dans le LOB.
    
    Attributes:
        order_id: Identifiant unique interne
        idNum: Identifiant externe (fourni par le trader)
        instrument: Symbole de l'instrument (ex: "AAPL", "EURUSD")
        side: Côté de l'ordre (BID ou ASK)
        order_type: Type d'ordre (LIMIT ou MARKET)
        price: Prix limite (None pour les ordres au marché)
        qty: Quantité demandée
        fulfilled: Quantité déjà exécutée
        timestamp: Horodatage de l'ordre
        trader_id: Identifiant du trader
        active: Si l'ordre est actif
        cancelled: Si l'ordre a été annulé
    
    Exemple:
        >>> order = Order(
        ...     order_id=1,
        ...     idNum=1001,
        ...     instrument="AAPL",
        ...     side=OrderSide.BID,
        ...     order_type=OrderType.LIMIT,
        ...     price=150.00,
        ...     qty=100,
        ...     trader_id=42
        ... )
        >>> print(f"Ordre d'achat: {order.qty} AAPL @ {order.price}$")
        Ordre d'achat: 100 AAPL @ 150.0$
    """
    order_id: int
    idNum: int
    instrument: str
    side: OrderSide
    order_type: OrderType
    price: Optional[float]
    qty: int
    fulfilled: int = 0
    timestamp: int = 0
    trader_id: int = 0
    active: bool = True
    cancelled: bool = False
    
    @property
    def remaining(self) -> int:
        """Quantité restante à exécuter."""
        return self.qty - self.fulfilled
    
    @property
    def is_filled(self) -> bool:
        """L'ordre est-il complètement exécuté ?"""
        return self.fulfilled >= self.qty
    
    def __repr__(self):
        status = "FILLED" if self.is_filled else f"{self.remaining} restants"
        price_str = f"@ {self.price}" if self.price else "MARKET"
        return f"Order({self.side.value} {self.qty} {self.instrument} {price_str}, {status})"


@dataclass
class Trade:
    """
    Représente une transaction exécutée.
    
    Un trade est créé lorsqu'un ordre entrant est apparié (matched)
    avec un ordre existant dans le carnet.
    
    Attributes:
        trade_id: Identifiant unique du trade
        bid_order_id: ID de l'ordre d'achat
        ask_order_id: ID de l'ordre de vente
        price: Prix d'exécution
        qty: Quantité échangée
        timestamp: Horodatage de l'exécution
    
    Exemple:
        >>> trade = Trade(
        ...     trade_id=1,
        ...     bid_order_id=101,
        ...     ask_order_id=201,
        ...     price=150.50,
        ...     qty=50,
        ...     timestamp=1234567890
        ... )
        >>> print(f"Trade: {trade.qty} @ {trade.price}")
        Trade: 50 @ 150.5
    """
    trade_id: int
    bid_order_id: int
    ask_order_id: int
    price: float
    qty: int
    timestamp: int
    
    def __repr__(self):
        return f"Trade(#{self.trade_id}: {self.qty} @ {self.price})"


@dataclass
class PriceLevel:
    """
    Représente un niveau de prix dans le carnet.
    
    Un niveau de prix agrège tous les ordres au même prix.
    
    Attributes:
        price: Le prix du niveau
        orders: Liste des ordres à ce prix (triés par temps)
        total_volume: Volume total à ce niveau
    
    Exemple:
        Pour le niveau de prix 100.00 :
        - Ordre 1: 50 unités (09:00:01)
        - Ordre 2: 30 unités (09:00:05)
        - Ordre 3: 20 unités (09:00:10)
        → Total: 100 unités à 100.00
    """
    price: float
    orders: List[Order] = field(default_factory=list)
    
    @property
    def total_volume(self) -> int:
        """Volume total disponible à ce niveau de prix."""
        return sum(order.remaining for order in self.orders if order.active)
    
    @property
    def order_count(self) -> int:
        """Nombre d'ordres actifs à ce niveau."""
        return len([o for o in self.orders if o.active])
```

## 3.3 Métriques du Carnet

```python
@dataclass
class OrderBookMetrics:
    """
    Métriques calculées à partir du carnet d'ordres.
    
    Ces métriques sont essentielles pour :
    - L'analyse de la liquidité
    - Le calcul du coût d'exécution
    - La détection d'opportunités de trading
    
    Attributes:
        best_bid: Meilleur prix d'achat (le plus élevé)
        best_ask: Meilleur prix de vente (le plus bas)
        spread: Écart entre best_ask et best_bid
        mid_price: Prix milieu = (best_bid + best_ask) / 2
        bid_volume: Volume total des achats
        ask_volume: Volume total des ventes
        imbalance: Déséquilibre = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    """
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float] = None
    spread_bps: Optional[float] = None  # Spread en points de base
    mid_price: Optional[float] = None
    bid_volume: int = 0
    ask_volume: int = 0
    imbalance: float = 0.0
    
    def __post_init__(self):
        if self.best_bid and self.best_ask:
            self.spread = self.best_ask - self.best_bid
            self.mid_price = (self.best_bid + self.best_ask) / 2
            self.spread_bps = (self.spread / self.mid_price) * 10000
        
        total_volume = self.bid_volume + self.ask_volume
        if total_volume > 0:
            self.imbalance = (self.bid_volume - self.ask_volume) / total_volume
    
    def __repr__(self):
        return (f"OrderBookMetrics(\n"
                f"  BBO (Best Bid/Offer): {self.best_bid} / {self.best_ask}\n"
                f"  Spread: {self.spread} ({self.spread_bps:.2f} bps)\n"
                f"  Mid Price: {self.mid_price}\n"
                f"  Volume: {self.bid_volume} (bid) / {self.ask_volume} (ask)\n"
                f"  Imbalance: {self.imbalance:+.2%}\n"
                f")")


def calculate_vwap(trades: List[Trade]) -> float:
    """
    Calcule le VWAP (Volume Weighted Average Price, Prix Moyen Pondéré par le Volume).
    
    VWAP = Σ(Prix × Volume) / Σ(Volume)
    
    Le VWAP est utilisé comme benchmark pour évaluer la qualité d'exécution.
    
    Args:
        trades: Liste des trades
    
    Returns:
        VWAP calculé
    
    Exemple:
        >>> trades = [
        ...     Trade(1, 1, 2, 100.0, 50, 0),   # 50 @ 100
        ...     Trade(2, 3, 4, 101.0, 30, 1),   # 30 @ 101
        ...     Trade(3, 5, 6, 99.5, 20, 2),    # 20 @ 99.5
        ... ]
        >>> vwap = calculate_vwap(trades)
        >>> print(f"VWAP = {vwap:.4f}")
        VWAP = 100.2500
        
        Calcul : (100×50 + 101×30 + 99.5×20) / (50+30+20) = 10025/100 = 100.25
    """
    if not trades:
        return 0.0
    
    total_value = sum(t.price * t.qty for t in trades)
    total_volume = sum(t.qty for t in trades)
    
    return total_value / total_volume if total_volume > 0 else 0.0


def calculate_market_impact(
    order_qty: int,
    side: OrderSide,
    price_levels: List[PriceLevel]
) -> Tuple[float, float]:
    """
    Calcule l'impact de marché d'un ordre.
    
    L'impact de marché mesure combien un ordre déplace le prix
    lors de son exécution.
    
    Args:
        order_qty: Quantité de l'ordre
        side: Côté de l'ordre
        price_levels: Niveaux de prix disponibles (triés)
    
    Returns:
        Tuple (prix moyen d'exécution, impact en %)
    
    Exemple:
        Pour acheter 100 unités avec le carnet suivant :
        - Ask: 101.00 x 30
        - Ask: 101.50 x 40
        - Ask: 102.00 x 50
        
        On achète :
        - 30 @ 101.00
        - 40 @ 101.50
        - 30 @ 102.00
        
        Prix moyen = (30×101 + 40×101.5 + 30×102) / 100 = 101.49
        Impact = (101.49 - 101.00) / 101.00 = 0.49%
    """
    if not price_levels:
        return 0.0, 0.0
    
    remaining = order_qty
    total_cost = 0.0
    initial_price = price_levels[0].price
    
    for level in price_levels:
        if remaining <= 0:
            break
        
        available = level.total_volume
        executed = min(remaining, available)
        total_cost += executed * level.price
        remaining -= executed
    
    if order_qty - remaining > 0:
        avg_price = total_cost / (order_qty - remaining)
        impact = (avg_price - initial_price) / initial_price
        return avg_price, impact * 100
    
    return 0.0, 0.0
```

---

# 4. Types d'Ordres

## 4.1 Ordre à Cours Limité (Limit Order)

Un **ordre limite** spécifie un prix maximal (pour l'achat) ou minimal (pour la vente) auquel le trader est prêt à exécuter.

**Caractéristiques** :
- Garantie de prix (jamais exécuté à un prix défavorable)
- Pas de garantie d'exécution (peut rester dans le carnet)
- Fournit de la liquidité au marché

```python
def create_limit_order(
    instrument: str,
    side: str,
    qty: int,
    price: float,
    trader_id: int
) -> dict:
    """
    Crée un ordre à cours limité (Limit Order).
    
    Un ordre limite est exécuté seulement si le marché atteint
    le prix spécifié ou un prix plus favorable.
    
    Args:
        instrument: Symbole de l'instrument (ex: "AAPL", "EURUSD")
        side: "bid" (achat) ou "ask" (vente)
        qty: Quantité à acheter/vendre
        price: Prix limite
        trader_id: ID du trader
    
    Returns:
        Dictionnaire représentant l'ordre
    
    Exemples:
        >>> # Ordre d'achat limite : Acheter 100 AAPL max à 150$
        >>> buy_order = create_limit_order("AAPL", "bid", 100, 150.00, 1)
        >>> 
        >>> # Ordre de vente limite : Vendre 50 AAPL min à 155$
        >>> sell_order = create_limit_order("AAPL", "ask", 50, 155.00, 2)
        
    Comportement :
        - BID (achat) @ 150$ : Exécuté si prix marché ≤ 150$
        - ASK (vente) @ 155$ : Exécuté si prix marché ≥ 155$
    """
    if side not in ('bid', 'ask'):
        raise ValueError("side doit être 'bid' ou 'ask'")
    if qty <= 0:
        raise ValueError("qty doit être > 0")
    if price <= 0:
        raise ValueError("price doit être > 0 pour un ordre limite")
    
    return {
        'type': 'limit',
        'side': side,
        'instrument': instrument,
        'qty': qty,
        'price': price,
        'tid': trader_id
    }


# Exemples concrets
print("=== ORDRES LIMITE ===\n")

# Scénario : Un trader veut acheter des actions
bid_order = create_limit_order(
    instrument="AAPL",
    side="bid",
    qty=100,
    price=150.00,
    trader_id=1
)
print(f"Ordre d'ACHAT limite:")
print(f"  → Acheter 100 AAPL à maximum 150.00$")
print(f"  → Si le meilleur ask est ≤ 150$, l'ordre est exécuté")
print(f"  → Sinon, l'ordre reste dans le carnet côté BID")

ask_order = create_limit_order(
    instrument="AAPL",
    side="ask",
    qty=50,
    price=155.00,
    trader_id=2
)
print(f"\nOrdre de VENTE limite:")
print(f"  → Vendre 50 AAPL à minimum 155.00$")
print(f"  → Si le meilleur bid est ≥ 155$, l'ordre est exécuté")
print(f"  → Sinon, l'ordre reste dans le carnet côté ASK")
```

## 4.2 Ordre au Marché (Market Order)

Un **ordre au marché** est exécuté immédiatement au meilleur prix disponible.

**Caractéristiques** :
- Garantie d'exécution (si liquidité disponible)
- Pas de garantie de prix
- Consomme de la liquidité

```python
def create_market_order(
    instrument: str,
    side: str,
    qty: int,
    trader_id: int
) -> dict:
    """
    Crée un ordre au marché (Market Order).
    
    Un ordre au marché est exécuté immédiatement au meilleur
    prix disponible dans le carnet.
    
    Args:
        instrument: Symbole de l'instrument
        side: "bid" (achat) ou "ask" (vente)
        qty: Quantité à acheter/vendre
        trader_id: ID du trader
    
    Returns:
        Dictionnaire représentant l'ordre
    
    Exemples:
        >>> # Acheter immédiatement 100 AAPL au prix du marché
        >>> buy_market = create_market_order("AAPL", "bid", 100, 1)
        >>> 
        >>> # Vendre immédiatement 50 AAPL au prix du marché
        >>> sell_market = create_market_order("AAPL", "ask", 50, 2)
    
    Attention :
        L'ordre au marché peut être exécuté à différents prix
        si la quantité dépasse le volume disponible au meilleur prix.
        C'est ce qu'on appelle le "slippage" (dérapage).
    """
    if side not in ('bid', 'ask'):
        raise ValueError("side doit être 'bid' ou 'ask'")
    if qty <= 0:
        raise ValueError("qty doit être > 0")
    
    return {
        'type': 'market',
        'side': side,
        'instrument': instrument,
        'qty': qty,
        'price': None,  # Pas de prix pour un ordre marché
        'tid': trader_id
    }


# Démonstration du slippage
print("\n=== ORDRES AU MARCHÉ ET SLIPPAGE ===\n")

print("Carnet d'ordres initial :")
print("  ASK: 101.00 x 30")
print("  ASK: 101.50 x 40")
print("  ASK: 102.00 x 50")
print()

print("Ordre d'achat marché: 100 unités")
print("Exécution :")
print("  - 30 @ 101.00 = 3030.00$")
print("  - 40 @ 101.50 = 4060.00$")
print("  - 30 @ 102.00 = 3060.00$")
print("  ----------------------------")
print("  Total: 100 unités pour 10150.00$")
print(f"  Prix moyen: {10150/100:.2f}$")
print(f"  Slippage: +{(10150/100 - 101):.2f}$ vs meilleur prix")
```

## 4.3 Comparaison des Types d'Ordres

```python
def compare_order_types():
    """
    Compare les ordres limite et marché.
    """
    print("\n" + "=" * 70)
    print("COMPARAISON : ORDRE LIMITE vs ORDRE AU MARCHÉ")
    print("=" * 70)
    
    comparison = """
    ┌──────────────────┬────────────────────┬────────────────────┐
    │    Critère       │   Ordre LIMITE     │   Ordre MARCHÉ     │
    ├──────────────────┼────────────────────┼────────────────────┤
    │ Prix d'exécution │ Garanti (ou mieux) │ Non garanti        │
    │ Exécution        │ Non garantie       │ Garantie (si liq.) │
    │ Liquidité        │ Fournit            │ Consomme           │
    │ Slippage         │ Aucun              │ Possible           │
    │ Dans le carnet   │ Oui (si non matchÃ©)│ Non (immédiat)     │
    │ Cas d'usage      │ Prix important     │ Rapidité importante│
    └──────────────────┴────────────────────┴────────────────────┘
    
    Quand utiliser un ordre LIMITE ?
    --------------------------------
    • Quand le prix d'exécution est crucial
    • Pour les marchés peu liquides
    • Pour les stratégies de market-making
    • Quand on peut attendre l'exécution
    
    Quand utiliser un ordre au MARCHÉ ?
    -----------------------------------
    • Quand la rapidité d'exécution est cruciale
    • Pour sortir rapidement d'une position
    • Sur des marchés très liquides (faible spread)
    • Pour les petits ordres (slippage négligeable)
    """
    print(comparison)


compare_order_types()
```

---

# 5. Mécanisme de Matching

## 5.1 Algorithme de Correspondance

```python
class MatchingEngine:
    """
    Moteur de correspondance des ordres (Matching Engine).
    
    Le matching engine est le cœur du LOB. Il détermine quels ordres
    peuvent être appariés et génère les trades correspondants.
    
    Règles de correspondance :
    1. Un ordre BID peut matcher avec des ASKs si bid_price >= ask_price
    2. Un ordre ASK peut matcher avec des BIDs si ask_price <= bid_price
    3. Le prix d'exécution est celui de l'ordre déjà dans le carnet
    4. Priorité : Prix > Temps (FIFO à prix égal)
    """
    
    def __init__(self, tick_size: float = 0.01):
        """
        Initialise le moteur de matching.
        
        Args:
            tick_size: Taille minimale de variation du prix (tick)
                      Ex: 0.01 pour 1 centime
        """
        self.tick_size = tick_size
        self.rounder = int(math.log10(1 / tick_size))
    
    def clip_price(self, price: float) -> float:
        """
        Arrondit le prix au tick le plus proche.
        
        Exemple avec tick_size = 0.01 :
            100.123 → 100.12
            100.126 → 100.13
        """
        return round(price, self.rounder)
    
    def can_match(
        self, 
        incoming_order: dict, 
        book_order: dict
    ) -> bool:
        """
        Vérifie si deux ordres peuvent être appariés.
        
        Args:
            incoming_order: Ordre entrant
            book_order: Ordre dans le carnet
        
        Returns:
            True si les ordres peuvent matcher
        
        Règles :
        - BID entrant vs ASK dans le carnet : bid_price >= ask_price
        - ASK entrant vs BID dans le carnet : ask_price <= bid_price
        - Ordres au marché : matchent toujours (pas de contrainte de prix)
        
        Exemple:
            BID @ 100 vs ASK @ 99 → Match (acheteur paie 99)
            BID @ 100 vs ASK @ 101 → Pas de match
        """
        incoming_price = incoming_order.get('price')
        book_price = book_order.get('price')
        incoming_side = incoming_order['side']
        
        # Ordres au marché matchent toujours
        if incoming_price is None or book_price is None:
            return True
        
        if incoming_side == 'bid':
            # Acheteur prêt à payer au moins ask_price
            return incoming_price >= book_price
        else:
            # Vendeur accepte au moins bid_price
            return incoming_price <= book_price
    
    def determine_trade_price(
        self, 
        incoming_order: dict, 
        book_order: dict
    ) -> float:
        """
        Détermine le prix d'exécution du trade.
        
        Règle : Le prix est celui de l'ordre PASSIF (dans le carnet).
        L'ordre passif a la priorité car il était là en premier.
        
        Args:
            incoming_order: Ordre entrant (agresseur)
            book_order: Ordre dans le carnet (passif)
        
        Returns:
            Prix d'exécution
        
        Exemple:
            - Ordre dans le carnet : ASK @ 100
            - Ordre entrant : BID @ 102
            - Prix d'exécution : 100 (prix du passif)
            - L'acheteur obtient un meilleur prix que demandé !
        """
        # Le prix passif a priorité
        if book_order.get('price') is not None:
            return book_order['price']
        elif incoming_order.get('price') is not None:
            return incoming_order['price']
        else:
            # Les deux sont des ordres marché - utiliser last price
            return None  # Nécessite un last price
    
    def match_orders(
        self, 
        incoming_order: dict,
        book_orders: list,
        verbose: bool = False
    ) -> Tuple[list, int]:
        """
        Exécute le matching d'un ordre entrant.
        
        Args:
            incoming_order: Ordre à matcher
            book_orders: Ordres disponibles dans le carnet (triés par priorité)
            verbose: Afficher les détails
        
        Returns:
            Tuple (liste des trades, quantité restante non exécutée)
        
        Exemple:
            >>> engine = MatchingEngine()
            >>> incoming = {'side': 'bid', 'price': 102, 'qty': 100}
            >>> book = [
            ...     {'side': 'ask', 'price': 100, 'qty': 30, 'order_id': 1},
            ...     {'side': 'ask', 'price': 101, 'qty': 50, 'order_id': 2},
            ... ]
            >>> trades, remaining = engine.match_orders(incoming, book)
            >>> # trades: [Trade(30@100), Trade(50@101)]
            >>> # remaining: 20 (100 - 30 - 50)
        """
        trades = []
        remaining_qty = incoming_order['qty']
        
        for book_order in book_orders:
            if remaining_qty <= 0:
                break
            
            if not self.can_match(incoming_order, book_order):
                continue
            
            # Quantité disponible dans l'ordre du carnet
            available = book_order['qty'] - book_order.get('fulfilled', 0)
            
            # Quantité à exécuter
            exec_qty = min(remaining_qty, available)
            
            # Prix d'exécution
            trade_price = self.determine_trade_price(incoming_order, book_order)
            
            # Créer le trade
            trade = {
                'bid_order_id': incoming_order.get('order_id') if incoming_order['side'] == 'bid' else book_order['order_id'],
                'ask_order_id': incoming_order.get('order_id') if incoming_order['side'] == 'ask' else book_order['order_id'],
                'price': trade_price,
                'qty': exec_qty
            }
            trades.append(trade)
            
            remaining_qty -= exec_qty
            
            if verbose:
                print(f">>> TRADE: {exec_qty} @ {trade_price}")
        
        return trades, remaining_qty


# Démonstration du matching
print("\n=== DÉMONSTRATION DU MATCHING ===\n")

engine = MatchingEngine(tick_size=0.01)

# Carnet simulé (côté ASK)
book_asks = [
    {'side': 'ask', 'price': 100.00, 'qty': 30, 'fulfilled': 0, 'order_id': 1},
    {'side': 'ask', 'price': 100.50, 'qty': 40, 'fulfilled': 0, 'order_id': 2},
    {'side': 'ask', 'price': 101.00, 'qty': 50, 'fulfilled': 0, 'order_id': 3},
]

# Ordre d'achat entrant
incoming_bid = {
    'side': 'bid',
    'price': 100.75,
    'qty': 60,
    'order_id': 100
}

print("Carnet d'ordres (ASK) :")
for o in book_asks:
    print(f"  {o['qty']} @ {o['price']}")

print(f"\nOrdre entrant : BID {incoming_bid['qty']} @ {incoming_bid['price']}")
print()

trades, remaining = engine.match_orders(incoming_bid, book_asks, verbose=True)

print(f"\nRésultat :")
print(f"  Trades exécutés : {len(trades)}")
for t in trades:
    print(f"    - {t['qty']} @ {t['price']}")
print(f"  Quantité restante : {remaining}")
if remaining > 0:
    print(f"  → {remaining} unités placées dans le carnet à {incoming_bid['price']}")
```

## 5.2 Requêtes SQL de Matching

PyLOB utilise SQLite pour le stockage et le matching. Voici les requêtes clés :

```python
"""
REQUÊTES SQL DE MATCHING - PyLOB
================================
"""

# Requête pour trouver les ordres correspondants
MATCHES_SQL = """
SELECT 
    order_id, 
    trader AS counterparty, 
    COALESCE(price, :price, :lastprice) AS price, 
    available
FROM best_quotes 
WHERE 
    instrument = :instrument AND 
    matching = :side AND
    (allow_self_matching = 1 OR trader <> :tid) AND 
    COALESCE(:price, price, :lastprice) IS NOT NULL AND 
    (:price IS NULL OR price IS NULL OR price * matching_order <= :price * matching_order)
-- ORDER BY clause appended
"""

# Explication de la requête :
# 1. best_quotes : Vue des ordres actifs avec leur volume disponible
# 2. matching = :side : Trouve le côté opposé (bid cherche ask et vice versa)
# 3. allow_self_matching : Empêche un trader de matcher avec lui-même
# 4. La condition de prix vérifie que les prix sont compatibles
#    - Pour un BID : bid_price >= ask_price
#    - Pour un ASK : ask_price <= bid_price


# Vue des meilleures cotations
BEST_QUOTES_VIEW = """
CREATE VIEW IF NOT EXISTS best_quotes AS
SELECT 
    order_id, 
    idNum, 
    side.side AS side, 
    price, 
    qty, 
    fulfilled, 
    qty - fulfilled AS available,  -- Volume disponible
    event_dt, 
    instrument, 
    trade_order.trader, 
    allow_self_matching, 
    matching, 
    matching_order
FROM trade_order
INNER JOIN trader ON trader.tid = trade_order.trader
INNER JOIN side ON side.side = trade_order.side
WHERE 
    active = 1 AND      -- Ordre actif
    cancel = 0 AND      -- Non annulé
    qty > fulfilled     -- Il reste du volume
ORDER BY 
    side ASC,
    -- Prix NULL (ordres marché) ont la meilleure priorité
    CASE WHEN price IS NULL THEN 0 ELSE 1 END ASC,
    -- Tri par prix selon le côté
    matching_order * COALESCE(price, 0) ASC,
    -- FIFO à prix égal
    event_dt ASC
;
"""

# Insertion d'un trade
INSERT_TRADE_SQL = """
INSERT INTO trade (
    bid_order,
    ask_order,
    event_dt,
    price,
    qty
)
VALUES (?, ?, ?, ?, ?)
"""

# Le trigger SQL met automatiquement à jour :
# 1. Le fulfilled des deux ordres
# 2. Les balances des deux traders
# 3. Les commissions
```

---

# 6. Structure de la Base de Données

## 6.1 Schéma Relationnel

```python
"""
SCHÉMA DE LA BASE DE DONNÉES PyLOB
==================================
"""

DATABASE_SCHEMA = """
┌─────────────────────────────────────────────────────────────────────┐
│                         SCHÉMA LOB                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     TRADER      │     │   INSTRUMENT    │     │      SIDE       │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ tid (PK)        │     │ symbol (PK)     │     │ side (PK)       │
│ name            │     │ currency        │     │ matching        │
│ currency (FK)   │────→│ lastprice       │     │ matching_order  │
│ commission_*    │     └─────────────────┘     └────────┬────────┘
│ allow_self_match│             │                        │
└────────┬────────┘             │                        │
         │                      │                        │
         │    ┌─────────────────┴────────────────────────┴─────┐
         │    │                                                │
         │    ▼                                                ▼
         │  ┌──────────────────────────────────────────────────┐
         └─→│               TRADE_ORDER                        │
            ├──────────────────────────────────────────────────┤
            │ order_id (PK)                                    │
            │ instrument (FK) ─────→ instrument.symbol         │
            │ order_type ('limit' | 'market')                  │
            │ side (FK) ───────────→ side.side                 │
            │ event_dt (timestamp)                             │
            │ qty (quantité demandée)                          │
            │ fulfilled (quantité exécutée)                    │
            │ price (prix limite, NULL pour market)            │
            │ idNum (ID externe)                               │
            │ trader (FK) ─────────→ trader.tid                │
            │ active (1=actif)                                 │
            │ cancel (1=annulé)                                │
            │ fulfill_price (prix moyen d'exécution)           │
            │ commission (frais calculés)                      │
            └───────────────────────────┬──────────────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
                        ▼                               ▼
            ┌───────────────────────────────────────────────────┐
            │                      TRADE                        │
            ├───────────────────────────────────────────────────┤
            │ trade_id (PK)                                     │
            │ bid_order (FK) ──────→ trade_order.order_id       │
            │ ask_order (FK) ──────→ trade_order.order_id       │
            │ event_dt (timestamp)                              │
            │ price (prix d'exécution)                          │
            │ qty (quantité échangée)                           │
            │ idNum (ID externe)                                │
            └───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                        TRADER_BALANCE                                 │
├───────────────────────────────────────────────────────────────────────┤
│ trader (FK) ────────→ trader.tid                                      │
│ instrument (FK) ────→ instrument.symbol                               │
│ amount (solde)                                                        │
│ PRIMARY KEY (trader, instrument)                                      │
└───────────────────────────────────────────────────────────────────────┘
"""

print(DATABASE_SCHEMA)
```

## 6.2 Création de la Base

```python
import sqlite3
from pathlib import Path

def create_lob_database(db_path: str = ":memory:") -> sqlite3.Connection:
    """
    Crée la base de données LOB avec tout le schéma nécessaire.
    
    Args:
        db_path: Chemin vers le fichier SQLite (":memory:" pour en mémoire)
    
    Returns:
        Connexion SQLite
    
    Exemple:
        >>> conn = create_lob_database("lob.db")
        >>> # La base est prête à être utilisée
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Accès par nom de colonne
    
    # Script de création
    create_script = """
    -- Activer les clés étrangères
    PRAGMA foreign_keys = ON;
    
    BEGIN TRANSACTION;
    
    -- Table des traders
    CREATE TABLE IF NOT EXISTS trader (
        tid INTEGER NOT NULL PRIMARY KEY,
        name TEXT,
        currency TEXT,
        commission_per_unit REAL DEFAULT(0),
        commission_min REAL DEFAULT(0),
        commission_max_percnt REAL DEFAULT(0),
        allow_self_matching INTEGER DEFAULT(0),
        FOREIGN KEY(currency) REFERENCES instrument(symbol)
    );
    
    -- Table des instruments
    CREATE TABLE IF NOT EXISTS instrument (
        symbol TEXT UNIQUE,
        currency TEXT,
        lastprice REAL
    );
    
    -- Insérer la devise de base
    INSERT INTO instrument (symbol, currency) 
    VALUES ('USD', NULL) 
    ON CONFLICT DO NOTHING;
    
    -- Table des balances
    CREATE TABLE IF NOT EXISTS trader_balance (
        trader INTEGER,
        instrument TEXT,
        amount REAL DEFAULT(0),
        PRIMARY KEY(trader, instrument),
        FOREIGN KEY(trader) REFERENCES trader(tid),
        FOREIGN KEY(instrument) REFERENCES instrument(symbol)
    );
    
    -- Table des côtés (bid/ask)
    CREATE TABLE IF NOT EXISTS side (
        side TEXT PRIMARY KEY,
        matching TEXT,
        matching_order INTEGER
    );
    
    INSERT INTO side (side, matching, matching_order) 
    VALUES 
        ('bid', 'ask', -1),
        ('ask', 'bid', 1)
    ON CONFLICT DO NOTHING;
    
    -- Table des ordres
    CREATE TABLE IF NOT EXISTS trade_order (
        order_id INTEGER PRIMARY KEY,
        instrument TEXT,
        order_type TEXT,
        side TEXT,
        event_dt INTEGER,
        qty INTEGER NOT NULL,
        fulfilled INTEGER DEFAULT(0),
        price REAL,
        idNum INTEGER,
        trader INTEGER,
        active INTEGER DEFAULT(1),
        cancel INTEGER DEFAULT(0),
        fulfill_price REAL DEFAULT(0),
        commission REAL NOT NULL DEFAULT(0),
        FOREIGN KEY(side) REFERENCES side(side),
        FOREIGN KEY(trader) REFERENCES trader(tid),
        FOREIGN KEY(instrument) REFERENCES instrument(symbol)
    );
    
    -- Index pour la performance
    CREATE INDEX IF NOT EXISTS order_priority 
        ON trade_order (side, instrument, price ASC, event_dt ASC);
    CREATE INDEX IF NOT EXISTS order_idnum 
        ON trade_order (idNum ASC);
    
    -- Table des trades
    CREATE TABLE IF NOT EXISTS trade (
        trade_id INTEGER PRIMARY KEY, 
        bid_order INTEGER,
        ask_order INTEGER,
        event_dt TEXT DEFAULT(datetime('now')),
        price REAL,
        qty INTEGER,
        idNum INTEGER,
        FOREIGN KEY(bid_order) REFERENCES trade_order(order_id),
        FOREIGN KEY(ask_order) REFERENCES trade_order(order_id)
    );
    
    COMMIT;
    """
    
    conn.executescript(create_script)
    return conn


def setup_test_data(conn: sqlite3.Connection, instrument: str = "AAPL"):
    """
    Configure des données de test.
    
    Args:
        conn: Connexion à la base
        instrument: Symbole de l'instrument
    """
    setup_script = f"""
    BEGIN TRANSACTION;
    
    -- Créer des traders de test
    INSERT INTO trader (tid, name, commission_min, commission_max_percnt, commission_per_unit) 
    VALUES 
        (1, 'Alice', 1.0, 0.1, 0.01),
        (2, 'Bob', 1.0, 0.1, 0.01),
        (3, 'Charlie', 1.0, 0.1, 0.01)
    ON CONFLICT DO NOTHING;
    
    -- Créer l'instrument
    INSERT INTO instrument (symbol, currency) 
    VALUES ('{instrument}', 'USD')
    ON CONFLICT DO NOTHING;
    
    COMMIT;
    """
    conn.executescript(setup_script)
```

---

# 7. Implémentation Python

## 7.1 Classe OrderBook Complète

```python
"""
=============================================================================
IMPLÉMENTATION COMPLÈTE DU LOB (LIMIT ORDER BOOK)
=============================================================================
"""

import sqlite3
import math
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


class OrderBook:
    """
    Carnet d'ordres à cours limité (Limit Order Book).
    
    Cette classe implémente un LOB complet avec :
    - Support des ordres limite et marché
    - Matching avec priorité prix-temps
    - Annulation et modification d'ordres
    - Calcul du volume à chaque niveau de prix
    - Stockage persistant via SQLite
    
    Attributes:
        db: Connexion SQLite
        tick_size: Taille minimale de variation du prix
        time: Horodatage interne (incrémental)
        lastPrice: Dernier prix de transaction par instrument
    
    Exemple d'utilisation:
        >>> conn = sqlite3.connect("lob.db")
        >>> lob = OrderBook(db=conn, tick_size=0.01)
        >>> 
        >>> # Soumettre un ordre limite
        >>> order = {
        ...     'type': 'limit',
        ...     'side': 'bid',
        ...     'instrument': 'AAPL',
        ...     'qty': 100,
        ...     'price': 150.00,
        ...     'tid': 1
        ... }
        >>> trades, order_info = lob.processOrder(order, fromData=False, verbose=True)
    """
    
    # Types d'ordres valides
    VALID_TYPES = ('market', 'limit')
    # Côtés valides
    VALID_SIDES = ('ask', 'bid')
    
    def __init__(self, db: sqlite3.Connection, tick_size: float = 0.0001):
        """
        Initialise le carnet d'ordres.
        
        Args:
            db: Connexion à la base de données SQLite
            tick_size: Taille minimale de variation du prix (tick)
                      Ex: 0.01 pour les actions, 0.0001 pour le forex
        
        Exemple:
            >>> conn = sqlite3.connect(":memory:")
            >>> # Créer le schéma...
            >>> lob = OrderBook(db=conn, tick_size=0.01)
        """
        self.db = db
        self.tickSize = tick_size
        self.rounder = int(math.log10(1 / self.tickSize))
        self.time = 0
        self.nextQuoteID = 0
        self.lastPrice: Dict[str, float] = {}
        self.lastTimestamp = 0
        
        # Charger les requêtes SQL
        self._load_sql_queries()
    
    def _load_sql_queries(self):
        """Charge les requêtes SQL depuis les fichiers ou définit en dur."""
        
        self.insert_order = """
        INSERT INTO trade_order (
            instrument, order_type, side, event_dt, qty, price, idNum, trader
        ) VALUES (
            :instrument, :type, :side, :timestamp, :qty, :price, :idNum, :tid
        )
        """
        
        self.insert_trade = """
        INSERT INTO trade (bid_order, ask_order, event_dt, price, qty)
        VALUES (?, ?, ?, ?, ?)
        """
        
        self.cancel_order = """
        UPDATE trade_order 
        SET cancel = :cancel
        WHERE idNum = :idNum AND side = :side
        """
        
        self.find_order = """
        SELECT side, instrument, price, qty, fulfilled, cancel, order_id, order_type 
        FROM trade_order 
        WHERE idNum = ?
        """
        
        self.modify_order = """
        UPDATE trade_order 
        SET 
            price = CASE WHEN order_type = 'market' 
                THEN NULL 
                ELSE CASE WHEN :price IS NULL THEN price ELSE :price END
            END,
            qty = (CASE WHEN :qty < fulfilled THEN fulfilled ELSE :qty END),
            event_dt = (CASE WHEN :qty > qty THEN :timestamp ELSE event_dt END)
        WHERE idNum = :idNum AND cancel = 0 AND side = :side
        """
        
        self.active_orders = """
        SELECT idNum, qty, fulfilled, price, event_dt, instrument
        FROM best_quotes
        WHERE side = :side AND instrument = :instrument
        """
        
        self.matches = """
        SELECT 
            order_id, trader AS counterparty, 
            COALESCE(price, :price, :lastprice) AS price, 
            available
        FROM best_quotes 
        WHERE 
            instrument = :instrument AND 
            matching = :side AND
            (allow_self_matching = 1 OR trader <> :tid) AND 
            COALESCE(:price, price, :lastprice) IS NOT NULL AND 
            (:price IS NULL OR price IS NULL OR price * matching_order <= :price * matching_order)
        """
        
        self.best_quotes_order_asc = """
        ORDER BY 
            instrument ASC, side ASC,
            CASE WHEN price IS NULL THEN 0 ELSE 1 END ASC,
            matching_order * COALESCE(price, 0) ASC,
            event_dt ASC
        """
        
        self.best_quotes_order_desc = self.best_quotes_order_asc.replace('ASC', 'DESC')
        
        self.volume_at_price = """
        SELECT COALESCE(SUM(available), 0) AS volume
        FROM best_quotes 
        WHERE 
            instrument = :instrument AND 
            side = :side AND
            (price IS NULL OR price * matching_order <= :price * matching_order)
        """
        
        self.select_trades = """
        SELECT * FROM trade
        """
        
        self.set_lastprice = """
        UPDATE instrument SET lastprice = :lastprice WHERE symbol = :instrument
        """
    
    def clipPrice(self, price: float) -> float:
        """
        Arrondit le prix au tick le plus proche.
        
        Args:
            price: Prix à arrondir
        
        Returns:
            Prix arrondi
        
        Exemple:
            >>> lob = OrderBook(conn, tick_size=0.01)
            >>> lob.clipPrice(100.123)
            100.12
            >>> lob.clipPrice(100.127)
            100.13
        """
        return round(price, self.rounder)
    
    def updateTime(self):
        """Incrémente l'horodatage interne."""
        self.time += 1
    
    def processOrder(
        self, 
        quote: dict, 
        fromData: bool = False, 
        verbose: bool = False
    ) -> Tuple[List[dict], dict]:
        """
        Traite un nouvel ordre.
        
        C'est la méthode principale pour soumettre un ordre au LOB.
        L'ordre est d'abord validé, puis matché contre les ordres existants.
        
        Args:
            quote: Dictionnaire contenant les détails de l'ordre
                   - type: 'limit' ou 'market'
                   - side: 'bid' ou 'ask'
                   - instrument: Symbole (ex: 'AAPL')
                   - qty: Quantité
                   - price: Prix (obligatoire pour limit, ignoré pour market)
                   - tid: ID du trader
            fromData: True si l'ordre vient de données historiques
            verbose: Afficher les détails des trades
        
        Returns:
            Tuple (liste des trades exécutés, info de l'ordre)
        
        Exemple:
            >>> order = {
            ...     'type': 'limit',
            ...     'side': 'bid',
            ...     'instrument': 'AAPL',
            ...     'qty': 100,
            ...     'price': 150.00,
            ...     'tid': 1
            ... }
            >>> trades, order_info = lob.processOrder(order, verbose=True)
            >>> if trades:
            ...     print(f"Exécuté: {len(trades)} trades")
            >>> else:
            ...     print(f"Ordre placé dans le carnet")
        """
        # Gestion du timestamp
        if fromData:
            self.time = quote['timestamp']
        else:
            self.updateTime()
            quote['timestamp'] = self.time
            self.nextQuoteID += 1
            quote['idNum'] = self.nextQuoteID
        
        # Validation
        if quote['qty'] <= 0:
            raise ValueError("processOrder() : qty doit être > 0")
        
        if quote['type'] not in self.VALID_TYPES:
            raise ValueError(f"processOrder() : type doit être dans {self.VALID_TYPES}")
        
        if quote['side'] not in self.VALID_SIDES:
            raise ValueError(f"processOrder() : side doit être dans {self.VALID_SIDES}")
        
        # Arrondir le prix si présent
        if quote.get('price'):
            quote['price'] = self.clipPrice(quote['price'])
        else:
            quote['price'] = None
        
        # Transaction DB
        crsr = self.db.cursor()
        crsr.execute('BEGIN TRANSACTION')
        
        # Insérer l'ordre
        crsr.execute(self.insert_order, quote)
        quote['order_id'] = crsr.lastrowid
        
        # Matching
        ret = self._processMatchesDB(quote, crsr, verbose)
        
        crsr.execute('COMMIT')
        
        return ret
    
    def _processMatchesDB(
        self, 
        quote: dict, 
        crsr: sqlite3.Cursor, 
        verbose: bool
    ) -> Tuple[List[dict], dict]:
        """
        Exécute le matching avec les ordres dans le carnet.
        
        Args:
            quote: Ordre à matcher
            crsr: Curseur de la transaction en cours
            verbose: Afficher les détails
        
        Returns:
            Tuple (liste des trades, info de l'ordre)
        """
        instrument = quote['instrument']
        quote['lastprice'] = self.lastPrice.get(instrument)
        
        qtyToExec = quote['qty']
        
        # Requête de matching
        sql_matches = self.matches + self.best_quotes_order_asc
        matches = crsr.execute(sql_matches, quote).fetchall()
        
        trades = []
        price = None
        
        for match in matches:
            if qtyToExec <= 0:
                break
            
            order_id, counterparty, price, available = match
            
            # Déterminer les IDs bid/ask
            bid_order = quote['order_id'] if quote['side'] == 'bid' else order_id
            ask_order = quote['order_id'] if quote['side'] == 'ask' else order_id
            
            # Quantité à exécuter
            qty = min(available, qtyToExec)
            qtyToExec -= qty
            
            # Créer le trade
            trade = (bid_order, ask_order, self.time, price, qty)
            trades.append(trade)
            
            if verbose:
                print(f">>> TRADE: t={self.time} price=${price} qty={qty} "
                      f"counterparty={counterparty} trader={quote['tid']}")
        
        # Enregistrer les trades
        if trades:
            crsr.executemany(self.insert_trade, trades)
            self.lastPrice[instrument] = price
            crsr.execute(self.set_lastprice, 
                        dict(instrument=instrument, lastprice=price))
        
        return trades, quote
    
    def cancelOrder(
        self, 
        side: str, 
        idNum: int, 
        time: int = None
    ) -> None:
        """
        Annule un ordre existant.
        
        Args:
            side: Côté de l'ordre ('bid' ou 'ask')
            idNum: Identifiant externe de l'ordre
            time: Timestamp (optionnel)
        
        Exemple:
            >>> # Annuler l'ordre #5 côté bid
            >>> lob.cancelOrder('bid', 5)
        """
        if time:
            self.time = time
        else:
            self.updateTime()
        
        crsr = self.db.cursor()
        crsr.execute('BEGIN TRANSACTION')
        crsr.execute(self.cancel_order, dict(cancel=1, idNum=idNum, side=side))
        crsr.execute('COMMIT')
    
    def modifyOrder(
        self, 
        idNum: int, 
        orderUpdate: dict, 
        time: int = None,
        verbose: bool = False
    ) -> Tuple[List[dict], dict]:
        """
        Modifie un ordre existant.
        
        ATTENTION : Modifier la quantité à la hausse ou améliorer le prix
        fait perdre la priorité temporelle !
        
        Args:
            idNum: Identifiant de l'ordre à modifier
            orderUpdate: Nouvelles valeurs (side, qty, price, tid)
            time: Timestamp (optionnel)
            verbose: Afficher les détails
        
        Returns:
            Tuple (trades si l'ordre s'exécute, info de l'ordre)
        
        Exemple:
            >>> # Augmenter la quantité de l'ordre #5
            >>> update = {'side': 'bid', 'qty': 20, 'price': 100, 'tid': 1}
            >>> trades, info = lob.modifyOrder(5, update)
        """
        if time:
            self.time = time
        else:
            self.updateTime()
        
        side = orderUpdate['side']
        orderUpdate['idNum'] = idNum
        orderUpdate['timestamp'] = self.time
        
        ret = ([], orderUpdate)
        
        crsr = self.db.cursor()
        crsr.execute('BEGIN TRANSACTION')
        
        # Trouver l'ordre existant
        row = crsr.execute(self.find_order, (idNum,)).fetchone()
        
        if row:
            side, instrument, price, qty, fulfilled, cancel, order_id, order_type = row
            
            orderUpdate.update(
                type=order_type,
                order_id=order_id,
                instrument=instrument,
            )
            
            if orderUpdate.get('price'):
                orderUpdate['price'] = self.clipPrice(orderUpdate['price'])
            
            # Appliquer la modification
            crsr.execute(self.modify_order, orderUpdate)
            
            # Si le prix s'améliore, tenter le matching
            if self._betterPrice(side, price, orderUpdate.get('price')):
                ret = self._processMatchesDB(orderUpdate, crsr, verbose)
        
        crsr.execute('COMMIT')
        return ret
    
    def _betterPrice(
        self, 
        side: str, 
        price: Optional[float], 
        comparedPrice: Optional[float]
    ) -> bool:
        """
        Vérifie si comparedPrice est meilleur que price pour le matching.
        
        Pour un BID : un prix plus élevé est meilleur
        Pour un ASK : un prix plus bas est meilleur
        """
        if price is None and comparedPrice is not None:
            return False
        if price is not None and comparedPrice is None:
            return True
        
        if side == 'bid':
            return price < comparedPrice
        elif side == 'ask':
            return price > comparedPrice
        else:
            raise ValueError("side doit être 'bid' ou 'ask'")
    
    def getVolumeAtPrice(
        self, 
        instrument: str, 
        side: str, 
        price: float
    ) -> int:
        """
        Retourne le volume disponible jusqu'à un certain prix.
        
        Args:
            instrument: Symbole de l'instrument
            side: 'bid' ou 'ask'
            price: Prix limite
        
        Returns:
            Volume total disponible
        
        Exemple:
            >>> # Combien puis-je acheter à 102$ max ?
            >>> volume = lob.getVolumeAtPrice('AAPL', 'ask', 102)
            >>> print(f"Volume disponible: {volume} unités")
        """
        price = self.clipPrice(price)
        crsr = self.db.cursor()
        params = dict(instrument=instrument, side=side, price=price)
        result = crsr.execute(self.volume_at_price, params).fetchone()
        
        if result:
            return result[0]
        return 0
    
    def getBestBid(self, instrument: str) -> Optional[float]:
        """
        Retourne le meilleur prix d'achat (le plus élevé).
        
        Args:
            instrument: Symbole de l'instrument
        
        Returns:
            Meilleur bid ou None si le carnet est vide
        """
        return self._getPrice(instrument, 'bid', 'DESC')
    
    def getBestAsk(self, instrument: str) -> Optional[float]:
        """
        Retourne le meilleur prix de vente (le plus bas).
        
        Args:
            instrument: Symbole de l'instrument
        
        Returns:
            Meilleur ask ou None si le carnet est vide
        """
        return self._getPrice(instrument, 'ask', 'ASC')
    
    def getWorstBid(self, instrument: str) -> Optional[float]:
        """Retourne le pire prix d'achat (le plus bas)."""
        return self._getPrice(instrument, 'bid', 'ASC')
    
    def getWorstAsk(self, instrument: str) -> Optional[float]:
        """Retourne le pire prix de vente (le plus élevé)."""
        return self._getPrice(instrument, 'ask', 'DESC')
    
    def _getPrice(
        self, 
        instrument: str, 
        side: str, 
        direction: str
    ) -> Optional[float]:
        """Helper pour obtenir un prix extrême."""
        crsr = self.db.cursor()
        
        order_clause = self.best_quotes_order_asc if direction == 'ASC' else self.best_quotes_order_desc
        sql = self.active_orders + order_clause + " LIMIT 1"
        
        result = crsr.execute(sql, dict(instrument=instrument, side=side)).fetchone()
        
        if result:
            return result[3]  # price est en position 3
        return None
    
    def getSpread(self, instrument: str) -> Optional[float]:
        """
        Calcule le spread (écart bid-ask).
        
        Spread = Best Ask - Best Bid
        
        Un spread faible indique un marché liquide.
        Un spread élevé indique un marché illiquide.
        
        Args:
            instrument: Symbole de l'instrument
        
        Returns:
            Spread ou None si incomplet
        
        Exemple:
            >>> spread = lob.getSpread('AAPL')
            >>> if spread:
            ...     print(f"Spread: ${spread:.2f}")
        """
        best_bid = self.getBestBid(instrument)
        best_ask = self.getBestAsk(instrument)
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def getMidPrice(self, instrument: str) -> Optional[float]:
        """
        Calcule le prix milieu (mid price).
        
        Mid Price = (Best Bid + Best Ask) / 2
        
        Le mid price est souvent utilisé comme proxy du "vrai" prix.
        
        Args:
            instrument: Symbole de l'instrument
        
        Returns:
            Mid price ou None si incomplet
        """
        best_bid = self.getBestBid(instrument)
        best_ask = self.getBestAsk(instrument)
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def print(self, instrument: str) -> str:
        """
        Affiche l'état actuel du carnet d'ordres.
        
        Args:
            instrument: Symbole de l'instrument
        
        Returns:
            Chaîne formatée du carnet
        """
        crsr = self.db.cursor()
        sql_active = self.active_orders + self.best_quotes_order_asc
        
        output = StringIO()
        
        output.write("\n" + "=" * 50 + "\n")
        output.write(f"    CARNET D'ORDRES - {instrument}\n")
        output.write("=" * 50 + "\n")
        
        # BIDS
        output.write("\n------ BIDS (Achats) -------\n")
        bids = crsr.execute(sql_active, dict(instrument=instrument, side='bid')).fetchall()
        for bid in bids:
            idNum, qty, fulfilled, price, event_dt, inst = bid
            remaining = qty - fulfilled
            output.write(f"  #{idNum}) {remaining}/{qty} @ ${price} (t={event_dt})\n")
        
        if not bids:
            output.write("  (vide)\n")
        
        # ASKS
        output.write("\n------ ASKS (Ventes) -------\n")
        asks = crsr.execute(sql_active, dict(instrument=instrument, side='ask')).fetchall()
        for ask in asks:
            idNum, qty, fulfilled, price, event_dt, inst = ask
            remaining = qty - fulfilled
            output.write(f"  #{idNum}) {remaining}/{qty} @ ${price} (t={event_dt})\n")
        
        if not asks:
            output.write("  (vide)\n")
        
        # Métriques
        output.write("\n------ MÉTRIQUES -------\n")
        output.write(f"  Best Bid: {self.getBestBid(instrument)}\n")
        output.write(f"  Best Ask: {self.getBestAsk(instrument)}\n")
        output.write(f"  Spread: {self.getSpread(instrument)}\n")
        output.write(f"  Mid Price: {self.getMidPrice(instrument)}\n")
        
        output.write("=" * 50 + "\n")
        
        result = output.getvalue()
        print(result)
        return result
```

---

# 8. Exemples d'Utilisation

## 8.1 Scénario Complet

```python
def run_complete_example():
    """
    Exemple complet d'utilisation du LOB.
    
    Ce scénario simule une journée de trading avec :
    1. Création de traders
    2. Soumission d'ordres limite
    3. Exécution d'ordres marché
    4. Modification et annulation
    """
    print("=" * 70)
    print("EXEMPLE COMPLET : SIMULATION D'UNE JOURNÉE DE TRADING")
    print("=" * 70)
    
    # 1. Initialisation
    print("\n[1] INITIALISATION")
    
    conn = create_lob_database(":memory:")
    setup_test_data(conn, "FAKE")
    lob = OrderBook(db=conn, tick_size=0.01)
    
    instrument = "FAKE"
    print(f"  Instrument: {instrument}")
    print(f"  Tick size: {lob.tickSize}")
    
    # 2. Soumission d'ordres limite initiaux
    print("\n[2] SOUMISSION D'ORDRES LIMITE")
    
    initial_orders = [
        # ASK (ventes)
        {'type': 'limit', 'side': 'ask', 'instrument': instrument, 
         'qty': 5, 'price': 101, 'tid': 1},
        {'type': 'limit', 'side': 'ask', 'instrument': instrument, 
         'qty': 5, 'price': 103, 'tid': 2},
        {'type': 'limit', 'side': 'ask', 'instrument': instrument, 
         'qty': 5, 'price': 101, 'tid': 3},
        # BID (achats)
        {'type': 'limit', 'side': 'bid', 'instrument': instrument, 
         'qty': 5, 'price': 99, 'tid': 1},
        {'type': 'limit', 'side': 'bid', 'instrument': instrument, 
         'qty': 5, 'price': 98, 'tid': 2},
        {'type': 'limit', 'side': 'bid', 'instrument': instrument, 
         'qty': 5, 'price': 99, 'tid': 3},
    ]
    
    for order in initial_orders:
        trades, info = lob.processOrder(order, False, False)
        print(f"  Ordre #{info['idNum']}: {order['side'].upper()} "
              f"{order['qty']} @ {order['price']}")
    
    print("\nCarnet après ordres initiaux:")
    lob.print(instrument)
    
    # 3. Ordre limite qui croise (crossing order)
    print("\n[3] ORDRE LIMITE QUI CROISE")
    print("  Un ordre d'achat à 102$ va matcher avec les ventes à 101$")
    
    crossing_order = {
        'type': 'limit', 
        'side': 'bid', 
        'instrument': instrument,
        'qty': 2, 
        'price': 102,
        'tid': 1
    }
    
    trades, info = lob.processOrder(crossing_order, False, True)
    print(f"\n  Résultat: {len(trades)} trade(s) exécuté(s)")
    
    lob.print(instrument)
    
    # 4. Gros ordre limite (partial fill)
    print("\n[4] GROS ORDRE LIMITE (EXÉCUTION PARTIELLE)")
    print("  Un ordre d'achat de 50 unités à 102$ va :")
    print("  - Exécuter ce qui est disponible à ≤102$")
    print("  - Placer le reste dans le carnet")
    
    big_order = {
        'type': 'limit', 
        'side': 'bid', 
        'instrument': instrument,
        'qty': 50, 
        'price': 102,
        'tid': 2
    }
    
    trades, info = lob.processOrder(big_order, False, True)
    print(f"\n  Trades exécutés: {len(trades)}")
    print(f"  Quantité restante dans le carnet: {50 - sum(t[4] for t in trades)}")
    
    lob.print(instrument)
    
    # 5. Ordre au marché
    print("\n[5] ORDRE AU MARCHÉ")
    print("  Une vente au marché de 40 unités prend les meilleurs bids")
    
    market_order = {
        'type': 'market', 
        'side': 'ask', 
        'instrument': instrument,
        'qty': 40, 
        'tid': 3
    }
    
    trades, info = lob.processOrder(market_order, False, True)
    
    lob.print(instrument)
    
    # 6. Annulation d'ordre
    print("\n[6] ANNULATION D'ORDRE")
    print("  Annulation de l'ordre bid #4")
    
    lob.cancelOrder('bid', 4)
    
    lob.print(instrument)
    
    # 7. Modification d'ordre
    print("\n[7] MODIFICATION D'ORDRE")
    print("  Modification de l'ordre #5: augmentation à 14 unités")
    print("  Note: Cela fait perdre la priorité temporelle")
    
    modify_update = {
        'side': 'bid', 
        'qty': 14, 
        'price': 99,
        'tid': 1
    }
    
    trades, info = lob.modifyOrder(5, modify_update)
    
    lob.print(instrument)
    
    print("\n[FIN DE LA SIMULATION]")
    
    conn.close()


# Exécuter l'exemple
run_complete_example()
```

## 8.2 Simulation de Market Making

```python
def simulate_market_making(
    n_rounds: int = 100,
    initial_price: float = 100.0,
    spread: float = 0.10,
    order_size: int = 10,
    volatility: float = 0.01
) -> dict:
    """
    Simule une stratégie de market making.
    
    Un Market Maker (Teneur de Marché) :
    - Place des ordres des deux côtés du carnet
    - Profite du spread bid-ask
    - Prend le risque d'inventaire
    
    Args:
        n_rounds: Nombre de tours de simulation
        initial_price: Prix initial de l'actif
        spread: Spread bid-ask ciblé
        order_size: Taille des ordres
        volatility: Volatilité du prix
    
    Returns:
        Statistiques de la simulation
    
    Exemple:
        >>> stats = simulate_market_making(n_rounds=1000)
        >>> print(f"PnL (Profit and Loss): ${stats['pnl']:.2f}")
    """
    import random
    
    print("=" * 70)
    print("SIMULATION DE MARKET MAKING")
    print("=" * 70)
    
    # Initialisation
    conn = create_lob_database(":memory:")
    setup_test_data(conn, "TEST")
    lob = OrderBook(db=conn, tick_size=0.01)
    
    instrument = "TEST"
    mm_trader_id = 1  # Market Maker
    
    # État du market maker
    mm_inventory = 0  # Position en actions
    mm_cash = 0  # Cash (commence à 0)
    price = initial_price
    
    stats = {
        'trades': [],
        'inventory_history': [],
        'cash_history': [],
        'pnl_history': []
    }
    
    for round_num in range(n_rounds):
        # 1. Le MM place ses ordres de chaque côté
        half_spread = spread / 2
        bid_price = price - half_spread
        ask_price = price + half_spread
        
        # Ordre d'achat (bid)
        bid_order = {
            'type': 'limit',
            'side': 'bid',
            'instrument': instrument,
            'qty': order_size,
            'price': round(bid_price, 2),
            'tid': mm_trader_id
        }
        
        # Ordre de vente (ask)
        ask_order = {
            'type': 'limit',
            'side': 'ask',
            'instrument': instrument,
            'qty': order_size,
            'price': round(ask_price, 2),
            'tid': mm_trader_id
        }
        
        lob.processOrder(bid_order, False, False)
        lob.processOrder(ask_order, False, False)
        
        # 2. Simulation d'un trader "noise" qui arrive
        noise_side = random.choice(['bid', 'ask'])
        noise_qty = random.randint(1, order_size * 2)
        
        noise_order = {
            'type': 'market',
            'side': noise_side,
            'instrument': instrument,
            'qty': noise_qty,
            'tid': 2  # Noise trader
        }
        
        trades, _ = lob.processOrder(noise_order, False, False)
        
        # 3. Mise à jour de l'état du MM
        for trade in trades:
            trade_qty = trade[4]
            trade_price = trade[3]
            
            if noise_side == 'bid':
                # Le noise trader achète → le MM vend
                mm_inventory -= trade_qty
                mm_cash += trade_qty * trade_price
            else:
                # Le noise trader vend → le MM achète
                mm_inventory += trade_qty
                mm_cash -= trade_qty * trade_price
            
            stats['trades'].append({
                'round': round_num,
                'side': 'sell' if noise_side == 'bid' else 'buy',
                'qty': trade_qty,
                'price': trade_price
            })
        
        # 4. Mouvement aléatoire du prix
        price_change = random.gauss(0, volatility) * price
        price = max(1, price + price_change)  # Prix minimum de 1
        
        # 5. Enregistrement des stats
        mark_to_market = mm_cash + mm_inventory * price
        stats['inventory_history'].append(mm_inventory)
        stats['cash_history'].append(mm_cash)
        stats['pnl_history'].append(mark_to_market)
        
        # Affichage périodique
        if (round_num + 1) % 20 == 0:
            print(f"Round {round_num + 1}: "
                  f"Prix={price:.2f}, "
                  f"Inventaire={mm_inventory}, "
                  f"Cash={mm_cash:.2f}, "
                  f"PnL={mark_to_market:.2f}")
    
    # Résultats finaux
    final_pnl = mm_cash + mm_inventory * price
    
    print("\n" + "=" * 50)
    print("RÉSULTATS FINAUX")
    print("=" * 50)
    print(f"Nombre de trades: {len(stats['trades'])}")
    print(f"Inventaire final: {mm_inventory}")
    print(f"Cash final: ${mm_cash:.2f}")
    print(f"Prix final: ${price:.2f}")
    print(f"PnL (Profit and Loss) final: ${final_pnl:.2f}")
    
    stats['pnl'] = final_pnl
    stats['final_inventory'] = mm_inventory
    stats['final_cash'] = mm_cash
    stats['final_price'] = price
    
    conn.close()
    
    return stats


# Exécuter la simulation
# stats = simulate_market_making(n_rounds=100)
```

---

# 9. Visualisation du Carnet d'Ordres

```python
"""
VISUALISATION DU LOB
====================
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_order_book(
    bids: List[Tuple[float, int]],
    asks: List[Tuple[float, int]],
    instrument: str = "ASSET",
    last_price: float = None
) -> None:
    """
    Visualise le carnet d'ordres en format graphique.
    
    Crée un graphique en escalier montrant :
    - Les bids (achats) en vert à gauche
    - Les asks (ventes) en rouge à droite
    - Le spread au milieu
    
    Args:
        bids: Liste de tuples (prix, volume) triés par prix décroissant
        asks: Liste de tuples (prix, volume) triés par prix croissant
        instrument: Nom de l'instrument
        last_price: Dernier prix de transaction
    
    Exemple:
        >>> bids = [(99.5, 100), (99.0, 150), (98.5, 200)]
        >>> asks = [(100.5, 80), (101.0, 120), (101.5, 90)]
        >>> visualize_order_book(bids, asks, "AAPL", 100.0)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Préparer les données des bids (cumul décroissant)
    if bids:
        bid_prices = [b[0] for b in bids]
        bid_volumes = [b[1] for b in bids]
        bid_cumulative = np.cumsum(bid_volumes)
        
        # Tracer en escalier
        ax.fill_between(
            bid_prices, 0, bid_cumulative,
            step='post', alpha=0.4, color='green', label='Bids (Achats)'
        )
        ax.step(bid_prices, bid_cumulative, where='post', color='darkgreen', linewidth=2)
    
    # Préparer les données des asks (cumul croissant)
    if asks:
        ask_prices = [a[0] for a in asks]
        ask_volumes = [a[1] for a in asks]
        ask_cumulative = np.cumsum(ask_volumes)
        
        ax.fill_between(
            ask_prices, 0, ask_cumulative,
            step='post', alpha=0.4, color='red', label='Asks (Ventes)'
        )
        ax.step(ask_prices, ask_cumulative, where='post', color='darkred', linewidth=2)
    
    # Ligne du dernier prix
    if last_price:
        ax.axvline(x=last_price, color='blue', linestyle='--', linewidth=2, 
                  label=f'Last Price: {last_price}')
    
    # Annotations
    if bids and asks:
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        ax.axvspan(best_bid, best_ask, alpha=0.1, color='yellow', label=f'Spread: {spread:.2f}')
        
        # Texte du spread
        ax.annotate(
            f'Spread\n{spread:.2f}',
            xy=(mid_price, ax.get_ylim()[1] * 0.8),
            ha='center', va='center',
            fontsize=12, fontweight='bold'
        )
    
    # Mise en forme
    ax.set_xlabel('Prix', fontsize=12)
    ax.set_ylabel('Volume Cumulé', fontsize=12)
    ax.set_title(f'Carnet d\'Ordres - {instrument}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('orderbook_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_depth_chart(
    bids: List[Tuple[float, int]],
    asks: List[Tuple[float, int]],
    instrument: str = "ASSET"
) -> None:
    """
    Crée un graphique de profondeur (depth chart) du carnet.
    
    Le depth chart montre le volume cumulé disponible à chaque niveau de prix.
    C'est une visualisation classique utilisée par les plateformes de trading.
    
    Args:
        bids: Liste de (prix, volume) côté achat
        asks: Liste de (prix, volume) côté vente
        instrument: Symbole de l'instrument
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Bids (de droite à gauche)
    if bids:
        bid_prices = [b[0] for b in sorted(bids, reverse=True)]
        bid_volumes = [b[1] for b in sorted(bids, key=lambda x: x[0], reverse=True)]
        bid_cumulative = np.cumsum(bid_volumes)
        
        ax.fill_between(bid_prices, bid_cumulative, 0, 
                       step='pre', alpha=0.5, color='#00C853')
        ax.plot(bid_prices, bid_cumulative, 
               drawstyle='steps-pre', color='#00C853', linewidth=2)
    
    # Asks (de gauche à droite)
    if asks:
        ask_prices = [a[0] for a in sorted(asks)]
        ask_volumes = [a[1] for a in sorted(asks, key=lambda x: x[0])]
        ask_cumulative = np.cumsum(ask_volumes)
        
        ax.fill_between(ask_prices, ask_cumulative, 0, 
                       step='post', alpha=0.5, color='#FF1744')
        ax.plot(ask_prices, ask_cumulative, 
               drawstyle='steps-post', color='#FF1744', linewidth=2)
    
    # Style
    ax.set_xlabel('Prix', fontsize=12)
    ax.set_ylabel('Volume Cumulé', fontsize=12)
    ax.set_title(f'Depth Chart - {instrument}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Annotations BBO
    if bids and asks:
        best_bid = max(b[0] for b in bids)
        best_ask = min(a[0] for a in asks)
        
        ax.axvline(x=best_bid, color='green', linestyle=':', alpha=0.7)
        ax.axvline(x=best_ask, color='red', linestyle=':', alpha=0.7)
        
        ax.annotate(f'Best Bid\n{best_bid}', 
                   xy=(best_bid, 0), xytext=(best_bid - 0.5, ax.get_ylim()[1] * 0.3),
                   ha='right', fontsize=10, color='green')
        ax.annotate(f'Best Ask\n{best_ask}', 
                   xy=(best_ask, 0), xytext=(best_ask + 0.5, ax.get_ylim()[1] * 0.3),
                   ha='left', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('depth_chart.png', dpi=150, bbox_inches='tight')
    plt.show()


# Exemple de visualisation
print("\n=== VISUALISATION DU CARNET D'ORDRES ===\n")

# Données d'exemple
example_bids = [
    (99.90, 150),
    (99.80, 200),
    (99.70, 300),
    (99.60, 250),
    (99.50, 400),
]

example_asks = [
    (100.10, 120),
    (100.20, 180),
    (100.30, 220),
    (100.40, 280),
    (100.50, 350),
]

# Décommenter pour afficher les graphiques
# visualize_order_book(example_bids, example_asks, "EXAMPLE", 100.0)
# visualize_depth_chart(example_bids, example_asks, "EXAMPLE")
```

---

# 10. Stratégies de Trading Automatisé

## 10.1 Types de Stratégies

```python
"""
STRATÉGIES DE TRADING AVEC LOB
==============================
"""

from abc import ABC, abstractmethod
from typing import Optional


class TradingStrategy(ABC):
    """
    Classe de base pour les stratégies de trading.
    
    Une stratégie analyse l'état du carnet et génère des signaux
    d'achat ou de vente.
    """
    
    def __init__(self, lob: OrderBook, instrument: str, trader_id: int):
        self.lob = lob
        self.instrument = instrument
        self.trader_id = trader_id
        self.position = 0
        self.pnl = 0.0
    
    @abstractmethod
    def generate_signal(self) -> Optional[dict]:
        """Génère un signal de trading (ordre à soumettre)."""
        pass
    
    def execute(self) -> Optional[Tuple[List, dict]]:
        """Exécute le signal généré."""
        signal = self.generate_signal()
        if signal:
            return self.lob.processOrder(signal, False, False)
        return None


class SpreadTradingStrategy(TradingStrategy):
    """
    Stratégie de capture de spread.
    
    Place des ordres des deux côtés pour capturer le spread.
    Similaire au market making mais plus simple.
    
    Logique :
    - Si spread > threshold, placer des ordres limite
    - Fermer les positions quand le spread se resserre
    """
    
    def __init__(
        self, 
        lob: OrderBook, 
        instrument: str, 
        trader_id: int,
        spread_threshold: float = 0.10,
        order_size: int = 10
    ):
        super().__init__(lob, instrument, trader_id)
        self.spread_threshold = spread_threshold
        self.order_size = order_size
    
    def generate_signal(self) -> Optional[dict]:
        """
        Génère un signal basé sur le spread.
        
        Returns:
            Ordre à soumettre ou None
        """
        best_bid = self.lob.getBestBid(self.instrument)
        best_ask = self.lob.getBestAsk(self.instrument)
        
        if best_bid is None or best_ask is None:
            return None
        
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Si le spread est suffisamment large
        if spread >= self.spread_threshold:
            # Placer un ordre d'achat juste au-dessus du best bid
            return {
                'type': 'limit',
                'side': 'bid',
                'instrument': self.instrument,
                'qty': self.order_size,
                'price': best_bid + 0.01,
                'tid': self.trader_id
            }
        
        return None


class MomentumStrategy(TradingStrategy):
    """
    Stratégie de momentum basée sur le déséquilibre du carnet.
    
    Le déséquilibre (imbalance) mesure la pression acheteuse vs vendeuse.
    
    Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
    
    - Imbalance > 0 : Pression acheteuse → Signal d'achat
    - Imbalance < 0 : Pression vendeuse → Signal de vente
    """
    
    def __init__(
        self, 
        lob: OrderBook, 
        instrument: str, 
        trader_id: int,
        imbalance_threshold: float = 0.3,
        order_size: int = 10,
        n_levels: int = 3
    ):
        """
        Args:
            imbalance_threshold: Seuil de déséquilibre pour déclencher
            n_levels: Nombre de niveaux de prix à analyser
        """
        super().__init__(lob, instrument, trader_id)
        self.imbalance_threshold = imbalance_threshold
        self.order_size = order_size
        self.n_levels = n_levels
    
    def calculate_imbalance(self) -> float:
        """
        Calcule le déséquilibre du carnet.
        
        Returns:
            Imbalance entre -1 et 1
        """
        best_bid = self.lob.getBestBid(self.instrument)
        best_ask = self.lob.getBestAsk(self.instrument)
        
        if best_bid is None or best_ask is None:
            return 0.0
        
        # Volume côté bid
        bid_volume = self.lob.getVolumeAtPrice(
            self.instrument, 'bid', 
            best_bid - 1.0  # Inclure quelques niveaux
        )
        
        # Volume côté ask
        ask_volume = self.lob.getVolumeAtPrice(
            self.instrument, 'ask',
            best_ask + 1.0
        )
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    def generate_signal(self) -> Optional[dict]:
        """
        Génère un signal basé sur le déséquilibre.
        
        - Fort déséquilibre positif → Achat agressif
        - Fort déséquilibre négatif → Vente agressive
        """
        imbalance = self.calculate_imbalance()
        
        if imbalance > self.imbalance_threshold:
            # Forte pression acheteuse → Acheter avant la hausse
            best_ask = self.lob.getBestAsk(self.instrument)
            if best_ask:
                return {
                    'type': 'market',  # Exécution immédiate
                    'side': 'bid',
                    'instrument': self.instrument,
                    'qty': self.order_size,
                    'tid': self.trader_id
                }
        
        elif imbalance < -self.imbalance_threshold:
            # Forte pression vendeuse → Vendre avant la baisse
            best_bid = self.lob.getBestBid(self.instrument)
            if best_bid:
                return {
                    'type': 'market',
                    'side': 'ask',
                    'instrument': self.instrument,
                    'qty': self.order_size,
                    'tid': self.trader_id
                }
        
        return None


class TWAPStrategy:
    """
    TWAP (Time Weighted Average Price) Strategy.
    
    Exécute un gros ordre en le divisant en petits morceaux
    répartis uniformément dans le temps.
    
    Objectif : Minimiser l'impact de marché d'un gros ordre.
    
    Exemple :
        Vendre 1000 actions en 10 tranches de 100 sur 1 heure
        → 1 tranche toutes les 6 minutes
    """
    
    def __init__(
        self,
        lob: OrderBook,
        instrument: str,
        trader_id: int,
        total_qty: int,
        side: str,
        n_slices: int = 10,
        use_limit: bool = True
    ):
        """
        Args:
            total_qty: Quantité totale à exécuter
            side: 'bid' (achat) ou 'ask' (vente)
            n_slices: Nombre de tranches
            use_limit: Utiliser des ordres limite (True) ou marché (False)
        """
        self.lob = lob
        self.instrument = instrument
        self.trader_id = trader_id
        self.total_qty = total_qty
        self.side = side
        self.n_slices = n_slices
        self.use_limit = use_limit
        
        self.slice_qty = total_qty // n_slices
        self.executed_qty = 0
        self.trades = []
    
    def execute_slice(self) -> Optional[Tuple[List, dict]]:
        """
        Exécute une tranche de l'ordre.
        
        Returns:
            Résultat de l'exécution (trades, order_info)
        """
        if self.executed_qty >= self.total_qty:
            return None
        
        remaining = self.total_qty - self.executed_qty
        qty = min(self.slice_qty, remaining)
        
        if self.use_limit:
            # Ordre limite au mid-price
            mid = self.lob.getMidPrice(self.instrument)
            if mid is None:
                return None
            
            # Ajuster le prix selon le côté
            if self.side == 'bid':
                price = mid - 0.01  # Légèrement sous le mid
            else:
                price = mid + 0.01  # Légèrement au-dessus
            
            order = {
                'type': 'limit',
                'side': self.side,
                'instrument': self.instrument,
                'qty': qty,
                'price': round(price, 2),
                'tid': self.trader_id
            }
        else:
            order = {
                'type': 'market',
                'side': self.side,
                'instrument': self.instrument,
                'qty': qty,
                'tid': self.trader_id
            }
        
        trades, info = self.lob.processOrder(order, False, False)
        
        executed = sum(t[4] for t in trades)
        self.executed_qty += executed
        self.trades.extend(trades)
        
        return trades, info
    
    def get_vwap(self) -> float:
        """
        Calcule le VWAP (Volume Weighted Average Price) des exécutions.
        
        VWAP = Σ(prix × volume) / Σ(volume)
        """
        if not self.trades:
            return 0.0
        
        total_value = sum(t[3] * t[4] for t in self.trades)  # price * qty
        total_qty = sum(t[4] for t in self.trades)
        
        return total_value / total_qty if total_qty > 0 else 0.0
    
    def status(self) -> dict:
        """Retourne le statut de l'exécution."""
        return {
            'total_qty': self.total_qty,
            'executed_qty': self.executed_qty,
            'remaining_qty': self.total_qty - self.executed_qty,
            'completion': self.executed_qty / self.total_qty * 100,
            'n_trades': len(self.trades),
            'vwap': self.get_vwap()
        }


# Démonstration des stratégies
print("\n=== STRATÉGIES DE TRADING ===\n")

print("""
STRATÉGIES IMPLÉMENTÉES :

1. SpreadTradingStrategy
   - Capture le spread bid-ask
   - Adapté aux marchés peu volatils

2. MomentumStrategy
   - Suit le déséquilibre du carnet
   - Anticipe les mouvements de prix

3. TWAPStrategy
   - Exécution progressive d'un gros ordre
   - Minimise l'impact de marché

Chaque stratégie peut être backtestée sur des données historiques
ou utilisée en temps réel avec le LOB.
""")
```

---

# Annexes

## A.1 Récapitulatif des Concepts

| Concept | Description |
|---------|-------------|
| **LOB** | Structure centrale qui stocke tous les ordres actifs |
| **BBO** | Best Bid and Offer - meilleurs prix d'achat et vente |
| **Spread** | Écart entre le meilleur ask et le meilleur bid |
| **Mid Price** | Prix milieu = (Bid + Ask) / 2 |
| **Tick** | Plus petite variation de prix autorisée |
| **FIFO** | First In First Out - priorité temporelle |
| **Market Order** | Ordre exécuté immédiatement au prix du marché |
| **Limit Order** | Ordre avec prix limite, peut rester dans le carnet |
| **Fill** | Exécution d'un ordre |
| **Partial Fill** | Exécution partielle |
| **Slippage** | Écart entre prix attendu et prix réel |
| **Market Impact** | Mouvement de prix causé par un ordre |

## A.2 Formules Importantes

```
Spread = Best Ask - Best Bid

Mid Price = (Best Bid + Best Ask) / 2

Spread (bps) = (Spread / Mid Price) × 10000

Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)

VWAP = Σ(Price × Volume) / Σ(Volume)

Market Impact = (Execution Price - Initial Price) / Initial Price
```

## A.3 Références

1. **PyLOB Original** : https://github.com/ab24v07/PyLOB
2. **Harris, L.** (2003). "Trading and Exchanges"
3. **Gould et al.** (2013). "Limit Order Books"
4. **Cont et al.** (2010). "A Stochastic Model for Order Book Dynamics"

---

*Document généré avec tous les acronymes développés et exemples de code fonctionnel.*

*Version: Janvier 2026*
