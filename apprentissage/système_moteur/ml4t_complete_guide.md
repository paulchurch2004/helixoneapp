# 🤖 Machine Learning for Algorithmic Trading
## Guide Complet / Complete Guide

**Auteur Original / Original Author**: Stefan Jansen  
**Documentation**: HelixOne Complete Reference  
**Version**: 2.0 - Second Edition

---

# 📑 TABLE DES MATIÈRES / TABLE OF CONTENTS

1. [Introduction](#1-introduction)
2. [Données de Marché et Fondamentales](#2-données-de-marché-et-fondamentales)
3. [Données Alternatives](#3-données-alternatives)
4. [Recherche de Facteurs Alpha](#4-recherche-de-facteurs-alpha)
5. [Évaluation de Stratégie](#5-évaluation-de-stratégie)
6. [Processus Machine Learning](#6-processus-machine-learning)
7. [Modèles Linéaires](#7-modèles-linéaires)
8. [Workflow ML4T Complet](#8-workflow-ml4t-complet)
9. [Modèles de Séries Temporelles](#9-modèles-de-séries-temporelles)
10. [Machine Learning Bayésien](#10-machine-learning-bayésien)
11. [Arbres de Décision et Forêts Aléatoires](#11-arbres-de-décision-et-forêts-aléatoires)
12. [Gradient Boosting Machines](#12-gradient-boosting-machines)
13. [Apprentissage Non Supervisé](#13-apprentissage-non-supervisé)
14. [Traitement du Langage Naturel (NLP)](#14-traitement-du-langage-naturel-nlp)
15. [Modélisation de Sujets (Topic Modeling)](#15-modélisation-de-sujets-topic-modeling)
16. [Embeddings de Mots](#16-embeddings-de-mots)
17. [Deep Learning - Réseaux Feedforward](#17-deep-learning---réseaux-feedforward)
18. [Réseaux de Neurones Convolutionnels (CNN)](#18-réseaux-de-neurones-convolutionnels-cnn)
19. [Réseaux de Neurones Récurrents (RNN)](#19-réseaux-de-neurones-récurrents-rnn)
20. [Autoencodeurs](#20-autoencodeurs)
21. [Réseaux Adverses Génératifs (GAN)](#21-réseaux-adverses-génératifs-gan)
22. [Apprentissage par Renforcement](#22-apprentissage-par-renforcement)
23. [Prochaines Étapes](#23-prochaines-étapes)
24. [Bibliothèque de Facteurs Alpha](#24-bibliothèque-de-facteurs-alpha)

---

# 1. INTRODUCTION

## 1.1 Qu'est-ce que le ML4T (Machine Learning for Trading)?

Le **ML4T** (Machine Learning for Trading - Apprentissage Automatique pour le Trading) est l'application des techniques de Machine Learning (ML) et Deep Learning (DL) pour:

1. **Générer des signaux de trading** (alpha factors - facteurs alpha)
2. **Optimiser les portefeuilles** (portfolio optimization - optimisation de portefeuille)
3. **Exécuter des ordres** (order execution - exécution d'ordres)
4. **Gérer les risques** (risk management - gestion des risques)

## 1.2 Architecture du Workflow ML4T

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WORKFLOW ML4T                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  DATA    │───▶│ FEATURES │───▶│  MODEL   │───▶│ BACKTEST │      │
│  │ SOURCES  │    │ENGINEERING│   │ TRAINING │    │          │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│       │              │               │               │              │
│       ▼              ▼               ▼               ▼              │
│  - Market Data   - Technical    - Linear Models  - Zipline         │
│  - Fundamental     Indicators   - Tree-based     - Backtrader      │
│  - Alternative   - Alpha        - Deep Learning  - PyFolio         │
│  - SEC Filings     Factors      - Ensemble       - Alphalens       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 1.3 Dépendances Python Principales

```python
# === CORE DATA SCIENCE ===
import numpy as np                    # Calcul numérique (numerical computing)
import pandas as pd                   # Manipulation de données (data manipulation)
from scipy import stats               # Statistiques (statistics)

# === MACHINE LEARNING ===
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb               # LightGBM (Light Gradient Boosting Machine)
import catboost as cb                # CatBoost (Categorical Boosting)
import xgboost as xgb                # XGBoost (eXtreme Gradient Boosting)

# === DEEP LEARNING ===
import tensorflow as tf              # TensorFlow (DL framework by Google)
from tensorflow import keras         # Keras (High-level DL API)
import torch                         # PyTorch (DL framework by Meta)
import torch.nn as nn                # Neural Network modules

# === FINANCE & TRADING ===
import yfinance as yf                # Yahoo Finance API
import pandas_datareader as web      # Financial data reader
from zipline.api import order_target_percent, record, symbol
from alphalens import utils, performance, plotting
import pyfolio as pf                 # Portfolio analysis
import talib                         # TA-Lib (Technical Analysis Library)

# === NLP (Natural Language Processing) ===
import spacy                         # spaCy NLP library
from gensim.models import Word2Vec, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# === VISUALIZATION ===
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```

## 1.4 Configuration de l'Environnement

```bash
# Créer l'environnement conda
conda create -n ml4t python=3.8

# Activer l'environnement
conda activate ml4t

# Installer les dépendances principales
conda install -c conda-forge \
    numpy pandas scipy scikit-learn \
    matplotlib seaborn plotly \
    jupyter jupyterlab \
    statsmodels arch \
    lightgbm catboost xgboost

# Installer les packages financiers
pip install yfinance alphalens-reloaded pyfolio-reloaded zipline-reloaded

# Installer TA-Lib (nécessite compilation)
conda install -c conda-forge ta-lib

# Installer les packages NLP
pip install spacy gensim textblob
python -m spacy download en_core_web_sm

# Installer TensorFlow et PyTorch
pip install tensorflow torch torchvision
```

---

# 2. DONNÉES DE MARCHÉ ET FONDAMENTALES
## Market and Fundamental Data

## 2.1 Sources de Données

| Source | Type | Fréquence | Accès |
|--------|------|-----------|-------|
| **Yahoo Finance** | Prix OHLCV (Open-High-Low-Close-Volume) | Journalier | Gratuit |
| **Quandl** | Multi-sources | Variable | Freemium |
| **NASDAQ ITCH** | Order book (carnet d'ordres) | Tick | Payant |
| **SEC EDGAR** | Filings (déclarations) | Événementiel | Gratuit |
| **AlgoSeek** | Intraday | Minute | Payant |

## 2.2 Téléchargement avec yfinance

```python
"""
yfinance - Téléchargement de données Yahoo Finance
==================================================
yfinance permet de télécharger gratuitement les données de prix historiques
depuis Yahoo Finance.

Exemple: Télécharger les données AAPL (Apple Inc.)
"""
import yfinance as yf
import pandas as pd

# === Méthode 1: Ticker unique ===
ticker = yf.Ticker("AAPL")

# Obtenir les données historiques
# period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
# interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
hist = ticker.history(period="1y", interval="1d")
print(hist.head())
"""
                  Open        High         Low       Close    Volume  Dividends  Stock Splits
Date                                                                                          
2023-01-03  130.279999  130.899994  124.169998  125.070000  112117500        0.0           0.0
2023-01-04  126.889999  128.660004  125.080002  126.360001   89113600        0.0           0.0
"""

# Informations sur l'entreprise
info = ticker.info
print(f"Entreprise: {info['longName']}")
print(f"Secteur: {info['sector']}")
print(f"Market Cap: ${info['marketCap']:,.0f}")

# === Méthode 2: Téléchargement multiple ===
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

# Télécharger toutes les données en une fois
data = yf.download(
    tickers=tickers,
    start="2020-01-01",
    end="2024-01-01",
    interval="1d",
    group_by="ticker",    # Grouper par ticker
    auto_adjust=True,     # Ajuster pour dividendes/splits
    threads=True          # Téléchargement parallèle
)

# Accéder aux prix de clôture ajustés
close_prices = data.xs('Close', axis=1, level=1)
print(close_prices.head())

# === Méthode 3: Données intraday ===
intraday = yf.download(
    tickers="SPY",
    period="5d",
    interval="5m"    # Barres de 5 minutes
)
print(f"Nombre de barres: {len(intraday)}")
```

## 2.3 Téléchargement avec pandas-datareader

```python
"""
pandas-datareader - Accès à multiples sources de données
=========================================================
Permet d'accéder à: FRED, Fama-French, World Bank, OECD, etc.
"""
import pandas_datareader as web
from datetime import datetime

# === Données Fama-French (Facteurs de risque) ===
# Les facteurs Fama-French sont utilisés pour expliquer les rendements
# Mkt-RF: Rendement du marché moins le taux sans risque
# SMB: Small Minus Big (petites caps vs grandes caps)
# HML: High Minus Low (value vs growth)
# RMW: Robust Minus Weak (profitabilité)
# CMA: Conservative Minus Aggressive (investissement)

ff_factors = web.DataReader(
    'F-F_Research_Data_5_Factors_2x3',
    'famafrench',
    start='2010-01-01'
)[0]

# Convertir en pourcentages décimaux
ff_factors = ff_factors / 100
print(ff_factors.head())
"""
              Mkt-RF     SMB     HML     RMW     CMA      RF
Date                                                        
2010-01   -0.0327 -0.0081  0.0058  0.0040 -0.0065  0.0000
2010-02    0.0309  0.0089 -0.0057  0.0126  0.0085  0.0000
"""

# === Données FRED (Federal Reserve Economic Data) ===
# Taux d'intérêt, inflation, PIB, etc.
fred_data = web.DataReader(
    ['GS10', 'TB3MS', 'CPIAUCSL'],  # 10Y Treasury, 3M T-Bill, CPI
    'fred',
    start='2010-01-01'
)
print(fred_data.head())

# === Données de la Banque Mondiale ===
from pandas_datareader import wb

gdp_data = wb.download(
    indicator='NY.GDP.MKTP.CD',  # GDP (current US$)
    country=['US', 'CN', 'JP', 'DE', 'FR'],
    start=2010,
    end=2023
)
print(gdp_data.head())
```

## 2.4 Parsing NASDAQ ITCH Order Flow

```python
"""
NASDAQ ITCH Protocol Parser
===========================
Le protocole ITCH (Integrated Trading and Clearing) est le format de données
brutes du NASDAQ. Il contient TOUS les messages du marché:
- Add Order: Nouvel ordre ajouté au carnet
- Order Executed: Ordre exécuté
- Order Cancel: Ordre annulé
- Trade: Transaction effectuée

Ce parsing est essentiel pour:
1. Reconstruire le carnet d'ordres (order book)
2. Analyser le flux d'ordres (order flow)
3. Détecter les patterns de trading haute fréquence (HFT)
"""
from pathlib import Path
from collections import namedtuple, Counter
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from struct import unpack

# === Définition des types de messages ITCH ===
# Chaque message a un format binaire spécifique

# Format des messages (nom: (format_struct, champs))
message_formats = {
    'S': ('4sHHI', ['event_code']),                    # System Event
    'R': ('4sHHI', ['stock', 'market_category']),     # Stock Directory
    'H': ('4sHHI', ['stock', 'trading_state']),       # Trading Action
    'Y': ('4sHHI', ['stock', 'reg_sho']),             # Reg SHO
    'L': ('4sHHI', ['mpid', 'stock']),                # Market Participant
    'A': ('4sHH6sIQI', ['ref', 'side', 'shares', 'stock', 'price']),  # Add Order
    'F': ('4sHH6sIQI4s', ['ref', 'side', 'shares', 'stock', 'price', 'mpid']),  # Add Order MPID
    'E': ('4sHHQI', ['ref', 'shares', 'match']),      # Order Executed
    'C': ('4sHHQIQ', ['ref', 'shares', 'match', 'printable', 'price']),  # Order Executed Price
    'X': ('4sHHQI', ['ref', 'shares']),               # Order Cancel
    'D': ('4sHHQ', ['ref']),                          # Order Delete
    'U': ('4sHHQQI', ['ref', 'new_ref', 'shares', 'price']),  # Order Replace
    'P': ('4sHHQI6sQQ', ['ref', 'side', 'shares', 'stock', 'price', 'match']),  # Trade
    'Q': ('4sHHQ6sQQ', ['shares', 'stock', 'price', 'match', 'cross']),  # Cross Trade
    'B': ('4sHHQ', ['match']),                        # Broken Trade
    'I': ('4sHH6sIIIQ', ['paired', 'imbalance', 'direction', 'stock', 'far', 'near', 'current']),  # NOII
}

def parse_itch_message(message_type, data):
    """
    Parse un message ITCH binaire.
    
    Args:
        message_type: Type de message (char)
        data: Données binaires
    
    Returns:
        dict: Message parsé avec les champs
    """
    if message_type not in message_formats:
        return None
    
    fmt, fields = message_formats[message_type]
    
    try:
        # Unpack les données binaires selon le format
        values = unpack('>' + fmt, data[:len(fmt)])
        
        # Créer le dictionnaire de résultat
        result = {'message_type': message_type}
        for field, value in zip(fields, values[1:]):  # Skip timestamp
            if isinstance(value, bytes):
                value = value.decode('ascii').strip()
            result[field] = value
        
        return result
    except Exception as e:
        return None


def read_itch_file(filepath, max_messages=None):
    """
    Lit un fichier ITCH et extrait les messages.
    
    Args:
        filepath: Chemin vers le fichier ITCH
        max_messages: Nombre maximum de messages à lire (None = tous)
    
    Returns:
        list: Liste des messages parsés
    """
    messages = []
    
    with open(filepath, 'rb') as f:
        while True:
            # Lire la taille du message (2 bytes, big-endian)
            size_data = f.read(2)
            if len(size_data) < 2:
                break
            
            message_size = unpack('>H', size_data)[0]
            
            # Lire le message
            message_data = f.read(message_size)
            if len(message_data) < message_size:
                break
            
            # Parser le message
            message_type = chr(message_data[0])
            parsed = parse_itch_message(message_type, message_data)
            
            if parsed:
                messages.append(parsed)
            
            if max_messages and len(messages) >= max_messages:
                break
    
    return messages


# === Exemple d'utilisation ===
# messages = read_itch_file('data/01302019.NASDAQ_ITCH50', max_messages=100000)
# df = pd.DataFrame(messages)
# print(df['message_type'].value_counts())
```

## 2.5 Reconstruction du Carnet d'Ordres (Order Book)

```python
"""
Order Book Reconstruction
=========================
Le carnet d'ordres (order book ou LOB - Limit Order Book) représente
l'état du marché à tout instant:
- BID (achat): Ordres d'achat en attente
- ASK (vente): Ordres de vente en attente
- Spread: Différence entre meilleur ask et meilleur bid

Structure:
    ASK (sell orders)      Prix
    ----------------       -----
    100 @ $150.05         150.05  <- Best Ask
    200 @ $150.06         150.06
    150 @ $150.07         150.07
    
    --- SPREAD: $0.03 ---
    
    BID (buy orders)       Prix
    ----------------       -----
    180 @ $150.02         150.02  <- Best Bid
    250 @ $150.01         150.01
    300 @ $150.00         150.00
"""
import pandas as pd
import numpy as np
from collections import defaultdict

class OrderBook:
    """
    Implémentation d'un carnet d'ordres.
    
    Maintient l'état du marché et permet:
    - Ajouter/supprimer des ordres
    - Exécuter des ordres
    - Calculer les métriques de microstructure
    """
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.bids = {}  # {order_ref: {'price': p, 'shares': s, 'timestamp': t}}
        self.asks = {}
        self.trades = []
        
    def add_order(self, order_ref, side, price, shares, timestamp):
        """
        Ajoute un ordre au carnet.
        
        Args:
            order_ref: Référence unique de l'ordre
            side: 'B' (buy/bid) ou 'S' (sell/ask)
            price: Prix en cents (ex: 15002 = $150.02)
            shares: Nombre d'actions
            timestamp: Horodatage
        """
        order = {
            'price': price,
            'shares': shares,
            'timestamp': timestamp
        }
        
        if side == 'B':
            self.bids[order_ref] = order
        else:
            self.asks[order_ref] = order
    
    def cancel_order(self, order_ref, shares_to_cancel):
        """
        Annule partiellement ou totalement un ordre.
        
        Args:
            order_ref: Référence de l'ordre
            shares_to_cancel: Nombre d'actions à annuler
        """
        for book in [self.bids, self.asks]:
            if order_ref in book:
                book[order_ref]['shares'] -= shares_to_cancel
                if book[order_ref]['shares'] <= 0:
                    del book[order_ref]
                return
    
    def delete_order(self, order_ref):
        """Supprime complètement un ordre."""
        for book in [self.bids, self.asks]:
            if order_ref in book:
                del book[order_ref]
                return
    
    def execute_order(self, order_ref, shares_executed, price, timestamp):
        """
        Exécute un ordre (trade).
        
        Args:
            order_ref: Référence de l'ordre exécuté
            shares_executed: Nombre d'actions exécutées
            price: Prix d'exécution
            timestamp: Horodatage
        """
        # Enregistrer le trade
        self.trades.append({
            'order_ref': order_ref,
            'shares': shares_executed,
            'price': price,
            'timestamp': timestamp
        })
        
        # Mettre à jour l'ordre
        for book in [self.bids, self.asks]:
            if order_ref in book:
                book[order_ref]['shares'] -= shares_executed
                if book[order_ref]['shares'] <= 0:
                    del book[order_ref]
                return
    
    def get_best_bid(self):
        """Retourne le meilleur bid (plus haut prix d'achat)."""
        if not self.bids:
            return None, 0
        best = max(self.bids.values(), key=lambda x: x['price'])
        total_shares = sum(o['shares'] for o in self.bids.values() 
                         if o['price'] == best['price'])
        return best['price'], total_shares
    
    def get_best_ask(self):
        """Retourne le meilleur ask (plus bas prix de vente)."""
        if not self.asks:
            return None, 0
        best = min(self.asks.values(), key=lambda x: x['price'])
        total_shares = sum(o['shares'] for o in self.asks.values() 
                         if o['price'] == best['price'])
        return best['price'], total_shares
    
    def get_spread(self):
        """
        Calcule le spread (écart bid-ask).
        
        Returns:
            float: Spread en cents, ou None si non disponible
        """
        bid_price, _ = self.get_best_bid()
        ask_price, _ = self.get_best_ask()
        
        if bid_price is None or ask_price is None:
            return None
        
        return ask_price - bid_price
    
    def get_midprice(self):
        """
        Calcule le prix médian (midprice).
        
        Le midprice est souvent utilisé comme estimation du "vrai" prix.
        
        Returns:
            float: (best_bid + best_ask) / 2
        """
        bid_price, _ = self.get_best_bid()
        ask_price, _ = self.get_best_ask()
        
        if bid_price is None or ask_price is None:
            return None
        
        return (bid_price + ask_price) / 2
    
    def get_depth(self, levels=5):
        """
        Retourne la profondeur du carnet sur N niveaux.
        
        Args:
            levels: Nombre de niveaux de prix à retourner
        
        Returns:
            dict: {'bids': [...], 'asks': [...]}
        """
        # Agréger par niveau de prix
        bid_levels = defaultdict(int)
        ask_levels = defaultdict(int)
        
        for order in self.bids.values():
            bid_levels[order['price']] += order['shares']
        
        for order in self.asks.values():
            ask_levels[order['price']] += order['shares']
        
        # Trier et prendre les N meilleurs niveaux
        sorted_bids = sorted(bid_levels.items(), key=lambda x: -x[0])[:levels]
        sorted_asks = sorted(ask_levels.items(), key=lambda x: x[0])[:levels]
        
        return {
            'bids': [{'price': p, 'shares': s} for p, s in sorted_bids],
            'asks': [{'price': p, 'shares': s} for p, s in sorted_asks]
        }
    
    def get_order_imbalance(self):
        """
        Calcule le déséquilibre d'ordres (order imbalance).
        
        L'imbalance est un indicateur de pression acheteuse/vendeuse:
        - Positif: Plus d'ordres d'achat (bullish)
        - Négatif: Plus d'ordres de vente (bearish)
        
        Returns:
            float: (bid_volume - ask_volume) / (bid_volume + ask_volume)
        """
        bid_volume = sum(o['shares'] for o in self.bids.values())
        ask_volume = sum(o['shares'] for o in self.asks.values())
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0
        
        return (bid_volume - ask_volume) / total


# === Exemple d'utilisation ===
book = OrderBook('AAPL')

# Simuler quelques ordres
book.add_order('O001', 'B', 15002, 100, '09:30:00')  # Buy 100 @ $150.02
book.add_order('O002', 'B', 15001, 200, '09:30:01')  # Buy 200 @ $150.01
book.add_order('O003', 'S', 15005, 150, '09:30:02')  # Sell 150 @ $150.05
book.add_order('O004', 'S', 15006, 100, '09:30:03')  # Sell 100 @ $150.06

print(f"Best Bid: ${book.get_best_bid()[0]/100:.2f}")
print(f"Best Ask: ${book.get_best_ask()[0]/100:.2f}")
print(f"Spread: ${book.get_spread()/100:.4f}")
print(f"Midprice: ${book.get_midprice()/100:.2f}")
print(f"Order Imbalance: {book.get_order_imbalance():.2%}")
print(f"\nDepth:\n{book.get_depth(3)}")
```

## 2.6 SEC EDGAR - Parsing XBRL

```python
"""
SEC EDGAR XBRL Parser
=====================
EDGAR (Electronic Data Gathering, Analysis, and Retrieval) est le système
de la SEC pour collecter les déclarations des entreprises cotées.

XBRL (eXtensible Business Reporting Language) est le format standard
pour les données financières structurées.

Types de filings courants:
- 10-K: Rapport annuel
- 10-Q: Rapport trimestriel  
- 8-K: Événements importants
- 13-F: Holdings des gestionnaires de fonds
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime

class SECEdgarClient:
    """
    Client pour accéder aux données SEC EDGAR.
    """
    
    BASE_URL = "https://www.sec.gov"
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    
    def __init__(self, user_agent):
        """
        Initialise le client.
        
        Args:
            user_agent: Votre email (requis par la SEC)
        """
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate'
        }
    
    def get_company_filings(self, cik, filing_type='10-K', count=10):
        """
        Récupère les filings d'une entreprise.
        
        Args:
            cik: Central Index Key (identifiant SEC)
            filing_type: Type de filing (10-K, 10-Q, 8-K, etc.)
            count: Nombre de filings à récupérer
        
        Returns:
            list: Liste des filings avec métadonnées
        """
        # Formater le CIK (10 digits avec leading zeros)
        cik = str(cik).zfill(10)
        
        # URL de l'API EDGAR
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': filing_type,
            'dateb': '',
            'owner': 'include',
            'count': count,
            'output': 'atom'
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Error fetching filings: {response.status_code}")
        
        # Parser le XML Atom
        soup = BeautifulSoup(response.content, 'xml')
        entries = soup.find_all('entry')
        
        filings = []
        for entry in entries:
            filing = {
                'title': entry.find('title').text if entry.find('title') else None,
                'link': entry.find('link')['href'] if entry.find('link') else None,
                'filing_date': entry.find('filing-date').text if entry.find('filing-date') else None,
                'accession_number': entry.find('accession-number').text if entry.find('accession-number') else None,
            }
            filings.append(filing)
        
        return filings
    
    def get_filing_documents(self, accession_number, cik):
        """
        Récupère la liste des documents d'un filing.
        
        Args:
            accession_number: Numéro d'accession du filing
            cik: CIK de l'entreprise
        
        Returns:
            list: Documents du filing
        """
        cik = str(cik).zfill(10)
        accession_formatted = accession_number.replace('-', '')
        
        url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{accession_formatted}/index.json"
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        return data.get('directory', {}).get('item', [])
    
    def parse_xbrl_facts(self, url):
        """
        Parse les faits XBRL d'un document.
        
        Args:
            url: URL du document XBRL
        
        Returns:
            dict: Faits financiers extraits
        """
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'lxml')
        
        facts = {}
        
        # Éléments financiers courants
        xbrl_elements = [
            'us-gaap:Assets',
            'us-gaap:Liabilities',
            'us-gaap:StockholdersEquity',
            'us-gaap:Revenues',
            'us-gaap:NetIncomeLoss',
            'us-gaap:EarningsPerShareBasic',
            'us-gaap:EarningsPerShareDiluted',
            'us-gaap:CashAndCashEquivalentsAtCarryingValue',
            'us-gaap:LongTermDebt',
            'us-gaap:CommonStockSharesOutstanding',
        ]
        
        for element_name in xbrl_elements:
            # Chercher l'élément (avec ou sans namespace)
            element = soup.find(element_name.lower().replace(':', '_'))
            if element is None:
                element = soup.find(element_name.split(':')[1].lower())
            
            if element:
                try:
                    value = float(element.text.replace(',', ''))
                    facts[element_name] = value
                except ValueError:
                    facts[element_name] = element.text
        
        return facts


def get_financial_ratios(facts):
    """
    Calcule les ratios financiers à partir des faits XBRL.
    
    Args:
        facts: Dictionnaire des faits financiers
    
    Returns:
        dict: Ratios calculés
    """
    ratios = {}
    
    # Current Ratio (liquidité)
    assets = facts.get('us-gaap:Assets', 0)
    liabilities = facts.get('us-gaap:Liabilities', 0)
    
    if liabilities > 0:
        ratios['debt_to_assets'] = liabilities / assets
    
    # Return on Equity (ROE)
    net_income = facts.get('us-gaap:NetIncomeLoss', 0)
    equity = facts.get('us-gaap:StockholdersEquity', 0)
    
    if equity > 0:
        ratios['roe'] = net_income / equity
    
    # Profit Margin
    revenue = facts.get('us-gaap:Revenues', 0)
    if revenue > 0:
        ratios['profit_margin'] = net_income / revenue
    
    return ratios


# === Exemple d'utilisation ===
# client = SECEdgarClient("votre.email@exemple.com")
# 
# # Apple Inc. CIK: 320193
# filings = client.get_company_filings('320193', '10-K', count=5)
# print(f"Derniers 10-K d'Apple: {len(filings)}")
# 
# for f in filings:
#     print(f"  {f['filing_date']}: {f['title']}")
```

## 2.7 Stockage des Données avec HDF5

```python
"""
HDF5 Storage for Financial Data
================================
HDF5 (Hierarchical Data Format) est idéal pour stocker de grandes
quantités de données financières car il offre:
- Compression efficace
- Accès rapide par chunks
- Structure hiérarchique (comme un système de fichiers)
- Support natif par pandas

Structure recommandée:
/prices
    /daily          - Prix OHLCV journaliers
    /minute         - Données minute
/fundamentals
    /quarterly      - Données trimestrielles
    /annual         - Données annuelles
/factors
    /fama_french    - Facteurs FF
    /custom         - Vos propres facteurs
"""
import pandas as pd
import numpy as np
from pathlib import Path

# === Création et écriture ===
DATA_STORE = 'data/assets.h5'

# Créer le fichier HDF5 et stocker des données
with pd.HDFStore(DATA_STORE, mode='w') as store:
    
    # Exemple: stocker des prix
    prices = pd.DataFrame({
        'AAPL': np.random.randn(1000).cumsum() + 150,
        'GOOGL': np.random.randn(1000).cumsum() + 2800,
        'MSFT': np.random.randn(1000).cumsum() + 330,
    }, index=pd.date_range('2020-01-01', periods=1000, freq='D'))
    
    store.put('prices/daily', prices)
    
    # Stocker avec compression
    store.put('prices/compressed', prices, 
              complevel=9,           # Niveau de compression (0-9)
              complib='blosc')       # Algorithme de compression
    
    # Stocker avec format 'table' (permet les requêtes)
    store.put('prices/queryable', prices, format='table')

# === Lecture ===
with pd.HDFStore(DATA_STORE, mode='r') as store:
    # Lire toutes les données
    prices = store['prices/daily']
    print(f"Shape: {prices.shape}")
    
    # Lister les clés
    print(f"Keys: {store.keys()}")
    
    # Requête sur données 'table'
    subset = store.select('prices/queryable', 
                         where='index >= "2020-06-01" and index < "2020-07-01"')
    print(f"Juin 2020: {len(subset)} jours")

# === Multi-Index Storage ===
# Pour stocker des données avec MultiIndex (ticker, date)
def create_multiindex_data():
    """Crée des données avec MultiIndex pour stockage efficace."""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Créer MultiIndex
    idx = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    # Créer DataFrame
    n = len(idx)
    data = pd.DataFrame({
        'open': np.random.randn(n) * 10 + 100,
        'high': np.random.randn(n) * 10 + 105,
        'low': np.random.randn(n) * 10 + 95,
        'close': np.random.randn(n) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, n)
    }, index=idx)
    
    return data

# Stocker et charger des données MultiIndex
# data = create_multiindex_data()
# data.to_hdf(DATA_STORE, 'prices/multiindex', format='table')

# Charger avec sélection
# with pd.HDFStore(DATA_STORE) as store:
#     aapl = store.select('prices/multiindex', 
#                         where="ticker == 'AAPL'")

# === Benchmark des formats de stockage ===
def benchmark_storage_formats(df, filename_base='benchmark'):
    """
    Compare les performances de différents formats de stockage.
    
    Args:
        df: DataFrame à stocker
        filename_base: Préfixe des fichiers
    
    Returns:
        dict: Résultats du benchmark
    """
    import time
    import os
    
    results = {}
    
    # CSV
    start = time.time()
    df.to_csv(f'{filename_base}.csv')
    csv_write = time.time() - start
    
    start = time.time()
    _ = pd.read_csv(f'{filename_base}.csv', index_col=0, parse_dates=True)
    csv_read = time.time() - start
    csv_size = os.path.getsize(f'{filename_base}.csv')
    
    results['csv'] = {
        'write_time': csv_write,
        'read_time': csv_read,
        'file_size': csv_size
    }
    
    # Parquet
    start = time.time()
    df.to_parquet(f'{filename_base}.parquet')
    parquet_write = time.time() - start
    
    start = time.time()
    _ = pd.read_parquet(f'{filename_base}.parquet')
    parquet_read = time.time() - start
    parquet_size = os.path.getsize(f'{filename_base}.parquet')
    
    results['parquet'] = {
        'write_time': parquet_write,
        'read_time': parquet_read,
        'file_size': parquet_size
    }
    
    # HDF5
    start = time.time()
    df.to_hdf(f'{filename_base}.h5', 'data', mode='w', complevel=9)
    hdf5_write = time.time() - start
    
    start = time.time()
    _ = pd.read_hdf(f'{filename_base}.h5', 'data')
    hdf5_read = time.time() - start
    hdf5_size = os.path.getsize(f'{filename_base}.h5')
    
    results['hdf5'] = {
        'write_time': hdf5_write,
        'read_time': hdf5_read,
        'file_size': hdf5_size
    }
    
    # Afficher les résultats
    print("\n" + "="*60)
    print("BENCHMARK STORAGE FORMATS")
    print("="*60)
    print(f"{'Format':<10} {'Write (s)':<12} {'Read (s)':<12} {'Size (MB)':<12}")
    print("-"*60)
    for fmt, res in results.items():
        print(f"{fmt:<10} {res['write_time']:<12.3f} {res['read_time']:<12.3f} "
              f"{res['file_size']/1e6:<12.2f}")
    
    return results
```

---

# 3. DONNÉES ALTERNATIVES
## Alternative Data

## 3.1 Web Scraping avec Scrapy

```python
"""
Web Scraping pour Données Alternatives
======================================
Les données alternatives (alternative data) incluent:
- Données de réservation (OpenTable, Booking)
- Sentiment social media (Twitter, Reddit)
- Trafic web (SimilarWeb)
- Données satellites (parkings, agriculture)
- Earnings calls transcripts

IMPORTANT: Toujours respecter:
1. robots.txt du site
2. Conditions d'utilisation
3. Rate limiting (délai entre requêtes)
"""

# === Scrapy Spider pour OpenTable ===
# Fichier: opentable/spiders/table_spider.py

"""
Spider Scrapy pour OpenTable
Ce spider collecte les données de réservation des restaurants.
"""
import scrapy
from scrapy.loader import ItemLoader
from ..items import RestaurantItem

class OpenTableSpider(scrapy.Spider):
    """
    Spider pour scraper les données OpenTable.
    
    Usage:
        scrapy crawl opentable -o restaurants.json
    """
    name = 'opentable'
    allowed_domains = ['opentable.com']
    
    # URL de départ (page de résultats)
    start_urls = [
        'https://www.opentable.com/new-york-restaurant-listings'
    ]
    
    # Configuration du spider
    custom_settings = {
        'DOWNLOAD_DELAY': 2,              # 2 secondes entre requêtes
        'RANDOMIZE_DOWNLOAD_DELAY': True,  # Randomiser le délai
        'CONCURRENT_REQUESTS': 1,          # Une requête à la fois
        'ROBOTSTXT_OBEY': True,           # Respecter robots.txt
    }
    
    def parse(self, response):
        """
        Parse la page de liste des restaurants.
        
        Args:
            response: Réponse HTTP
        
        Yields:
            dict: Données du restaurant ou Request pour la page suivante
        """
        # Extraire les cartes de restaurants
        restaurant_cards = response.css('div.restaurant-card')
        
        for card in restaurant_cards:
            # Extraire les informations de base
            yield {
                'name': card.css('h2.restaurant-name::text').get(),
                'cuisine': card.css('span.cuisine-type::text').get(),
                'price_range': card.css('span.price-range::text').get(),
                'rating': card.css('span.rating-score::text').get(),
                'reviews_count': card.css('span.review-count::text').get(),
                'neighborhood': card.css('span.neighborhood::text').get(),
                'available_slots': card.css('span.time-slot::text').getall(),
            }
        
        # Pagination - suivre la page suivante
        next_page = response.css('a.pagination-next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)


# === Alternative: Selenium pour sites JavaScript ===
"""
Selenium pour sites dynamiques
Certains sites chargent le contenu via JavaScript,
ce qui nécessite un navigateur headless.
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

def scrape_with_selenium(url, wait_element_id):
    """
    Scrape une page avec Selenium.
    
    Args:
        url: URL à scraper
        wait_element_id: ID de l'élément à attendre avant de scraper
    
    Returns:
        str: HTML de la page
    """
    # Configuration Chrome headless
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/91.0.4472.124 Safari/537.36')
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)
        
        # Attendre que l'élément soit chargé (max 10 secondes)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, wait_element_id))
        )
        
        # Pause supplémentaire pour le JavaScript
        time.sleep(2)
        
        return driver.page_source
    
    finally:
        driver.quit()


# === Scraping des Earnings Calls (SeekingAlpha) ===
"""
ATTENTION: SeekingAlpha bloque maintenant le scraping avec CAPTCHA.
Ce code est fourni à titre éducatif uniquement.
Utilisez des APIs officielles ou des fournisseurs de données.
"""
import requests
from bs4 import BeautifulSoup

def get_earnings_call_transcript(ticker, api_key=None):
    """
    Récupère le transcript d'un earnings call.
    
    Pour une utilisation en production, utilisez:
    - API SeekingAlpha (payante)
    - Refinitiv/Reuters
    - Bloomberg
    - Quandl
    
    Args:
        ticker: Symbol du ticker
        api_key: Clé API (si disponible)
    
    Returns:
        str: Transcript de l'earnings call
    """
    # Exemple avec API fictive
    if api_key:
        url = f"https://api.provider.com/transcripts/{ticker}"
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get(url, headers=headers)
        return response.json().get('transcript', '')
    
    # Sans API - méthode manuelle non recommandée
    print("ATTENTION: Le scraping de SeekingAlpha n'est plus possible.")
    print("Utilisez une API officielle ou un fournisseur de données.")
    return None
```

---

# 4. RECHERCHE DE FACTEURS ALPHA
## Alpha Factor Research

## 4.1 Qu'est-ce qu'un Alpha Factor?

Un **alpha factor** (facteur alpha) est un signal prédictif qui aide à prévoir les rendements futurs d'un actif. L'alpha représente le rendement excédentaire par rapport à un benchmark.

**Types de facteurs:**
- **Value (Valeur)**: P/E, P/B, Dividend Yield
- **Momentum**: Rendements passés, RSI (Relative Strength Index)
- **Quality (Qualité)**: ROE (Return on Equity), dette/capitaux propres
- **Volatility (Volatilité)**: Volatilité historique, beta
- **Size (Taille)**: Market cap, volume

## 4.2 Feature Engineering pour le Trading

```python
"""
Feature Engineering pour Alpha Factors
======================================
Le feature engineering est l'étape la plus importante du ML4T.
Un bon feature engineering peut transformer une stratégie médiocre
en une stratégie rentable.

Catégories de features:
1. Rendements (Returns) - Différentes périodes
2. Volatilité (Volatility) - Risque
3. Momentum - Force de la tendance
4. Mean Reversion - Retour à la moyenne
5. Volume - Liquidité et activité
6. Fondamentaux - Ratios financiers
"""
import pandas as pd
import numpy as np
import talib
from scipy import stats

def compute_returns(prices, periods=[1, 5, 10, 21, 63, 126, 252]):
    """
    Calcule les rendements sur différentes périodes.
    
    Args:
        prices: Series ou DataFrame de prix
        periods: Liste des périodes en jours
    
    Returns:
        DataFrame: Rendements pour chaque période
    
    Exemple:
        >>> prices = pd.Series([100, 101, 102, 103, 104])
        >>> returns = compute_returns(prices, [1, 2])
        >>> print(returns)
           return_1d  return_2d
        0        NaN        NaN
        1      0.010        NaN
        2      0.010      0.020
        3      0.010      0.020
        4      0.010      0.020
    """
    returns = pd.DataFrame(index=prices.index)
    
    for period in periods:
        col_name = f'return_{period}d'
        returns[col_name] = prices.pct_change(period)
    
    return returns


def compute_volatility(prices, windows=[5, 10, 21, 63]):
    """
    Calcule la volatilité (écart-type des rendements) sur différentes fenêtres.
    
    Args:
        prices: Series de prix
        windows: Liste des tailles de fenêtre
    
    Returns:
        DataFrame: Volatilité pour chaque fenêtre
    
    Note:
        La volatilité est annualisée en multipliant par sqrt(252).
    """
    returns = prices.pct_change()
    vol = pd.DataFrame(index=prices.index)
    
    for window in windows:
        col_name = f'vol_{window}d'
        vol[col_name] = returns.rolling(window).std() * np.sqrt(252)
    
    return vol


def compute_momentum_indicators(prices, high=None, low=None, volume=None):
    """
    Calcule les indicateurs de momentum avec TA-Lib.
    
    Args:
        prices: Series de prix de clôture
        high: Series de prix hauts (optionnel)
        low: Series de prix bas (optionnel)
        volume: Series de volume (optionnel)
    
    Returns:
        DataFrame: Indicateurs de momentum
    
    Indicateurs inclus:
        - RSI (Relative Strength Index): Mesure la vitesse des changements de prix
          * > 70: Suracheté (overbought)
          * < 30: Survendu (oversold)
        
        - MACD (Moving Average Convergence Divergence): Différence entre EMA rapide et lente
          * Signal > 0: Momentum haussier
          * Signal < 0: Momentum baissier
        
        - Stochastic: Position du prix dans la range récente
          * > 80: Suracheté
          * < 20: Survendu
        
        - ADX (Average Directional Index): Force de la tendance
          * > 25: Tendance forte
          * < 20: Pas de tendance claire
    """
    close = prices.values
    indicators = pd.DataFrame(index=prices.index)
    
    # RSI - Relative Strength Index
    # Mesure la magnitude des gains récents vs pertes
    indicators['rsi_14'] = talib.RSI(close, timeperiod=14)
    indicators['rsi_7'] = talib.RSI(close, timeperiod=7)
    
    # MACD - Moving Average Convergence Divergence
    macd, macd_signal, macd_hist = talib.MACD(close, 
                                               fastperiod=12, 
                                               slowperiod=26, 
                                               signalperiod=9)
    indicators['macd'] = macd
    indicators['macd_signal'] = macd_signal
    indicators['macd_hist'] = macd_hist
    
    # Williams %R - Similaire au stochastique
    if high is not None and low is not None:
        indicators['willr'] = talib.WILLR(high.values, low.values, close, timeperiod=14)
        
        # Stochastic
        slowk, slowd = talib.STOCH(high.values, low.values, close,
                                    fastk_period=14, slowk_period=3, slowd_period=3)
        indicators['stoch_k'] = slowk
        indicators['stoch_d'] = slowd
        
        # ADX - Average Directional Index
        indicators['adx'] = talib.ADX(high.values, low.values, close, timeperiod=14)
        
        # CCI - Commodity Channel Index
        indicators['cci'] = talib.CCI(high.values, low.values, close, timeperiod=14)
        
        # ATR - Average True Range (volatilité)
        indicators['atr'] = talib.ATR(high.values, low.values, close, timeperiod=14)
    
    # ROC - Rate of Change
    indicators['roc_10'] = talib.ROC(close, timeperiod=10)
    indicators['roc_20'] = talib.ROC(close, timeperiod=20)
    
    # MOM - Momentum
    indicators['mom_10'] = talib.MOM(close, timeperiod=10)
    
    # OBV - On Balance Volume
    if volume is not None:
        indicators['obv'] = talib.OBV(close, volume.values)
    
    return indicators


def compute_mean_reversion_indicators(prices):
    """
    Calcule les indicateurs de mean reversion (retour à la moyenne).
    
    Args:
        prices: Series de prix
    
    Returns:
        DataFrame: Indicateurs de mean reversion
    
    Indicateurs inclus:
        - Bollinger Bands: Bandes de volatilité autour de la moyenne mobile
          * Prix > Upper Band: Suracheté
          * Prix < Lower Band: Survendu
        
        - Z-Score: Nombre d'écarts-types par rapport à la moyenne
          * |Z| > 2: Signal potentiel de retour à la moyenne
        
        - Distance to MA: Écart par rapport à la moyenne mobile
    """
    close = prices.values
    indicators = pd.DataFrame(index=prices.index)
    
    # Bollinger Bands
    for window in [20, 50]:
        upper, middle, lower = talib.BBANDS(close, 
                                             timeperiod=window,
                                             nbdevup=2, 
                                             nbdevdn=2)
        indicators[f'bb_upper_{window}'] = upper
        indicators[f'bb_middle_{window}'] = middle
        indicators[f'bb_lower_{window}'] = lower
        
        # Position dans les bandes (0 = lower, 1 = upper)
        indicators[f'bb_position_{window}'] = (close - lower) / (upper - lower)
    
    # Z-Score
    for window in [10, 20, 50]:
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        indicators[f'zscore_{window}'] = (prices - rolling_mean) / rolling_std
    
    # Distance to Moving Average (en %)
    for window in [10, 20, 50, 200]:
        ma = prices.rolling(window).mean()
        indicators[f'dist_ma_{window}'] = (prices - ma) / ma * 100
    
    # Percentile Rank
    for window in [20, 63, 252]:
        indicators[f'pct_rank_{window}'] = prices.rolling(window).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )
    
    return indicators


def compute_volume_indicators(prices, volume, windows=[5, 10, 20]):
    """
    Calcule les indicateurs basés sur le volume.
    
    Args:
        prices: Series de prix
        volume: Series de volume
        windows: Fenêtres pour les moyennes mobiles
    
    Returns:
        DataFrame: Indicateurs de volume
    
    Indicateurs inclus:
        - Volume Ratio: Volume actuel / Moyenne mobile
          * > 2: Volume anormalement élevé (possible signal)
        
        - VWAP (Volume Weighted Average Price): Prix moyen pondéré par volume
        
        - Money Flow: Flux monétaire entrant/sortant
    """
    indicators = pd.DataFrame(index=prices.index)
    
    # Volume relatif
    for window in windows:
        vol_ma = volume.rolling(window).mean()
        indicators[f'vol_ratio_{window}'] = volume / vol_ma
    
    # Volume trend
    indicators['vol_trend_5'] = volume.rolling(5).mean() / volume.rolling(20).mean()
    
    # Price-Volume Correlation
    indicators['pv_corr_20'] = prices.rolling(20).corr(volume)
    
    # Dollar Volume (proxy de liquidité)
    dollar_vol = prices * volume
    indicators['dollar_vol_ma_20'] = dollar_vol.rolling(20).mean()
    
    # Volume Spike (z-score du volume)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    indicators['vol_zscore'] = (volume - vol_mean) / vol_std
    
    return indicators


class AlphaFactorPipeline:
    """
    Pipeline complet de création de facteurs alpha.
    
    Cette classe encapsule tout le processus de feature engineering
    pour le trading quantitatif.
    
    Usage:
        pipeline = AlphaFactorPipeline()
        features = pipeline.fit_transform(data)
    """
    
    def __init__(self, include_volume=True, include_fundamentals=False):
        """
        Initialise le pipeline.
        
        Args:
            include_volume: Inclure les indicateurs de volume
            include_fundamentals: Inclure les ratios fondamentaux
        """
        self.include_volume = include_volume
        self.include_fundamentals = include_fundamentals
    
    def fit_transform(self, data):
        """
        Génère tous les alpha factors.
        
        Args:
            data: DataFrame avec colonnes 'open', 'high', 'low', 'close', 'volume'
        
        Returns:
            DataFrame: Tous les facteurs calculés
        """
        features = pd.DataFrame(index=data.index)
        
        close = data['close']
        
        # 1. Rendements
        returns = compute_returns(close)
        features = features.join(returns)
        
        # 2. Volatilité
        vol = compute_volatility(close)
        features = features.join(vol)
        
        # 3. Momentum
        momentum = compute_momentum_indicators(
            close,
            high=data.get('high'),
            low=data.get('low'),
            volume=data.get('volume')
        )
        features = features.join(momentum)
        
        # 4. Mean Reversion
        mean_rev = compute_mean_reversion_indicators(close)
        features = features.join(mean_rev)
        
        # 5. Volume
        if self.include_volume and 'volume' in data.columns:
            vol_ind = compute_volume_indicators(close, data['volume'])
            features = features.join(vol_ind)
        
        # 6. Nettoyer les NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        return features


# === Exemple d'utilisation ===
if __name__ == "__main__":
    # Simuler des données
    np.random.seed(42)
    n = 500
    
    data = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 102,
        'low': np.random.randn(n).cumsum() + 98,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, n)
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    # Créer le pipeline
    pipeline = AlphaFactorPipeline()
    features = pipeline.fit_transform(data)
    
    print(f"Nombre de features: {features.shape[1]}")
    print(f"\nFeatures disponibles:")
    print(features.columns.tolist())
```

## 4.3 Débruitage avec Filtre de Kalman et Wavelets

```python
"""
Débruitage des Séries Temporelles Financières
=============================================
Les prix financiers sont bruités. Le débruitage permet d'extraire
le signal sous-jacent pour de meilleures prédictions.

Méthodes:
1. Filtre de Kalman (Kalman Filter)
   - Modèle d'état-espace
   - Estimation optimale en temps réel
   - Adaptatif aux changements de régime

2. Transformée en Ondelettes (Wavelets)
   - Décomposition multi-échelle
   - Conservation des discontinuités
   - Flexible pour différents types de signaux
"""
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import pywt

# === FILTRE DE KALMAN ===
class KalmanSmoother:
    """
    Lissage de séries temporelles avec filtre de Kalman.
    
    Le filtre de Kalman modélise le prix comme un processus d'état caché
    observé avec du bruit. Il estime récursivement l'état réel.
    
    Modèle:
        x(t) = A * x(t-1) + w(t)    # Équation d'état
        y(t) = H * x(t) + v(t)      # Équation d'observation
    
    où:
        x(t): État caché (vrai prix)
        y(t): Observation (prix bruité)
        w(t): Bruit de processus
        v(t): Bruit d'observation
    """
    
    def __init__(self, observation_covariance=1, transition_covariance=0.01):
        """
        Initialise le filtre de Kalman.
        
        Args:
            observation_covariance: Variance du bruit d'observation (plus grand = plus de lissage)
            transition_covariance: Variance du bruit de processus (plus grand = plus réactif)
        """
        self.observation_covariance = observation_covariance
        self.transition_covariance = transition_covariance
        self.kf = None
    
    def fit(self, observations):
        """
        Ajuste le filtre de Kalman aux données.
        
        Args:
            observations: Array de prix observés
        
        Returns:
            self
        """
        n_timesteps = len(observations)
        
        # Définir le filtre de Kalman
        self.kf = KalmanFilter(
            transition_matrices=[1],                          # A: marche aléatoire
            observation_matrices=[1],                         # H: observation directe
            initial_state_mean=observations[0],               # État initial
            initial_state_covariance=1,                       # Incertitude initiale
            observation_covariance=self.observation_covariance,
            transition_covariance=self.transition_covariance
        )
        
        return self
    
    def smooth(self, observations):
        """
        Applique le lissage de Kalman.
        
        Args:
            observations: Array de prix observés
        
        Returns:
            tuple: (état_lissé, covariance)
        """
        if self.kf is None:
            self.fit(observations)
        
        # Lissage (utilise toutes les observations)
        state_means, state_covariances = self.kf.smooth(observations)
        
        return state_means.flatten(), state_covariances.flatten()
    
    def filter(self, observations):
        """
        Applique le filtrage de Kalman (temps réel).
        
        Contrairement au lissage, le filtrage n'utilise que les observations
        passées et présentes (pas de look-ahead).
        
        Args:
            observations: Array de prix observés
        
        Returns:
            tuple: (état_filtré, covariance)
        """
        if self.kf is None:
            self.fit(observations)
        
        # Filtrage (temps réel)
        state_means, state_covariances = self.kf.filter(observations)
        
        return state_means.flatten(), state_covariances.flatten()


# === ONDELETTES (WAVELETS) ===
class WaveletDenoiser:
    """
    Débruitage par ondelettes.
    
    Les ondelettes décomposent le signal en composantes de différentes
    fréquences, permettant de filtrer le bruit haute fréquence tout en
    préservant les discontinuités (changements abrupts).
    
    Ondelettes courantes:
        - 'db4': Daubechies 4 (bon compromis)
        - 'sym8': Symlet 8 (symétrique)
        - 'coif5': Coiflet 5 (moments nuls)
        - 'haar': Haar (simple, discontinuités)
    """
    
    def __init__(self, wavelet='db4', level=None, threshold_mode='soft'):
        """
        Initialise le débruiteur par ondelettes.
        
        Args:
            wavelet: Type d'ondelette ('db4', 'sym8', 'coif5', 'haar')
            level: Niveau de décomposition (None = maximum)
            threshold_mode: 'soft' (doux) ou 'hard' (dur)
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
    
    def denoise(self, signal, threshold_method='universal'):
        """
        Débruite un signal avec les ondelettes.
        
        Args:
            signal: Array du signal à débruiter
            threshold_method: 'universal' (Donoho) ou 'bayesian'
        
        Returns:
            array: Signal débruité
        """
        signal = np.array(signal)
        
        # Déterminer le niveau de décomposition
        if self.level is None:
            self.level = pywt.dwt_max_level(len(signal), self.wavelet)
        
        # Décomposition en ondelettes
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Calculer le seuil
        if threshold_method == 'universal':
            # Seuil universel de Donoho-Johnstone
            # σ * sqrt(2 * log(n))
            sigma = self._estimate_noise(coeffs[-1])
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        else:
            # Seuil adaptatif par niveau
            threshold = None
        
        # Appliquer le seuillage aux coefficients de détail
        denoised_coeffs = [coeffs[0]]  # Garder les coefficients d'approximation
        
        for i, coeff in enumerate(coeffs[1:]):
            if threshold_method == 'universal':
                thresh = threshold
            else:
                # BayesShrink
                sigma = self._estimate_noise(coeff)
                sigma_signal = np.sqrt(max(np.var(coeff) - sigma**2, 0))
                thresh = sigma**2 / sigma_signal if sigma_signal > 0 else np.max(np.abs(coeff))
            
            # Appliquer le seuil
            denoised_coeff = pywt.threshold(coeff, thresh, mode=self.threshold_mode)
            denoised_coeffs.append(denoised_coeff)
        
        # Reconstruction
        denoised_signal = pywt.waverec(denoised_coeffs, self.wavelet)
        
        # Ajuster la longueur (peut différer légèrement)
        return denoised_signal[:len(signal)]
    
    def _estimate_noise(self, detail_coeffs):
        """
        Estime le niveau de bruit à partir des coefficients de détail.
        
        Utilise la MAD (Median Absolute Deviation) qui est robuste aux outliers.
        
        Args:
            detail_coeffs: Coefficients de détail du niveau le plus fin
        
        Returns:
            float: Estimation de sigma
        """
        # MAD / 0.6745 est un estimateur robuste de sigma
        return np.median(np.abs(detail_coeffs)) / 0.6745
    
    def decompose(self, signal, return_details=True):
        """
        Décompose le signal en composantes de différentes échelles.
        
        Args:
            signal: Signal à décomposer
            return_details: Si True, retourne aussi les détails
        
        Returns:
            dict: Composantes à différentes échelles
        """
        signal = np.array(signal)
        
        if self.level is None:
            self.level = pywt.dwt_max_level(len(signal), self.wavelet)
        
        # Décomposition
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Reconstruction par niveau
        result = {}
        
        # Approximation (tendance basse fréquence)
        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        result['approximation'] = pywt.waverec(approx_coeffs, self.wavelet)[:len(signal)]
        
        if return_details:
            # Détails à chaque niveau
            for i in range(1, len(coeffs)):
                detail_coeffs = [np.zeros_like(coeffs[0])]
                for j in range(1, len(coeffs)):
                    if j == i:
                        detail_coeffs.append(coeffs[j])
                    else:
                        detail_coeffs.append(np.zeros_like(coeffs[j]))
                
                result[f'detail_level_{i}'] = pywt.waverec(detail_coeffs, self.wavelet)[:len(signal)]
        
        return result


# === Comparaison des méthodes ===
def compare_denoising_methods(prices, plot=True):
    """
    Compare les méthodes de débruitage.
    
    Args:
        prices: Series de prix
        plot: Si True, affiche un graphique
    
    Returns:
        DataFrame: Prix débruités par chaque méthode
    """
    prices_array = prices.values
    
    results = pd.DataFrame(index=prices.index)
    results['original'] = prices_array
    
    # Filtre de Kalman
    kalman = KalmanSmoother(observation_covariance=0.1, transition_covariance=0.01)
    results['kalman_smooth'], _ = kalman.smooth(prices_array)
    results['kalman_filter'], _ = kalman.filter(prices_array)
    
    # Ondelettes
    wavelet = WaveletDenoiser(wavelet='db4')
    results['wavelet_db4'] = wavelet.denoise(prices_array)
    
    wavelet_sym = WaveletDenoiser(wavelet='sym8')
    results['wavelet_sym8'] = wavelet_sym.denoise(prices_array)
    
    # Moving Average (baseline)
    results['ma_20'] = prices.rolling(20).mean()
    
    # EMA (Exponential Moving Average)
    results['ema_20'] = prices.ewm(span=20).mean()
    
    if plot:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Toutes les méthodes
        ax1 = axes[0]
        ax1.plot(results['original'], alpha=0.5, label='Original', linewidth=0.5)
        ax1.plot(results['kalman_smooth'], label='Kalman Smooth', linewidth=1.5)
        ax1.plot(results['wavelet_db4'], label='Wavelet DB4', linewidth=1.5)
        ax1.plot(results['ma_20'], label='MA 20', linestyle='--')
        ax1.legend()
        ax1.set_title('Comparison of Denoising Methods')
        
        # Plot 2: Zoom sur une période
        zoom_start = len(results) // 2
        zoom_end = zoom_start + 100
        ax2 = axes[1]
        ax2.plot(results['original'].iloc[zoom_start:zoom_end], 
                alpha=0.5, label='Original', linewidth=0.5)
        ax2.plot(results['kalman_smooth'].iloc[zoom_start:zoom_end], 
                label='Kalman', linewidth=2)
        ax2.plot(results['wavelet_db4'].iloc[zoom_start:zoom_end], 
                label='Wavelet', linewidth=2)
        ax2.legend()
        ax2.set_title('Zoomed View')
        
        plt.tight_layout()
        plt.show()
    
    return results


# === Exemple d'utilisation ===
if __name__ == "__main__":
    # Créer un signal synthétique bruité
    np.random.seed(42)
    n = 500
    
    # Signal vrai: tendance + oscillation
    t = np.linspace(0, 10, n)
    true_signal = 100 + 0.5 * t + 5 * np.sin(t)
    
    # Ajouter du bruit
    noise = np.random.randn(n) * 2
    noisy_signal = true_signal + noise
    
    prices = pd.Series(noisy_signal, 
                       index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    # Comparer les méthodes
    results = compare_denoising_methods(prices, plot=False)
    
    # Calculer les erreurs
    print("\nMean Squared Error vs True Signal:")
    print("-" * 40)
    for col in results.columns:
        if col != 'original':
            mse = np.nanmean((results[col].values - true_signal)**2)
            print(f"{col:<20}: {mse:.4f}")
```

---

# 5. ÉVALUATION DE STRATÉGIE
## Strategy Evaluation

## 5.1 Métriques de Performance

```python
"""
Métriques de Performance pour Stratégies de Trading
===================================================
Ces métriques permettent d'évaluer objectivement une stratégie:

1. Rendement (Return): Performance absolue
2. Risque (Risk): Volatilité, drawdown
3. Ratio risque/rendement: Sharpe, Sortino, Calmar
4. Stabilité: Consistance des rendements
"""
import numpy as np
import pandas as pd
from scipy import stats

def calculate_returns(prices):
    """
    Calcule les rendements à partir des prix.
    
    Args:
        prices: Series ou array de prix
    
    Returns:
        Series: Rendements journaliers
    """
    return pd.Series(prices).pct_change().dropna()


def annualized_return(returns, periods_per_year=252):
    """
    Calcule le rendement annualisé.
    
    Formule: (1 + rendement_moyen)^252 - 1
    
    Args:
        returns: Series de rendements journaliers
        periods_per_year: Nombre de périodes par an (252 jours de trading)
    
    Returns:
        float: Rendement annualisé
    
    Exemple:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        >>> ann_ret = annualized_return(returns)
        >>> print(f"Rendement annualisé: {ann_ret:.2%}")
    """
    total_return = (1 + returns).prod()
    n_periods = len(returns)
    return total_return ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns, periods_per_year=252):
    """
    Calcule la volatilité annualisée.
    
    Formule: σ_journalier * √252
    
    Args:
        returns: Series de rendements journaliers
        periods_per_year: Nombre de périodes par an
    
    Returns:
        float: Volatilité annualisée
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calcule le ratio de Sharpe.
    
    Le Sharpe mesure le rendement excédentaire par unité de risque:
    Sharpe = (Rendement - Taux sans risque) / Volatilité
    
    Interprétation:
        < 1.0  : Sous-performance
        1.0-2.0: Acceptable
        2.0-3.0: Très bon
        > 3.0  : Excellent (ou suspect!)
    
    Args:
        returns: Series de rendements
        risk_free_rate: Taux sans risque annuel
        periods_per_year: Nombre de périodes par an
    
    Returns:
        float: Ratio de Sharpe
    """
    # Convertir le taux sans risque en journalier
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Rendement excédentaire
    excess_returns = returns - rf_per_period
    
    # Sharpe annualisé
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calcule le ratio de Sortino.
    
    Similaire au Sharpe mais ne pénalise que la volatilité négative
    (downside risk), pas la volatilité positive.
    
    Sortino = (Rendement - Taux sans risque) / Downside Deviation
    
    Args:
        returns: Series de rendements
        risk_free_rate: Taux sans risque annuel
        periods_per_year: Nombre de périodes par an
    
    Returns:
        float: Ratio de Sortino
    """
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    
    # Downside deviation: écart-type des rendements négatifs seulement
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.inf
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def max_drawdown(prices):
    """
    Calcule le maximum drawdown (perte maximale depuis un pic).
    
    Le drawdown mesure la perte maximale qu'un investisseur aurait
    subie en entrant au pire moment.
    
    Args:
        prices: Series de prix ou valeur du portefeuille
    
    Returns:
        float: Maximum drawdown (négatif)
    
    Exemple:
        >>> prices = pd.Series([100, 110, 90, 95, 120, 100])
        >>> mdd = max_drawdown(prices)
        >>> print(f"Max Drawdown: {mdd:.2%}")  # -18.18%
    """
    prices = pd.Series(prices)
    
    # Pic cumulatif
    cumulative_max = prices.cummax()
    
    # Drawdown à chaque point
    drawdown = (prices - cumulative_max) / cumulative_max
    
    return drawdown.min()


def calmar_ratio(returns, prices=None, periods_per_year=252):
    """
    Calcule le ratio de Calmar.
    
    Calmar = Rendement annualisé / |Max Drawdown|
    
    Plus le Calmar est élevé, meilleur est le rendement par rapport
    au risque de perte maximale.
    
    Args:
        returns: Series de rendements
        prices: Series de prix (optionnel, calculé si non fourni)
        periods_per_year: Nombre de périodes par an
    
    Returns:
        float: Ratio de Calmar
    """
    ann_ret = annualized_return(returns, periods_per_year)
    
    if prices is None:
        prices = (1 + returns).cumprod()
    
    mdd = abs(max_drawdown(prices))
    
    if mdd == 0:
        return np.inf
    
    return ann_ret / mdd


def information_ratio(returns, benchmark_returns):
    """
    Calcule le ratio d'information.
    
    IR = Alpha / Tracking Error
    
    Mesure la performance ajustée au risque par rapport à un benchmark.
    
    Args:
        returns: Series de rendements de la stratégie
        benchmark_returns: Series de rendements du benchmark
    
    Returns:
        float: Ratio d'information
    """
    # Rendement actif (différence avec le benchmark)
    active_returns = returns - benchmark_returns
    
    # Tracking error (volatilité du rendement actif)
    tracking_error = active_returns.std() * np.sqrt(252)
    
    if tracking_error == 0:
        return np.inf
    
    return (active_returns.mean() * 252) / tracking_error


def win_rate(returns):
    """
    Calcule le taux de gains.
    
    Args:
        returns: Series de rendements
    
    Returns:
        float: Pourcentage de périodes positives
    """
    return (returns > 0).mean()


def profit_factor(returns):
    """
    Calcule le profit factor.
    
    Profit Factor = Somme des gains / |Somme des pertes|
    
    Args:
        returns: Series de rendements
    
    Returns:
        float: Profit factor (> 1 = profitable)
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf
    
    return gains / losses


def calculate_all_metrics(returns, prices=None, benchmark_returns=None, 
                          risk_free_rate=0.02):
    """
    Calcule toutes les métriques de performance.
    
    Args:
        returns: Series de rendements
        prices: Series de prix (optionnel)
        benchmark_returns: Series de rendements du benchmark (optionnel)
        risk_free_rate: Taux sans risque annuel
    
    Returns:
        dict: Toutes les métriques
    """
    if prices is None:
        prices = (1 + returns).cumprod() * 100  # Partir de 100
    
    metrics = {
        # Rendement
        'total_return': (prices.iloc[-1] / prices.iloc[0]) - 1,
        'annualized_return': annualized_return(returns),
        'cagr': (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(returns)) - 1,
        
        # Risque
        'annualized_volatility': annualized_volatility(returns),
        'max_drawdown': max_drawdown(prices),
        'var_95': returns.quantile(0.05),  # Value at Risk 95%
        'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),  # Conditional VaR
        
        # Ratios
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate),
        'calmar_ratio': calmar_ratio(returns, prices),
        
        # Trading
        'win_rate': win_rate(returns),
        'profit_factor': profit_factor(returns),
        'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
        'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
        
        # Distribution
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
    }
    
    if benchmark_returns is not None:
        metrics['information_ratio'] = information_ratio(returns, benchmark_returns)
        metrics['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
        metrics['alpha'] = metrics['annualized_return'] - metrics['beta'] * annualized_return(benchmark_returns)
    
    return metrics


# === Affichage des métriques ===
def print_performance_report(metrics):
    """
    Affiche un rapport de performance formaté.
    
    Args:
        metrics: dict retourné par calculate_all_metrics
    """
    print("\n" + "="*60)
    print("RAPPORT DE PERFORMANCE / PERFORMANCE REPORT")
    print("="*60)
    
    print("\n📈 RENDEMENT / RETURN")
    print("-"*40)
    print(f"  Rendement total      : {metrics['total_return']:>10.2%}")
    print(f"  Rendement annualisé  : {metrics['annualized_return']:>10.2%}")
    print(f"  CAGR                 : {metrics['cagr']:>10.2%}")
    
    print("\n📉 RISQUE / RISK")
    print("-"*40)
    print(f"  Volatilité annualisée: {metrics['annualized_volatility']:>10.2%}")
    print(f"  Max Drawdown         : {metrics['max_drawdown']:>10.2%}")
    print(f"  VaR 95%              : {metrics['var_95']:>10.2%}")
    print(f"  CVaR 95%             : {metrics['cvar_95']:>10.2%}")
    
    print("\n📊 RATIOS")
    print("-"*40)
    print(f"  Sharpe Ratio         : {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio        : {metrics['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio         : {metrics['calmar_ratio']:>10.2f}")
    
    print("\n🎯 TRADING")
    print("-"*40)
    print(f"  Win Rate             : {metrics['win_rate']:>10.2%}")
    print(f"  Profit Factor        : {metrics['profit_factor']:>10.2f}")
    print(f"  Gain moyen           : {metrics['avg_win']:>10.2%}")
    print(f"  Perte moyenne        : {metrics['avg_loss']:>10.2%}")
    
    if 'information_ratio' in metrics:
        print("\n📎 VS BENCHMARK")
        print("-"*40)
        print(f"  Information Ratio    : {metrics['information_ratio']:>10.2f}")
        print(f"  Beta                 : {metrics['beta']:>10.2f}")
        print(f"  Alpha                : {metrics['alpha']:>10.2%}")
    
    print("\n" + "="*60)
```

## 5.2 Backtesting Vectorisé

```python
"""
Backtesting Vectorisé
=====================
Le backtesting vectorisé utilise les opérations pandas/numpy pour
simuler une stratégie sur données historiques.

Avantages:
- Très rapide (pas de boucles)
- Facile à implémenter
- Bon pour le prototypage

Inconvénients:
- Pas de gestion des ordres complexes
- Pas de slippage réaliste
- Look-ahead bias potentiel
"""
import numpy as np
import pandas as pd

class VectorizedBacktest:
    """
    Backtester vectorisé simple.
    
    Cette classe permet de tester rapidement des stratégies basées
    sur des signaux.
    """
    
    def __init__(self, prices, signals, initial_capital=100000, 
                 transaction_cost=0.001):
        """
        Initialise le backtester.
        
        Args:
            prices: Series de prix
            signals: Series de signaux (-1, 0, 1) pour short/flat/long
            initial_capital: Capital initial
            transaction_cost: Coût de transaction (0.1% = 0.001)
        """
        self.prices = prices
        self.signals = signals
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Résultats
        self.positions = None
        self.portfolio_value = None
        self.returns = None
    
    def run(self):
        """
        Exécute le backtest.
        
        Returns:
            DataFrame: Résultats du backtest
        """
        # Aligner les données
        prices = self.prices.copy()
        signals = self.signals.reindex(prices.index).fillna(0)
        
        # Positions: signal décalé d'un jour (on trade le lendemain du signal)
        self.positions = signals.shift(1).fillna(0)
        
        # Rendements du marché
        market_returns = prices.pct_change()
        
        # Rendements de la stratégie (position * rendement marché)
        strategy_returns = self.positions * market_returns
        
        # Coûts de transaction
        # On paie quand la position change
        position_changes = self.positions.diff().abs()
        costs = position_changes * self.transaction_cost
        
        # Rendements nets
        self.returns = strategy_returns - costs
        
        # Valeur du portefeuille
        self.portfolio_value = self.initial_capital * (1 + self.returns).cumprod()
        
        # Créer le DataFrame de résultats
        results = pd.DataFrame({
            'price': prices,
            'signal': signals,
            'position': self.positions,
            'market_return': market_returns,
            'strategy_return': strategy_returns,
            'costs': costs,
            'net_return': self.returns,
            'portfolio_value': self.portfolio_value
        })
        
        return results
    
    def get_metrics(self):
        """
        Calcule les métriques de performance.
        
        Returns:
            dict: Métriques de performance
        """
        if self.returns is None:
            self.run()
        
        return calculate_all_metrics(self.returns, self.portfolio_value)
    
    def plot_results(self, benchmark_prices=None):
        """
        Affiche les résultats du backtest.
        
        Args:
            benchmark_prices: Series de prix du benchmark (optionnel)
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. Valeur du portefeuille
        ax1 = axes[0]
        ax1.plot(self.portfolio_value, label='Strategy', linewidth=2)
        
        if benchmark_prices is not None:
            benchmark_value = self.initial_capital * (benchmark_prices / benchmark_prices.iloc[0])
            ax1.plot(benchmark_value, label='Benchmark', linewidth=2, alpha=0.7)
        
        ax1.set_title('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[1]
        cummax = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - cummax) / cummax
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.5, color='red')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # 3. Positions
        ax3 = axes[2]
        ax3.plot(self.positions, label='Position', drawstyle='steps-post')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax3.set_title('Positions')
        ax3.set_ylim(-1.5, 1.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# === Stratégies de base ===
def moving_average_crossover(prices, fast_window=10, slow_window=50):
    """
    Stratégie de croisement de moyennes mobiles.
    
    Signal LONG quand MA rapide > MA lente (tendance haussière)
    Signal SHORT quand MA rapide < MA lente (tendance baissière)
    
    Args:
        prices: Series de prix
        fast_window: Fenêtre de la MA rapide
        slow_window: Fenêtre de la MA lente
    
    Returns:
        Series: Signaux (-1, 0, 1)
    """
    fast_ma = prices.rolling(fast_window).mean()
    slow_ma = prices.rolling(slow_window).mean()
    
    signals = pd.Series(0, index=prices.index)
    signals[fast_ma > slow_ma] = 1    # Long
    signals[fast_ma < slow_ma] = -1   # Short
    
    return signals


def mean_reversion(prices, window=20, threshold=2):
    """
    Stratégie de retour à la moyenne.
    
    SHORT quand le prix est trop au-dessus de la moyenne (surachat)
    LONG quand le prix est trop en-dessous de la moyenne (survente)
    
    Args:
        prices: Series de prix
        window: Fenêtre pour la moyenne et écart-type
        threshold: Nombre d'écarts-types pour déclencher un signal
    
    Returns:
        Series: Signaux (-1, 0, 1)
    """
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    
    z_score = (prices - rolling_mean) / rolling_std
    
    signals = pd.Series(0, index=prices.index)
    signals[z_score > threshold] = -1   # Short (surachat)
    signals[z_score < -threshold] = 1   # Long (survente)
    
    return signals


def momentum_strategy(prices, lookback=20, top_pct=0.2):
    """
    Stratégie momentum.
    
    LONG sur les actifs avec les meilleurs rendements passés.
    
    Args:
        prices: Series de prix
        lookback: Période de calcul du momentum
        top_pct: Pourcentage des meilleurs (ex: 0.2 = top 20%)
    
    Returns:
        Series: Signaux (0, 1)
    """
    returns = prices.pct_change(lookback)
    
    signals = pd.Series(0, index=prices.index)
    signals[returns > returns.quantile(1 - top_pct)] = 1
    
    return signals


# === Exemple complet ===
def run_backtest_example():
    """Exemple complet de backtest."""
    import yfinance as yf
    
    # Télécharger les données
    ticker = "SPY"
    data = yf.download(ticker, start="2018-01-01", end="2023-12-31")
    prices = data['Adj Close']
    
    # Créer les signaux
    signals = moving_average_crossover(prices, fast_window=20, slow_window=50)
    
    # Exécuter le backtest
    bt = VectorizedBacktest(prices, signals, transaction_cost=0.001)
    results = bt.run()
    
    # Afficher les métriques
    metrics = bt.get_metrics()
    print_performance_report(metrics)
    
    # Afficher les graphiques
    bt.plot_results(benchmark_prices=prices)
    
    return results


# Pour exécuter:
# results = run_backtest_example()
```

## 5.3 Optimisation de Portefeuille Mean-Variance

```python
"""
Optimisation de Portefeuille Mean-Variance (Markowitz)
======================================================
La théorie moderne du portefeuille (MPT - Modern Portfolio Theory) de
Harry Markowitz (1952) cherche à maximiser le rendement pour un niveau
de risque donné, ou minimiser le risque pour un rendement cible.

Concepts clés:
- Frontière efficiente: Ensemble des portefeuilles optimaux
- Portefeuille tangent: Meilleur ratio Sharpe
- Diversification: Réduction du risque non-systématique
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class MeanVarianceOptimizer:
    """
    Optimiseur Mean-Variance (Markowitz).
    
    Trouve les poids optimaux pour un portefeuille d'actifs.
    """
    
    def __init__(self, returns, risk_free_rate=0.02):
        """
        Initialise l'optimiseur.
        
        Args:
            returns: DataFrame de rendements (colonnes = actifs)
            risk_free_rate: Taux sans risque annuel
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        
        # Calculer les statistiques
        self.mean_returns = returns.mean() * 252  # Annualisé
        self.cov_matrix = returns.cov() * 252     # Annualisé
        self.n_assets = len(returns.columns)
        self.asset_names = returns.columns.tolist()
    
    def portfolio_return(self, weights):
        """
        Calcule le rendement attendu du portefeuille.
        
        Args:
            weights: Array de poids
        
        Returns:
            float: Rendement annualisé
        """
        return np.dot(weights, self.mean_returns)
    
    def portfolio_volatility(self, weights):
        """
        Calcule la volatilité du portefeuille.
        
        Args:
            weights: Array de poids
        
        Returns:
            float: Volatilité annualisée
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe(self, weights):
        """
        Calcule le ratio de Sharpe du portefeuille.
        
        Args:
            weights: Array de poids
        
        Returns:
            float: Ratio de Sharpe
        """
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol
    
    def optimize_sharpe(self, allow_short=False):
        """
        Trouve le portefeuille avec le meilleur ratio de Sharpe.
        
        Args:
            allow_short: Autoriser les positions short
        
        Returns:
            dict: Résultat de l'optimisation
        """
        # Fonction objectif: minimiser -Sharpe (pour maximiser Sharpe)
        def neg_sharpe(weights):
            return -self.portfolio_sharpe(weights)
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Somme = 1
        ]
        
        # Bornes
        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Point de départ: équipondéré
        init_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimisation
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        return {
            'weights': dict(zip(self.asset_names, optimal_weights)),
            'return': self.portfolio_return(optimal_weights),
            'volatility': self.portfolio_volatility(optimal_weights),
            'sharpe': self.portfolio_sharpe(optimal_weights)
        }
    
    def optimize_min_volatility(self, target_return=None):
        """
        Trouve le portefeuille de variance minimale.
        
        Args:
            target_return: Rendement cible (optionnel)
        
        Returns:
            dict: Résultat de l'optimisation
        """
        # Fonction objectif: minimiser la volatilité
        def volatility(weights):
            return self.portfolio_volatility(weights)
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: self.portfolio_return(x) - target_return
            })
        
        # Bornes (long only)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Point de départ
        init_weights = np.ones(self.n_assets) / self.n_assets
        
        # Optimisation
        result = minimize(
            volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        return {
            'weights': dict(zip(self.asset_names, optimal_weights)),
            'return': self.portfolio_return(optimal_weights),
            'volatility': self.portfolio_volatility(optimal_weights),
            'sharpe': self.portfolio_sharpe(optimal_weights)
        }
    
    def efficient_frontier(self, n_points=50):
        """
        Calcule la frontière efficiente.
        
        Args:
            n_points: Nombre de points sur la frontière
        
        Returns:
            DataFrame: Points de la frontière efficiente
        """
        # Range de rendements cibles
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        
        for target in target_returns:
            try:
                result = self.optimize_min_volatility(target_return=target)
                frontier.append({
                    'target_return': target,
                    'return': result['return'],
                    'volatility': result['volatility'],
                    'sharpe': result['sharpe']
                })
            except:
                continue
        
        return pd.DataFrame(frontier)
    
    def plot_efficient_frontier(self, n_points=50, show_assets=True):
        """
        Affiche la frontière efficiente.
        
        Args:
            n_points: Nombre de points sur la frontière
            show_assets: Afficher les actifs individuels
        """
        # Calculer la frontière
        frontier = self.efficient_frontier(n_points)
        
        # Portefeuille optimal (max Sharpe)
        optimal = self.optimize_sharpe()
        
        # Portefeuille min variance
        min_vol = self.optimize_min_volatility()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Frontière efficiente
        ax.plot(frontier['volatility'], frontier['return'], 
                'b-', linewidth=2, label='Efficient Frontier')
        
        # Portefeuille optimal
        ax.scatter(optimal['volatility'], optimal['return'], 
                  marker='*', s=300, c='red', label='Max Sharpe')
        
        # Portefeuille min variance
        ax.scatter(min_vol['volatility'], min_vol['return'], 
                  marker='o', s=200, c='green', label='Min Volatility')
        
        # Actifs individuels
        if show_assets:
            for i, asset in enumerate(self.asset_names):
                ret = self.mean_returns.iloc[i]
                vol = np.sqrt(self.cov_matrix.iloc[i, i])
                ax.scatter(vol, ret, s=100, alpha=0.7)
                ax.annotate(asset, (vol, ret), fontsize=10)
        
        # Capital Market Line (CML)
        max_sharpe_ret = optimal['return']
        max_sharpe_vol = optimal['volatility']
        cml_x = np.linspace(0, max_sharpe_vol * 1.5, 100)
        cml_y = self.risk_free_rate + (max_sharpe_ret - self.risk_free_rate) / max_sharpe_vol * cml_x
        ax.plot(cml_x, cml_y, 'r--', label='Capital Market Line')
        
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Expected Return (Annualized)')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


# === Kelly Criterion ===
def kelly_criterion(win_prob, win_loss_ratio):
    """
    Calcule la fraction Kelly optimale.
    
    La formule Kelly détermine la fraction optimale du capital à risquer
    pour maximiser la croissance à long terme.
    
    f* = (p * b - q) / b
    
    où:
        p = probabilité de gain
        q = probabilité de perte (1 - p)
        b = ratio gain/perte
    
    Args:
        win_prob: Probabilité de gain (ex: 0.55)
        win_loss_ratio: Ratio gain moyen / perte moyenne (ex: 1.5)
    
    Returns:
        float: Fraction Kelly (ex: 0.2 = 20% du capital)
    """
    q = 1 - win_prob
    kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
    return max(0, kelly)  # Ne pas retourner de valeur négative


def half_kelly(win_prob, win_loss_ratio):
    """
    Calcule le demi-Kelly (plus conservateur).
    
    En pratique, on utilise souvent une fraction du Kelly (1/2, 1/4)
    car la formule suppose des paramètres parfaitement connus.
    
    Args:
        win_prob: Probabilité de gain
        win_loss_ratio: Ratio gain/perte
    
    Returns:
        float: Demi-fraction Kelly
    """
    return kelly_criterion(win_prob, win_loss_ratio) / 2


# === Exemple ===
def run_portfolio_optimization_example():
    """Exemple d'optimisation de portefeuille."""
    import yfinance as yf
    
    # Télécharger les données
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JPM', 'JNJ', 'XOM', 'PG']
    data = yf.download(tickers, start='2019-01-01', end='2023-12-31')['Adj Close']
    
    # Calculer les rendements
    returns = data.pct_change().dropna()
    
    # Créer l'optimiseur
    optimizer = MeanVarianceOptimizer(returns)
    
    # Optimiser
    print("="*60)
    print("PORTEFEUILLE OPTIMAL (MAX SHARPE)")
    print("="*60)
    optimal = optimizer.optimize_sharpe()
    print(f"\nRendement: {optimal['return']:.2%}")
    print(f"Volatilité: {optimal['volatility']:.2%}")
    print(f"Sharpe Ratio: {optimal['sharpe']:.2f}")
    print("\nPoids:")
    for asset, weight in sorted(optimal['weights'].items(), key=lambda x: -x[1]):
        if abs(weight) > 0.01:
            print(f"  {asset}: {weight:.2%}")
    
    print("\n" + "="*60)
    print("PORTEFEUILLE MIN VARIANCE")
    print("="*60)
    min_vol = optimizer.optimize_min_volatility()
    print(f"\nRendement: {min_vol['return']:.2%}")
    print(f"Volatilité: {min_vol['volatility']:.2%}")
    print(f"Sharpe Ratio: {min_vol['sharpe']:.2f}")
    
    # Afficher la frontière efficiente
    optimizer.plot_efficient_frontier()
    
    return optimizer


# Pour exécuter:
# optimizer = run_portfolio_optimization_example()
```

---

# 6. PROCESSUS MACHINE LEARNING
## ML Workflow

## 6.1 Cross-Validation pour Séries Temporelles

```python
"""
Cross-Validation pour Séries Temporelles Financières
====================================================
La cross-validation standard (K-Fold) ne fonctionne PAS pour les séries
temporelles car elle crée un look-ahead bias (utiliser des données futures
pour prédire le passé).

Solutions:
1. TimeSeriesSplit: Validation glissante
2. Walk-Forward: Ré-entraînement périodique
3. Purged K-Fold: Avec gap entre train et test
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

class TimeSeriesSplitCustom(BaseCrossValidator):
    """
    Split temporel personnalisé avec gap et embargo.
    
    Le gap empêche le look-ahead bias en créant un tampon entre
    les données d'entraînement et de test.
    
    Exemple visuel:
    
    |------- Train -------|  Gap  |--- Test ---|
    |=====================|       |============|
    t0                   t1       t2           t3
    """
    
    def __init__(self, n_splits=5, train_period_length=252, 
                 test_period_length=63, gap=5):
        """
        Initialise le splitter.
        
        Args:
            n_splits: Nombre de splits
            train_period_length: Taille de la période d'entraînement (en jours)
            test_period_length: Taille de la période de test
            gap: Nombre de jours entre train et test (pour éviter look-ahead)
        """
        self.n_splits = n_splits
        self.train_length = train_period_length
        self.test_length = test_period_length
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        """
        Génère les indices train/test.
        
        Args:
            X: Features
            y: Target (optionnel)
            groups: Groupes (optionnel)
        
        Yields:
            tuple: (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculer la taille totale nécessaire par split
        total_per_split = self.train_length + self.gap + self.test_length
        
        # Point de départ pour le premier split
        # On part de la fin et on recule
        for i in range(self.n_splits):
            # Fin du test = n_samples - i * test_length
            test_end = n_samples - i * self.test_length
            test_start = test_end - self.test_length
            
            # Gap
            train_end = test_start - self.gap
            train_start = train_end - self.train_length
            
            if train_start < 0:
                break
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class WalkForwardValidator:
    """
    Walk-Forward Validation avec ré-entraînement.
    
    Le modèle est ré-entraîné à chaque étape avec les nouvelles données,
    simulant une utilisation en temps réel.
    
    Schéma:
    Step 1: [====Train====]  [Test]
    Step 2:     [====Train====]  [Test]
    Step 3:         [====Train====]  [Test]
    """
    
    def __init__(self, train_window=252, test_window=21, 
                 step_size=21, expanding=False):
        """
        Initialise le validateur.
        
        Args:
            train_window: Taille de la fenêtre d'entraînement
            test_window: Taille de la fenêtre de test
            step_size: Pas entre chaque split
            expanding: Si True, la fenêtre d'entraînement grandit
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.expanding = expanding
    
    def split(self, X):
        """
        Génère les splits.
        
        Args:
            X: Features
        
        Yields:
            tuple: (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Premier point de départ
        train_start = 0
        train_end = self.train_window
        
        while train_end + self.test_window <= n_samples:
            test_start = train_end
            test_end = test_start + self.test_window
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
            
            # Avancer
            if not self.expanding:
                train_start += self.step_size
            train_end += self.step_size


class MultipleTimeSeriesCV:
    """
    Cross-validation pour données multi-actifs (panel data).
    
    Gère correctement les données avec MultiIndex (date, ticker).
    Évite le look-ahead bias en purgeant les observations qui chevauchent.
    """
    
    def __init__(self, n_splits=3, train_period_length=126,
                 test_period_length=21, lookahead=None,
                 date_idx='date', shuffle=False):
        """
        Initialise le validateur.
        
        Args:
            n_splits: Nombre de splits
            train_period_length: Taille de la période d'entraînement
            test_period_length: Taille de la période de test
            lookahead: Horizon de prédiction (pour purging)
            date_idx: Nom de l'index de date
            shuffle: Mélanger les données d'entraînement
        """
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx
    
    def split(self, X, y=None, groups=None):
        """
        Génère les splits pour données panel.
        
        Args:
            X: DataFrame avec MultiIndex (date, ticker)
        
        Yields:
            tuple: (train_indices, test_indices)
        """
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + (self.lookahead or 0) - 1
            train_start_idx = train_end_idx + self.train_length + (self.lookahead or 0) - 1
            split_idx.append([train_start_idx, train_end_idx,
                             test_start_idx, test_end_idx])
        
        dates = X.reset_index()[[self.date_idx]]
        
        for train_start, train_end, test_start, test_end in split_idx:
            if train_start >= len(days):
                continue
                
            train_idx = dates[(dates[self.date_idx] > days[train_start])
                             & (dates[self.date_idx] <= days[train_end])].index
            test_idx = dates[(dates[self.date_idx] > days[test_start])
                            & (dates[self.date_idx] <= days[test_end])].index
            
            if self.shuffle:
                np.random.shuffle(train_idx.to_numpy())
            
            yield train_idx.to_numpy(), test_idx.to_numpy()
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# === Exemple d'utilisation ===
def demonstrate_cv():
    """Démontre les différentes méthodes de cross-validation."""
    
    # Créer des données
    n_samples = 500
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)
    
    print("TimeSeriesSplit Custom")
    print("-" * 40)
    cv = TimeSeriesSplitCustom(n_splits=3, train_period_length=200, 
                                test_period_length=50, gap=5)
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"Split {i+1}:")
        print(f"  Train: {train_idx[0]} - {train_idx[-1]} ({len(train_idx)} samples)")
        print(f"  Test:  {test_idx[0]} - {test_idx[-1]} ({len(test_idx)} samples)")
    
    print("\nWalk-Forward Validation")
    print("-" * 40)
    wf = WalkForwardValidator(train_window=200, test_window=20, step_size=20)
    
    splits = list(wf.split(X))
    print(f"Nombre de splits: {len(splits)}")
    print(f"Premier split - Train: {splits[0][0][0]}-{splits[0][0][-1]}, "
          f"Test: {splits[0][1][0]}-{splits[0][1][-1]}")
    print(f"Dernier split - Train: {splits[-1][0][0]}-{splits[-1][0][-1]}, "
          f"Test: {splits[-1][1][0]}-{splits[-1][1][-1]}")


# demonstrate_cv()
```

## 6.2 Information Mutuelle pour Sélection de Features

```python
"""
Information Mutuelle pour Sélection de Features
===============================================
L'information mutuelle (MI) mesure la dépendance entre deux variables.
Contrairement à la corrélation, elle capture les relations non-linéaires.

MI(X, Y) = 0: X et Y sont indépendants
MI(X, Y) > 0: X contient de l'information sur Y
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

def calculate_mutual_information(X, y, task='regression', n_neighbors=3):
    """
    Calcule l'information mutuelle entre features et target.
    
    Args:
        X: DataFrame de features
        y: Series cible
        task: 'regression' ou 'classification'
        n_neighbors: Nombre de voisins pour l'estimation
    
    Returns:
        Series: MI pour chaque feature, triée décroissante
    """
    if task == 'regression':
        mi_scores = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)
    else:
        mi_scores = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=42)
    
    mi_series = pd.Series(mi_scores, index=X.columns)
    return mi_series.sort_values(ascending=False)


def select_features_by_mi(X, y, n_features=10, task='regression'):
    """
    Sélectionne les meilleures features par information mutuelle.
    
    Args:
        X: DataFrame de features
        y: Series cible
        n_features: Nombre de features à sélectionner
        task: 'regression' ou 'classification'
    
    Returns:
        list: Noms des features sélectionnées
    """
    mi_scores = calculate_mutual_information(X, y, task)
    return mi_scores.head(n_features).index.tolist()


def plot_mutual_information(mi_scores, top_n=20):
    """
    Affiche les scores d'information mutuelle.
    
    Args:
        mi_scores: Series de MI scores
        top_n: Nombre de features à afficher
    """
    import matplotlib.pyplot as plt
    
    top_scores = mi_scores.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_scores.plot(kind='barh', ax=ax)
    ax.set_xlabel('Mutual Information')
    ax.set_title(f'Top {top_n} Features by Mutual Information')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# === Exemple ===
def demonstrate_mi():
    """Démontre la sélection de features par MI."""
    
    # Créer des données synthétiques
    np.random.seed(42)
    n = 1000
    
    # Features avec différentes relations avec la target
    X = pd.DataFrame({
        'linear': np.random.randn(n),           # Relation linéaire
        'quadratic': np.random.randn(n),        # Relation quadratique
        'sine': np.random.randn(n),             # Relation sinusoïdale
        'noise1': np.random.randn(n),           # Bruit
        'noise2': np.random.randn(n),           # Bruit
    })
    
    # Target avec relations non-linéaires
    y = (2 * X['linear'] + 
         X['quadratic']**2 + 
         np.sin(X['sine'] * 3) + 
         np.random.randn(n) * 0.5)
    
    # Calculer MI
    mi_scores = calculate_mutual_information(X, y)
    
    print("Information Mutuelle:")
    print("-" * 30)
    for feature, score in mi_scores.items():
        print(f"  {feature}: {score:.4f}")
    
    return mi_scores


# mi_scores = demonstrate_mi()
```

---

# 7. MODÈLES LINÉAIRES
## Linear Models

## 7.1 Régression Linéaire pour Prédiction de Rendements

```python
"""
Régression Linéaire pour Finance
================================
La régression linéaire est le point de départ pour la prédiction
de rendements. Malgré sa simplicité, elle reste très utilisée car:
- Interprétable
- Rapide
- Base pour les modèles plus complexes (Ridge, Lasso)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

class LinearReturnPredictor:
    """
    Prédicteur de rendements basé sur régression linéaire.
    """
    
    def __init__(self, regularization='none', alpha=1.0, l1_ratio=0.5):
        """
        Initialise le prédicteur.
        
        Args:
            regularization: 'none', 'ridge' (L2), 'lasso' (L1), 'elasticnet'
            alpha: Force de la régularisation
            l1_ratio: Ratio L1 pour ElasticNet (0=Ridge, 1=Lasso)
        """
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        # Créer le modèle
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        elif regularization == 'elasticnet':
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        else:
            self.model = LinearRegression()
        
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, X, y):
        """
        Entraîne le modèle.
        
        Args:
            X: Features (DataFrame ou array)
            y: Target (rendements)
        
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Standardiser les features
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraîner
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """
        Prédit les rendements.
        
        Args:
            X: Features
        
        Returns:
            array: Prédictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_coefficients(self):
        """
        Retourne les coefficients du modèle.
        
        Returns:
            Series: Coefficients par feature
        """
        coef = self.model.coef_
        
        if self.feature_names:
            return pd.Series(coef, index=self.feature_names).sort_values(
                key=abs, ascending=False
            )
        return pd.Series(coef)
    
    def evaluate(self, X, y):
        """
        Évalue le modèle.
        
        Args:
            X: Features
            y: Vraies valeurs
        
        Returns:
            dict: Métriques d'évaluation
        """
        predictions = self.predict(X)
        
        return {
            'r2': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': np.mean(np.abs(y - predictions)),
            'ic': np.corrcoef(y, predictions)[0, 1],  # Information Coefficient
            'ic_rank': pd.Series(y).corr(pd.Series(predictions), method='spearman')
        }


def fama_macbeth_regression(data, factor_columns, return_column='forward_return'):
    """
    Régression Fama-MacBeth pour panel data.
    
    La régression Fama-MacBeth est standard en finance pour estimer
    les primes de risque des facteurs. Elle se fait en deux étapes:
    
    1. Cross-sectional: Pour chaque date, régresser les rendements
       sur les facteurs pour obtenir les primes de risque
    2. Time-series: Calculer la moyenne et l'écart-type des primes
    
    Args:
        data: DataFrame avec MultiIndex (date, ticker)
        factor_columns: Liste des colonnes de facteurs
        return_column: Nom de la colonne de rendement
    
    Returns:
        DataFrame: Primes de risque avec t-stats
    """
    # Grouper par date
    dates = data.index.get_level_values('date').unique()
    
    # Stocker les primes de risque
    risk_premia = []
    
    for date in dates:
        # Données de cette date
        cross_section = data.loc[date]
        
        if len(cross_section) < len(factor_columns) + 5:  # Minimum d'observations
            continue
        
        # Régression cross-sectionnelle
        X = cross_section[factor_columns]
        y = cross_section[return_column]
        
        # Ajouter constante
        X_const = sm.add_constant(X)
        
        try:
            model = sm.OLS(y, X_const).fit()
            risk_premia.append(model.params)
        except:
            continue
    
    # Convertir en DataFrame
    risk_premia_df = pd.DataFrame(risk_premia)
    
    # Statistiques
    results = pd.DataFrame({
        'mean': risk_premia_df.mean(),
        'std': risk_premia_df.std(),
        't_stat': risk_premia_df.mean() / (risk_premia_df.std() / np.sqrt(len(risk_premia_df))),
        'p_value': 2 * (1 - stats.t.cdf(
            abs(risk_premia_df.mean() / (risk_premia_df.std() / np.sqrt(len(risk_premia_df)))),
            df=len(risk_premia_df) - 1
        ))
    })
    
    return results


# === Exemple ===
def demonstrate_linear_models():
    """Démontre les modèles linéaires."""
    
    # Créer des données
    np.random.seed(42)
    n = 1000
    
    # Features
    X = pd.DataFrame({
        'momentum': np.random.randn(n),
        'value': np.random.randn(n),
        'size': np.random.randn(n),
        'volatility': np.random.randn(n),
        'quality': np.random.randn(n),
    })
    
    # Target: combinaison linéaire + bruit
    y = (0.05 * X['momentum'] + 
         0.03 * X['value'] + 
         -0.02 * X['size'] + 
         0.01 * X['quality'] +
         np.random.randn(n) * 0.1)
    
    # Split train/test
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Comparer les modèles
    models = {
        'OLS': LinearReturnPredictor(regularization='none'),
        'Ridge': LinearReturnPredictor(regularization='ridge', alpha=1.0),
        'Lasso': LinearReturnPredictor(regularization='lasso', alpha=0.01),
        'ElasticNet': LinearReturnPredictor(regularization='elasticnet', alpha=0.1),
    }
    
    print("Comparaison des modèles linéaires")
    print("=" * 60)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        print(f"\n{name}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  IC: {metrics['ic']:.4f}")
        print(f"  IC (rank): {metrics['ic_rank']:.4f}")
        
        print(f"  Coefficients:")
        for feat, coef in model.get_coefficients().items():
            print(f"    {feat}: {coef:.4f}")


# demonstrate_linear_models()
```

## 7.2 Régression Logistique pour Classification

```python
"""
Régression Logistique pour Classification de Mouvements de Prix
===============================================================
La régression logistique prédit la probabilité d'un événement binaire:
- Le prix va-t-il monter ou descendre?
- Y aura-t-il un mouvement significatif?
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)

class PriceMovementClassifier:
    """
    Classifieur de mouvements de prix.
    """
    
    def __init__(self, threshold=0, regularization='l2', C=1.0):
        """
        Initialise le classifieur.
        
        Args:
            threshold: Seuil pour définir up/down (0 = simple sign)
            regularization: 'l1', 'l2', ou 'elasticnet'
            C: Inverse de la force de régularisation
        """
        self.threshold = threshold
        
        self.model = LogisticRegression(
            penalty=regularization,
            C=C,
            solver='saga' if regularization == 'elasticnet' else 'lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def _create_labels(self, returns):
        """
        Crée les labels binaires à partir des rendements.
        
        Args:
            returns: Series de rendements
        
        Returns:
            array: Labels (1 = up, 0 = down)
        """
        return (returns > self.threshold).astype(int)
    
    def fit(self, X, returns):
        """
        Entraîne le classifieur.
        
        Args:
            X: Features
            returns: Rendements (seront convertis en labels)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        y = self._create_labels(returns)
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """
        Prédit les labels.
        
        Args:
            X: Features
        
        Returns:
            array: Labels prédits (0 ou 1)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Prédit les probabilités.
        
        Args:
            X: Features
        
        Returns:
            array: Probabilités de chaque classe
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X, returns):
        """
        Évalue le classifieur.
        
        Args:
            X: Features
            returns: Vrais rendements
        
        Returns:
            dict: Métriques d'évaluation
        """
        y_true = self._create_labels(returns)
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def get_feature_importance(self):
        """
        Retourne l'importance des features.
        
        Returns:
            Series: Importance (coefficients) par feature
        """
        importance = self.model.coef_[0]
        
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(
                key=abs, ascending=False
            )
        return pd.Series(importance)


# === Exemple ===
def demonstrate_logistic_regression():
    """Démontre la régression logistique."""
    
    np.random.seed(42)
    n = 1000
    
    # Features
    X = pd.DataFrame({
        'momentum': np.random.randn(n),
        'rsi': np.random.randn(n),
        'macd': np.random.randn(n),
        'volume_ratio': np.random.randn(n),
    })
    
    # Rendements (légèrement prévisibles)
    returns = (0.3 * X['momentum'] + 
               0.2 * X['rsi'] + 
               0.1 * X['macd'] +
               np.random.randn(n) * 0.5)
    
    # Split
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    ret_train, ret_test = returns[:train_size], returns[train_size:]
    
    # Entraîner
    clf = PriceMovementClassifier(threshold=0, C=0.1)
    clf.fit(X_train, ret_train)
    
    # Évaluer
    metrics = clf.evaluate(X_test, ret_test)
    
    print("Résultats du classifieur")
    print("=" * 40)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1']:.2%}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    print("\nImportance des features:")
    for feat, imp in clf.get_feature_importance().items():
        print(f"  {feat}: {imp:.4f}")


# demonstrate_logistic_regression()
```

---

# 8. WORKFLOW ML4T COMPLET
## Complete ML4T Workflow

## 8.1 Deflated Sharpe Ratio

```python
"""
Deflated Sharpe Ratio
=====================
Le Deflated Sharpe Ratio (DSR) corrige le Sharpe Ratio pour tenir compte
du multiple testing (test de nombreuses stratégies).

Quand on teste N stratégies, la meilleure aura un Sharpe élevé par chance
même si aucune n'est vraiment profitable.

Le DSR estime la probabilité que le Sharpe observé soit dû au hasard.
"""
import numpy as np
from scipy import stats

def deflated_sharpe_ratio(sharpe_observed, n_trials, variance_sharpe=1,
                          skewness_returns=0, kurtosis_returns=3,
                          n_observations=252):
    """
    Calcule le Deflated Sharpe Ratio.
    
    Args:
        sharpe_observed: Sharpe ratio observé de la meilleure stratégie
        n_trials: Nombre de stratégies testées
        variance_sharpe: Variance des Sharpe ratios (généralement 1)
        skewness_returns: Skewness des rendements (0 = normal)
        kurtosis_returns: Kurtosis des rendements (3 = normal)
        n_observations: Nombre d'observations
    
    Returns:
        float: Probabilité que le Sharpe soit dû au skill (pas au hasard)
    """
    # Sharpe ratio attendu de la meilleure stratégie sous l'hypothèse nulle
    # (toutes les stratégies ont un vrai Sharpe de 0)
    expected_max_sharpe = (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_trials) +
        np.euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))
    ) * np.sqrt(variance_sharpe)
    
    # Variance du Sharpe ratio estimé
    var_sharpe = (
        (1 + 0.25 * sharpe_observed**2 * (kurtosis_returns - 1) -
         sharpe_observed * skewness_returns) / n_observations
    )
    
    # Test statistique
    z_stat = (sharpe_observed - expected_max_sharpe) / np.sqrt(var_sharpe)
    
    # Probabilité (p-value one-sided)
    prob_skill = stats.norm.cdf(z_stat)
    
    return prob_skill


def minimum_track_record_length(sharpe_target, sharpe_benchmark=0,
                                skewness=0, kurtosis=3, alpha=0.05):
    """
    Calcule la durée minimale de track record nécessaire.
    
    Combien d'observations faut-il pour être confiant que le Sharpe
    observé n'est pas dû au hasard?
    
    Args:
        sharpe_target: Sharpe ratio cible
        sharpe_benchmark: Sharpe ratio du benchmark (généralement 0)
        skewness: Skewness des rendements
        kurtosis: Kurtosis des rendements
        alpha: Niveau de significativité
    
    Returns:
        int: Nombre minimum d'observations requises
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    
    # Formule de Bailey et Lopez de Prado
    min_length = (
        (z_alpha / (sharpe_target - sharpe_benchmark))**2 *
        (1 + 0.25 * sharpe_target**2 * (kurtosis - 1) - sharpe_target * skewness)
    )
    
    return int(np.ceil(min_length))


# === Exemple ===
def demonstrate_dsr():
    """Démontre le Deflated Sharpe Ratio."""
    
    print("Deflated Sharpe Ratio")
    print("=" * 50)
    
    # Scénario: On a testé 100 stratégies
    # La meilleure a un Sharpe de 2.0
    sharpe = 2.0
    n_trials = 100
    
    dsr = deflated_sharpe_ratio(sharpe, n_trials)
    
    print(f"\nSharpe observé: {sharpe}")
    print(f"Nombre de stratégies testées: {n_trials}")
    print(f"Probabilité de skill (DSR): {dsr:.2%}")
    
    if dsr > 0.95:
        print("→ Forte probabilité que ce soit du skill")
    elif dsr > 0.50:
        print("→ Résultat incertain, prudence recommandée")
    else:
        print("→ Probablement dû au hasard (data mining)")
    
    # Minimum track record
    print("\n" + "-" * 50)
    print("Track record minimum pour différents Sharpe cibles:")
    
    for target_sharpe in [0.5, 1.0, 1.5, 2.0, 2.5]:
        min_obs = minimum_track_record_length(target_sharpe)
        years = min_obs / 252
        print(f"  Sharpe {target_sharpe}: {min_obs} jours ({years:.1f} années)")


# demonstrate_dsr()
```

## 8.2 Backtrader - Framework de Backtesting

```python
"""
Backtrader - Framework de Backtesting Professionnel
===================================================
Backtrader est un framework Python complet pour le backtesting
et le trading algorithmique.

Avantages:
- Event-driven (simulation réaliste)
- Support multi-timeframe
- Gestion des ordres complexes
- Intégration avec brokers (IB, Oanda)
"""
import backtrader as bt
import pandas as pd
import numpy as np

class MovingAverageCrossStrategy(bt.Strategy):
    """
    Stratégie de croisement de moyennes mobiles.
    
    Long quand MA rapide > MA lente
    Flat quand MA rapide < MA lente
    """
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('printlog', True),
    )
    
    def __init__(self):
        """Initialise les indicateurs."""
        # Moyennes mobiles
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period
        )
        
        # Signal de crossover
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Pour le logging
        self.order = None
    
    def next(self):
        """
        Appelé à chaque nouvelle barre.
        
        Logique de trading:
        - Si pas en position et crossover up → acheter
        - Si en position et crossover down → vendre
        """
        # Vérifier si un ordre est en attente
        if self.order:
            return
        
        # Vérifier si on est en position
        if not self.position:
            # Pas en position
            if self.crossover > 0:  # MA rapide croise au-dessus
                self.order = self.buy()
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
        else:
            # En position
            if self.crossover < 0:  # MA rapide croise en-dessous
                self.order = self.sell()
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
    
    def notify_order(self, order):
        """Appelé quand un ordre change de statut."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def log(self, txt, dt=None):
        """Logging helper."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')


class MomentumStrategy(bt.Strategy):
    """
    Stratégie momentum basée sur le RSI (Relative Strength Index).
    
    - Achète quand RSI sort de la zone de survente (< 30)
    - Vend quand RSI entre dans la zone de surachat (> 70)
    """
    
    params = (
        ('rsi_period', 14),
        ('oversold', 30),
        ('overbought', 70),
        ('stake', 100),  # Nombre d'actions par trade
    )
    
    def __init__(self):
        """Initialise les indicateurs."""
        self.rsi = bt.indicators.RSI(
            self.data.close, period=self.params.rsi_period
        )
        self.order = None
    
    def next(self):
        """Logique de trading."""
        if self.order:
            return
        
        if not self.position:
            # Pas en position - chercher signal d'achat
            if self.rsi < self.params.oversold:
                self.order = self.buy(size=self.params.stake)
        else:
            # En position - chercher signal de vente
            if self.rsi > self.params.overbought:
                self.order = self.sell(size=self.params.stake)
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


def run_backtrader_backtest(data, strategy_class, strategy_params=None,
                            initial_cash=100000, commission=0.001):
    """
    Exécute un backtest avec Backtrader.
    
    Args:
        data: DataFrame avec OHLCV (index = dates)
        strategy_class: Classe de stratégie Backtrader
        strategy_params: Paramètres de la stratégie
        initial_cash: Capital initial
        commission: Commission par transaction
    
    Returns:
        dict: Résultats du backtest
    """
    # Créer l'instance Cerebro
    cerebro = bt.Cerebro()
    
    # Ajouter la stratégie
    if strategy_params:
        cerebro.addstrategy(strategy_class, **strategy_params)
    else:
        cerebro.addstrategy(strategy_class)
    
    # Convertir les données en format Backtrader
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # Utiliser l'index
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    
    cerebro.adddata(data_feed)
    
    # Configuration
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # Ajouter les analyseurs
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Exécuter
    print(f'Starting Portfolio Value: ${initial_cash:,.2f}')
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: ${final_value:,.2f}')
    
    # Extraire les résultats
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    results_dict = {
        'initial_value': initial_cash,
        'final_value': final_value,
        'total_return': (final_value - initial_cash) / initial_cash,
        'sharpe_ratio': sharpe.get('sharperatio', None),
        'max_drawdown': drawdown.get('max', {}).get('drawdown', None),
        'total_trades': trades.get('total', {}).get('total', 0),
    }
    
    return results_dict, cerebro


# === Exemple d'utilisation ===
def backtrader_example():
    """Exemple de backtest avec Backtrader."""
    import yfinance as yf
    
    # Télécharger les données
    data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Exécuter le backtest
    results, cerebro = run_backtrader_backtest(
        data,
        MovingAverageCrossStrategy,
        strategy_params={'fast_period': 10, 'slow_period': 30, 'printlog': False}
    )
    
    print("\nRésultats du backtest:")
    print("-" * 40)
    print(f"Rendement total: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}" if results['sharpe_ratio'] else "N/A")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}" if results['max_drawdown'] else "N/A")
    print(f"Nombre de trades: {results['total_trades']}")
    
    # Afficher le graphique
    cerebro.plot(style='candlestick')
    
    return results


# Pour exécuter:
# results = backtrader_example()
```

---

# 9. MODÈLES DE SÉRIES TEMPORELLES
## Time Series Models

## 9.1 Stationnarité et Tests

```python
"""
Stationnarité des Séries Temporelles
====================================
Une série temporelle est stationnaire si ses propriétés statistiques
(moyenne, variance, autocorrélation) ne changent pas dans le temps.

C'est important car la plupart des modèles (ARIMA, etc.) supposent
la stationnarité.

Types de non-stationnarité:
1. Tendance (trend): La moyenne change
2. Saisonnalité: Patterns périodiques
3. Hétéroscédasticité: La variance change (GARCH)
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def adf_test(series, significance_level=0.05, verbose=True):
    """
    Test de Dickey-Fuller Augmenté (ADF) pour la stationnarité.
    
    H0: La série a une racine unitaire (non-stationnaire)
    H1: La série est stationnaire
    
    Args:
        series: Series temporelle
        significance_level: Niveau de significativité
        verbose: Afficher les résultats
    
    Returns:
        dict: Résultats du test
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'n_observations': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level
    }
    
    if verbose:
        print("Test ADF (Augmented Dickey-Fuller)")
        print("=" * 50)
        print(f"Test Statistic: {output['test_statistic']:.4f}")
        print(f"p-value: {output['p_value']:.4f}")
        print(f"Lags Used: {output['lags_used']}")
        print(f"Number of Observations: {output['n_observations']}")
        print("Critical Values:")
        for key, value in output['critical_values'].items():
            print(f"  {key}: {value:.4f}")
        
        if output['is_stationary']:
            print(f"\n✓ La série est STATIONNAIRE (p-value < {significance_level})")
        else:
            print(f"\n✗ La série est NON-STATIONNAIRE (p-value >= {significance_level})")
    
    return output


def kpss_test(series, regression='c', significance_level=0.05, verbose=True):
    """
    Test KPSS pour la stationnarité.
    
    Contrairement à ADF, KPSS teste:
    H0: La série est stationnaire
    H1: La série a une racine unitaire
    
    Args:
        series: Series temporelle
        regression: 'c' (constant) ou 'ct' (constant + trend)
        significance_level: Niveau de significativité
        verbose: Afficher les résultats
    
    Returns:
        dict: Résultats du test
    """
    result = kpss(series.dropna(), regression=regression)
    
    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'critical_values': result[3],
        'is_stationary': result[1] > significance_level
    }
    
    if verbose:
        print("\nTest KPSS")
        print("=" * 50)
        print(f"Test Statistic: {output['test_statistic']:.4f}")
        print(f"p-value: {output['p_value']:.4f}")
        print(f"Lags Used: {output['lags_used']}")
        print("Critical Values:")
        for key, value in output['critical_values'].items():
            print(f"  {key}: {value:.4f}")
        
        if output['is_stationary']:
            print(f"\n✓ La série est STATIONNAIRE (p-value > {significance_level})")
        else:
            print(f"\n✗ La série est NON-STATIONNAIRE (p-value <= {significance_level})")
    
    return output


def make_stationary(series, method='diff', order=1):
    """
    Transforme une série en série stationnaire.
    
    Args:
        series: Series non-stationnaire
        method: 'diff' (différenciation), 'log_diff' (rendements log),
                'pct_change' (rendements)
        order: Ordre de différenciation
    
    Returns:
        Series: Série transformée
    """
    if method == 'diff':
        return series.diff(order).dropna()
    elif method == 'log_diff':
        return np.log(series).diff(order).dropna()
    elif method == 'pct_change':
        return series.pct_change(order).dropna()
    else:
        raise ValueError(f"Méthode inconnue: {method}")


def plot_stationarity_diagnostics(series, lags=40, title=''):
    """
    Affiche les diagnostics de stationnarité.
    
    Args:
        series: Series temporelle
        lags: Nombre de lags pour ACF/PACF
        title: Titre du graphique
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Série originale
    axes[0, 0].plot(series)
    axes[0, 0].set_title(f'{title} - Time Series')
    
    # 2. Distribution
    series.hist(ax=axes[0, 1], bins=50, density=True)
    axes[0, 1].set_title('Distribution')
    
    # 3. ACF (AutoCorrelation Function)
    plot_acf(series.dropna(), ax=axes[1, 0], lags=lags)
    axes[1, 0].set_title('Autocorrelation (ACF)')
    
    # 4. PACF (Partial ACF)
    plot_pacf(series.dropna(), ax=axes[1, 1], lags=lags)
    axes[1, 1].set_title('Partial Autocorrelation (PACF)')
    
    plt.tight_layout()
    plt.show()


# === Exemple ===
def demonstrate_stationarity():
    """Démontre les tests de stationnarité."""
    import yfinance as yf
    
    # Télécharger des données
    data = yf.download('SPY', start='2020-01-01', end='2023-12-31')
    prices = data['Adj Close']
    returns = prices.pct_change().dropna()
    
    print("PRIX (non-stationnaire)")
    print("=" * 60)
    adf_test(prices)
    kpss_test(prices)
    
    print("\n\nRENDEMENTS (stationnaire)")
    print("=" * 60)
    adf_test(returns)
    kpss_test(returns)


# demonstrate_stationarity()
```

## 9.2 Modèles ARIMA

```python
"""
Modèles ARIMA (AutoRegressive Integrated Moving Average)
========================================================
ARIMA(p, d, q) combine trois composantes:
- AR(p): AutoRégressive - utilise les p observations passées
- I(d): Intégration - d différenciations pour rendre stationnaire
- MA(q): Moving Average - utilise les q erreurs passées

Formule:
y'(t) = c + φ₁y'(t-1) + ... + φₚy'(t-p) + θ₁ε(t-1) + ... + θqε(t-q) + ε(t)

où y'(t) est la série différenciée d fois.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

def select_arima_order(series, max_p=5, max_d=2, max_q=5, criterion='aic'):
    """
    Sélectionne automatiquement les ordres ARIMA optimaux.
    
    Args:
        series: Series temporelle
        max_p: Maximum pour p
        max_d: Maximum pour d
        max_q: Maximum pour q
        criterion: 'aic' ou 'bic'
    
    Returns:
        tuple: (p, d, q) optimaux
    """
    best_score = np.inf
    best_order = (0, 0, 0)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    results = model.fit()
                    
                    score = results.aic if criterion == 'aic' else results.bic
                    
                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                except:
                    continue
    
    return best_order


class ARIMAForecaster:
    """
    Forecaster basé sur ARIMA.
    """
    
    def __init__(self, order=None, auto_select=True, seasonal_order=None):
        """
        Initialise le forecaster.
        
        Args:
            order: (p, d, q) ou None pour auto-sélection
            auto_select: Sélectionner automatiquement l'ordre
            seasonal_order: (P, D, Q, S) pour SARIMA
        """
        self.order = order
        self.auto_select = auto_select
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
    
    def fit(self, series):
        """
        Entraîne le modèle.
        
        Args:
            series: Series temporelle
        
        Returns:
            self
        """
        if self.auto_select and self.order is None:
            self.order = select_arima_order(series)
            print(f"Ordre sélectionné: ARIMA{self.order}")
        
        if self.seasonal_order:
            self.model = SARIMAX(
                series, 
                order=self.order, 
                seasonal_order=self.seasonal_order
            )
        else:
            self.model = ARIMA(series, order=self.order)
        
        self.results = self.model.fit()
        
        return self
    
    def predict(self, steps=1):
        """
        Prédit les valeurs futures.
        
        Args:
            steps: Nombre de pas à prédire
        
        Returns:
            Series: Prédictions
        """
        forecast = self.results.forecast(steps=steps)
        return forecast
    
    def get_summary(self):
        """
        Retourne le résumé du modèle.
        
        Returns:
            str: Résumé statistique
        """
        return self.results.summary()
    
    def diagnostic_plots(self):
        """Affiche les diagnostics du modèle."""
        self.results.plot_diagnostics(figsize=(14, 10))
        plt.tight_layout()
        plt.show()


# === Exemple ===
def demonstrate_arima():
    """Démontre les modèles ARIMA."""
    import yfinance as yf
    
    # Données
    data = yf.download('SPY', start='2020-01-01', end='2023-12-31')
    returns = data['Adj Close'].pct_change().dropna() * 100  # En pourcentage
    
    # Split train/test
    train = returns[:-30]
    test = returns[-30:]
    
    # Créer et entraîner le modèle
    forecaster = ARIMAForecaster(auto_select=True)
    forecaster.fit(train)
    
    print(forecaster.get_summary())
    
    # Prédictions
    predictions = forecaster.predict(steps=30)
    
    # Évaluer
    mae = np.mean(np.abs(test.values - predictions.values))
    rmse = np.sqrt(np.mean((test.values - predictions.values)**2))
    
    print(f"\nMétriques de prédiction:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return forecaster


# forecaster = demonstrate_arima()
```

## 9.3 Modèles GARCH pour la Volatilité

```python
"""
Modèles GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)
=========================================================================
Les modèles GARCH modélisent la volatilité qui varie dans le temps.

GARCH(p, q):
σ²(t) = ω + Σαᵢε²(t-i) + Σβⱼσ²(t-j)

où:
- σ²(t): Variance conditionnelle
- ε(t): Résidus
- α: Impact des chocs passés (ARCH)
- β: Persistance de la volatilité (GARCH)

Applications:
- Prévision de volatilité
- Value at Risk (VaR)
- Option pricing
"""
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

class GARCHVolatilityModel:
    """
    Modèle GARCH pour la prévision de volatilité.
    """
    
    def __init__(self, p=1, q=1, vol='Garch', dist='normal'):
        """
        Initialise le modèle.
        
        Args:
            p: Ordre GARCH (persistance)
            q: Ordre ARCH (chocs)
            vol: Type de volatilité ('Garch', 'EGarch', 'GJR-GARCH')
            dist: Distribution des erreurs ('normal', 't', 'skewt')
        """
        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self.model = None
        self.results = None
    
    def fit(self, returns, rescale=True):
        """
        Entraîne le modèle.
        
        Args:
            returns: Series de rendements (en pourcentage recommandé)
            rescale: Multiplier par 100 si nécessaire
        
        Returns:
            self
        """
        # Rescale si les rendements sont petits
        if rescale and returns.std() < 0.1:
            returns = returns * 100
        
        self.model = arch_model(
            returns,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist
        )
        
        self.results = self.model.fit(disp='off')
        
        return self
    
    def forecast_volatility(self, horizon=1, method='analytic'):
        """
        Prévoit la volatilité future.
        
        Args:
            horizon: Nombre de périodes
            method: 'analytic', 'simulation', ou 'bootstrap'
        
        Returns:
            DataFrame: Prévisions de volatilité
        """
        forecast = self.results.forecast(horizon=horizon, method=method)
        
        # Retourner l'écart-type (pas la variance)
        return np.sqrt(forecast.variance)
    
    def conditional_volatility(self):
        """
        Retourne la volatilité conditionnelle historique.
        
        Returns:
            Series: Volatilité conditionnelle
        """
        return np.sqrt(self.results.conditional_volatility)
    
    def get_summary(self):
        """Retourne le résumé du modèle."""
        return self.results.summary()
    
    def calculate_var(self, confidence_level=0.05, horizon=1):
        """
        Calcule la Value at Risk (VaR).
        
        Args:
            confidence_level: Niveau de confiance (0.05 = 95%)
            horizon: Horizon en jours
        
        Returns:
            float: VaR
        """
        from scipy import stats
        
        # Prévision de volatilité
        vol_forecast = self.forecast_volatility(horizon=horizon)
        vol = vol_forecast.values[-1, 0]
        
        # VaR paramétrique
        z = stats.norm.ppf(confidence_level)
        var = z * vol * np.sqrt(horizon)
        
        return var
    
    def plot_volatility(self, figsize=(14, 8)):
        """Affiche la volatilité conditionnelle."""
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 1. Rendements
        ax1 = axes[0]
        self.results.resid.plot(ax=ax1, alpha=0.7)
        ax1.set_title('Returns')
        
        # 2. Volatilité conditionnelle
        ax2 = axes[1]
        vol = self.conditional_volatility()
        vol.plot(ax=ax2, color='red')
        ax2.set_title('Conditional Volatility (GARCH)')
        ax2.fill_between(vol.index, 0, vol, alpha=0.3, color='red')
        
        plt.tight_layout()
        plt.show()


# === Exemple ===
def demonstrate_garch():
    """Démontre les modèles GARCH."""
    import yfinance as yf
    
    # Données
    data = yf.download('SPY', start='2015-01-01', end='2023-12-31')
    returns = data['Adj Close'].pct_change().dropna() * 100
    
    # Créer et entraîner le modèle
    garch = GARCHVolatilityModel(p=1, q=1, vol='Garch', dist='t')
    garch.fit(returns, rescale=False)
    
    print(garch.get_summary())
    
    # Prévision
    vol_forecast = garch.forecast_volatility(horizon=5)
    print(f"\nPrévision de volatilité (5 jours):")
    print(vol_forecast)
    
    # VaR
    var_95 = garch.calculate_var(confidence_level=0.05)
    print(f"\nVaR 95% (1 jour): {var_95:.2f}%")
    
    # Plot
    garch.plot_volatility()
    
    return garch


# garch = demonstrate_garch()
```

## 9.4 Cointegration et Pairs Trading

```python
"""
Cointegration et Pairs Trading
==============================
Deux séries sont cointégrées si leur combinaison linéaire est stationnaire,
même si chaque série ne l'est pas individuellement.

Exemple: Prix de Coca-Cola et Pepsi
- Chaque prix peut être non-stationnaire (tendance)
- Mais le spread (Coca - β*Pepsi) peut être stationnaire

C'est la base du pairs trading:
1. Trouver des paires cointégrées
2. Calculer le spread
3. Trader quand le spread s'écarte de sa moyenne
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def test_cointegration(series1, series2, significance_level=0.05):
    """
    Teste la cointegration entre deux séries.
    
    Utilise le test d'Engle-Granger:
    1. Régresse series1 sur series2
    2. Teste si les résidus sont stationnaires
    
    Args:
        series1: Première série
        series2: Deuxième série
        significance_level: Niveau de significativité
    
    Returns:
        dict: Résultats du test
    """
    # Test de cointegration
    score, pvalue, _ = coint(series1, series2)
    
    # Régression pour obtenir le coefficient
    series2_const = sm.add_constant(series2)
    model = OLS(series1, series2_const).fit()
    
    # Spread (résidus)
    spread = series1 - model.params[1] * series2 - model.params[0]
    
    # Test ADF sur le spread
    adf_result = adfuller(spread)
    
    return {
        'coint_stat': score,
        'coint_pvalue': pvalue,
        'is_cointegrated': pvalue < significance_level,
        'hedge_ratio': model.params[1],
        'intercept': model.params[0],
        'spread': spread,
        'spread_mean': spread.mean(),
        'spread_std': spread.std(),
        'adf_stat': adf_result[0],
        'adf_pvalue': adf_result[1]
    }


def find_cointegrated_pairs(prices_df, significance_level=0.05):
    """
    Trouve toutes les paires cointégrées dans un DataFrame.
    
    Args:
        prices_df: DataFrame de prix (colonnes = actifs)
        significance_level: Niveau de significativité
    
    Returns:
        list: Liste des paires cointégrées avec leurs statistiques
    """
    n = len(prices_df.columns)
    tickers = prices_df.columns.tolist()
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            ticker1, ticker2 = tickers[i], tickers[j]
            
            result = test_cointegration(
                prices_df[ticker1], 
                prices_df[ticker2],
                significance_level
            )
            
            if result['is_cointegrated']:
                pairs.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'pvalue': result['coint_pvalue'],
                    'hedge_ratio': result['hedge_ratio']
                })
    
    return sorted(pairs, key=lambda x: x['pvalue'])


class PairsTradingStrategy:
    """
    Stratégie de pairs trading basée sur la cointegration.
    """
    
    def __init__(self, entry_zscore=2.0, exit_zscore=0.5, 
                 lookback=252, hedge_ratio=None):
        """
        Initialise la stratégie.
        
        Args:
            entry_zscore: Z-score pour entrer en position
            exit_zscore: Z-score pour sortir
            lookback: Période pour calculer la moyenne/std du spread
            hedge_ratio: Ratio de hedge fixe (None = calculer dynamiquement)
        """
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.lookback = lookback
        self.hedge_ratio = hedge_ratio
    
    def calculate_spread(self, prices1, prices2):
        """
        Calcule le spread entre deux séries.
        
        Args:
            prices1: Prix de l'actif 1
            prices2: Prix de l'actif 2
        
        Returns:
            Series: Spread
        """
        if self.hedge_ratio is None:
            # Calculer le hedge ratio par régression rolling
            # Simplifié: ratio constant
            result = test_cointegration(prices1, prices2)
            hr = result['hedge_ratio']
        else:
            hr = self.hedge_ratio
        
        return prices1 - hr * prices2
    
    def calculate_zscore(self, spread):
        """
        Calcule le z-score du spread.
        
        Args:
            spread: Series du spread
        
        Returns:
            Series: Z-score
        """
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        
        return (spread - mean) / std
    
    def generate_signals(self, prices1, prices2):
        """
        Génère les signaux de trading.
        
        Args:
            prices1: Prix de l'actif 1
            prices2: Prix de l'actif 2
        
        Returns:
            DataFrame: Signaux pour chaque actif
        """
        spread = self.calculate_spread(prices1, prices2)
        zscore = self.calculate_zscore(spread)
        
        signals = pd.DataFrame(index=prices1.index)
        signals['zscore'] = zscore
        signals['signal1'] = 0  # Signal pour actif 1
        signals['signal2'] = 0  # Signal pour actif 2
        
        # Entrée short spread (spread trop haut)
        signals.loc[zscore > self.entry_zscore, 'signal1'] = -1  # Short asset 1
        signals.loc[zscore > self.entry_zscore, 'signal2'] = 1   # Long asset 2
        
        # Entrée long spread (spread trop bas)
        signals.loc[zscore < -self.entry_zscore, 'signal1'] = 1  # Long asset 1
        signals.loc[zscore < -self.entry_zscore, 'signal2'] = -1 # Short asset 2
        
        # Sortie
        signals.loc[abs(zscore) < self.exit_zscore, 'signal1'] = 0
        signals.loc[abs(zscore) < self.exit_zscore, 'signal2'] = 0
        
        return signals
    
    def backtest(self, prices1, prices2):
        """
        Backteste la stratégie.
        
        Args:
            prices1: Prix de l'actif 1
            prices2: Prix de l'actif 2
        
        Returns:
            DataFrame: Résultats du backtest
        """
        signals = self.generate_signals(prices1, prices2)
        
        # Rendements
        returns1 = prices1.pct_change()
        returns2 = prices2.pct_change()
        
        # Rendements de la stratégie
        strategy_returns = (
            signals['signal1'].shift(1) * returns1 +
            signals['signal2'].shift(1) * returns2
        ) / 2  # Normalisé
        
        # Résultats
        results = pd.DataFrame(index=prices1.index)
        results['spread'] = self.calculate_spread(prices1, prices2)
        results['zscore'] = signals['zscore']
        results['signal1'] = signals['signal1']
        results['signal2'] = signals['signal2']
        results['strategy_return'] = strategy_returns
        results['cumulative_return'] = (1 + strategy_returns).cumprod()
        
        return results


# === Exemple ===
def demonstrate_pairs_trading():
    """Démontre le pairs trading."""
    import yfinance as yf
    
    # Télécharger des données (secteur financier)
    tickers = ['JPM', 'BAC', 'C', 'WFC', 'GS']
    data = yf.download(tickers, start='2018-01-01', end='2023-12-31')['Adj Close']
    
    # Trouver les paires cointégrées
    print("Recherche de paires cointégrées...")
    pairs = find_cointegrated_pairs(data, significance_level=0.05)
    
    print(f"\nPaires trouvées: {len(pairs)}")
    for pair in pairs[:5]:
        print(f"  {pair['ticker1']}-{pair['ticker2']}: "
              f"p-value={pair['pvalue']:.4f}, "
              f"hedge_ratio={pair['hedge_ratio']:.2f}")
    
    if pairs:
        # Backtester la meilleure paire
        best = pairs[0]
        print(f"\nBacktest de {best['ticker1']}-{best['ticker2']}")
        
        strategy = PairsTradingStrategy(
            entry_zscore=2.0,
            exit_zscore=0.5,
            hedge_ratio=best['hedge_ratio']
        )
        
        results = strategy.backtest(
            data[best['ticker1']], 
            data[best['ticker2']]
        )
        
        # Métriques
        total_return = results['cumulative_return'].iloc[-1] - 1
        sharpe = np.sqrt(252) * results['strategy_return'].mean() / results['strategy_return'].std()
        
        print(f"\nRésultats:")
        print(f"  Rendement total: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        
        return results, strategy
    
    return None, None


# results, strategy = demonstrate_pairs_trading()
```

                                        reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        return self
    
    def predict(self, X):
        """Prédit les valeurs."""
        preds = self.model.predict(X)
        
        if self.task == 'classification':
            return (preds > 0.5).astype(int)
        return preds
    
    def predict_proba(self, X):
        """Prédit les probabilités."""
        return self.model.predict(X)
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Retourne l'importance des features.
        
        Args:
            importance_type: 'gain', 'split', ou 'shap'
        """
        importance = self.model.feature_importance(importance_type=importance_type)
        
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importance)
    
    def cross_validate(self, X, y, n_splits=5):
        """
        Cross-validation temporelle.
        
        Args:
            X: Features
            y: Target
            n_splits: Nombre de splits
        
        Returns:
            dict: Résultats de CV
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.fit(X_train, y_train, X_val, y_val,
                    num_boost_round=500, early_stopping_rounds=30)
            
            if self.task == 'classification':
                from sklearn.metrics import roc_auc_score
                preds = self.predict_proba(X_val)
                score = roc_auc_score(y_val, preds)
            else:
                from sklearn.metrics import r2_score
                preds = self.predict(X_val)
                score = r2_score(y_val, preds)
            
            scores.append(score)
            print(f"Fold {fold+1}: {score:.4f}")
        
        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }


# === CatBoost ===
"""
CatBoost pour le Trading
========================
CatBoost (Categorical Boosting) est optimisé pour les features catégorielles.

Avantages:
- Gestion native des catégorielles (pas besoin d'encoding)
- Ordered Target Statistics (évite target leakage)
- Très performant out-of-the-box
"""
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

class CatBoostTrader:
    """
    CatBoost pour prédiction de trading.
    """
    
    def __init__(self, task='classification', params=None, cat_features=None):
        """
        Initialise CatBoost.
        
        Args:
            task: 'classification' ou 'regression'
            params: Paramètres CatBoost
            cat_features: Liste des features catégorielles
        """
        self.task = task
        self.cat_features = cat_features or []
        
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'min_child_samples': 100,
            'random_seed': 42,
            'verbose': 100,
        }
        
        self.params = {**default_params, **(params or {})}
        
        ModelClass = CatBoostClassifier if task == 'classification' else CatBoostRegressor
        self.model = ModelClass(**self.params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """Entraîne le modèle."""
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        
        eval_set = None
        if X_val is not None:
            eval_set = Pool(X_val, y_val, cat_features=self.cat_features)
        
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True
        )
        
        return self
    
    def predict(self, X):
        """Prédit."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Prédit les probabilités."""
        if self.task == 'classification':
            return self.model.predict_proba(X)[:, 1]
        raise ValueError("predict_proba only for classification")
    
    def get_feature_importance(self):
        """Retourne l'importance des features."""
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)


# === SHAP pour Interprétabilité ===
"""
SHAP (SHapley Additive exPlanations)
====================================
SHAP explique les prédictions individuelles en attribuant
une contribution à chaque feature.

Basé sur la théorie des jeux (valeurs de Shapley).
"""
import shap

def explain_with_shap(model, X, feature_names=None, plot=True):
    """
    Explique les prédictions avec SHAP.
    
    Args:
        model: Modèle entraîné (LightGBM, XGBoost, CatBoost, etc.)
        X: Features
        feature_names: Noms des features
        plot: Afficher les graphiques
    
    Returns:
        shap_values: Valeurs SHAP
    """
    # Créer l'explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(X)
    
    if plot:
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.show()
        
        # Feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.show()
    
    return shap_values


def explain_single_prediction(model, X_single, X_background, feature_names=None):
    """
    Explique une prédiction individuelle.
    
    Args:
        model: Modèle entraîné
        X_single: Une seule observation à expliquer
        X_background: Données de background pour SHAP
        feature_names: Noms des features
    """
    explainer = shap.TreeExplainer(model, X_background)
    shap_values = explainer.shap_values(X_single)
    
    # Waterfall plot
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0] if isinstance(shap_values, list) else shap_values,
        base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X_single.values[0] if hasattr(X_single, 'values') else X_single[0],
        feature_names=feature_names
    ))


# === Exemple complet GBM ===
def demonstrate_gbm_trading():
    """Démontre LightGBM et CatBoost pour le trading."""
    np.random.seed(42)
    n = 10000
    
    # Créer des données
    X = pd.DataFrame({
        'momentum_1m': np.random.randn(n),
        'momentum_3m': np.random.randn(n),
        'volatility': np.abs(np.random.randn(n)) + 0.1,
        'rsi': np.random.uniform(20, 80, n),
        'macd': np.random.randn(n),
        'volume_ratio': np.random.lognormal(0, 0.3, n),
        'sector': np.random.choice(['Tech', 'Finance', 'Health', 'Energy'], n),  # Catégorielle
    })
    
    # Encoder la catégorielle pour LightGBM
    X_lgb = X.copy()
    X_lgb['sector'] = X_lgb['sector'].astype('category').cat.codes
    
    # Target
    y = ((X['momentum_1m'] > 0.3) & (X['rsi'] < 65) | 
         (X['macd'] > 0.5)).astype(int)
    
    # Split temporel
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X_lgb[:train_end], y[:train_end]
    X_val, y_val = X_lgb[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X_lgb[val_end:], y[val_end:]
    
    # LightGBM
    print("=" * 60)
    print("LightGBM")
    print("=" * 60)
    
    lgbm = LightGBMTrader(task='classification')
    lgbm.fit(X_train, y_train, X_val, y_val)
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    y_pred_lgb = lgbm.predict(X_test)
    y_proba_lgb = lgbm.predict_proba(X_test)
    
    print(f"\nLightGBM Results:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_lgb):.2%}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_proba_lgb):.4f}")
    
    print(f"\nFeature Importance (Gain):")
    for feat, imp in lgbm.get_feature_importance('gain').head(5).items():
        print(f"  {feat}: {imp:.0f}")
    
    # CatBoost
    print("\n" + "=" * 60)
    print("CatBoost")
    print("=" * 60)
    
    # CatBoost avec features originales (gère les catégorielles)
    X_cat_train = X[:train_end]
    X_cat_val = X[train_end:val_end]
    X_cat_test = X[val_end:]
    
    catboost = CatBoostTrader(task='classification', cat_features=['sector'])
    catboost.fit(X_cat_train, y_train, X_cat_val, y_val)
    
    y_pred_cat = catboost.predict(X_cat_test)
    y_proba_cat = catboost.predict_proba(X_cat_test)
    
    print(f"\nCatBoost Results:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_cat):.2%}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_proba_cat):.4f}")
    
    return lgbm, catboost


# lgbm, catboost = demonstrate_gbm_trading()
```

---

# 13. APPRENTISSAGE NON SUPERVISÉ
## Unsupervised Learning

## 13.1 PCA (Principal Component Analysis)

```python
"""
PCA pour la Finance
===================
PCA (Principal Component Analysis - Analyse en Composantes Principales)
réduit la dimensionnalité en trouvant les directions de variance maximale.

Applications en finance:
- Extraction de facteurs de risque
- Compression de données
- Détection d'anomalies
- Construction d'eigenportfolios
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class FinancialPCA:
    """
    PCA pour données financières.
    """
    
    def __init__(self, n_components=None, variance_threshold=0.95):
        """
        Initialise PCA.
        
        Args:
            n_components: Nombre de composantes (None = auto)
            variance_threshold: Seuil de variance expliquée si n_components=None
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None
    
    def fit(self, X):
        """
        Ajuste PCA aux données.
        
        Args:
            X: DataFrame ou array de features
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Standardiser
        X_scaled = self.scaler.fit_transform(X)
        
        # Si n_components est None, trouver automatiquement
        if self.n_components is None:
            # D'abord faire PCA complet
            pca_full = PCA()
            pca_full.fit(X_scaled)
            
            # Trouver le nombre de composantes pour atteindre le seuil
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            print(f"Composantes sélectionnées: {self.n_components} "
                  f"({cumsum[self.n_components-1]:.1%} de variance)")
        
        # PCA final
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        return self
    
    def transform(self, X):
        """Transforme les données en composantes principales."""
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, X):
        """Fit et transforme."""
        self.fit(X)
        return self.transform(X)
    
    def get_loadings(self):
        """
        Retourne les loadings (poids des features originales).
        
        Returns:
            DataFrame: Loadings par composante
        """
        loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_names or range(len(self.pca.components_[0])),
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        return loadings
    
    def explained_variance(self):
        """Retourne la variance expliquée par composante."""
        return pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f'PC{i+1}' for i in range(self.n_components)]
        )
    
    def plot_explained_variance(self):
        """Affiche la variance expliquée."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Variance par composante
        ax1 = axes[0]
        var_ratio = self.pca.explained_variance_ratio_
        ax1.bar(range(1, len(var_ratio)+1), var_ratio, alpha=0.7)
        ax1.set_xlabel('Composante')
        ax1.set_ylabel('Variance Expliquée')
        ax1.set_title('Variance par Composante')
        
        # Variance cumulée
        ax2 = axes[1]
        cumsum = np.cumsum(var_ratio)
        ax2.plot(range(1, len(cumsum)+1), cumsum, 'bo-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax2.set_xlabel('Nombre de Composantes')
        ax2.set_ylabel('Variance Cumulée')
        ax2.set_title('Variance Cumulée')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


def extract_risk_factors(returns_df, n_factors=5):
    """
    Extrait les facteurs de risque à partir des rendements.
    
    Args:
        returns_df: DataFrame de rendements (colonnes = actifs)
        n_factors: Nombre de facteurs à extraire
    
    Returns:
        tuple: (facteurs, loadings, pca_object)
    """
    pca = FinancialPCA(n_components=n_factors)
    factors = pca.fit_transform(returns_df)
    
    factors_df = pd.DataFrame(
        factors,
        index=returns_df.index,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    
    loadings = pca.get_loadings()
    
    return factors_df, loadings, pca


def build_eigenportfolios(returns_df, n_portfolios=5):
    """
    Construit des eigenportfolios à partir de PCA.
    
    Les eigenportfolios sont des portefeuilles dont les poids
    correspondent aux loadings des composantes principales.
    
    Args:
        returns_df: DataFrame de rendements
        n_portfolios: Nombre d'eigenportfolios
    
    Returns:
        DataFrame: Rendements des eigenportfolios
    """
    pca = FinancialPCA(n_components=n_portfolios)
    pca.fit(returns_df)
    
    # Les loadings sont les poids (normalisés)
    loadings = pca.get_loadings()
    
    # Calculer les rendements des eigenportfolios
    eigen_returns = pd.DataFrame(index=returns_df.index)
    
    for i in range(n_portfolios):
        weights = loadings[f'PC{i+1}'].values
        weights = weights / np.sum(np.abs(weights))  # Normaliser
        
        eigen_returns[f'EigenPF_{i+1}'] = returns_df.dot(weights)
    
    return eigen_returns, loadings


# === Exemple ===
def demonstrate_pca():
    """Démontre PCA pour la finance."""
    import yfinance as yf
    
    # Télécharger des données
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'JPM', 'BAC', 'GS', 'XOM', 'CVX']
    data = yf.download(tickers, start='2020-01-01', end='2023-12-31')['Adj Close']
    returns = data.pct_change().dropna()
    
    print("PCA sur les rendements boursiers")
    print("=" * 50)
    
    # Extraire les facteurs
    factors, loadings, pca = extract_risk_factors(returns, n_factors=3)
    
    print(f"\nVariance expliquée:")
    for comp, var in pca.explained_variance().items():
        print(f"  {comp}: {var:.1%}")
    
    print(f"\nLoadings PC1 (facteur marché):")
    pc1_loadings = loadings['PC1'].sort_values(ascending=False)
    for ticker, loading in pc1_loadings.items():
        print(f"  {ticker}: {loading:.3f}")
    
    # Eigenportfolios
    eigen_ret, _ = build_eigenportfolios(returns, n_portfolios=3)
    
    print(f"\nPerformance des Eigenportfolios:")
    for col in eigen_ret.columns:
        sharpe = np.sqrt(252) * eigen_ret[col].mean() / eigen_ret[col].std()
        print(f"  {col}: Sharpe = {sharpe:.2f}")
    
    # Plot
    pca.plot_explained_variance()
    
    return factors, loadings, pca


# factors, loadings, pca = demonstrate_pca()
```

## 13.2 Clustering (K-Means, Hierarchical)

```python
"""
Clustering pour la Finance
==========================
Le clustering groupe les actifs similaires ensemble.

Applications:
- Allocation d'actifs
- Détection de régimes de marché
- Diversification de portefeuille
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def cluster_assets(returns_df, n_clusters=5, method='kmeans'):
    """
    Cluster les actifs basé sur leurs rendements.
    
    Args:
        returns_df: DataFrame de rendements
        n_clusters: Nombre de clusters
        method: 'kmeans', 'hierarchical', ou 'dbscan'
    
    Returns:
        dict: Assignments et statistiques
    """
    # Calculer les features de chaque actif
    features = pd.DataFrame(index=returns_df.columns)
    features['mean_return'] = returns_df.mean() * 252
    features['volatility'] = returns_df.std() * np.sqrt(252)
    features['sharpe'] = features['mean_return'] / features['volatility']
    features['skewness'] = returns_df.skew()
    features['kurtosis'] = returns_df.kurtosis()
    
    # Standardiser
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Clustering
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=2)
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    labels = model.fit_predict(features_scaled)
    
    # Silhouette score
    if len(set(labels)) > 1:
        sil_score = silhouette_score(features_scaled, labels)
    else:
        sil_score = None
    
    # Résultats
    features['cluster'] = labels
    
    return {
        'features': features,
        'labels': labels,
        'silhouette_score': sil_score,
        'cluster_stats': features.groupby('cluster').mean()
    }


def plot_hierarchical_clustering(returns_df, method='ward'):
    """
    Affiche le dendrogramme du clustering hiérarchique.
    
    Args:
        returns_df: DataFrame de rendements
        method: Méthode de linkage ('ward', 'complete', 'average', 'single')
    """
    # Calculer la matrice de corrélation
    corr = returns_df.corr()
    
    # Convertir en distance
    distance = 1 - corr
    
    # Linkage
    Z = linkage(distance, method=method)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    dendrogram(Z, labels=returns_df.columns, ax=ax, leaf_rotation=90)
    ax.set_title(f'Hierarchical Clustering (method={method})')
    ax.set_ylabel('Distance')
    plt.tight_layout()
    plt.show()
    
    return Z


def find_optimal_clusters(returns_df, max_clusters=10):
    """
    Trouve le nombre optimal de clusters.
    
    Args:
        returns_df: DataFrame de rendements
        max_clusters: Nombre maximum de clusters à tester
    
    Returns:
        dict: Scores pour chaque k
    """
    # Features
    features = pd.DataFrame(index=returns_df.columns)
    features['mean_return'] = returns_df.mean() * 252
    features['volatility'] = returns_df.std() * np.sqrt(252)
    features['sharpe'] = features['mean_return'] / features['volatility']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {'k': [], 'inertia': [], 'silhouette': []}
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(features_scaled, labels))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    axes[0].plot(results['k'], results['inertia'], 'bo-')
    axes[0].set_xlabel('Nombre de clusters')
    axes[0].set_ylabel('Inertie')
    axes[0].set_title('Méthode du coude (Elbow)')
    
    # Silhouette
    axes[1].plot(results['k'], results['silhouette'], 'go-')
    axes[1].set_xlabel('Nombre de clusters')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Score Silhouette')
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(results)
```

## 13.3 Hierarchical Risk Parity (HRP)

```python
"""
Hierarchical Risk Parity (HRP)
==============================
HRP est une méthode d'allocation de portefeuille qui utilise
le clustering hiérarchique pour construire des portefeuilles diversifiés.

Avantages vs Markowitz:
- Pas d'inversion de matrice (plus stable)
- Meilleures performances out-of-sample
- Plus intuitif (diversification hiérarchique)

Algorithme:
1. Tree Clustering: Grouper les actifs par corrélation
2. Quasi-Diagonalization: Réorganiser la matrice de covariance
3. Recursive Bisection: Allouer le capital récursivement
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

class HierarchicalRiskParity:
    """
    Implémentation de Hierarchical Risk Parity.
    """
    
    def __init__(self):
        self.weights = None
        self.linkage = None
        self.sorted_indices = None
    
    def fit(self, returns):
        """
        Calcule les poids HRP.
        
        Args:
            returns: DataFrame de rendements
        
        Returns:
            Series: Poids optimaux
        """
        # Étape 1: Matrice de corrélation et covariance
        corr = returns.corr()
        cov = returns.cov()
        
        # Étape 2: Tree Clustering
        dist = self._correlation_distance(corr)
        self.linkage = linkage(squareform(dist), method='single')
        
        # Étape 3: Quasi-Diagonalization
        self.sorted_indices = self._get_quasi_diag(self.linkage)
        
        # Étape 4: Recursive Bisection
        weights = self._recursive_bisection(cov, self.sorted_indices)
        
        self.weights = pd.Series(weights, index=returns.columns)
        
        return self.weights
    
    def _correlation_distance(self, corr):
        """
        Convertit la corrélation en distance.
        
        d = sqrt(0.5 * (1 - corr))
        """
        return np.sqrt(0.5 * (1 - corr))
    
    def _get_quasi_diag(self, link):
        """
        Réorganise les indices pour quasi-diagonaliser la matrice.
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        
        return sort_ix.tolist()
    
    def _recursive_bisection(self, cov, sorted_indices):
        """
        Alloue le capital récursivement.
        """
        weights = pd.Series(1.0, index=sorted_indices)
        clusters = [sorted_indices]
        
        while len(clusters) > 0:
            # Bisection
            clusters = [
                cluster[j:k]
                for cluster in clusters
                for j, k in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                if len(cluster) > 1
            ]
            
            for i in range(0, len(clusters), 2):
                if i + 1 < len(clusters):
                    cluster0 = clusters[i]
                    cluster1 = clusters[i + 1]
                    
                    # Variance de chaque cluster
                    var0 = self._get_cluster_var(cov, cluster0)
                    var1 = self._get_cluster_var(cov, cluster1)
                    
                    # Allocation inversement proportionnelle à la variance
                    alpha = 1 - var0 / (var0 + var1)
                    
                    weights[cluster0] *= alpha
                    weights[cluster1] *= 1 - alpha
        
        return weights
    
    def _get_cluster_var(self, cov, cluster_items):
        """
        Calcule la variance d'un cluster.
        
        Utilise l'inverse-variance weighting au sein du cluster.
        """
        cov_slice = cov.iloc[cluster_items, cluster_items]
        
        # Poids inverse-variance
        ivp = 1 / np.diag(cov_slice)
        ivp /= ivp.sum()
        
        # Variance du cluster
        return np.dot(ivp, np.dot(cov_slice, ivp))
    
    def plot_dendrogram(self, labels=None):
        """Affiche le dendrogramme."""
        if self.linkage is None:
            raise ValueError("Appelez fit() d'abord")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        dendrogram(self.linkage, labels=labels, ax=ax, leaf_rotation=90)
        ax.set_title('Hierarchical Risk Parity - Dendrogram')
        plt.tight_layout()
        plt.show()


def compare_portfolio_methods(returns, risk_free_rate=0.02):
    """
    Compare différentes méthodes d'allocation.
    
    Args:
        returns: DataFrame de rendements
        risk_free_rate: Taux sans risque
    
    Returns:
        DataFrame: Comparaison des méthodes
    """
    results = {}
    
    # 1. Equal Weight
    n = len(returns.columns)
    ew_weights = pd.Series(1/n, index=returns.columns)
    ew_returns = returns.dot(ew_weights)
    results['Equal Weight'] = {
        'weights': ew_weights,
        'return': ew_returns.mean() * 252,
        'volatility': ew_returns.std() * np.sqrt(252),
    }
    results['Equal Weight']['sharpe'] = (
        (results['Equal Weight']['return'] - risk_free_rate) / 
        results['Equal Weight']['volatility']
    )
    
    # 2. HRP
    hrp = HierarchicalRiskParity()
    hrp_weights = hrp.fit(returns)
    hrp_returns = returns.dot(hrp_weights)
    results['HRP'] = {
        'weights': hrp_weights,
        'return': hrp_returns.mean() * 252,
        'volatility': hrp_returns.std() * np.sqrt(252),
    }
    results['HRP']['sharpe'] = (
        (results['HRP']['return'] - risk_free_rate) / 
        results['HRP']['volatility']
    )
    
    # 3. Inverse Volatility
    vol = returns.std()
    iv_weights = (1/vol) / (1/vol).sum()
    iv_returns = returns.dot(iv_weights)
    results['Inverse Vol'] = {
        'weights': iv_weights,
        'return': iv_returns.mean() * 252,
        'volatility': iv_returns.std() * np.sqrt(252),
    }
    results['Inverse Vol']['sharpe'] = (
        (results['Inverse Vol']['return'] - risk_free_rate) / 
        results['Inverse Vol']['volatility']
    )
    
    # Afficher les résultats
    print("Comparaison des méthodes d'allocation")
    print("=" * 60)
    
    for method, res in results.items():
        print(f"\n{method}:")
        print(f"  Rendement: {res['return']:.2%}")
        print(f"  Volatilité: {res['volatility']:.2%}")
        print(f"  Sharpe: {res['sharpe']:.2f}")
    
    return results


# === Exemple ===
def demonstrate_hrp():
    """Démontre HRP."""
    import yfinance as yf
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'BAC', 'XOM', 'CVX', 'JNJ', 'PG']
    data = yf.download(tickers, start='2019-01-01', end='2023-12-31')['Adj Close']
    returns = data.pct_change().dropna()
    
    # HRP
    hrp = HierarchicalRiskParity()
    weights = hrp.fit(returns)
    
    print("Poids HRP:")
    print("-" * 30)
    for ticker, weight in weights.sort_values(ascending=False).items():
        print(f"  {ticker}: {weight:.2%}")
    
    # Dendrogramme
    hrp.plot_dendrogram(labels=returns.columns.tolist())
    
    # Comparaison
    results = compare_portfolio_methods(returns)
    
    return hrp, results


# hrp, results = demonstrate_hrp()
```

---

# 14. TRAITEMENT DU LANGAGE NATUREL (NLP)
## Natural Language Processing

## 14.1 Pipeline NLP avec spaCy

```python
"""
NLP pour la Finance avec spaCy
==============================
Le NLP permet d'extraire des informations des textes financiers:
- Rapports d'analystes
- Articles de presse
- Filings SEC
- Réseaux sociaux
- Earnings calls transcripts

Pipeline NLP typique:
1. Tokenization: Découper en mots/phrases
2. POS Tagging: Identifier les parties du discours
3. NER: Extraire les entités nommées
4. Lemmatization: Réduire aux formes de base
5. Sentiment Analysis: Évaluer la tonalité
"""
import spacy
import pandas as pd
import numpy as np

# Charger le modèle spaCy
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

def analyze_text(text):
    """
    Analyse complète d'un texte avec spaCy.
    
    Args:
        text: Texte à analyser
    
    Returns:
        dict: Résultats de l'analyse
    """
    doc = nlp(text)
    
    # Tokens avec leurs propriétés
    tokens = [{
        'text': token.text,
        'lemma': token.lemma_,
        'pos': token.pos_,          # Part of Speech (Noun, Verb, etc.)
        'tag': token.tag_,          # Fine-grained POS
        'dep': token.dep_,          # Dependency relation
        'is_stop': token.is_stop,   # Est-ce un stopword?
        'is_alpha': token.is_alpha  # Est-ce alphabétique?
    } for token in doc]
    
    # Entités nommées
    entities = [{
        'text': ent.text,
        'label': ent.label_,        # Type d'entité (ORG, PERSON, MONEY, etc.)
        'start': ent.start_char,
        'end': ent.end_char
    } for ent in doc.ents]
    
    # Phrases
    sentences = [sent.text for sent in doc.sents]
    
    # Noun chunks (groupes nominaux)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    
    return {
        'tokens': tokens,
        'entities': entities,
        'sentences': sentences,
        'noun_chunks': noun_chunks,
        'n_tokens': len(doc),
        'n_sentences': len(sentences),
        'n_entities': len(entities)
    }


def extract_financial_entities(text):
    """
    Extrait les entités financières d'un texte.
    
    Args:
        text: Texte financier
    
    Returns:
        dict: Entités par catégorie
    """
    doc = nlp(text)
    
    entities = {
        'MONEY': [],      # Montants ($10 million)
        'PERCENT': [],    # Pourcentages (15%)
        'ORG': [],        # Organisations (Apple Inc.)
        'PERSON': [],     # Personnes (Tim Cook)
        'DATE': [],       # Dates (Q3 2023)
        'GPE': [],        # Pays/Villes (United States)
        'CARDINAL': [],   # Nombres
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities


# === TextBlob pour Sentiment ===
from textblob import TextBlob

def analyze_sentiment_textblob(text):
    """
    Analyse le sentiment avec TextBlob.
    
    Args:
        text: Texte à analyser
    
    Returns:
        dict: Polarité (-1 à 1) et subjectivité (0 à 1)
    """
    blob = TextBlob(text)
    
    return {
        'polarity': blob.sentiment.polarity,      # -1 (négatif) à 1 (positif)
        'subjectivity': blob.sentiment.subjectivity,  # 0 (objectif) à 1 (subjectif)
        'sentences': [{
            'text': str(sentence),
            'polarity': sentence.sentiment.polarity,
            'subjectivity': sentence.sentiment.subjectivity
        } for sentence in blob.sentences]
    }


def batch_sentiment_analysis(texts):
    """
    Analyse le sentiment d'une liste de textes.
    
    Args:
        texts: Liste de textes
    
    Returns:
        DataFrame: Résultats d'analyse
    """
    results = []
    
    for text in texts:
        sentiment = analyze_sentiment_textblob(text)
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'polarity': sentiment['polarity'],
            'subjectivity': sentiment['subjectivity'],
            'sentiment_label': 'positive' if sentiment['polarity'] > 0.1 else 
                              ('negative' if sentiment['polarity'] < -0.1 else 'neutral')
        })
    
    return pd.DataFrame(results)


# === Exemple ===
def demonstrate_nlp():
    """Démontre le NLP financier."""
    
    # Texte d'exemple (earnings call fictif)
    text = """
    Apple Inc. reported strong Q3 2023 results with revenue of $81.8 billion, 
    up 5% year-over-year. CEO Tim Cook stated that iPhone sales exceeded 
    expectations, particularly in China. The company announced a $90 billion 
    share buyback program. Analysts from Goldman Sachs raised their price 
    target to $200. However, iPad sales declined 8% due to supply chain issues.
    """
    
    print("Analyse NLP du texte financier")
    print("=" * 60)
    
    # Entités
    entities = extract_financial_entities(text)
    print("\nEntités extraites:")
    for entity_type, values in entities.items():
        if values:
            print(f"  {entity_type}: {', '.join(set(values))}")
    
    # Sentiment
    sentiment = analyze_sentiment_textblob(text)
    print(f"\nSentiment global:")
    print(f"  Polarité: {sentiment['polarity']:.2f}")
    print(f"  Subjectivité: {sentiment['subjectivity']:.2f}")
    
    print(f"\nSentiment par phrase:")
    for sent in sentiment['sentences']:
        label = 'positive' if sent['polarity'] > 0.1 else ('negative' if sent['polarity'] < -0.1 else 'neutral')
        print(f"  [{label:8}] {sent['text'][:60]}...")
    
    return entities, sentiment


# entities, sentiment = demonstrate_nlp()
```

## 14.2 Document-Term Matrix et TF-IDF

```python
"""
Représentation Vectorielle des Documents
========================================
Pour utiliser le ML sur du texte, il faut convertir les documents
en vecteurs numériques.

Méthodes:
1. Bag of Words (BoW): Compte des mots
2. TF-IDF: Pondération par importance relative
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np

def create_document_term_matrix(documents, method='tfidf', 
                                 max_features=1000, ngram_range=(1, 2)):
    """
    Crée une matrice document-terme.
    
    Args:
        documents: Liste de textes
        method: 'count' (BoW) ou 'tfidf'
        max_features: Nombre maximum de features
        ngram_range: Range de n-grams (1,1)=unigrams, (1,2)=uni+bigrams
    
    Returns:
        tuple: (matrice, vectorizer)
    """
    if method == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,           # Apparaît dans au moins 2 docs
            max_df=0.95         # Pas plus de 95% des docs
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True   # Utilise 1 + log(tf) au lieu de tf
        )
    
    dtm = vectorizer.fit_transform(documents)
    
    # Convertir en DataFrame pour visualisation
    feature_names = vectorizer.get_feature_names_out()
    dtm_df = pd.DataFrame(
        dtm.toarray(),
        columns=feature_names
    )
    
    return dtm_df, vectorizer


def get_top_terms(dtm_df, n_terms=20):
    """
    Retourne les termes les plus fréquents.
    
    Args:
        dtm_df: Document-term matrix
        n_terms: Nombre de termes
    
    Returns:
        Series: Top termes avec leurs scores
    """
    term_freq = dtm_df.sum().sort_values(ascending=False)
    return term_freq.head(n_terms)


def classify_documents(train_texts, train_labels, test_texts, vectorizer_type='tfidf'):
    """
    Classification de documents avec Naive Bayes.
    
    Args:
        train_texts: Textes d'entraînement
        train_labels: Labels
        test_texts: Textes de test
        vectorizer_type: 'tfidf' ou 'count'
    
    Returns:
        tuple: (predictions, probabilities)
    """
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    
    # Pipeline
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    else:
        vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB())
    ])
    
    # Entraîner
    pipeline.fit(train_texts, train_labels)
    
    # Prédire
    predictions = pipeline.predict(test_texts)
    probabilities = pipeline.predict_proba(test_texts)
    
    return predictions, probabilities, pipeline
```

---

# 15. TOPIC MODELING
## Topic Modeling

## 15.1 LDA (Latent Dirichlet Allocation)

```python
"""
LDA pour l'Analyse de Thèmes Financiers
=======================================
LDA découvre automatiquement les thèmes latents dans un corpus de documents.

Applications en finance:
- Analyser les thèmes des earnings calls
- Détecter les sujets d'actualité
- Classifier les articles de presse
- Identifier les préoccupations des investisseurs
"""
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

class LDATopicModel:
    """
    Modèle LDA pour découverte de thèmes.
    """
    
    def __init__(self, n_topics=10, max_features=5000, n_top_words=10):
        """
        Initialise LDA.
        
        Args:
            n_topics: Nombre de thèmes
            max_features: Vocabulaire maximum
            n_top_words: Mots par thème à afficher
        """
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=5,
            max_df=0.9
        )
        
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=20,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        self.feature_names = None
    
    def fit(self, documents):
        """
        Entraîne LDA sur les documents.
        
        Args:
            documents: Liste de textes
        """
        # Vectoriser
        dtm = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Entraîner LDA
        self.lda.fit(dtm)
        
        return self
    
    def transform(self, documents):
        """
        Calcule la distribution de thèmes pour les documents.
        
        Args:
            documents: Liste de textes
        
        Returns:
            array: Distribution de thèmes (n_docs x n_topics)
        """
        dtm = self.vectorizer.transform(documents)
        return self.lda.transform(dtm)
    
    def get_topics(self):
        """
        Retourne les thèmes avec leurs mots clés.
        
        Returns:
            dict: Thèmes avec mots et poids
        """
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_indices = topic.argsort()[:-self.n_top_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_indices]
            top_weights = topic[top_indices]
            
            topics[f'Topic_{topic_idx}'] = {
                'words': top_words,
                'weights': top_weights.tolist()
            }
        
        return topics
    
    def print_topics(self):
        """Affiche les thèmes."""
        topics = self.get_topics()
        
        print("="*60)
        print("THÈMES DÉCOUVERTS")
        print("="*60)
        
        for topic_name, topic_data in topics.items():
            print(f"\n{topic_name}:")
            for word, weight in zip(topic_data['words'], topic_data['weights']):
                print(f"  {word}: {weight:.3f}")
    
    def get_document_topics(self, documents, threshold=0.1):
        """
        Assigne les thèmes dominants à chaque document.
        
        Args:
            documents: Liste de textes
            threshold: Seuil minimum pour considérer un thème
        
        Returns:
            DataFrame: Thèmes par document
        """
        topic_dist = self.transform(documents)
        
        results = []
        for i, dist in enumerate(topic_dist):
            dominant_topic = np.argmax(dist)
            dominant_prob = dist[dominant_topic]
            
            # Tous les thèmes au-dessus du seuil
            significant_topics = [
                f'Topic_{j}' for j, p in enumerate(dist) if p >= threshold
            ]
            
            results.append({
                'document_idx': i,
                'dominant_topic': f'Topic_{dominant_topic}',
                'dominant_prob': dominant_prob,
                'significant_topics': significant_topics
            })
        
        return pd.DataFrame(results)


# === Exemple avec Gensim (plus avancé) ===
"""
Gensim offre plus de contrôle et de fonctionnalités pour LDA.
"""
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

def train_lda_gensim(documents, n_topics=10, passes=10):
    """
    Entraîne LDA avec Gensim.
    
    Args:
        documents: Liste de documents (chaque doc = liste de mots)
        n_topics: Nombre de thèmes
        passes: Nombre de passes sur le corpus
    
    Returns:
        tuple: (model, dictionary, corpus)
    """
    # Créer le dictionnaire
    dictionary = corpora.Dictionary(documents)
    
    # Filtrer les termes rares et trop fréquents
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # Créer le corpus (bag of words)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    
    # Entraîner LDA
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=passes,
        workers=4,
        random_state=42
    )
    
    # Cohérence (mesure de qualité)
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=documents,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence = coherence_model.get_coherence()
    
    print(f"Coherence Score: {coherence:.4f}")
    
    return lda_model, dictionary, corpus


def find_optimal_topics(documents, min_topics=5, max_topics=20, step=2):
    """
    Trouve le nombre optimal de thèmes par cohérence.
    
    Args:
        documents: Liste de documents tokenisés
        min_topics: Minimum de thèmes
        max_topics: Maximum de thèmes
        step: Pas
    
    Returns:
        dict: Scores de cohérence par nombre de thèmes
    """
    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    
    coherence_scores = {}
    
    for n in range(min_topics, max_topics + 1, step):
        print(f"Testing {n} topics...")
        
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n,
            passes=5,
            workers=4,
            random_state=42
        )
        
        cm = CoherenceModel(
            model=model,
            texts=documents,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        coherence_scores[n] = cm.get_coherence()
    
    # Meilleur nombre
    best_n = max(coherence_scores, key=coherence_scores.get)
    print(f"\nMeilleur nombre de thèmes: {best_n} (coherence: {coherence_scores[best_n]:.4f})")
    
    return coherence_scores
```

---

# 16. WORD EMBEDDINGS
## Word Embeddings

## 16.1 Word2Vec

```python
"""
Word Embeddings pour la Finance
===============================
Les word embeddings représentent les mots comme des vecteurs denses
capturant les relations sémantiques.

word2vec: "king" - "man" + "woman" ≈ "queen"
finance:  "stock" - "equity" + "debt" ≈ "bond"

Avantages:
- Capture les relations sémantiques
- Réduit la dimensionnalité
- Permet le transfer learning
"""
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd

def train_word2vec(sentences, vector_size=100, window=5, min_count=5,
                   workers=4, epochs=10):
    """
    Entraîne un modèle Word2Vec.
    
    Args:
        sentences: Liste de phrases tokenisées
        vector_size: Dimension des vecteurs
        window: Taille de la fenêtre de contexte
        min_count: Fréquence minimum
        workers: Nombre de workers
        epochs: Nombre d'époques
    
    Returns:
        Word2Vec: Modèle entraîné
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,  # Skip-gram (1) vs CBOW (0)
        seed=42
    )
    
    return model


def load_pretrained_embeddings(path):
    """
    Charge des embeddings pré-entraînés.
    
    Args:
        path: Chemin vers le fichier (GloVe, Word2Vec, etc.)
    
    Returns:
        KeyedVectors: Vecteurs de mots
    """
    # Pour GloVe format texte:
    # from gensim.scripts.glove2word2vec import glove2word2vec
    # glove2word2vec('glove.6B.100d.txt', 'glove.6B.100d.w2v.txt')
    
    return KeyedVectors.load_word2vec_format(path, binary=False)


class FinancialWordEmbeddings:
    """
    Word embeddings spécialisés pour la finance.
    """
    
    def __init__(self, model):
        """
        Initialise avec un modèle Word2Vec.
        
        Args:
            model: Word2Vec ou KeyedVectors
        """
        if isinstance(model, Word2Vec):
            self.wv = model.wv
        else:
            self.wv = model
    
    def get_vector(self, word):
        """Retourne le vecteur d'un mot."""
        try:
            return self.wv[word]
        except KeyError:
            return None
    
    def most_similar(self, word, topn=10):
        """
        Trouve les mots les plus similaires.
        
        Args:
            word: Mot de référence
            topn: Nombre de résultats
        
        Returns:
            list: Mots similaires avec scores
        """
        try:
            return self.wv.most_similar(word, topn=topn)
        except KeyError:
            return []
    
    def analogy(self, positive, negative, topn=5):
        """
        Résout une analogie: positive[0] - negative[0] + positive[1] = ?
        
        Exemple: king - man + woman = queen
        
        Args:
            positive: Liste de mots à ajouter
            negative: Liste de mots à soustraire
            topn: Nombre de résultats
        
        Returns:
            list: Résultats de l'analogie
        """
        try:
            return self.wv.most_similar(
                positive=positive,
                negative=negative,
                topn=topn
            )
        except KeyError:
            return []
    
    def document_vector(self, tokens, method='mean'):
        """
        Calcule le vecteur d'un document.
        
        Args:
            tokens: Liste de mots
            method: 'mean' ou 'sum'
        
        Returns:
            array: Vecteur du document
        """
        vectors = []
        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            return np.zeros(self.wv.vector_size)
        
        vectors = np.array(vectors)
        
        if method == 'mean':
            return vectors.mean(axis=0)
        return vectors.sum(axis=0)
    
    def similarity(self, word1, word2):
        """Calcule la similarité cosinus entre deux mots."""
        try:
            return self.wv.similarity(word1, word2)
        except KeyError:
            return 0.0


# === Doc2Vec pour documents ===
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def train_doc2vec(documents, vector_size=100, epochs=20):
    """
    Entraîne Doc2Vec pour représenter des documents entiers.
    
    Args:
        documents: Liste de (tokens, tag)
        vector_size: Dimension des vecteurs
        epochs: Nombre d'époques
    
    Returns:
        Doc2Vec: Modèle entraîné
    """
    # Préparer les documents tagués
    tagged_docs = [
        TaggedDocument(words=tokens, tags=[str(i)])
        for i, tokens in enumerate(documents)
    ]
    
    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=vector_size,
        window=5,
        min_count=5,
        workers=4,
        epochs=epochs,
        dm=1  # Distributed Memory (1) vs DBOW (0)
    )
    
    return model


# === Exemple ===
def demonstrate_embeddings():
    """Démontre les word embeddings."""
    
    # Corpus d'exemple (phrases tokenisées)
    sentences = [
        ['stock', 'price', 'increased', 'today'],
        ['bond', 'yields', 'fell', 'sharply'],
        ['fed', 'raised', 'interest', 'rates'],
        ['earnings', 'beat', 'expectations'],
        ['market', 'crashed', 'on', 'news'],
        ['investors', 'bought', 'stocks'],
        ['company', 'announced', 'dividend'],
        ['volatility', 'increased', 'significantly'],
    ] * 100  # Répéter pour avoir assez de données
    
    # Entraîner Word2Vec
    print("Entraînement Word2Vec...")
    model = train_word2vec(sentences, vector_size=50, epochs=20)
    
    embeddings = FinancialWordEmbeddings(model)
    
    print("\nMots similaires à 'stock':")
    for word, score in embeddings.most_similar('stock', topn=5):
        print(f"  {word}: {score:.3f}")
    
    print(f"\nSimilarité 'stock' - 'market': {embeddings.similarity('stock', 'market'):.3f}")
    
    # Vecteur de document
    doc = ['stock', 'price', 'increased']
    doc_vec = embeddings.document_vector(doc)
    print(f"\nVecteur du document (dim {len(doc_vec)}): {doc_vec[:5]}...")
    
    return model, embeddings


# model, embeddings = demonstrate_embeddings()
```

---

# 17. DEEP LEARNING - RÉSEAUX FEEDFORWARD
## Deep Learning - Feedforward Networks

## 17.1 Introduction au Deep Learning

```python
"""
Deep Learning pour le Trading
=============================
Les réseaux de neurones profonds peuvent capturer des patterns
complexes et non-linéaires dans les données financières.

Architecture de base:
Input → Hidden Layers → Output

Chaque couche: z = W·x + b, a = activation(z)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class TradingNeuralNetwork:
    """
    Réseau de neurones pour prédiction de trading.
    """
    
    def __init__(self, input_dim, task='classification', 
                 hidden_layers=[64, 32], dropout_rate=0.3,
                 learning_rate=0.001):
        """
        Initialise le réseau.
        
        Args:
            input_dim: Dimension d'entrée
            task: 'classification' ou 'regression'
            hidden_layers: Liste des tailles de couches cachées
            dropout_rate: Taux de dropout
            learning_rate: Taux d'apprentissage
        """
        self.input_dim = input_dim
        self.task = task
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
    
    def _build_model(self):
        """Construit le modèle Keras."""
        model = keras.Sequential()
        
        # Première couche
        model.add(layers.Dense(
            self.hidden_layers[0],
            activation='relu',
            input_shape=(self.input_dim,)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.dropout_rate))
        
        # Couches cachées
        for units in self.hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))
        
        # Couche de sortie
        if self.task == 'classification':
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            model.add(layers.Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compiler
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32, early_stopping=True):
        """
        Entraîne le modèle.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation
            y_val: Target de validation
            epochs: Nombre d'époques
            batch_size: Taille des batches
            early_stopping: Arrêt anticipé
        """
        callbacks = []
        
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ))
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Prédit."""
        preds = self.model.predict(X)
        
        if self.task == 'classification':
            return (preds > 0.5).astype(int).flatten()
        return preds.flatten()
    
    def predict_proba(self, X):
        """Prédit les probabilités."""
        return self.model.predict(X).flatten()
    
    def evaluate(self, X, y):
        """Évalue le modèle."""
        return self.model.evaluate(X, y, verbose=0)
    
    def summary(self):
        """Affiche le résumé du modèle."""
        return self.model.summary()


# === PyTorch Alternative ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TradingNetPyTorch(nn.Module):
    """
    Réseau de neurones PyTorch pour le trading.
    """
    
    def __init__(self, input_dim, hidden_layers=[64, 32], dropout_rate=0.3):
        super().__init__()
        
        layers_list = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers_list.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers_list.append(nn.Linear(prev_dim, 1))
        layers_list.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers_list)
    
    def forward(self, x):
        return self.network(x)


def train_pytorch_model(model, X_train, y_train, X_val=None, y_val=None,
                        epochs=100, batch_size=32, lr=0.001):
    """
    Entraîne un modèle PyTorch.
    
    Args:
        model: Modèle PyTorch
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        epochs: Nombre d'époques
        batch_size: Taille des batches
        lr: Learning rate
    
    Returns:
        dict: Historique d'entraînement
    """
    # Convertir en tenseurs
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimiseur et loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val)
                y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}", end='')
            if X_val is not None:
                print(f", Val Loss: {val_loss:.4f}")
            else:
                print()
    
    return history


# === Exemple ===
def demonstrate_neural_network():
    """Démontre les réseaux de neurones pour le trading."""
    np.random.seed(42)
    n = 5000
    
    # Créer des données
    X = np.random.randn(n, 10)
    y = ((X[:, 0] > 0.5) & (X[:, 1] < 0) | (X[:, 2] > 1)).astype(int)
    
    # Split
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size+val_size]
    y_test = y[train_size+val_size:]
    
    # Normaliser
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Entraîner (Keras)
    print("Entraînement Keras...")
    nn = TradingNeuralNetwork(
        input_dim=10,
        task='classification',
        hidden_layers=[64, 32, 16]
    )
    nn.summary()
    
    history = nn.fit(X_train, y_train, X_val, y_val, epochs=50)
    
    # Évaluer
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    y_pred = nn.predict(X_test)
    y_proba = nn.predict_proba(X_test)
    
    print(f"\nRésultats:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    return nn, history


# nn, history = demonstrate_neural_network()
```

---

# 18. RÉSEAUX DE NEURONES CONVOLUTIONNELS (CNN)
## Convolutional Neural Networks

## 18.1 CNN pour Séries Temporelles Financières

```python
"""
CNN pour le Trading
===================
Les CNN peuvent extraire des patterns locaux dans les séries temporelles
en traitant les données comme des "images" 1D ou 2D.

Applications:
- Détection de patterns techniques (head & shoulders, etc.)
- Extraction de features à partir de données OHLCV
- Classification d'images de graphiques
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

class CNNTimeSeries:
    """
    CNN pour séries temporelles financières.
    """
    
    def __init__(self, sequence_length, n_features, task='classification',
                 conv_filters=[64, 128], kernel_size=3, pool_size=2,
                 dense_units=[64], dropout_rate=0.3):
        """
        Initialise le CNN.
        
        Args:
            sequence_length: Longueur de la séquence
            n_features: Nombre de features (ex: OHLCV = 5)
            task: 'classification' ou 'regression'
            conv_filters: Filtres par couche conv
            kernel_size: Taille du kernel
            pool_size: Taille du pooling
            dense_units: Unités des couches denses
            dropout_rate: Taux de dropout
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.task = task
        
        self.model = self._build_model(
            conv_filters, kernel_size, pool_size, 
            dense_units, dropout_rate
        )
    
    def _build_model(self, conv_filters, kernel_size, pool_size,
                     dense_units, dropout_rate):
        """Construit le modèle CNN."""
        model = tf.keras.Sequential()
        
        # Input shape: (sequence_length, n_features)
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))
        
        # Couches convolutionnelles
        for i, filters in enumerate(conv_filters):
            model.add(layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu'
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling1D(pool_size=pool_size))
            model.add(layers.Dropout(dropout_rate))
        
        # Flatten
        model.add(layers.GlobalAveragePooling1D())
        
        # Couches denses
        for units in dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output
        if self.task == 'classification':
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(layers.Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def prepare_data(self, data, target_col, lookback):
        """
        Prépare les données pour le CNN.
        
        Args:
            data: DataFrame OHLCV
            target_col: Nom de la colonne cible
            lookback: Fenêtre de lookback
        
        Returns:
            tuple: (X, y)
        """
        X, y = [], []
        
        feature_cols = [c for c in data.columns if c != target_col]
        
        for i in range(lookback, len(data)):
            X.append(data[feature_cols].iloc[i-lookback:i].values)
            y.append(data[target_col].iloc[i])
        
        return np.array(X), np.array(y)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32):
        """Entraîne le modèle."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Prédit."""
        preds = self.model.predict(X)
        if self.task == 'classification':
            return (preds > 0.5).astype(int).flatten()
        return preds.flatten()
    
    def predict_proba(self, X):
        """Prédit les probabilités."""
        return self.model.predict(X).flatten()


def create_image_from_ohlcv(ohlcv_data, image_size=(64, 64)):
    """
    Convertit des données OHLCV en image pour CNN 2D.
    
    Args:
        ohlcv_data: DataFrame avec colonnes OHLCV
        image_size: Taille de l'image (height, width)
    
    Returns:
        array: Image normalisée
    """
    import cv2
    
    # Normaliser les données
    normalized = (ohlcv_data - ohlcv_data.min()) / (ohlcv_data.max() - ohlcv_data.min())
    
    # Créer l'image
    n_bars = len(ohlcv_data)
    bar_width = image_size[1] // n_bars
    
    image = np.zeros(image_size)
    
    for i, (_, row) in enumerate(normalized.iterrows()):
        x = i * bar_width
        
        # Dessiner la bougie
        open_y = int((1 - row['open']) * image_size[0])
        close_y = int((1 - row['close']) * image_size[0])
        high_y = int((1 - row['high']) * image_size[0])
        low_y = int((1 - row['low']) * image_size[0])
        
        # Mèche
        image[high_y:low_y, x:x+bar_width//3] = 0.5
        
        # Corps
        body_top = min(open_y, close_y)
        body_bottom = max(open_y, close_y)
        color = 1.0 if close_y < open_y else 0.3  # Vert si hausse
        image[body_top:body_bottom, x:x+bar_width] = color
    
    return image


# === Exemple ===
def demonstrate_cnn_trading():
    """Démontre CNN pour le trading."""
    np.random.seed(42)
    
    # Créer des données OHLCV synthétiques
    n = 2000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    close = np.cumsum(np.random.randn(n) * 0.02) + 100
    data = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.5,
        'high': close + np.abs(np.random.randn(n)),
        'low': close - np.abs(np.random.randn(n)),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)
    
    # Normaliser
    data_norm = (data - data.mean()) / data.std()
    
    # Target: rendement positif le lendemain
    data_norm['target'] = (data['close'].pct_change().shift(-1) > 0).astype(int)
    data_norm = data_norm.dropna()
    
    # Préparer les séquences
    lookback = 20
    cnn = CNNTimeSeries(
        sequence_length=lookback,
        n_features=5,
        task='classification'
    )
    
    X, y = cnn.prepare_data(data_norm, 'target', lookback)
    
    # Split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Entraîner
    print("Entraînement du CNN...")
    cnn.model.summary()
    history = cnn.fit(X_train, y_train, epochs=30, batch_size=32)
    
    # Évaluer
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    y_pred = cnn.predict(X_test)
    y_proba = cnn.predict_proba(X_test)
    
    print(f"\nRésultats CNN:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    return cnn


# cnn = demonstrate_cnn_trading()
```

---

# 19. RÉSEAUX DE NEURONES RÉCURRENTS (RNN)
## Recurrent Neural Networks

## 19.1 LSTM pour Séries Temporelles

```python
"""
LSTM pour Prédiction Financière
===============================
Les LSTM (Long Short-Term Memory) sont conçus pour capturer
les dépendances à long terme dans les séries temporelles.

Structure d'une cellule LSTM:
- Forget gate: Quoi oublier de l'état précédent
- Input gate: Quoi ajouter à l'état
- Output gate: Quoi outputter

Avantages pour la finance:
- Capture les patterns temporels complexes
- Gère les dépendances à long terme
- Robuste au vanishing gradient
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class LSTMTradingModel:
    """
    LSTM pour prédiction de trading.
    """
    
    def __init__(self, sequence_length, n_features, task='regression',
                 lstm_units=[64, 32], dense_units=[16], dropout_rate=0.2,
                 recurrent_dropout=0.2):
        """
        Initialise le LSTM.
        
        Args:
            sequence_length: Longueur de la séquence d'entrée
            n_features: Nombre de features
            task: 'classification' ou 'regression'
            lstm_units: Unités par couche LSTM
            dense_units: Unités des couches denses
            dropout_rate: Dropout standard
            recurrent_dropout: Dropout récurrent
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.task = task
        
        self.model = self._build_model(
            lstm_units, dense_units, dropout_rate, recurrent_dropout
        )
    
    def _build_model(self, lstm_units, dense_units, dropout_rate, recurrent_dropout):
        """Construit le modèle LSTM."""
        model = tf.keras.Sequential()
        
        # Input
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))
        
        # Couches LSTM
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout
            ))
        
        # Couches denses
        for units in dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output
        if self.task == 'classification':
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(layers.Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def create_sequences(self, data, target=None):
        """
        Crée des séquences pour le LSTM.
        
        Args:
            data: array de features (n_samples, n_features)
            target: array de target (optionnel)
        
        Returns:
            tuple: (X, y) ou X seul
        """
        X = []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
        
        X = np.array(X)
        
        if target is not None:
            y = target[self.sequence_length:]
            return X, y
        
        return X
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32):
        """Entraîne le modèle."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Prédit."""
        preds = self.model.predict(X)
        if self.task == 'classification':
            return (preds > 0.5).astype(int).flatten()
        return preds.flatten()
    
    def predict_proba(self, X):
        """Prédit les probabilités."""
        return self.model.predict(X).flatten()


class BidirectionalLSTM:
    """
    LSTM bidirectionnel pour une meilleure compréhension du contexte.
    """
    
    def __init__(self, sequence_length, n_features, task='classification',
                 lstm_units=64, dense_units=[32]):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.task = task
        
        self.model = self._build_model(lstm_units, dense_units)
    
    def _build_model(self, lstm_units, dense_units):
        """Construit le modèle BiLSTM."""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # BiLSTM
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(inputs)
        x = layers.Bidirectional(layers.LSTM(lstm_units // 2))(x)
        
        # Dense
        for units in dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
        
        # Output
        if self.task == 'classification':
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(1)(x)
            loss = 'mse'
        
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy' if self.task == 'classification' else 'mae'])
        
        return model


# === Exemple ===
def demonstrate_lstm_trading():
    """Démontre LSTM pour le trading."""
    np.random.seed(42)
    
    # Créer des données
    n = 2000
    
    # Features: returns, volatility, momentum
    returns = np.random.randn(n) * 0.02
    volatility = np.abs(np.random.randn(n)) * 0.1 + 0.1
    momentum = np.convolve(returns, np.ones(5)/5, mode='same')
    
    data = np.column_stack([returns, volatility, momentum])
    
    # Target: direction du prochain return
    target = (np.roll(returns, -1) > 0).astype(int)
    
    # Normaliser
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Créer le modèle
    sequence_length = 20
    lstm = LSTMTradingModel(
        sequence_length=sequence_length,
        n_features=3,
        task='classification',
        lstm_units=[64, 32]
    )
    
    # Créer les séquences
    X, y = lstm.create_sequences(data_scaled, target)
    
    # Split temporel
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Entraîner
    print("Entraînement du LSTM...")
    lstm.model.summary()
    history = lstm.fit(X_train, y_train, epochs=30, batch_size=32)
    
    # Évaluer
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    y_pred = lstm.predict(X_test)
    y_proba = lstm.predict_proba(X_test)
    
    print(f"\nRésultats LSTM:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    return lstm


# lstm = demonstrate_lstm_trading()
```

---

# 20. AUTOENCODEURS
## Autoencoders

## 20.1 Autoencodeur pour Feature Learning

```python
"""
Autoencodeurs pour la Finance
=============================
Les autoencodeurs apprennent des représentations compressées (latentes)
des données en les encodant puis décodant.

Applications:
- Réduction de dimensionnalité non-linéaire
- Détection d'anomalies
- Débruitage (denoising)
- Génération de features
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

class FinancialAutoencoder:
    """
    Autoencodeur pour données financières.
    """
    
    def __init__(self, input_dim, encoding_dim=32, hidden_layers=[64]):
        """
        Initialise l'autoencodeur.
        
        Args:
            input_dim: Dimension d'entrée
            encoding_dim: Dimension de l'espace latent
            hidden_layers: Couches cachées entre input et encoding
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        self.autoencoder, self.encoder, self.decoder = self._build_model(hidden_layers)
    
    def _build_model(self, hidden_layers):
        """Construit l'autoencodeur."""
        # Encoder
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        
        for units in hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder
        x = encoded
        for units in reversed(hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        decoded = layers.Dense(self.input_dim, activation='linear')(x)
        
        # Modèles
        autoencoder = Model(inputs, decoded, name='autoencoder')
        encoder = Model(inputs, encoded, name='encoder')
        
        # Decoder séparé
        decoder_input = layers.Input(shape=(self.encoding_dim,))
        decoder_layers = autoencoder.layers[len(hidden_layers)+3:]  # Skip encoder layers
        x = decoder_input
        for layer in decoder_layers:
            x = layer(x)
        decoder = Model(decoder_input, x, name='decoder')
        
        # Compiler
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder, encoder, decoder
    
    def fit(self, X, validation_split=0.1, epochs=100, batch_size=32):
        """Entraîne l'autoencodeur."""
        from tensorflow.keras.callbacks import EarlyStopping
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True)
        ]
        
        history = self.autoencoder.fit(
            X, X,  # Input = Output
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def encode(self, X):
        """Encode les données dans l'espace latent."""
        return self.encoder.predict(X)
    
    def decode(self, Z):
        """Décode depuis l'espace latent."""
        return self.decoder.predict(Z)
    
    def reconstruct(self, X):
        """Reconstruit les données."""
        return self.autoencoder.predict(X)
    
    def reconstruction_error(self, X):
        """
        Calcule l'erreur de reconstruction.
        
        Utile pour la détection d'anomalies.
        """
        reconstructed = self.reconstruct(X)
        return np.mean((X - reconstructed) ** 2, axis=1)


class VariationalAutoencoder:
    """
    Variational Autoencoder (VAE) pour génération de données.
    
    Le VAE apprend une distribution dans l'espace latent,
    permettant de générer de nouvelles données.
    """
    
    def __init__(self, input_dim, latent_dim=16, hidden_layers=[64, 32]):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder, self.decoder, self.vae = self._build_model(hidden_layers)
    
    def _sampling(self, args):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _build_model(self, hidden_layers):
        """Construit le VAE."""
        # Encoder
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        for units in hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = layers.Lambda(self._sampling, name='z')([z_mean, z_log_var])
        
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = latent_inputs
        for units in reversed(hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
        outputs = layers.Dense(self.input_dim)(x)
        
        decoder = Model(latent_inputs, outputs, name='decoder')
        
        # VAE
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')
        
        # Loss: reconstruction + KL divergence
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.mse(inputs, outputs)
        ) * self.input_dim
        
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        
        vae.add_loss(reconstruction_loss + kl_loss)
        vae.compile(optimizer='adam')
        
        return encoder, decoder, vae
    
    def fit(self, X, epochs=100, batch_size=32):
        """Entraîne le VAE."""
        return self.vae.fit(X, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def encode(self, X):
        """Encode dans l'espace latent."""
        z_mean, z_log_var, z = self.encoder.predict(X)
        return z
    
    def generate(self, n_samples):
        """Génère de nouvelles données."""
        z = np.random.normal(size=(n_samples, self.latent_dim))
        return self.decoder.predict(z)


# === Détection d'anomalies ===
def detect_anomalies_with_autoencoder(autoencoder, X, threshold_percentile=95):
    """
    Détecte les anomalies avec l'erreur de reconstruction.
    
    Args:
        autoencoder: Autoencodeur entraîné
        X: Données à analyser
        threshold_percentile: Percentile pour le seuil
    
    Returns:
        tuple: (anomalies, scores)
    """
    # Calculer les erreurs
    errors = autoencoder.reconstruction_error(X)
    
    # Définir le seuil
    threshold = np.percentile(errors, threshold_percentile)
    
    # Identifier les anomalies
    anomalies = errors > threshold
    
    return anomalies, errors, threshold


# === Exemple ===
def demonstrate_autoencoder():
    """Démontre les autoencodeurs."""
    np.random.seed(42)
    
    # Créer des données
    n = 2000
    n_features = 20
    
    # Données normales
    X_normal = np.random.randn(n, n_features)
    
    # Ajouter quelques anomalies
    n_anomalies = 50
    X_anomalies = np.random.randn(n_anomalies, n_features) * 3 + 5
    X = np.vstack([X_normal, X_anomalies])
    labels = np.array([0] * n + [1] * n_anomalies)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, labels = X[idx], labels[idx]
    
    # Split (entraîner sur données normales uniquement)
    X_train = X_normal[:int(0.8*n)]
    X_test = X
    
    # Entraîner l'autoencodeur
    print("Entraînement de l'autoencodeur...")
    ae = FinancialAutoencoder(
        input_dim=n_features,
        encoding_dim=8,
        hidden_layers=[32, 16]
    )
    ae.fit(X_train, epochs=50)
    
    # Détecter les anomalies
    anomalies_detected, scores, threshold = detect_anomalies_with_autoencoder(ae, X_test)
    
    # Évaluer
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    print(f"\nDétection d'anomalies:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Precision: {precision_score(labels, anomalies_detected):.2%}")
    print(f"  Recall: {recall_score(labels, anomalies_detected):.2%}")
    print(f"  F1: {f1_score(labels, anomalies_detected):.2%}")
    
    return ae


# ae = demonstrate_autoencoder()
```

---

# 21. RÉSEAUX ADVERSES GÉNÉRATIFS (GAN)
## Generative Adversarial Networks

## 21.1 TimeGAN pour Données Synthétiques

```python
"""
TimeGAN pour Génération de Séries Temporelles Financières
=========================================================
TimeGAN génère des séries temporelles synthétiques réalistes
qui préservent les propriétés temporelles des données originales.

Applications:
- Augmentation de données
- Test de stratégies sur scénarios synthétiques
- Préservation de la confidentialité
- Simulation de stress tests
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

class SimpleTimeGAN:
    """
    Version simplifiée de TimeGAN.
    
    TimeGAN complet a 4 composants:
    1. Embedding: Encode les séquences
    2. Recovery: Décode les séquences
    3. Generator: Génère des séquences latentes
    4. Discriminator: Distingue vrai de faux
    """
    
    def __init__(self, seq_len, n_features, hidden_dim=24, latent_dim=24):
        """
        Initialise TimeGAN.
        
        Args:
            seq_len: Longueur des séquences
            n_features: Nombre de features
            hidden_dim: Dimension cachée
            latent_dim: Dimension latente
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.embedder, self.recovery, self.generator, self.discriminator = self._build_models()
    
    def _build_models(self):
        """Construit les 4 réseaux."""
        # Embedder (Real → Hidden)
        emb_input = layers.Input(shape=(self.seq_len, self.n_features))
        e = layers.GRU(self.hidden_dim, return_sequences=True)(emb_input)
        e = layers.GRU(self.hidden_dim, return_sequences=True)(e)
        embedder = Model(emb_input, e, name='embedder')
        
        # Recovery (Hidden → Real)
        rec_input = layers.Input(shape=(self.seq_len, self.hidden_dim))
        r = layers.GRU(self.hidden_dim, return_sequences=True)(rec_input)
        r = layers.Dense(self.n_features)(r)
        recovery = Model(rec_input, r, name='recovery')
        
        # Generator (Noise → Hidden)
        gen_input = layers.Input(shape=(self.seq_len, self.latent_dim))
        g = layers.GRU(self.hidden_dim, return_sequences=True)(gen_input)
        g = layers.GRU(self.hidden_dim, return_sequences=True)(g)
        generator = Model(gen_input, g, name='generator')
        
        # Discriminator (Hidden → Real/Fake)
        disc_input = layers.Input(shape=(self.seq_len, self.hidden_dim))
        d = layers.GRU(self.hidden_dim, return_sequences=True)(disc_input)
        d = layers.GRU(self.hidden_dim)(d)
        d = layers.Dense(1, activation='sigmoid')(d)
        discriminator = Model(disc_input, d, name='discriminator')
        
        return embedder, recovery, generator, discriminator
    
    def train(self, real_data, epochs=1000, batch_size=32):
        """
        Entraîne TimeGAN.
        
        Args:
            real_data: Données réelles (n_samples, seq_len, n_features)
            epochs: Nombre d'époques
            batch_size: Taille des batches
        """
        optimizer_e = tf.keras.optimizers.Adam(0.001)
        optimizer_g = tf.keras.optimizers.Adam(0.001)
        optimizer_d = tf.keras.optimizers.Adam(0.001)
        
        n_samples = len(real_data)
        
        for epoch in range(epochs):
            # Mini-batch
            idx = np.random.randint(0, n_samples, batch_size)
            real_batch = real_data[idx]
            
            # 1. Entraîner Embedder + Recovery (reconstruction)
            with tf.GradientTape() as tape:
                h_real = self.embedder(real_batch, training=True)
                x_reconstructed = self.recovery(h_real, training=True)
                e_loss = tf.reduce_mean(tf.square(real_batch - x_reconstructed))
            
            e_vars = self.embedder.trainable_variables + self.recovery.trainable_variables
            grads = tape.gradient(e_loss, e_vars)
            optimizer_e.apply_gradients(zip(grads, e_vars))
            
            # 2. Entraîner Generator
            noise = np.random.normal(size=(batch_size, self.seq_len, self.latent_dim))
            
            with tf.GradientTape() as tape:
                h_fake = self.generator(noise, training=True)
                y_fake = self.discriminator(h_fake, training=True)
                g_loss = -tf.reduce_mean(tf.math.log(y_fake + 1e-8))
            
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            optimizer_g.apply_gradients(zip(g_grads, self.generator.trainable_variables))
            
            # 3. Entraîner Discriminator
            with tf.GradientTape() as tape:
                h_real = self.embedder(real_batch, training=False)
                h_fake = self.generator(noise, training=False)
                
                y_real = self.discriminator(h_real, training=True)
                y_fake = self.discriminator(h_fake, training=True)
                
                d_loss = -tf.reduce_mean(tf.math.log(y_real + 1e-8) + 
                                        tf.math.log(1 - y_fake + 1e-8))
            
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            optimizer_d.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - E_loss: {e_loss:.4f}, G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")
    
    def generate(self, n_samples):
        """
        Génère des séquences synthétiques.
        
        Args:
            n_samples: Nombre de séquences à générer
        
        Returns:
            array: Séquences synthétiques
        """
        noise = np.random.normal(size=(n_samples, self.seq_len, self.latent_dim))
        h_fake = self.generator.predict(noise)
        x_fake = self.recovery.predict(h_fake)
        return x_fake


def evaluate_synthetic_data(real_data, synthetic_data):
    """
    Évalue la qualité des données synthétiques.
    
    Métriques:
    - Distribution: Les distributions sont-elles similaires?
    - Diversité: Les données sont-elles variées?
    - Utilité: Peuvent-elles entraîner des modèles?
    """
    from scipy import stats
    
    results = {}
    
    # 1. Comparaison des distributions (par feature)
    n_features = real_data.shape[2]
    ks_stats = []
    
    for f in range(n_features):
        real_flat = real_data[:, :, f].flatten()
        synth_flat = synthetic_data[:, :, f].flatten()
        ks_stat, _ = stats.ks_2samp(real_flat, synth_flat)
        ks_stats.append(ks_stat)
    
    results['ks_statistic_mean'] = np.mean(ks_stats)
    
    # 2. Corrélation temporelle
    real_autocorr = np.mean([
        np.corrcoef(real_data[i, :-1, 0], real_data[i, 1:, 0])[0, 1]
        for i in range(len(real_data))
    ])
    
    synth_autocorr = np.mean([
        np.corrcoef(synthetic_data[i, :-1, 0], synthetic_data[i, 1:, 0])[0, 1]
        for i in range(len(synthetic_data))
    ])
    
    results['autocorr_real'] = real_autocorr
    results['autocorr_synthetic'] = synth_autocorr
    results['autocorr_diff'] = abs(real_autocorr - synth_autocorr)
    
    return results


# === Exemple ===
def demonstrate_timegan():
    """Démontre TimeGAN."""
    np.random.seed(42)
    
    # Créer des données réalistes (marche aléatoire avec momentum)
    n_samples = 500
    seq_len = 24
    n_features = 3
    
    real_data = []
    for _ in range(n_samples):
        # Simuler une trajectoire de prix
        returns = np.random.randn(seq_len) * 0.02
        returns = np.convolve(returns, [0.3, 0.5, 0.2], mode='same')  # Momentum
        
        price = 100 * np.exp(np.cumsum(returns))
        volume = np.abs(np.random.randn(seq_len)) * 1000000
        volatility = np.abs(returns) * 10
        
        seq = np.column_stack([price, volume, volatility])
        real_data.append(seq)
    
    real_data = np.array(real_data)
    
    # Normaliser
    mean = real_data.mean(axis=(0, 1))
    std = real_data.std(axis=(0, 1))
    real_data_norm = (real_data - mean) / std
    
    # Entraîner TimeGAN
    print("Entraînement de TimeGAN...")
    tgan = SimpleTimeGAN(seq_len, n_features, hidden_dim=16, latent_dim=16)
    tgan.train(real_data_norm.astype(np.float32), epochs=500, batch_size=32)
    
    # Générer des données synthétiques
    synthetic_data_norm = tgan.generate(100)
    synthetic_data = synthetic_data_norm * std + mean
    
    # Évaluer
    eval_results = evaluate_synthetic_data(real_data_norm, synthetic_data_norm)
    
    print("\nÉvaluation des données synthétiques:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
    
    return tgan, real_data, synthetic_data


# tgan, real_data, synthetic_data = demonstrate_timegan()
```

---

# 22. APPRENTISSAGE PAR RENFORCEMENT
## Reinforcement Learning

## 22.1 Q-Learning pour Trading

```python
"""
Reinforcement Learning pour le Trading
======================================
L'agent RL apprend une politique de trading en interagissant
avec l'environnement (le marché).

Composants:
- État (State): Représentation du marché (features)
- Action: Buy, Hold, Sell
- Récompense (Reward): P&L, Sharpe, etc.
- Politique (Policy): État → Action

Algorithmes:
- Q-Learning: Table de Q-values
- Deep Q-Network (DQN): Neural network pour Q-values
- Policy Gradient: Optimise directement la politique
"""
import numpy as np
import pandas as pd
from collections import deque
import random

class TradingEnvironment:
    """
    Environnement de trading pour RL.
    
    Actions:
        0: HOLD
        1: BUY
        2: SELL
    """
    
    def __init__(self, prices, features, initial_balance=10000,
                 transaction_cost=0.001, max_position=1):
        """
        Initialise l'environnement.
        
        Args:
            prices: Series de prix
            features: DataFrame de features
            initial_balance: Capital initial
            transaction_cost: Coût de transaction
            max_position: Position max (-1 à 1)
        """
        self.prices = prices.values
        self.features = features.values
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        self.n_features = features.shape[1]
        self.n_steps = len(prices)
        
        self.reset()
    
    def reset(self):
        """Réinitialise l'environnement."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1, 0, ou 1
        self.portfolio_value = self.initial_balance
        self.done = False
        
        return self._get_state()
    
    def _get_state(self):
        """Retourne l'état actuel."""
        market_state = self.features[self.current_step]
        position_state = np.array([self.position, self.balance / self.initial_balance])
        return np.concatenate([market_state, position_state])
    
    def step(self, action):
        """
        Exécute une action.
        
        Args:
            action: 0 (HOLD), 1 (BUY), 2 (SELL)
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        current_price = self.prices[self.current_step]
        
        # Exécuter l'action
        old_position = self.position
        
        if action == 1 and self.position < self.max_position:  # BUY
            self.position = min(self.position + 1, self.max_position)
        elif action == 2 and self.position > -self.max_position:  # SELL
            self.position = max(self.position - 1, -self.max_position)
        
        # Coût de transaction si changement de position
        if self.position != old_position:
            transaction_cost = abs(self.position - old_position) * current_price * self.transaction_cost
            self.balance -= transaction_cost
        
        # Avancer d'un pas
        self.current_step += 1
        
        if self.current_step >= self.n_steps - 1:
            self.done = True
        
        # Calculer la récompense (P&L)
        if not self.done:
            next_price = self.prices[self.current_step]
            price_change = (next_price - current_price) / current_price
            reward = self.position * price_change  # Position * Return
        else:
            reward = 0
        
        # Mettre à jour la valeur du portefeuille
        if not self.done:
            self.portfolio_value = self.balance + self.position * self.prices[self.current_step]
        
        next_state = self._get_state() if not self.done else None
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance
        }
        
        return next_state, reward, self.done, info


class DQNAgent:
    """
    Agent Deep Q-Network pour le trading.
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialise l'agent DQN.
        
        Args:
            state_size: Taille de l'état
            action_size: Nombre d'actions
            learning_rate: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Exploration initiale
            epsilon_min: Exploration minimale
            epsilon_decay: Décroissance de l'exploration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.memory = deque(maxlen=10000)
        self.model = self._build_model(learning_rate)
    
    def _build_model(self, learning_rate):
        """Construit le réseau Q."""
        from tensorflow.keras import layers, Model
        
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='mse'
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choisit une action.
        
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy
        
        Returns:
            int: Action choisie
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """
        Entraîne sur un batch d'expériences.
        
        Args:
            batch_size: Taille du batch
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] if e[3] is not None else np.zeros(self.state_size) for e in minibatch])
        dones = np.array([e[4] for e in minibatch])
        
        # Q-values actuels
        q_values = self.model.predict(states, verbose=0)
        
        # Q-values cibles
        next_q_values = self.model.predict(next_states, verbose=0)
        
        # Mise à jour Q-learning
        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, q_values, epochs=1, verbose=0)
        
        # Décroissance de l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn_agent(env, agent, episodes=100, batch_size=32):
    """
    Entraîne l'agent DQN.
    
    Args:
        env: Environnement de trading
        agent: Agent DQN
        episodes: Nombre d'épisodes
        batch_size: Taille des batches
    
    Returns:
        list: Historique des récompenses
    """
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            
            state = next_state
            
            if done:
                break
            
            agent.replay(batch_size)
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history


# === Exemple ===
def demonstrate_rl_trading():
    """Démontre le RL pour le trading."""
    np.random.seed(42)
    
    # Créer des données
    n = 500
    
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Features
    features = pd.DataFrame({
        'return_1d': pd.Series(prices).pct_change().fillna(0),
        'return_5d': pd.Series(prices).pct_change(5).fillna(0),
        'volatility': pd.Series(prices).pct_change().rolling(10).std().fillna(0.01),
        'momentum': (pd.Series(prices).rolling(10).mean() / pd.Series(prices).rolling(30).mean()).fillna(1) - 1
    })
    
    # Créer l'environnement
    env = TradingEnvironment(
        prices=pd.Series(prices),
        features=features,
        initial_balance=10000
    )
    
    # Créer l'agent
    state_size = env.n_features + 2  # Features + position + balance
    action_size = 3  # HOLD, BUY, SELL
    
    agent = DQNAgent(state_size, action_size)
    
    # Entraîner
    print("Entraînement de l'agent DQN...")
    rewards = train_dqn_agent(env, agent, episodes=50)
    
    # Évaluer
    state = env.reset()
    agent.epsilon = 0  # Pas d'exploration
    
    positions = []
    portfolio_values = []
    
    while True:
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        positions.append(info['position'])
        portfolio_values.append(info['portfolio_value'])
        
        if done:
            break
        
        state = next_state
    
    # Résultats
    final_value = portfolio_values[-1]
    total_return = (final_value - 10000) / 10000
    
    print(f"\nRésultats:")
    print(f"  Valeur finale: ${final_value:,.2f}")
    print(f"  Rendement total: {total_return:.2%}")
    
    # Comparer à Buy & Hold
    buy_hold_return = (prices[-1] - prices[0]) / prices[0]
    print(f"  Buy & Hold: {buy_hold_return:.2%}")
    
    return agent, rewards


# agent, rewards = demonstrate_rl_trading()
```

---

# 23. PROCHAINES ÉTAPES ET RESSOURCES

## 23.1 Bonnes Pratiques

```python
"""
Bonnes Pratiques pour le ML en Finance
======================================
"""

# 1. ÉVITER LE LOOK-AHEAD BIAS
# Ne jamais utiliser des données futures pour prédire le passé
# - Toujours utiliser TimeSeriesSplit pour la cross-validation
# - Décaler les features d'au moins 1 période
# - Vérifier les timestamps des données

# 2. GÉRER L'OVERFITTING
# Les données financières sont bruitées et non-stationnaires
# - Utiliser la régularisation (L1, L2, Dropout)
# - Limiter la complexité du modèle
# - Valider sur des périodes out-of-sample
# - Utiliser le Deflated Sharpe Ratio pour le multiple testing

# 3. COÛTS DE TRANSACTION
# Toujours inclure les coûts réalistes
# - Commission du broker
# - Slippage (différence entre prix prévu et exécuté)
# - Market impact pour grandes positions

# 4. DONNÉES DE QUALITÉ
# La qualité des données est plus importante que le modèle
# - Vérifier les ajustements (dividendes, splits)
# - Gérer les survivorship bias
# - Attention aux données de point-in-time

# 5. INTERPRÉTABILITÉ
# Comprendre pourquoi le modèle fonctionne
# - Utiliser SHAP pour l'interprétation
# - Vérifier que les features ont du sens économique
# - Tester sur différentes périodes et marchés
```

## 23.2 Ressources Additionnelles

```markdown
## Livres Recommandés
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Asset Managers" - Marcos López de Prado
- "Machine Learning for Algorithmic Trading" - Stefan Jansen
- "Deep Learning" - Ian Goodfellow

## Cours en Ligne
- Coursera: Machine Learning (Andrew Ng)
- Fast.ai: Practical Deep Learning
- Udacity: AI for Trading

## Bibliothèques Python
- scikit-learn: ML classique
- TensorFlow/Keras: Deep Learning
- PyTorch: Deep Learning
- LightGBM, CatBoost, XGBoost: Gradient Boosting
- Zipline, Backtrader: Backtesting
- Alphalens, PyFolio: Évaluation de stratégies

## Datasets
- Yahoo Finance (yfinance)
- Quandl
- WRDS (académique)
- SEC EDGAR (filings)
- Alpha Vantage
```

---

# FIN DU DOCUMENT

Ce document complet couvre l'ensemble du livre "Machine Learning for Algorithmic Trading" 
de Stefan Jansen. Il fournit les implémentations Python détaillées pour chaque technique
abordée, avec des explications en français et anglais.

**Total: ~8000 lignes de code et documentation**
