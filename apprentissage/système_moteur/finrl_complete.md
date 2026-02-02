# 📊 FinRL - Deep Reinforcement Learning pour la Finance
## Guide Complet pour HelixOne Visual Code

---

# Table des Matières

1. [Introduction au DRL (Deep Reinforcement Learning)](#1-introduction-au-drl)
2. [Installation et Configuration](#2-installation-et-configuration)
3. [Architecture FinRL](#3-architecture-finrl)
4. [Configuration et Paramètres](#4-configuration-et-paramètres)
5. [Data Processing - Traitement des Données](#5-data-processing)
6. [Environnements de Trading](#6-environnements-de-trading)
7. [Agents DRL - Algorithmes](#7-agents-drl)
8. [Pipeline Train-Test-Trade](#8-pipeline-train-test-trade)
9. [Backtesting et Visualisation](#9-backtesting-et-visualisation)
10. [Paper Trading - Trading Simulé](#10-paper-trading)
11. [Exemples Complets](#11-exemples-complets)
12. [Optimisation des Hyperparamètres](#12-optimisation-hyperparametres)
13. [Glossaire DRL](#13-glossaire-drl)

---

# 1. Introduction au DRL

## 1.1 Qu'est-ce que le Deep Reinforcement Learning ?

Le **DRL (Deep Reinforcement Learning)** combine:
- **RL (Reinforcement Learning)** = Apprentissage par Renforcement : un agent apprend en interagissant avec un environnement
- **Deep Learning** = Réseaux de neurones profonds pour approximer les fonctions de valeur ou les politiques

### Concepts Fondamentaux

```
┌─────────────────────────────────────────────────────────────┐
│                    CYCLE DRL                                │
│                                                             │
│    ┌─────────┐     Action (a)      ┌─────────────────┐     │
│    │  AGENT  │ ─────────────────►  │  ENVIRONNEMENT  │     │
│    │  (DRL)  │                     │    (Marché)     │     │
│    │         │ ◄─────────────────  │                 │     │
│    └─────────┘   État (s), Reward  └─────────────────┘     │
│                       (r)                                   │
└─────────────────────────────────────────────────────────────┘
```

| Terme | Signification | Exemple Trading |
|-------|---------------|-----------------|
| **Agent** | L'algorithme qui prend des décisions | Notre modèle DRL |
| **Environment** | Le monde avec lequel l'agent interagit | Le marché financier |
| **State (s)** | L'observation actuelle | Prix, indicateurs techniques, positions |
| **Action (a)** | Décision de l'agent | Acheter, Vendre, Conserver |
| **Reward (r)** | Récompense/pénalité | Profit ou perte réalisé |
| **Policy (π)** | Stratégie de l'agent | Fonction qui mappe état → action |

## 1.2 Pourquoi le DRL pour le Trading ?

### Avantages
1. **Apprentissage End-to-End** : Pas besoin de règles manuelles
2. **Adaptation** : S'adapte aux conditions de marché changeantes
3. **Gestion du risque** : Peut intégrer la turbulence/volatilité
4. **Multi-actifs** : Gère des portefeuilles complexes

### Défis
1. **Non-stationnarité** : Les marchés évoluent
2. **Données bruitées** : Signal/bruit faible
3. **Coûts de transaction** : Impact sur les stratégies
4. **Overfitting** : Risque de surajustement

## 1.3 Algorithmes DRL Supportés par FinRL

| Algorithme | Type | Description |
|------------|------|-------------|
| **A2C** | On-Policy | Advantage Actor-Critic (Synchrone) |
| **PPO** | On-Policy | Proximal Policy Optimization |
| **DDPG** | Off-Policy | Deep Deterministic Policy Gradient |
| **TD3** | Off-Policy | Twin Delayed DDPG |
| **SAC** | Off-Policy | Soft Actor-Critic |

### On-Policy vs Off-Policy

- **On-Policy** (A2C, PPO) : Apprend uniquement des actions de la politique actuelle
  - Plus stable mais moins efficient en données
- **Off-Policy** (DDPG, TD3, SAC) : Peut apprendre d'expériences passées (Replay Buffer)
  - Plus efficient en données mais moins stable

---

# 2. Installation et Configuration

## 2.1 Installation de Base

```python
# ============================================================
# INSTALLATION FINRL
# ============================================================

# Installation via pip
# Note: Nécessite Python 3.8+
!pip install finrl

# OU installation depuis GitHub (dernière version)
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

# Dépendances supplémentaires
!pip install swig              # Pour box2d (certains environnements)
!pip install wrds              # Wharton Research Data Services
!pip install pyportfolioopt    # Optimisation de portefeuille classique
```

## 2.2 Installation Complète (avec tous les frameworks DRL)

```python
# ============================================================
# INSTALLATION COMPLÈTE AVEC TOUS LES BACKENDS
# ============================================================

# Système Linux (apt-get)
!apt-get update -y -qq
!apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig

# Stable Baselines 3 (recommandé pour débuter)
!pip install stable-baselines3[extra]

# ElegantRL (performances optimisées)
!pip install elegantrl

# Ray RLlib (computing distribué)
!pip install "ray[rllib]"

# FinRL
!pip install finrl

# Vérification de l'installation
import finrl
print(f"FinRL version: {finrl.__version__}")
```

## 2.3 Structure des Imports

```python
# ============================================================
# IMPORTS STANDARD FINRL
# ============================================================

# Bibliothèques Python standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import itertools
from pprint import pprint

# Configuration FinRL
from finrl import config
from finrl import config_tickers
from finrl.config import (
    DATA_SAVE_DIR,        # Dossier de sauvegarde des données
    TRAINED_MODEL_DIR,    # Dossier des modèles entraînés
    TENSORBOARD_LOG_DIR,  # Dossier des logs TensorBoard
    RESULTS_DIR,          # Dossier des résultats
    INDICATORS,           # Liste des indicateurs techniques
    TRAIN_START_DATE,     # Date début entraînement
    TRAIN_END_DATE,       # Date fin entraînement
    TEST_START_DATE,      # Date début test
    TEST_END_DATE,        # Date fin test
    TRADE_START_DATE,     # Date début trading
    TRADE_END_DATE,       # Date fin trading
)

# Data Processing
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.data_processor import DataProcessor

# Environnements
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnvNP
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv

# Agents DRL
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_ERL
from finrl.agents.rllib.models import DRLAgent as DRLAgent_RLlib

# Visualisation et Backtesting
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# Utilitaires
from finrl.main import check_and_make_directories
```

---

# 3. Architecture FinRL

## 3.1 Vue d'Ensemble

FinRL suit une architecture en **3 couches** :

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATIONS LAYER                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐│
│  │Stock Trading │ │  Portfolio   │ │    Crypto    │ │     HFT     ││
│  │              │ │ Allocation   │ │   Trading    │ │             ││
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                          AGENTS LAYER                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│  │  ElegantRL   │ │    RLlib     │ │Stable Base-  │                 │
│  │              │ │              │ │  lines 3     │                 │
│  │  - PPO       │ │  - PPO       │ │  - A2C       │                 │
│  │  - A2C       │ │  - A2C       │ │  - PPO       │                 │
│  │  - SAC       │ │  - DDPG      │ │  - DDPG      │                 │
│  │  - DDPG      │ │  - TD3       │ │  - TD3       │                 │
│  │  - TD3       │ │  - SAC       │ │  - SAC       │                 │
│  └──────────────┘ └──────────────┘ └──────────────┘                 │
├─────────────────────────────────────────────────────────────────────┤
│                      ENVIRONMENT LAYER (Meta)                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Data Processors                           │   │
│  │  Yahoo Finance | Alpaca | WRDS | CCXT | Binance | ...        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Trading Environments                        │   │
│  │  StockTradingEnv | PortfolioEnv | CryptoEnv | ...            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.2 Structure des Fichiers

```
FinRL/
├── finrl/                          # Package principal
│   ├── agents/                     # Implémentations des agents DRL
│   │   ├── elegantrl/              # Wrapper ElegantRL
│   │   │   └── models.py           # DRLAgent pour ElegantRL
│   │   ├── rllib/                  # Wrapper Ray RLlib
│   │   │   └── models.py           # DRLAgent pour RLlib
│   │   └── stablebaselines3/       # Wrapper Stable Baselines 3
│   │       ├── models.py           # DRLAgent pour SB3
│   │       └── hyperparams_opt.py  # Optimisation hyperparamètres
│   │
│   ├── meta/                       # Couche Meta (données + environnements)
│   │   ├── data_processor.py       # Classe unifiée DataProcessor
│   │   ├── data_processors/        # Processeurs spécifiques
│   │   │   ├── processor_yahoofinance.py
│   │   │   ├── processor_alpaca.py
│   │   │   ├── processor_wrds.py
│   │   │   └── processor_ccxt.py
│   │   │
│   │   ├── env_stock_trading/      # Environnements de trading
│   │   │   ├── env_stocktrading.py        # Env principal
│   │   │   ├── env_stocktrading_np.py     # Version NumPy optimisée
│   │   │   └── env_stock_papertrading.py  # Paper trading
│   │   │
│   │   ├── env_portfolio_allocation/  # Allocation de portefeuille
│   │   │   └── env_portfolio.py
│   │   │
│   │   └── preprocessor/           # Prétraitement des données
│   │       ├── preprocessors.py    # FeatureEngineer
│   │       └── yahoodownloader.py  # Téléchargement Yahoo
│   │
│   ├── config.py                   # Configuration globale
│   ├── config_tickers.py           # Listes de tickers
│   ├── train.py                    # Script d'entraînement
│   ├── test.py                     # Script de test
│   ├── trade.py                    # Script de trading
│   └── plot.py                     # Fonctions de visualisation
│
└── examples/                       # Notebooks d'exemples
    ├── Stock_NeurIPS2018_SB3.ipynb
    └── FinRL_Ensemble_StockTrading.ipynb
```

---

# 4. Configuration et Paramètres

## 4.1 Fichier de Configuration Principal (config.py)

```python
# ============================================================
# CONFIGURATION FINRL - config.py
# ============================================================

from __future__ import annotations

# ============================================================
# RÉPERTOIRES
# ============================================================
DATA_SAVE_DIR = "datasets"           # Sauvegarde des données téléchargées
TRAINED_MODEL_DIR = "trained_models" # Modèles entraînés
TENSORBOARD_LOG_DIR = "tensorboard_log"  # Logs pour TensorBoard
RESULTS_DIR = "results"              # Résultats de backtesting

# ============================================================
# DATES - Définition des périodes
# ============================================================
# Format: 'YYYY-MM-DD' (année-mois-jour)

# Période d'entraînement
TRAIN_START_DATE = "2014-01-06"  # Début (lundi pour éviter les problèmes de weekend)
TRAIN_END_DATE = "2020-07-31"    # Fin entraînement

# Période de test (validation)
TEST_START_DATE = "2020-08-01"   # Début test
TEST_END_DATE = "2021-10-01"     # Fin test

# Période de trading (paper trading ou live)
TRADE_START_DATE = "2021-11-01"  # Début trading
TRADE_END_DATE = "2021-12-01"    # Fin trading

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================
# Liste des indicateurs calculés par stockstats
# Documentation: https://pypi.org/project/stockstats/

INDICATORS = [
    "macd",         # MACD (Moving Average Convergence Divergence)
                    # Différence entre EMA(12) et EMA(26)
    
    "boll_ub",      # Bande de Bollinger Supérieure (Upper Band)
                    # SMA(20) + 2 * std(20)
    
    "boll_lb",      # Bande de Bollinger Inférieure (Lower Band)
                    # SMA(20) - 2 * std(20)
    
    "rsi_30",       # RSI (Relative Strength Index) sur 30 périodes
                    # Mesure la force relative des mouvements haussiers
    
    "cci_30",       # CCI (Commodity Channel Index) sur 30 périodes
                    # Identifie les conditions de surachat/survente
    
    "dx_30",        # DX (Directional Movement Index) sur 30 périodes
                    # Force de la tendance
    
    "close_30_sma", # SMA (Simple Moving Average) sur 30 périodes
                    # Moyenne mobile simple
    
    "close_60_sma", # SMA sur 60 périodes
                    # Tendance à plus long terme
]

# ============================================================
# PARAMÈTRES DES MODÈLES DRL
# ============================================================

# A2C (Advantage Actor-Critic)
# Algorithme on-policy synchrone
A2C_PARAMS = {
    "n_steps": 5,           # Nombre de pas avant mise à jour
    "ent_coef": 0.01,       # Coefficient d'entropie (exploration)
    "learning_rate": 0.0007 # Taux d'apprentissage
}

# PPO (Proximal Policy Optimization)
# Algorithme on-policy avec contrainte de divergence
PPO_PARAMS = {
    "n_steps": 2048,        # Pas par rollout (collecte d'expérience)
    "ent_coef": 0.01,       # Entropie pour exploration
    "learning_rate": 0.00025,  # Taux d'apprentissage
    "batch_size": 64        # Taille des mini-batches
}

# DDPG (Deep Deterministic Policy Gradient)
# Algorithme off-policy pour actions continues
DDPG_PARAMS = {
    "batch_size": 128,      # Taille batch d'entraînement
    "buffer_size": 50000,   # Taille du replay buffer
    "learning_rate": 0.001  # Taux d'apprentissage
}

# TD3 (Twin Delayed DDPG)
# Amélioration de DDPG avec réseaux jumeaux
TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000, # Buffer plus grand
    "learning_rate": 0.001
}

# SAC (Soft Actor-Critic)
# Algorithme off-policy avec maximisation d'entropie
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,  # Pas avant début entraînement
    "ent_coef": "auto_0.1"   # Entropie automatique avec target 0.1
}

# ElegantRL (paramètres génériques)
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,         # Facteur de discount
    "seed": 312,            # Graine aléatoire
    "net_dimension": 512,   # Dimension des réseaux de neurones
    "target_step": 5000,    # Pas par épisode
    "eval_gap": 30,         # Évaluation tous les N épisodes
    "eval_times": 64        # Nombre d'évaluations
}

# RLlib (Ray)
RLlib_PARAMS = {
    "lr": 5e-5,             # Learning rate
    "train_batch_size": 500,
    "gamma": 0.99           # Discount factor
}

# ============================================================
# FUSEAUX HORAIRES
# ============================================================
TIME_ZONE_SHANGHAI = "Asia/Shanghai"   # HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"     # Dow, Nasdaq, S&P
TIME_ZONE_PARIS = "Europe/Paris"       # CAC
TIME_ZONE_BERLIN = "Europe/Berlin"     # DAX

# ============================================================
# API KEYS (à configurer dans config_private.py)
# ============================================================
ALPACA_API_KEY = "xxx"
ALPACA_API_SECRET = "xxx"
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"
BINANCE_BASE_URL = "https://data.binance.vision/"
```

## 4.2 Listes de Tickers (config_tickers.py)

```python
# ============================================================
# LISTES DE TICKERS PRÉ-DÉFINIES
# ============================================================

# Ticker unique pour tests rapides
SINGLE_TICKER = ["AAPL"]

# Dow Jones 30 (USA)
DOW_30_TICKER = [
    "AXP",   # American Express
    "AMGN",  # Amgen
    "AAPL",  # Apple
    "BA",    # Boeing
    "CAT",   # Caterpillar
    "CSCO",  # Cisco
    "CVX",   # Chevron
    "GS",    # Goldman Sachs
    "HD",    # Home Depot
    "HON",   # Honeywell
    "IBM",   # IBM
    "INTC",  # Intel
    "JNJ",   # Johnson & Johnson
    "KO",    # Coca-Cola
    "JPM",   # JPMorgan Chase
    "MCD",   # McDonald's
    "MMM",   # 3M
    "MRK",   # Merck
    "MSFT",  # Microsoft
    "NKE",   # Nike
    "PG",    # Procter & Gamble
    "TRV",   # Travelers
    "UNH",   # UnitedHealth
    "CRM",   # Salesforce
    "VZ",    # Verizon
    "V",     # Visa
    "WBA",   # Walgreens
    "WMT",   # Walmart
    "DIS",   # Disney
    "DOW",   # Dow Inc.
]

# NASDAQ 100 (partiellement)
NAS_100_TICKER = [
    "AMGN", "AAPL", "AMAT", "INTC", "PCAR", "PAYX", "MSFT", "ADBE",
    "CSCO", "XLNX", "QCOM", "COST", "SBUX", "FISV", "CTXS", "INTU",
    "AMZN", "EBAY", "BIIB", "CHKP", "GILD", "NLOK", "CMCSA", "FAST",
    "ADSK", "CTSH", "NVDA", "GOOGL", "ISRG", "VRTX", # ... etc
]

# S&P 500
SP_500_TICKER = [
    "A", "AAL", "AAP", "AAPL", "ABBV", "ABC", # ... (liste complète de ~500 tickers)
]

# CAC 40 (France)
CAC_40_TICKER = [
    "AC.PA",   # Accor
    "AI.PA",   # Air Liquide
    "AIR.PA",  # Airbus
    "BNP.PA",  # BNP Paribas
    "OR.PA",   # L'Oréal
    "MC.PA",   # LVMH
    "SAN.PA",  # Sanofi
    "FP.PA",   # Total
    # ... etc
]

# DAX 30 (Allemagne)
DAX_30_TICKER = [
    "ALV.DE",  # Allianz
    "BAS.DE",  # BASF
    "BAYN.DE", # Bayer
    "BMW.DE",  # BMW
    "SAP.DE",  # SAP
    "SIE.DE",  # Siemens
    "VOW3.DE", # Volkswagen
    # ... etc
]

# Cryptomonnaies (format CCXT)
CRYPTO_TICKER = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
]
```

## 4.3 Création des Répertoires

```python
# ============================================================
# INITIALISATION DES RÉPERTOIRES
# ============================================================

import os
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR
)

def check_and_make_directories(directories: list):
    """
    Vérifie et crée les répertoires nécessaires.
    
    Parameters:
    -----------
    directories : list
        Liste des chemins de répertoires à créer
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Répertoire créé: {directory}")
        else:
            print(f"Répertoire existant: {directory}")

# Utilisation
check_and_make_directories([
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR
])
```

---

# 5. Data Processing

## 5.1 Téléchargement des Données

### 5.1.1 Avec YahooDownloader (Simple)

```python
# ============================================================
# TÉLÉCHARGEMENT DE DONNÉES AVEC YAHOO FINANCE
# ============================================================

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.config_tickers import DOW_30_TICKER

# Définir les dates
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2023-03-01'

# Télécharger les données
# Retourne un DataFrame avec colonnes: date, open, high, low, close, volume, tic
df = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    ticker_list=DOW_30_TICKER  # Liste de 30 tickers
).fetch_data()

print(f"Shape des données: {df.shape}")
print(f"Colonnes: {df.columns.tolist()}")
print(f"Tickers uniques: {df.tic.unique()}")
print(f"Période: {df.date.min()} à {df.date.max()}")

# Aperçu des données
df.head()
```

### 5.1.2 Avec DataProcessor (Unifié)

```python
# ============================================================
# DATAPROCESSOR - INTERFACE UNIFIÉE
# ============================================================

from finrl.meta.data_processor import DataProcessor

# Créer le processeur pour Yahoo Finance
dp = DataProcessor(data_source="yahoofinance")

# Télécharger les données
# time_interval: "1D" (journalier), "1H" (horaire), "1Min" (minute)
df = dp.download_data(
    ticker_list=DOW_30_TICKER,
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval="1D"  # Données journalières
)

# Nettoyer les données (gestion des NaN, ajustement des prix)
df = dp.clean_data(df)

print(f"Données nettoyées: {df.shape}")
```

### 5.1.3 Avec Alpaca (Trading US)

```python
# ============================================================
# ALPACA - DONNÉES EN TEMPS RÉEL
# ============================================================

from finrl.meta.data_processor import DataProcessor

# Configuration Alpaca (nécessite un compte)
ALPACA_API_KEY = "votre_api_key"
ALPACA_API_SECRET = "votre_api_secret"
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"

# Créer le processeur Alpaca
dp = DataProcessor(
    data_source="alpaca",
    API_KEY=ALPACA_API_KEY,
    API_SECRET=ALPACA_API_SECRET,
    API_BASE_URL=ALPACA_API_BASE_URL
)

# Télécharger (même interface)
df = dp.download_data(
    ticker_list=["AAPL", "MSFT", "GOOGL"],
    start_date="2022-01-01",
    end_date="2023-01-01",
    time_interval="1D"
)
```

## 5.2 Feature Engineering (Ingénierie des Caractéristiques)

```python
# ============================================================
# FEATURE ENGINEERING - AJOUT D'INDICATEURS TECHNIQUES
# ============================================================

from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import INDICATORS

# Afficher les indicateurs utilisés
print("Indicateurs techniques:")
for ind in INDICATORS:
    print(f"  - {ind}")

# Créer le FeatureEngineer
fe = FeatureEngineer(
    use_technical_indicator=True,       # Ajouter indicateurs techniques
    tech_indicator_list=INDICATORS,     # Liste des indicateurs
    use_vix=True,                        # Ajouter VIX (indice de volatilité)
    use_turbulence=True,                 # Ajouter indice de turbulence
    user_defined_feature=False           # Pas de features personnalisées
)

# Prétraiter les données
processed = fe.preprocess_data(df)

print(f"\nNouvelles colonnes ajoutées:")
new_cols = [c for c in processed.columns if c not in df.columns]
for col in new_cols:
    print(f"  - {col}")

# Vérifier les données
print(f"\nShape après traitement: {processed.shape}")
processed.head()
```

### Explication des Indicateurs Techniques

```python
# ============================================================
# EXPLICATION DES INDICATEURS
# ============================================================

"""
INDICATEURS TECHNIQUES DANS FINRL
=================================

1. MACD (Moving Average Convergence Divergence)
   - Formule: EMA(12) - EMA(26)
   - Signal: Croisement avec la ligne de signal (EMA(9) du MACD)
   - Interprétation: MACD > 0 = tendance haussière

2. Bandes de Bollinger (boll_ub, boll_lb)
   - Upper Band (boll_ub): SMA(20) + 2 * StdDev(20)
   - Lower Band (boll_lb): SMA(20) - 2 * StdDev(20)
   - Interprétation: Prix proche de boll_ub = surachat potentiel

3. RSI (Relative Strength Index)
   - Formule: 100 - (100 / (1 + RS))
   - RS = Moyenne des hausses / Moyenne des baisses
   - Interprétation: RSI > 70 = surachat, RSI < 30 = survente

4. CCI (Commodity Channel Index)
   - Formule: (Typical Price - SMA) / (0.015 * Mean Deviation)
   - Typical Price = (High + Low + Close) / 3
   - Interprétation: CCI > 100 = surachat, CCI < -100 = survente

5. DX (Directional Movement Index)
   - Mesure la force de la tendance
   - DX élevé = tendance forte

6. SMA (Simple Moving Average)
   - close_30_sma: Moyenne des 30 derniers prix de clôture
   - close_60_sma: Moyenne des 60 derniers prix de clôture
   - Utilisation: Identifier la tendance

7. VIX (CBOE Volatility Index)
   - "Indice de la peur" - mesure la volatilité attendue du S&P 500
   - VIX élevé = haute incertitude/volatilité

8. Turbulence
   - Mesure la déviation des rendements par rapport à leur distribution historique
   - Turbulence élevée = conditions de marché anormales
"""
```

## 5.3 Gestion des Données Manquantes

```python
# ============================================================
# GESTION DES DONNÉES MANQUANTES
# ============================================================

import itertools
import pandas as pd

def fill_missing_data(processed_df):
    """
    Remplit les données manquantes pour garantir la cohérence
    temporelle entre tous les tickers.
    
    Parameters:
    -----------
    processed_df : pd.DataFrame
        DataFrame avec colonnes ['date', 'tic', ...features...]
    
    Returns:
    --------
    pd.DataFrame
        DataFrame complet sans données manquantes
    """
    # Obtenir tous les tickers uniques
    list_ticker = processed_df["tic"].unique().tolist()
    
    # Obtenir toutes les dates de trading (du min au max)
    list_date = list(pd.date_range(
        processed_df['date'].min(),
        processed_df['date'].max()
    ).astype(str))
    
    # Créer toutes les combinaisons possibles (date, ticker)
    combination = list(itertools.product(list_date, list_ticker))
    
    # Créer un DataFrame avec toutes les combinaisons
    processed_full = pd.DataFrame(
        combination, 
        columns=["date", "tic"]
    ).merge(processed_df, on=["date", "tic"], how="left")
    
    # Garder seulement les dates qui sont dans les données originales
    # (exclure weekends et jours fériés)
    processed_full = processed_full[
        processed_full['date'].isin(processed_df['date'])
    ]
    
    # Trier par date et ticker
    processed_full = processed_full.sort_values(['date', 'tic'])
    
    # Remplir les valeurs manquantes par 0 (ou autre stratégie)
    processed_full = processed_full.fillna(0)
    
    return processed_full

# Utilisation
processed_full = fill_missing_data(processed)
print(f"Shape final: {processed_full.shape}")
```

## 5.4 Division Train/Test/Trade

```python
# ============================================================
# DIVISION DES DONNÉES
# ============================================================

from finrl.meta.preprocessor.preprocessors import data_split

# Définir les périodes
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2023-03-01'

# Diviser les données
train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

# Statistiques
print("="*50)
print("STATISTIQUES DES DONNÉES")
print("="*50)
print(f"\nPériode d'entraînement: {TRAIN_START_DATE} à {TRAIN_END_DATE}")
print(f"  - Lignes: {len(train)}")
print(f"  - Jours de trading: {len(train.date.unique())}")

print(f"\nPériode de trading: {TRADE_START_DATE} à {TRADE_END_DATE}")
print(f"  - Lignes: {len(trade)}")
print(f"  - Jours de trading: {len(trade.date.unique())}")

# Vérifier la continuité
print(f"\nDernière date train: {train.date.max()}")
print(f"Première date trade: {trade.date.min()}")
```

## 5.5 Conversion en Arrays NumPy

```python
# ============================================================
# CONVERSION POUR LES ENVIRONNEMENTS NUMPY
# ============================================================

from finrl.meta.data_processor import DataProcessor

def prepare_arrays(df, indicators, if_vix=True):
    """
    Convertit un DataFrame en arrays NumPy pour les environnements.
    
    Returns:
    --------
    price_array : np.ndarray
        Prix de clôture, shape (n_days, n_stocks)
    tech_array : np.ndarray
        Indicateurs techniques, shape (n_days, n_stocks * n_indicators)
    turbulence_array : np.ndarray
        Indice de turbulence/VIX, shape (n_days,)
    """
    dp = DataProcessor(data_source="yahoofinance")
    
    # Utiliser la méthode df_to_array
    price_array, tech_array, turbulence_array = dp.processor.df_to_array(
        df=df,
        tech_indicator_list=indicators,
        if_vix=if_vix
    )
    
    # Nettoyer les NaN et Inf
    tech_array = np.nan_to_num(tech_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Price array shape: {price_array.shape}")
    print(f"Tech array shape: {tech_array.shape}")
    print(f"Turbulence array shape: {turbulence_array.shape}")
    
    return price_array, tech_array, turbulence_array

# Utilisation
price_array, tech_array, turb_array = prepare_arrays(
    processed_full, 
    INDICATORS, 
    if_vix=True
)
```

---

# 6. Environnements de Trading

## 6.1 Vue d'Ensemble des Environnements

FinRL propose plusieurs environnements compatibles OpenAI Gym / Gymnasium :

| Environnement | Fichier | Usage |
|---------------|---------|-------|
| `StockTradingEnv` | env_stocktrading.py | Trading multi-actions, DataFrame |
| `StockTradingEnv (NP)` | env_stocktrading_np.py | Version NumPy optimisée |
| `StockPortfolioEnv` | env_portfolio.py | Allocation de portefeuille |
| `AlpacaPaperTrading` | env_stock_papertrading.py | Paper trading temps réel |

## 6.2 StockTradingEnv (Principal)

```python
# ============================================================
# ENVIRONNEMENT DE TRADING PRINCIPAL
# ============================================================

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS

# Calculer les dimensions
stock_dimension = len(train.tic.unique())

# State space = cash + prix * n_stocks + positions * n_stocks + indicateurs
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}")
print(f"State Space: {state_space}")

# Configuration de l'environnement
env_kwargs = {
    # Paramètres de trading
    "hmax": 100,                     # Nombre max d'actions par transaction
    "initial_amount": 1_000_000,     # Capital initial ($1M)
    
    # Positions initiales (0 = pas de positions)
    "num_stock_shares": [0] * stock_dimension,
    
    # Coûts de transaction (0.1% = 0.001)
    "buy_cost_pct": [0.001] * stock_dimension,   # 0.1% pour acheter
    "sell_cost_pct": [0.001] * stock_dimension,  # 0.1% pour vendre
    
    # Dimensions
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "action_space": stock_dimension,  # Une action par stock
    
    # Indicateurs techniques
    "tech_indicator_list": INDICATORS,
    
    # Scaling de la récompense (important pour la stabilité)
    "reward_scaling": 1e-4,
    
    # Gestion du risque (optionnel)
    "turbulence_threshold": None,  # Seuil de turbulence
    "risk_indicator_col": "vix",   # Colonne de risque
    
    # Verbosité
    "print_verbosity": 10,  # Afficher stats tous les 10 épisodes
}

# Créer l'environnement d'entraînement
e_train_gym = StockTradingEnv(df=train, **env_kwargs)

# Obtenir l'environnement vectorisé pour Stable Baselines
env_train, obs = e_train_gym.get_sb_env()

print(f"\nType d'environnement: {type(env_train)}")
print(f"Observation shape: {obs.shape}")
print(f"Action space: {e_train_gym.action_space}")
print(f"Observation space: {e_train_gym.observation_space}")
```

### Structure de l'État (State)

```python
# ============================================================
# STRUCTURE DE L'ÉTAT DANS STOCKTRADINGENV
# ============================================================

"""
L'état (state) est un vecteur 1D contenant:

state = [
    cash,                        # Index 0: Montant de cash disponible
    price_1, price_2, ...,       # Index 1 à stock_dim: Prix de clôture
    shares_1, shares_2, ...,     # Index stock_dim+1 à 2*stock_dim: Positions
    tech_1_1, tech_1_2, ...,     # Indicateurs pour stock 1
    tech_2_1, tech_2_2, ...,     # Indicateurs pour stock 2
    ...
]

Exemple avec 30 stocks et 8 indicateurs:
- state[0] = cash
- state[1:31] = 30 prix
- state[31:61] = 30 positions
- state[61:301] = 30 * 8 = 240 indicateurs techniques

Total: 1 + 30 + 30 + 240 = 301 dimensions
"""

# Vérification
state_example = e_train_gym.state
print(f"Longueur de l'état: {len(state_example)}")
print(f"Cash: ${state_example[0]:,.2f}")
print(f"Premier prix: ${state_example[1]:.2f}")
```

### Structure des Actions

```python
# ============================================================
# STRUCTURE DES ACTIONS
# ============================================================

"""
L'espace d'action est continu, shape = (stock_dim,)
Chaque action est dans [-1, 1]

Interprétation:
- action[i] > 0: Acheter min(hmax * action[i], cash_available) actions du stock i
- action[i] < 0: Vendre min(hmax * |action[i]|, current_holdings) actions du stock i
- action[i] ≈ 0: Conserver (hold)

Exemple avec hmax=100:
- action = [0.5, -0.3, 0.0, ...]
  → Stock 0: Acheter 50 actions (0.5 * 100)
  → Stock 1: Vendre 30 actions (0.3 * 100)
  → Stock 2: Hold
"""

# Test d'une action
import numpy as np
sample_action = np.random.uniform(-1, 1, size=(stock_dimension,))
print(f"Action sample shape: {sample_action.shape}")
print(f"Actions (premiers 5): {sample_action[:5]}")
```

## 6.3 Environnement NumPy Optimisé

```python
# ============================================================
# ENVIRONNEMENT NUMPY (HAUTE PERFORMANCE)
# ============================================================

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnvNP

# Préparer la configuration
env_config = {
    "price_array": price_array,       # Prix, shape (n_days, n_stocks)
    "tech_array": tech_array,         # Indicateurs, shape (n_days, n_features)
    "turbulence_array": turb_array,   # Turbulence, shape (n_days,)
    "if_train": True                   # Mode entraînement
}

# Créer l'environnement
env_np = StockTradingEnvNP(
    config=env_config,
    initial_account=1_000_000,     # Capital initial
    gamma=0.99,                     # Facteur de discount
    turbulence_thresh=99,           # Seuil de turbulence
    min_stock_rate=0.1,             # Taux minimum pour trader
    max_stock=100,                  # Nombre max d'actions par trade
    buy_cost_pct=0.001,             # Coût d'achat (0.1%)
    sell_cost_pct=0.001,            # Coût de vente (0.1%)
    reward_scaling=2**-11,          # Scaling de récompense
)

print(f"State dim: {env_np.state_dim}")
print(f"Action dim: {env_np.action_dim}")
print(f"Max step: {env_np.max_step}")
```

## 6.4 Environnement d'Allocation de Portefeuille

```python
# ============================================================
# ENVIRONNEMENT PORTFOLIO ALLOCATION
# ============================================================

from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv

"""
Différence avec StockTradingEnv:
- Actions = poids du portefeuille (somment à 1)
- Pas de trading discret, mais réallocation continue
- Utilise une matrice de covariance
"""

# Préparer les données avec covariance
def add_covariance(df, lookback=252):
    """Ajoute une matrice de covariance roulante."""
    df_pivot = df.pivot(index='date', columns='tic', values='close')
    
    # Calculer les rendements
    returns = df_pivot.pct_change()
    
    # Covariance roulante
    cov_list = []
    for i in range(len(returns)):
        if i < lookback:
            cov_list.append(np.eye(len(df_pivot.columns)))
        else:
            cov_list.append(returns.iloc[i-lookback:i].cov().values)
    
    # Ajouter au DataFrame
    df_cov = df_pivot.copy()
    df_cov['cov_list'] = cov_list
    
    return df_cov

# Configuration de l'environnement
portfolio_env_kwargs = {
    "hmax": 100,
    "initial_amount": 1_000_000,
    "transaction_cost_pct": 0.001,  # 0.1%
    "reward_scaling": 1,
    "state_space": stock_dimension,
    "stock_dim": stock_dimension,
    "action_space": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "lookback": 252,  # 1 an de lookback pour covariance
}

# L'environnement utilise softmax pour normaliser les poids
# Donc actions peuvent être n'importe quelles valeurs
```

## 6.5 Gestion de la Turbulence

```python
# ============================================================
# GESTION DE LA TURBULENCE (RISK MANAGEMENT)
# ============================================================

"""
La turbulence est calculée comme la distance de Mahalanobis 
des rendements actuels par rapport à leur distribution historique.

Formule:
turbulence_t = (r_t - μ)' * Σ^(-1) * (r_t - μ)

où:
- r_t = vecteur des rendements au temps t
- μ = moyenne historique des rendements
- Σ = matrice de covariance historique

Si turbulence > seuil:
→ L'agent liquide toutes ses positions (risk-off)
"""

# Calculer le seuil de turbulence sur les données d'entraînement
data_risk = processed_full[
    (processed_full.date < TRAIN_END_DATE) & 
    (processed_full.date >= TRAIN_START_DATE)
]
insample_risk = data_risk.drop_duplicates(subset=['date'])

# Statistiques de turbulence
print("Statistiques de turbulence (in-sample):")
print(insample_risk.turbulence.describe())

# Utiliser le percentile 99.6 comme seuil
turbulence_threshold = insample_risk.turbulence.quantile(0.996)
print(f"\nSeuil de turbulence (99.6%): {turbulence_threshold:.2f}")

# Statistiques VIX
print("\nStatistiques VIX (in-sample):")
print(insample_risk.vix.describe())
vix_threshold = insample_risk.vix.quantile(0.996)
print(f"Seuil VIX (99.6%): {vix_threshold:.2f}")

# Créer l'environnement de trading avec seuil
e_trade_gym = StockTradingEnv(
    df=trade,
    turbulence_threshold=turbulence_threshold,  # Activer le risk management
    risk_indicator_col='turbulence',  # Ou 'vix'
    **env_kwargs
)
```

---

# 7. Agents DRL

## 7.1 Stable Baselines 3 (Recommandé)

### 7.1.1 Configuration de Base

```python
# ============================================================
# AGENTS STABLE BASELINES 3
# ============================================================

from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

# Créer l'agent
agent = DRLAgent(env=env_train)

# Modèles disponibles
MODELS_AVAILABLE = ["a2c", "ppo", "ddpg", "td3", "sac"]
print(f"Modèles disponibles: {MODELS_AVAILABLE}")
```

### 7.1.2 A2C (Advantage Actor-Critic)

```python
# ============================================================
# A2C - ADVANTAGE ACTOR-CRITIC
# ============================================================

"""
A2C (Advantage Actor-Critic):
- Algorithme on-policy synchrone
- Utilise plusieurs workers en parallèle
- Actor: prédit l'action optimale
- Critic: évalue la valeur de l'état

Avantages:
- Simple et stable
- Bon pour commencer
- Fonctionne bien avec peu de données

Inconvénients:
- Moins efficient que PPO
- Pas de replay buffer
"""

# Paramètres A2C
A2C_PARAMS = {
    "n_steps": 5,           # Nombre de pas avant mise à jour
                            # Plus petit = mises à jour plus fréquentes
    
    "ent_coef": 0.01,       # Coefficient d'entropie
                            # Plus élevé = plus d'exploration
    
    "learning_rate": 0.0007 # Taux d'apprentissage
}

# Créer le modèle A2C
agent = DRLAgent(env=env_train)
model_a2c = agent.get_model(
    "a2c",
    model_kwargs=A2C_PARAMS
)

# Configurer le logger TensorBoard
tmp_path = RESULTS_DIR + '/a2c'
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model_a2c.set_logger(new_logger)

# Entraîner
trained_a2c = agent.train_model(
    model=model_a2c,
    tb_log_name='a2c',
    total_timesteps=100_000  # Nombre total de pas d'entraînement
)

# Sauvegarder
trained_a2c.save(f"{TRAINED_MODEL_DIR}/a2c_model")
```

### 7.1.3 PPO (Proximal Policy Optimization)

```python
# ============================================================
# PPO - PROXIMAL POLICY OPTIMIZATION
# ============================================================

"""
PPO (Proximal Policy Optimization):
- Algorithme on-policy
- Limite les mises à jour de politique pour la stabilité
- Utilise clipping sur le ratio de probabilité

Avantages:
- Très stable
- Bon compromis exploration/exploitation
- Standard de l'industrie

Paramètres importants:
- n_steps: taille du rollout buffer
- batch_size: taille des mini-batches
- ent_coef: exploration via entropie
"""

PPO_PARAMS = {
    "n_steps": 2048,         # Nombre de pas par rollout
                             # Plus grand = apprentissage plus stable
    
    "ent_coef": 0.01,        # Coefficient d'entropie
    
    "learning_rate": 0.00025, # Taux d'apprentissage
                             # Plus petit que A2C pour stabilité
    
    "batch_size": 128,       # Taille des mini-batches
                             # Doit diviser n_steps
    
    "n_epochs": 10,          # Nombre d'époques par mise à jour
    
    "gamma": 0.99,           # Facteur de discount
    
    "gae_lambda": 0.95,      # Lambda pour GAE (Generalized Advantage Estimation)
    
    "clip_range": 0.2,       # Range de clipping (ratio de politique)
}

# Créer et entraîner PPO
agent = DRLAgent(env=env_train)
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

# Logger
tmp_path = RESULTS_DIR + '/ppo'
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model_ppo.set_logger(new_logger)

# Entraînement
trained_ppo = agent.train_model(
    model=model_ppo,
    tb_log_name='ppo',
    total_timesteps=200_000
)

trained_ppo.save(f"{TRAINED_MODEL_DIR}/ppo_model")
```

### 7.1.4 DDPG (Deep Deterministic Policy Gradient)

```python
# ============================================================
# DDPG - DEEP DETERMINISTIC POLICY GRADIENT
# ============================================================

"""
DDPG (Deep Deterministic Policy Gradient):
- Algorithme off-policy pour actions continues
- Utilise un replay buffer
- Actor: politique déterministe
- Critic: Q-function

Avantages:
- Efficace en données (replay buffer)
- Bon pour actions continues

Inconvénients:
- Peut être instable
- Sensible aux hyperparamètres
"""

DDPG_PARAMS = {
    "batch_size": 128,        # Taille des batches du replay buffer
    
    "buffer_size": 50_000,    # Taille du replay buffer
                              # Plus grand = plus de mémoire
    
    "learning_rate": 0.001,   # Taux d'apprentissage
    
    "tau": 0.005,             # Coefficient de soft update des target networks
    
    "gamma": 0.99,            # Facteur de discount
    
    # Bruit d'exploration (Ornstein-Uhlenbeck)
    "action_noise": "ornstein_uhlenbeck"
}

# Créer et entraîner DDPG
agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)

# Logger
tmp_path = RESULTS_DIR + '/ddpg'
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model_ddpg.set_logger(new_logger)

# Entraînement
trained_ddpg = agent.train_model(
    model=model_ddpg,
    tb_log_name='ddpg',
    total_timesteps=100_000
)

trained_ddpg.save(f"{TRAINED_MODEL_DIR}/ddpg_model")
```

### 7.1.5 TD3 (Twin Delayed DDPG)

```python
# ============================================================
# TD3 - TWIN DELAYED DDPG
# ============================================================

"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient):
- Amélioration de DDPG
- Deux réseaux Critic (twin) pour réduire overestimation
- Mise à jour delayed de l'Actor
- Ajout de bruit sur les target actions

Avantages:
- Plus stable que DDPG
- Meilleure performance en général
"""

TD3_PARAMS = {
    "batch_size": 100,
    
    "buffer_size": 1_000_000,  # Buffer très large
    
    "learning_rate": 0.001,
    
    "tau": 0.005,
    
    "gamma": 0.99,
    
    "policy_delay": 2,         # Mise à jour de l'actor tous les 2 pas
    
    "target_policy_noise": 0.2,  # Bruit ajouté aux target actions
    
    "target_noise_clip": 0.5,    # Clipping du bruit
}

# Créer et entraîner TD3
agent = DRLAgent(env=env_train)
model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

tmp_path = RESULTS_DIR + '/td3'
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model_td3.set_logger(new_logger)

trained_td3 = agent.train_model(
    model=model_td3,
    tb_log_name='td3',
    total_timesteps=100_000
)

trained_td3.save(f"{TRAINED_MODEL_DIR}/td3_model")
```

### 7.1.6 SAC (Soft Actor-Critic)

```python
# ============================================================
# SAC - SOFT ACTOR-CRITIC
# ============================================================

"""
SAC (Soft Actor-Critic):
- Algorithme off-policy basé sur maximum entropy RL
- Maximise reward + entropie de la politique
- Exploration automatique via entropie

Avantages:
- Très stable
- Exploration robuste
- Souvent le meilleur pour trading

Particularité:
- ent_coef peut être appris automatiquement ("auto")
"""

SAC_PARAMS = {
    "batch_size": 128,
    
    "buffer_size": 100_000,
    
    "learning_rate": 0.0001,   # Learning rate plus petit
    
    "learning_starts": 100,    # Pas avant de commencer l'entraînement
                               # Permet de remplir le buffer
    
    "tau": 0.005,
    
    "gamma": 0.99,
    
    "ent_coef": "auto_0.1",    # Entropie automatique
                               # "auto" ou "auto_X" où X est la target
                               # Plus élevé = plus d'exploration
    
    "target_entropy": "auto",  # Target entropy automatique
}

# Créer et entraîner SAC
agent = DRLAgent(env=env_train)
model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

tmp_path = RESULTS_DIR + '/sac'
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model_sac.set_logger(new_logger)

trained_sac = agent.train_model(
    model=model_sac,
    tb_log_name='sac',
    total_timesteps=100_000
)

trained_sac.save(f"{TRAINED_MODEL_DIR}/sac_model")
```

## 7.2 ElegantRL (Haute Performance)

```python
# ============================================================
# ELEGANTRL - AGENT HAUTE PERFORMANCE
# ============================================================

from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_ERL
from finrl.config import ERL_PARAMS

"""
ElegantRL:
- Framework DRL optimisé pour la performance
- Implémentation PyTorch efficace
- Support GPU natif
- Entraînement parallèle
"""

# Configuration ElegantRL
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,           # Facteur de discount
    "seed": 312,
    "net_dimension": 512,     # Dimension des couches cachées
    "target_step": 5000,      # Pas par épisode d'entraînement
    "eval_gap": 30,           # Évaluation tous les 30 épisodes
    "eval_times": 64,         # Nombre d'évaluations
}

# Créer l'agent ElegantRL
agent_erl = DRLAgent_ERL(
    env=StockTradingEnvNP,
    price_array=price_array,
    tech_array=tech_array,
    turbulence_array=turb_array
)

# Obtenir le modèle (PPO dans cet exemple)
model_erl = agent_erl.get_model(
    model_name="ppo",
    model_kwargs=ERL_PARAMS
)

# Entraîner
agent_erl.train_model(
    model=model_erl,
    cwd="./trained_models/elegantrl_ppo",
    total_timesteps=100_000
)
```

## 7.3 Ray RLlib (Distribué)

```python
# ============================================================
# RLLIB - APPRENTISSAGE DISTRIBUÉ
# ============================================================

import ray
from finrl.agents.rllib.models import DRLAgent as DRLAgent_RLlib
from finrl.config import RLlib_PARAMS

"""
Ray RLlib:
- Framework pour RL distribué
- Scaling horizontal sur cluster
- Support multi-GPU
"""

# Initialiser Ray
ray.shutdown()  # Fermer session précédente si existante
ray.init(ignore_reinit_error=True)

# Paramètres RLlib
RLlib_PARAMS = {
    "lr": 5e-5,               # Learning rate
    "train_batch_size": 500,  # Taille batch d'entraînement
    "gamma": 0.99,            # Discount factor
}

# Créer l'agent RLlib
agent_rllib = DRLAgent_RLlib(
    env=StockTradingEnvNP,
    price_array=price_array,
    tech_array=tech_array,
    turbulence_array=turb_array
)

# Obtenir le modèle
model_rllib, model_config = agent_rllib.get_model("ppo")

# Configurer
model_config["lr"] = RLlib_PARAMS["lr"]
model_config["train_batch_size"] = RLlib_PARAMS["train_batch_size"]
model_config["gamma"] = RLlib_PARAMS["gamma"]

# Entraîner
trained_rllib = agent_rllib.train_model(
    model=model_rllib,
    model_name="ppo",
    model_config=model_config,
    total_episodes=100
)

# Sauvegarder
trained_rllib.save("./trained_models/rllib_ppo")

# Fermer Ray
ray.shutdown()
```

## 7.4 Comparaison des Frameworks

| Aspect | Stable Baselines 3 | ElegantRL | RLlib |
|--------|-------------------|-----------|-------|
| **Facilité** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Scaling** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Debugging** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

**Recommandation:**
- **Débutant** → Stable Baselines 3
- **Production** → ElegantRL
- **Cluster/Cloud** → RLlib

---

# 8. Pipeline Train-Test-Trade

## 8.1 Script d'Entraînement (train.py)

```python
# ============================================================
# PIPELINE D'ENTRAÎNEMENT COMPLET
# ============================================================

from finrl.train import train
from finrl.config import (
    TRAIN_START_DATE, TRAIN_END_DATE,
    INDICATORS, ERL_PARAMS
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

def run_training(
    model_name="ppo",
    drl_lib="stable_baselines3",
    total_timesteps=100_000
):
    """
    Exécute le pipeline d'entraînement complet.
    
    Parameters:
    -----------
    model_name : str
        Nom du modèle ("ppo", "a2c", "ddpg", "td3", "sac")
    drl_lib : str
        Librairie DRL ("stable_baselines3", "elegantrl", "rllib")
    total_timesteps : int
        Nombre de pas d'entraînement
    """
    
    # Appeler la fonction train
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib=drl_lib,
        env=StockTradingEnv,
        model_name=model_name,
        cwd=f"./trained_models/{model_name}",
        if_vix=True,
        
        # Paramètres spécifiques au framework
        erl_params=ERL_PARAMS if drl_lib == "elegantrl" else None,
        break_step=total_timesteps if drl_lib == "elegantrl" else None,
        total_timesteps=total_timesteps if drl_lib == "stable_baselines3" else None,
    )
    
    print(f"✅ Entraînement terminé pour {model_name} avec {drl_lib}")

# Exécution
if __name__ == "__main__":
    # Entraîner PPO avec Stable Baselines 3
    run_training(
        model_name="ppo",
        drl_lib="stable_baselines3",
        total_timesteps=100_000
    )
```

## 8.2 Script de Test (test.py)

```python
# ============================================================
# PIPELINE DE TEST/VALIDATION
# ============================================================

from finrl.test import test
from finrl.config import (
    TEST_START_DATE, TEST_END_DATE,
    INDICATORS
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

def run_test(
    model_name="ppo",
    drl_lib="stable_baselines3",
    cwd="./trained_models/ppo"
):
    """
    Teste un modèle entraîné sur la période de test.
    
    Returns:
    --------
    episode_total_assets : list
        Liste des valeurs totales du portefeuille à chaque pas
    """
    
    account_value = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib=drl_lib,
        env=StockTradingEnv,
        model_name=model_name,
        cwd=cwd,
        if_vix=True,
        net_dimension=512,  # Pour ElegantRL
    )
    
    print(f"✅ Test terminé")
    print(f"   Capital initial: $1,000,000")
    print(f"   Capital final: ${account_value[-1]:,.2f}")
    print(f"   Rendement: {(account_value[-1]/1_000_000 - 1)*100:.2f}%")
    
    return account_value

# Exécution
if __name__ == "__main__":
    assets = run_test(
        model_name="ppo",
        drl_lib="stable_baselines3",
        cwd="./trained_models/ppo"
    )
```

## 8.3 Script de Trading (trade.py)

```python
# ============================================================
# PIPELINE DE TRADING
# ============================================================

from finrl.trade import trade
from finrl.config import (
    TRADE_START_DATE, TRADE_END_DATE,
    INDICATORS, ALPACA_API_BASE_URL
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

def run_trade(
    model_name="ppo",
    drl_lib="stable_baselines3",
    trade_mode="backtesting",  # ou "paper_trading"
    cwd="./trained_models/ppo",
    api_key=None,
    api_secret=None
):
    """
    Exécute le trading avec un modèle entraîné.
    
    Parameters:
    -----------
    trade_mode : str
        "backtesting" : simulation sur données historiques
        "paper_trading" : trading simulé en temps réel via Alpaca
    """
    
    kwargs = {}
    
    if trade_mode == "paper_trading":
        # Dimensions nécessaires pour paper trading
        kwargs["state_dim"] = len(DOW_30_TICKER) * (len(INDICATORS) + 3) + 3
        kwargs["action_dim"] = len(DOW_30_TICKER)
        kwargs["net_dimension"] = 512
    
    trade(
        start_date=TRADE_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib=drl_lib,
        env=StockTradingEnv,
        model_name=model_name,
        API_KEY=api_key or "xxx",
        API_SECRET=api_secret or "xxx",
        API_BASE_URL=ALPACA_API_BASE_URL,
        trade_mode=trade_mode,
        cwd=cwd,
        if_vix=True,
        **kwargs
    )
    
    print(f"✅ Trading {'simulé' if trade_mode == 'backtesting' else 'paper'} terminé")

# Exécution
if __name__ == "__main__":
    # Backtesting
    run_trade(
        model_name="ppo",
        trade_mode="backtesting"
    )
```

## 8.4 Pipeline Complet avec Prédiction

```python
# ============================================================
# PRÉDICTION AVEC MODÈLE ENTRAÎNÉ
# ============================================================

from finrl.agents.stablebaselines3.models import DRLAgent

def predict_with_model(
    model_name,
    trade_df,
    env_kwargs,
    model_path
):
    """
    Effectue des prédictions avec un modèle entraîné.
    
    Parameters:
    -----------
    model_name : str
        Nom du modèle
    trade_df : pd.DataFrame
        Données de trading
    env_kwargs : dict
        Configuration de l'environnement
    model_path : str
        Chemin vers le modèle sauvegardé
    
    Returns:
    --------
    df_account_value : pd.DataFrame
        Valeur du compte à chaque étape
    df_actions : pd.DataFrame
        Actions prises à chaque étape
    """
    
    # Créer l'environnement de trading
    e_trade = StockTradingEnv(
        df=trade_df,
        turbulence_threshold=70,
        risk_indicator_col='vix',
        **env_kwargs
    )
    
    # Charger et prédire
    df_account_value, df_actions = DRLAgent.DRL_prediction_load_from_file(
        model_name=model_name,
        environment=e_trade,
        cwd=model_path,
        deterministic=True  # Actions déterministes (pas d'exploration)
    )
    
    return df_account_value, df_actions

# Utilisation
df_account, df_actions = predict_with_model(
    model_name="ppo",
    trade_df=trade,
    env_kwargs=env_kwargs,
    model_path="./trained_models/ppo"
)

print(f"Rendement final: {(df_account['account_value'].iloc[-1] / 1_000_000 - 1) * 100:.2f}%")
```

---

# 9. Backtesting et Visualisation

## 9.1 Statistiques de Performance

```python
# ============================================================
# CALCUL DES STATISTIQUES DE BACKTESTING
# ============================================================

from finrl.plot import backtest_stats, get_daily_return

def calculate_performance_metrics(df_account_value):
    """
    Calcule les métriques de performance standard.
    
    Parameters:
    -----------
    df_account_value : pd.DataFrame
        DataFrame avec colonnes ['date', 'account_value']
    
    Returns:
    --------
    dict : Dictionnaire des métriques
    """
    
    # Utiliser pyfolio pour les stats
    perf_stats = backtest_stats(
        df_account_value,
        value_col_name="account_value"
    )
    
    return perf_stats

# Exemple de sortie:
"""
                             Backtest
Annual return                  23.5%
Cumulative returns             89.2%
Annual volatility              15.3%
Sharpe ratio                    1.54
Calmar ratio                    2.31
Stability                       0.95
Max drawdown                  -10.2%
Omega ratio                     1.28
Sortino ratio                   2.15
Skew                           -0.12
Kurtosis                        3.45
Tail ratio                      1.08
Daily value at risk            -1.5%
"""
```

## 9.2 Comparaison avec Benchmark

```python
# ============================================================
# COMPARAISON AVEC UN BENCHMARK (ex: DJI)
# ============================================================

from finrl.plot import get_baseline, backtest_plot
import matplotlib.pyplot as plt

def compare_with_benchmark(
    df_account_value,
    benchmark_ticker="^DJI",
    start_date=TRADE_START_DATE,
    end_date=TRADE_END_DATE,
    initial_amount=1_000_000
):
    """
    Compare la stratégie DRL avec un benchmark.
    """
    
    # Obtenir les données du benchmark
    df_baseline = get_baseline(
        ticker=benchmark_ticker,
        start=start_date,
        end=end_date
    )
    
    # Normaliser le benchmark au même capital initial
    df_baseline_normalized = df_baseline.copy()
    df_baseline_normalized['account_value'] = (
        df_baseline['close'] / df_baseline['close'].iloc[0] * initial_amount
    )
    
    # Statistiques du benchmark
    print("="*50)
    print(f"STATISTIQUES DU BENCHMARK ({benchmark_ticker})")
    print("="*50)
    baseline_stats = backtest_stats(df_baseline, value_col_name='close')
    
    # Créer le graphique complet (tear sheet)
    backtest_plot(
        df_account_value,
        baseline_start=start_date,
        baseline_end=end_date,
        baseline_ticker=benchmark_ticker,
        value_col_name="account_value"
    )
    
    return df_baseline_normalized

# Exécution
df_dji = compare_with_benchmark(df_account_value_ppo)
```

## 9.3 Visualisation des Rendements

```python
# ============================================================
# VISUALISATION DES RENDEMENTS
# ============================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_cumulative_returns(results_dict, initial_amount=1_000_000):
    """
    Trace les rendements cumulés de plusieurs stratégies.
    
    Parameters:
    -----------
    results_dict : dict
        {nom_strategie: df_account_value}
    """
    
    plt.figure(figsize=(15, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'black']
    
    for i, (name, df) in enumerate(results_dict.items()):
        # Calculer les rendements cumulés
        returns = (df['account_value'] / initial_amount - 1) * 100
        plt.plot(df['date'], returns, label=name, color=colors[i % len(colors)], linewidth=1.5)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rendement Cumulé (%)', fontsize=12)
    plt.title('Comparaison des Stratégies DRL', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('results/cumulative_returns.png', dpi=300)
    plt.show()

# Utilisation
results = {
    'A2C': df_account_value_a2c,
    'PPO': df_account_value_ppo,
    'DDPG': df_account_value_ddpg,
    'TD3': df_account_value_td3,
    'SAC': df_account_value_sac,
    'DJI (Benchmark)': df_dji,
}

plot_cumulative_returns(results)
```

## 9.4 Analyse des Transactions

```python
# ============================================================
# ANALYSE DES TRANSACTIONS
# ============================================================

from finrl.plot import trx_plot

def analyze_transactions(df_trade, df_actions, ticker_list):
    """
    Visualise les signaux d'achat/vente pour chaque actif.
    """
    
    # Tracer pour chaque ticker
    trx_plot(
        df_trade=df_trade,
        df_actions=df_actions,
        ticker_list=ticker_list
    )

# Analyse détaillée des actions
def summarize_actions(df_actions):
    """
    Résume les actions prises par l'agent.
    """
    
    # Compter les transactions par ticker
    transactions = {}
    for col in df_actions.columns:
        if col != 'date':
            buys = (df_actions[col] > 0).sum()
            sells = (df_actions[col] < 0).sum()
            holds = (df_actions[col] == 0).sum()
            transactions[col] = {'Achats': buys, 'Ventes': sells, 'Holds': holds}
    
    df_summary = pd.DataFrame(transactions).T
    df_summary['Total Transactions'] = df_summary['Achats'] + df_summary['Ventes']
    
    print("="*60)
    print("RÉSUMÉ DES TRANSACTIONS PAR ACTIF")
    print("="*60)
    print(df_summary.sort_values('Total Transactions', ascending=False).head(10))
    
    return df_summary

# Exécution
summarize_actions(df_actions_ppo)
```

## 9.5 Tableau de Bord Complet

```python
# ============================================================
# TABLEAU DE BORD DE PERFORMANCE
# ============================================================

def create_performance_dashboard(
    strategy_name,
    df_account_value,
    df_actions,
    df_baseline,
    initial_amount=1_000_000
):
    """
    Crée un tableau de bord complet de performance.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Évolution du portefeuille
    ax1 = axes[0, 0]
    ax1.plot(df_account_value['date'], df_account_value['account_value'], 
             label=strategy_name, color='blue')
    ax1.plot(df_baseline['date'], df_baseline['account_value'], 
             label='Benchmark', color='gray', linestyle='--')
    ax1.set_title('Évolution du Portefeuille')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Valeur ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    running_max = df_account_value['account_value'].cummax()
    drawdown = (df_account_value['account_value'] - running_max) / running_max * 100
    ax2.fill_between(df_account_value['date'], drawdown, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution des rendements
    ax3 = axes[1, 0]
    daily_returns = df_account_value['account_value'].pct_change().dropna() * 100
    ax3.hist(daily_returns, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(daily_returns.mean(), color='red', linestyle='--', 
                label=f'Moyenne: {daily_returns.mean():.2f}%')
    ax3.set_title('Distribution des Rendements Journaliers')
    ax3.set_xlabel('Rendement (%)')
    ax3.set_ylabel('Fréquence')
    ax3.legend()
    
    # 4. Métriques clés
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculer les métriques
    final_value = df_account_value['account_value'].iloc[-1]
    total_return = (final_value / initial_amount - 1) * 100
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    max_dd = drawdown.min()
    
    metrics_text = f"""
    {'='*40}
    MÉTRIQUES DE PERFORMANCE - {strategy_name}
    {'='*40}
    
    Capital Initial:     ${initial_amount:,.0f}
    Capital Final:       ${final_value:,.0f}
    
    Rendement Total:     {total_return:.2f}%
    Sharpe Ratio:        {sharpe:.2f}
    Max Drawdown:        {max_dd:.2f}%
    
    Rendement Annualisé: {total_return * 252 / len(daily_returns):.2f}%
    Volatilité Ann.:     {daily_returns.std() * np.sqrt(252):.2f}%
    
    Nombre de Jours:     {len(daily_returns)}
    Jours Positifs:      {(daily_returns > 0).sum()}
    Jours Négatifs:      {(daily_returns < 0).sum()}
    """
    
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'results/dashboard_{strategy_name}.png', dpi=300)
    plt.show()

# Utilisation
create_performance_dashboard(
    strategy_name="PPO",
    df_account_value=df_account_value_ppo,
    df_actions=df_actions_ppo,
    df_baseline=df_dji
)
```

---

# 10. Paper Trading

## 10.1 Configuration Alpaca

```python
# ============================================================
# PAPER TRADING AVEC ALPACA
# ============================================================

"""
Alpaca Markets offre:
- API gratuite pour paper trading
- Données en temps réel
- Exécution simulée réaliste

Étapes:
1. Créer un compte sur https://alpaca.markets
2. Obtenir API Key et Secret
3. Configurer l'environnement
"""

# Configuration API
ALPACA_CONFIG = {
    "API_KEY": "votre_api_key",
    "API_SECRET": "votre_api_secret",
    "API_BASE_URL": "https://paper-api.alpaca.markets",  # Paper trading
    # Pour live trading: "https://api.alpaca.markets"
}
```

## 10.2 Classe AlpacaPaperTrading

```python
# ============================================================
# ENVIRONNEMENT DE PAPER TRADING
# ============================================================

from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading

def start_paper_trading(
    model_name,
    model_path,
    ticker_list,
    indicators,
    api_key,
    api_secret,
    api_base_url
):
    """
    Démarre le paper trading en temps réel.
    """
    
    # Calculer les dimensions
    stock_dim = len(ticker_list)
    state_dim = 1 + 2 + 3 * stock_dim + len(indicators) * stock_dim
    action_dim = stock_dim
    
    # Créer l'instance de paper trading
    paper_trading = AlpacaPaperTrading(
        ticker_list=ticker_list,
        time_interval="1Min",         # Intervalle de trading (1 minute)
        drl_lib="stable_baselines3",
        model_name=model_name,
        cwd=model_path,
        net_dim=512,                  # Dimension du réseau
        state_dim=state_dim,
        action_dim=action_dim,
        API_KEY=api_key,
        API_SECRET=api_secret,
        API_BASE_URL=api_base_url,
        tech_indicator_list=indicators,
        turbulence_thresh=30,         # Seuil de turbulence
        max_stock=1e2,                # Position max par stock
        latency=None                  # Pas de latence simulée
    )
    
    # Lancer le trading (boucle infinie)
    print("🚀 Démarrage du paper trading...")
    print("   Appuyez sur Ctrl+C pour arrêter")
    
    try:
        paper_trading.run()
    except KeyboardInterrupt:
        print("\n⏹️ Paper trading arrêté")

# Exécution
if __name__ == "__main__":
    start_paper_trading(
        model_name="ppo",
        model_path="./trained_models/ppo",
        ticker_list=["AAPL", "MSFT", "GOOGL"],  # Petit portefeuille pour tester
        indicators=INDICATORS,
        api_key=ALPACA_CONFIG["API_KEY"],
        api_secret=ALPACA_CONFIG["API_SECRET"],
        api_base_url=ALPACA_CONFIG["API_BASE_URL"]
    )
```

## 10.3 Monitoring du Paper Trading

```python
# ============================================================
# MONITORING DU PAPER TRADING
# ============================================================

import alpaca_trade_api as tradeapi

def monitor_account(api_key, api_secret, api_base_url):
    """
    Affiche l'état actuel du compte Alpaca.
    """
    
    api = tradeapi.REST(api_key, api_secret, api_base_url, api_version='v2')
    
    # Informations du compte
    account = api.get_account()
    
    print("="*50)
    print("ÉTAT DU COMPTE ALPACA")
    print("="*50)
    print(f"ID du compte: {account.id}")
    print(f"Status: {account.status}")
    print(f"\nCash: ${float(account.cash):,.2f}")
    print(f"Valeur du portefeuille: ${float(account.portfolio_value):,.2f}")
    print(f"Equity: ${float(account.equity):,.2f}")
    
    # Positions actuelles
    positions = api.list_positions()
    
    if positions:
        print(f"\nPOSITIONS ({len(positions)}):")
        print("-"*50)
        for pos in positions:
            pl = float(pos.unrealized_pl)
            pl_pct = float(pos.unrealized_plpc) * 100
            print(f"  {pos.symbol}: {pos.qty} actions @ ${float(pos.avg_entry_price):.2f}")
            print(f"    P/L: ${pl:,.2f} ({pl_pct:+.2f}%)")
    else:
        print("\nAucune position ouverte")
    
    # Ordres récents
    orders = api.list_orders(status='all', limit=5)
    
    if orders:
        print(f"\nDERNIERS ORDRES:")
        print("-"*50)
        for order in orders:
            print(f"  {order.side.upper()} {order.symbol}: {order.qty} @ {order.type}")
            print(f"    Status: {order.status}")

# Exécution
monitor_account(
    ALPACA_CONFIG["API_KEY"],
    ALPACA_CONFIG["API_SECRET"],
    ALPACA_CONFIG["API_BASE_URL"]
)
```

---

# 11. Exemples Complets

## 11.1 Exemple Complet: Trading DOW 30

```python
# ============================================================
# EXEMPLE COMPLET: TRADING DOW 30 AVEC MULTIPLE MODÈLES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, get_baseline
from finrl.config import INDICATORS
from finrl.config_tickers import DOW_30_TICKER
from stable_baselines3.common.logger import configure
import itertools
import os

# ============================================================
# ÉTAPE 1: CONFIGURATION
# ============================================================

TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2023-03-01'

# Créer les répertoires
os.makedirs('datasets', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("📊 Configuration:")
print(f"   Train: {TRAIN_START_DATE} → {TRAIN_END_DATE}")
print(f"   Trade: {TRADE_START_DATE} → {TRADE_END_DATE}")
print(f"   Tickers: {len(DOW_30_TICKER)} actions (DOW 30)")

# ============================================================
# ÉTAPE 2: TÉLÉCHARGEMENT ET PRÉTRAITEMENT
# ============================================================

print("\n📥 Téléchargement des données...")
df = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    ticker_list=DOW_30_TICKER
).fetch_data()

print(f"   Données brutes: {df.shape}")

# Feature Engineering
print("\n🔧 Feature Engineering...")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False
)
processed = fe.preprocess_data(df)

# Compléter les données manquantes
list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(
    processed['date'].min(), 
    processed['date'].max()
).astype(str))

combination = list(itertools.product(list_date, list_ticker))
processed_full = pd.DataFrame(
    combination, 
    columns=["date", "tic"]
).merge(processed, on=["date", "tic"], how="left")

processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic']).fillna(0)

print(f"   Données traitées: {processed_full.shape}")

# ============================================================
# ÉTAPE 3: DIVISION TRAIN/TRADE
# ============================================================

train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)

print(f"\n📊 Division des données:")
print(f"   Train: {len(train)} lignes, {len(train.date.unique())} jours")
print(f"   Trade: {len(trade)} lignes, {len(trade.date.unique())} jours")

# ============================================================
# ÉTAPE 4: CONFIGURATION DE L'ENVIRONNEMENT
# ============================================================

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1_000_000,
    "num_stock_shares": [0] * stock_dimension,
    "buy_cost_pct": [0.001] * stock_dimension,
    "sell_cost_pct": [0.001] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

print(f"\n🎮 Environnement configuré:")
print(f"   Stock dimension: {stock_dimension}")
print(f"   State space: {state_space}")

# ============================================================
# ÉTAPE 5: ENTRAÎNEMENT DES MODÈLES
# ============================================================

TIMESTEPS = 50_000  # Réduire pour test rapide

models = {}
results = {}

# Liste des modèles à entraîner
model_configs = {
    "a2c": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
    "ppo": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64},
    "ddpg": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
    "sac": {"batch_size": 64, "buffer_size": 100000, "learning_rate": 0.0001, "learning_starts": 100, "ent_coef": "auto_0.1"},
}

for model_name, params in model_configs.items():
    print(f"\n🚀 Entraînement {model_name.upper()}...")
    
    agent = DRLAgent(env=env_train)
    model = agent.get_model(model_name, model_kwargs=params)
    
    # Logger
    tmp_path = f'results/{model_name}'
    os.makedirs(tmp_path, exist_ok=True)
    logger = configure(tmp_path, ["stdout", "csv"])
    model.set_logger(logger)
    
    # Entraînement
    trained = agent.train_model(
        model=model,
        tb_log_name=model_name,
        total_timesteps=TIMESTEPS
    )
    
    trained.save(f"trained_models/{model_name}")
    models[model_name] = trained
    print(f"   ✅ {model_name.upper()} entraîné et sauvegardé")

# ============================================================
# ÉTAPE 6: TEST ET PRÉDICTION
# ============================================================

print("\n📈 Test des modèles...")

# Seuil de risque
data_risk = processed_full[
    (processed_full.date < TRAIN_END_DATE) & 
    (processed_full.date >= TRAIN_START_DATE)
].drop_duplicates(subset=['date'])
turbulence_thresh = data_risk.turbulence.quantile(0.996)

# Environnement de trading
e_trade_gym = StockTradingEnv(
    df=trade,
    turbulence_threshold=turbulence_thresh,
    risk_indicator_col='turbulence',
    **env_kwargs
)

for model_name, trained_model in models.items():
    print(f"\n   Testing {model_name.upper()}...")
    
    df_account, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym
    )
    
    results[model_name] = {
        'account': df_account,
        'actions': df_actions,
        'final_value': df_account['account_value'].iloc[-1],
        'return': (df_account['account_value'].iloc[-1] / 1_000_000 - 1) * 100
    }
    
    print(f"      Rendement: {results[model_name]['return']:.2f}%")

# ============================================================
# ÉTAPE 7: BENCHMARK
# ============================================================

print("\n📊 Calcul du benchmark...")

df_dji = get_baseline(
    ticker="^DJI",
    start=TRADE_START_DATE,
    end=TRADE_END_DATE
)

# Normaliser
initial_price = df_dji['close'].iloc[0]
df_dji['account_value'] = df_dji['close'] / initial_price * 1_000_000
benchmark_return = (df_dji['account_value'].iloc[-1] / 1_000_000 - 1) * 100

print(f"   DJI Benchmark: {benchmark_return:.2f}%")

# ============================================================
# ÉTAPE 8: VISUALISATION FINALE
# ============================================================

print("\n📊 Création des visualisations...")

plt.figure(figsize=(15, 8))

colors = {'a2c': 'blue', 'ppo': 'green', 'ddpg': 'orange', 'sac': 'red'}

for model_name, result in results.items():
    df = result['account']
    returns = (df['account_value'] / 1_000_000 - 1) * 100
    plt.plot(df['date'], returns, label=f"{model_name.upper()} ({result['return']:.1f}%)", 
             color=colors.get(model_name, 'gray'), linewidth=1.5)

# Benchmark
bench_returns = (df_dji['account_value'] / 1_000_000 - 1) * 100
plt.plot(df_dji['date'], bench_returns, label=f"DJI ({benchmark_return:.1f}%)", 
         color='black', linestyle='--', linewidth=2)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Rendement Cumulé (%)', fontsize=12)
plt.title('Comparaison des Stratégies DRL vs Benchmark (DOW 30)', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/comparison_final.png', dpi=300)
plt.show()

# ============================================================
# ÉTAPE 9: RAPPORT FINAL
# ============================================================

print("\n" + "="*60)
print("RAPPORT FINAL")
print("="*60)
print(f"\nPériode de trading: {TRADE_START_DATE} → {TRADE_END_DATE}")
print(f"Capital initial: $1,000,000")
print("\n" + "-"*60)
print(f"{'Modèle':<15} {'Valeur Finale':>20} {'Rendement':>15}")
print("-"*60)

for model_name in sorted(results.keys(), key=lambda x: results[x]['return'], reverse=True):
    r = results[model_name]
    print(f"{model_name.upper():<15} ${r['final_value']:>18,.0f} {r['return']:>14.2f}%")

print("-"*60)
print(f"{'DJI (Benchmark)':<15} ${df_dji['account_value'].iloc[-1]:>18,.0f} {benchmark_return:>14.2f}%")
print("="*60)

# Meilleur modèle
best_model = max(results.keys(), key=lambda x: results[x]['return'])
print(f"\n🏆 Meilleur modèle: {best_model.upper()} avec {results[best_model]['return']:.2f}%")
```

---

# 12. Optimisation des Hyperparamètres

## 12.1 Optuna pour l'Optimisation

```python
# ============================================================
# OPTIMISATION AVEC OPTUNA
# ============================================================

import optuna
from stable_baselines3 import PPO

def objective(trial):
    """
    Fonction objectif pour Optuna.
    Retourne le rendement négatif (Optuna minimise).
    """
    
    # Hyperparamètres à optimiser
    params = {
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
    }
    
    # Créer et entraîner le modèle
    agent = DRLAgent(env=env_train)
    model = agent.get_model("ppo", model_kwargs=params)
    
    trained = agent.train_model(
        model=model,
        tb_log_name="optuna",
        total_timesteps=20_000  # Réduit pour optimisation rapide
    )
    
    # Évaluer
    df_account, _ = DRLAgent.DRL_prediction(
        model=trained,
        environment=e_trade_gym
    )
    
    final_return = df_account['account_value'].iloc[-1] / 1_000_000 - 1
    
    return -final_return  # Négatif car Optuna minimise

# Lancer l'optimisation
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials ou 1 heure

print("Meilleurs hyperparamètres:")
print(study.best_params)
print(f"Meilleur rendement: {-study.best_value * 100:.2f}%")
```

## 12.2 Grid Search Manuel

```python
# ============================================================
# GRID SEARCH MANUEL
# ============================================================

from itertools import product

def grid_search_ppo():
    """
    Grid search sur les hyperparamètres PPO.
    """
    
    # Grille de paramètres
    param_grid = {
        "n_steps": [1024, 2048],
        "batch_size": [64, 128],
        "learning_rate": [1e-4, 2.5e-4],
        "ent_coef": [0.01, 0.02],
    }
    
    # Générer toutes les combinaisons
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    
    results = []
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        print(f"\nTest: {params}")
        
        try:
            agent = DRLAgent(env=env_train)
            model = agent.get_model("ppo", model_kwargs=params)
            
            trained = agent.train_model(
                model=model,
                tb_log_name="grid",
                total_timesteps=30_000
            )
            
            df_account, _ = DRLAgent.DRL_prediction(
                model=trained,
                environment=e_trade_gym
            )
            
            final_return = (df_account['account_value'].iloc[-1] / 1_000_000 - 1) * 100
            
            results.append({**params, 'return': final_return})
            print(f"   Rendement: {final_return:.2f}%")
            
        except Exception as e:
            print(f"   Erreur: {e}")
            results.append({**params, 'return': None})
    
    # Trier par rendement
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('return', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS")
    print("="*60)
    print(df_results.head())
    
    return df_results

# Exécution
# results_df = grid_search_ppo()
```

---

# 13. Glossaire DRL

## Termes Généraux

| Acronyme | Signification | Description |
|----------|---------------|-------------|
| **DRL** | Deep Reinforcement Learning | Apprentissage par renforcement avec réseaux de neurones profonds |
| **RL** | Reinforcement Learning | Apprentissage par renforcement |
| **MDP** | Markov Decision Process | Formalisation mathématique du problème RL |
| **POMDP** | Partially Observable MDP | MDP avec observations partielles |

## Algorithmes

| Acronyme | Signification | Type |
|----------|---------------|------|
| **A2C** | Advantage Actor-Critic | On-Policy |
| **A3C** | Asynchronous A2C | On-Policy, Distribué |
| **PPO** | Proximal Policy Optimization | On-Policy |
| **TRPO** | Trust Region Policy Optimization | On-Policy |
| **DDPG** | Deep Deterministic Policy Gradient | Off-Policy |
| **TD3** | Twin Delayed DDPG | Off-Policy |
| **SAC** | Soft Actor-Critic | Off-Policy |
| **DQN** | Deep Q-Network | Off-Policy, Discret |

## Concepts RL

| Terme | Description |
|-------|-------------|
| **Policy (π)** | Stratégie qui mappe états vers actions |
| **Value Function (V)** | Espérance des récompenses futures depuis un état |
| **Q-Function (Q)** | Espérance des récompenses futures pour un couple (état, action) |
| **Advantage (A)** | Différence entre Q et V: A(s,a) = Q(s,a) - V(s) |
| **Reward (r)** | Récompense immédiate après une action |
| **Discount (γ)** | Facteur d'actualisation des récompenses futures |
| **Episode** | Une séquence complète du début à la fin |
| **Timestep** | Un pas dans l'environnement |
| **Rollout** | Collection d'expériences pendant plusieurs timesteps |

## Indicateurs Techniques

| Acronyme | Signification |
|----------|---------------|
| **MACD** | Moving Average Convergence Divergence |
| **RSI** | Relative Strength Index |
| **CCI** | Commodity Channel Index |
| **SMA** | Simple Moving Average |
| **EMA** | Exponential Moving Average |
| **VIX** | CBOE Volatility Index |

## Métriques de Performance

| Métrique | Description |
|----------|-------------|
| **Sharpe Ratio** | Rendement excédentaire / Volatilité |
| **Sortino Ratio** | Sharpe avec volatilité à la baisse seulement |
| **Max Drawdown** | Perte maximale depuis un pic |
| **Calmar Ratio** | Rendement annualisé / Max Drawdown |
| **Alpha** | Rendement excédentaire vs benchmark |
| **Beta** | Sensibilité au marché |

---

## 📚 Ressources Additionnelles

### Documentation Officielle
- [FinRL GitHub](https://github.com/AI4Finance-Foundation/FinRL)
- [FinRL Documentation](https://finrl.readthedocs.io/)
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)

### Papers Académiques
- "Practical Deep Reinforcement Learning Approach for Stock Trading" (NeurIPS 2018)
- "Deep Reinforcement Learning for Automated Stock Trading" (ICAIF 2020)
- "FinRL-Meta: Market Environments and Benchmarks" (NeurIPS 2022)

### Tutoriels
- [FinRL Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials)
- [AI4Finance YouTube](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g)

---

**Document créé pour HelixOne Visual Code**
**Version: 1.0**
**Date: 2025**
