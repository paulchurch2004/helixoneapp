# ML Trading System - Documentation Complète

> Système de Machine Learning pour prédictions boursières avec XGBoost + LSTM

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Entraînement](#entraînement)
- [Backtesting](#backtesting)
- [API Reference](#api-reference)
- [Performance](#performance)

---

## 🎯 Vue d'ensemble

Ce système ML combine **XGBoost** (classification) et **LSTM** (régression) pour prédire les mouvements de prix d'actions avec une précision cible de **75-80%** et un Sharpe ratio **>2.0**.

### Caractéristiques principales

- ✅ **100% Gratuit** - Yahoo Finance, FRED API, pas de coûts
- ✅ **50+ Features** - Indicateurs techniques, macro, sentiment, volume
- ✅ **Multi-horizon** - Prédictions 1j, 3j, 7j
- ✅ **Ensemble learning** - Combine XGBoost + LSTM
- ✅ **Backtesting réaliste** - Backtrader avec slippage et commissions
- ✅ **Google Colab** - Entraînement GPU gratuit
- ✅ **Walk-forward validation** - Évite l'overfitting
- ✅ **Monte Carlo** - 10k+ simulations pour risk management
- ✅ **Dashboards Plotly** - Visualisations interactives

### Objectifs de performance

| Métrique | Objectif |
|----------|----------|
| Accuracy (classification) | >72% |
| MAPE (régression) | <5% |
| Sharpe Ratio | >2.0 |
| Max Drawdown | <20% |
| Win Rate | >55% |

---

## 🏗️ Architecture

```
ml_models/
├── data_collection/          # Téléchargement données
│   ├── yahoo_finance_downloader.py
│   ├── fred_macro_downloader.py
│   └── data_cache.py
│
├── feature_engineering/      # Création features
│   ├── technical_indicators.py   # RSI, MACD, Bollinger, etc.
│   ├── macro_features.py         # Fed funds, inflation, VIX
│   ├── sentiment_features.py     # Reddit, news, StockTwits
│   ├── volume_features.py        # Volume analysis
│   └── feature_selector.py       # Top 50 selection
│
├── models/                   # Modèles ML
│   ├── xgboost_classifier.py     # Classification UP/DOWN/FLAT
│   ├── lstm_predictor.py         # LSTM price prediction
│   ├── ensemble_model.py         # Combine les deux
│   └── model_trainer.py          # Script d'entraînement
│
├── backtesting/              # Backtesting & validation
│   ├── backtest_engine.py        # Backtrader strategy
│   ├── cost_models.py            # Commissions, slippage
│   ├── walk_forward_validator.py # Validation rolling
│   ├── performance_metrics.py    # Sharpe, Sortino, Calmar
│   └── monte_carlo_simulator.py  # Simulations MC
│
├── visualization/            # Dashboards
│   └── visualization.py          # Plotly charts
│
└── saved_models/             # Modèles entraînés
    └── {TICKER}/
        ├── xgboost/
        ├── lstm/
        └── ensemble/
```

---

## 📦 Installation

### Prérequis

- Python 3.9+
- pip

### Installation locale

```bash
# 1. Cloner le repo
cd helixone-backend

# 2. Installer dépendances ML
pip install -r requirements_ml.txt

# 3. Configuration FRED API (gratuit)
# S'inscrire sur https://fred.stlouisfed.org/docs/api/api_key.html
# Créer .env avec:
FRED_API_KEY=your_api_key_here
```

### Google Colab (GPU gratuit)

```python
# Dans un notebook Colab:

# 1. Uploader ml_models/ et requirements_ml.txt

# 2. Installer dépendances
!pip install -r requirements_ml.txt

# 3. Monter Google Drive (pour sauvegarder modèles)
from google.colab import drive
drive.mount('/content/drive')

# 4. Prêt à entraîner!
```

---

## 🚀 Quick Start

### 1. Entraîner un modèle (5 minutes)

```bash
# Entraîner AAPL avec ensemble (XGBoost + LSTM)
python ml_models/model_trainer.py --ticker AAPL --mode ensemble --lstm-epochs 50

# Outputs:
# - Modèles: ml_models/saved_models/AAPL/
# - Dataset: ml_models/results/AAPL_dataset.csv
# - Logs: terminal
```

### 2. Backtester la stratégie

```python
from ml_models.backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine()

results = engine.run_backtest(
    ticker='AAPL',
    model_path='ml_models/saved_models/AAPL/ensemble',
    features=['rsi_14', 'macd', 'bb_width', 'sma_20', 'volume_ratio'],
    start_date='2022-01-01',
    initial_cash=100000
)

# Résultats:
# - Total return: +XX%
# - Sharpe ratio: X.XX
# - Max drawdown: -XX%
# - Win rate: XX%
```

### 3. Obtenir un signal de trading

```python
from ml_models.models.ensemble_model import MultiHorizonEnsemble
from ml_models.data_collection.data_cache import DataCache

# Charger modèle
ensemble = MultiHorizonEnsemble()
ensemble.load_all('ml_models/saved_models/AAPL/ensemble')

# Télécharger données récentes
cache = DataCache()
data = cache.get_ml_dataset(['AAPL'], start_date='2023-01-01')
df = data['AAPL']

# Ajouter features (simplified)
from ml_models.feature_engineering.technical_indicators import TechnicalIndicators
tech = TechnicalIndicators()
df = tech.add_all_indicators(df)

# Obtenir signal
signal = ensemble.get_multi_horizon_signals(df, features=df.columns)

print(signal)
# {
#   'signals': {
#     '1d': {'action': 'BUY', 'confidence': 0.85, ...},
#     '3d': {'action': 'BUY', 'confidence': 0.78, ...},
#     '7d': {'action': 'HOLD', 'confidence': 0.65, ...}
#   },
#   'consensus': {'action': 'BUY', 'score': 66.7, 'confidence': 0.76}
# }
```

---

## 📊 Modules

### 1. Data Collection

#### Yahoo Finance Downloader

Télécharge données historiques avec cache SQLite.

```python
from ml_models.data_collection.yahoo_finance_downloader import YahooFinanceDownloader

downloader = YahooFinanceDownloader()

# Télécharger 1 ticker
data = downloader.download_historical_data(
    tickers=['AAPL'],
    start_date='2020-01-01'
)

# Télécharger S&P 500 complet (parallèle)
sp500_data = downloader.download_sp500(
    start_date='2020-01-01',
    max_workers=10
)
```

**Features**:
- Cache SQLite (évite re-téléchargement)
- Téléchargement parallèle
- Mises à jour incrémentales
- S&P 500 auto-download

#### FRED Macro Downloader

Télécharge 20+ indicateurs macro-économiques.

```python
from ml_models.data_collection.fred_macro_downloader import FredMacroDownloader

downloader = FredMacroDownloader(api_key='your_key')

# Télécharger tous indicateurs
macro_data = downloader.download_all_indicators(
    start_date='2020-01-01'
)

# Colonnes:
# - DFF (Fed Funds Rate)
# - DGS10, DGS2 (Treasury yields)
# - CPIAUCSL (Inflation)
# - UNRATE (Unemployment)
# - VIXCLS (VIX)
# + 15 autres + derived indicators
```

### 2. Feature Engineering

#### Technical Indicators

50+ indicateurs techniques avec pandas-ta.

```python
from ml_models.feature_engineering.technical_indicators import TechnicalIndicators

tech = TechnicalIndicators()
df = tech.add_all_indicators(df)

# Ajoute:
# - Trend: SMA, EMA, MACD, ADX
# - Momentum: RSI, Stochastic, ROC, Williams %R
# - Volatility: Bollinger Bands, ATR, Keltner Channel
# - Volume: OBV, CMF, MFI
# - Candlestick patterns
```

#### Feature Selector

Sélectionne top 50 features les plus prédictives.

```python
from ml_models.feature_engineering.feature_selector import FeatureSelector

selector = FeatureSelector(max_features=50)

selected_features = selector.select_features(
    X=df[all_features],
    y=labels,
    method='xgboost'  # ou 'rf', 'rfe'
)

# Méthodes:
# 1. Variance threshold (éliminer constantes)
# 2. Correlation (éliminer >0.95 corrélées)
# 3. XGBoost feature importance
```

### 3. ML Models

#### XGBoost Classifier

Classification UP/DOWN/FLAT (3 classes).

```python
from ml_models.models.xgboost_classifier import MultiHorizonClassifier

clf = MultiHorizonClassifier()

# Entraîner 3 horizons (1j, 3j, 7j)
clf.train_all(
    df=df,
    features=selected_features,
    train_split=0.8,
    optimize=True,       # Optuna hyperparameter tuning
    n_trials=50
)

# Obtenir signal
signal = clf.get_multi_horizon_signal(df[features])
# {'1d': {'prediction': 'UP', 'confidence': 0.85, 'action': 'BUY'}, ...}
```

**Classes**:
- **UP**: Price change > +1%
- **DOWN**: Price change < -1%
- **FLAT**: Price change entre -1% et +1%

#### LSTM Predictor

Régression de prix avec LSTM.

```python
from ml_models.models.lstm_predictor import MultiHorizonLSTM

lstm = MultiHorizonLSTM(lookback_window=30, lstm_units=[64, 32])

# Entraîner
lstm.train_all(
    df=df,
    features=features,
    epochs=100,
    batch_size=32
)

# Prédire
predictions = lstm.get_multi_horizon_predictions(df, features)
# {
#   '1d': {'predicted_price': 152.50, 'price_change_pct': +2.3, ...},
#   '3d': {'predicted_price': 155.00, 'price_change_pct': +4.1, ...},
#   ...
# }
```

#### Ensemble Model

Combine XGBoost + LSTM avec weighted average.

```python
from ml_models.models.ensemble_model import MultiHorizonEnsemble

ensemble = MultiHorizonEnsemble(xgb_weight=0.5, lstm_weight=0.5)

# Entraîner (entraîne XGBoost ET LSTM)
ensemble.train_all(
    df=df,
    features=features,
    xgb_trials=30,
    lstm_epochs=100
)

# Signal combiné
signal = ensemble.get_multi_horizon_signals(df, features)
# Consensus entre XGBoost et LSTM
```

### 4. Backtesting

#### Backtest Engine

Backtrader avec stratégie ML.

```python
from ml_models.backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine()

results = engine.run_backtest(
    ticker='AAPL',
    model_path='ml_models/saved_models/AAPL/ensemble',
    features=selected_features,
    start_date='2022-01-01',
    initial_cash=100000,
    commission=0.001,           # 0.1%
    confidence_threshold=0.6,   # Trade si confiance >60%
    stop_loss_pct=-0.10,        # Stop loss -10%
    take_profit_pct=0.20        # Take profit +20%
)

# Résultats:
# - Sharpe ratio
# - Max drawdown
# - Win rate
# - Total return
# - Liste des trades
```

#### Performance Metrics

Métriques de trading professionnelles.

```python
from ml_models.backtesting.performance_metrics import PerformanceMetrics

calc = PerformanceMetrics(risk_free_rate=0.02)

metrics = calc.calculate_all(prices=equity_curve)

# Métriques:
# - Total return, CAGR
# - Volatility, Max drawdown
# - Sharpe, Sortino, Calmar
# - VaR, CVaR
# - Win rate, Profit factor
```

#### Monte Carlo Simulator

Simulations pour quantifier l'incertitude.

```python
from ml_models.backtesting.monte_carlo_simulator import MonteCarloSimulator

sim = MonteCarloSimulator(n_simulations=10000, forecast_days=252)

results = sim.run_simulation(
    returns=historical_returns,
    initial_value=100000,
    method='historical'  # ou 'normal', 't-student'
)

# Résultats:
# - Percentiles (P5, P25, P50, P75, P95)
# - VaR, CVaR
# - Probabilités (profit, loss >10%, gain >20%)
# - 10k trajectoires
```

### 5. Visualization

Dashboards Plotly interactifs.

```python
from ml_models.visualization.visualization import MLVisualizer

viz = MLVisualizer(template='plotly_dark')

# Dashboard complet
fig = viz.create_dashboard(
    equity=equity_curve,
    returns=returns,
    benchmark=sp500_benchmark,
    feature_importance=importances
)

# Sauvegarder HTML
viz.save_html(fig, 'results/dashboard.html')

# Ou afficher
fig.show()
```

**Charts**:
- Equity curve vs benchmark
- Drawdown
- Returns distribution + Q-Q plot
- Monthly returns heatmap
- Feature importance
- Rolling Sharpe
- Monte Carlo fan chart

---

## 🎓 Entraînement

### Option 1: Single Ticker (local)

```bash
python ml_models/model_trainer.py \
    --ticker AAPL \
    --mode ensemble \
    --start-date 2018-01-01 \
    --xgb-trials 50 \
    --lstm-epochs 100
```

### Option 2: Multiple Tickers

```bash
python ml_models/model_trainer.py \
    --tickers "AAPL,MSFT,GOOGL,AMZN,TSLA" \
    --mode ensemble \
    --xgb-trials 20 \
    --lstm-epochs 50
```

### Option 3: Google Colab (GPU)

```python
# Notebook Colab
!python ml_models/model_trainer.py \
    --ticker AAPL \
    --mode ensemble \
    --lstm-epochs 200 \  # Plus d'époques avec GPU
    --output-dir /content/drive/MyDrive/models
```

### Walk-Forward Validation

```python
from ml_models.backtesting.walk_forward_validator import WalkForwardValidator

validator = WalkForwardValidator(
    train_window_days=252,  # 1 an train
    test_window_days=63,    # 3 mois test
    step_days=21            # Avancer 1 mois
)

results = validator.validate(
    df=df,
    train_fn=train_function,
    predict_fn=predict_function,
    metric_fn=accuracy_function
)

# Résultats:
# - Score moyen sur toutes les windows
# - Écart-type (stabilité)
# - Min/max scores
```

---

## 📈 Performance Attendue

### Métriques de prédiction

| Horizon | Accuracy | MAPE | R² |
|---------|----------|------|-----|
| 1 jour  | 72-75%  | 2-3% | 0.65-0.75 |
| 3 jours | 70-73%  | 3-4% | 0.60-0.70 |
| 7 jours | 68-71%  | 4-5% | 0.55-0.65 |

### Métriques de trading

| Métrique | Valeur attendue |
|----------|-----------------|
| Sharpe Ratio | 1.8 - 2.5 |
| Max Drawdown | -15% à -20% |
| Win Rate | 55-60% |
| Profit Factor | 1.5 - 2.0 |
| CAGR | 15-25% |

---

## 🔧 API Reference

### model_trainer.py

```
Arguments:
  --ticker TICKER           Single ticker à entraîner
  --tickers TICKERS         Liste de tickers (comma-separated)
  --mode {xgboost,lstm,ensemble}
  --start-date DATE         Date de début (YYYY-MM-DD)
  --no-optimize             Désactiver Optuna
  --xgb-trials N            Nombre de trials Optuna (défaut: 30)
  --lstm-epochs N           Nombre d'époques LSTM (défaut: 100)
  --output-dir PATH         Répertoire output (défaut: ml_models/saved_models)
  --log-level {DEBUG,INFO,WARNING}
```

---

## 💡 Best Practices

### 1. Éviter l'overfitting

- ✅ Utiliser walk-forward validation
- ✅ Ne pas optimiser sur données de test
- ✅ Feature selection (éliminer features non prédictives)
- ✅ Regularization (dropout LSTM, reg_alpha/lambda XGBoost)
- ✅ Early stopping

### 2. Data quality

- ✅ Vérifier NaN (dropna ou fillna intelligent)
- ✅ Normaliser features pour LSTM
- ✅ Aligner dates (merge macro avec prix)
- ✅ Lookback minimum pour LSTM (60+ jours)

### 3. Production

- ✅ Sauvegarder scaler avec LSTM
- ✅ Versionner modèles (dates dans noms)
- ✅ Re-entraîner régulièrement (monthly)
- ✅ Monitor drift (accuracy baisse = re-train)
- ✅ Logging complet

---

## 🎯 Prochaines étapes

### Phase 2: Alternative Data (Semaine 4-6)

- [ ] Web scraping légal (Yahoo Finance news)
- [ ] Reddit sentiment (PRAW API)
- [ ] GitHub activity (commits, stars)
- [ ] Job postings scraping

### Phase 3: Market Microstructure (Semaine 7-9)

- [ ] Order flow imbalance
- [ ] Bid-ask spread
- [ ] Trade size distribution
- [ ] Time & Sales analysis

### Phase 4: Network Analysis (Semaine 10-12)

- [ ] Corrélations sectorielles
- [ ] Supply chain networks
- [ ] Insider trading networks

### Phase 5: Deploy (Semaine 13-15)

- [ ] API REST FastAPI
- [ ] Scheduler automatique
- [ ] Alertes email/SMS
- [ ] Dashboard temps réel

---

## 📚 Ressources

### Documentation

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow/Keras LSTM](https://www.tensorflow.org/guide/keras/rnn)
- [Backtrader](https://www.backtrader.com/)
- [Plotly](https://plotly.com/python/)
- [pandas-ta](https://github.com/twopirllc/pandas-ta)

### Papers

- "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
- "LSTM Networks for Stock Market Prediction" (Various)
- "The Sharpe Ratio" (Sharpe, 1994)

---

## ⚠️ Disclaimer

Ce système est à but éducatif. Le trading comporte des risques. Past performance ne garantit pas future results. Toujours tester en paper trading avant le live.

---

## 📞 Support

Questions? Check:
1. README (ce fichier)
2. Docstrings dans le code
3. Exemples `if __name__ == '__main__'`
4. GitHub issues

---

**Version**: 1.0
**Dernière mise à jour**: 2024-01-XX
**License**: MIT
