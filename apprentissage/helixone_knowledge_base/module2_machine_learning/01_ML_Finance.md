# Machine Learning pour la Finance

> **Sources**: Cours ENSAE/Polytechnique, NYU Tandon, Stanford CS229
> **Références**: López de Prado "Advances in Financial ML", Gu, Kelly, Xiu (2020)
> **Extrait pour la base de connaissances HelixOne**

---

## 1. Introduction

### 1.1 Pourquoi le ML en Finance?

**Défis spécifiques:**
- **Signal-to-noise ratio très faible**: Les marchés sont proches de l'efficience
- **Non-stationnarité**: Les relations changent dans le temps
- **Régimes multiples**: Crises, bulles, périodes calmes
- **Données limitées**: Peu d'observations indépendantes
- **Overfitting facile**: Beaucoup de variables, peu de signal

**Applications:**
- Prédiction des rendements (alpha generation)
- Construction de portefeuille
- Gestion des risques
- Détection de fraude
- Trading algorithmique
- Pricing de dérivés

### 1.2 Types de Problèmes

| Type | Exemples en Finance |
|------|---------------------|
| **Régression** | Prédiction de rendements, volatilité |
| **Classification** | Direction du marché, défaut de crédit |
| **Clustering** | Segmentation de clients, régimes de marché |
| **Séries temporelles** | Forecasting, détection d'anomalies |
| **Reinforcement Learning** | Trading optimal, exécution |

---

## 2. Préparation des Données Financières

### 2.1 Types de Données

**Données de marché:**
- Prix (OHLCV)
- Order book
- Trades tick-by-tick

**Données fondamentales:**
- États financiers
- Ratios financiers
- Earnings

**Données alternatives:**
- Sentiment (news, social media)
- Satellite imagery
- Web scraping

### 2.2 Feature Engineering

**Features de prix:**
```python
# Rendements
returns = np.log(prices / prices.shift(1))
returns_5d = np.log(prices / prices.shift(5))

# Volatilité réalisée
volatility = returns.rolling(20).std() * np.sqrt(252)

# Momentum
momentum = prices / prices.rolling(20).mean() - 1

# RSI (Relative Strength Index)
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

**Features techniques:**
- Moving averages (SMA, EMA)
- Bollinger Bands
- MACD
- Volume indicators

### 2.3 Problèmes de Data Leakage

**Sources de leakage:**
1. **Look-ahead bias**: Utiliser des données futures
2. **Survivorship bias**: Ignorer les actifs qui ont disparu
3. **Point-in-time data**: Utiliser des données révisées

**Solutions:**
```python
# Cross-validation temporelle correcte
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Entraîner et évaluer
```

### 2.4 Labeling

**Triple Barrier Method** (López de Prado):
```python
def triple_barrier_labels(prices, horizon, pt_sl=[1, 1], min_ret=0.01):
    """
    Labels basés sur les barrières:
    - Upper barrier: take profit
    - Lower barrier: stop loss
    - Vertical barrier: max holding period
    """
    labels = []
    for i in range(len(prices) - horizon):
        window = prices[i:i+horizon]
        ret = (window / prices[i]) - 1
        
        # Check barriers
        upper_hit = ret.max() >= pt_sl[0] * min_ret
        lower_hit = ret.min() <= -pt_sl[1] * min_ret
        
        if upper_hit and lower_hit:
            # Première barrière touchée
            upper_idx = ret[ret >= pt_sl[0]*min_ret].index[0]
            lower_idx = ret[ret <= -pt_sl[1]*min_ret].index[0]
            labels.append(1 if upper_idx < lower_idx else -1)
        elif upper_hit:
            labels.append(1)
        elif lower_hit:
            labels.append(-1)
        else:
            labels.append(0)
    
    return np.array(labels)
```

---

## 3. Modèles de Régression

### 3.1 Régression Linéaire

**Modèle:**
```
y = X·β + ε
```

**Estimateur OLS:**
```
β̂ = (X'X)^{-1} X'y
```

**Application**: Modèle de facteurs (Fama-French)
```python
import statsmodels.api as sm

# Régression Fama-French 3 facteurs
X = sm.add_constant(df[['Mkt-RF', 'SMB', 'HML']])
y = df['excess_return']
model = sm.OLS(y, X).fit()
print(model.summary())
```

### 3.2 Régularisation

**Ridge Regression (L2):**
```
β̂_ridge = argmin ||y - Xβ||² + λ||β||²
```

**LASSO (L1):**
```
β̂_lasso = argmin ||y - Xβ||² + λ||β||₁
```

**Elastic Net:**
```
β̂_en = argmin ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²
```

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Ridge avec CV
ridge = Ridge()
params = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(ridge, params, cv=TimeSeriesSplit(5))
grid.fit(X_train, y_train)
```

### 3.3 Random Forests

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=50,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

### 3.4 Gradient Boosting

```python
import xgboost as xgb
import lightgbm as lgb

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,  # L1
    reg_lambda=1  # L2
)

# LightGBM (plus rapide)
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.8,
    feature_fraction=0.8,
    reg_alpha=1,
    reg_lambda=1
)
```

---

## 4. Modèles de Classification

### 4.1 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    penalty='l1',
    solver='saga',
    C=1.0,  # inverse de la régularisation
    class_weight='balanced'
)
log_reg.fit(X_train, y_train)

# Probabilités
proba = log_reg.predict_proba(X_test)
```

### 4.2 Support Vector Machines

```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    class_weight='balanced'
)
svm.fit(X_train, y_train)
```

### 4.3 Métriques de Classification en Finance

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

def evaluate_classifier(y_true, y_pred, y_proba=None):
    """Évaluation complète d'un classifieur."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
    }
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
    return metrics
```

**Important**: En finance, l'accuracy n'est pas suffisante!
- Précision importante si coût des faux positifs élevé
- Recall important si coût des faux négatifs élevé

---

## 5. Deep Learning pour la Finance

### 5.1 Feedforward Neural Networks

```python
import torch
import torch.nn as nn

class FactorModel(nn.Module):
    def __init__(self, n_features, hidden_sizes=[64, 32]):
        super().__init__()
        layers = []
        input_size = n_features
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### 5.2 LSTM pour Séries Temporelles

```python
class LSTMPredictor(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)
        # Prendre le dernier timestep
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)
```

### 5.3 Transformer pour Finance

```python
class TransformerPredictor(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x[-1, :, :]  # Dernier timestep
        return self.fc(x)
```

### 5.4 Training Loop

```python
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred.squeeze(), y_batch).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
```

---

## 6. Factor Investing avec ML

### 6.1 Gu, Kelly, Xiu (2020) Framework

**Objectif**: Prédire les rendements en cross-section avec des caractéristiques d'entreprise.

```python
# Caractéristiques (94 dans le paper original)
characteristics = [
    # Value
    'book_to_market', 'earnings_to_price', 'cash_flow_to_price',
    # Momentum
    'momentum_12m', 'momentum_6m', 'momentum_1m',
    # Size
    'log_market_cap', 'log_book_equity',
    # Profitability
    'roe', 'roa', 'gross_profit_margin',
    # Investment
    'asset_growth', 'investment_to_assets',
    # Trading
    'turnover', 'volatility', 'beta',
    # ...
]
```

**Architecture recommandée:**
- Neural network avec 3-5 couches
- Batch normalization
- Dropout
- Ensemble de plusieurs modèles

### 6.2 Construction de Portefeuille

```python
def construct_portfolio(predictions, returns, n_long=50, n_short=50):
    """
    Construit un portefeuille long-short basé sur les prédictions.
    """
    # Rank stocks by predicted return
    ranks = predictions.rank(pct=True)
    
    # Long top stocks, short bottom stocks
    weights = np.zeros(len(predictions))
    
    # Top n_long stocks
    long_mask = ranks >= (1 - n_long/len(predictions))
    weights[long_mask] = 1 / n_long
    
    # Bottom n_short stocks
    short_mask = ranks <= n_short/len(predictions)
    weights[short_mask] = -1 / n_short
    
    # Portfolio return
    portfolio_return = (weights * returns).sum()
    
    return weights, portfolio_return
```

---

## 7. Backtesting

### 7.1 Walk-Forward Optimization

```python
def walk_forward_backtest(data, model_class, window_train=252, window_test=21):
    """
    Backtesting avec walk-forward optimization.
    """
    results = []
    
    for i in range(window_train, len(data) - window_test, window_test):
        # Training window
        train_data = data[i-window_train:i]
        # Test window
        test_data = data[i:i+window_test]
        
        # Train model
        model = model_class()
        model.fit(train_data['X'], train_data['y'])
        
        # Predict
        predictions = model.predict(test_data['X'])
        
        # Evaluate
        actual_returns = test_data['returns']
        results.append({
            'date': test_data['date'].iloc[0],
            'predictions': predictions,
            'returns': actual_returns
        })
    
    return pd.DataFrame(results)
```

### 7.2 Métriques de Performance

```python
def calculate_performance_metrics(returns):
    """
    Calcule les métriques de performance d'une stratégie.
    """
    # Annualized return
    ann_return = returns.mean() * 252
    
    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = ann_return / ann_vol
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = ann_return / downside_vol if downside_vol > 0 else np.nan
    
    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    return {
        'Annual Return': ann_return,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar
    }
```

---

## 8. Techniques Avancées

### 8.1 Ensemble Methods

```python
from sklearn.ensemble import VotingRegressor, StackingRegressor

# Voting ensemble
ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', xgb.XGBRegressor(n_estimators=100)),
    ('lgb', lgb.LGBMRegressor(n_estimators=100))
])

# Stacking
stack = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('xgb', xgb.XGBRegressor(n_estimators=100))
    ],
    final_estimator=Ridge(alpha=1.0)
)
```

### 8.2 Hyperparameter Tuning avec Optuna

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # Time series cross-validation
    scores = []
    for train_idx, val_idx in TimeSeriesSplit(5).split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        scores.append(np.corrcoef(pred, y[val_idx])[0, 1])
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 8.3 Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression
)
from boruta import BorutaPy

# Mutual Information
selector = SelectKBest(mutual_info_regression, k=50)
X_selected = selector.fit_transform(X, y)

# Boruta (wrapper method)
rf = RandomForestRegressor(n_jobs=-1)
boruta = BorutaPy(rf, n_estimators='auto', random_state=42)
boruta.fit(X.values, y.values)
selected_features = X.columns[boruta.support_]
```

---

## 9. Bonnes Pratiques

### 9.1 Éviter l'Overfitting

1. **Régularisation** (L1, L2, dropout)
2. **Cross-validation temporelle** (pas de shuffle!)
3. **Out-of-sample testing** sur période séparée
4. **Simplicité** (Occam's razor)
5. **Ensemble methods**

### 9.2 Gérer le Signal Faible

1. **Feature engineering** robuste
2. **Combinaison de modèles**
3. **Attentes réalistes** (Sharpe ~1-2 est excellent)
4. **Long historique** si possible

### 9.3 Checklist Avant Mise en Production

- [ ] Données point-in-time correctes?
- [ ] Pas de look-ahead bias?
- [ ] Survivorship bias corrigé?
- [ ] Performance robuste across time periods?
- [ ] Coûts de transaction inclus?
- [ ] Slippage estimé?
- [ ] Capacity constraints considérés?

---

## Références

1. López de Prado, M. (2018). "Advances in Financial Machine Learning", Wiley.
2. Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning", Review of Financial Studies.
3. Dixon, M., Halperin, I., & Bilokon, P. (2020). "Machine Learning in Finance", Springer.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning", Springer.
5. Chollet, F. (2021). "Deep Learning with Python", Manning.

---

*Document synthétisé pour la base de connaissances HelixOne. Machine Learning appliqué à la finance quantitative.*
