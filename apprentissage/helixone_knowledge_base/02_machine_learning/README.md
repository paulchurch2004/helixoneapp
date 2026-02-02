# 📗 MODULE 2: MACHINE LEARNING POUR LA FINANCE
## Techniques ML Adaptées aux Marchés Financiers

---

## 📚 SOURCES PRINCIPALES
- **ENSAE ML for Finance**: https://www.master-statistique-finance.com/ip_paris/courses/Machine%20Learning%20for%20finance_Eng.pdf
- **NYU FRE-7773**: https://engineering.nyu.edu/sites/default/files/2022-09/FRE-7773-SandeepJain-Syllabus_v1.pdf
- **Lopez de Prado - Advances in Financial ML**

---

## 🎯 OBJECTIFS
1. Adapter les techniques ML aux données financières
2. Gérer les spécificités (non-stationnarité, bruit, régimes)
3. Construire des features financières pertinentes
4. Éviter l'overfitting et le data snooping

---

## ⚠️ PIÈGES DU ML EN FINANCE

### 1. Leakage (Fuite d'Information)
```python
# ❌ MAUVAIS - Utilise le futur
df['feature'] = df['returns'].rolling(20).mean()  # Inclut la période actuelle!

# ✅ BON - Utilise seulement le passé
df['feature'] = df['returns'].shift(1).rolling(20).mean()
```

### 2. Look-Ahead Bias
```python
# ❌ MAUVAIS - Normalise avec données futures
X_scaled = (X - X.mean()) / X.std()

# ✅ BON - Normalise récursivement
def expanding_normalize(X):
    mean = X.expanding().mean().shift(1)
    std = X.expanding().std().shift(1)
    return (X - mean) / std
```

### 3. Survivorship Bias
- Inclure les entreprises qui ont fait faillite
- Utiliser point-in-time datasets

### 4. Overfitting
```python
# Cross-validation pour séries temporelles
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

---

## 🔧 FEATURE ENGINEERING FINANCIER

### 1. Features Techniques

```python
import pandas as pd
import numpy as np

def create_technical_features(df):
    """
    Create common technical features
    df must have 'open', 'high', 'low', 'close', 'volume' columns
    """
    features = pd.DataFrame(index=df.index)
    
    # Returns
    features['returns_1d'] = df['close'].pct_change()
    features['returns_5d'] = df['close'].pct_change(5)
    features['returns_20d'] = df['close'].pct_change(20)
    
    # Volatility
    features['volatility_20d'] = features['returns_1d'].rolling(20).std()
    features['volatility_60d'] = features['returns_1d'].rolling(60).std()
    
    # Moving averages
    features['sma_20'] = df['close'].rolling(20).mean() / df['close'] - 1
    features['sma_50'] = df['close'].rolling(50).mean() / df['close'] - 1
    features['sma_200'] = df['close'].rolling(200).mean() / df['close'] - 1
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    features['bb_upper'] = (sma + 2*std) / df['close'] - 1
    features['bb_lower'] = (sma - 2*std) / df['close'] - 1
    features['bb_width'] = (features['bb_upper'] - features['bb_lower'])
    
    # Volume features
    features['volume_sma'] = df['volume'].rolling(20).mean()
    features['volume_ratio'] = df['volume'] / features['volume_sma']
    
    # Range features
    features['high_low_range'] = (df['high'] - df['low']) / df['close']
    features['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    return features.shift(1)  # Shift to avoid look-ahead


def create_microstructure_features(trades_df, quotes_df):
    """
    Features from microstructure data
    """
    features = pd.DataFrame()
    
    # Order flow imbalance
    features['buy_volume'] = trades_df[trades_df['side'] == 'buy']['quantity'].resample('1H').sum()
    features['sell_volume'] = trades_df[trades_df['side'] == 'sell']['quantity'].resample('1H').sum()
    features['ofi'] = (features['buy_volume'] - features['sell_volume']) / \
                      (features['buy_volume'] + features['sell_volume'])
    
    # Spread features
    features['spread'] = quotes_df['ask'] - quotes_df['bid']
    features['spread_mean'] = features['spread'].rolling(20).mean()
    
    # Depth imbalance
    features['depth_imb'] = (quotes_df['bid_size'] - quotes_df['ask_size']) / \
                            (quotes_df['bid_size'] + quotes_df['ask_size'])
    
    return features
```

### 2. Features Fondamentales

```python
def create_fundamental_features(financials_df):
    """
    Fundamental financial ratios
    """
    features = pd.DataFrame(index=financials_df.index)
    
    # Valuation
    features['pe_ratio'] = financials_df['price'] / financials_df['eps']
    features['pb_ratio'] = financials_df['price'] / financials_df['book_value_per_share']
    features['ev_ebitda'] = (financials_df['market_cap'] + financials_df['debt'] - \
                            financials_df['cash']) / financials_df['ebitda']
    
    # Profitability
    features['roe'] = financials_df['net_income'] / financials_df['equity']
    features['roa'] = financials_df['net_income'] / financials_df['assets']
    features['gross_margin'] = financials_df['gross_profit'] / financials_df['revenue']
    features['operating_margin'] = financials_df['operating_income'] / financials_df['revenue']
    
    # Growth
    features['revenue_growth'] = financials_df['revenue'].pct_change(4)  # YoY
    features['earnings_growth'] = financials_df['net_income'].pct_change(4)
    
    # Leverage
    features['debt_to_equity'] = financials_df['debt'] / financials_df['equity']
    features['interest_coverage'] = financials_df['ebit'] / financials_df['interest_expense']
    
    return features
```

---

## 🤖 MODÈLES POUR LA FINANCE

### 1. Régression avec Régularisation

```python
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_regularized_model(alpha=1.0, l1_ratio=0.5):
    """
    ElasticNet for sparse, regularized predictions
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
    ])
    return pipeline


# Cross-validated hyperparameter tuning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def tune_model(X, y):
    """
    Tune model with time series cross-validation
    """
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'model__l1_ratio': [0.1, 0.5, 0.9]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    model = build_regularized_model()
    
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=tscv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_
```

### 2. Gradient Boosting (XGBoost/LightGBM)

```python
import lightgbm as lgb

def build_lgbm_model(X_train, y_train, X_val, y_val):
    """
    LightGBM for tabular financial data
    """
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data_in_leaf': 50,  # Prevent overfitting
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def feature_importance_analysis(model, feature_names):
    """
    Analyze feature importance
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    return importance
```

### 3. Neural Networks pour Time Series

```python
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """
    LSTM for financial time series prediction
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


class TransformerPredictor(nn.Module):
    """
    Transformer for financial sequence prediction
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        # Use CLS token or mean pooling
        x = x.mean(dim=1)  # Mean pooling
        return self.fc(x)


def train_neural_network(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """
    Training loop with early stopping
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss += criterion(pred.squeeze(), y_batch).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

---

## 📊 ÉVALUATION DES MODÈLES

### Métriques Financières

```python
import numpy as np

def evaluate_predictions(y_true, y_pred, returns):
    """
    Comprehensive evaluation for financial predictions
    """
    metrics = {}
    
    # Statistical metrics
    metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    
    # Directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    metrics['directional_accuracy'] = np.mean(direction_true == direction_pred)
    
    # Information Coefficient (IC)
    metrics['ic'] = np.corrcoef(y_true, y_pred)[0, 1]
    metrics['ic_ranked'] = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
    
    # Trading metrics (if using predictions as signals)
    signal = np.sign(y_pred)
    strategy_returns = signal[:-1] * returns[1:]  # Trade next day
    
    metrics['sharpe'] = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    metrics['max_drawdown'] = compute_max_drawdown(strategy_returns)
    metrics['hit_rate'] = np.mean(strategy_returns > 0)
    
    return metrics


def compute_max_drawdown(returns):
    """
    Compute maximum drawdown from returns series
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def walk_forward_validation(model_fn, X, y, train_size, test_size, step_size):
    """
    Walk-forward validation for time series
    """
    results = []
    
    for start in range(0, len(X) - train_size - test_size, step_size):
        train_end = start + train_size
        test_end = train_end + test_size
        
        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        
        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_predictions(y_test, y_pred, y_test)
        metrics['period_start'] = train_end
        metrics['period_end'] = test_end
        results.append(metrics)
    
    return pd.DataFrame(results)
```

---

## 🔗 RÉFÉRENCES
1. López de Prado, M. - Advances in Financial Machine Learning
2. Dixon, M., Halperin, I., Bilokon, P. - Machine Learning in Finance
3. Fabozzi, F.J. et al. - Quantitative Equity Investing
