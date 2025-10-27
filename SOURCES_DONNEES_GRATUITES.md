# 📊 Sources de Données Financières Gratuites

## 🎯 Sources à Implémenter (Par Priorité)

### 1. **Finnhub** ⭐⭐⭐⭐⭐
- **Limite gratuite**: 60 requêtes/minute
- **Données disponibles**:
  - Prix en temps réel
  - Données historiques
  - Actualités financières
  - Données fondamentales (P/E, EPS, etc.)
  - Sentiment des actualités
  - Insider transactions
- **Inscription**: https://finnhub.io/register
- **Avantages**: Très complète, bonne limite gratuite

### 2. **Alpha Vantage** ⭐⭐⭐⭐
- **Limite gratuite**: 5 requêtes/minute, 500/jour
- **Données disponibles**:
  - Prix en temps réel et historiques
  - Indicateurs techniques (RSI, MACD, etc.)
  - Données fondamentales
  - Données forex et crypto
  - Données sectorielles
- **Inscription**: https://www.alphavantage.co/support/#api-key
- **Avantages**: Indicateurs techniques intégrés

### 3. **Financial Modeling Prep (FMP)** ⭐⭐⭐⭐
- **Limite gratuite**: 250 requêtes/jour
- **Données disponibles**:
  - États financiers complets (bilan, compte de résultat, flux de trésorerie)
  - Ratios financiers détaillés
  - Prix historiques
  - Profils d'entreprises
  - Actualités
  - Calendrier économique
- **Inscription**: https://site.financialmodelingprep.com/developer/docs
- **Avantages**: Excellente pour les fondamentaux

### 4. **Polygon.io** ⭐⭐⭐⭐
- **Limite gratuite**: 5 requêtes/minute
- **Données disponibles**:
  - Prix en temps réel (avec 15 min de délai)
  - Données historiques
  - Agrégations (OHLC)
  - Splits et dividendes
- **Inscription**: https://polygon.io/
- **Avantages**: Données de qualité institutionnelle

### 5. **Twelve Data** ⭐⭐⭐
- **Limite gratuite**: 8 requêtes/minute, 800/jour
- **Données disponibles**:
  - Prix en temps réel et historiques
  - Indicateurs techniques
  - Données forex, crypto, ETF
  - Fondamentaux basiques
- **Inscription**: https://twelvedata.com/
- **Avantages**: Bonne couverture internationale

### 6. **IEX Cloud** ⭐⭐⭐
- **Limite gratuite**: 50,000 messages/mois
- **Données disponibles**:
  - Prix en temps réel
  - Actualités
  - Fondamentaux
  - Données sociales
- **Inscription**: https://iexcloud.io/
- **Avantages**: Très utilisé, fiable

### 7. **Marketstack** ⭐⭐⭐
- **Limite gratuite**: 100 requêtes/mois (limité)
- **Données disponibles**:
  - Prix historiques EOD
  - 50+ exchanges
  - Dividendes et splits
- **Inscription**: https://marketstack.com/
- **Avantages**: Bonne couverture internationale

### 8. **EOD Historical Data** ⭐⭐
- **Limite gratuite**: 20 requêtes/jour (très limité)
- **Données disponibles**:
  - Prix historiques
  - Fondamentaux
  - Calendrier économique
- **Inscription**: https://eodhistoricaldata.com/
- **Avantages**: Données de qualité

### 9. **CoinGecko** (Pour Crypto) ⭐⭐⭐⭐
- **Limite gratuite**: 50 requêtes/minute
- **Données disponibles**:
  - Prix crypto en temps réel
  - Données historiques crypto
  - Données DeFi
- **Inscription**: Pas besoin de clé API
- **Avantages**: Meilleure API crypto gratuite

### 10. **Federal Reserve Economic Data (FRED)** ⭐⭐⭐⭐⭐
- **Limite gratuite**: Illimitée!
- **Données disponibles**:
  - Indicateurs macro-économiques (GDP, inflation, chômage)
  - Taux d'intérêt
  - Données monétaires
  - Plus de 800,000 séries temporelles
- **Inscription**: https://fred.stlouisfed.org/docs/api/api_key.html
- **Avantages**: Gratuit et illimité, données officielles

## 📋 Sources Déjà Implémentées

- ✅ **Yahoo Finance** (via yfinance) - Illimitée mais rate limiting

## 🎯 Plan d'Implémentation

### Phase 1: Sources Principales (Cette session)
1. Finnhub - Meilleure API gratuite
2. Alpha Vantage - Indicateurs techniques
3. Financial Modeling Prep - Fondamentaux détaillés
4. FRED - Données macro-économiques

### Phase 2: Sources Complémentaires
5. Polygon.io
6. Twelve Data
7. IEX Cloud

### Phase 3: Sources Spécialisées
8. CoinGecko (si besoin crypto)
9. Marketstack (couverture internationale)

## 🔑 Gestion des Clés API

Toutes les clés seront stockées dans `.env`:

```env
# Yahoo Finance (pas de clé nécessaire)
YAHOO_FINANCE_ENABLED=true

# Finnhub
FINNHUB_API_KEY=your_key_here
FINNHUB_ENABLED=true

# Alpha Vantage
ALPHA_VANTAGE_API_KEY=your_key_here
ALPHA_VANTAGE_ENABLED=true

# Financial Modeling Prep
FMP_API_KEY=your_key_here
FMP_ENABLED=true

# Polygon.io
POLYGON_API_KEY=your_key_here
POLYGON_ENABLED=true

# Twelve Data
TWELVE_DATA_API_KEY=your_key_here
TWELVE_DATA_ENABLED=true

# IEX Cloud
IEX_CLOUD_API_KEY=your_key_here
IEX_CLOUD_ENABLED=true

# FRED (Federal Reserve)
FRED_API_KEY=your_key_here
FRED_ENABLED=true
```

## 💡 Stratégie d'Agrégation

1. **Priorité par qualité**: Finnhub > FMP > Alpha Vantage > Yahoo Finance
2. **Fallback automatique**: Si une source échoue, essayer la suivante
3. **Cache intelligent**: Mettre en cache les résultats pour éviter de dépasser les limites
4. **Rotation des sources**: Alterner entre les sources pour optimiser les limites
5. **Agrégation de données**: Combiner les données de plusieurs sources pour avoir le maximum d'informations

## 📊 Estimation des Capacités

Avec toutes ces sources combinées:
- **~150 requêtes/minute** au total
- **~2000 requêtes/jour**
- **Couverture**: Actions US + International + Crypto + Macro
- **Qualité**: Données redondantes = plus fiable
