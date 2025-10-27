# 🔑 Configuration des Clés API - HelixOne

Ce guide vous explique comment obtenir vos clés API gratuites pour collecter des données financières de qualité institutionnelle.

---

## 📊 Sources de Données Implémentées

### 1. Alpha Vantage (GRATUIT)
**Données**: Marché, Fondamentaux, Indicateurs techniques
**Limite**: 500 requêtes/jour (gratuit)
**Qualité**: ⭐⭐⭐⭐

### 2. FRED (GRATUIT)
**Données**: Macro USA (PIB, inflation, emploi, taux)
**Limite**: ILLIMITÉ
**Qualité**: ⭐⭐⭐⭐⭐ (Federal Reserve)

---

## 🚀 Obtenir vos Clés API (2 minutes)

### Alpha Vantage

1. Aller sur: https://www.alphavantage.co/support/#api-key
2. Entrer votre email
3. Cliquer sur "GET FREE API KEY"
4. Copier la clé (format: `XXXXXXXXX`)

**Avantages**:
- ✅ Gratuit à vie
- ✅ 500 requêtes/jour
- ✅ Données historiques illimitées
- ✅ Pas de carte bancaire requise

### FRED (Federal Reserve)

1. Aller sur: https://fred.stlouisfed.org/
2. Créer un compte (gratuit)
3. Aller sur: https://fred.stlouisfed.org/docs/api/api_key.html
4. Cliquer sur "Request API Key"
5. Remplir le formulaire simple
6. Copier la clé (format: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

**Avantages**:
- ✅ Gratuit à vie
- ✅ ILLIMITÉ (pas de limite de requêtes)
- ✅ 500,000+ séries économiques
- ✅ Qualité institutionnelle (Fed)

---

## ⚙️ Configuration dans HelixOne

### Option 1: Variables d'environnement (Recommandé)

Créer un fichier `.env` à la racine du projet:

```bash
# Dans /Users/macintosh/Desktop/helixone/.env

# Alpha Vantage (marché + fondamentaux)
ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY_HERE

# FRED (données macro USA)
FRED_API_KEY=YOUR_FRED_KEY_HERE
```

### Option 2: Configuration directe

Éditer les fichiers de service:

**Alpha Vantage** (`helixone-backend/app/services/alpha_vantage_collector.py`):
```python
ALPHA_VANTAGE_API_KEY = "YOUR_KEY_HERE"
```

**FRED** (`helixone-backend/app/services/fred_collector.py`):
```python
FRED_API_KEY = "YOUR_KEY_HERE"
```

---

## ✅ Vérifier la Configuration

### Test Alpha Vantage

```python
from app.services.alpha_vantage_collector import get_alpha_vantage_collector

# Initialiser
av = get_alpha_vantage_collector()

# Test: récupérer la quote temps réel
quote = av.get_quote("AAPL")
print(f"AAPL: ${quote['price']:.2f}")

# Test: récupérer company overview
overview = av.get_company_overview("AAPL")
print(f"{overview['name']} - {overview['sector']}")
```

### Test FRED

```python
from app.services.fred_collector import get_fred_collector

# Initialiser
fred = get_fred_collector()

# Test: récupérer taux Fed Funds
fed_funds = fred.get_series('DFF')
print(f"Fed Funds Rate: {fed_funds.iloc[-1]:.2f}%")

# Test: récupérer inflation (CPI)
cpi = fred.get_series('CPIAUCSL')
print(f"CPI: {cpi.iloc[-1]:.2f}")
```

---

## 📊 Données Disponibles

### Alpha Vantage

| Catégorie | Fonction | Exemple |
|-----------|----------|---------|
| **Prix journaliers** | `get_daily_data()` | OHLCV 20+ ans |
| **Prix intraday** | `get_intraday_data()` | 1min, 5min, 15min, 30min, 60min |
| **Quote temps réel** | `get_quote()` | Prix actuel + volume |
| **Company overview** | `get_company_overview()` | Secteur, industrie, market cap, PE, beta |
| **Income statement** | `get_income_statement()` | Compte de résultat |
| **Balance sheet** | `get_balance_sheet()` | Bilan |
| **Cash flow** | `get_cash_flow()` | Flux de trésorerie |
| **RSI** | `get_rsi()` | Relative Strength Index |
| **MACD** | `get_macd()` | MACD, signal, histogram |
| **Bollinger Bands** | `get_bbands()` | Upper, middle, lower bands |

### FRED

| Catégorie | Indicateurs | Exemples |
|-----------|-------------|----------|
| **Taux d'intérêt** | Fed Funds, Treasury yields | DFF, DGS10, DGS2 |
| **Inflation** | CPI, PCE, PPI | CPIAUCSL, PCE, PPIACO |
| **PIB** | GDP nominal, real, growth | GDP, GDPC1, A191RL1Q225SBEA |
| **Emploi** | Unemployment, payrolls | UNRATE, PAYEMS, ICSA |
| **Immobilier** | Housing starts, sales | HOUST, HSN1F, CSUSHPISA |
| **Consommation** | Retail sales, sentiment | RSXFS, UMCSENT |
| **Production** | Industrial production | INDPRO, TCU |
| **Monnaie** | M1, M2, crédit | M1SL, M2SL, TOTLL |
| **Indices** | S&P 500, NASDAQ, VIX | SP500, NASDAQCOM, VIXCLS |
| **Dette** | Federal debt, debt/GDP | GFDEBTN, GFDEGDQ188S |

---

## 🎯 Exemples d'Utilisation

### Collecter des données pour le moteur de scénarios

```python
from app.services.alpha_vantage_collector import get_alpha_vantage_collector
from app.services.fred_collector import get_fred_collector
from datetime import datetime, timedelta

av = get_alpha_vantage_collector()
fred = get_fred_collector()

# 1. Collecter prix historiques pour un portefeuille
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

for symbol in symbols:
    # Prix journaliers (20 ans)
    data, meta = av.get_daily_data(symbol, outputsize='full')
    print(f"✅ {symbol}: {len(data)} jours collectés")

# 2. Collecter données macro pour analyse de crise
start_date = datetime(2007, 1, 1)  # Début crise 2008
end_date = datetime(2009, 12, 31)

# Taux d'intérêt
rates = fred.get_interest_rates(start_date, end_date)
print(f"✅ Taux d'intérêt: {len(rates)} observations")

# Inflation
inflation = fred.get_inflation_data(start_date, end_date)
print(f"✅ Inflation: {len(inflation)} observations")

# Emploi
employment = fred.get_employment_data(start_date, end_date)
print(f"✅ Emploi: {len(employment)} observations")

# 3. Calculer yield spread (indicateur de récession)
spread = fred.calculate_yield_spread()
print(f"📊 Yield Spread 10Y-2Y: {spread.iloc[-1]:.2f}%")

if spread.iloc[-1] < 0:
    print("⚠️ ALERTE: Courbe inversée - Risque de récession!")
```

### Analyse fondamentale d'une action

```python
av = get_alpha_vantage_collector()

# Company overview
overview = av.get_company_overview("AAPL")
print(f"""
{overview['name']}
Secteur: {overview['sector']}
Industrie: {overview['industry']}
Market Cap: ${overview['market_cap']:,.0f}
P/E: {overview['pe_ratio']}
Beta: {overview['beta']}
Dividend Yield: {overview['dividend_yield']}%
""")

# États financiers
income = av.get_income_statement("AAPL")
print(f"Historique revenus: {len(income)} années")

balance = av.get_balance_sheet("AAPL")
print(f"Historique bilan: {len(balance)} années")

cashflow = av.get_cash_flow("AAPL")
print(f"Historique cash flow: {len(cashflow)} années")
```

---

## 💡 Bonnes Pratiques

### Rate Limiting

**Alpha Vantage**: Le collecteur gère automatiquement le rate limiting (5 req/min max)
```python
# Pas besoin de gérer manuellement, c'est fait automatiquement
av = get_alpha_vantage_collector()

# Collecte de 10 symboles avec rate limiting auto
for symbol in symbols:
    data = av.get_daily_data(symbol)  # Attente automatique entre requêtes
```

**FRED**: Pas de limite (illimité)
```python
# Collecte illimitée
fred = get_fred_collector()
indicators = fred.get_all_key_indicators()  # Collecte tous les indicateurs
```

### Caching

Les données sont automatiquement stockées en base de données après collecte:
```python
# Première collecte: appel API
data1 = av.get_daily_data("AAPL")  # API call

# Futures lectures: depuis la BDD (pas d'API call)
# À implémenter dans le service data_collector
```

---

## 🎯 Prochaines Étapes

1. ✅ Obtenir vos clés API (2 minutes)
2. ✅ Configurer dans `.env`
3. ⏳ Tester les collecteurs
4. ⏳ Collecter données historiques pour crises
5. ⏳ Entraîner les modèles ML

---

## 📝 Notes Importantes

### Alpha Vantage
- ⚠️ Limite quotidienne: 500 requêtes/jour
- ✅ Réinitialisation: Tous les jours à minuit (UTC)
- 💡 Astuce: Prioritiser la collecte de données historiques (une seule fois), puis maintenir à jour quotidiennement

### FRED
- ✅ Aucune limite
- ✅ Mise à jour automatique des séries
- 💡 Astuce: Collecter toutes les données macro en une fois (rapide et gratuit)

### Sécurité
- ⚠️ Ne jamais commit les clés API dans git
- ✅ Ajouter `.env` dans `.gitignore`
- ✅ Utiliser des variables d'environnement

---

## 🆘 Dépannage

### "Invalid API call"
- Vérifier que la clé API est correcte
- Vérifier que vous n'avez pas dépassé la limite quotidienne (Alpha Vantage)

### "Connection error"
- Vérifier la connexion internet
- Vérifier que l'API n'est pas temporairement indisponible

### "No data returned"
- Certains symboles peuvent ne pas avoir de données fondamentales
- Certaines séries FRED peuvent être discontinues

---

## 📚 Ressources

- **Alpha Vantage Documentation**: https://www.alphavantage.co/documentation/
- **FRED API Documentation**: https://fred.stlouisfed.org/docs/api/fred/
- **FRED Series Search**: https://fred.stlouisfed.org/
- **HelixOne Data Collection Guide**: `DATA_COLLECTION_GUIDE.md`
- **HelixOne Master Plan**: `DATA_SOURCES_MASTER_PLAN.md`

---

**Avec ces 2 sources gratuites, vous avez accès à des données de qualité institutionnelle!** 🚀
