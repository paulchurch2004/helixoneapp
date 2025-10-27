# 🔑 Guide : Obtenir une Clé API Tiingo

**Alternative à IEX Cloud - MEILLEURE !**

**Temps estimé** : 5 minutes
**Coût** : GRATUIT

---

## 🌟 Pourquoi Tiingo au lieu de IEX Cloud ?

### Avantages de Tiingo Free Tier

| Fonctionnalité | Tiingo Free | IEX Cloud Free |
|----------------|-------------|----------------|
| **Requêtes/heure** | 500 | ~1,667/jour (~69/h) |
| **Requêtes/jour** | 12,000 | 1,667 |
| **Requêtes/mois** | ~360,000 | 50,000 |
| **Délai données** | End-of-day gratuit | 15 min delay |
| **Historique** | **30 ans** | 5 ans |
| **Actions US** | ✅ Toutes | ✅ |
| **Forex** | ✅ | ✅ |
| **Crypto** | ✅ | ✅ |
| **News API** | ✅ Excellent | ❌ Payant |
| **Carte bancaire** | ❌ Pas requise | ❌ Pas requise |

**Verdict** : Tiingo offre **7x plus de requêtes** que IEX Cloud ! 🚀

---

## 📝 Étapes d'Inscription

### 1. Créer un Compte

1. **Aller sur** : https://www.tiingo.com/account/api/token

   OU

   https://www.tiingo.com/signup

2. **Remplir le formulaire** :
   - Email
   - Mot de passe (min 8 caractères)
   - Prénom / Nom

3. **Cliquer sur** "Sign Up"

4. **Vérifier votre email** :
   - Ouvrez l'email de Tiingo
   - Cliquez sur le lien de vérification
   - Vous serez automatiquement connecté

---

### 2. Obtenir la Clé API

**C'est ultra simple !**

1. **Après connexion**, vous êtes redirigé automatiquement vers :
   https://www.tiingo.com/account/api/token

2. **Votre clé API est déjà affichée !** 🎉
   ```
   Token: abc123def456ghi789jkl012mno345pqrst678uvw901xyz234
   ```

3. **Copiez cette clé** (bouton "Copy" à droite)

**C'est tout !** Pas besoin de choisir un plan, c'est automatiquement le free tier.

---

### 3. Tester la Clé

Testez dans votre terminal :

```bash
curl "https://api.tiingo.com/tiingo/daily/AAPL/prices?token=VOTRE_CLE_ICI"
```

Vous devriez voir des données historiques pour Apple.

**Exemple de réponse** :
```json
[
  {
    "date": "2025-10-22T00:00:00.000Z",
    "close": 259.0,
    "high": 261.5,
    "low": 257.8,
    "open": 258.3,
    "volume": 45678901,
    "adjClose": 259.0,
    "adjHigh": 261.5,
    "adjLow": 257.8,
    "adjOpen": 258.3,
    "adjVolume": 45678901,
    "divCash": 0.0,
    "splitFactor": 1.0
  }
]
```

---

## 🔧 Configuration dans HelixOne

### Option A : Ligne de Commande (Rapide)

```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend
echo "TIINGO_API_KEY=votre_cle_tiingo_ici" >> .env
```

### Option B : Éditeur Nano

```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend
nano .env
```

Trouvez la ligne :
```bash
TIINGO_API_KEY=
```

Remplacez par :
```bash
TIINGO_API_KEY=abc123def456ghi789jkl012mno345pqrst678uvw901xyz234
```

**Sauvegardez** : `Ctrl+O`, `Enter`, `Ctrl+X`

---

## 🧪 Test d'Intégration

### Test Simple

```bash
cd /Users/macintosh/Desktop/helixone

./venv/bin/python -c "
import os
import requests
from dotenv import load_dotenv

load_dotenv('helixone-backend/.env')

tiingo_key = os.getenv('TIINGO_API_KEY')

if tiingo_key:
    print(f'✅ Tiingo Key trouvée: {tiingo_key[:15]}...')

    # Test End-of-Day data
    url = f'https://api.tiingo.com/tiingo/daily/AAPL/prices'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {tiingo_key}'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            latest = data[-1]
            print(f'✅ API fonctionne!')
            print(f'   Date: {latest[\"date\"][:10]}')
            print(f'   Prix: \${latest[\"close\"]}')
            print(f'   Volume: {latest[\"volume\"]:,}')
        else:
            print('⚠️  Aucune donnée reçue')
    else:
        print(f'❌ Erreur API: {response.status_code}')
        print(f'   Message: {response.text[:200]}')
else:
    print('❌ Clé Tiingo non trouvée dans .env')
"
```

### Test Intraday (si disponible)

```bash
./venv/bin/python -c "
import os
import requests
from dotenv import load_dotenv

load_dotenv('helixone-backend/.env')

tiingo_key = os.getenv('TIINGO_API_KEY')

# Test IEX Intraday data (certaines actions)
url = f'https://api.tiingo.com/iex/AAPL/prices'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Token {tiingo_key}'
}

params = {
    'resampleFreq': '1min'
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    print(f'✅ Intraday data disponible: {len(data)} points')
elif response.status_code == 403:
    print('⚠️  Intraday nécessite un upgrade (end-of-day gratuit)')
else:
    print(f'Status: {response.status_code}')
"
```

---

## 📊 Limites du Free Tier

### Détails Officiels

| Limite | Valeur | Commentaire |
|--------|--------|-------------|
| **Requêtes/heure** | 500 | Très généreux |
| **Requêtes/jour** | ~12,000 | 24h x 500 |
| **Requêtes/mois** | ~360,000 | 7x plus que IEX Cloud |
| **Historique** | 30 ans | Excellent |
| **End-of-Day** | ✅ Gratuit | Toutes actions US |
| **Intraday (IEX)** | ⚠️ Limité | ~50 top actions gratuites |
| **News** | ✅ Gratuit | Excellent |
| **Fundamentals** | ❌ Payant | Utiliser FMP/Alpha Vantage |

### Calcul de Capacité

**Free Tier** :
- 500 requêtes/heure = **1 requête toutes les 7.2 secondes**
- 12,000 requêtes/jour = **500 symboles x 24 fois/jour**
- 360,000 requêtes/mois = suffisant pour un portfolio de 500 actions

**Comparaison** :
- Tiingo : 360,000/mois
- IEX Cloud : 50,000/mois
- **Tiingo gagne 7x !** ��

---

## 📖 API Endpoints Disponibles (Free)

### 1. End-of-Day Prices (EOD)

**Le plus utile pour du trading éducatif !**

```bash
GET https://api.tiingo.com/tiingo/daily/{ticker}/prices
```

**Paramètres** :
- `startDate` : Date de début (YYYY-MM-DD)
- `endDate` : Date de fin
- `resampleFreq` : daily, weekly, monthly, annually

**Exemple** :
```bash
curl -H "Authorization: Token YOUR_KEY" \
  "https://api.tiingo.com/tiingo/daily/AAPL/prices?startDate=2024-01-01&endDate=2025-10-22"
```

### 2. Latest Price

```bash
GET https://api.tiingo.com/tiingo/daily/{ticker}
```

**Retourne** : Métadonnées + dernier prix

### 3. News API

**GRATUIT et excellent !**

```bash
GET https://api.tiingo.com/tiingo/news
```

**Paramètres** :
- `tickers` : AAPL,MSFT,GOOGL
- `startDate` / `endDate`
- `limit` : Nombre d'articles (max 100)

**Exemple** :
```bash
curl -H "Authorization: Token YOUR_KEY" \
  "https://api.tiingo.com/tiingo/news?tickers=AAPL&limit=10"
```

### 4. Crypto Prices

```bash
GET https://api.tiingo.com/tiingo/crypto/prices
```

**Tickers** : btcusd, ethusd, etc.

### 5. Forex

```bash
GET https://api.tiingo.com/tiingo/fx/{ticker}/prices
```

**Tickers** : eurusd, gbpusd, usdjpy, etc.

---

## 💡 Bonnes Pratiques

### 1. Headers Recommandés

Toujours utiliser les headers :

```python
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Token {tiingo_api_key}'
}
```

### 2. Rate Limiting

```python
import time

# Respecter 500 req/heure = 1 req/7.2s
time.sleep(8)  # 8 secondes pour être sûr
```

### 3. Cache Local

Tiingo recommande de cacher les données end-of-day :

```python
from functools import lru_cache
from datetime import date

@lru_cache(maxsize=500)
def get_eod_price(ticker, date_str):
    """Cache EOD prices (ne changent pas)"""
    return tiingo.get_prices(ticker, date_str)
```

### 4. Bulk Requests

Au lieu de 100 requêtes individuelles, utilisez :

```python
# Plusieurs tickers en une requête
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
for ticker in tickers:
    data = get_eod_price(ticker, today)
    time.sleep(8)  # Rate limiting
```

---

## 🎯 Comparaison avec Autres Sources

| Source | Requêtes/mois | Historique | News | Points Forts |
|--------|---------------|------------|------|--------------|
| **Tiingo** | **360,000** | **30 ans** | ✅ | **Best free tier** |
| Twelve Data | 24,000 | Illimité | ❌ | Intraday gratuit |
| Alpha Vantage | 15,000 | 20+ ans | ❌ | Fondamentaux |
| Finnhub | 172,800 | Limité | ✅ | ESG scores |
| FMP | 7,500 | 10+ ans | ✅ | Ratios financiers |
| IEX Cloud | 50,000 | 5 ans | ❌ | Intraday (payant) |

**Tiingo est le meilleur free tier pour données historiques et news !** 🏆

---

## 🔄 Collector HelixOne

Si vous avez un collector Tiingo, vérifiez qu'il est configuré :

```python
# helixone-backend/app/services/tiingo_source.py

import os
import requests
from typing import Dict, List, Optional
from datetime import datetime

class TiingoSource:
    """Tiingo Data Source - Free tier: 500 req/hour"""

    def __init__(self):
        self.api_key = os.getenv('TIINGO_API_KEY')
        self.base_url = "https://api.tiingo.com"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }

    def get_eod_prices(self, ticker: str, start_date: str = None, end_date: str = None):
        """Get end-of-day prices"""
        url = f"{self.base_url}/tiingo/daily/{ticker}/prices"

        params = {}
        if start_date:
            params['startDate'] = start_date
        if end_date:
            params['endDate'] = end_date

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        return response.json()

    def get_latest_price(self, ticker: str):
        """Get latest price"""
        url = f"{self.base_url}/tiingo/daily/{ticker}"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_news(self, tickers: List[str], limit: int = 10):
        """Get news for tickers"""
        url = f"{self.base_url}/tiingo/news"

        params = {
            'tickers': ','.join(tickers),
            'limit': limit
        }

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        return response.json()
```

---

## ✅ Checklist Finale

- [ ] Compte Tiingo créé
- [ ] Email vérifié
- [ ] Clé API copiée
- [ ] Clé ajoutée à `helixone-backend/.env`
- [ ] Test end-of-day réussi
- [ ] Test news API réussi
- [ ] (Optionnel) Collector Tiingo configuré

---

## 📞 Support

### Tiingo
- **Documentation** : https://api.tiingo.com/documentation/general/overview
- **Pricing** : https://api.tiingo.com/about/pricing
- **Support** : support@tiingo.com
- **Discord** : https://discord.gg/tiingo

---

## 🎉 Résumé

**Avec Tiingo, vous obtenez** :
- ✅ **7x plus de requêtes** que IEX Cloud (360k vs 50k/mois)
- ✅ **30 ans d'historique** (vs 5 ans)
- ✅ **News API gratuit** (payant chez IEX)
- ✅ **Données end-of-day** pour toutes actions US
- ✅ **Setup en 5 minutes**
- ✅ **GRATUIT à vie**

**Tiingo est objectivement supérieur à IEX Cloud pour le free tier !** 🚀

---

*Guide créé le 2025-10-22*
*Alternative recommandée à IEX Cloud*
*HelixOne - Plateforme de Trading Éducative*
