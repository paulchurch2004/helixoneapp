# 🔑 Guide : Obtenir les Clés API IEX Cloud & Twelve Data

**Temps total estimé** : 15-20 minutes
**Coût** : GRATUIT (plans free tier)

---

## 📊 1. IEX Cloud

### Avantages
- ✅ **50,000 requêtes/mois** gratuites
- ✅ Prix en temps réel (15 min delay sur free tier)
- ✅ Données fondamentales
- ✅ Actualités financières
- ✅ Données historiques (5 ans)
- ✅ Coverage : Actions US principalement

### Étapes d'Inscription

#### 1.1 Créer un Compte

1. **Aller sur** : https://iexcloud.io/cloud-login#/register

2. **Remplir le formulaire** :
   - Prénom / Nom
   - Email (utilisez une adresse valide)
   - Mot de passe (min 8 caractères)
   - Cochez "I agree to terms"

3. **Cliquer sur** "Create Account"

4. **Vérifier votre email** :
   - Ouvrez l'email de IEX Cloud
   - Cliquez sur le lien de vérification

#### 1.2 Choisir le Plan Gratuit

1. **Après connexion**, vous serez sur le dashboard

2. **Sélectionner "Start" plan** (Free tier) :
   - 50,000 messages/mois
   - Core data (prix, fondamentaux, news)
   - Délai de 15 minutes pour les prix

3. **Pas de carte bancaire requise** pour le plan gratuit

#### 1.3 Obtenir la Clé API

1. **Dans le menu de gauche**, cliquez sur **"API Tokens"**

2. **Vous verrez deux types de clés** :
   - **Publishable Token** (commence par `pk_`)
   - **Secret Token** (commence par `sk_`)

3. **Copiez le "Publishable Token"** (pk_...)
   - C'est celui qu'on utilisera
   - Le token secret est pour des opérations sensibles

4. **Exemple** :
   ```
   pk_1234567890abcdef1234567890abcdef
   ```

#### 1.4 Tester la Clé

Testez dans votre terminal :

```bash
curl "https://cloud.iexapis.com/stable/stock/AAPL/quote?token=VOTRE_CLE_ICI"
```

Vous devriez voir des données JSON pour Apple.

---

## 📈 2. Twelve Data

### Avantages
- ✅ **800 requêtes/jour** gratuites (8 req/minute)
- ✅ Excellente couverture internationale
- ✅ Forex, Crypto, Indices
- ✅ Données intraday (1min, 5min, 15min, etc.)
- ✅ Indicateurs techniques
- ✅ Coverage : Actions mondiales, Forex, Crypto

### Étapes d'Inscription

#### 2.1 Créer un Compte

1. **Aller sur** : https://twelvedata.com/register

2. **Remplir le formulaire** :
   - Email
   - Mot de passe
   - Nom / Prénom
   - Cochez "I agree to terms"

3. **Cliquer sur** "Sign Up"

4. **Vérifier votre email** :
   - Ouvrez l'email de Twelve Data
   - Cliquez sur le lien de confirmation

#### 2.2 Plan Gratuit (Basic)

Le plan gratuit est automatiquement sélectionné :
- **800 requêtes/jour**
- **8 requêtes/minute**
- Accès à toutes les données de base
- Pas de carte bancaire requise

#### 2.3 Obtenir la Clé API

1. **Après connexion**, vous êtes redirigé vers le dashboard

2. **La clé API est affichée immédiatement** en haut de la page :
   ```
   Your API Key: abc123def456ghi789jkl012mno345pq
   ```

3. **Copiez cette clé**

4. **Alternative** : Allez dans **"API"** → **"API Key"** dans le menu

#### 2.4 Tester la Clé

Testez dans votre terminal :

```bash
curl "https://api.twelvedata.com/time_series?symbol=AAPL&interval=1day&apikey=VOTRE_CLE_ICI"
```

Vous devriez voir des données de séries temporelles pour Apple.

---

## 🔧 3. Configuration dans HelixOne

### 3.1 Modifier le fichier .env

Il y a **deux fichiers .env** à mettre à jour :

#### A. Backend (.env principal)

```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend
nano .env
```

Ajoutez ces lignes :

```bash
# IEX Cloud
IEX_CLOUD_API_KEY=pk_votre_cle_iex_ici

# Twelve Data
TWELVEDATA_API_KEY=votre_cle_twelvedata_ici
```

**Sauvegardez** : `Ctrl+O`, `Enter`, `Ctrl+X`

#### B. Root (si nécessaire)

```bash
cd /Users/macintosh/Desktop/helixone
nano .env
```

Ajoutez les mêmes lignes si ce fichier existe.

### 3.2 Vérifier la Configuration

```bash
cd /Users/macintosh/Desktop/helixone

# Vérifier que les variables sont définies
grep -E "(IEX_CLOUD|TWELVEDATA)" helixone-backend/.env
```

Vous devriez voir :
```
IEX_CLOUD_API_KEY=pk_...
TWELVEDATA_API_KEY=...
```

---

## 🧪 4. Tester les Intégrations

### 4.1 Test IEX Cloud

Créez un script de test rapide :

```bash
cd /Users/macintosh/Desktop/helixone
./venv/bin/python -c "
import os
from dotenv import load_dotenv

# Charger .env
load_dotenv('helixone-backend/.env')

iex_key = os.getenv('IEX_CLOUD_API_KEY')

if iex_key:
    print(f'✅ IEX Cloud Key trouvée: {iex_key[:10]}...')

    # Test API
    import requests
    url = f'https://cloud.iexapis.com/stable/stock/AAPL/quote?token={iex_key}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f'✅ API fonctionne! Prix AAPL: \${data[\"latestPrice\"]}')
    else:
        print(f'❌ Erreur API: {response.status_code}')
else:
    print('❌ Clé IEX Cloud non trouvée dans .env')
"
```

### 4.2 Test Twelve Data

```bash
./venv/bin/python -c "
import os
from dotenv import load_dotenv

load_dotenv('helixone-backend/.env')

twelve_key = os.getenv('TWELVEDATA_API_KEY')

if twelve_key:
    print(f'✅ Twelve Data Key trouvée: {twelve_key[:10]}...')

    import requests
    url = f'https://api.twelvedata.com/time_series?symbol=AAPL&interval=1day&outputsize=1&apikey={twelve_key}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'values' in data:
            print(f'✅ API fonctionne! Dernier prix: \${data[\"values\"][0][\"close\"]}')
        else:
            print(f'⚠️  Réponse reçue mais format inattendu: {data}')
    else:
        print(f'❌ Erreur API: {response.status_code}')
else:
    print('❌ Clé Twelve Data non trouvée dans .env')
"
```

### 4.3 Test avec les Collectors HelixOne

Si vous avez des collectors pour IEX et Twelve Data :

```bash
# Test IEX Cloud collector
./venv/bin/python -c "
from helixone-backend.app.services.iex_cloud_collector import get_iex_cloud_collector

iex = get_iex_cloud_collector()
quote = iex.get_quote('AAPL')
print(f'✅ IEX Cloud: AAPL = \${quote[\"latestPrice\"]}')
"

# Test Twelve Data collector
./venv/bin/python -c "
from helixone-backend.app.services.twelvedata_collector import get_twelvedata_collector

twelve = get_twelvedata_collector()
price = twelve.get_latest_price('AAPL')
print(f'✅ Twelve Data: AAPL = \${price}')
"
```

---

## 📊 5. Limites des Plans Gratuits

### IEX Cloud - Free Tier

| Limite | Valeur |
|--------|--------|
| Requêtes/mois | 50,000 |
| Requêtes/seconde | ~10 (non officiel) |
| Délai données | 15 minutes |
| Historique | 5 ans |
| Actions US | ✅ |
| Actions internationales | ⚠️ Limité |
| Crypto | ✅ |
| Forex | ✅ |

**Calcul** : 50,000 req/mois = ~1,667 req/jour = ~69 req/heure

### Twelve Data - Basic Plan

| Limite | Valeur |
|--------|--------|
| Requêtes/jour | 800 |
| Requêtes/minute | 8 |
| Délai données | Temps réel |
| Historique | Illimité |
| Actions mondiales | ✅ |
| Crypto | ✅ |
| Forex | ✅ |
| Indices | ✅ |

**Calcul** : 800 req/jour = ~33 req/heure = 8 req/minute max

---

## ⚠️ 6. Bonnes Pratiques

### 6.1 Gestion du Rate Limiting

**Pour éviter de dépasser les limites** :

```python
import time

# IEX Cloud - Espacer les requêtes
time.sleep(0.1)  # 100ms entre requêtes

# Twelve Data - 8 req/minute max
time.sleep(7.5)  # 7.5 secondes entre requêtes pour être sûr
```

### 6.2 Cache Local

Implémentez un cache pour éviter les requêtes répétées :

```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=100)
def get_cached_quote(symbol, cache_time):
    """Cache for 5 minutes"""
    return iex.get_quote(symbol)

# Utilisation
cache_key = datetime.now().strftime("%Y%m%d%H%M") // 5
quote = get_cached_quote("AAPL", cache_key)
```

### 6.3 Monitoring des Limites

Gardez une trace de votre utilisation :

```python
# Compteur simple
api_calls = {
    'iex': 0,
    'twelve': 0
}

def track_api_call(source):
    api_calls[source] += 1
    if api_calls['iex'] > 1600:  # ~50k/31 jours
        print("⚠️ Approaching IEX limit!")
    if api_calls['twelve'] > 750:
        print("⚠️ Approaching Twelve Data daily limit!")
```

---

## 🎯 7. Résumé des Clés Obtenues

Après avoir suivi ce guide, vous devriez avoir :

```bash
# helixone-backend/.env

# ===== SOURCES DÉJÀ CONFIGURÉES =====
FINNHUB_API_KEY=votre_clé_finnhub
ALPHA_VANTAGE_API_KEY=votre_clé_alphavantage
FRED_API_KEY=votre_clé_fred
FMP_API_KEY=votre_clé_fmp

# ===== NOUVELLES SOURCES =====
IEX_CLOUD_API_KEY=pk_1234567890abcdef  # ← NOUVEAU
TWELVEDATA_API_KEY=abc123def456ghi789  # ← NOUVEAU
```

---

## ✅ 8. Vérification Finale

Checklist complète :

- [ ] Compte IEX Cloud créé
- [ ] Email IEX Cloud vérifié
- [ ] Clé API IEX Cloud copiée (pk_...)
- [ ] Compte Twelve Data créé
- [ ] Email Twelve Data vérifié
- [ ] Clé API Twelve Data copiée
- [ ] Clés ajoutées au fichier `.env`
- [ ] Tests de connexion réussis
- [ ] Sources apparaissent comme "disponibles" dans HelixOne

---

## 🚀 9. Prochaines Étapes

Avec ces deux nouvelles sources, vous avez maintenant :

**Sources de Marché Actives** :
1. ✅ Alpha Vantage (500/jour)
2. ✅ Finnhub (60/min)
3. ✅ FMP (250/jour)
4. ✅ **IEX Cloud** (50k/mois) 🆕
5. ✅ **Twelve Data** (800/jour) 🆕

**Coverage Totale** :
- 📈 Actions US : **5 sources**
- 🌍 Actions internationales : **3 sources** (Finnhub, Twelve Data, FMP)
- 💱 Forex : **2 sources** (IEX Cloud, Twelve Data)
- 🪙 Crypto : **2 sources** (IEX Cloud, Twelve Data)

**Capacité quotidienne estimée** :
- ~52,400 requêtes/jour sur toutes les sources combinées
- Redondance excellente (si une source tombe, 4 autres disponibles)

---

## 📞 Support

### IEX Cloud
- **Documentation** : https://iexcloud.io/docs/
- **Support** : support@iexcloud.io
- **Status** : https://status.iexcloud.io/

### Twelve Data
- **Documentation** : https://twelvedata.com/docs
- **Support** : support@twelvedata.com
- **FAQ** : https://twelvedata.com/faq

---

**Temps total** : ⏱️ 15-20 minutes
**Coût** : 💰 GRATUIT
**Résultat** : 🎉 +2 sources de données professionnelles !

---

*Guide créé le 2025-10-22*
*HelixOne - Plateforme de Trading Éducative*
