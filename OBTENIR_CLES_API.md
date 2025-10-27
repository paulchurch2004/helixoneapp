# 🔑 Guide: Obtenir les Clés API Gratuites

Ce guide vous explique comment obtenir **gratuitement** toutes les clés API nécessaires pour HelixOne.

## ⚡ Clés Prioritaires (À obtenir en premier)

### 1. **Finnhub** (⭐⭐⭐⭐⭐ PRIORITÉ #1)
**Limite**: 60 requêtes/minute - Excellent!

**Étapes**:
1. Allez sur https://finnhub.io/register
2. Créez un compte (email + mot de passe)
3. Confirmez votre email
4. Allez sur https://finnhub.io/dashboard
5. Copiez votre **API Key**
6. Ajoutez dans `.env`: `FINNHUB_API_KEY=votre_clé_ici`

**Temps**: ~2 minutes

---

### 2. **Alpha Vantage** (⭐⭐⭐⭐ PRIORITÉ #2)
**Limite**: 5 requêtes/minute, 500/jour

**Étapes**:
1. Allez sur https://www.alphavantage.co/support/#api-key
2. Entrez votre email et cliquez "GET FREE API KEY"
3. La clé s'affiche immédiatement (pas de confirmation email)
4. Copiez votre **API Key**
5. Ajoutez dans `.env`: `ALPHA_VANTAGE_API_KEY=votre_clé_ici`

**Temps**: ~1 minute

---

### 3. **Financial Modeling Prep** (⭐⭐⭐⭐ PRIORITÉ #3)
**Limite**: 250 requêtes/jour

**Étapes**:
1. Allez sur https://site.financialmodelingprep.com/developer/docs
2. Cliquez sur "Get your FREE API KEY here"
3. Créez un compte
4. Confirmez votre email
5. Connectez-vous et allez dans votre dashboard
6. Copiez votre **API Key**
7. Ajoutez dans `.env`: `FMP_API_KEY=votre_clé_ici`

**Temps**: ~3 minutes

---

### 4. **FRED** (⭐⭐⭐⭐⭐ GRATUIT ILLIMITÉ!)
**Limite**: AUCUNE! Données macro-économiques officielles

**Étapes**:
1. Allez sur https://fred.stlouisfed.org/
2. Cliquez sur "My Account" puis "Register"
3. Créez un compte
4. Allez sur https://fredaccount.stlouisfed.org/apikeys
5. Cliquez "Request API Key"
6. Remplissez le formulaire simple
7. Copiez votre **API Key**
8. Ajoutez dans `.env`: `FRED_API_KEY=votre_clé_ici`

**Temps**: ~3 minutes

---

## 🎯 Clés Complémentaires (Optionnelles)

### 5. **Polygon.io**
**Limite**: 5 requêtes/minute

1. https://polygon.io/
2. Créez un compte gratuit
3. Dashboard → API Keys
4. `.env`: `POLYGON_API_KEY=votre_clé`

---

### 6. **Twelve Data**
**Limite**: 8 requêtes/minute, 800/jour

1. https://twelvedata.com/register
2. Créez un compte
3. Dashboard → API Keys
4. `.env`: `TWELVEDATA_API_KEY=votre_clé`

---

### 7. **IEX Cloud**
**Limite**: 50,000 messages/mois

1. https://iexcloud.io/cloud-login#/register
2. Créez un compte
3. Console → API Tokens
4. Utilisez le token "Publishable"
5. `.env`: `IEX_CLOUD_API_KEY=votre_clé`

---

## 📋 Checklist Rapide

```bash
# Copiez et configurez votre .env
cd /Users/macintosh/Desktop/helixone/helixone-backend
cp .env.example .env
nano .env  # ou utilisez votre éditeur préféré
```

### Minimum pour démarrer (10 minutes):
- [ ] Finnhub
- [ ] Alpha Vantage
- [ ] Financial Modeling Prep
- [ ] FRED

### Pour aller plus loin (optionnel):
- [ ] Polygon.io
- [ ] Twelve Data
- [ ] IEX Cloud

---

## ✅ Vérifier que tout fonctionne

```bash
# Testez chaque source
cd /Users/macintosh/Desktop/helixone/helixone-backend

# Créez un script de test
cat > test_sources.py << 'EOF'
import os
from app.services.data_sources.finnhub_source import FinnhubSource
from app.services.data_sources.yahoo_finance import YahooFinanceSource
import asyncio

async def test_sources():
    print("🧪 Test des sources de données\n")

    # Yahoo Finance (pas de clé nécessaire)
    print("1. Yahoo Finance...")
    yahoo = YahooFinanceSource()
    quote = await yahoo.get_quote("AAPL")
    print(f"   {'✅' if quote else '❌'} Yahoo Finance")

    # Finnhub
    print("2. Finnhub...")
    finnhub = FinnhubSource()
    if finnhub.is_available():
        quote = await finnhub.get_quote("AAPL")
        print(f"   {'✅' if quote else '❌'} Finnhub")
    else:
        print("   ⚠️  Finnhub - Clé API manquante")

    # TODO: Ajoutez les autres sources ici

asyncio.run(test_sources())
EOF

../venv/bin/python test_sources.py
```

---

## 💡 Conseils

1. **Gardez vos clés secrètes**: Ne les commitez JAMAIS sur Git
2. **Fichier .env**: Assurez-vous qu'il est dans `.gitignore`
3. **Rotation**: Régénérez vos clés si vous pensez qu'elles sont compromises
4. **Limites**: Respectez les limites gratuites pour éviter d'être bloqué

---

## 🚀 Une fois les clés configurées

```bash
# Installez les nouvelles dépendances
cd /Users/macintosh/Desktop/helixone
./venv/bin/pip install -r helixone-backend/requirements.txt

# Relancez le backend
cd helixone-backend
../venv/bin/python -m uvicorn app.main:app --reload

# Testez une analyse
# Elle devrait maintenant utiliser plusieurs sources!
```

---

## ⏱️ Temps Total Estimé

- **Minimum (4 clés)**: ~10 minutes
- **Complet (7+ clés)**: ~20 minutes

**Astuce**: Ouvrez tous les sites dans des onglets différents et inscrivez-vous en parallèle! 🚀
