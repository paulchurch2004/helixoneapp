# 🧪 Guide de Test Finnhub

## Étape 1: Obtenir une clé API Finnhub (2 minutes)

1. **Ouvrez votre navigateur** et allez sur: https://finnhub.io/register

2. **Créez un compte**:
   - Entrez votre email
   - Créez un mot de passe
   - Cliquez sur "Sign Up"

3. **Confirmez votre email**:
   - Vérifiez votre boîte de réception
   - Cliquez sur le lien de confirmation

4. **Récupérez votre clé**:
   - Allez sur https://finnhub.io/dashboard
   - Votre **API Key** s'affiche en haut
   - Copiez-la (elle ressemble à: `abc123xyz...`)

## Étape 2: Configurer la clé (1 minute)

Vous avez 2 options:

### Option A: Variable d'environnement (Temporaire)
```bash
export FINNHUB_API_KEY='votre_clé_ici'
```

### Option B: Fichier .env (Permanent - Recommandé)
```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend

# Éditez le fichier .env
nano .env

# Ajoutez cette ligne:
FINNHUB_API_KEY=votre_clé_ici

# Sauvegardez (Ctrl+X, puis Y, puis Enter)
```

## Étape 3: Lancer le test

```bash
cd /Users/macintosh/Desktop/helixone

# Lancer le script de test
./venv/bin/python test_finnhub.py
```

## ✅ Résultats Attendus

Si tout fonctionne, vous devriez voir:

```
======================================================================
🧪 TEST FINNHUB
======================================================================
✅ Clé API configurée: abc123xyz...
📊 Initialisation de Finnhub...
✅ Finnhub initialisé

----------------------------------------------------------------------
TEST 1: Récupération d'un prix (Quote)
----------------------------------------------------------------------
🔍 Récupération du prix de AAPL...
✅ Quote récupérée!
   Ticker: AAPL
   Nom: Apple Inc
   Prix: $178.50
   Change: 2.15 (1.22%)
   Source: finnhub
   Timestamp: 2025-10-14 12:34:56

----------------------------------------------------------------------
TEST 2: Données historiques
----------------------------------------------------------------------
🔍 Récupération des données de AAPL du 2025-09-14 au 2025-10-14...
✅ Données historiques récupérées!
   Nombre de jours: 30
   Premier jour: 2025-09-14 - $175.20
   Dernier jour: 2025-10-14 - $178.50
   Source: finnhub

----------------------------------------------------------------------
TEST 3: Données fondamentales
----------------------------------------------------------------------
🔍 Récupération des fondamentaux de AAPL...
✅ Fondamentaux récupérés!
   Market Cap: $2,800,000,000,000
   P/E Ratio: 28.5
   EPS: 6.25
   ROE: 147.3%
   Beta: 1.24
   Secteur: Technology
   Source: finnhub

----------------------------------------------------------------------
TEST 4: Actualités avec sentiment
----------------------------------------------------------------------
🔍 Récupération des actualités de AAPL...
✅ 5 articles récupérés!
   Article 1:
   📰 Apple Announces New iPhone
   🔗 https://...
   📅 2025-10-13 15:30:00
   😊 Sentiment: 0.85

======================================================================
📊 RÉSUMÉ DU TEST
======================================================================
✅ Finnhub fonctionne correctement!
✅ Données disponibles: Prix, Historique, Fondamentaux, News
✅ Prêt pour l'intégration dans l'aggregator

💡 Prochaine étape: Ajouter les autres sources (Alpha Vantage, FMP, FRED)
```

## ❌ En Cas d'Erreur

### Erreur: "Clé API Finnhub non configurée"
→ Vérifiez que vous avez bien configuré la clé dans `.env` ou via `export`

### Erreur: "Module finnhub manquant"
```bash
cd /Users/macintosh/Desktop/helixone
./venv/bin/pip install finnhub-python
```

### Erreur: "401 Unauthorized"
→ Votre clé API est invalide. Revérifiez-la sur https://finnhub.io/dashboard

### Erreur: "429 Too Many Requests"
→ Vous avez dépassé la limite de 60 requêtes/minute. Attendez 1 minute.

## 🎯 Une Fois le Test Réussi

1. **Notez votre clé** quelque part en sécurité
2. **Continuez** avec l'implémentation des autres sources
3. Ou **testez l'intégration** dans l'application complète

## 📞 Besoin d'Aide?

Si le test échoue, partagez:
1. Le message d'erreur complet
2. Les 10 premiers caractères de votre clé API
3. La sortie du script

---

**Temps total estimé**: 3-5 minutes ⏱️
