# 🔑 Guide Complet - Obtenir les Clés API

**Temps total estimé** : 15-20 minutes
**Résultat** : **12/13 sources fonctionnelles** (92%)

---

## 📋 Récapitulatif des Clés

| # | Source | Status | Priorité | Temps |
|---|--------|--------|----------|-------|
| 1 | **Finnhub** | ⚠️ Invalide | 🔴 Haute | 5 min |
| 2 | **NewsAPI** | ❌ Manquante | 🟡 Moyenne | 2 min |
| 3 | **Quandl** | ❌ Manquante | 🟢 Basse (optionnel) | 2 min |

---

## 1️⃣ Finnhub (Priorité HAUTE - 5 minutes)

### Pourquoi ?
- Source déjà intégrée mais clé invalide
- **60 requêtes/minute** gratuit
- Données: stocks, forex, crypto, news

### Étapes Détaillées

#### Étape 1 : Se connecter à Finnhub
```
1. Ouvrir: https://finnhub.io/dashboard
2. Se connecter avec votre compte
   (Si pas de compte: https://finnhub.io/register)
```

#### Étape 2 : Obtenir la Clé API
```
1. Dans le Dashboard, section "API Key"
2. Copier la clé affichée (format: xxxxxxxxxxxxxxxx)

   OU

3. Si expirée, cliquer "Regenerate API Key"
4. Copier la nouvelle clé
```

#### Étape 3 : Configurer dans HelixOne
```bash
# Ouvrir le fichier .env
nano helixone-backend/.env

# Ou avec VSCode
code helixone-backend/.env

# Remplacer la ligne:
FINNHUB_API_KEY=d3mob9hr01qmso34p190d3mob9hr01qmso34p19g

# Par:
FINNHUB_API_KEY=votre_nouvelle_clé_ici

# Sauvegarder (Ctrl+O, Enter, Ctrl+X pour nano)
```

#### Étape 4 : Tester
```bash
./venv/bin/python helixone-backend/test_all_sources.py
```

### Résultat Attendu
```
10. Finnhub... ✅ OK (AAPL=$XXX.XX)
```

---

## 2️⃣ NewsAPI.org (Priorité MOYENNE - 2 minutes)

### Pourquoi ?
- **80,000+ sources** de news mondiales
- **100 requêtes/jour** gratuit
- News filtrées par: pays, langue, catégorie, source

### Étapes Détaillées

#### Étape 1 : S'inscrire (1 minute)
```
1. Ouvrir: https://newsapi.org/register

2. Remplir le formulaire:
   - Email: votre_email@example.com
   - Password: choisir mot de passe
   - First name: Votre prénom
   - Country: Votre pays

3. Cliquer "Submit"

4. Vérifier votre email et confirmer
```

#### Étape 2 : Obtenir la Clé (30 secondes)
```
1. Après confirmation, vous serez redirigé vers le Dashboard
2. Votre clé API est affichée directement en haut:

   Your API key is: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

3. Copier cette clé
```

#### Étape 3 : Configurer dans HelixOne
```bash
# Ouvrir .env
nano helixone-backend/.env

# Ajouter cette ligne à la fin:
NEWSAPI_API_KEY=votre_clé_newsapi_ici

# Sauvegarder
```

#### Étape 4 : Tester
```bash
./venv/bin/python helixone-backend/test_newsapi.py
```

### Résultat Attendu
```
📚 Test 1: Sources disponibles (business)
----------------------------------------------------------------------

✅ 67 sources business trouvées

ID                   Nom                            Pays
----------------------------------------------------------------------
abc-news             ABC News                       US
associated-press     Associated Press               US
bloomberg            Bloomberg                      US
business-insider     Business Insider               US
cnbc                 CNBC                           US
financial-times      Financial Times                GB
reuters              Reuters                        US
the-wall-street-journal The Wall Street Journal    US
```

---

## 3️⃣ Quandl/Nasdaq Data Link (OPTIONNEL - 2 minutes)

### Pourquoi Optionnel ?
- ✅ **Alpha Vantage Commodities déjà fonctionnel**
- Quandl offre les mêmes données commodités
- Utile seulement pour redondance

### Si Vous Voulez Quand Même

#### Étape 1 : S'inscrire (1 minute)
```
1. Ouvrir: https://data.nasdaq.com/sign-up

2. Remplir:
   - Email
   - Password
   - First/Last name

3. Cliquer "Create Free Account"

4. Vérifier email et confirmer
```

#### Étape 2 : Obtenir la Clé (30 secondes)
```
1. Aller sur: https://data.nasdaq.com/account/profile
2. Section "API KEY"
3. Copier la clé affichée
```

#### Étape 3 : Configurer
```bash
# Ouvrir .env
nano helixone-backend/.env

# Ajouter:
QUANDL_API_KEY=votre_clé_quandl_ici

# Sauvegarder
```

#### Étape 4 : Tester
```bash
./venv/bin/python helixone-backend/test_quandl.py
```

---

## ⚡ Configuration Rapide - Tout en Une Fois

### Script de Configuration

Voici toutes les clés à ajouter/modifier dans `.env`:

```bash
# 1. Ouvrir .env
nano helixone-backend/.env

# 2. Modifier/ajouter ces lignes:

# Finnhub (REMPLACER la clé existante)
FINNHUB_API_KEY=votre_nouvelle_clé_finnhub

# NewsAPI (AJOUTER à la fin si pas déjà là)
NEWSAPI_API_KEY=votre_clé_newsapi

# Quandl (AJOUTER - optionnel)
QUANDL_API_KEY=votre_clé_quandl

# 3. Sauvegarder (Ctrl+O, Enter, Ctrl+X)
```

### Exemple de .env Complet

```bash
# Clés API déjà configurées
ALPHA_VANTAGE_API_KEY=PEHB0Q9ZHXMWFM0X
FRED_API_KEY=2eb1601f70b8771864fd98d891879301
FMP_API_KEY=kPPYlq9KldwfsuQJ1RIWXpuLsPKSnwvN
TWELVEDATA_API_KEY=9f2f7efc5a1b400bba397a8c9356b172
IEX_CLOUD_API_KEY=e09023906db18cbf26c4dc22879c5f79fa4cb6d0

# Clés à renouveler/ajouter
FINNHUB_API_KEY=votre_nouvelle_clé_finnhub      # ⚠️ REMPLACER
NEWSAPI_API_KEY=votre_clé_newsapi               # ➕ AJOUTER
QUANDL_API_KEY=votre_clé_quandl                 # ➕ AJOUTER (optionnel)
```

---

## ✅ Vérification Finale

### Test Complet de Toutes les Sources

```bash
# Lancer le test global
./venv/bin/python helixone-backend/test_all_sources.py
```

### Résultat Attendu (Après Toutes les Clés)

```
================================================================================
📊 RÉSUMÉ
================================================================================

✅ Fonctionnelles:       12/19  (63%)
❌ En erreur:           0/19   (0%)
⏳ Config requise:      0/19   (0%)
⚠️  Cassées (migration): 2/19   (11%)
⏭️  Skipped (lent):      5/19   (26%)

📊 Taux de succès: 12/12 = 100% 🎉
```

### Sources Opérationnelles (12)

```
✅ CoinGecko                 BTC=$XXX,XXX
✅ NewsAPI                   XX sources          ← NOUVEAU
✅ Quandl                    Gold=$X,XXX/oz      ← NOUVEAU (optionnel)
✅ Alpha Vantage +           AAPL=$XXX.XX
✅ Fear & Greed              XX/100
✅ Carbon Intensity          XXX gCO2/kWh
✅ USAspending.gov           Contrats OK
✅ FRED                      GDP=$XX,XXXT
✅ SEC Edgar                 10,142 companies
✅ Finnhub                   AAPL=$XXX.XX        ← RENOUVELÉ
✅ FMP                       AAPL=$XXX.XX
✅ Twelve Data               AAPL=$XXX.XX
```

---

## 🎯 Feuille de Route

### Scénario 1 : Minimum Viable (5 minutes)
```
☑️ Finnhub seulement
Résultat: 10/13 sources = 77%
```

### Scénario 2 : Recommandé (7 minutes)
```
☑️ Finnhub
☑️ NewsAPI
Résultat: 11/13 sources = 85%
```

### Scénario 3 : Maximum (10 minutes)
```
☑️ Finnhub
☑️ NewsAPI
☑️ Quandl
Résultat: 12/13 sources = 92%
```

---

## 🆘 Dépannage

### Problème 1 : Clé API ne fonctionne pas

```bash
# Vérifier que la clé est bien dans .env
cat helixone-backend/.env | grep FINNHUB

# Résultat attendu:
FINNHUB_API_KEY=votre_clé_sans_espaces

# Pas d'espaces avant/après le =
# Pas de guillemets
```

### Problème 2 : Test échoue après ajout clé

```bash
# Redémarrer le terminal ou recharger .env
source helixone-backend/.env

# Ou relancer Python fresh
./venv/bin/python helixone-backend/test_all_sources.py
```

### Problème 3 : Clé NewsAPI invalide

```
Erreur: "apiKey parameter is missing"

Solution:
1. Vérifier l'orthographe: NEWSAPI_API_KEY (pas NEWS_API_KEY)
2. Vérifier que la clé est bien copiée (32 caractères)
3. Pas d'espaces dans la clé
```

---

## 📞 Support

### Liens Utiles

- **Finnhub Dashboard**: https://finnhub.io/dashboard
- **NewsAPI Dashboard**: https://newsapi.org/account
- **Quandl Dashboard**: https://data.nasdaq.com/account/profile

### Documentation

- [RAPPORT_CORRECTIONS.md](RAPPORT_CORRECTIONS.md) - Détails corrections
- [STATUS_SOURCES_FINAL.md](STATUS_SOURCES_FINAL.md) - Status toutes sources
- [RESUME_TESTS.md](RESUME_TESTS.md) - Résumé tests

---

## 🎉 Félicitations !

Une fois les clés configurées, vous aurez :

✅ **12 sources de données** de niveau institutionnel
✅ **92% de couverture** globale
✅ **100% gratuit** - toutes les sources
✅ **Données en temps réel** : crypto, stocks, commodités, news, ESG

**HelixOne est prêt pour le trading éducatif !** 🚀

---

*Guide créé le 2025-10-22*
*Temps total: 10-20 minutes*
*Résultat: 12/13 sources fonctionnelles*
