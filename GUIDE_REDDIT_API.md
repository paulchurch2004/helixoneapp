# 🔑 Guide Configuration Reddit API

## Étape 1: Créer votre Application Reddit (5 minutes)

### 1.1 Connectez-vous à Reddit
- Allez sur: https://reddit.com
- Connectez-vous avec votre compte (ou créez-en un)

### 1.2 Accédez à la page des applications
- **URL directe:** https://www.reddit.com/prefs/apps
- **Ou via menu:** Préférences → Onglet "apps"

### 1.3 Créez une nouvelle app
1. Cliquez sur le bouton **"create another app..."** (en bas de la page)

2. Remplissez le formulaire comme ceci:
   ```
   Name:        HelixOne Data Collector
   App type:    ☑ script (important: sélectionner "script", pas "web app")
   Description: Sentiment analysis for HelixOne trading platform
   About url:   (laissez vide)
   Redirect uri: http://localhost:8080
   ```

3. Cliquez sur **"create app"**

### 1.4 Récupérez vos credentials

Vous verrez maintenant votre app affichée ainsi:

```
┌─────────────────────────────────────┐
│ HelixOne Data Collector             │
│ personal use script                 │
│ Ab12CdE34FgH5I  ← CLIENT_ID (14 car)│
│                                     │
│ secret: xYz789AbC012dEf345GhI678... │
│         ↑ CLIENT_SECRET             │
│                                     │
│ redirect uri: http://localhost:8080 │
└─────────────────────────────────────┘
```

**Copiez ces 2 valeurs:**
- **CLIENT_ID:** Le texte sous "personal use script" (14 caractères)
- **CLIENT_SECRET:** La valeur après "secret:" (environ 27 caractères)

---

## Étape 2: Configurer HelixOne

### 2.1 Ouvrez le fichier .env
```bash
nano helixone-backend/.env
# ou
code helixone-backend/.env  # si vous utilisez VSCode
```

### 2.2 Trouvez la section Reddit API (lignes 32-35):
```bash
# Reddit API (pour sentiment analysis)
REDDIT_CLIENT_ID=votre_client_id_ici
REDDIT_CLIENT_SECRET=votre_client_secret_ici
REDDIT_USER_AGENT=HelixOne:v1.0.0 (by /u/votre_username)
```

### 2.3 Remplacez avec vos valeurs:
```bash
# Reddit API (pour sentiment analysis)
REDDIT_CLIENT_ID=Ab12CdE34FgH5I
REDDIT_CLIENT_SECRET=xYz789AbC012dEf345GhI678jKl901MnO
REDDIT_USER_AGENT=HelixOne:v1.0.0 (by /u/votre_username_reddit)
```

**Notes importantes:**
- Remplacez `votre_username_reddit` par votre vrai username Reddit
- Le USER_AGENT doit suivre ce format exact
- Ne mettez PAS de guillemets autour des valeurs
- Ne partagez JAMAIS ces clés publiquement

### 2.4 Sauvegardez le fichier
```bash
# Ctrl+O puis Enter (nano)
# ou Ctrl+S (VSCode)
```

---

## Étape 3: Tester la Configuration

### 3.1 Testez Reddit API
```bash
./venv/bin/python test_reddit_quick.py
```

**Résultat attendu:**
```
✅ Hot posts r/wallstreetbets...
   Post 1: "TSLA TO THE MOON 🚀" (Score: 15420)
   Post 2: "SPY puts printing" (Score: 8903)
   ...

✅ Ticker mentions (top posts 24h)...
   TSLA: 1,234 mentions
   SPY: 892 mentions
   NVDA: 675 mentions
   ...

✅ Trending tickers...
   #1: TSLA (+156% vs 7d avg)
   #2: GME (+89% vs 7d avg)
   ...
```

Si vous voyez encore **"401 HTTP response"**, vérifiez:
- ✓ CLIENT_ID est bien de 14 caractères
- ✓ CLIENT_SECRET est bien de ~27 caractères
- ✓ Pas de guillemets autour des valeurs
- ✓ USER_AGENT contient bien votre username Reddit

---

## 🎯 Fonctionnalités Débloquées

Une fois configuré, vous aurez accès à:

### 1. Sentiment WallStreetBets
```python
from app.services.reddit_source import RedditSource

reddit = RedditSource()
mentions = reddit.get_ticker_mentions('wallstreetbets', limit=100)
print(mentions)
# {'TSLA': 234, 'SPY': 189, 'NVDA': 156, ...}
```

### 2. Trending Tickers
```python
trending = reddit.get_trending_tickers(['wallstreetbets', 'stocks'], min_change_pct=50)
print(trending)
# [
#   {'ticker': 'TSLA', 'mentions': 234, 'change_pct': 156.7},
#   {'ticker': 'GME', 'mentions': 145, 'change_pct': 89.3}
# ]
```

### 3. Top Posts avec Tickers
```python
posts = reddit.get_top_posts('wallstreetbets', time_filter='day', limit=50)
for post in posts[:5]:
    print(f"{post['title']} - Tickers: {post['tickers']}")
```

### 4. Multi-Subreddit Analysis
```python
analysis = reddit.analyze_multiple_subreddits(
    subreddits=['wallstreetbets', 'stocks', 'investing'],
    time_filter='day'
)
print(f"Total mentions: {analysis['total_mentions']}")
print(f"Top ticker: {analysis['top_tickers'][0]}")
```

---

## 🔒 Sécurité

### ⚠️ IMPORTANT - Ne JAMAIS:
- ❌ Commiter le fichier .env sur Git
- ❌ Partager vos clés API publiquement
- ❌ Screenshot votre CLIENT_SECRET
- ❌ Pusher vos credentials sur GitHub

### ✅ Le .gitignore est déjà configuré
Le fichier `.gitignore` contient déjà:
```
.env
.env.*
```

Vos clés sont protégées! 🔒

---

## 📊 Limites Reddit API (gratuit)

Reddit API gratuit a ces limites:
- **60 requêtes / minute**
- **600 requêtes / 10 minutes**

Notre source respecte automatiquement ces limites avec un rate limiter intégré.

---

## ❓ Troubleshooting

### Erreur: "401 Unauthorized"
**Cause:** Credentials incorrects
**Solution:**
1. Vérifiez CLIENT_ID et CLIENT_SECRET
2. Vérifiez qu'il n'y a pas d'espaces avant/après
3. Vérifiez USER_AGENT format

### Erreur: "429 Too Many Requests"
**Cause:** Rate limit dépassé
**Solution:** Attendez 1 minute, le rate limiter va s'ajuster

### Erreur: "403 Forbidden"
**Cause:** App type incorrect
**Solution:** Recréez l'app en sélectionnant bien "script" (pas "web app")

---

## 🚀 Prochaine Étape

Une fois Reddit configuré, vous pouvez aussi configurer:

### Google Trends (optionnel)
Pas de clé API nécessaire! Mais rate limité par Google.

### Quandl (optionnel)
Pour données commodities historiques:
1. Créez compte sur https://data.nasdaq.com/sign-up
2. Récupérez clé API gratuite
3. Ajoutez dans .env: `QUANDL_API_KEY=votre_clé`

---

## 📝 Résumé Configuration

✅ Compte Reddit créé
✅ App "script" créée sur reddit.com/prefs/apps
✅ CLIENT_ID copié (14 caractères)
✅ CLIENT_SECRET copié (27 caractères)
✅ .env mis à jour avec credentials
✅ Test passé: `./venv/bin/python test_reddit_quick.py`

**Temps estimé:** 5 minutes ⏱️
**Coût:** 0€/mois 💰
**Impact:** Sentiment analysis activé! 🚀

---

## 📞 Support

Si vous rencontrez des problèmes:
1. Vérifiez que l'app type est bien "script"
2. Vérifiez le format du USER_AGENT
3. Testez avec curl:
   ```bash
   curl -A "HelixOne:v1.0.0" \
        -u "CLIENT_ID:CLIENT_SECRET" \
        -X POST https://www.reddit.com/api/v1/access_token \
        -d "grant_type=client_credentials"
   ```

Vous devriez recevoir un access_token si les credentials sont corrects.

---

**Bonne configuration!** 🎉
