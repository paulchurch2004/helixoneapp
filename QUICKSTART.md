# ⚡ Démarrage Rapide - HelixOne

## 🎯 Installation en 30 secondes

```bash
# 1. Cloner le projet
git clone https://github.com/votre-repo/helixone.git
cd helixone

# 2. Installer
./setup.sh

# 3. Lancer
./start.sh
```

**C'est tout !** 🎉

## 📱 Première utilisation

1. **L'interface s'ouvre** → Écran d'accueil
2. **Cliquez sur "S'inscrire"** → Créez votre compte
3. **Connectez-vous** avec vos identifiants
4. **Ajoutez une action** (ex: AAPL) dans votre watchlist
5. **Lancez une analyse ML** → Attendez 10-30 secondes
6. **Consultez les prédictions** et recommandations

## 🔑 Clés API (optionnel)

Pour débloquer toutes les fonctionnalités, ajoutez vos clés API gratuites :

### 1. Finnhub (recommandé)
```bash
# Inscription: https://finnhub.io/register
# Gratuit: 60 requêtes/minute
# Données: Prix en temps réel, news, fondamentaux
```

### 2. FRED (Federal Reserve)
```bash
# Inscription: https://fred.stlouisfed.org/docs/api/api_key.html
# Gratuit: ILLIMITÉ
# Données: Taux d'intérêt, inflation, indicateurs économiques
```

### 3. Alpha Vantage
```bash
# Inscription: https://www.alphavantage.co/support/#api-key
# Gratuit: 5 requêtes/minute
# Données: Prix historiques, indicateurs techniques
```

**Ajout des clés :**
```bash
# Éditez le fichier
nano helixone-backend/.env

# Ajoutez vos clés
FINNHUB_API_KEY=votre_cle_ici
FRED_API_KEY=votre_cle_ici
ALPHA_VANTAGE_API_KEY=votre_cle_ici

# Sauvegardez (Ctrl+O puis Ctrl+X)

# Relancez l'application
./start.sh
```

## 🎓 Tutoriels

### Analyser une action
1. Cherchez le ticker (ex: AAPL, TSLA, MSFT)
2. Cliquez sur "Analyser"
3. Choisissez le mode :
   - **Standard** : Analyse rapide (30 sec)
   - **Approfondie** : Analyse complète (1-2 min)
4. Consultez les résultats

### Configurer des alertes
1. Allez dans Portfolio → Alertes
2. Créez une alerte de prix
3. Recevez des notifications en temps réel

### Formation au trading
1. Menu → Formation
2. Suivez les modules interactifs
3. Pratiquez en mode paper trading

## 🆘 Problèmes fréquents

### Le backend ne démarre pas
```bash
# Nettoyer le port 8000
lsof -ti:8000 | xargs kill -9

# Relancer
./start.sh
```

### Erreur "Module not found"
```bash
# Réinstaller les dépendances
./venv/bin/pip install -r helixone-backend/requirements.txt
```

### L'interface ne s'ouvre pas
```bash
# Vérifier que le backend tourne
curl http://127.0.0.1:8000/health

# Si ça ne répond pas, consulter les logs
cat backend.log
```

## 💡 Conseils

- 🚀 **Utilisez `./dev.sh`** pour le mode développement avec logs détaillés
- 💾 **Sauvegardez `helixone.db`** régulièrement (votre base de données)
- 🔐 **Activez le 2FA** dans Paramètres → Sécurité
- 📊 **Laissez tourner la nuit** pour bénéficier des analyses automatiques (7h et 17h)

## 📚 Documentation complète

Consultez [README.md](README.md) pour la documentation complète.

---

**Bon trading ! 🚀📈**
