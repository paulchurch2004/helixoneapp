# 🚀 HelixOne - Plateforme de Trading Intelligente

HelixOne est une plateforme complète d'analyse boursière avec intelligence artificielle, analyse de portfolio et trading automatisé.

## ✨ Fonctionnalités

- 🤖 **Analyses ML** - Prédictions machine learning (XGBoost, LSTM)
- 📊 **Portfolio Analysis** - Analyse complète de votre portefeuille
- 🔔 **Système d'alertes** - Notifications en temps réel
- 📈 **Graphiques avancés** - Visualisations interactives
- 🎓 **Formation trading** - Module d'apprentissage intégré
- 🔐 **Authentification 2FA** - Sécurité maximale
- 🌐 **35+ sources de données** - Reddit, News, Google Trends, FRED, etc.

## 📋 Prérequis

- **Python 3.9 ou supérieur** ([Télécharger](https://www.python.org/downloads/))
- **macOS, Linux ou Windows**
- **4 GB RAM minimum** (8 GB recommandé pour ML)

## 🔧 Installation (3 étapes)

### 1️⃣ Cloner le projet

```bash
git clone https://github.com/votre-repo/helixone.git
cd helixone
```

### 2️⃣ Lancer l'installation automatique

```bash
./setup.sh
```

Le script va automatiquement :
- ✅ Créer l'environnement virtuel Python
- ✅ Installer toutes les dépendances
- ✅ Configurer la base de données SQLite
- ✅ Générer une clé de sécurité unique
- ✅ Créer les dossiers nécessaires

### 3️⃣ Lancer HelixOne

```bash
./start.sh
```

L'interface graphique va s'ouvrir et vous pourrez créer votre compte ! 🎉

## 🎯 Premier lancement

1. **Créez votre compte** dans l'écran d'accueil
2. **Connectez-vous** avec vos identifiants
3. **Ajoutez des actions** à votre watchlist (ex: AAPL, MSFT, TSLA)
4. **Lancez une analyse** pour obtenir des prédictions ML

## 🔑 Configuration des clés API (optionnel)

Pour accéder à toutes les fonctionnalités, configurez vos clés API gratuites :

1. Éditez le fichier `helixone-backend/.env`
2. Ajoutez vos clés API :

```bash
# Clés API recommandées (toutes gratuites)
FINNHUB_API_KEY=votre_cle_ici       # https://finnhub.io/register
FRED_API_KEY=votre_cle_ici          # https://fred.stlouisfed.org
ALPHA_VANTAGE_API_KEY=votre_cle_ici # https://www.alphavantage.co
```

**Sans clés API** : L'application fonctionnera en mode limité avec Yahoo Finance uniquement.

## 📖 Utilisation

### Mode Normal
```bash
./start.sh  # Lance backend + frontend
```

### Mode Développement
```bash
./dev.sh    # Logs détaillés pour debugging
```

### Arrêter l'application
Fermez simplement la fenêtre ou appuyez sur `Ctrl+C` dans le terminal.

## 🏗️ Architecture

```
helixone/
├── helixone-backend/     # API FastAPI
│   ├── app/
│   │   ├── api/          # Routes API
│   │   ├── services/     # Logique métier
│   │   ├── models/       # Modèles de données
│   │   └── ml_models/    # Modèles ML
├── src/
│   └── interface/        # Interface CustomTkinter
├── data/                 # Données utilisateur
├── assets/               # Images et sons
├── setup.sh              # Installation automatique
├── start.sh              # Lancement normal
└── dev.sh                # Lancement développement
```

## 🔧 Dépannage

### Le backend ne démarre pas
```bash
# Vérifier que le port 8000 est libre
lsof -ti:8000 | xargs kill -9

# Vérifier les logs
cat backend.log
```

### Erreur de dépendances
```bash
# Réinstaller les dépendances
./venv/bin/pip install -r helixone-backend/requirements.txt
```

### Base de données corrompue
```bash
# Supprimer et recréer la DB
rm helixone-backend/helixone.db
./start.sh  # La DB sera recréée automatiquement
```

## 🛡️ Sécurité

- ✅ Authentification JWT sécurisée
- ✅ 2FA (Two-Factor Authentication) disponible
- ✅ Mots de passe hashés avec bcrypt
- ✅ SECRET_KEY unique générée automatiquement
- ✅ Rate limiting sur l'API
- ✅ CORS configuré

## 📊 Sources de données

HelixOne agrège des données depuis :
- 📈 Yahoo Finance (gratuit)
- 📰 News API (gratuit avec limite)
- 🔥 Reddit (via PRAW)
- 📊 Google Trends
- 🏦 FRED (Federal Reserve)
- 💹 Finnhub (gratuit 60 req/min)
- 📈 Alpha Vantage (gratuit 5 req/min)
- Et bien plus...

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📝 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🆘 Support

Pour toute question ou problème :
- 📧 Email: support@helixone.com
- 💬 Discord: [Lien vers Discord]
- 📖 Documentation complète: [Lien vers docs]

## 🙏 Remerciements

Construit avec :
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web moderne
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Interface graphique moderne
- [XGBoost](https://xgboost.readthedocs.io/) - Machine Learning
- [yfinance](https://github.com/ranaroussi/yfinance) - Données financières

---

**Fait avec ❤️ pour la communauté des traders**

🚀 **Bon trading !**
