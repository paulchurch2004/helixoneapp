# 🚀 Guide de Lancement HelixOne

## ✅ Corrections Effectuées

1. **Token JWT** configuré dès le démarrage en mode DEV
2. **Attribut macro_score** ajouté à CompatibilityResult
3. **get_analysis()** modifié pour appeler l'API backend

## 📋 Étapes de Lancement

### Option 1: Lancement Automatique (Recommandé)

```bash
cd /Users/macintosh/Desktop/helixone
HELIXONE_DEV=1 python3 run.py
```

Le script `run.py` lance automatiquement:
- Le backend API sur le port 8000
- L'interface graphique

### Option 2: Lancement Manuel (Pour Debug)

**Terminal 1 - Backend**:
```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend
../venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Interface**:
```bash
cd /Users/macintosh/Desktop/helixone
HELIXONE_DEV=1 python3 run.py
```

## 🔧 En Cas de Problème

### Port 8000 déjà utilisé

```bash
# Tuer le processus sur le port 8000
lsof -ti:8000 | xargs kill -9
```

### Vérifier que le backend fonctionne

```bash
curl http://127.0.0.1:8000/health
```

Devrait retourner:
```json
{
  "status": "healthy",
  "app_name": "HelixOne API",
  "version": "1.0.0"
}
```

### Tester l'analyse directement

```bash
cd /Users/macintosh/Desktop/helixone

# Créer un script de test
cat > test_analysis.py << 'EOF'
import os
import sys
sys.path.insert(0, "/Users/macintosh/Desktop/helixone/src")

# Configurer le token
from src.auth_session import set_auth_token
set_auth_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMjI2ZjI0MDctNGY2Yi00ODMyLWJjMTQtZGZhNzQ4M2JmY2Y0IiwiZW1haWwiOiJ0ZXN0QGhlbGl4b25lLmNvbSIsImV4cCI6MTc5MTkzMDA2N30.DDnZTWxmHCfPW6mVJrhKCU0HJeD7vCxcPTTIXwjmq5M")

# Tester l'analyse
from src.fxi_engine import get_analysis
result = get_analysis("AAPL", "Standard")

print(f"Status: {result.get('status')}")
print(f"Score: {result.get('score_fxi')}")
print(f"Recommandation: {result.get('recommandation')}")
EOF

./venv/bin/python test_analysis.py
```

## 📊 Ce Qui Devrait Se Passer

1. **Intro Plasma** (5 secondes)
2. **Écran de connexion** (connexion auto en mode DEV)
3. **Interface principale** avec barre de recherche
4. **Recherche d'un ticker** (ex: AAPL, TSLA, MSFT)
5. **Affichage des résultats**:
   - Score FXI global
   - Recommandation
   - Scores détaillés (technique, fondamental, sentiment, risque, macro)
   - Graphiques et visualisations

## ⚠️ Limitations Connues

- **Yahoo Finance rate limiting**: Si vous effectuez trop de recherches rapidement, vous pourriez obtenir des erreurs 429
- **Données limitées**: Sans token Yahoo Finance premium, certaines données peuvent être incomplètes
- **Mode DEV**: Le token est valide 1 an, mais doit être régénéré après expiration

## 🔑 Régénérer un Token

Si le token expire:

```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend

cat > generate_token.py << 'EOF'
import sys
sys.path.insert(0, "/Users/macintosh/Desktop/helixone/helixone-backend")

from app.core.database import SessionLocal
from app.core.security import create_access_token
from app.models import User
from datetime import timedelta

db = SessionLocal()
test_user = db.query(User).filter(User.email == "test@helixone.com").first()

if test_user:
    token = create_access_token(
        data={"user_id": test_user.id, "email": "test@helixone.com"},
        expires_delta=timedelta(days=365)
    )
    print(f"Nouveau token:\n{token}")
else:
    print("Utilisateur test non trouvé")

db.close()
EOF

../venv/bin/python generate_token.py
```

Puis mettez à jour le token dans `/Users/macintosh/Desktop/helixone/run.py` ligne 42.
