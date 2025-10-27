# 🚀 Quick Start - Moteur de Scénarios

**Temps estimé**: 5 minutes

---

## 1. Lancer le Backend

```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend
../venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Vous devriez voir:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## 2. Vérifier que l'API fonctionne

```bash
curl http://127.0.0.1:8000/health
```

Résultat:
```json
{
  "status": "healthy",
  "app_name": "HelixOne API",
  "version": "1.0.0",
  "environment": "development",
  "database": "connected"
}
```

---

## 3. Voir les Scénarios Disponibles

```bash
# Token de développement (1 an de validité)
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMjI2ZjI0MDctNGY2Yi00ODMyLWJjMTQtZGZhNzQ4M2JmY2Y0IiwiZW1haWwiOiJ0ZXN0QGhlbGl4b25lLmNvbSIsImV4cCI6MTc5MTkzMDA2N30.DDnZTWxmHCfPW6mVJrhKCU0HJeD7vCxcPTTIXwjmq5M"

curl -X GET "http://127.0.0.1:8000/api/scenarios/predefined" \
  -H "Authorization: Bearer $TOKEN"
```

Vous verrez:
- 4 stress tests (market_crash, rate_shock, volatility_spike...)
- 4 événements historiques (2008, COVID, dot-com, Black Monday)

---

## 4. Lancer un Stress Test

```bash
curl -X POST "http://127.0.0.1:8000/api/scenarios/stress-test" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "positions": {
        "AAPL": 100,
        "MSFT": 50,
        "TSLA": 30
      },
      "cash": 10000
    },
    "scenario_type": "market_crash",
    "shock_percent": -0.30
  }'
```

---

## 5. Résultat Attendu

```json
{
  "scenario_name": "Market Crash",
  "scenario_type": "stress_test",
  "portfolio_value_before": 85000.0,
  "portfolio_value_after": 58350.0,
  "total_impact_pct": -31.4,
  "metrics": {
    "var_95": -31.4,
    "cvar_95": -37.7,
    "max_drawdown": 31.4,
    "stress_score": 57,
    "recovery_time_days": 94
  },
  "recommendations": [
    {
      "type": "hedge",
      "action": "Acheter un ETF inverse (SQQQ, SPXU) pour hedge",
      "reason": "Impact de -31.4% très élevé",
      "amount": 8500.0,
      "priority": 5
    }
  ],
  "worst_position": {
    "ticker": "TSLA",
    "impact": -48.2
  }
}
```

---

## 6. Voir l'Historique

```bash
curl -X GET "http://127.0.0.1:8000/api/scenarios/history" \
  -H "Authorization: Bearer $TOKEN"
```

Vous verrez toutes vos simulations passées.

---

## 7. Tester d'Autres Scénarios

### Choc de Taux d'Intérêt (+5%)
```bash
curl -X POST "http://127.0.0.1:8000/api/scenarios/stress-test" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "positions": {"AAPL": 100, "MSFT": 50}
    },
    "scenario_type": "interest_rate_shock",
    "shock_percent": 0.05
  }'
```

### Spike de Volatilité (VIX x3)
```bash
curl -X POST "http://127.0.0.1:8000/api/scenarios/stress-test" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio": {
      "positions": {"AAPL": 100, "TSLA": 50}
    },
    "scenario_type": "volatility_spike"
  }'
```

---

## 8. Documentation Swagger

Ouvrez dans votre navigateur:
**http://127.0.0.1:8000/docs**

Interface interactive pour tester tous les endpoints!

---

## ✅ Checklist de Test

- [ ] Backend lancé et répond sur http://127.0.0.1:8000
- [ ] `/health` retourne "healthy"
- [ ] `/api/scenarios/predefined` liste les scénarios
- [ ] Stress test fonctionne et retourne des résultats
- [ ] Historique sauvegarde les simulations
- [ ] Swagger docs accessible

---

## 🐛 Dépannage

### Erreur: "Address already in use"
```bash
# Tuer le processus sur le port 8000
lsof -ti:8000 | xargs kill -9
```

### Erreur: "Token expired"
Régénérer un token:
```bash
cd /Users/macintosh/Desktop/helixone/helixone-backend
../venv/bin/python << 'EOF'
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
```

### Erreur: "Module not found"
```bash
# Installer les dépendances manquantes
cd /Users/macintosh/Desktop/helixone
./venv/bin/pip install scipy numpy
```

---

## 📊 Prochaines Étapes

1. ✅ Tester manuellement l'API
2. 🔧 Ajouter d'autres scénarios
3. 🎨 Créer l'interface frontend
4. 🧠 Implémenter le ML

Voir: [`SCENARIO_ENGINE_IMPLEMENTATION.md`](SCENARIO_ENGINE_IMPLEMENTATION.md) pour détails.

---

**Besoin d'aide?** Consultez:
- Documentation complète: [`SCENARIO_ENGINE_DESIGN.md`](SCENARIO_ENGINE_DESIGN.md)
- Implémentation: [`SCENARIO_ENGINE_IMPLEMENTATION.md`](SCENARIO_ENGINE_IMPLEMENTATION.md)
