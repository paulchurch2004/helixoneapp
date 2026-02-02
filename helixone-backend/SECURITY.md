# 🔐 Guide de Sécurité - HelixOne

Documentation complète des pratiques de sécurité pour HelixOne.

## 📋 Table des Matières

1. [Gestion des Secrets](#gestion-des-secrets)
2. [Authentification](#authentification)
3. [Sécurité des API](#sécurité-des-api)
4. [Base de Données](#base-de-données)
5. [Audit et Logging](#audit-et-logging)
6. [Bonnes Pratiques](#bonnes-pratiques)

---

## 🔑 Gestion des Secrets

### Secrets Manager

HelixOne utilise un gestionnaire de secrets centralisé dans `app/core/secrets_manager.py`.

**Features:**
- Chargement sécurisé depuis variables d'environnement
- Validation automatique au démarrage
- Rotation des secrets
- Audit logging
- Aucun secret dans les logs

### Configuration des Secrets

#### 1. Fichier .env

Créer un fichier `.env` à la racine du backend:

```bash
# Application
SECRET_KEY=<généré avec scripts/generate_secret_key.py>
DATABASE_URL=postgresql://user:pass@localhost/helixone

# API Keys (optionnelles)
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Redis (optionnel)
REDIS_URL=redis://:password@localhost:6379/0

# Sentry (optionnel)
SENTRY_DSN=your_sentry_dsn
```

#### 2. Générer une SECRET_KEY Sécurisée

```bash
# Générer une clé 32 bytes (recommandé)
python scripts/generate_secret_key.py

# Générer une clé plus longue
python scripts/generate_secret_key.py --length 64

# Générer plusieurs clés (dev, staging, prod)
python scripts/generate_secret_key.py --multiple 3
```

**Règles pour SECRET_KEY:**
- ✅ Minimum 32 caractères
- ✅ Générée aléatoirement (cryptographiquement sûr)
- ✅ Différente pour chaque environnement (dev/staging/prod)
- ❌ JAMAIS commitée dans git
- ❌ JAMAIS partagée par email/Slack

#### 3. Vérifier les Secrets

```bash
# Vérifier quels secrets sont configurés
python scripts/rotate_secrets.py --check

# Vérifier la force des secrets
python -c "from app.core.config import validate_settings; validate_settings()"
```

### Rotation des Secrets

Les secrets doivent être rotés régulièrement pour limiter l'impact en cas de compromission.

**Calendrier recommandé:**
- SECRET_KEY: tous les 90 jours
- API Keys: tous les 180 jours
- Database passwords: tous les 90 jours

#### Rotation Manuelle

```bash
# 1. Vérifier quels secrets nécessitent rotation
python scripts/rotate_secrets.py --check

# 2. Simulation (dry-run)
python scripts/rotate_secrets.py --rotate-all --dry-run

# 3. Rotation réelle
python scripts/rotate_secrets.py --rotate-all

# 4. Rotation d'un secret spécifique
python scripts/rotate_secrets.py --rotate SECRET_KEY
```

#### Rotation Automatique (TODO)

Pour production, implémenter rotation automatique:
- HashiCorp Vault Dynamic Secrets
- AWS Secrets Manager avec Lambda
- Kubernetes Secrets avec rotation

---

## 🔐 Authentification

### JWT Tokens

HelixOne utilise JWT (JSON Web Tokens) pour l'authentification.

**Configuration:**
```python
# app/core/config.py
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080  # 7 jours
```

**Sécurité:**
- ✅ Tokens signés avec SECRET_KEY forte
- ✅ Expiration automatique
- ✅ Validation stricte (signature + expiration)
- ❌ Pas de refresh tokens (TODO)
- ❌ Pas de token blacklist (TODO)

### Passwords

**Hashing:**
- Algorithme: **bcrypt**
- Rounds: **12** (2^12 = 4096 iterations)
- Salt: Automatique et unique par password

**Règles de mot de passe (TODO):**
- Minimum 8 caractères
- Au moins 1 majuscule, 1 minuscule, 1 chiffre
- Vérifier contre liste de mots de passe communs
- Rate limiting sur login

---

## 🛡️ Sécurité des API

### Rate Limiting

**Configuration actuelle:**
```python
RATE_LIMIT_ENABLED = True
RATE_LIMIT_PER_MINUTE = 60  # Global
```

**Améliorations nécessaires (P1):**
- [ ] Rate limiting per-user
- [ ] Endpoints sensibles avec limites plus strictes
- [ ] Circuit breakers pour sources externes

### CORS

**Configuration:**
```python
CORS_ORIGINS = ["http://localhost", "helixone://"]
```

**Production:**
- [ ] Restreindre aux domaines spécifiques
- [ ] Pas de wildcard "*"
- [ ] Vérifier credentials

### HTTPS/TLS

**Status:** ⚠️ Non implémenté (P1)

**À faire:**
```bash
# Générer certificats self-signed (dev)
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Lancer avec TLS
uvicorn app.main:app \
  --ssl-keyfile=./key.pem \
  --ssl-certfile=./cert.pem \
  --host 0.0.0.0 --port 8443
```

**Production:** Utiliser Let's Encrypt + Nginx reverse proxy

### Input Validation

**Status:** ✅ Implémenté avec Pydantic

Tous les inputs API sont validés automatiquement via Pydantic schemas.

**Exemple:**
```python
class UserRegister(BaseModel):
    email: EmailStr  # Validation email automatique
    password: str
    first_name: Optional[str] = None
```

### Vulnérabilités Prévenues

- ✅ **SQL Injection:** Utilisation de SQLAlchemy ORM
- ✅ **XSS:** Pas de HTML rendering côté backend
- ❌ **CSRF:** Protection manquante (P1)
- ⚠️ **Command Injection:** subprocess.run() dans run.py (partiellement corrigé)

---

## 🗄️ Base de Données

### Connexion Sécurisée

**PostgreSQL:**
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/helixone?sslmode=require
```

**Bonnes pratiques:**
- ✅ SSL/TLS requis en production (`sslmode=require`)
- ✅ User avec privilèges minimaux (pas root)
- ✅ Passwords forts et rotés
- ❌ Connection pooling explicite (TODO)

### Chiffrement

**Au repos:**
- PostgreSQL: Activer encryption at rest
- Backups: Chiffrer avec GPG

**En transit:**
- SSL/TLS pour toutes les connexions
- Certificats valides

### Backups

**Stratégie recommandée:**
```bash
# Backup quotidien
pg_dump -h localhost -U user -Fc helixone > backup_$(date +%Y%m%d).dump

# Chiffrer
gpg --encrypt --recipient admin@helixone.com backup_*.dump

# Uploader vers S3 (chiffré)
aws s3 cp backup_*.dump.gpg s3://helixone-backups/
```

---

## 📊 Audit et Logging

### Événements à Logger

**Obligatoires:**
- [ ] Authentification (login/logout/échecs)
- [ ] Accès aux données sensibles (trades, portfolio)
- [ ] Modifications de configuration
- [ ] Erreurs d'API
- [ ] Accès refusés (401/403)

**Format recommandé:**
```json
{
  "timestamp": "2025-11-10T12:34:56Z",
  "event": "user_login",
  "user_id": "uuid",
  "ip": "1.2.3.4",
  "success": true,
  "metadata": {}
}
```

### Logs de Sécurité

**Ne JAMAIS logger:**
- ❌ Passwords (même hashés)
- ❌ Tokens JWT complets
- ❌ API Keys
- ❌ Secrets

**Logger uniquement:**
- ✅ User IDs
- ✅ Actions
- ✅ Timestamps
- ✅ IPs (anonymisés en prod)

---

## ✅ Bonnes Pratiques

### Checklist Développement

- [ ] Tests de sécurité écrits
- [ ] Pas de secrets hardcodés
- [ ] Validation stricte des inputs
- [ ] Gestion d'erreurs appropriée (pas de stack traces en prod)
- [ ] Logging d'audit
- [ ] Rate limiting testé

### Checklist Déploiement

- [ ] SECRET_KEY unique générée
- [ ] HTTPS/TLS activé
- [ ] Database password roté
- [ ] Firewall configuré
- [ ] Monitoring activé (Sentry)
- [ ] Logs centralisés (ELK)
- [ ] Backups testés

### Checklist Maintenance

- [ ] Dépendances à jour (`pip-audit`)
- [ ] Secrets rotés (90 jours)
- [ ] Logs d'audit revus
- [ ] Scan de vulnérabilités (`bandit`)
- [ ] Pentest annuel (production)

### Outils de Sécurité

```bash
# Audit de sécurité du code
bandit -r app/

# Audit des dépendances
pip-audit
safety check

# Tests de sécurité
pytest -m security

# Format + lint
black app/
flake8 app/
mypy app/
```

---

## 🚨 Incident Response

### En cas de compromission de SECRET_KEY

1. **Immédiatement:**
   ```bash
   # Générer nouvelle clé
   python scripts/generate_secret_key.py

   # Mettre à jour .env
   SECRET_KEY=<nouvelle_clé>

   # Redémarrer l'application
   ```

2. **Invalider tous les tokens:**
   - Tous les utilisateurs doivent se reconnecter
   - Implémenter token blacklist si nécessaire

3. **Audit:**
   - Vérifier les logs pour accès non autorisés
   - Notifier les utilisateurs si données compromises

### En cas de fuite d'API Key

1. Révoquer immédiatement la clé compromise
2. Générer nouvelle clé chez le provider
3. Mettre à jour dans .env
4. Vérifier les logs pour usage non autorisé

---

## 📚 Ressources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)
- [bcrypt Explained](https://en.wikipedia.org/wiki/Bcrypt)

---

## 📞 Contact Sécurité

Pour rapporter une vulnérabilité: security@helixone.com

**Responsible Disclosure:**
- Nous répondons sous 48h
- Correction sous 30 jours
- Crédit public si souhaité
