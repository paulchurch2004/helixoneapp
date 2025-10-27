# ⚡ Actions Immédiates - HelixOne

**Ce document contient les actions à faire MAINTENANT pour sécuriser et améliorer rapidement le projet.**

---

## 🔴 CRITIQUE - À FAIRE AUJOURD'HUI

### 1. Sécuriser les Clés API (30 minutes)

#### Étape 1 : Rotation des clés

```bash
# ⚠️ VOS CLÉS SONT EXPOSÉES PUBLIQUEMENT !

# 1. Alpha Vantage
# Aller sur: https://www.alphavantage.co/support/#api-key
# - Révoquer: PEHB0Q9ZHXMWFM0X
# - Créer nouvelle clé

# 2. FRED (Federal Reserve)
# Aller sur: https://fred.stlouisfed.org/docs/api/api_key.html
# - Révoquer: 2eb1601f70b8771864fd98d891879301
# - Créer nouvelle clé
```

#### Étape 2 : Supprimer .env de Git

```bash
cd /Users/macintosh/Desktop/helixone

# 1. Supprimer de l'historique Git
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# 2. Force push (si repo distant)
# ⚠️ ATTENTION: Cela réécrit l'historique !
# git push origin --force --all

# 3. Nettoyer les refs
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

#### Étape 3 : Mettre à jour .gitignore

```bash
# Vérifier que .gitignore contient:
cat >> .gitignore << 'EOF'

# Environment variables
.env
.env.*
*.env
.env.local
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

EOF

git add .gitignore
git commit -m "security: update .gitignore to exclude sensitive files"
```

#### Étape 4 : Créer .env.example

```bash
cat > .env.example << 'EOF'
# HelixOne Environment Variables Template
# Copy to .env and fill with your actual values

# API Keys (Get your own!)
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
NEWS_API_KEY=your_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/helixone

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=generate_a_strong_random_key
ALGORITHM=HS256

# OpenAI (Optional)
OPENAI_API_KEY=your_key_here

# Sentry (Optional)
SENTRY_DSN=your_sentry_dsn_here

# IBKR (Optional)
IBKR_GATEWAY_HOST=localhost
IBKR_GATEWAY_PORT=7497
EOF

git add .env.example
git commit -m "docs: add .env.example template"
```

**✅ Vérification** :
```bash
# .env ne doit PAS apparaître
git ls-files | grep ".env$"

# Doit retourner vide (aucun résultat)
```

---

### 2. Nettoyer le Repository (15 minutes)

```bash
cd /Users/macintosh/Desktop/helixone

# 1. Supprimer tous les caches Python
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# 2. Nettoyer fichiers temporaires
find . -name ".DS_Store" -delete
find . -name "*.swp" -delete
find . -name "*.swo" -delete
find . -name "*~" -delete

# 3. Supprimer logs anciens
find . -name "*.log" -mtime +30 -delete

echo "✅ Repository nettoyé"
```

---

### 3. Fix .gitignore Complet (5 minutes)

```bash
cat > .gitignore << 'EOF'
# ============================================================================
# HelixOne .gitignore
# ============================================================================

# Environment Variables & Secrets
.env
.env.*
*.env
.env.local
.env.development
.env.production
.env.test
config/secrets.yaml
config/credentials.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject
.settings/
*.sublime-project
*.sublime-workspace

# OS Files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# Jupyter Notebooks
.ipynb_checkpoints
*.ipynb

# Database
*.db
*.sqlite
*.sqlite3

# Logs
*.log
logs/
helixone-backend/logs/

# Cache
.cache/
*.cache
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Coverage
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover

# Data files (exemples - ajuster selon besoins)
data/*.csv
data/*.json
!data/example.csv
ml_models/saved_models/

# Temporary files
tmp/
temp/
*.tmp

# Archives
*.zip
*.tar.gz
*.rar

# Compiled files
*.com
*.class
*.dll
*.exe
*.o
*.so

EOF

git add .gitignore
git commit -m "chore: comprehensive .gitignore"
```

---

## 🟠 IMPORTANT - À FAIRE CETTE SEMAINE

### 4. Setup Pre-commit Hooks (1 heure)

```bash
# Installation
pip install pre-commit

# Configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  # Code formatting
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.11

  # Linting
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=120, --ignore=E203,W503]

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  # Security
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: [-ll, -i]

  # Remove trailing whitespace
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-json
      - id: check-merge-conflict
      - id: detect-private-key

EOF

# Installer hooks
pre-commit install

# Test
pre-commit run --all-files

git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hooks"
```

---

### 5. Setup Logging Proper (2 heures)

```python
# helixone-backend/app/core/logging_config.py
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

def setup_logging(
    app_name: str = "helixone",
    log_level: str = "INFO",
    log_dir: Path = Path("logs")
):
    """Configure logging pour toute l'application"""

    # Créer répertoire logs
    log_dir.mkdir(exist_ok=True)

    # Format
    log_format = (
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s:%(funcName)s:%(lineno)d | "
        "%(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(log_format, date_format)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler - Tous les logs
    file_handler = RotatingFileHandler(
        log_dir / f"{app_name}.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # File handler - Erreurs seulement
    error_handler = RotatingFileHandler(
        log_dir / f"{app_name}_errors.log",
        maxBytes=10_000_000,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # File handler - Daily rotation
    daily_handler = TimedRotatingFileHandler(
        log_dir / f"{app_name}_daily.log",
        when='midnight',
        interval=1,
        backupCount=30,  # Garder 30 jours
        encoding='utf-8'
    )
    daily_handler.setLevel(logging.INFO)
    daily_handler.setFormatter(formatter)
    root_logger.addHandler(daily_handler)

    # Silencer certains loggers verbeux
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    return root_logger

# Usage dans main.py
from app.core.logging_config import setup_logging

logger = setup_logging(log_level="INFO")
logger.info("HelixOne backend started")
```

---

### 6. Script de Migration print() → logger (2 heures)

```python
#!/usr/bin/env python3
"""
Script pour remplacer tous les print() par logger.info()
"""
import re
import os
from pathlib import Path

def add_logger_import(content: str) -> str:
    """Ajoute import logging si absent"""
    if "import logging" not in content:
        # Trouver les imports
        imports_end = content.find("\n\n")
        if imports_end == -1:
            imports_end = 0

        insert_pos = imports_end
        content = (
            content[:insert_pos] +
            "\nimport logging\n\nlogger = logging.getLogger(__name__)\n" +
            content[insert_pos:]
        )
    elif "logger = logging.getLogger" not in content:
        # Import existe mais pas logger
        match = re.search(r"import logging.*?\n", content)
        if match:
            insert_pos = match.end()
            content = (
                content[:insert_pos] +
                "\nlogger = logging.getLogger(__name__)\n" +
                content[insert_pos:]
            )

    return content

def replace_prints(content: str) -> tuple[str, int]:
    """Remplace print() par logger"""
    count = 0

    # Pattern 1: print("string")
    def replace_simple(match):
        nonlocal count
        count += 1
        return f"logger.info({match.group(1)})"

    content = re.sub(r'print\((.*?)\)(?!\s*#\s*noqa)', replace_simple, content)

    # Pattern 2: print(f"...")
    # Déjà géré par le pattern ci-dessus

    return content, count

def process_file(filepath: Path) -> int:
    """Process un fichier Python"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip si déjà migré
        if "# LOGGING_MIGRATED" in content:
            return 0

        original = content
        content = add_logger_import(content)
        content, count = replace_prints(content)

        if count > 0:
            # Ajouter marqueur
            content = f"# LOGGING_MIGRATED\n{content}"

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ {filepath}: {count} print() remplacés")

        return count

    except Exception as e:
        print(f"❌ Erreur {filepath}: {e}")
        return 0

def main():
    """Main function"""
    root = Path("/Users/macintosh/Desktop/helixone")

    # Directories à scanner
    directories = [
        root / "helixone-backend" / "app",
        root / "src",
    ]

    total = 0
    for directory in directories:
        if not directory.exists():
            continue

        print(f"\n📂 Scanning {directory}...")

        for filepath in directory.rglob("*.py"):
            # Skip venv, tests, migrations
            if any(part in filepath.parts for part in ["venv", "__pycache__", "migrations", "alembic"]):
                continue

            count = process_file(filepath)
            total += count

    print(f"\n✅ TOTAL: {total} print() remplacés")

if __name__ == "__main__":
    main()
```

**Usage** :
```bash
chmod +x migrate_logging.py
./migrate_logging.py

# Review changes
git diff

# Commit si OK
git add -A
git commit -m "refactor: replace print() with logger"
```

---

### 7. Consolider Documentation (4 heures)

```bash
# Créer structure /docs/
mkdir -p docs/{getting-started,architecture,user-guide,developer}

# Fusionner fichiers similaires
cat README.md README_FINAL.md LANCER_MAINTENANT.md > docs/getting-started/quickstart.md

# Supprimer duplicatas
rm README_FINAL.md LANCER_MAINTENANT.md DEMARRAGE_SIMPLE.md

# Organiser par thème
mv ANALYSE_*.md docs/architecture/
mv STATUS_*.md docs/developer/
mv DATA_SOURCES_*.md docs/architecture/

# Créer index
cat > docs/README.md << 'EOF'
# HelixOne Documentation

## Getting Started
- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Configuration](getting-started/configuration.md)

## Architecture
- [Overview](architecture/overview.md)
- [Backend](architecture/backend.md)
- [Frontend](architecture/frontend.md)
- [Data Sources](architecture/data-sources.md)

## User Guide
- [Search & Analysis](user-guide/search.md)
- [Portfolio Management](user-guide/portfolio.md)
- [Alerts](user-guide/alerts.md)

## Developer
- [Contributing](developer/contributing.md)
- [API Reference](developer/api.md)
- [Testing](developer/testing.md)
EOF

git add docs/
git commit -m "docs: consolidate documentation in /docs/"
```

---

## 📊 Checklist de Validation

Après avoir fait ces actions, vérifiez :

### Sécurité
- [ ] .env supprimé de Git
- [ ] .env dans .gitignore
- [ ] API keys rotées
- [ ] .env.example créé
- [ ] Pas de credentials en dur dans le code

### Code Quality
- [ ] Pre-commit hooks installés
- [ ] Logging configuré
- [ ] print() remplacés par logger
- [ ] Repository nettoyé (__pycache__ supprimés)

### Documentation
- [ ] < 20 fichiers .md au root
- [ ] /docs/ structure créée
- [ ] Duplicatas supprimés

---

## 🧪 Tests de Validation

```bash
# 1. Vérifier .env n'est pas dans Git
git ls-files | grep "\.env$"
# → Doit être vide

# 2. Vérifier pas de __pycache__
find . -name "__pycache__" -type d
# → Doit être vide

# 3. Vérifier pre-commit
pre-commit run --all-files
# → Doit passer tous les checks

# 4. Vérifier logging
grep -r "print(" --include="*.py" helixone-backend/app/ | wc -l
# → Devrait être < 10

# 5. Lancer les tests
pytest tests/ -v
```

---

## ⏱️ Timing Total

| Action | Temps | Priorité |
|--------|-------|----------|
| Sécuriser clés API | 30min | 🔴 CRITIQUE |
| Nettoyer repo | 15min | 🔴 CRITIQUE |
| Fix .gitignore | 5min | 🔴 CRITIQUE |
| Pre-commit hooks | 1h | 🟠 Important |
| Setup logging | 2h | 🟠 Important |
| Migrer print() | 2h | 🟠 Important |
| Consolider docs | 4h | 🟡 Moyen |
| **TOTAL** | **~10h** | **1-2 jours** |

---

## 🎯 Résultat Attendu

Après ces actions :
- ✅ **Sécurité** : Clés protégées, .env sécurisé
- ✅ **Code Quality** : Logging proper, pre-commit actif
- ✅ **Documentation** : Structure claire, pas de duplicatas
- ✅ **Repository** : Propre, professionnel

**→ Projet prêt pour développement collaboratif et mise en production**

---

**Créé par** : Claude
**Date** : 27 Octobre 2025
**Priorité** : IMMÉDIATE
