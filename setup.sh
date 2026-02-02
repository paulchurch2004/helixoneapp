#!/bin/bash
# Script d'installation automatique pour HelixOne
# Utilisez ce script pour configurer HelixOne sur une nouvelle machine

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🚀 HELIXONE - Installation"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# 1. Vérifier Python 3.9+
# ============================================================================
echo "📋 Vérification des prérequis..."
echo ""

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 n'est pas installé${NC}"
    echo "Installez Python 3.9 ou supérieur depuis https://www.python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION trouvé${NC}"

# ============================================================================
# 2. Créer l'environnement virtuel
# ============================================================================
echo ""
echo "🔧 Configuration de l'environnement virtuel..."

if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  venv existe déjà, on le supprime et on recrée...${NC}"
    rm -rf venv
fi

python3 -m venv venv
echo -e "${GREEN}✅ Environnement virtuel créé${NC}"

# ============================================================================
# 3. Installer les dépendances
# ============================================================================
echo ""
echo "📦 Installation des dépendances Python..."
echo -e "${BLUE}Cela peut prendre quelques minutes...${NC}"
echo ""

# Mettre à jour pip
./venv/bin/pip install --upgrade pip > /dev/null 2>&1

# Installer les dépendances backend
echo "  → Installation dépendances backend..."
./venv/bin/pip install -q -r helixone-backend/requirements.txt

# Installer les dépendances frontend (si requirements.txt existe)
if [ -f "requirements.txt" ]; then
    echo "  → Installation dépendances frontend..."
    ./venv/bin/pip install -q -r requirements.txt
fi

echo -e "${GREEN}✅ Toutes les dépendances installées${NC}"

# ============================================================================
# 4. Configurer le fichier .env
# ============================================================================
echo ""
echo "⚙️  Configuration de l'environnement (.env)..."

if [ ! -f "helixone-backend/.env" ]; then
    # Copier le template
    cp helixone-backend/.env.example helixone-backend/.env

    # Modifier pour utiliser SQLite par défaut (plus simple)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' 's|^DATABASE_URL=postgresql.*|DATABASE_URL=sqlite:///./helixone.db|' helixone-backend/.env
        sed -i '' 's|^REDIS_URL=redis.*|# REDIS_URL=redis://localhost:6379/0  # Optionnel|' helixone-backend/.env
    else
        # Linux
        sed -i 's|^DATABASE_URL=postgresql.*|DATABASE_URL=sqlite:///./helixone.db|' helixone-backend/.env
        sed -i 's|^REDIS_URL=redis.*|# REDIS_URL=redis://localhost:6379/0  # Optionnel|' helixone-backend/.env
    fi

    # Générer une SECRET_KEY unique
    NEW_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|SECRET_KEY=.*|SECRET_KEY=$NEW_SECRET_KEY|" helixone-backend/.env
    else
        sed -i "s|SECRET_KEY=.*|SECRET_KEY=$NEW_SECRET_KEY|" helixone-backend/.env
    fi

    echo -e "${GREEN}✅ Fichier .env créé avec configuration SQLite${NC}"
    echo -e "${YELLOW}ℹ️  SECRET_KEY unique générée automatiquement${NC}"
else
    echo -e "${YELLOW}⚠️  Le fichier .env existe déjà, on ne le modifie pas${NC}"
fi

# ============================================================================
# 5. Créer les dossiers nécessaires
# ============================================================================
echo ""
echo "📁 Création des dossiers..."

mkdir -p data/watchlist
mkdir -p data/formation_commerciale
mkdir -p data/avatars
mkdir -p helixone-backend/ml_models/saved_models
mkdir -p helixone-backend/ml_models/results

echo -e "${GREEN}✅ Dossiers créés${NC}"

# ============================================================================
# 6. Initialiser la base de données
# ============================================================================
echo ""
echo "🗄️  Initialisation de la base de données..."

# La base de données sera créée automatiquement au premier démarrage
# grâce à Base.metadata.create_all() dans main.py

echo -e "${GREEN}✅ Base de données sera créée au premier démarrage${NC}"

# ============================================================================
# 7. Rendre les scripts exécutables
# ============================================================================
echo ""
echo "🔐 Configuration des permissions..."

chmod +x start.sh
chmod +x dev.sh
chmod +x setup.sh

echo -e "${GREEN}✅ Scripts rendus exécutables${NC}"

# ============================================================================
# RÉSUMÉ
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}✅ INSTALLATION TERMINÉE !${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📝 Prochaines étapes :"
echo ""
echo "1. (OPTIONNEL) Configurez vos clés API dans helixone-backend/.env"
echo "   → Éditez le fichier pour ajouter vos clés (Finnhub, Alpha Vantage, etc.)"
echo "   → L'application fonctionnera en mode limité sans clés API"
echo ""
echo "2. Lancez HelixOne :"
echo -e "   ${BLUE}./start.sh${NC}  (mode normal)"
echo -e "   ${BLUE}./dev.sh${NC}    (mode développement)"
echo ""
echo "3. Créez votre compte dans l'interface"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}💡 Conseil :${NC} Pour obtenir des clés API gratuites :"
echo "   • Finnhub: https://finnhub.io/register (60 req/min gratuit)"
echo "   • Alpha Vantage: https://www.alphavantage.co/support/#api-key (5 req/min)"
echo "   • FRED: https://fred.stlouisfed.org/docs/api/api_key.html (gratuit illimité)"
echo ""
echo "Bonne utilisation ! 🚀"
echo ""
