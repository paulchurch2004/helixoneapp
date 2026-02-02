#!/bin/bash
# Script de build pour créer l'application HelixOne
# Crée une application .app sur macOS ou .exe sur Windows

set -e  # Arrêter en cas d'erreur

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  📦 HELIXONE - BUILD"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# 1. Vérifier l'environnement virtuel
# ============================================================================
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Environnement virtuel non trouvé${NC}"
    echo "Lancez d'abord: ./setup.sh"
    exit 1
fi

echo -e "${GREEN}✅ Environnement virtuel trouvé${NC}"

# ============================================================================
# 2. Installer PyInstaller si nécessaire
# ============================================================================
echo ""
echo "📦 Vérification de PyInstaller..."

if ! ./venv/bin/pip show pyinstaller > /dev/null 2>&1; then
    echo -e "${YELLOW}⚙️  Installation de PyInstaller...${NC}"
    ./venv/bin/pip install -q pyinstaller
    echo -e "${GREEN}✅ PyInstaller installé${NC}"
else
    echo -e "${GREEN}✅ PyInstaller déjà installé${NC}"
fi

# ============================================================================
# 3. Nettoyer les anciens builds
# ============================================================================
echo ""
echo "🧹 Nettoyage des anciens builds..."

rm -rf build/ dist/ HelixOne.app 2>/dev/null
echo -e "${GREEN}✅ Nettoyage terminé${NC}"

# ============================================================================
# 4. Build de l'application
# ============================================================================
echo ""
echo "🔨 Construction de l'application..."
echo -e "${BLUE}Cela peut prendre 2-5 minutes...${NC}"
echo ""

# Déterminer le fichier .spec selon l'OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    SPEC_FILE="HelixOne.spec"
    echo "🍎 Build pour macOS..."
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    SPEC_FILE="HelixOne_Windows.spec"
    echo "🪟 Build pour Windows..."
else
    SPEC_FILE="HelixOne.spec"
    echo "🐧 Build pour Linux..."
fi

# Stamp build date
BUILD_DATE=$(date +%Y-%m-%d)
sed -i '' "s/BUILD_DATE = .*/BUILD_DATE = \"$BUILD_DATE\"/" src/updater/version.py
echo -e "${GREEN}✅ Build date: $BUILD_DATE${NC}"

# Lancer PyInstaller
./venv/bin/pyinstaller "$SPEC_FILE" --clean --noconfirm

# ============================================================================
# 5. Vérifier le résultat
# ============================================================================
echo ""
echo "✅ Vérification du build..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - chercher le .app
    if [ -d "dist/HelixOne.app" ]; then
        APP_SIZE=$(du -sh dist/HelixOne.app | awk '{print $1}')
        echo -e "${GREEN}✅ Application créée : dist/HelixOne.app ($APP_SIZE)${NC}"

        # Créer le DMG pour distribution
        echo ""
        echo "📀 Création du DMG..."
        rm -f dist/HelixOne.dmg 2>/dev/null
        hdiutil create -volname "HelixOne" -srcfolder dist/HelixOne.app \
            -ov -format UDZO dist/HelixOne.dmg
        if [ -f "dist/HelixOne.dmg" ]; then
            DMG_SIZE=$(du -sh dist/HelixOne.dmg | awk '{print $1}')
            echo -e "${GREEN}✅ DMG créé : dist/HelixOne.dmg ($DMG_SIZE)${NC}"
        fi

        # Copier .app à la racine pour faciliter l'accès
        cp -r dist/HelixOne.app ./HelixOne.app
        echo -e "${GREEN}✅ Copié à la racine : HelixOne.app${NC}"
    else
        echo -e "${RED}❌ Erreur : HelixOne.app non trouvé${NC}"
        exit 1
    fi
else
    # Windows/Linux - chercher l'exécutable
    if [ -f "dist/HelixOne/HelixOne.exe" ] || [ -f "dist/HelixOne/HelixOne" ]; then
        APP_SIZE=$(du -sh dist/HelixOne | awk '{print $1}')
        echo -e "${GREEN}✅ Application créée : dist/HelixOne/ ($APP_SIZE)${NC}"
    else
        echo -e "${RED}❌ Erreur : Exécutable non trouvé${NC}"
        exit 1
    fi
fi

# ============================================================================
# 6. Instructions finales
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}✅ BUILD TERMINÉ !${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🎉 Votre application est prête !"
    echo ""
    echo "📱 Pour lancer l'application :"
    echo "   → Double-cliquez sur HelixOne.app"
    echo "   → Ou faites: open HelixOne.app"
    echo ""
    echo "📦 Pour distribuer :"
    echo "   1. Compressez: zip -r HelixOne.zip HelixOne.app"
    echo "   2. Ou créez un DMG: ./create_dmg.sh (si disponible)"
    echo ""
    echo "⚠️  Note : Au premier lancement, macOS peut demander"
    echo "   d'autoriser l'application (Préférences Système → Sécurité)"
else
    echo "🎉 Votre application est prête !"
    echo ""
    echo "📱 Pour lancer l'application :"
    echo "   → Naviguez vers dist/HelixOne/"
    echo "   → Double-cliquez sur HelixOne.exe (Windows) ou HelixOne (Linux)"
    echo ""
    echo "📦 Pour distribuer :"
    echo "   → Compressez le dossier dist/HelixOne/"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
