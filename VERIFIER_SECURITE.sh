#!/bin/bash
# Script de vérification rapide de la sécurité HelixOne

echo "🔍 Vérification de Sécurité HelixOne"
echo "===================================="
echo ""

# Couleurs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction de vérification
check_item() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅${NC} $2"
    else
        echo -e "${RED}❌${NC} $2"
    fi
}

check_warning() {
    echo -e "${YELLOW}⚠️${NC}  $1"
}

# 1. Vérifier .env.example
echo "📋 1. Fichiers de configuration"
echo "--------------------------------"
if [ -f ".env.example" ]; then
    check_item 0 ".env.example existe"
else
    check_item 1 ".env.example manquant"
fi

if [ -f ".env" ]; then
    check_item 0 ".env existe (utilisé par l'app)"

    # Vérifier si les clés sont encore des placeholders
    if grep -q "your_.*_key_here" .env 2>/dev/null; then
        check_warning "Des placeholders détectés dans .env - avez-vous mis vos vraies clés?"
    fi
else
    check_item 1 ".env manquant - copiez .env.example vers .env"
fi

echo ""

# 2. Vérifier .gitignore
echo "🔒 2. Protection Git"
echo "--------------------"
if [ -f ".gitignore" ]; then
    check_item 0 ".gitignore existe"

    # Vérifier que .env est dans .gitignore
    if grep -q "^\.env$" .gitignore 2>/dev/null; then
        check_item 0 ".env est protégé dans .gitignore"
    else
        check_item 1 ".env n'est PAS protégé dans .gitignore"
    fi
else
    check_item 1 ".gitignore manquant"
fi

# Vérifier si Git est initialisé
if [ -d ".git" ]; then
    check_item 0 "Git initialisé"

    # Vérifier si .env est tracké
    if git ls-files --error-unmatch .env 2>/dev/null; then
        check_item 1 "⚠️ CRITIQUE: .env est tracké par Git!"
        echo "   → Exécutez: git rm --cached .env && git commit -m 'Remove .env from tracking'"
    else
        check_item 0 ".env n'est pas tracké par Git"
    fi
else
    check_warning "Git non initialisé - exécutez 'git init'"
fi

echo ""

# 3. Vérifier le cache Python
echo "🧹 3. Propreté du Code"
echo "----------------------"
PYCACHE_COUNT=$(find . -type d -name __pycache__ 2>/dev/null | wc -l | xargs)
PYC_COUNT=$(find . -name "*.pyc" 2>/dev/null | wc -l | xargs)

if [ "$PYCACHE_COUNT" -eq 0 ]; then
    check_item 0 "Aucun __pycache__ ($PYCACHE_COUNT)"
else
    check_item 1 "$PYCACHE_COUNT répertoires __pycache__ trouvés"
    echo "   → Exécutez: find . -type d -name __pycache__ -exec rm -rf {} +"
fi

if [ "$PYC_COUNT" -eq 0 ]; then
    check_item 0 "Aucun fichier .pyc ($PYC_COUNT)"
else
    check_item 1 "$PYC_COUNT fichiers .pyc trouvés"
    echo "   → Exécutez: find . -name '*.pyc' -delete"
fi

echo ""

# 4. Vérifier les clés sensibles
echo "🔑 4. Clés API et Secrets"
echo "-------------------------"
if [ -f ".env" ]; then
    # Vérifier si des clés ressemblent à des vraies clés
    if grep -E "^[A-Z_]+_KEY=[a-zA-Z0-9]{20,}" .env >/dev/null 2>&1; then
        check_item 0 "Clés API configurées (ne pas afficher par sécurité)"
    else
        check_warning "Clés API potentiellement manquantes ou invalides"
    fi

    # Vérifier SECRET_KEY
    if grep -q "^SECRET_KEY=GENERATE_A_RANDOM_SECRET_KEY_HERE" .env 2>/dev/null; then
        check_item 1 "SECRET_KEY utilise encore le placeholder"
        echo "   → Exécutez: ./venv/bin/python -c 'import secrets; print(secrets.token_hex(32))'"
    fi
else
    check_warning "Impossible de vérifier les clés (.env manquant)"
fi

echo ""

# 5. Vérifier les fichiers sensibles exposés
echo "🚨 5. Fichiers Sensibles Exposés"
echo "---------------------------------"
SENSITIVE_FILES=(
    ".env"
    "config/secrets.yaml"
    "config/credentials.json"
    "helixone.db"
)

for file in "${SENSITIVE_FILES[@]}"; do
    if [ -f "$file" ]; then
        if [ -d ".git" ]; then
            if git ls-files --error-unmatch "$file" 2>/dev/null; then
                check_item 1 "$file est tracké par Git (DANGEREUX!)"
            else
                check_item 0 "$file existe mais n'est pas tracké"
            fi
        else
            check_warning "$file existe (Git non initialisé, impossible de vérifier)"
        fi
    fi
done

echo ""

# Résumé
echo "📊 RÉSUMÉ"
echo "========="
echo ""
echo "Fichiers de sécurité:"
echo "  - .env.example: $([ -f '.env.example' ] && echo '✅' || echo '❌')"
echo "  - .env: $([ -f '.env' ] && echo '✅' || echo '❌')"
echo "  - .gitignore: $([ -f '.gitignore' ] && echo '✅' || echo '❌')"
echo ""
echo "Propreté du code:"
echo "  - __pycache__: $PYCACHE_COUNT"
echo "  - *.pyc: $PYC_COUNT"
echo ""
echo "Git:"
echo "  - Initialisé: $([ -d '.git' ] && echo '✅' || echo '❌')"
echo ""

# Score de sécurité
SCORE=0
MAX_SCORE=10

[ -f ".env.example" ] && SCORE=$((SCORE+1))
[ -f ".env" ] && SCORE=$((SCORE+1))
[ -f ".gitignore" ] && SCORE=$((SCORE+1))
grep -q "^\.env$" .gitignore 2>/dev/null && SCORE=$((SCORE+1))
[ "$PYCACHE_COUNT" -eq 0 ] && SCORE=$((SCORE+2))
[ "$PYC_COUNT" -eq 0 ] && SCORE=$((SCORE+1))
[ -d ".git" ] && SCORE=$((SCORE+1))
! git ls-files --error-unmatch .env 2>/dev/null && SCORE=$((SCORE+2)) || SCORE=$((SCORE+0))

echo "🎯 Score de Sécurité: $SCORE/$MAX_SCORE"
echo ""

if [ $SCORE -ge 9 ]; then
    echo -e "${GREEN}✨ Excellent! Votre configuration de sécurité est très bonne.${NC}"
elif [ $SCORE -ge 7 ]; then
    echo -e "${YELLOW}⚠️  Bien, mais quelques améliorations sont recommandées.${NC}"
else
    echo -e "${RED}🚨 Attention! Des problèmes de sécurité importants ont été détectés.${NC}"
    echo "   → Consultez SECURITE_CORRECTIONS_EFFECTUEES.md"
fi

echo ""
echo "Pour plus de détails, voir: SECURITE_CORRECTIONS_EFFECTUEES.md"
