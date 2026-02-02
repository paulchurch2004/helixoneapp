#!/bin/bash
# ============================================
# HELIXONE - INTERFACE SEULE
# Lance l'interface CustomTkinter
# ============================================

cd "$(dirname "$0")"

echo ""
echo "╔════════════════════════════════════════╗"
echo "║     🖥️  HELIXONE - INTERFACE  🖥️      ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Vérifier que le backend tourne
echo "🔍 Vérification du backend..."
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ Backend détecté sur port 8000"
else
    echo "❌ Backend non détecté sur port 8000"
    echo ""
    echo "⚠️  Veuillez d'abord démarrer le backend:"
    echo "   Terminal 1: ./START_DEV.sh"
    echo "   Terminal 2: ./START_INTERFACE.sh"
    echo ""
    exit 1
fi
echo ""

# Nettoyer les anciennes instances d'interface
echo "🧹 Nettoyage des anciennes instances..."
pkill -9 -f "python.*main_app" 2>/dev/null
sleep 1
echo "✅ Prêt"
echo ""

echo "╔════════════════════════════════════════╗"
echo "║  TERMINAL 2: Interface CustomTkinter   ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "🖥️  Démarrage de l'interface..."
echo "📡 Backend: http://127.0.0.1:8000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Configurer PYTHONPATH (utilise le répertoire courant)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"

# Lancer l'interface (utilise __main__.py)
exec "${SCRIPT_DIR}/venv/bin/python" -m src.interface
