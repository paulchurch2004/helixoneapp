#!/bin/bash
# Script pour démarrer le backend HelixOne proprement

# Déterminer le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}/helixone-backend"

echo "🧹 Nettoyage du cache Python..."
cd "${BACKEND_DIR}"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "🛑 Arrêt des processus existants..."
pkill -f "uvicorn app.main:app" 2>/dev/null
lsof -ti:8000 | xargs kill 2>/dev/null
sleep 2

echo "🚀 Démarrage du backend..."
echo ""
"${SCRIPT_DIR}/venv/bin/python" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
