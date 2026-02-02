#!/bin/bash
# Script pour redémarrer le backend HelixOne

# Déterminer le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🛑 Arrêt du backend en cours..."
# Arrêter proprement avec SIGTERM d'abord
pkill -f "uvicorn app.main:app" 2>/dev/null
sleep 2

# Si toujours en cours, forcer l'arrêt
if pgrep -f "uvicorn app.main:app" > /dev/null; then
    pkill -9 -f "uvicorn app.main:app" 2>/dev/null
    sleep 1
fi

echo "🚀 Démarrage du backend..."
cd "${SCRIPT_DIR}/helixone-backend"

# Activer l'environnement virtuel et lancer (127.0.0.1 pour sécurité en dev)
source "${SCRIPT_DIR}/venv/bin/activate"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 &

echo "✅ Backend redémarré"
echo "📡 Backend accessible sur http://127.0.0.1:8000"
echo ""
echo "Pour vérifier l'API:"
echo "curl http://127.0.0.1:8000/health"
