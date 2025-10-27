#!/bin/bash
# Script pour démarrer le backend HelixOne proprement

echo "🧹 Nettoyage du cache Python..."
cd /Users/macintosh/Desktop/helixone/helixone-backend
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "🛑 Arrêt des processus existants..."
killall -9 python Python uvicorn 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 2

echo "🚀 Démarrage du backend..."
echo ""
../venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
