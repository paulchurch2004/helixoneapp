#!/bin/bash
# Script de démarrage simple du backend HelixOne

cd /Users/macintosh/Desktop/helixone/helixone-backend

echo "🚀 Démarrage du backend HelixOne..."
echo "📂 Répertoire: $(pwd)"
echo ""

# Activer venv et lancer
../venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
