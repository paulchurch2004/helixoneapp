#!/bin/bash
# Script pour redémarrer le backend HelixOne

echo "🛑 Arrêt du backend en cours..."
# Trouver et tuer le processus uvicorn
pkill -f "uvicorn.*helixone-backend"

sleep 2

echo "🚀 Démarrage du backend avec le nouvel endpoint..."
cd helixone-backend

# Activer l'environnement virtuel et lancer
source ../venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &

echo "✅ Backend redémarré avec l'endpoint /stock-deep-analysis"
echo "📡 Backend accessible sur http://localhost:8000"
echo ""
echo "Pour vérifier que l'endpoint existe:"
echo "curl http://localhost:8000/docs | grep stock-deep-analysis"
