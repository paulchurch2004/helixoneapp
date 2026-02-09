#!/bin/bash
# Script ultra-simple pour lancer HelixOne en mode DEV

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

clear
echo "🚀 HelixOne - Mode DEV"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Nettoyer
echo "🧹 Nettoyage..."
pkill -9 -f "python.*uvicorn" 2>/dev/null
pkill -9 -f "python.*main_app" 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 2
echo "✅ Nettoyage terminé"
echo ""

# Lancer backend en arrière-plan
echo "🔧 Démarrage backend..."
cd "$SCRIPT_DIR/helixone-backend"
HELIXONE_DEV=1 ../venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > /dev/null 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Attendre le backend
echo ""
echo "⏳ Vérification backend..."
for i in {1..10}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✅ Backend prêt!"
        break
    fi
    sleep 1
done

# Lancer frontend
echo ""
echo "🎨 Lancement frontend..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
HELIXONE_ENV=local HELIXONE_DEV=1 ./venv/bin/python -m src.interface

# Nettoyage à la sortie
echo ""
echo "🧹 Nettoyage final..."
kill $BACKEND_PID 2>/dev/null
pkill -9 -f "python.*uvicorn" 2>/dev/null
echo "✅ Terminé!"
