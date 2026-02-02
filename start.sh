#!/bin/bash

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "════════════════════════════════════════════════════════════════"
echo "  🚀 HELIXONE - Lancement"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Nettoyer les anciens processus
echo "🧹 Nettoyage des anciens processus..."
pkill -9 -f "python.*uvicorn" 2>/dev/null
pkill -9 -f "python.*main_app" 2>/dev/null
pkill -9 -f "python.*run.py" 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 2

# Lancer le backend
echo "🔧 Démarrage du backend..."
cd "$SCRIPT_DIR/helixone-backend"
../venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level warning > ../backend.log 2>&1 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Attendre que le backend soit prêt
echo "⏳ Attente du backend..."
for i in {1..15}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✅ Backend prêt!"
        break
    fi
    sleep 1
done

# Vérifier si le backend est prêt
if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "❌ Erreur: Le backend n'a pas démarré"
    echo "Consultez backend.log pour plus d'informations"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ HELIXONE EST PRÊT!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Backend URL: http://127.0.0.1:8000"
echo ""
echo "🎨 Lancement de l'interface graphique..."
echo ""

# Lancer l'interface
PYTHONPATH="$SCRIPT_DIR" ./venv/bin/python run.py

# Nettoyer à la fermeture
echo ""
echo "🧹 Nettoyage..."
kill $BACKEND_PID 2>/dev/null
pkill -9 -f "python.*uvicorn" 2>/dev/null
echo "✅ Terminé!"
