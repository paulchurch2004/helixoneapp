#!/bin/bash
# Script pour démarrer l'interface HelixOne

cd /Users/macintosh/Desktop/helixone

echo "🧹 Nettoyage du cache Python..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "🚀 Lancement de l'interface HelixOne..."
echo ""
./venv/bin/python -m src.main
