#!/bin/bash
#
# ========================================
#   HELIXONE - Double-cliquez pour lancer
# ========================================
#

cd "$(dirname "$0")"

echo ""
echo "======================================"
echo "   Lancement de HelixOne..."
echo "======================================"
echo ""

# Verifier Python
if ! command -v python3 &> /dev/null; then
    echo "ERREUR: Python 3 n'est pas installe."
    echo ""
    echo "Installez Python depuis: https://www.python.org/downloads/"
    echo ""
    read -p "Appuyez sur Entree pour fermer..."
    exit 1
fi

echo "Python trouve: $(python3 --version)"

# Verifier/installer les dependances frontend
echo "Verification des dependances..."
if ! python3 -c "import customtkinter" 2>/dev/null; then
    echo ""
    echo "Installation des dependances interface..."
    pip3 install customtkinter pillow requests pandas yfinance --quiet
fi

# Verifier/installer les dependances backend
if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo ""
    echo "Installation des dependances serveur..."
    pip3 install uvicorn fastapi sqlalchemy passlib python-jose bcrypt --quiet
fi

# Lancer l'application
echo ""
echo "Demarrage de l'interface..."
echo ""

python3 run.py

# Si erreur
if [ $? -ne 0 ]; then
    echo ""
    echo "ERREUR: L'application s'est fermee avec une erreur."
    echo ""
    read -p "Appuyez sur Entree pour fermer..."
fi
