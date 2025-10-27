#!/bin/bash

# HelixOne Development Script
# Lance le backend et le frontend en mode développement

set -e

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  HelixOne - Mode Développement${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Vérifier l'environnement virtuel
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Environnement virtuel non trouvé${NC}"
    echo -e "${YELLOW}Création de l'environnement virtuel...${NC}"
    python3.11 -m venv venv
    ./venv/bin/pip install --upgrade pip
fi

# Vérifier les dépendances
echo -e "${YELLOW}📦 Vérification des dépendances...${NC}"
if [ -f "requirements.txt" ]; then
    ./venv/bin/pip install -q -r requirements.txt
fi

# Vérifier si le port 8000 est libre
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port 8000 déjà utilisé${NC}"
    read -p "Voulez-vous tuer le processus existant? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:8000 | xargs kill -9
        echo -e "${GREEN}✅ Processus tué${NC}"
        sleep 1
    else
        echo -e "${RED}❌ Impossible de continuer${NC}"
        exit 1
    fi
fi

# Créer les logs
LOG_DIR="logs"
mkdir -p $LOG_DIR
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

# Nettoyer les anciens logs
> $BACKEND_LOG
> $FRONTEND_LOG

echo -e "\n${GREEN}🚀 Démarrage du backend...${NC}"
cd helixone-backend
../venv/bin/python -m uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload \
    --log-level info > ../$BACKEND_LOG 2>&1 &
BACKEND_PID=$!
cd ..

# Attendre que le backend soit prêt
echo -e "${YELLOW}⏳ Attente du démarrage du backend...${NC}"
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend démarré (PID: $BACKEND_PID)${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Timeout - Backend n'a pas démarré${NC}"
        echo -e "${YELLOW}Logs:${NC}"
        tail -20 $BACKEND_LOG
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
done

# Tester la connexion API
echo -e "\n${YELLOW}🔍 Test de l'API...${NC}"
API_RESPONSE=$(curl -s http://127.0.0.1:8000/health)
if echo "$API_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✅ API opérationnelle${NC}"
    echo "$API_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}❌ API non fonctionnelle${NC}"
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Backend: ${NC}http://127.0.0.1:8000"
echo -e "${GREEN}📚 Docs API: ${NC}http://127.0.0.1:8000/docs"
echo -e "${GREEN}📊 ReDoc: ${NC}http://127.0.0.1:8000/redoc"
echo -e "${GREEN}📝 Logs Backend: ${NC}$BACKEND_LOG"
echo -e "${BLUE}========================================${NC}\n"

# Demander si on lance le frontend
echo -e "${YELLOW}Voulez-vous lancer le frontend? (y/n)${NC}"
read -p "" -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}🖥️  Démarrage du frontend...${NC}"
    ./venv/bin/python src/main.py > $FRONTEND_LOG 2>&1 &
    FRONTEND_PID=$!
    echo -e "${GREEN}✅ Frontend démarré (PID: $FRONTEND_PID)${NC}"
    echo -e "${GREEN}📝 Logs Frontend: ${NC}$FRONTEND_LOG"

    # Sauvegarder les PIDs
    echo $BACKEND_PID > $LOG_DIR/backend.pid
    echo $FRONTEND_PID > $LOG_DIR/frontend.pid
else
    # Sauvegarder seulement le PID du backend
    echo $BACKEND_PID > $LOG_DIR/backend.pid
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Mode développement actif!${NC}"
echo -e "${YELLOW}Pour arrêter:${NC} ./stop.sh"
echo -e "${YELLOW}Pour voir les logs:${NC} tail -f logs/*.log"
echo -e "${BLUE}========================================${NC}\n"

# Fonction de nettoyage
cleanup() {
    echo -e "\n${YELLOW}🛑 Arrêt des services...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Backend arrêté${NC}"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Frontend arrêté${NC}"
    fi
    exit 0
}

# Capturer Ctrl+C
trap cleanup INT TERM

# Afficher les logs en temps réel
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}📊 Logs en temps réel (Ctrl+C pour arrêter):${NC}\n"
    tail -f $BACKEND_LOG $FRONTEND_LOG
else
    echo -e "${YELLOW}📊 Logs backend en temps réel (Ctrl+C pour arrêter):${NC}\n"
    tail -f $BACKEND_LOG
fi
