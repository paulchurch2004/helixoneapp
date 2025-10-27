#!/bin/bash

# HelixOne Stop Script
# Arrête tous les services en cours

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Arrêt de HelixOne...${NC}\n"

# Arrêter via les PIDs sauvegardés
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        kill $BACKEND_PID
        echo -e "${GREEN}✅ Backend arrêté (PID: $BACKEND_PID)${NC}"
    fi
    rm logs/backend.pid
fi

if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        kill $FRONTEND_PID
        echo -e "${GREEN}✅ Frontend arrêté (PID: $FRONTEND_PID)${NC}"
    fi
    rm logs/frontend.pid
fi

# Tuer tous les processus sur le port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    echo -e "${GREEN}✅ Port 8000 libéré${NC}"
fi

# Tuer les processus uvicorn
pkill -9 -f "uvicorn app.main:app" 2>/dev/null && echo -e "${GREEN}✅ Uvicorn arrêté${NC}"

# Tuer les processus python helixone
pkill -9 -f "python.*src/main.py" 2>/dev/null && echo -e "${GREEN}✅ Frontend Python arrêté${NC}"

echo -e "\n${GREEN}✅ Tous les services sont arrêtés${NC}"
