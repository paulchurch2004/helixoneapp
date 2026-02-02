#!/bin/bash
# ============================================
# HELIXONE - Script de démarrage principal
# ============================================
# Usage:
#   ./helixone.sh          # Mode normal (GUI)
#   ./helixone.sh dev      # Mode développement
#   ./helixone.sh docker   # Mode Docker
#   ./helixone.sh backend  # Backend seul
#   ./helixone.sh check    # Vérification système
# ============================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
BACKEND_PORT=8000
BACKEND_HOST="127.0.0.1"
HEALTH_TIMEOUT=20
LOG_DIR="$SCRIPT_DIR/logs"
VENV_DIR="$SCRIPT_DIR/venv"

# Créer répertoire logs
mkdir -p "$LOG_DIR"

# ============================================
# Fonctions utilitaires
# ============================================

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  🚀 HELIXONE - $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# ============================================
# Vérification système
# ============================================

check_dependencies() {
    print_header "Vérification des dépendances"

    local errors=0

    # Python
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $py_version"
    else
        print_error "Python 3 non trouvé"
        ((errors++))
    fi

    # Virtual environment
    if [ -d "$VENV_DIR" ]; then
        print_success "Virtual environment trouvé"
    else
        print_warning "Virtual environment non trouvé - création en cours..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment créé"
    fi

    # PostgreSQL (optionnel)
    if command -v psql &> /dev/null; then
        print_success "PostgreSQL client disponible"
    else
        print_warning "PostgreSQL client non trouvé (optionnel)"
    fi

    # Redis (optionnel)
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            print_success "Redis disponible et connecté"
        else
            print_warning "Redis installé mais non connecté (cache désactivé)"
        fi
    else
        print_warning "Redis non trouvé (cache désactivé)"
    fi

    # Docker (optionnel)
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            print_success "Docker disponible"
        else
            print_warning "Docker installé mais démon non actif"
        fi
    else
        print_info "Docker non installé (mode conteneur non disponible)"
    fi

    echo ""

    if [ $errors -gt 0 ]; then
        print_error "$errors erreur(s) critique(s) trouvée(s)"
        return 1
    fi

    print_success "Toutes les dépendances critiques sont présentes"
    return 0
}

check_ports() {
    print_info "Vérification des ports..."

    if lsof -i:$BACKEND_PORT &> /dev/null; then
        print_warning "Port $BACKEND_PORT déjà utilisé - libération..."
        lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    fi

    print_success "Port $BACKEND_PORT disponible"
}

# ============================================
# Nettoyage
# ============================================

cleanup() {
    print_info "Nettoyage des processus..."
    pkill -9 -f "python.*uvicorn" 2>/dev/null || true
    pkill -9 -f "python.*main_app" 2>/dev/null || true
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
    print_success "Processus nettoyés"
}

# ============================================
# Démarrage backend
# ============================================

start_backend() {
    print_info "Démarrage du backend..."

    cd "$SCRIPT_DIR/helixone-backend"

    # Activer venv et lancer
    "$VENV_DIR/bin/python" -m uvicorn app.main:app \
        --host $BACKEND_HOST \
        --port $BACKEND_PORT \
        --log-level warning \
        > "$LOG_DIR/backend.log" 2>&1 &

    BACKEND_PID=$!
    echo $BACKEND_PID > "$LOG_DIR/backend.pid"

    cd "$SCRIPT_DIR"

    # Attendre que le backend soit prêt
    print_info "Attente du backend (timeout: ${HEALTH_TIMEOUT}s)..."

    for i in $(seq 1 $HEALTH_TIMEOUT); do
        if curl -s "http://$BACKEND_HOST:$BACKEND_PORT/health" > /dev/null 2>&1; then
            print_success "Backend prêt! (PID: $BACKEND_PID)"

            # Afficher status
            local health=$(curl -s "http://$BACKEND_HOST:$BACKEND_PORT/health")
            local status=$(echo "$health" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            print_info "Status: $status"

            return 0
        fi
        sleep 1
    done

    print_error "Backend n'a pas démarré dans les temps"
    print_error "Consultez $LOG_DIR/backend.log"
    cat "$LOG_DIR/backend.log" | tail -20
    return 1
}

# ============================================
# Démarrage interface
# ============================================

start_gui() {
    print_info "Lancement de l'interface graphique..."

    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    "$VENV_DIR/bin/python" "$SCRIPT_DIR/run.py"
}

# ============================================
# Mode Docker
# ============================================

start_docker() {
    print_header "Mode Docker"

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "docker-compose non trouvé"
        return 1
    fi

    cd "$SCRIPT_DIR/helixone-backend"

    print_info "Construction et démarrage des conteneurs..."
    docker compose up --build -d

    print_info "Attente des services..."
    sleep 10

    # Vérifier health
    if curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        print_success "Services Docker démarrés!"
        docker compose ps
    else
        print_error "Services non disponibles"
        docker compose logs --tail=50
        return 1
    fi

    cd "$SCRIPT_DIR"
}

stop_docker() {
    print_info "Arrêt des conteneurs Docker..."
    cd "$SCRIPT_DIR/helixone-backend"
    docker compose down
    print_success "Conteneurs arrêtés"
    cd "$SCRIPT_DIR"
}

# ============================================
# Installation des dépendances
# ============================================

install_deps() {
    print_header "Installation des dépendances"

    print_info "Activation du virtual environment..."
    source "$VENV_DIR/bin/activate"

    print_info "Mise à jour pip..."
    pip install --upgrade pip > /dev/null

    print_info "Installation des dépendances backend..."
    pip install -r "$SCRIPT_DIR/helixone-backend/requirements.txt" > /dev/null

    print_success "Dépendances installées"
}

# ============================================
# Point d'entrée principal
# ============================================

main() {
    local mode="${1:-normal}"

    case "$mode" in
        check)
            check_dependencies
            ;;

        install)
            check_dependencies
            install_deps
            ;;

        dev)
            print_header "Mode Développement"
            cleanup
            check_ports
            export HELIXONE_DEV=1
            start_backend
            start_gui
            cleanup
            ;;

        backend)
            print_header "Backend Seul"
            cleanup
            check_ports
            start_backend
            print_info "Backend en cours d'exécution..."
            print_info "Appuyez sur Ctrl+C pour arrêter"
            wait
            ;;

        docker)
            start_docker
            ;;

        docker-stop)
            stop_docker
            ;;

        normal|"")
            print_header "Démarrage"
            check_dependencies || exit 1
            cleanup
            check_ports
            start_backend || exit 1
            echo ""
            print_success "HELIXONE EST PRÊT!"
            echo ""
            print_info "Backend: http://$BACKEND_HOST:$BACKEND_PORT"
            print_info "Health:  http://$BACKEND_HOST:$BACKEND_PORT/health"
            echo ""
            start_gui
            cleanup
            ;;

        *)
            echo "Usage: $0 [mode]"
            echo ""
            echo "Modes disponibles:"
            echo "  (vide)     - Mode normal avec GUI"
            echo "  dev        - Mode développement"
            echo "  backend    - Backend seul"
            echo "  docker     - Mode Docker (démarre tous les services)"
            echo "  docker-stop- Arrête les conteneurs Docker"
            echo "  check      - Vérifie les dépendances"
            echo "  install    - Installe les dépendances"
            exit 1
            ;;
    esac
}

# Trap pour nettoyage à la sortie
trap cleanup EXIT

# Lancer
main "$@"
