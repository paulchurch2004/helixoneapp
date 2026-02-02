#!/bin/bash
# Déterminer le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../venv/bin/activate"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000