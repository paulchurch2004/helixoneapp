# ============================================
# HELIXONE BACKEND - Dockerfile
# ============================================
# Multi-stage build pour une image optimisée
#
# Build: docker build -t helixone-backend .
# Run:   docker run -p 8000:8000 helixone-backend
# ============================================

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Installer les dépendances de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Installer les dépendances runtime uniquement
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash helixone

# Copier les packages Python depuis le builder
COPY --from=builder /root/.local /home/helixone/.local

# S'assurer que les scripts Python sont dans le PATH
ENV PATH=/home/helixone/.local/bin:$PATH

# Copier le code de l'application
COPY --chown=helixone:helixone . .

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs /app/data && chown -R helixone:helixone /app

# Passer à l'utilisateur non-root
USER helixone

# Variables d'environnement par défaut
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    PORT=8000

# Exposer le port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de démarrage avec workers pour production
# Note: 0.0.0.0 est correct dans Docker (accessible depuis l'extérieur du conteneur)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
