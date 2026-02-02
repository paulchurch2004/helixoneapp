"""
Configuration de la base de données SQLAlchemy
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings


# Créer le moteur de base de données
# check_same_thread est seulement pour SQLite
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args
)

# Créer une session locale
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles
Base = declarative_base()


# Fonction helper pour obtenir une session DB
def get_db():
    """
    Dépendance pour obtenir une session de base de données
    Utilisée dans les routes FastAPI
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()