"""
Configuration centralisee du client HelixOne

Ce fichier permet de basculer facilement entre les environnements:
- LOCAL: Pour le developpement (backend sur localhost:8000)
- PRODUCTION: Pour l'application distribuee (backend sur le cloud)

Pour changer d'environnement, modifiez simplement ENVIRONMENT ci-dessous.
"""

import os

# ============================================
# ENVIRONNEMENT ACTIF
# ============================================
# Changer ici pour basculer entre les environnements
# Options: "local" pour dev, "production" pour distribution
# Par défaut "production" pour l'app packagée
ENVIRONMENT = os.environ.get("HELIXONE_ENV", "production")


# ============================================
# CONFIGURATION PAR ENVIRONNEMENT
# ============================================

_CONFIGS = {
    "local": {
        "API_BASE_URL": "http://127.0.0.1:8000",
        "API_TIMEOUT": 30,
        "DEBUG": True,
    },
    "production": {
        # URL du backend deploye sur Render
        "API_BASE_URL": os.environ.get("HELIXONE_API_URL", "https://helixone-apii.onrender.com"),
        "API_TIMEOUT": 45,
        "DEBUG": False,
    }
}


# ============================================
# ACCES A LA CONFIGURATION
# ============================================

def get_config():
    """Retourne la configuration active"""
    return _CONFIGS.get(ENVIRONMENT, _CONFIGS["local"])


def get_api_url():
    """Retourne l'URL de base de l'API"""
    return get_config()["API_BASE_URL"]


def get_api_timeout():
    """Retourne le timeout pour les appels API"""
    return get_config()["API_TIMEOUT"]


def is_debug():
    """Retourne True si en mode debug"""
    return get_config()["DEBUG"]


def is_production():
    """Retourne True si en mode production"""
    return ENVIRONMENT == "production"


# ============================================
# URLS SPECIFIQUES
# ============================================

def get_auth_url():
    """URL pour l'authentification"""
    return f"{get_api_url()}/auth"


def get_ibkr_url():
    """URL pour l'API IBKR"""
    return f"{get_api_url()}/api/ibkr"


def get_analysis_url():
    """URL pour l'API d'analyse"""
    return f"{get_api_url()}/api/analysis"


def get_portfolio_url():
    """URL pour l'API portfolio"""
    return f"{get_api_url()}/api/portfolio"


def get_risk_url():
    """URL pour l'API risk"""
    return f"{get_api_url()}/api/risk"


# ============================================
# AFFICHAGE DE LA CONFIG ACTIVE
# ============================================

if __name__ == "__main__":
    print(f"Environnement actif: {ENVIRONMENT}")
    print(f"API URL: {get_api_url()}")
    print(f"Timeout: {get_api_timeout()}s")
    print(f"Debug: {is_debug()}")
