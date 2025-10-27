"""
Module pour gérer la session d'authentification globale de l'application
Stocke le token JWT pour être utilisé par tous les modules
"""

import os

# Token JWT global
_auth_token = None

def set_auth_token(token: str):
    """
    Définit le token d'authentification JWT

    Args:
        token: Token JWT reçu de l'API
    """
    global _auth_token
    _auth_token = token
    # Également définir dans les variables d'environnement pour compatibilité
    os.environ["HELIXONE_API_TOKEN"] = token
    print(f"✅ Token d'authentification configuré")

def get_auth_token() -> str:
    """
    Récupère le token d'authentification JWT

    Returns:
        Token JWT ou None si non authentifié
    """
    global _auth_token
    return _auth_token or os.environ.get("HELIXONE_API_TOKEN")

def clear_auth_token():
    """
    Efface le token d'authentification (déconnexion)
    """
    global _auth_token
    _auth_token = None
    if "HELIXONE_API_TOKEN" in os.environ:
        del os.environ["HELIXONE_API_TOKEN"]
    print("🔒 Session fermée")

def is_authenticated() -> bool:
    """
    Vérifie si l'utilisateur est authentifié

    Returns:
        True si un token existe
    """
    return get_auth_token() is not None
