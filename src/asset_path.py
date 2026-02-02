"""
Helper pour trouver les assets dans l'app packagée ou en dev
"""
import os
import sys

def get_base_path():
    """Retourne le chemin de base selon le mode d'exécution"""
    if getattr(sys, 'frozen', False):
        # Mode packagé (PyInstaller)
        return sys._MEIPASS
    else:
        # Mode développement
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_asset_path(filename):
    """Retourne le chemin complet vers un asset"""
    base = get_base_path()
    return os.path.join(base, "assets", filename)
