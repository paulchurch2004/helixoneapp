#!/usr/bin/env python3
"""
Point d'entrée pour lancer l'interface graphique HelixOne
Usage: python -m src.interface

Ce module lance le flux complet:
1. Vérification du backend
2. Animation PlasmaIntro
3. HomePanel puis MainApp
"""

import os
import sys
import time
import requests
import logging
import customtkinter as ctk

# Import de la configuration centralisee
from src.config import get_api_url

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_backend(max_tries=10):
    """Vérifier si le backend est accessible"""
    api_url = get_api_url()
    logger.info(f"⏳ Vérification du backend ({api_url})...")
    for i in range(max_tries):
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Backend prêt!")
                return True
        except Exception:
            pass
        time.sleep(0.5)
    logger.warning("⚠️ Backend non détecté, l'application continuera quand même")
    return False

if __name__ == "__main__":
    logger.info("🚀 Démarrage HelixOne...")

    # Vérifier le backend
    check_backend()

    # Vérifier les mises à jour en arrière-plan
    try:
        from src.updater.auto_updater import check_updates_on_startup
        check_updates_on_startup()
        logger.info("Vérification des mises à jour lancée")
    except Exception as e:
        logger.warning(f"Impossible de vérifier les mises à jour: {e}")

    # Configuration CustomTkinter
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    # Lancer l'animation PlasmaIntro qui enchaîne sur le HomePanel
    try:
        from src.plasma_intro import PlasmaIntro
        PlasmaIntro().mainloop()
    except Exception as e:
        logger.error(f"Erreur PlasmaIntro: {e}")
        # Fallback: lancer directement l'interface
        from src.interface.main_window import launch_helixone_ui
        launch_helixone_ui()
