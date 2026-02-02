#!/usr/bin/env python3
"""
HelixOne - Application packagée pour distribution
Se connecte automatiquement au backend cloud
"""

import sys
import os

# Forcer le mode production
os.environ["HELIXONE_ENV"] = "production"

import customtkinter as ctk
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajustement des chemins pour les imports
if getattr(sys, 'frozen', False):
    # Mode packagé (PyInstaller)
    base_path = sys._MEIPASS
else:
    # Mode développement
    base_path = os.path.dirname(__file__)

src_path = os.path.join(base_path, "src")
interface_path = os.path.join(src_path, "interface")

for path in [base_path, src_path, interface_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

def main():
    """Point d'entrée principal de l'application packagée"""
    logger.info("🚀 Démarrage HelixOne (mode cloud)...")

    # Configuration CustomTkinter
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    # Lancement avec le splash screen
    try:
        from src.plasma_intro import PlasmaIntro
        PlasmaIntro().mainloop()
    except Exception as e:
        logger.error(f"Erreur PlasmaIntro: {e}")
        # Fallback: lancer directement l'interface
        from src.interface.main_window import launch_helixone_ui
        launch_helixone_ui()

if __name__ == "__main__":
    main()
