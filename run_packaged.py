#!/usr/bin/env python3
"""
HelixOne - Application packagée pour distribution
Se connecte automatiquement au backend cloud
"""

import sys
import os
import traceback

# Forcer le mode production
os.environ["HELIXONE_ENV"] = "production"

# Écrire TOUT dans un fichier log sur le bureau
_log_path = os.path.join(os.path.expanduser("~"), "Desktop", "helixone_debug.log")
_log_file = open(_log_path, "w")
sys.stdout = _log_file
sys.stderr = _log_file
print("=== HelixOne Debug Log ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Frozen: {getattr(sys, 'frozen', False)}", flush=True)
print(f"MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}", flush=True)

try:
    import customtkinter as ctk
    print("customtkinter imported OK", flush=True)
except Exception as e:
    print(f"ERREUR customtkinter: {e}", flush=True)
    traceback.print_exc(file=_log_file)
    _log_file.flush()

import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=_log_file
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

    # Sur macOS packagé, forcer l'app au premier plan
    if sys.platform == "darwin" and getattr(sys, 'frozen', False):
        try:
            os.system('''/usr/bin/osascript -e 'tell app "System Events" to set frontmost of the first process whose unix id is ''' + str(os.getpid()) + ''' to true' &''')
        except Exception:
            pass

    # Lancement avec le splash screen vidéo
    try:
        from src.plasma_intro import VideoIntro
        VideoIntro().mainloop()
    except Exception as e:
        logger.error(f"Erreur PlasmaIntro: {e}")
        from src.interface.main_window import launch_helixone_ui
        launch_helixone_ui()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=_log_file)
        _log_file.flush()
        _log_file.close()
