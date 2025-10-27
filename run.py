import sys
import os
import threading
import subprocess
import customtkinter as ctk

# 🎯 Ajustement des chemins pour les imports
base_path = os.path.dirname(__file__)
src_path = os.path.join(base_path, "src")
interface_path = os.path.join(src_path, "interface")

for path in [src_path, interface_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 🔁 Lancer l'API FastAPI dans un thread
def lancer_api():
    try:
        backend_path = os.path.join(base_path, "helixone-backend")
        subprocess.Popen(
            ["../venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=open('uvicorn.log', 'w'),
            stderr=subprocess.STDOUT,
            cwd=backend_path
        )
        print("✅ Backend API lancé sur http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Erreur lancement API : {e}")

# 🎯 Lancement de l'interface principale après l’intro
def lancer_interface():
    from src.interface.main_window import launch_helixone_ui
    launch_helixone_ui()

# 🎬 Point d'entrée principal
if __name__ == "__main__":
    # 🔑 Si mode DEV, définir le token d'authentification immédiatement
    if os.environ.get("HELIXONE_DEV") == "1":
        print("[⚙️ MODE DEV] Configuration du token d'authentification...")
        from src.auth_session import set_auth_token
        # Token valide 1 an pour l'utilisateur test
        set_auth_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMjI2ZjI0MDctNGY2Yi00ODMyLWJjMTQtZGZhNzQ4M2JmY2Y0IiwiZW1haWwiOiJ0ZXN0QGhlbGl4b25lLmNvbSIsImV4cCI6MTc5MTkzMDA2N30.DDnZTWxmHCfPW6mVJrhKCU0HJeD7vCxcPTTIXwjmq5M")

    threading.Thread(target=lancer_api, daemon=True).start()

    # 📺 Splash Plasma Intro
    from src.plasma_intro import PlasmaIntro
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    PlasmaIntro().mainloop()
