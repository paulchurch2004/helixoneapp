import customtkinter as ctk
from src.interface.theme_modern import apply_modern_theme
from src.interface.router import Router
from src.interface.home_panel import HomePanel
from src.interface.main_app import launch_main_app


def launch_helixone_ui():
    # 🧪 Initialisation CustomTkinter
    apply_modern_theme()  # <- correction ici

    app = ctk.CTk()
    app.geometry("1200x800")
    app.title("HelixOne - Analyse Boursière")

    # 🧱 Frame principale
    main_frame = ctk.CTkFrame(app)
    main_frame.pack(fill="both", expand=True)

    # ✅ Callback : appelé après le bouton "Entrer dans HelixOne"
    def show_main_app():
        main_frame.destroy()
        launch_main_app(app)

    # 🖼️ Page d’accueil HelixOne (fond, logo, indices, bouton)
    accueil = HomePanel(main_frame, on_continue_callback=show_main_app)
    accueil.place(relx=0, rely=0, relwidth=1, relheight=1)

    # ▶️ Démarrer GUI
    app.mainloop()
