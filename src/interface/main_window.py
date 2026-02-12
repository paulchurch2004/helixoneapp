import customtkinter as ctk
from src.interface.theme_modern import apply_modern_theme
from src.interface.home_panel import HomePanel
from src.interface.main_app import launch_main_app
from src.updater.auto_updater import check_updates_on_startup


def launch_helixone_ui():
    apply_modern_theme()

    # Couleur de fond uniforme pour éviter les bandes noires
    BG_COLOR = "#0a0a0f"

    app = ctk.CTk()
    app.geometry("1200x800")
    app.minsize(900, 650)
    app.title("HelixOne - Formation Trading")
    app.configure(fg_color=BG_COLOR)

    # Frame principale - remplit tout l'espace sans coins arrondis
    main_frame = ctk.CTkFrame(app, fg_color=BG_COLOR, corner_radius=0)
    main_frame.pack(fill="both", expand=True)

    # Page d'accueil avec login, création compte et indices boursiers
    accueil = HomePanel(main_frame, on_continue_callback=None)

    # Callback : appelé après connexion réussie
    def show_main_app():
        auth_manager = accueil.auth_manager

        def logout_callback():
            """Déconnexion et retour à l'écran de connexion"""
            auth_manager.logout()
            # Recréer l'écran d'accueil
            for widget in app.winfo_children():
                widget.destroy()
            new_frame = ctk.CTkFrame(app, fg_color=BG_COLOR, corner_radius=0)
            new_frame.pack(fill="both", expand=True)
            new_accueil = HomePanel(new_frame, on_continue_callback=None)
            new_accueil.on_continue_callback = show_main_app
            new_accueil.pack(fill="both", expand=True)

        main_frame.destroy()
        launch_main_app(app, auth_manager=auth_manager, logout_callback=logout_callback)

    accueil.on_continue_callback = show_main_app
    accueil.pack(fill="both", expand=True)

    # Vérifier les mises à jour en arrière-plan
    check_updates_on_startup(parent=app)

    app.mainloop()
