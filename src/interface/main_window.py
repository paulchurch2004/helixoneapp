import customtkinter as ctk
from src.interface.theme_modern import apply_modern_theme
from src.interface.home_panel import HomePanel
from src.interface.main_app import launch_main_app


def launch_helixone_ui():
    apply_modern_theme()

    app = ctk.CTk()
    app.geometry("1200x800")
    app.minsize(900, 650)
    app.title("HelixOne - Formation Trading")

    # Frame principale
    main_frame = ctk.CTkFrame(app)
    main_frame.pack(fill="both", expand=True)

    # Callback : appelé après connexion réussie
    def show_main_app():
        main_frame.destroy()
        launch_main_app(app)

    # Page d'accueil avec login, création compte et indices boursiers
    accueil = HomePanel(main_frame, on_continue_callback=show_main_app)
    accueil.place(relx=0, rely=0, relwidth=1, relheight=1)

    app.mainloop()
