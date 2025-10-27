import customtkinter as ctk

class FavoritesPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.label = ctk.CTkLabel(self, text="Favoris", font=("Roboto", 16, "bold"))
        self.label.pack(pady=5)
        
        self.fav_list = ctk.CTkTextbox(self, height=200)
        self.fav_list.pack(fill="both", expand=True, padx=10, pady=5)
        self.fav_list.insert("end", "Aucun favori pour le moment.")
        self.fav_list.configure(state="disabled")
