import customtkinter as ctk

class SettingsPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.label = ctk.CTkLabel(self, text="Paramètres", font=("Roboto", 16, "bold"))
        self.label.pack(pady=5)

        self.theme_switch = ctk.CTkSwitch(self, text="Mode Sombre")
        self.theme_switch.select()
        self.theme_switch.pack(pady=10)
