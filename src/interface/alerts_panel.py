import customtkinter as ctk

class AlertsPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.label = ctk.CTkLabel(self, text="Alertes FXI", font=("Roboto", 16, "bold"))
        self.label.pack(pady=5)

        self.alert_box = ctk.CTkTextbox(self, height=150)
        self.alert_box.pack(fill="both", expand=True, padx=10, pady=5)
        self.alert_box.insert("end", "Aucune alerte active.")
        self.alert_box.configure(state="disabled")
