import customtkinter as ctk

class ResultDisplay(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.label = ctk.CTkLabel(self, text="Résultats d’analyse", font=("Roboto", 18, "bold"))
        self.label.pack(pady=10)

        self.result_text = ctk.CTkTextbox(self, height=300, wrap="word")
        self.result_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.result_text.insert("end", "Sélectionnez un ticker pour afficher les données.")
        self.result_text.configure(state="disabled")

    def show_analysis(self, data: dict):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        for k, v in data.items():
            self.result_text.insert("end", f"{k.upper()}: {v}\n")
        self.result_text.configure(state="disabled")
