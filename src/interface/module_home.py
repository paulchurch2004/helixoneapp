import customtkinter as ctk
import os
import json

FORMATION_PATH = os.path.join("data", "formation")
MODULES_META = os.path.join(FORMATION_PATH, "modules.json")


class ModuleHome(ctk.CTkFrame):
    def __init__(self, master, on_module_click, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="#1C1F26", padx=20, pady=20)
        self.on_module_click = on_module_click
        self.build_ui()

    def build_ui(self):
        # Titre
        ctk.CTkLabel(
            self,
            text="📘 Bienvenue dans la Formation",
            font=("Consolas", 22, "bold"),
            text_color="#00C9FF"
        ).pack(anchor="w", pady=(10, 20))

        # Chargement modules
        try:
            with open(MODULES_META, "r", encoding="utf-8") as f:
                modules = json.load(f)
        except Exception as e:
            ctk.CTkLabel(self, text=f"[Erreur chargement modules] {e}").pack()
            return

        for module in modules:
            self.ajouter_module(module)

    def ajouter_module(self, module):
        color = self.get_color_by_level(module["niveau"])

        btn = ctk.CTkButton(
            self,
            text=f"{module['titre']} - {module['description']}",
            fg_color=color,
            hover_color="#333333",
            anchor="w",
            font=("Segoe UI", 15),
            command=lambda m=module: self.on_module_click(m)
        )
        btn.pack(fill="x", pady=8)

    def get_color_by_level(self, niveau):
        couleur = {
            "Débutant": "#00796B",
            "Intermédiaire": "#FFA000",
            "Avancé": "#1976D2",
            "Expert": "#C62828"
        }
        return couleur.get(niveau, "#444444")
