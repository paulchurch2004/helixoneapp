import customtkinter as ctk
import os
import json
from tkinter import messagebox

class ModuleViewer(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="#1c1f1f")
        self.pack(fill="both", expand=True, padx=20, pady=20)

        self.modules_data = self.load_all_modules()
        self.active_module = None

        # === Barre latérale ===
        self.sidebar = ctk.CTkFrame(self, width=250, fg_color="#161a1f")
        self.sidebar.pack(side="left", fill="y", padx=(0, 10))

        self.module_buttons = []
        for module in self.modules_data:
            btn = ctk.CTkButton(
                self.sidebar,
                text=module["titre"],
                anchor="w",
                command=lambda m=module: self.afficher_module(m),
                fg_color="#003366",
                hover_color="#0055AA"
            )
            btn.pack(fill="x", padx=10, pady=5)
            self.module_buttons.append(btn)

        # === Zone principale d’affichage ===
        self.content_box = ctk.CTkTextbox(
            self,
            wrap="word",
            font=("Segoe UI", 13),
            text_color="#DDDDDD",
            fg_color="#101418"
        )
        self.content_box.pack(fill="both", expand=True)

        if self.modules_data:
            self.afficher_module(self.modules_data[0])

    def load_all_modules(self):
        modules = []
        base_path = os.path.join("data", "formation")
        niveaux = [f"niveau_{i}" for i in range(1, 5)]

        for niveau in niveaux:
            niveau_path = os.path.join(base_path, niveau)
            if not os.path.isdir(niveau_path):
                continue

            for file_name in sorted(os.listdir(niveau_path)):
                if file_name.endswith(".json"):
                    file_path = os.path.join(niveau_path, file_name)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            module = json.load(f)
                            modules.append(module)
                    except Exception as e:
                        print(f"[❌] Erreur chargement {file_name} : {e}")
        return modules

    def afficher_module(self, module):
        self.active_module = module
        self.content_box.delete("1.0", "end")

        def section(titre):
            return f"\n📘 {titre}\n{'-' * len(titre)}\n"

        def paragraphe(txt):
            return f"{txt}\n"

        def item(txt):
            return f"• {txt}\n"

        content = f"🟢 {module['titre']}\n"
        content += f"{module.get('niveau', '')}\n\n"

        content += section("🎯 Objectifs")
        content += "\n".join([item(obj) for obj in module.get("objectifs", [])])
        content += "\n"

        contenu = module.get("contenu")
        if isinstance(contenu, str):
            content += contenu + "\n"
        elif isinstance(contenu, list):
            for bloc in contenu:
                content += section(bloc.get("titre", ""))
                for p in bloc.get("paragraphes", []):
                    content += paragraphe(p)
        else:
            content += "[⚠️] Contenu non reconnu\n"

        if "conclusion" in module:
            content += section("📌 Conclusion")
            content += paragraphe(module["conclusion"])

        self.content_box.insert("1.0", content)


def afficher_interface_formation(parent):
    print("[DEBUG] Interface formations lancée ✅")
    for widget in parent.winfo_children():
        widget.destroy()
    ModuleViewer(parent)
