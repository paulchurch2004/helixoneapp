# src/interface/module_viewer.py

import customtkinter as ctk

class ModuleViewer(ctk.CTkFrame):
    def __init__(self, master, module_data):
        super().__init__(master)
        self.pack_propagate(False)

        textbox = ctk.CTkTextbox(self, wrap="word", font=("Segoe UI", 13))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)

        content = f"📘 {module_data['titre']}\n"
        content += f"Niveau : {module_data['niveau']}\n"
        content += f"Durée : {module_data.get('duree', 'N/A')}\n\n"

        content += "🎯 Objectifs :\n"
        for objectif in module_data.get("objectifs", []):
            content += f"• {objectif}\n"

        content += "\n📚 Contenu :\n"
        content += module_data.get("contenu", "")

        content += "\n\n🧠 Quiz :\n"
        for i, q in enumerate(module_data.get("quiz", []), 1):
            content += f"{i}. {q['question']}\n"
            for r in q['reponses']:
                content += f"   {r}\n"
            content += f"(Bonne réponse : {q['bonne_reponse']})\n\n"

        content += f"\n🎓 Conclusion :\n{module_data.get('conclusion', '')}"

        textbox.insert("1.0", content)
        textbox.configure(state="disabled")
