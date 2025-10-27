# src/formation/module_manager.py
import json
import os
from datetime import datetime

class ModuleManager:
    def __init__(self):
        self.data_path = "data/formation_commerciale"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        os.makedirs(self.data_path, exist_ok=True)
    
    def load_modules(self):
        modules_file = os.path.join(self.data_path, "modules_complets.json")
        try:
            with open(modules_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_default_modules()
    
    def create_default_modules(self):
        # Création des modules par défaut
        modules = [
            {
                "id": "basics_1",
                "parcours": "débutant", 
                "titre": "🎯 Fondamentaux de la Bourse",
                "description": "Comprenez le fonctionnement des marchés",
                "durée": "45 min",
                "points_xp": 100,
                "contenu": "Contenu détaillé du module...",
                "quiz": [
                    {
                        "question": "Qu'est-ce qu'une action ?",
                        "options": ["Prêt", "Part de propriété", "Dérivé", "Obligation"],
                        "bonne_reponse": 1,
                        "explication": "Une action = part de propriété"
                    }
                ]
            }
            # Ajouter plus de modules...
        ]
        
        self.save_modules(modules)
        return modules
    
    def save_modules(self, modules):
        modules_file = os.path.join(self.data_path, "modules_complets.json")
        with open(modules_file, 'w', encoding='utf-8') as f:
            json.dump(modules, f, indent=2, ensure_ascii=False)