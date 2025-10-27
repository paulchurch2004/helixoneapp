"""
Fenêtre de connexion HelixOne
"""

import customtkinter as ctk
from tkinter import messagebox
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixone_client import HelixOneAPIError


class LoginWindow(ctk.CTk):
    """
    Fenêtre de connexion
    """
    
    def __init__(self, auth_manager):
        super().__init__()
        
        self.auth_manager = auth_manager
        self.register_callback = None
        self.success_callback = None
        
        # Configuration fenêtre
        self.title("HelixOne - Connexion")
        self.geometry("450x550")
        self.resizable(False, False)
        
        # Centrer la fenêtre
        self.center_window()
        
        # Créer l'interface
        self.create_ui()
    
    def center_window(self):
        """Centrer la fenêtre sur l'écran"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_ui(self):
        """Créer l'interface utilisateur"""
        
        # Container principal
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Logo / Titre
        title_label = ctk.CTkLabel(
            main_frame,
            text="HelixOne",
            font=("Arial", 36, "bold"),
            text_color="#1f538d"
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ctk.CTkLabel(
            main_frame,
            text="Analyse d'actions avec IA",
            font=("Arial", 14),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 40))
        
        # Email
        email_label = ctk.CTkLabel(
            main_frame,
            text="Email",
            font=("Arial", 12, "bold"),
            anchor="w"
        )
        email_label.pack(fill="x", pady=(0, 5))
        
        self.email_entry = ctk.CTkEntry(
            main_frame,
            height=40,
            placeholder_text="votre@email.com",
            font=("Arial", 12)
        )
        self.email_entry.pack(fill="x", pady=(0, 20))
        
        # Mot de passe
        password_label = ctk.CTkLabel(
            main_frame,
            text="Mot de passe",
            font=("Arial", 12, "bold"),
            anchor="w"
        )
        password_label.pack(fill="x", pady=(0, 5))
        
        self.password_entry = ctk.CTkEntry(
            main_frame,
            height=40,
            show="•",
            placeholder_text="Votre mot de passe",
            font=("Arial", 12)
        )
        self.password_entry.pack(fill="x", pady=(0, 10))
        
        # Bind Enter key
        self.password_entry.bind("<Return>", lambda e: self.handle_login())
        
        # Message d'erreur
        self.error_label = ctk.CTkLabel(
            main_frame,
            text="",
            text_color="red",
            font=("Arial", 11)
        )
        self.error_label.pack(pady=(0, 10))
        
        # Bouton connexion
        login_button = ctk.CTkButton(
            main_frame,
            text="Se connecter",
            height=40,
            font=("Arial", 14, "bold"),
            command=self.handle_login
        )
        login_button.pack(fill="x", pady=(10, 20))
        
        # Séparateur
        separator_frame = ctk.CTkFrame(main_frame, height=1, fg_color="gray")
        separator_frame.pack(fill="x", pady=20)
        
        # Texte "Pas de compte ?"
        no_account_label = ctk.CTkLabel(
            main_frame,
            text="Pas encore de compte ?",
            font=("Arial", 11),
            text_color="gray"
        )
        no_account_label.pack()
        
        # Bouton inscription
        register_button = ctk.CTkButton(
            main_frame,
            text="Créer un compte",
            height=40,
            font=("Arial", 13),
            fg_color="transparent",
            border_width=2,
            border_color="#1f538d",
            hover_color="#e8f0f8",
            text_color="#1f538d",
            command=self.show_register
        )
        register_button.pack(fill="x", pady=(10, 0))
    
    def handle_login(self):
        """Gérer la connexion"""
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        
        # Validation
        if not email:
            self.error_label.configure(text="⚠️ Veuillez entrer votre email")
            return
        
        if not password:
            self.error_label.configure(text="⚠️ Veuillez entrer votre mot de passe")
            return
        
        # Désactiver le bouton pendant la requête
        self.error_label.configure(text="⏳ Connexion en cours...", text_color="blue")
        self.update()
        
        try:
            # Appeler l'API
            result = self.auth_manager.login(email, password)
            
            # Récupérer la licence
            license = self.auth_manager.get_license_info()
            
            # Afficher message de succès
            messagebox.showinfo(
                "Connexion réussie",
                f"Bienvenue !\n\n"
                f"Licence : {license['license_type'].upper()}\n"
                f"Jours restants : {license['days_remaining']}"
            )
            
            # Callback de succès
            if self.success_callback:
                self.success_callback()
            
            # Fermer la fenêtre
            self.destroy()
            
        except HelixOneAPIError as e:
            error_msg = str(e)
            if "401" in error_msg or "incorrect" in error_msg.lower():
                self.error_label.configure(
                    text="❌ Email ou mot de passe incorrect",
                    text_color="red"
                )
            else:
                self.error_label.configure(
                    text=f"❌ {error_msg}",
                    text_color="red"
                )
        except Exception as e:
            self.error_label.configure(
                text=f"❌ Erreur : {str(e)}",
                text_color="red"
            )
    
    def show_register(self):
        if self.register_callback:
            self.register_callback()
       
    def set_register_callback(self, callback):
        """Définir la fonction à appeler pour l'inscription"""
        self.register_callback = callback
    
    def set_success_callback(self, callback):
        """Définir la fonction à appeler après connexion réussie"""
        self.success_callback = callback


# Test du module
if __name__ == "__main__":
    from src.auth_manager import AuthManager
    
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
    auth = AuthManager()
    
    def on_success():
        print("✅ Connexion réussie !")
    
    def on_register():
        print("📝 Afficher l'inscription")
    
    app = LoginWindow(auth)
    app.set_success_callback(on_success)
    app.set_register_callback(on_register)
    app.mainloop()
