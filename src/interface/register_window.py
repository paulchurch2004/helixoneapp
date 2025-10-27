"""
Fenêtre d'inscription HelixOne
"""

import customtkinter as ctk
from tkinter import messagebox
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixone_client import HelixOneAPIError


class RegisterWindow(ctk.CTk):
    """
    Fenêtre d'inscription
    """
    
    def __init__(self, auth_manager):
        super().__init__()
        
        self.auth_manager = auth_manager
        self.login_callback = None
        self.success_callback = None
        
        # Configuration fenêtre
        self.title("HelixOne - Inscription")
        self.geometry("450x700")
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
        
        # Titre
        title_label = ctk.CTkLabel(
            main_frame,
            text="Créer un compte",
            font=("Arial", 28, "bold"),
            text_color="#1f538d"
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ctk.CTkLabel(
            main_frame,
            text="Essai gratuit 14 jours",
            font=("Arial", 14),
            text_color="green"
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Prénom
        firstname_label = ctk.CTkLabel(
            main_frame,
            text="Prénom",
            font=("Arial", 12, "bold"),
            anchor="w"
        )
        firstname_label.pack(fill="x", pady=(0, 5))
        
        self.firstname_entry = ctk.CTkEntry(
            main_frame,
            height=40,
            placeholder_text="Votre prénom",
            font=("Arial", 12)
        )
        self.firstname_entry.pack(fill="x", pady=(0, 15))
        
        # Nom
        lastname_label = ctk.CTkLabel(
            main_frame,
            text="Nom",
            font=("Arial", 12, "bold"),
            anchor="w"
        )
        lastname_label.pack(fill="x", pady=(0, 5))
        
        self.lastname_entry = ctk.CTkEntry(
            main_frame,
            height=40,
            placeholder_text="Votre nom",
            font=("Arial", 12)
        )
        self.lastname_entry.pack(fill="x", pady=(0, 15))
        
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
        self.email_entry.pack(fill="x", pady=(0, 15))
        
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
            placeholder_text="Au moins 8 caractères",
            font=("Arial", 12)
        )
        self.password_entry.pack(fill="x", pady=(0, 10))
        
        # Bind Enter key
        self.password_entry.bind("<Return>", lambda e: self.handle_register())
        
        # Message
        self.message_label = ctk.CTkLabel(
            main_frame,
            text="",
            font=("Arial", 11)
        )
        self.message_label.pack(pady=(0, 10))
        
        # Bouton inscription
        register_button = ctk.CTkButton(
            main_frame,
            text="Créer mon compte",
            height=40,
            font=("Arial", 14, "bold"),
            command=self.handle_register
        )
        register_button.pack(fill="x", pady=(10, 20))
        
        # Séparateur
        separator_frame = ctk.CTkFrame(main_frame, height=1, fg_color="gray")
        separator_frame.pack(fill="x", pady=15)
        
        # Texte "Déjà un compte ?"
        have_account_label = ctk.CTkLabel(
            main_frame,
            text="Vous avez déjà un compte ?",
            font=("Arial", 11),
            text_color="gray"
        )
        have_account_label.pack()
        
        # Bouton retour login
        login_button = ctk.CTkButton(
            main_frame,
            text="Se connecter",
            height=40,
            font=("Arial", 13),
            fg_color="transparent",
            border_width=2,
            border_color="#1f538d",
            hover_color="#e8f0f8",
            text_color="#1f538d",
            command=self.show_login
        )
        login_button.pack(fill="x", pady=(10, 0))
    
    def handle_register(self):
        """Gérer l'inscription"""
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        firstname = self.firstname_entry.get().strip()
        lastname = self.lastname_entry.get().strip()
        
        # Validation
        if not firstname:
            self.message_label.configure(text="⚠️ Veuillez entrer votre prénom", text_color="red")
            return
        
        if not lastname:
            self.message_label.configure(text="⚠️ Veuillez entrer votre nom", text_color="red")
            return
        
        if not email:
            self.message_label.configure(text="⚠️ Veuillez entrer votre email", text_color="red")
            return
        
        if not password or len(password) < 8:
            self.message_label.configure(text="⚠️ Le mot de passe doit contenir au moins 8 caractères", text_color="red")
            return
        
        # Indication de chargement
        self.message_label.configure(text="⏳ Création du compte...", text_color="blue")
        self.update()
        
        try:
            # Appeler l'API
            result = self.auth_manager.register(
                email=email,
                password=password,
                first_name=firstname,
                last_name=lastname
            )
            
            # Récupérer la licence (créée automatiquement)
            license = self.auth_manager.get_license_info()
            
            # Afficher message de succès
            messagebox.showinfo(
                "Compte créé !",
                f"Bienvenue {firstname} !\n\n"
                f"✨ Votre licence d'essai de {license['days_remaining']} jours est active.\n\n"
                f"Vous pouvez effectuer {license['quota_daily_analyses']} analyses par jour.\n\n"
                f"Profitez de HelixOne !"
            )
            
            # Callback de succès
            if self.success_callback:
                self.success_callback()
            
            # Fermer la fenêtre
            self.destroy()
            
        except HelixOneAPIError as e:
            error_msg = str(e)
            if "déjà utilisé" in error_msg.lower() or "already" in error_msg.lower():
                self.message_label.configure(
                    text="❌ Cet email est déjà utilisé",
                    text_color="red"
                )
            else:
                self.message_label.configure(
                    text=f"❌ {error_msg}",
                    text_color="red"
                )
        except Exception as e:
            self.message_label.configure(
                text=f"❌ Erreur : {str(e)}",
                text_color="red"
            )
    
    def show_login(self):
        """Afficher l'écran de connexion"""
        if self.login_callback:
            self.login_callback()
        self.destroy()
    
    def set_login_callback(self, callback):
        """Définir la fonction à appeler pour la connexion"""
        self.login_callback = callback
    
    def set_success_callback(self, callback):
        """Définir la fonction à appeler après inscription réussie"""
        self.success_callback = callback


# Test du module
if __name__ == "__main__":
    from src.auth_manager import AuthManager
    
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
    auth = AuthManager()
    
    def on_success():
        print("✅ Inscription réussie !")
    
    def on_login():
        print("🔐 Afficher la connexion")
    
    app = RegisterWindow(auth)
    app.set_success_callback(on_success)
    app.set_login_callback(on_login)
    app.mainloop()
